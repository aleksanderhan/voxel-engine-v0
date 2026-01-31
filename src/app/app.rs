use std::sync::Arc;
use std::time::Instant;

use winit::{
    event::*,
    event_loop::EventLoopWindowTarget,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use glam::{Mat4, Vec2, Vec3, Vec4};

use super::{camera::Camera, config, input::InputState, profiler::FrameProf};

use crate::{
    clipmap::Clipmap,
    render::{BallGpu, CameraGpu, ClipmapGpu, OverlayGpu, Renderer},
    streaming::ChunkManager,
    world::WorldGen,
};

pub struct App {
    window: Arc<Window>,
    start_time: Instant,

    _instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    _adapter: wgpu::Adapter,
    _surface_format: wgpu::TextureFormat,
    config_sc: wgpu::SurfaceConfiguration,

    renderer: Renderer,

    world: Arc<WorldGen>,
    chunks: ChunkManager,

    clipmap: Clipmap,

    input: InputState,
    camera: Camera,

    fps_value: u32,
    fps_frames: u32,
    fps_last: Instant,

    frame_index: u32,

    profiler: FrameProf,

    last_frame: Instant,

    prev_view_proj: Mat4,
    has_prev_vp: bool,

    last_stream_update: Instant,
    stream_period: std::time::Duration,

    physics: crate::physics::Physics,
    physics_player: crate::physics::CharacterState,
    physics_ctrl: crate::physics::CharacterController,
    free_cam: bool,
}

impl App {
    pub fn window(&self) -> &Arc<Window> {
        &self.window
    }

    pub async fn new(window: Arc<Window>) -> Self {
        let start_time = Instant::now();
        let size = window.inner_size();

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats[0];

        let present_mode = if surface_caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else if surface_caps.present_modes.contains(&wgpu::PresentMode::Fifo) {
            wgpu::PresentMode::Fifo
        } else if surface_caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
            wgpu::PresentMode::Immediate
        } else {
            surface_caps.present_modes[0]
        };

        let config_sc = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 3,
        };

        let renderer =
            Renderer::new(&adapter, surface_format, config_sc.width, config_sc.height).await;

        surface.configure(renderer.device(), &config_sc);

        let world = Arc::new(WorldGen::new(12345));
        let chunks = ChunkManager::new(world.clone());

        let camera = Camera::new(config_sc.width as f32 / config_sc.height as f32);
        let input = InputState::default();

        let clipmap = Clipmap::new();

        let start_pos = Vec3::new(
            (config::CHUNK_SIZE as f32 * config::VOXEL_SIZE_M_F32) * 0.5,
            50.0,
            -20.0,
        );
        let physics = crate::physics::Physics::new(start_pos);

        let player_pos = camera.position();
        let physics_player = crate::physics::CharacterState {
            pos: player_pos,
            vel: Vec3::ZERO,
            on_ground: false,
        };
        let physics_ctrl = crate::physics::CharacterController::default();

        Self {
            window,
            start_time,
            _instance: instance,
            surface,
            _adapter: adapter,
            _surface_format: surface_format,
            config_sc,
            renderer,
            world,
            chunks,
            clipmap,
            input,
            camera,
            fps_value: 0,
            fps_frames: 0,
            fps_last: Instant::now(),
            frame_index: 0,
            profiler: FrameProf::new(),
            last_frame: Instant::now(),
            prev_view_proj: Mat4::IDENTITY,
            has_prev_vp: false,
            last_stream_update: Instant::now(),
            stream_period: std::time::Duration::from_millis(33),
            physics,
            physics_player,
            physics_ctrl,
            free_cam: false,
        }
    }

    pub fn handle_event(&mut self, event: Event<()>, elwt: &EventLoopWindowTarget<()>) {
        match event {
            Event::DeviceEvent { event, .. } => {
                self.input.on_device_event(&event);
            }

            Event::WindowEvent { event, .. } => {
                let _ = self.input.on_window_event(&event, &self.window);

                match event {
                    WindowEvent::CloseRequested => elwt.exit(),

                    WindowEvent::Resized(new_size) => {
                        self.config_sc.width = new_size.width.max(1);
                        self.config_sc.height = new_size.height.max(1);

                        self.surface.configure(self.renderer.device(), &self.config_sc);
                        self.renderer
                            .resize_output(self.config_sc.width, self.config_sc.height);

                        // IMPORTANT: the resize recreated clip_height, so force reupload next frame
                        self.clipmap.invalidate_all();

                        self.has_prev_vp = false;
                    }

                    _ => {}
                }
            }

            _ => {}
        }
    }

    pub fn frame(&mut self, elwt: &EventLoopWindowTarget<()>) {
        let frame_t0 = Instant::now();

        // dt (seconds)
        let now = Instant::now();
        let mut dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        dt = dt.clamp(0.0, 0.05);

        // 0) input -> camera look only (mouse), NO translation here
        self.camera.integrate_input(&mut self.input, 0.0);

        // 1) camera mode toggle + physics step + camera update
        let t0 = Instant::now();

        let shoot = self.input.take_lmb_pressed();

        let balls_gpu: Vec<BallGpu> = self.physics
            .balls_iter()
            .take(config::MAX_BALLS as usize)
            .map(|b| BallGpu {
                center_radius: [b.pos.x, b.pos.y, b.pos.z, config::BALL_RADIUS_M],
                material: 42,
                _pad0: 0, _pad1: 0, _pad2: 0,
            })
            .collect();
        let ball_count: u32 = balls_gpu.len() as u32;
        self.renderer.write_balls(&balls_gpu);

        if self.input.take_c_pressed() {
            self.free_cam = !self.free_cam;

            if self.free_cam {
                let eye = self.physics.player.pos + self.physics.eye_offset;
                self.camera.set_position(eye);
                self.camera.set_yaw_pitch(self.physics.yaw, self.physics.pitch);
            } else {
                let (yaw, pitch) = self.camera.yaw_pitch();
                self.physics.yaw = yaw;
                self.physics.pitch = pitch;
            }
        }

        let q = crate::physics::ChunkManagerQuery {
            mgr: &self.chunks,
            world: Some(self.world.as_ref()),
        };

        let (eye, forward) = if self.free_cam {
            self.physics.step_frame_player_only(dt, &q);
            self.camera.integrate_input(&mut self.input, dt);

            let eye = self.camera.position();
            let fwd = self.camera.forward();
            (eye, fwd)
        } else {
            let eye = self.physics.step_frame(&mut self.input, dt, &q);
            self.camera.set_position(eye);
            self.camera.set_yaw_pitch(self.physics.yaw, self.physics.pitch);

            let fwd = self.camera.forward();
            (eye, fwd)
        };

        if shoot {
            self.physics.spawn_ball(eye, forward);
        }

        self.frame_index = self.frame_index.wrapping_add(1);
        self.profiler.cam(FrameProf::mark_ms(t0));

        // 2) streaming update (THROTTLED)
        let t0 = Instant::now();

        let grid_changed_pump = self.chunks.pump_completed();

        let mut grid_changed = grid_changed_pump;
        if self.last_stream_update.elapsed() >= self.stream_period {
            let cam_pos = self.camera.position();
            let cam_fwd = self.camera.forward();
            grid_changed |= self.chunks.update(&self.world, cam_pos, cam_fwd);
            self.last_stream_update = Instant::now();
        }

        if grid_changed {
            self.renderer.write_chunk_grid(self.chunks.chunk_grid());
        }

        self.profiler.stream(FrameProf::mark_ms(t0));

        // 3) clipmap update (CPU only)
        let t0 = Instant::now();
        let t = self.start_time.elapsed().as_secs_f32();
        let cam_pos = self.camera.position();
        let (clip_params_cpu, clip_uploads) = self.clipmap.update(self.world.as_ref(), cam_pos, t);
        let clip_gpu = ClipmapGpu::from_cpu(&clip_params_cpu);
        self.profiler.clip_update(FrameProf::mark_ms(t0));

        let clip_bytes: usize = clip_uploads
            .iter()
            .map(|u| (u.w as usize) * (u.h as usize) * 2)
            .sum();
        self.profiler.add_clip_uploads(clip_uploads.len(), clip_bytes);

        // 4) camera matrices -> CameraGpu + write
        let t0 = Instant::now();
        let aspect = self.config_sc.width as f32 / self.config_sc.height as f32;
        let cf = self.camera.frame_matrices(aspect);

        let view = cf.view_inv.inverse();
        let proj = cf.proj_inv.inverse();
        let vp = proj * view;

        let prev_vp = if self.has_prev_vp { self.prev_view_proj } else { vp };

        let max_steps = (config::CHUNK_SIZE * 2).clamp(64, 256);
        let (rw, rh) = self.renderer.internal_dims();

        let cam_gpu = CameraGpu {
            view_inv: cf.view_inv.to_cols_array_2d(),
            proj_inv: cf.proj_inv.to_cols_array_2d(),

            view_proj: vp.to_cols_array_2d(),
            prev_view_proj: prev_vp.to_cols_array_2d(),

            cam_pos: [cf.pos.x, cf.pos.y, cf.pos.z, 1.0],

            chunk_size: config::CHUNK_SIZE,
            chunk_count: self.chunks.chunk_count(),
            max_steps,
            frame_index: self.frame_index,

            voxel_params: [config::VOXEL_SIZE_M_F32, t, 2.0, 0.002],

            grid_origin_chunk: [
                self.chunks.grid_origin()[0],
                self.chunks.grid_origin()[1],
                self.chunks.grid_origin()[2],
                0,
            ],
            grid_dims: [
                self.chunks.grid_dims()[0],
                self.chunks.grid_dims()[1],
                self.chunks.grid_dims()[2],
                0,
            ],
            render_present_px: [rw, rh, self.config_sc.width, self.config_sc.height],
            dyn_counts: [ball_count, 0, 0, 0],
        };

        self.renderer.write_camera(&cam_gpu);
        self.profiler.cam_write(FrameProf::mark_ms(t0));

        // 5) fps overlay
        let t0 = Instant::now();
        self.fps_frames += 1;
        let dt_fps = self.fps_last.elapsed().as_secs_f32();
        if dt_fps >= 0.25 {
            let fps = (self.fps_frames as f32) / dt_fps;
            self.fps_value = fps.round() as u32;
            self.fps_frames = 0;
            self.fps_last = Instant::now();
        }

        let overlay = OverlayGpu::from_fps_and_dims(
            self.fps_value,
            self.config_sc.width,
            self.config_sc.height,
            8,
        );
        self.renderer.write_overlay(&overlay);
        self.profiler.overlay(FrameProf::mark_ms(t0));

        // 6) scene uploads if changed
        let t0 = Instant::now();
        let chunk_uploads = self.chunks.take_uploads_budgeted();
        self.profiler.add_chunk_uploads(chunk_uploads.len());

        self.renderer.apply_chunk_uploads(&chunk_uploads);

        let grid_changed2 = self.chunks.commit_uploads_applied(&chunk_uploads);
        if grid_changed2 {
            self.renderer.write_chunk_grid(self.chunks.chunk_grid());
        }

        self.profiler.chunk_up(FrameProf::mark_ms(t0));

        // 7) encode passes
        let mut encoder = self
            .renderer
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });

        // Clipmap uploads + uniform update
        let t0 = Instant::now();
        self.renderer.write_clipmap_updates(&clip_gpu, &clip_uploads);
        self.profiler.enc_clip(FrameProf::mark_ms(t0));

        // Compute passes
        let t0 = Instant::now();
        self.renderer
            .encode_compute(&mut encoder, self.config_sc.width, self.config_sc.height);
        self.profiler.enc_comp(FrameProf::mark_ms(t0));

        // Acquire swapchain as late as possible
        let t0 = Instant::now();
        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(self.renderer.device(), &self.config_sc);
                return;
            }
            Err(wgpu::SurfaceError::Timeout) => return,
            Err(wgpu::SurfaceError::OutOfMemory) => {
                elwt.exit();
                return;
            }
        };
        self.profiler.acq_swapchain(FrameProf::mark_ms(t0));

        let frame_view = frame.texture.create_view(&Default::default());

        // Blit pass
        let t0 = Instant::now();
        self.renderer.encode_blit(&mut encoder, &frame_view);
        self.profiler.enc_blit(FrameProf::mark_ms(t0));

        self.renderer.encode_timestamp_resolve(&mut encoder);

        // submit
        let t0 = Instant::now();
        self.renderer.queue().submit(Some(encoder.finish()));
        self.profiler.submit(FrameProf::mark_ms(t0));

        // poll
        let t0 = Instant::now();
        self.renderer.device().poll(wgpu::Maintain::Poll);
        self.profiler.poll_wait(FrameProf::mark_ms(t0));

        // present
        let t0 = Instant::now();
        frame.present();
        self.profiler.present(FrameProf::mark_ms(t0));

        self.prev_view_proj = vp;
        self.has_prev_vp = true;

        // end-of-frame
        let gpu = if self.profiler.should_print() {
            self.renderer.read_gpu_timings_ms_blocking()
        } else {
            None
        };

        let ss = if self.profiler.should_print() {
            self.chunks.stats()
        } else {
            None
        };

        let frame_ms = frame_t0.elapsed().as_secs_f64() * 1000.0;
        self.profiler.end_frame(frame_ms, ss, gpu);
    }
}


pub fn camera_ray_dir_from_cursor(
    cursor_px: Vec2,
    present_size_px: Vec2,
    proj_inv: Mat4,
    view_inv: Mat4,
) -> Vec3 {
    // NDC (note Y flip matches WGSL)
    let ndc = Vec4::new(
        2.0 * cursor_px.x / present_size_px.x - 1.0,
        1.0 - 2.0 * cursor_px.y / present_size_px.y,
        1.0,
        1.0,
    );

    let view = proj_inv * ndc;
    let vdir = Vec4::new(view.x / view.w, view.y / view.w, view.z / view.w, 0.0);
    let wdir = (view_inv * vdir).truncate();
    wdir.normalize_or_zero()
}

pub async fn run(event_loop: EventLoop<()>, window: Arc<Window>) {
    let mut app = App::new(window).await;

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll); // Poll while profiling

            match &event {
                Event::AboutToWait => {
                    app.window().request_redraw(); // schedule next frame
                }
                Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                    app.frame(elwt); // render here only
                }
                _ => {
                    app.handle_event(event, elwt);
                }
            }
        })
        .unwrap();
}
