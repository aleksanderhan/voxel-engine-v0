// src/app/app.rs
// --------------
//
// Application loop + per-frame orchestration.
//
// Key rule for correctness:
// - Clipmap texture patch uploads and the clipmap uniform update must be encoded
//   in the same command encoder, before the compute pass. This prevents the
//   uniform (origin/offset) from getting ahead of the texture data on the GPU.

use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::{Vec2, Vec3, Vec4};

use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget},
    window::Window,
};

use super::profiler;
use crate::app::camera::Camera;
use crate::app::config;
use crate::app::input::InputState;
use crate::{
    clipmap::Clipmap,
    render::{gpu_types::PRIMARY_PROFILE_COUNT, CameraGpu, ClipmapGpu, OverlayGpu, Renderer},
    streaming::ChunkManager,
    world::WorldGen,
};
use crate::world::materials::{AIR, DIRT, STONE, WOOD, LIGHT};

pub async fn run(event_loop: EventLoop<()>, window: Arc<Window>) {
    let mut app = App::new(window).await;

    event_loop
        .run(move |event, elwt| {
            // Poll yields the lowest latency and best profiling determinism.
            elwt.set_control_flow(ControlFlow::Poll);

            match &event {
                Event::AboutToWait => {
                    // Schedule the next frame; the redraw will arrive as WindowEvent::RedrawRequested.
                    app.window.request_redraw();
                }
                Event::WindowEvent {
                    event: WindowEvent::RedrawRequested,
                    ..
                } => {
                    // Render only on RedrawRequested (keeps event handling lightweight).
                    app.render_frame(elwt);
                }
                _ => {
                    app.handle_event(event, elwt);
                }
            }
        })
        .unwrap();
}

struct ClipmapCpuUpdate {
    clip_gpu: ClipmapGpu,
    clip_uploads: Vec<crate::clipmap::ClipmapUpload>,
    time_seconds: f32,
}

pub struct App {
    window: Arc<Window>,
    start_time: Instant,

    // Keep handles alive for the app lifetime.
    _instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    _adapter: wgpu::Adapter,
    _surface_format: wgpu::TextureFormat,
    surface_config: wgpu::SurfaceConfiguration,

    renderer: Renderer,

    world: Arc<WorldGen>,
    chunks: ChunkManager,
    clipmap: Clipmap,

    input: InputState,
    camera: Camera,

    // FPS display (smoothed in a coarse window).
    fps_value: u32,
    fps_frames: u32,
    fps_last_update: Instant,

    frame_index: u32,

    profiler: profiler::FrameProf,
    last_frame_time: Instant,

    // Motion vectors / temporal reprojection needs last frame's VP.
    prev_view_proj: glam::Mat4,
    has_prev_view_proj: bool,

    // Streaming update throttle.
    last_stream_update: Instant,
    stream_period: Duration,

    physics: crate::physics::Physics,
    free_cam: bool,

    edit_mode: usize,
    edit_modes: Vec<EditMode>,

    show_profile_hud: bool,
    primary_profile_counts: [u32; PRIMARY_PROFILE_COUNT],
}

#[derive(Clone, Copy)]
enum EditMode {
    Dig,        // set hit voxel to AIR
    Place(u32), // place material into prev-voxel
}

impl App {
    pub async fn new(window: Arc<Window>) -> Self {
        let start_time = Instant::now();
        let initial_size = window.inner_size();

        // --- GPU/Surface bootstrap -----------------------------------------------------------
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
        let present_mode = choose_present_mode(&surface_caps);
        println!("present_mode = {:?}", present_mode);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: initial_size.width.max(1),
            height: initial_size.height.max(1),
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let renderer =
            Renderer::new(&adapter, surface_format, surface_config.width, surface_config.height)
                .await;

        surface.configure(renderer.device(), &surface_config);

        // --- World / streaming / camera ------------------------------------------------------
        let world = Arc::new(WorldGen::new(12345));
        let chunks = ChunkManager::new(world.clone());

        let camera = Camera::new(surface_config.width as f32 / surface_config.height as f32);
        let input = InputState::default();

        let clipmap = Clipmap::new();

        let start_pos = Vec3::new(
            (config::CHUNK_SIZE as f32 * config::VOXEL_SIZE_M_F32) * 0.5,
            50.0,
            -20.0,
        );
        let physics = crate::physics::Physics::new(start_pos);

        let edit_modes = vec![
            EditMode::Dig,
            EditMode::Place(DIRT),
            EditMode::Place(STONE),
            EditMode::Place(WOOD),
            EditMode::Place(LIGHT),
        ];

        // --- Profiling (default off) ---------------------------------------------------------
        let (prof_enabled, prof_every) = profiler::settings_from_args();
        let profiler = profiler::FrameProf::new(prof_enabled, prof_every);

        Self {
            window,
            start_time,
            _instance: instance,
            surface,
            _adapter: adapter,
            _surface_format: surface_format,
            surface_config,
            renderer,
            world,
            chunks,
            clipmap,
            input,
            camera,
            fps_value: 0,
            fps_frames: 0,
            fps_last_update: Instant::now(),
            frame_index: 0,
            profiler,
            last_frame_time: Instant::now(),
            prev_view_proj: glam::Mat4::IDENTITY,
            has_prev_view_proj: false,
            last_stream_update: Instant::now(),
            stream_period: Duration::from_millis(33), // 30 Hz
            physics,
            free_cam: false,
            edit_mode: 0,
            edit_modes,
            show_profile_hud: false,
            primary_profile_counts: [0; PRIMARY_PROFILE_COUNT],
        }
    }

    pub fn handle_event(&mut self, event: Event<()>, elwt: &EventLoopWindowTarget<()>) {
        match event {
            Event::DeviceEvent { event, .. } => {
                self.input.on_device_event(&event);
            }
            Event::WindowEvent { event, .. } => {
                // Input gets first look; it can capture mouse/keyboard state.
                let _ = self.input.on_window_event(&event, &self.window);

                match event {
                    WindowEvent::CloseRequested => elwt.exit(),

                    WindowEvent::Resized(new_size) => {
                        self.handle_resize(new_size);
                    }

                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn handle_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        // WGPU requires non-zero surface size.
        self.surface_config.width = new_size.width.max(1);
        self.surface_config.height = new_size.height.max(1);

        self.surface
            .configure(self.renderer.device(), &self.surface_config);
        self.renderer
            .resize_output(self.surface_config.width, self.surface_config.height);

        // Resize recreates internal clip-height resources, so force a full reupload next frame.
        self.clipmap.invalidate_all();

        // Temporal history is invalid after resize due to projection mismatch.
        self.has_prev_view_proj = false;
    }

    fn render_frame(&mut self, elwt: &EventLoopWindowTarget<()>) {
        let frame_start = Instant::now();

        let delta_seconds = self.compute_frame_dt_seconds();

        self.update_camera(delta_seconds);
        self.update_editor();
        self.update_streaming();
        let clipmap_update = self.update_clipmap_cpu();
        let camera_gpu = self.build_camera_gpu(clipmap_update.time_seconds);
        self.write_overlay_fps();

        self.apply_chunk_uploads_and_refresh_grid();

        // Encode all GPU work (compute + blit) in one submission.
        let mut encoder = self.create_frame_encoder();

        // IMPORTANT ORDER:
        // 1) clipmap texture uploads and clipmap uniform updates
        // 2) compute pass
        // 3) acquire swapchain
        // 4) blit to swapchain
        self.encode_clipmap_updates(&clipmap_update, &mut encoder);
        self.encode_compute_pass(&mut encoder);

        let swapchain_frame = match self.acquire_swapchain_frame(elwt) {
            Some(frame) => frame,
            None => return,
        };
        let swapchain_view = swapchain_frame.texture.create_view(&Default::default());

        self.encode_blit_pass(&swapchain_view, &mut encoder);
        self.renderer.encode_timestamp_resolve(&mut encoder);

        self.submit_and_present(encoder, swapchain_frame);

        // Record temporal history for next frame.
        self.prev_view_proj = glam::Mat4::from_cols_array_2d(&camera_gpu.view_proj);
        self.has_prev_view_proj = true;

        // IMPORTANT: "render_ms" should exclude finish_frame_profiling overhead
        let render_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
        self.finish_frame_profiling(render_ms);
    }

    fn compute_frame_dt_seconds(&mut self) -> f32 {
        let now = Instant::now();
        let raw_dt = (now - self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;

        // Clamp prevents giant dt spikes from causing unstable camera integration.
        raw_dt.clamp(0.0, 0.05)
    }

    fn update_camera(&mut self, delta_seconds: f32) {
        let t0 = self.profiler.start();

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

        let (_eye, _forward) = if self.free_cam {
            // free cam: camera owns mouse + movement
            self.physics.step_frame_player_only(delta_seconds, &q);
            self.camera.integrate_input(&mut self.input, delta_seconds);
            (self.camera.position(), self.camera.forward())
        } else {
            // player cam: physics owns mouse + movement; camera just mirrors
            let eye = self.physics.step_frame(&mut self.input, delta_seconds, &q);
            self.camera.set_position(eye);
            self.camera
                .set_yaw_pitch(self.physics.yaw, self.physics.pitch);
            (eye, self.camera.forward())
        };

        self.frame_index = self.frame_index.wrapping_add(1);
        self.profiler.cam(profiler::FrameProf::end_ms(t0));
    }

    fn update_streaming(&mut self) {
        let t0 = self.profiler.start();

        // Always do cheap housekeeping each frame.
        let grid_changed_by_pump = self.chunks.pump_completed();

        // Do expensive planning at a fixed cadence.
        let mut grid_changed = grid_changed_by_pump;
        if self.last_stream_update.elapsed() >= self.stream_period {
            let camera_position = self.camera.position();
            let camera_forward = self.camera.forward();

            grid_changed |= self
                .chunks
                .update(&self.world, camera_position, camera_forward);

            self.last_stream_update = Instant::now();
        }

        if grid_changed {
            self.renderer.write_chunk_grid(self.chunks.chunk_grid());
        }

        self.profiler.stream(profiler::FrameProf::end_ms(t0));
    }

    fn update_clipmap_cpu(&mut self) -> ClipmapCpuUpdate {
        let t0 = self.profiler.start();

        let time_seconds = self.start_time.elapsed().as_secs_f32();
        let camera_position = self.camera.position();

        let (clip_params_cpu, clip_uploads) =
            self.clipmap.update(self.world.as_ref(), camera_position, time_seconds);
        let clip_gpu = ClipmapGpu::from_cpu(&clip_params_cpu);

        // R16 = 2 bytes per texel.
        let upload_bytes = clip_uploads
            .iter()
            .map(|upload| (upload.w as usize) * (upload.h as usize) * 2)
            .sum::<usize>();

        self.profiler
            .clip_update(profiler::FrameProf::end_ms(t0));
        self.profiler
            .add_clip_uploads(clip_uploads.len(), upload_bytes);

        ClipmapCpuUpdate {
            clip_gpu,
            clip_uploads,
            time_seconds,
        }
    }

    fn build_camera_gpu(&mut self, time_seconds: f32) -> CameraGpu {
        let t0 = self.profiler.start();

        let aspect = self.surface_config.width as f32 / self.surface_config.height as f32;
        let camera_frame = self.camera.frame_matrices(aspect);

        let view = camera_frame.view_inv.inverse();
        let proj = camera_frame.proj_inv.inverse();
        let view_proj = proj * view;

        let previous_view_proj = if self.has_prev_view_proj {
            self.prev_view_proj
        } else {
            view_proj
        };

        let max_steps = (config::CHUNK_SIZE * 2).clamp(64, 256);
        let (internal_width, internal_height) = self.renderer.internal_dims();
        let render_res = Vec2::new(internal_width as f32, internal_height as f32);

        let ray_dir_at = |px: Vec2| -> Vec3 {
            let ndc = Vec4::new(
                2.0 * px.x / render_res.x - 1.0,
                1.0 - 2.0 * px.y / render_res.y,
                1.0,
                1.0,
            );
            let view_pos = camera_frame.proj_inv * ndc;
            let vdir = Vec4::new(
                view_pos.x / view_pos.w,
                view_pos.y / view_pos.w,
                view_pos.z / view_pos.w,
                0.0,
            );
            (camera_frame.view_inv * vdir).truncate()
        };

        let ray00 = ray_dir_at(Vec2::new(0.5, 0.5));
        let ray10 = ray_dir_at(Vec2::new(1.5, 0.5));
        let ray01 = ray_dir_at(Vec2::new(0.5, 1.5));
        let ray_dx = ray10 - ray00;
        let ray_dy = ray01 - ray00;

        let camera_gpu = CameraGpu {
            view_inv: camera_frame.view_inv.to_cols_array_2d(),
            proj_inv: camera_frame.proj_inv.to_cols_array_2d(),

            view_proj: view_proj.to_cols_array_2d(),
            prev_view_proj: previous_view_proj.to_cols_array_2d(),

            cam_pos: [camera_frame.pos.x, camera_frame.pos.y, camera_frame.pos.z, 1.0],
            ray00: [ray00.x, ray00.y, ray00.z, 0.0],
            ray_dx: [ray_dx.x, ray_dx.y, ray_dx.z, 0.0],
            ray_dy: [ray_dy.x, ray_dy.y, ray_dy.z, 0.0],

            chunk_size: config::CHUNK_SIZE,
            chunk_count: self.chunks.chunk_count(),
            max_steps,
            frame_index: self.frame_index,

            // [voxel_size_m, time_seconds, ???, ???]
            // NOTE: Keep exact layout/values stable; shader expects this packing.
            voxel_params: [config::VOXEL_SIZE_M_F32, time_seconds, 2.0, 0.002],

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
            render_present_px: [
                internal_width,
                internal_height,
                self.surface_config.width,
                self.surface_config.height,
            ],
            profile_flags: if self.show_profile_hud { 1 } else { 0 },
            _pad_profile: [0; 3],
            _pad0: [0; 4],
            _pad1: [0; 4],
            _pad2: [0; 4],

        };

        self.renderer.write_camera(&camera_gpu);
        self.profiler.cam_write(profiler::FrameProf::end_ms(t0));

        camera_gpu
    }

    fn write_overlay_fps(&mut self) {
        let t0 = self.profiler.start();

        self.fps_frames += 1;

        let elapsed_seconds = self.fps_last_update.elapsed().as_secs_f32();
        if elapsed_seconds >= 0.25 {
            let fps = (self.fps_frames as f32) / elapsed_seconds;
            self.fps_value = fps.round() as u32;

            self.fps_frames = 0;
            self.fps_last_update = Instant::now();
        }

        let edit_value = match self.edit_modes[self.edit_mode] {
            EditMode::Dig => crate::world::materials::AIR,   // or keep a dedicated “DIG” path if you prefer
            EditMode::Place(m) => m,
        };

        let profile_lines = if self.show_profile_hud {
            self.build_profile_lines()
        } else {
            Vec::new()
        };
        let profile_refs: Vec<&str> = profile_lines.iter().map(String::as_str).collect();

        let overlay = OverlayGpu::from_fps_edit_profile(
            self.fps_value,
            edit_value,
            self.surface_config.width,
            self.surface_config.height,
            8,
            &profile_refs,
        );
        self.renderer.write_overlay(&overlay);

        self.profiler.overlay(profiler::FrameProf::end_ms(t0));
    }

    fn build_profile_lines(&self) -> Vec<String> {
        let counts = self.primary_profile_counts;
        vec![
            format!("VOX {}", Self::format_profile_count(counts[0])),
            format!("GRS {}", Self::format_profile_count(counts[1])),
            format!("HDR {}", Self::format_profile_count(counts[2])),
            format!("FOG {}", Self::format_profile_count(counts[3])),
            format!("SHD {}", Self::format_profile_count(counts[4])),
        ]
    }

    fn format_profile_count(value: u32) -> String {
        let raw = value.to_string();
        let mut out = String::with_capacity(raw.len() + raw.len() / 3);
        let mut count = 0usize;
        for ch in raw.chars().rev() {
            if count != 0 && count % 3 == 0 {
                out.push('.');
            }
            out.push(ch);
            count += 1;
        }
        out.chars().rev().collect()
    }


    fn apply_chunk_uploads_and_refresh_grid(&mut self) {
        let t0 = self.profiler.start();

        let chunk_uploads = self.chunks.take_uploads_budgeted();
        self.profiler.add_chunk_uploads(chunk_uploads.len());

        self.renderer.apply_chunk_uploads(&chunk_uploads);

        // After applying uploads, the chunk grid may change again (e.g. new resident chunks).
        let grid_changed = self.chunks.commit_uploads_applied(&chunk_uploads);
        if grid_changed {
            self.renderer.write_chunk_grid(self.chunks.chunk_grid());
        }

        self.profiler.chunk_up(profiler::FrameProf::end_ms(t0));
    }

    fn create_frame_encoder(&self) -> wgpu::CommandEncoder {
        self.renderer
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame_encoder"),
            })
    }

    fn encode_clipmap_updates(
        &mut self,
        clipmap_update: &ClipmapCpuUpdate,
        _encoder: &mut wgpu::CommandEncoder,
    ) {
        let t0 = self.profiler.start();

        // This call should encode both:
        // - texture uploads (staging -> clipmap textures)
        // - clipmap uniform updates
        // so the compute pass always sees matching data and parameters.
        self.renderer
            .write_clipmap_updates(&clipmap_update.clip_gpu, &clipmap_update.clip_uploads);

        self.profiler.enc_clip(profiler::FrameProf::end_ms(t0));
    }

    fn encode_compute_pass(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let t0 = self.profiler.start();

        if self.show_profile_hud {
            self.renderer.reset_primary_profile_counts();
        }

        self.renderer.encode_compute(
            encoder,
            self.surface_config.width,
            self.surface_config.height,
        );

        if self.show_profile_hud {
            self.renderer.encode_primary_profile_readback(encoder);
        }

        self.profiler.enc_comp(profiler::FrameProf::end_ms(t0));
    }

    fn acquire_swapchain_frame(
        &mut self,
        elwt: &EventLoopWindowTarget<()>,
    ) -> Option<wgpu::SurfaceTexture> {
        let t0 = self.profiler.start();

        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,

            // Recoverable errors: reconfigure and try next frame.
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface
                    .configure(self.renderer.device(), &self.surface_config);
                return None;
            }

            // Transient: skip this frame.
            Err(wgpu::SurfaceError::Timeout) => return None,

            // Fatal: exit the app.
            Err(wgpu::SurfaceError::OutOfMemory) => {
                elwt.exit();
                return None;
            }
        };

        self.profiler
            .acq_swapchain(profiler::FrameProf::end_ms(t0));

        Some(frame)
    }

    fn encode_blit_pass(
        &mut self,
        swapchain_view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let t0 = self.profiler.start();

        self.renderer.encode_blit(encoder, swapchain_view);

        self.profiler.enc_blit(profiler::FrameProf::end_ms(t0));
    }

    fn submit_and_present(&mut self, encoder: wgpu::CommandEncoder, frame: wgpu::SurfaceTexture) {
        // Submit
        let t0 = self.profiler.start();
        self.renderer.queue().submit(Some(encoder.finish()));
        self.profiler.submit(profiler::FrameProf::end_ms(t0));

        // Poll (keeps mapping/timestamp queries flowing)
        let t0 = self.profiler.start();
        self.renderer.device().poll(wgpu::Maintain::Poll);
        self.profiler.poll_wait(profiler::FrameProf::end_ms(t0));

        // Present
        let t0 = self.profiler.start();
        frame.present();
        self.profiler.present(profiler::FrameProf::end_ms(t0));
    }

    fn finish_frame_profiling(&mut self, render_ms: f64) {
        if self.show_profile_hud {
            if let Some(counts) = self.renderer.read_primary_profile_counts_blocking() {
                self.primary_profile_counts = counts;
            }
        }

        if !self.profiler.enabled() {
            return;
        }

        let prof_start = Instant::now();

        let gpu_timings_ms = if self.profiler.should_print() {
            self.renderer.read_gpu_timings_ms_blocking()
        } else {
            None
        };

        let streaming_stats = if self.profiler.should_print() {
            self.chunks.stats()
        } else {
            None
        };

        let prof_overhead_ms = prof_start.elapsed().as_secs_f64() * 1000.0;
        self.profiler.end_frame(render_ms, prof_overhead_ms, streaming_stats, gpu_timings_ms);
    }


    fn update_editor(&mut self) {
        if self.input.take_p_pressed() {
            self.show_profile_hud = !self.show_profile_hud;
        }

        // scroll wheel selects mode
        let steps = self.input.take_wheel_steps();
        if steps != 0 {
            let n = self.edit_modes.len() as i32;
            let mut i = self.edit_mode as i32;
            i = (i + steps).rem_euclid(n);
            self.edit_mode = i as usize;
        }

        // click applies operation (one-shot per click)
        if self.input.take_lmb_pressed() {
            self.apply_edit_click();
        }
    }

    fn enqueue_chunk_rebuild_now(&mut self, key: crate::streaming::types::ChunkKey) {
        let Some(center) = self.chunks.build.last_center else { return; };

        crate::streaming::manager::build::request_edit_refresh(&mut self.chunks, key);
        crate::streaming::manager::build::dispatch_builds(&mut self.chunks, center);
    }

    fn apply_edit_click(&mut self) {
        let eye = self.camera.position();
        let dir = self.camera.forward();

        let max_dist_m = 80.0;
        let voxel = config::VOXEL_SIZE_M_F32;

        let Some((hit_w, place_w, enter_n)) = self.raycast_voxel(eye, dir, max_dist_m, voxel) else {
            return;
        };

        println!("HIT  = {:?} mat={}", hit_w, self.sample_voxel_material(hit_w.0, hit_w.1, hit_w.2));
        println!("PLACE= {:?} mat={}", place_w, self.sample_voxel_material(place_w.0, place_w.1, place_w.2));
        println!("enter_n = {:?}", enter_n);

        // Decide edit operation.
        // Rule:
        // - Dig replaces the HIT voxel with AIR.
        // - Place writes ONLY into the PREV voxel (adjacent empty). Never overwrite solids.
        let (tx, ty, tz, mat) = match self.edit_modes[self.edit_mode] {
            EditMode::Dig => (hit_w.0, hit_w.1, hit_w.2, crate::world::materials::AIR),

            EditMode::Place(m) => {
                // Absolute guarantee: we never write into hit voxel when placing.
                if place_w == hit_w {
                    panic!("BUG: place_w == hit_w  hit={:?} place={:?} enter_n={:?}", hit_w, place_w, enter_n);
                }

                // Only place into empty.
                let place_mat = self.sample_voxel_material(place_w.0, place_w.1, place_w.2);
                if place_mat != AIR {
                    println!("blocked: place_w {:?} is not AIR (mat={})", place_w, place_mat);
                    return;
                }

                (place_w.0, place_w.1, place_w.2, m)
            }

        };

        // chunk key + local coords
        let (key, lx, ly, lz) = crate::world::edits::voxel_to_chunk_local(&self.world, tx, ty, tz);

        println!("edit key={:?} state={:?}", key, self.chunks.build.chunks.get(&key));

        // pin so it will never unload
        self.chunks.pinned.insert(key);

        // write edit override
        self.chunks.edits.apply_voxel(key, lx, ly, lz, mat);

        let (hit_key, hit_lx, hit_ly, hit_lz) =
            crate::world::edits::voxel_to_chunk_local(&self.world, hit_w.0, hit_w.1, hit_w.2);
        let (pl_key, pl_lx, pl_ly, pl_lz) =
            crate::world::edits::voxel_to_chunk_local(&self.world, tx, ty, tz);

        let hit_override = self.chunks.edits.get_override(hit_key, hit_lx, hit_ly, hit_lz);
        let pl_override  = self.chunks.edits.get_override(pl_key, pl_lx, pl_ly, pl_lz);

        println!("override hit  {:?} local=({}, {}, {}) -> {:?}", hit_key, hit_lx, hit_ly, hit_lz, hit_override);
        println!("override place {:?} local=({}, {}, {}) -> {:?}", pl_key, pl_lx, pl_ly, pl_lz, pl_override);


        let verify = self.chunks.edits.get_override(key, lx, ly, lz);
        println!(
            "edit verify key={:?} local=({}, {}, {}) wrote={} readback={:?}",
            key, lx, ly, lz, mat, verify
        );

        // force a rebuild soon
        self.enqueue_chunk_rebuild_now(key);
    }

    fn raycast_voxel(
        &self,
        origin_m: glam::Vec3,
        dir_m: glam::Vec3,
        max_dist_m: f32,
        voxel_m: f32,
    ) -> Option<((i32, i32, i32), (i32, i32, i32), (i32, i32, i32))> {
        let dir = dir_m.normalize();

        // Nudge origin slightly along the ray to avoid starting exactly on voxel boundaries.
        let origin_m = origin_m + dir * (voxel_m * 1e-4);

        // Start voxel (cell containing origin)
        let mut vx = (origin_m.x / voxel_m).floor() as i32;
        let mut vy = (origin_m.y / voxel_m).floor() as i32;
        let mut vz = (origin_m.z / voxel_m).floor() as i32;

        let step_x = if dir.x >= 0.0 { 1 } else { -1 };
        let step_y = if dir.y >= 0.0 { 1 } else { -1 };
        let step_z = if dir.z >= 0.0 { 1 } else { -1 };

        let next_boundary = |v: i32, step: i32| -> f32 {
            if step > 0 {
                (v as f32 + 1.0) * voxel_m
            } else {
                (v as f32) * voxel_m
            }
        };

        let mut t_max_x = if dir.x.abs() < 1e-6 {
            f32::INFINITY
        } else {
            (next_boundary(vx, step_x) - origin_m.x) / dir.x
        };
        let mut t_max_y = if dir.y.abs() < 1e-6 {
            f32::INFINITY
        } else {
            (next_boundary(vy, step_y) - origin_m.y) / dir.y
        };
        let mut t_max_z = if dir.z.abs() < 1e-6 {
            f32::INFINITY
        } else {
            (next_boundary(vz, step_z) - origin_m.z) / dir.z
        };

        let t_delta_x = if dir.x.abs() < 1e-6 {
            f32::INFINITY
        } else {
            voxel_m / dir.x.abs()
        };
        let t_delta_y = if dir.y.abs() < 1e-6 {
            f32::INFINITY
        } else {
            voxel_m / dir.y.abs()
        };
        let t_delta_z = if dir.z.abs() < 1e-6 {
            f32::INFINITY
        } else {
            voxel_m / dir.z.abs()
        };

        let max_t = max_dist_m;

        // Deterministic single-axis stepping:
        // - We step EXACTLY ONE axis per iteration.
        // - On ties we use a stable priority (X then Y then Z).
        // This removes edge/corner ambiguity that causes placement to "flip".
        for _ in 0..512 {
            // Choose next crossing axis (stable tie-break)
            let mut axis = 0; // 0=x, 1=y, 2=z
            let mut t_next = t_max_x;

            if t_max_y < t_next || (t_max_y == t_next && axis > 1) {
                axis = 1;
                t_next = t_max_y;
            }
            if t_max_z < t_next || (t_max_z == t_next && axis > 2) {
                axis = 2;
                t_next = t_max_z;
            }

            if t_next > max_t {
                break;
            }

            // Step exactly one axis, and set enter_n to the face we crossed.
            let enter_n = match axis {
                0 => {
                    vx += step_x;
                    t_max_x += t_delta_x;
                    (-step_x, 0, 0)
                }
                1 => {
                    vy += step_y;
                    t_max_y += t_delta_y;
                    (0, -step_y, 0)
                }
                _ => {
                    vz += step_z;
                    t_max_z += t_delta_z;
                    (0, 0, -step_z)
                }
            };

            // Now inside the newly-entered voxel.
            if self.sample_voxel_material(vx, vy, vz) != AIR {
                let hit_w = (vx, vy, vz);
                // Place in the adjacent voxel on the entered face (outside the solid).
                let place_w = (vx + enter_n.0, vy + enter_n.1, vz + enter_n.2);
                return Some((hit_w, place_w, enter_n));
            }
        }

        None
    }


    fn sample_voxel_material(&self, wx: i32, wy: i32, wz: i32) -> u32 {
        let cs = config::CHUNK_SIZE as i32;
        let cx = wx.div_euclid(cs);
        let cy = wy.div_euclid(cs);
        let cz = wz.div_euclid(cs);

        let lx = wx.rem_euclid(cs);
        let ly = wy.rem_euclid(cs);
        let lz = wz.rem_euclid(cs);

        let key = crate::streaming::types::ChunkKey { x: cx, y: cy, z: cz };

        if let Some(m) = self.chunks.edits.get_override(key, lx, ly, lz) {
            return m;
        }

        // Procedural fallback (with edits context available, if WorldGen uses it)
        self.world
            .material_at_voxel_with_edits(&self.chunks.edits, wx, wy, wz)
    }

}

fn choose_present_mode(caps: &wgpu::SurfaceCapabilities) -> wgpu::PresentMode {
    // For profiling: avoid vsync blocking.
    // Use --profile as the switch (or add a dedicated flag).
    let profiling = std::env::args().any(|a| a == "--profile");

    if profiling && caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
        return wgpu::PresentMode::Immediate;
    }

    if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
        wgpu::PresentMode::Mailbox
    } else if caps.present_modes.contains(&wgpu::PresentMode::FifoRelaxed) {
        wgpu::PresentMode::FifoRelaxed
    } else if caps.present_modes.contains(&wgpu::PresentMode::Fifo) {
        wgpu::PresentMode::Fifo
    } else if caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
        wgpu::PresentMode::Immediate
    } else {
        caps.present_modes[0]
    }
}
