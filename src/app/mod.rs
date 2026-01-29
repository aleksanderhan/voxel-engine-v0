// src/app/mod.rs
// --------------
//
// Clipmap fix (A): encode clipmap texture patches + clipmap uniform update
// into the SAME command encoder, BEFORE the compute pass.
//
// This prevents uniforms (origin/offset) from getting ahead of the texture uploads.

use std::sync::Arc;
use std::time::Instant;

use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use crate::{
    camera::Camera,
    clipmap::Clipmap,
    config,
    input::InputState,
    render::{CameraGpu, ClipmapGpu, OverlayGpu, Renderer},
    streaming::ChunkManager,
    world::WorldGen,
};

pub async fn run(event_loop: EventLoop<()>, window: Arc<Window>) {
    let mut app = App::new(window).await;

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Wait);

            match &event {
                Event::AboutToWait => {
                    app.window.request_redraw();
                }
                Event::WindowEvent {
                    event: WindowEvent::RedrawRequested,
                    ..
                } => {
                    app.frame(elwt);
                }
                _ => {
                    app.handle_event(event, elwt);
                }
            }
        })
        .unwrap();
}

pub struct App {
    window: Arc<Window>,
    start_time: Instant,

    _instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    _adapter: wgpu::Adapter,
    _surface_format: wgpu::TextureFormat,
    config: wgpu::SurfaceConfiguration,

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
}

impl App {
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

        let config_sc = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let renderer =
            Renderer::new(&adapter, surface_format, config_sc.width, config_sc.height).await;

        surface.configure(renderer.device(), &config_sc);

        let world = Arc::new(WorldGen::new(12345));
        let chunks = ChunkManager::new(world.clone());

        let camera = Camera::new(config_sc.width as f32 / config_sc.height as f32);
        let input = InputState::default();

        let clipmap = Clipmap::new();

        Self {
            window,
            start_time,
            _instance: instance,
            surface,
            _adapter: adapter,
            _surface_format: surface_format,
            config: config_sc,
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
        }
    }

    pub fn handle_event(
        &mut self,
        event: Event<()>,
        elwt: &winit::event_loop::EventLoopWindowTarget<()>,
    ) {
        match event {
            Event::DeviceEvent { event, .. } => {
                self.input.on_device_event(&event);
            }

            Event::WindowEvent { event, .. } => {
                let _ = self.input.on_window_event(&event, &self.window);

                match event {
                    WindowEvent::CloseRequested => elwt.exit(),

                    WindowEvent::Resized(new_size) => {
                        self.config.width = new_size.width.max(1);
                        self.config.height = new_size.height.max(1);

                        self.surface.configure(self.renderer.device(), &self.config);
                        self.renderer
                            .resize_output(self.config.width, self.config.height);

                        // IMPORTANT: the resize recreated clip_height, so force reupload next frame
                        self.clipmap.invalidate_all();
                    }

                    _ => {}
                }
            }

            Event::AboutToWait => self.frame(elwt),

            _ => {}
        }
    }

    fn frame(&mut self, elwt: &winit::event_loop::EventLoopWindowTarget<()>) {
        // 1) camera integrate
        self.camera.integrate_input(&mut self.input);
        self.frame_index = self.frame_index.wrapping_add(1);

        // 2) streaming update
        let cam_pos = self.camera.position();
        let cam_fwd = self.camera.forward();
        let grid_changed = self.chunks.update(&self.world, cam_pos, cam_fwd);
        if grid_changed {
            self.renderer.write_chunk_grid(self.chunks.chunk_grid());
        }

        // 3) clipmap update (CPU only; DO NOT write to GPU here)
        let t = self.start_time.elapsed().as_secs_f32();
        let (clip_params_cpu, clip_uploads) = self.clipmap.update(self.world.as_ref(), cam_pos, t);
        let clip_gpu = ClipmapGpu::from_cpu(&clip_params_cpu);

        // 4) camera matrices -> CameraGpu
        let aspect = self.config.width as f32 / self.config.height as f32;
        let cf = self.camera.frame_matrices(aspect);

        let max_steps = (config::CHUNK_SIZE * 2).clamp(64, 256);

        let cam_gpu = CameraGpu {
            view_inv: cf.view_inv.to_cols_array_2d(),
            proj_inv: cf.proj_inv.to_cols_array_2d(),
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
        };

        self.renderer.write_camera(&cam_gpu);

        // 5) fps overlay
        self.fps_frames += 1;
        let dt = self.fps_last.elapsed().as_secs_f32();
        if dt >= 0.25 {
            let fps = (self.fps_frames as f32) / dt;
            self.fps_value = fps.round() as u32;
            self.fps_frames = 0;
            self.fps_last = Instant::now();
        }

        let overlay = OverlayGpu::from_fps_and_dims(
            self.fps_value,
            self.config.width,
            self.config.height,
            8, // scale
        );
        self.renderer.write_overlay(&overlay);

        // 6) update scene buffers if changed
        self.renderer.apply_chunk_uploads(self.chunks.take_uploads());

        // 7) acquire frame + encode passes
        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,

            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(self.renderer.device(), &self.config);
                return;
            }

            Err(wgpu::SurfaceError::Timeout) => return,

            Err(wgpu::SurfaceError::OutOfMemory) => {
                elwt.exit();
                return;
            }
        };

        let frame_view = frame.texture.create_view(&Default::default());

        let mut encoder = self
            .renderer
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });

        // IMPORTANT: clipmap uploads + uniform update go into the same encoder,
        // and must happen BEFORE encode_compute() (which samples the clipmap).
        self.renderer
            .encode_clipmap_updates(&mut encoder, &clip_gpu, &clip_uploads);

        self.renderer
            .encode_compute(&mut encoder, self.config.width, self.config.height);

        self.renderer.encode_blit(&mut encoder, &frame_view);

        self.renderer.queue().submit(Some(encoder.finish()));
        frame.present();
    }
}
