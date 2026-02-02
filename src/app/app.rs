// src/app/mod.rs
//
// Application loop + per-frame orchestration.
//
// Key rule for correctness:
// - Clipmap texture patch uploads and the clipmap uniform update must be encoded
//   in the same command encoder, before the compute pass. This prevents the
//   uniform (origin/offset) from getting ahead of the texture data on the GPU.

use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget},
    window::Window,
};

use super::profiler;
use crate::app::input::InputState;
use crate::app::camera::Camera;
use crate::app::config;
use crate::{
    clipmap::Clipmap,
    render::{CameraGpu, ClipmapGpu, OverlayGpu, Renderer},
    streaming::ChunkManager,
    world::WorldGen,
};

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
    upload_bytes: usize,
    time_seconds: f32,
    camera_position: glam::Vec3,
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

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: initial_size.width.max(1),
            height: initial_size.height.max(1),
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 3,
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
            profiler: profiler::FrameProf::new(),
            last_frame_time: Instant::now(),
            prev_view_proj: glam::Mat4::IDENTITY,
            has_prev_view_proj: false,
            last_stream_update: Instant::now(),
            stream_period: Duration::from_millis(33), // 30 Hz
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

        self.finish_frame_profiling(frame_start);
    }

    fn compute_frame_dt_seconds(&mut self) -> f32 {
        let now = Instant::now();
        let raw_dt = (now - self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;

        // Clamp prevents giant dt spikes from causing unstable camera integration.
        raw_dt.clamp(0.0, 0.05)
    }

    fn update_camera(&mut self, delta_seconds: f32) {
        let cpu_start = Instant::now();

        self.camera.integrate_input(&mut self.input, delta_seconds);
        self.frame_index = self.frame_index.wrapping_add(1);

        self.profiler.cam(profiler::FrameProf::mark_ms(cpu_start));
    }

    fn update_streaming(&mut self) {
        let cpu_start = Instant::now();

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

        self.profiler.stream(profiler::FrameProf::mark_ms(cpu_start));
    }

    fn update_clipmap_cpu(&mut self) -> ClipmapCpuUpdate {
        let cpu_start = Instant::now();

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
            .clip_update(profiler::FrameProf::mark_ms(cpu_start));
        self.profiler
            .add_clip_uploads(clip_uploads.len(), upload_bytes);

        ClipmapCpuUpdate {
            clip_gpu,
            clip_uploads,
            upload_bytes,
            time_seconds,
            camera_position,
        }
    }

    fn build_camera_gpu(&mut self, time_seconds: f32) -> CameraGpu {
        let cpu_start = Instant::now();

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

        let camera_gpu = CameraGpu {
            view_inv: camera_frame.view_inv.to_cols_array_2d(),
            proj_inv: camera_frame.proj_inv.to_cols_array_2d(),

            view_proj: view_proj.to_cols_array_2d(),
            prev_view_proj: previous_view_proj.to_cols_array_2d(),

            cam_pos: [camera_frame.pos.x, camera_frame.pos.y, camera_frame.pos.z, 1.0],

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
        };

        self.renderer.write_camera(&camera_gpu);
        self.profiler
            .cam_write(profiler::FrameProf::mark_ms(cpu_start));

        camera_gpu
    }

    fn write_overlay_fps(&mut self) {
        let cpu_start = Instant::now();

        self.fps_frames += 1;

        let elapsed_seconds = self.fps_last_update.elapsed().as_secs_f32();
        if elapsed_seconds >= 0.25 {
            let fps = (self.fps_frames as f32) / elapsed_seconds;
            self.fps_value = fps.round() as u32;

            self.fps_frames = 0;
            self.fps_last_update = Instant::now();
        }

        let overlay = OverlayGpu::from_fps_and_dims(
            self.fps_value,
            self.surface_config.width,
            self.surface_config.height,
            8,
        );
        self.renderer.write_overlay(&overlay);

        self.profiler
            .overlay(profiler::FrameProf::mark_ms(cpu_start));
    }

    fn apply_chunk_uploads_and_refresh_grid(&mut self) {
        let cpu_start = Instant::now();

        let chunk_uploads = self.chunks.take_uploads_budgeted();
        self.profiler.add_chunk_uploads(chunk_uploads.len());

        self.renderer.apply_chunk_uploads(&chunk_uploads);

        // After applying uploads, the chunk grid may change again (e.g. new resident chunks).
        let grid_changed = self.chunks.commit_uploads_applied(&chunk_uploads);
        if grid_changed {
            self.renderer.write_chunk_grid(self.chunks.chunk_grid());
        }

        self.profiler
            .chunk_up(profiler::FrameProf::mark_ms(cpu_start));
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
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let cpu_start = Instant::now();

        // This call should encode both:
        // - texture uploads (staging -> clipmap textures)
        // - clipmap uniform updates
        // so the compute pass always sees matching data and parameters.
        self.renderer
            .write_clipmap_updates(&clipmap_update.clip_gpu, &clipmap_update.clip_uploads);

        self.profiler
            .enc_clip(profiler::FrameProf::mark_ms(cpu_start));
    }

    fn encode_compute_pass(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let cpu_start = Instant::now();

        self.renderer.encode_compute(
            encoder,
            self.surface_config.width,
            self.surface_config.height,
        );

        self.profiler
            .enc_comp(profiler::FrameProf::mark_ms(cpu_start));
    }

    fn acquire_swapchain_frame(
        &mut self,
        elwt: &EventLoopWindowTarget<()>,
    ) -> Option<wgpu::SurfaceTexture> {
        let cpu_start = Instant::now();

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
            .acq_swapchain(profiler::FrameProf::mark_ms(cpu_start));

        Some(frame)
    }

    fn encode_blit_pass(&mut self, swapchain_view: &wgpu::TextureView, encoder: &mut wgpu::CommandEncoder) {
        let cpu_start = Instant::now();

        self.renderer.encode_blit(encoder, swapchain_view);

        self.profiler
            .enc_blit(profiler::FrameProf::mark_ms(cpu_start));
    }

    fn submit_and_present(
        &mut self,
        encoder: wgpu::CommandEncoder,
        frame: wgpu::SurfaceTexture,
    ) {
        // Submit
        let cpu_start = Instant::now();
        self.renderer.queue().submit(Some(encoder.finish()));
        self.profiler
            .submit(profiler::FrameProf::mark_ms(cpu_start));

        // Poll (keeps mapping/timestamp queries flowing)
        let cpu_start = Instant::now();
        self.renderer.device().poll(wgpu::Maintain::Poll);
        self.profiler
            .poll_wait(profiler::FrameProf::mark_ms(cpu_start));

        // Present
        let cpu_start = Instant::now();
        frame.present();
        self.profiler
            .present(profiler::FrameProf::mark_ms(cpu_start));
    }

    fn finish_frame_profiling(&mut self, frame_start: Instant) {
        // Readback timings only when printing; keep default path cheap.
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

        let frame_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
        self.profiler.end_frame(frame_ms, streaming_stats, gpu_timings_ms);
    }
}

fn choose_present_mode(surface_caps: &wgpu::SurfaceCapabilities) -> wgpu::PresentMode {
    // Mailbox: low latency + avoids tearing (if supported).
    // Fifo: always supported, vsync.
    // Immediate: lowest latency but can tear.
    if surface_caps
        .present_modes
        .contains(&wgpu::PresentMode::Mailbox)
    {
        wgpu::PresentMode::Mailbox
    } else if surface_caps.present_modes.contains(&wgpu::PresentMode::Fifo) {
        wgpu::PresentMode::Fifo
    } else if surface_caps
        .present_modes
        .contains(&wgpu::PresentMode::Immediate)
    {
        wgpu::PresentMode::Immediate
    } else {
        surface_caps.present_modes[0]
    }
}
