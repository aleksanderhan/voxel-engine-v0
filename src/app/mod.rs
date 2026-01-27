// src/app/mod.rs
//
// High-level application glue:
// - Owns the window + wgpu surface configuration (swapchain-ish state).
// - Owns input + camera state.
// - Owns world/chunk streaming state.
// - Delegates all GPU resource ownership and rendering work to `Renderer`.
//
// The core loop is driven by winit events. We run with `ControlFlow::Poll`,
// so we continually render (and handle input) as fast as the system allows.

use std::sync::Arc;
use std::time::Instant;

use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use crate::{
    camera::Camera,
    config,
    input::InputState,
    render::{CameraGpu, OverlayGpu, Renderer},
    streaming::ChunkManager,
    world::WorldGen,
};

/// Entrypoint called by main:
/// - Builds the `App` (async, because wgpu adapter/device acquisition is async).
/// - Starts the winit event loop and forwards events into the app.
pub async fn run(event_loop: EventLoop<()>, window: Arc<Window>) {
    let mut app = App::new(window).await;

    // winit's `run` never returns in normal operation (it exits the process/event loop).
    // We set control flow to Poll so we get `AboutToWait` continuously and can render frames.
    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);
            app.handle_event(event, elwt);
        })
        .unwrap();
}

/// Application state that lives for the duration of the event loop.
///
/// Ownership split:
/// - `App` owns the *presentation surface* and surface config (format/size/present mode).
/// - `Renderer` owns the wgpu `Device` and `Queue` and all GPU resources/pipelines.
///   This keeps GPU ownership centralized and makes `App` mostly orchestration.
pub struct App {
    /// Shared window handle (Arc keeps it alive as long as needed).
    window: Arc<Window>,

    /// Time origin for animations / time-based shader params.
    start_time: Instant,

    // --- WGPU "presentation" state (kept here, not in Renderer) ---
    //
    // NOTE: The underscore-prefixed fields are intentionally retained so their
    // lifetimes match expectations and to avoid "never read" warnings.
    //
    // - Instance: top-level wgpu context (backend selection, surface creation).
    // - Surface: platform swapchain surface tied to the window.
    // - Adapter: selected physical GPU.
    // - Surface format: chosen swapchain pixel format.
    // - Surface configuration: width/height/present mode/usage, etc.
    _instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    _adapter: wgpu::Adapter,
    _surface_format: wgpu::TextureFormat,
    config: wgpu::SurfaceConfiguration,

    /// All actual GPU pipelines/resources and per-frame encoding helpers.
    renderer: Renderer,

    // --- World + streaming state ---
    /// Procedural world generator (shared, so streaming can keep refs cheaply).
    world: Arc<WorldGen>,
    /// Chunk streaming / residency manager (decides what to load/unload/upload).
    chunks: ChunkManager,

    // --- Interaction state ---
    /// Aggregated input state updated from winit events.
    input: InputState,
    /// Camera pose/orientation and input integration.
    camera: Camera,

    // --- FPS overlay bookkeeping ---
    /// Most recent FPS value (rounded).
    fps_value: u32,
    /// Frames counted since last FPS update.
    fps_frames: u32,
    /// Timestamp of last FPS update sample window.
    fps_last: Instant,
}

impl App {
    /// Build a new `App`.
    ///
    /// Steps:
    /// 1) Create wgpu instance + surface from the window.
    /// 2) Pick adapter (GPU).
    /// 3) Choose surface format + configure surface for the initial window size.
    /// 4) Create `Renderer` (device/queue + pipelines).
    /// 5) Create world generator, chunk manager, camera, and input.
    pub async fn new(window: Arc<Window>) -> Self {
        let start_time = Instant::now();

        // Initial size (may be zero on some platforms during resize/minimize, so we clamp later).
        let size = window.inner_size();

        // Create the wgpu instance (selects backend internally).
        let instance = wgpu::Instance::default();

        // Surface must outlive the event loop; Arc<Window> makes this easy.
        // The 'static surface lifetime comes from the fact the window is kept alive.
        let surface = instance.create_surface(window.clone()).unwrap();

        // Ask wgpu to pick an adapter (GPU) that can present to this surface.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                // Prefer a discrete / high perf GPU if available.
                power_preference: wgpu::PowerPreference::HighPerformance,
                // Don't force a fallback adapter (software/low-feature) unless needed.
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        // Query what the surface supports with this adapter (formats, modes, etc.).
        let surface_caps = surface.get_capabilities(&adapter);

        // Pick the first advertised format.
        // (Often this is a sensible SRGB-ish format, but you could choose based on preference.)
        let surface_format = surface_caps.formats[0];

        // Create the surface configuration ("swapchain config"):
        // - usage: we render into it as a render attachment.
        // - width/height: clamp to >= 1 to avoid invalid zero-sized surfaces.
        // - present_mode/alpha_mode: choose first supported.
        // - desired_maximum_frame_latency: reduce queued frames (helps latency).
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

        // Renderer owns the real device/queue; App owns surface/config.
        // Renderer is async because it requests/creates the wgpu Device/Queue.
        let renderer =
            Renderer::new(&adapter, surface_format, config_sc.width, config_sc.height).await;

        // Configure once, with the real device we will render with.
        surface.configure(renderer.device(), &config_sc);

        // Create world generation & chunk streaming controller.
        let world = Arc::new(WorldGen::new(12345));
        let chunks = ChunkManager::new(world.clone());

        // Camera starts with an initial aspect ratio derived from the surface size.
        let camera = Camera::new(config_sc.width as f32 / config_sc.height as f32);

        // Input begins empty (no keys pressed, no mouse delta, etc.).
        let input = InputState::default();

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
            input,
            camera,
            fps_value: 0,
            fps_frames: 0,
            fps_last: Instant::now(),
        }
    }

    /// Central event dispatcher called by the event loop.
    ///
    /// We route:
    /// - `DeviceEvent` into raw input (mouse motion, etc.).
    /// - `WindowEvent` into window-related input and resize/close handling.
    /// - `AboutToWait` as a "tick" to render a frame.
    pub fn handle_event(
        &mut self,
        event: Event<()>,
        elwt: &winit::event_loop::EventLoopWindowTarget<()>,
    ) {
        match event {
            // DeviceEvent fires for raw input independent of focus/window coords
            // (e.g. mouse delta from high precision devices).
            Event::DeviceEvent { event, .. } => {
                self.input.on_device_event(&event);
            }

            // WindowEvent includes keyboard, mouse buttons, focus, resize, etc.
            Event::WindowEvent { event, .. } => {
                // Let input layer consume/track events first (focus changes, key state, etc.).
                // Return value is ignored here, but could indicate "consumed".
                let _ = self.input.on_window_event(&event, &self.window);

                match event {
                    // OS requested the window close (Alt+F4, close button, etc.).
                    WindowEvent::CloseRequested => elwt.exit(),

                    // Window was resized; update surface config and renderer output.
                    WindowEvent::Resized(new_size) => {
                        // Clamp to avoid 0-sized surfaces when minimized.
                        self.config.width = new_size.width.max(1);
                        self.config.height = new_size.height.max(1);

                        // Reconfigure the surface swapchain and notify renderer
                        // so any size-dependent textures can be resized.
                        self.surface.configure(self.renderer.device(), &self.config);
                        self.renderer
                            .resize_output(self.config.width, self.config.height);
                    }

                    _ => {}
                }
            }

            // `AboutToWait` is emitted once winit has processed all pending events
            // and is about to sleep. Under Poll control flow, this is effectively
            // our per-frame callback.
            Event::AboutToWait => self.frame(elwt),

            _ => {}
        }
    }

    /// Render/update one frame.
    ///
    /// Pipeline:
    /// 1) Integrate input into camera state.
    /// 2) Update chunk streaming decisions based on camera position/forward.
    /// 3) Build GPU camera parameters (inverse matrices, chunk grid info, etc.).
    /// 4) Update FPS overlay values and write overlay uniforms.
    /// 5) Upload any newly-streamed chunk data to GPU.
    /// 6) Acquire swapchain image, encode compute + blit passes, submit, present.
    fn frame(&mut self, elwt: &winit::event_loop::EventLoopWindowTarget<()>) {
        // 1) camera integrate
        //
        // Convert accumulated input state (keys/mouse deltas) into updated camera pose.
        self.camera.integrate_input(&mut self.input);

        // 2) streaming update
        //
        // Use camera pose to decide which chunks should be present/resident.
        // Then write the chunk grid metadata (addresses/ids) to the renderer.
        let cam_pos = self.camera.position();
        let cam_fwd = self.camera.forward();
        let grid_changed = self.chunks.update(&self.world, cam_pos, cam_fwd);
        if grid_changed {
            self.renderer.write_chunk_grid(self.chunks.chunk_grid());
        }


        // 3) camera matrices -> CameraGpu
        //
        // Compute view/projection matrices and pack their inverses for shader usage.
        // Inverse matrices let shaders go from screen-space rays back into world-space.
        let aspect = self.config.width as f32 / self.config.height as f32;
        let cf = self.camera.frame_matrices(aspect);

        // Elapsed time since startup (typically used for animation/noise jitter/etc.).
        let t = self.start_time.elapsed().as_secs_f32();

        // Raymarch/trace step limit: derived from chunk size but clamped to a sane band.
        // (Avoids tiny chunk sizes producing too few steps, and huge sizes producing too many.)
        let max_steps = (config::CHUNK_SIZE * 2).clamp(48, 96);

        // Camera uniform/SSBO payload for GPU.
        let cam_gpu = CameraGpu {
            view_inv: cf.view_inv.to_cols_array_2d(),
            proj_inv: cf.proj_inv.to_cols_array_2d(),
            cam_pos: [cf.pos.x, cf.pos.y, cf.pos.z, 1.0],

            // Chunking parameters used by shaders to interpret the streamed grid.
            chunk_size: config::CHUNK_SIZE,
            chunk_count: self.chunks.chunk_count(),
            max_steps,
            _pad0: 0,

            // Misc voxel/shader params:
            // [voxel_size_in_meters, time, ?, ?] (the last two likely tune lighting/density).
            voxel_params: [config::VOXEL_SIZE_M_F32, t, 2.0, 0.002],

            // Chunk grid origin and dimensions in chunk coordinates.
            // Packed as ivec-ish arrays with a trailing padding element.
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

        // Upload camera params to GPU.
        self.renderer.write_camera(&cam_gpu);

        // 4) fps overlay
        //
        // We update FPS roughly 4 times per second (every 0.25s) to smooth noise.
        self.fps_frames += 1;
        let dt = self.fps_last.elapsed().as_secs_f32();
        if dt >= 0.25 {
            let fps = (self.fps_frames as f32) / dt;
            self.fps_value = fps.round() as u32;
            self.fps_frames = 0;
            self.fps_last = Instant::now();
        }

        // Overlay uniform payload for the GPU overlay pass.
        let overlay = OverlayGpu {
            fps: self.fps_value,
            width: self.config.width,
            height: self.config.height,
            _pad0: 0,
        };
        self.renderer.write_overlay(&overlay);

        // 5) update scene buffers if changed
        //
        // If the chunk manager has produced new/updated chunk data (meshes/voxels/etc),
        // apply those uploads to GPU resources before encoding this frame.
        self.renderer.apply_chunk_uploads(self.chunks.take_uploads());

        // 6) acquire frame + encode passes
        //
        // Acquire the next drawable surface texture. Handle common surface errors.
        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,

            // Surface got invalidated (resize, display mode change, etc.) -> reconfigure.
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(self.renderer.device(), &self.config);
                return;
            }

            // Temporary issue: skip this frame.
            Err(wgpu::SurfaceError::Timeout) => return,

            // Fatal-ish: GPU memory exhaustion -> exit.
            Err(wgpu::SurfaceError::OutOfMemory) => {
                elwt.exit();
                return;
            }
        };

        // View into the swapchain image used as render target in the blit pass.
        let frame_view = frame.texture.create_view(&Default::default());

        // Command encoder collects GPU commands for this frame into a single submission.
        let mut encoder = self
            .renderer
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });

        // First: run compute (likely raymarching / voxel traversal) into an offscreen target.
        self.renderer
            .encode_compute(&mut encoder, self.config.width, self.config.height);

        // Then: blit (copy/compose) the offscreen output into the swapchain image.
        self.renderer.encode_blit(&mut encoder, &frame_view);

        // Submit GPU work and present the swapchain image.
        self.renderer.queue().submit(Some(encoder.finish()));
        frame.present();
    }
}
