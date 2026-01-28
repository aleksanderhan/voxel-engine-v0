// src/app/mod.rs
// --------------

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
    world,
};

pub async fn run(event_loop: EventLoop<()>, window: Arc<Window>) {
    let mut app = App::new(window).await;

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);
            app.handle_event(event, elwt);
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

    input: InputState,
    camera: Camera,

    fps_value: u32,
    fps_frames: u32,
    fps_last: Instant,

    last_chunk: world::ChunkCoord,
    last_frame: Instant,
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

        let (rw, rh) = config::render_dims(config_sc.width, config_sc.height);
        let renderer = Renderer::new(&adapter, surface_format, rw, rh).await;

        surface.configure(renderer.device(), &config_sc);

        let camera = Camera::new(config_sc.width as f32 / config_sc.height as f32);
        let last_chunk = world::ChunkCoord::from_world_pos_m(camera.position());

        let input = InputState::default();

        let mut app = Self {
            window,
            start_time,
            _instance: instance,
            surface,
            _adapter: adapter,
            _surface_format: surface_format,
            config: config_sc,
            renderer,
            input,
            camera,
            fps_value: 0,
            fps_frames: 0,
            fps_last: Instant::now(),
            last_chunk,
            last_frame: Instant::now(),
        };

        app.renderer.update_center_chunk(last_chunk);
        app
    }

    pub fn handle_event(
        &mut self,
        event: Event<()>,
        elwt: &winit::event_loop::EventLoopWindowTarget<()>,
    ) {
        match event {
            Event::DeviceEvent { event, .. } => self.input.on_device_event(&event),

            Event::WindowEvent { event, .. } => {
                let _ = self.input.on_window_event(&event, &self.window);

                match event {
                    WindowEvent::CloseRequested => elwt.exit(),

                    WindowEvent::Resized(new_size) => {
                        self.config.width = new_size.width.max(1);
                        self.config.height = new_size.height.max(1);

                        self.surface.configure(self.renderer.device(), &self.config);
                        self.renderer.resize_output(self.config.width, self.config.height);
                    }

                    _ => {}
                }
            }

            Event::AboutToWait => self.frame(elwt),
            _ => {}
        }
    }

    fn frame(&mut self, elwt: &winit::event_loop::EventLoopWindowTarget<()>) {
        let dt = self.last_frame.elapsed().as_secs_f32();
        self.last_frame = Instant::now();

        self.camera.integrate_input(&mut self.input, dt);

        let mut now_chunk = world::ChunkCoord::from_world_pos_m(self.camera.position());
        now_chunk.y = 0;
        if now_chunk != self.last_chunk {
            self.last_chunk = now_chunk;
            self.renderer.update_center_chunk(now_chunk);
        }

        let aspect = self.config.width as f32 / self.config.height as f32;
        let cf = self.camera.frame_matrices(aspect);
        let t = self.start_time.elapsed().as_secs_f32();

        let fog = 0.02;
        let cam_gpu = CameraGpu {
            view_inv: cf.view_inv.to_cols_array_2d(),
            proj_inv: cf.proj_inv.to_cols_array_2d(),
            cam_pos: [cf.pos.x, cf.pos.y, cf.pos.z, 1.0],
            params: [t, fog, config::VOXEL_SIZE_M_F32, 1337.0],
        };
        self.renderer.write_camera(&cam_gpu);

        // FPS overlay (~4x/sec)
        self.fps_frames += 1;
        let dt_fps = self.fps_last.elapsed().as_secs_f32();
        if dt_fps >= 0.25 {
            let fps = (self.fps_frames as f32) / dt_fps;
            self.fps_value = fps.round() as u32;
            self.fps_frames = 0;
            self.fps_last = Instant::now();
        }

        let overlay = OverlayGpu {
            fps: self.fps_value,
            width: self.config.width,
            height: self.config.height,
            _pad0: 0,
        };
        self.renderer.write_overlay(&overlay);

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

        self.renderer.encode_compute(&mut encoder);
        self.renderer.encode_blit(&mut encoder, &frame_view);

        self.renderer.queue().submit(Some(encoder.finish()));
        frame.present();
    }
}
