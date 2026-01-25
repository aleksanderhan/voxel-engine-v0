//! main.rs
//!
//! Multi-chunk Sparse Voxel Octree (SVO) + GPU ray traversal MVP.
//!
//! What changed vs your 1-chunk version:
//! - CPU builds SVOs for an ACTIVE GRID of chunks around the camera (e.g. 3x3).
//! - CPU packs ALL chunk node arrays into ONE big GPU storage buffer.
//! - CPU uploads a chunk metadata list (origin + node_base) to the GPU.
//! - GPU traces multiple chunks and picks the nearest hit (your WGSL must match the bindings/structs).
//!
//! Notes on abbreviations (first use):
//! - SVO  = Sparse Voxel Octree
//! - DDA  = Digital Differential Analyzer

use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    event::*,
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Fullscreen, WindowBuilder},
};

mod worldgen;
mod svo;

use worldgen::WorldGen;
use svo::{build_chunk_svo_sparse, NodeGpu};

/// Chunk edge length in voxels (power of two).
const CHUNK_SIZE: u32 = 128;

/// Active chunk radius in X/Z:
/// 1 => 3x3 chunks
/// 2 => 5x5 chunks
const ACTIVE_RADIUS: i32 = 1;

/// Chunk metadata passed to GPU (must match WGSL struct).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ChunkMetaGpu {
    /// Chunk origin in world voxel coords (x,y,z) + padding.
    origin: [i32; 4],
    /// Base index into packed nodes buffer.
    node_base: u32,
    /// Node count (optional; useful for debugging).
    node_count: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Camera / scene parameters passed to the compute shader.
///
/// We pass inverse matrices so WGSL can reconstruct rays:
/// - proj_inv: clip/NDC -> view
/// - view_inv: view -> world
///
/// Multi-chunk change:
/// - no single chunk_origin here anymore
/// - chunk_count tells WGSL how many entries are in the chunk meta buffer
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraGpu {
    view_inv: [[f32; 4]; 4],
    proj_inv: [[f32; 4]; 4],
    cam_pos: [f32; 4],

    chunk_size: u32,
    chunk_count: u32,
    max_steps: u32,
    _pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OverlayGpu {
    fps: u32,
    width: u32,
    height: u32,
    _pad0: u32,
}

#[derive(Default)]
struct KeyState {
    w: bool,
    a: bool,
    s: bool,
    d: bool,
    space: bool,
    alt: bool,
}

impl KeyState {
    fn set(&mut self, code: KeyCode, down: bool) {
        match code {
            KeyCode::KeyW => self.w = down,
            KeyCode::KeyA => self.a = down,
            KeyCode::KeyS => self.s = down,
            KeyCode::KeyD => self.d = down,
            KeyCode::Space => self.space = down,
            KeyCode::AltLeft | KeyCode::AltRight => self.alt = down,
            _ => {}
        }
    }
}

struct OutputTex {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
}

fn create_output_texture(device: &wgpu::Device, w: u32, h: u32) -> OutputTex {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("output_tex"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    OutputTex { tex, view }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("SVO MVP (multi-chunk)")
            .with_inner_size(PhysicalSize::new(1280, 720))
            .build(&event_loop)
            .unwrap(),
    );

    window.set_fullscreen(Some(Fullscreen::Borderless(None)));

    pollster::block_on(run(event_loop, window));
}

async fn run(event_loop: EventLoop<()>, window: Arc<winit::window::Window>) {
    // --- WGPU init ------------------------------------------------------------
    let size = window.inner_size();
    let instance = wgpu::Instance::default();

    let surface_window = window.clone();
    let surface = instance.create_surface(surface_window.as_ref()).unwrap();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
        })
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .unwrap();

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps.formats[0];

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: surface_caps.present_modes[0],
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    // --- World + chunk state --------------------------------------------------
    let gen = WorldGen::new(12345);

    // Center chunk coordinate currently loaded (cx, cz).
    let mut active_center = (i32::MIN, i32::MIN);

    // Packed nodes and chunk metadata for ACTIVE grid.
    let mut nodes_packed: Vec<NodeGpu> = Vec::new();
    let mut chunks_meta: Vec<ChunkMetaGpu> = Vec::new();

    // Dummy buffers (replaced on first chunk build).
    let mut node_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("svo_nodes_packed"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let mut chunk_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("chunk_meta"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    // --- Camera state ---------------------------------------------------------
    let mut camera_pos = Vec3::new(CHUNK_SIZE as f32 * 0.5, 20.0, -20.0);
    let mut yaw: f32 = 0.0;
    let mut pitch: f32 = 0.15;
    let mut focused = false;

    let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("camera_buf"),
        size: std::mem::size_of::<CameraGpu>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let overlay_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("overlay_buf"),
        size: std::mem::size_of::<OverlayGpu>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut fps_value: u32 = 0;
    let mut fps_frames: u32 = 0;
    let mut fps_last = Instant::now();

    // --- Output texture -------------------------------------------------------
    let mut output = create_output_texture(&device, config.width, config.height);

    // --- Shaders --------------------------------------------------------------
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ray_cs"),
        source: wgpu::ShaderSource::Wgsl(include_str!("ray_cs.wgsl").into()),
    });
    let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("blit_fs"),
        source: wgpu::ShaderSource::Wgsl(include_str!("blit.wgsl").into()),
    });

    // --- Bind group layouts ---------------------------------------------------
    //
    // IMPORTANT: This assumes your WGSL bindings are:
    //   @group(0) @binding(0) cam uniform
    //   @group(0) @binding(1) chunks meta storage (read)
    //   @group(0) @binding(2) nodes storage (read)
    //   @group(0) @binding(3) output storage texture (write)
    //
    let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("compute_bgl"),
        entries: &[
            // cam uniform
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // chunks meta
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // packed nodes
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // output storage texture
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });

    let mut compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute_bg"),
        layout: &compute_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: chunk_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: node_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&output.view),
            },
        ],
    });

    let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("blit_bgl"),
        entries: &[
            // sampled output texture
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // sampler
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // overlay uniform
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // --- Pipelines ------------------------------------------------------------
    let compute_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("compute_pl"),
        bind_group_layouts: &[&compute_bgl],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute_pipeline"),
        layout: Some(&compute_pl),
        module: &cs_module,
        entry_point: "main",
        compilation_options: Default::default(),
    });

    let blit_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("blit_pl"),
        bind_group_layouts: &[&blit_bgl],
        push_constant_ranges: &[],
    });

    let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("blit_pipeline"),
        layout: Some(&blit_pl),
        vertex: wgpu::VertexState {
            module: &fs_module,
            entry_point: "vs_main",
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &fs_module,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    // --- Blit resources -------------------------------------------------------
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("nearest_sampler"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let mut blit_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blit_bg"),
        layout: &blit_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&output.view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: overlay_buf.as_entire_binding(),
            },
        ],
    });

    // --- Input state ----------------------------------------------------------
    let mut keys = KeyState::default();

    // --- Main loop ------------------------------------------------------------
    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);

            match event {
                Event::DeviceEvent { event, .. } => {
                    if focused {
                        if let DeviceEvent::MouseMotion { delta } = event {
                            let (dx, dy) = (delta.0 as f32, delta.1 as f32);
                            let sens = 0.0025;
                            yaw -= dx * sens;
                            pitch = (pitch - dy * sens).clamp(-1.55, 1.55);
                        }
                    }
                }

                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => elwt.exit(),

                    WindowEvent::Resized(new_size) => {
                        config.width = new_size.width.max(1);
                        config.height = new_size.height.max(1);
                        surface.configure(&device, &config);

                        // IMPORTANT: recreate output texture on resize
                        output = create_output_texture(&device, config.width, config.height);

                        // Rebuild blit bind group with new output view
                        blit_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("blit_bg"),
                            layout: &blit_bgl,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(&output.view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(&sampler),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: overlay_buf.as_entire_binding(),
                                },
                            ],
                        });

                        // Rebuild compute bind group with new output view (buffers unchanged)
                        compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("compute_bg"),
                            layout: &compute_bgl,
                            entries: &[
                                wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 1, resource: chunk_buf.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 2, resource: node_buf.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&output.view) },
                            ],
                        });
                    }

                    WindowEvent::Focused(f) => {
                        focused = f;
                        if focused {
                            let _ = window
                                .set_cursor_grab(CursorGrabMode::Locked)
                                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
                            window.set_cursor_visible(false);
                        } else {
                            let _ = window.set_cursor_grab(CursorGrabMode::None);
                            window.set_cursor_visible(true);
                        }
                    }

                    WindowEvent::KeyboardInput { event, .. } => {
                        if let KeyEvent {
                            physical_key: PhysicalKey::Code(code),
                            state,
                            ..
                        } = event
                        {
                            let down = state == ElementState::Pressed;
                            keys.set(code, down);

                            if down && code == KeyCode::Escape {
                                focused = false;
                                let _ = window.set_cursor_grab(CursorGrabMode::None);
                                window.set_cursor_visible(true);
                            }
                        }
                    }

                    _ => {}
                },

                Event::AboutToWait => {
                    // Camera basis (right-handed)
                    let forward = Vec3::new(
                        yaw.sin() * pitch.cos(),
                        pitch.sin(),
                        yaw.cos() * pitch.cos(),
                    )
                    .normalize();

                    let right = forward.cross(Vec3::Y).normalize();
                    let up = right.cross(forward).normalize();

                    // Movement (still per-frame)
                    let mut vel = Vec3::ZERO;
                    if keys.w { vel += forward; }
                    if keys.s { vel -= forward; }
                    if keys.d { vel += right; }
                    if keys.a { vel -= right; }
                    if keys.space { vel += up; }
                    if keys.alt { vel -= up; }

                    if vel.length_squared() > 0.0 {
                        camera_pos += vel.normalize() * 0.35;
                    }

                    // Matrices for ray reconstruction
                    let view = Mat4::look_at_rh(camera_pos, camera_pos + forward, Vec3::Y);
                    let proj = Mat4::perspective_rh(
                        60.0f32.to_radians(),
                        config.width as f32 / config.height as f32,
                        0.1,
                        1000.0,
                    );

                    // Which chunk are we centered in?
                    let ccx = (camera_pos.x.floor() as i32).div_euclid(CHUNK_SIZE as i32);
                    let ccz = (camera_pos.z.floor() as i32).div_euclid(CHUNK_SIZE as i32);

                    // Rebuild the ACTIVE grid only when center chunk changes.
                    if (ccx, ccz) != active_center {
                        active_center = (ccx, ccz);

                        nodes_packed.clear();
                        chunks_meta.clear();

                        for dz in -ACTIVE_RADIUS..=ACTIVE_RADIUS {
                            for dx in -ACTIVE_RADIUS..=ACTIVE_RADIUS {
                                let cx = ccx + dx;
                                let cz = ccz + dz;

                                let origin = [
                                    cx * CHUNK_SIZE as i32,
                                    0,
                                    cz * CHUNK_SIZE as i32,
                                ];

                                let node_base = nodes_packed.len() as u32;
                                let nodes = build_chunk_svo_sparse(&gen, origin, CHUNK_SIZE);

                                nodes_packed.extend_from_slice(&nodes);

                                chunks_meta.push(ChunkMetaGpu {
                                    origin: [origin[0], origin[1], origin[2], 0],
                                    node_base,
                                    node_count: nodes.len() as u32,
                                    _pad0: 0,
                                    _pad1: 0,
                                });
                            }
                        }

                        node_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("svo_nodes_packed"),
                            contents: bytemuck::cast_slice(&nodes_packed),
                            usage: wgpu::BufferUsages::STORAGE,
                        });

                        chunk_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("chunk_meta"),
                            contents: bytemuck::cast_slice(&chunks_meta),
                            usage: wgpu::BufferUsages::STORAGE,
                        });

                        // Rebuild compute bind group because node_buf/chunk_buf changed
                        compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("compute_bg"),
                            layout: &compute_bgl,
                            entries: &[
                                wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 1, resource: chunk_buf.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 2, resource: node_buf.as_entire_binding() },
                                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&output.view) },
                            ],
                        });
                    }

                    // Upload camera + chunk_count for WGSL loop
                    let cam_gpu = CameraGpu {
                        view_inv: view.inverse().to_cols_array_2d(),
                        proj_inv: proj.inverse().to_cols_array_2d(),
                        cam_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 1.0],

                        chunk_size: CHUNK_SIZE,
                        chunk_count: chunks_meta.len() as u32,
                        max_steps: 128, // multi-chunk: keep this lower; increase if you miss hits
                        _pad0: 0,
                    };
                    queue.write_buffer(&camera_buf, 0, bytemuck::bytes_of(&cam_gpu));

                    // FPS overlay
                    fps_frames += 1;
                    let dt = fps_last.elapsed().as_secs_f32();
                    if dt >= 0.25 {
                        let fps = (fps_frames as f32) / dt;
                        fps_value = fps.round() as u32;
                        fps_frames = 0;
                        fps_last = Instant::now();
                    }

                    let overlay = OverlayGpu {
                        fps: fps_value,
                        width: config.width,
                        height: config.height,
                        _pad0: 0,
                    };
                    queue.write_buffer(&overlay_buf, 0, bytemuck::bytes_of(&overlay));

                    // Swapchain frame
                    let frame = match surface.get_current_texture() {
                        Ok(f) => f,
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            surface.configure(&device, &config);
                            return;
                        }
                        Err(wgpu::SurfaceError::Timeout) => return,
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            elwt.exit();
                            return;
                        }
                    };
                    let frame_view = frame.texture.create_view(&Default::default());

                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("encoder"),
                    });

                    // Compute pass: raycast into output texture
                    {
                        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("ray_pass"),
                            timestamp_writes: None,
                        });
                        cpass.set_pipeline(&compute_pipeline);
                        cpass.set_bind_group(0, &compute_bg, &[]);
                        let gx = (config.width + 7) / 8;
                        let gy = (config.height + 7) / 8;
                        cpass.dispatch_workgroups(gx, gy, 1);
                    }

                    // Render pass: blit to screen
                    {
                        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("blit_pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &frame_view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                        rpass.set_pipeline(&blit_pipeline);
                        rpass.set_bind_group(0, &blit_bg, &[]);
                        rpass.draw(0..3, 0..1);
                    }

                    queue.submit(Some(encoder.finish()));
                    frame.present();
                }

                _ => {}
            }
        })
        .unwrap();
}
