//! main.rs
//!
//! Minimal Sparse Voxel Octree (SVO) + GPU ray-march MVP.
//!
//! What this keeps (the important stuff):
//! - CPU builds an SVO for ONE chunk.
//! - GPU compute shader does ray traversal with 3D DDA (Digital Differential Analyzer).
//! - Each stepped voxel queries the SVO (point lookup) for material.
//! - Output is written to a storage texture and blitted fullscreen.
//!
//! Scene content (simple, deterministic):
//! - A single-layer floor at y==0 (material 1)
//! - A few voxels above it (material 2)

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
use svo::build_chunk_svo_sparse;


/// Chunk edge length in voxels (must be a power of two for a balanced octree).
/// 32 keeps the MVP fast and the octree small.
const CHUNK_SIZE: u32 = 32;

/// GPU node layout: must match the WGSL struct exactly.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct NodeGpu {
    /// Index of first child if internal node; 0xFFFF_FFFF means leaf.
    child_base: u32,
    /// Bitmask for which children exist (bits 0..7).
    child_mask: u32,
    /// Material id for leaf nodes (0 = empty / air).
    material: u32,
    /// Padding for 16-byte alignment.
    _pad: u32,
}

/// Camera / scene parameters passed to the compute shader.
///
/// We pass inverse matrices so WGSL can reconstruct rays:
/// - proj_inv: clip/NDC -> view
/// - view_inv: view -> world
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraGpu {
    view_inv: [[f32; 4]; 4],
    proj_inv: [[f32; 4]; 4],
    cam_pos: [f32; 4],

    /// Chunk origin in world voxel coords (integer).
    chunk_origin: [i32; 4],

    chunk_size: u32,
    max_steps: u32,
    _pad0: u32,
    _pad1: u32,
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
    _tex: wgpu::Texture,
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
    OutputTex { _tex: tex, view }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("SVO MVP (minimal)")
            .with_inner_size(PhysicalSize::new(1280, 720))
            .build(&event_loop)
            .unwrap(),
    );

    // Borderless fullscreen tends to be the most consistent cross-platform “fullscreen”.
    window.set_fullscreen(Some(Fullscreen::Borderless(None)));

    pollster::block_on(run(event_loop, window));
}

async fn run(event_loop: EventLoop<()>, window: Arc<winit::window::Window>) {
    // --- WGPU init ------------------------------------------------------------
    let size = window.inner_size();
    let instance = wgpu::Instance::default();

    // Surface holds references tied to the window; cloning the Arc avoids borrow/move issues.
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

    // --- CPU builds the SVO ----------------------
    let gen = WorldGen::new(12345);

    // One chunk for now. Later: origin = [chunk_x * CHUNK_SIZE as i32, 0, chunk_z * ...]
    let chunk_origin = [0, 0, 0];
    let nodes = build_chunk_svo_sparse(&gen, chunk_origin, CHUNK_SIZE);

    let node_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("svo_nodes"),
        contents: bytemuck::cast_slice(&nodes),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // --- Camera state ---------------------------------------------------------
    let mut camera_pos = Vec3::new(CHUNK_SIZE as f32 * 0.5, 8.0, -20.0);
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
    let output = create_output_texture(&device, config.width, config.height);

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
            // nodes
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
            // output storage texture
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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
                resource: node_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
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
            // overlay uniform (frame counter)
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
                // Raw mouse deltas (FPS-style look). Only apply when focused.
                Event::DeviceEvent { event, .. } => {
                    if focused {
                        if let DeviceEvent::MouseMotion { delta } = event {
                            let (dx, dy) = (delta.0 as f32, delta.1 as f32);
                            let sens = 0.0025;

                            // Mouse mapping:
                            // - Right moves camera right: yaw decreases (you tuned this)
                            // - Up/down sign varies by platform; this uses +dy to match your setup.
                            yaw -= dx * sens;
                            pitch = (pitch - dy * sens).clamp(-1.55, 1.55);
                        }
                    }
                }

                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => elwt.exit(),

                    // Resize: rebuild swapchain + output texture and rebind for blit.
                    WindowEvent::Resized(new_size) => {
                        config.width = new_size.width.max(1);
                        config.height = new_size.height.max(1);
                        surface.configure(&device, &config);

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

                        compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("compute_bg"),
                            layout: &compute_bgl,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: camera_buf.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: node_buf.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::TextureView(&output.view),
                                },
                            ],
                        });

                    }

                    // Focus: grab/hide cursor so raw mouse motion can drive camera.
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

                    // Keyboard: WASD + Space/Alt. Esc releases mouse look.
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

                // Frame tick: update camera, dispatch compute, blit.
                Event::AboutToWait => {
                    // Camera basis (right-handed):
                    let forward = Vec3::new(
                        yaw.sin() * pitch.cos(),
                        pitch.sin(),
                        yaw.cos() * pitch.cos(),
                    )
                    .normalize();

                    // This cross order preserves your A/D direction.
                    let right = forward.cross(Vec3::Y).normalize();
                    let up = right.cross(forward).normalize();

                    // Simple “constant speed per frame” movement (later you add delta time).
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

                    let cam_gpu = CameraGpu {
                        view_inv: view.inverse().to_cols_array_2d(),
                        proj_inv: proj.inverse().to_cols_array_2d(),
                        cam_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 1.0],
                        chunk_origin: [chunk_origin[0], chunk_origin[1], chunk_origin[2], 0],
                        chunk_size: CHUNK_SIZE,
                        max_steps: 512,
                        _pad0: 0,
                        _pad1: 0,
                    };
                    queue.write_buffer(&camera_buf, 0, bytemuck::bytes_of(&cam_gpu));

                    fps_frames += 1;
                    let dt = fps_last.elapsed().as_secs_f32();

                    // Update 4 times per second for stable numbers
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
                        Err(_) => {
                            surface.configure(&device, &config);
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

/// Build a tiny deterministic scene:
/// - Floor: y==0 everywhere in the chunk (material 1)
/// - A few “blocks” above the floor (material 2)
///
/// Then build a full SVO by recursive uniformity testing.
fn build_simple_svo_chunk() -> Vec<NodeGpu> {
    fn material_at(x: i32, y: i32, z: i32) -> u32 {
        if y == 0 { return 1; }
        if (x == 8 && z == 8 && (y == 1 || y == 2 || y == 3))
            || (x == 12 && z == 10 && (y == 1 || y == 2))
            || (x == 16 && z == 14 && y == 1)
            || (x == 10 && z == 18 && (y == 1 || y == 2 || y == 3 || y == 4))
        { return 2; }
        0
    }

    fn is_empty_leaf(n: &NodeGpu) -> bool {
        n.child_base == 0xFFFF_FFFF && n.material == 0
    }

    fn build_node(
        nodes: &mut Vec<NodeGpu>,
        ox: i32, oy: i32, oz: i32,
        size: i32,
        mat_fn: &dyn Fn(i32,i32,i32)->u32,
    ) -> NodeGpu {
        // Uniformity test
        let first = mat_fn(ox, oy, oz);
        let mut uniform = true;
        'outer: for z in oz..(oz + size) {
            for y in oy..(oy + size) {
                for x in ox..(ox + size) {
                    if mat_fn(x, y, z) != first {
                        uniform = false;
                        break 'outer;
                    }
                }
            }
        }

        if uniform || size == 1 {
            return NodeGpu {
                child_base: 0xFFFF_FFFF,
                child_mask: 0,
                material: first,
                _pad: 0,
            };
        }

        let half = size / 2;

        // Build the 8 child ROOT structs first (their subtrees get appended during recursion)
        let mut child_roots: [NodeGpu; 8] = [NodeGpu {
            child_base: 0xFFFF_FFFF, child_mask: 0, material: 0, _pad: 0
        }; 8];

        for ci in 0..8 {
            let dx = if (ci & 1) != 0 { half } else { 0 };
            let dy = if (ci & 2) != 0 { half } else { 0 };
            let dz = if (ci & 4) != 0 { half } else { 0 };

            child_roots[ci] = build_node(nodes, ox + dx, oy + dy, oz + dz, half, mat_fn);
        }

        // Compact: append only present children roots contiguously (in ci order)
        let mut mask: u32 = 0;
        let base = nodes.len() as u32;

        for ci in 0..8 {
            if !is_empty_leaf(&child_roots[ci]) {
                mask |= 1u32 << ci;
                nodes.push(child_roots[ci]);
            }
        }

        // If everything was empty, collapse to empty leaf
        if mask == 0 {
            return NodeGpu {
                child_base: 0xFFFF_FFFF,
                child_mask: 0,
                material: 0,
                _pad: 0,
            };
        }

        NodeGpu {
            child_base: base,
            child_mask: mask,
            material: 0,
            _pad: 0,
        }
    }

    // Root at index 0 must exist for GPU
    let mut nodes = vec![NodeGpu {
        child_base: 0xFFFF_FFFF,
        child_mask: 0,
        material: 0,
        _pad: 0,
    }];

    let root = build_node(&mut nodes, 0, 0, 0, CHUNK_SIZE as i32, &material_at);
    nodes[0] = root;
    nodes
}
