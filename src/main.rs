//! main.rs
//!
//! Multi-chunk Sparse Voxel Octree (SVO) + GPU ray traversal MVP.
//!
//! Streaming + off-thread build change:
//! - Maintain chunk states + build queue.
//! - Every frame computes desired ACTIVE set (ground-anchored Y).
//! - Add KEEP_RADIUS hysteresis so ground chunks don't vanish while new ones build.
//! - Missing chunks are queued and dispatched to worker threads (priority: near-first).
//! - Worker threads build SVO nodes and send results back.
//! - Main thread repacks READY chunks into GPU buffers when changed.
//! - Compute dispatch ALWAYS runs; sky fallback handles chunk_count==0.
//!
//! Notes on abbreviations (first use):
//! - SVO  = Sparse Voxel Octree
//! - DDA  = Digital Differential Analyzer

use std::collections::{HashMap, HashSet, VecDeque};
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

// Off-thread messaging
use crossbeam_channel::{unbounded, Receiver, Sender};

mod worldgen;
mod svo;

use worldgen::WorldGen;
use svo::{build_chunk_svo_sparse, NodeGpu};

/// Chunk edge length in voxels (power of two).
const CHUNK_SIZE: u32 = 64;

/// Active chunk radius in X/Z:
/// 1 => 3x3 chunks
/// 2 => 5x5 chunks
const ACTIVE_RADIUS: i32 = 2;

/// Keep a larger ring cached to prevent "holes" while chunks build.
const KEEP_RADIUS: i32 = ACTIVE_RADIUS + 2;

const VOXEL_SIZE_M: f32 = 0.10; // 10 cm voxels

/// Worker threads (SVO building).
const WORKER_THREADS: usize = 4;

/// Max jobs in flight (limits memory + CPU spikes).
const MAX_IN_FLIGHT: usize = 8;

const NODE_BUDGET_BYTES: usize = 120 * 1024 * 1024; // 120 MiB safety


/// Chunk metadata passed to GPU (must match WGSL struct).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ChunkMetaGpu {
    origin: [i32; 4],   // 16 bytes
    node_base: u32,     // +4
    node_count: u32,    // +4
    _pad0: u32,         // +4
    _pad1: u32,         // +4  => total 32 bytes
}

/// Camera / scene parameters passed to the compute shader.
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

    voxel_params: [f32; 4],
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

// ------------------------------------------------------------
// Chunk streaming state
// ------------------------------------------------------------

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
struct ChunkKey {
    x: i32,
    y: i32,
    z: i32,
}

struct ChunkCpu {
    nodes: Vec<NodeGpu>, // per-chunk nodes; child_base is chunk-relative (WGSL does node_base + child_base)
}

enum ChunkState {
    Missing,
    Queued,
    Building,
    Ready(ChunkCpu),
}

/// Desired chunk list around a center. Vertical band is tuned for terrain+trees.
fn desired_chunks(center: ChunkKey, radius: i32) -> Vec<ChunkKey> {
    let mut out = Vec::new();

    // Terrain world: keep some below, and enough above for crowns.
    for dy in -1..=2 {
        for dz in -radius..=radius {
            for dx in -radius..=radius {
                out.push(ChunkKey {
                    x: center.x + dx,
                    y: center.y + dy,
                    z: center.z + dz,
                });
            }
        }
    }
    out
}

// Job + result messages
#[derive(Clone, Copy, Debug)]
struct BuildJob {
    key: ChunkKey,
}

struct BuildDone {
    key: ChunkKey,
    nodes: Vec<NodeGpu>,
}

fn spawn_workers(gen: Arc<WorldGen>, rx_job: Receiver<BuildJob>, tx_done: Sender<BuildDone>) {
    for _ in 0..WORKER_THREADS {
        let gen = gen.clone();
        let rx_job = rx_job.clone();
        let tx_done = tx_done.clone();

        std::thread::spawn(move || {
            while let Ok(job) = rx_job.recv() {
                let k = job.key;
                let origin = [
                    k.x * CHUNK_SIZE as i32,
                    k.y * CHUNK_SIZE as i32,
                    k.z * CHUNK_SIZE as i32,
                ];

                let nodes = build_chunk_svo_sparse(&gen, origin, CHUNK_SIZE);

                if tx_done.send(BuildDone { key: k, nodes }).is_err() {
                    break;
                }
            }
        });
    }
}

fn sort_queue_near_first(queue: &mut VecDeque<ChunkKey>, center: ChunkKey) {
    let mut v: Vec<ChunkKey> = queue.drain(..).collect();
    v.sort_by_key(|k| {
        // prioritize XZ strongly; Y moderately
        (k.x - center.x).abs() + (k.z - center.z).abs() + 2 * (k.y - center.y).abs()
    });
    queue.extend(v);
}

fn main() {
    let event_loop = EventLoop::new().unwrap();

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("SVO MVP (multi-chunk, streaming, workers)")
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

    let adapter_limits = adapter.limits();

    // Request as much as the adapter supports (within reason).
    let required_limits = wgpu::Limits {
        max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
        max_buffer_size: adapter_limits.max_buffer_size,
        ..wgpu::Limits::default()
    };

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("device"),
                required_features: wgpu::Features::empty(),
                required_limits,
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

    // --- World + streaming chunk state ---------------------------------------
    let gen = Arc::new(WorldGen::new(12345));

    let mut chunks: HashMap<ChunkKey, ChunkState> = HashMap::new();
    let mut build_queue: VecDeque<ChunkKey> = VecDeque::new();

    let mut nodes_packed: Vec<NodeGpu> = Vec::new();
    let mut chunks_meta: Vec<ChunkMetaGpu> = Vec::new();

    // Dummy node buffer (non-zero)
    let dummy_node = NodeGpu {
        child_base: 0xFFFF_FFFF,
        child_mask: 0,
        material: 0,
        _pad: 0,
    };
    let mut node_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("svo_nodes_dummy_1"),
        contents: bytemuck::bytes_of(&dummy_node),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Dummy chunk meta buffer (non-zero)
    let dummy_chunk = ChunkMetaGpu {
        origin: [0, 0, 0, 0],
        node_base: 0,
        node_count: 0,
        _pad0: 0,
        _pad1: 0,
    };
    let mut chunk_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chunk_meta_dummy_1"),
        contents: bytemuck::bytes_of(&dummy_chunk),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Worker messaging
    let (tx_job, rx_job) = unbounded::<BuildJob>();
    let (tx_done, rx_done) = unbounded::<BuildDone>();
    spawn_workers(gen.clone(), rx_job, tx_done);
    let mut in_flight: usize = 0;

    // --- Camera state ---------------------------------------------------------
    let mut camera_pos = Vec3::new(
        (CHUNK_SIZE as f32 * VOXEL_SIZE_M) * 0.5,
        20.0,
        -20.0,
    );

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

                        output = create_output_texture(&device, config.width, config.height);

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

                    // Camera position in VOXELS for chunk selection / height queries
                    let cam_vx = (camera_pos.x / VOXEL_SIZE_M).floor() as i32;
                    let cam_vy = (camera_pos.y / VOXEL_SIZE_M).floor() as i32;
                    let cam_vz = (camera_pos.z / VOXEL_SIZE_M).floor() as i32;

                    let ccx = cam_vx.div_euclid(CHUNK_SIZE as i32);
                    let ccz = cam_vz.div_euclid(CHUNK_SIZE as i32);

                    // Ground-anchored streaming Y
                    let ground_y_vox = gen.ground_height(cam_vx, cam_vz);
                    let ground_cy = ground_y_vox.div_euclid(CHUNK_SIZE as i32);

                    // Two rings: desired (ACTIVE) + keep (hysteresis)
                    let center = ChunkKey { x: ccx, y: ground_cy, z: ccz };
                    let desired = desired_chunks(center, ACTIVE_RADIUS);
                    let keep = desired_chunks(center, KEEP_RADIUS);

                    let desired_set: HashSet<ChunkKey> = desired.iter().copied().collect();
                    let keep_set: HashSet<ChunkKey> = keep.iter().copied().collect();

                    let mut changed = false;

                    // Queue missing chunks in desired first (high priority)
                    for k in &desired {
                        match chunks.get(k) {
                            None | Some(ChunkState::Missing) => {
                                chunks.insert(*k, ChunkState::Queued);
                                build_queue.push_back(*k);
                            }
                            _ => {}
                        }
                    }

                    // Optionally queue the keep ring too (low priority)
                    nodes_packed.clear();
                    chunks_meta.clear();

                    let mut used_bytes: usize = 0;
                    let node_stride = std::mem::size_of::<NodeGpu>();

                    for k in &keep {
                        let Some(ChunkState::Ready(cpu)) = chunks.get(k) else { continue; };

                        let chunk_bytes = cpu.nodes.len() * node_stride;
                        if used_bytes + chunk_bytes > NODE_BUDGET_BYTES {
                            continue; // skip this chunk for this frame's packed buffer
                        }

                        let origin = [
                            k.x * CHUNK_SIZE as i32,
                            k.y * CHUNK_SIZE as i32,
                            k.z * CHUNK_SIZE as i32,
                        ];

                        let node_base = nodes_packed.len() as u32;
                        nodes_packed.extend_from_slice(&cpu.nodes);
                        used_bytes += chunk_bytes;

                        chunks_meta.push(ChunkMetaGpu {
                            origin: [origin[0], origin[1], origin[2], 0],
                            node_base,
                            node_count: cpu.nodes.len() as u32,
                            _pad0: 0,
                            _pad1: 0,
                        });
                    }


                    // Only unload chunks that are OUTSIDE keep_set.
                    let keys_snapshot: Vec<ChunkKey> = chunks.keys().copied().collect();
                    for k in keys_snapshot {
                        if !keep_set.contains(&k) {
                            match chunks.get(&k) {
                                Some(ChunkState::Ready(_)) => {
                                    chunks.insert(k, ChunkState::Missing);
                                    changed = true;
                                }
                                Some(ChunkState::Queued) | Some(ChunkState::Building) => {
                                    chunks.insert(k, ChunkState::Missing);
                                }
                                _ => {}
                            }
                        }
                    }

                    // Reorder queue to build near-first (helps prevent holes)
                    sort_queue_near_first(&mut build_queue, center);

                    // Dispatch queued work up to MAX_IN_FLIGHT
                    while in_flight < MAX_IN_FLIGHT {
                        let Some(k) = build_queue.pop_front() else { break; };

                        // If not even in keep_set anymore, drop it.
                        if !keep_set.contains(&k) {
                            chunks.insert(k, ChunkState::Missing);
                            continue;
                        }

                        match chunks.get(&k) {
                            Some(ChunkState::Queued) => {
                                chunks.insert(k, ChunkState::Building);
                                if tx_job.send(BuildJob { key: k }).is_ok() {
                                    in_flight += 1;
                                } else {
                                    chunks.insert(k, ChunkState::Queued);
                                    break;
                                }
                            }
                            _ => {}
                        }
                    }

                    // Harvest finished builds (non-blocking)
                    while let Ok(done) = rx_done.try_recv() {
                        if in_flight > 0 {
                            in_flight -= 1;
                        }

                        // If it finished but is no longer in keep_set, drop it.
                        if !keep_set.contains(&done.key) {
                            chunks.insert(done.key, ChunkState::Missing);
                            continue;
                        }

                        chunks.insert(done.key, ChunkState::Ready(ChunkCpu { nodes: done.nodes }));
                        changed = true;
                    }

                    // Repack READY chunks into GPU buffers if changed.
                    // IMPORTANT: pack from KEEP set, not only desired, to prevent gaps.
                    if changed {
                        nodes_packed.clear();
                        chunks_meta.clear();

                        for k in &keep {
                            if let Some(ChunkState::Ready(cpu)) = chunks.get(k) {
                                let origin = [
                                    k.x * CHUNK_SIZE as i32,
                                    k.y * CHUNK_SIZE as i32,
                                    k.z * CHUNK_SIZE as i32,
                                ];

                                let node_base = nodes_packed.len() as u32;
                                nodes_packed.extend_from_slice(&cpu.nodes);

                                chunks_meta.push(ChunkMetaGpu {
                                    origin: [origin[0], origin[1], origin[2], 0],
                                    node_base,
                                    node_count: cpu.nodes.len() as u32,
                                    _pad0: 0,
                                    _pad1: 0,
                                });
                            }
                        }

                        // Node buffer: ensure non-zero sized
                        let node_bytes: &[u8];
                        let tmp_node: [NodeGpu; 1];

                        if nodes_packed.is_empty() {
                            tmp_node = [dummy_node];
                            node_bytes = bytemuck::cast_slice(&tmp_node);
                        } else {
                            node_bytes = bytemuck::cast_slice(&nodes_packed);
                        }

                        node_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("svo_nodes_packed"),
                            contents: node_bytes,
                            usage: wgpu::BufferUsages::STORAGE,
                        });

                        // Chunk meta buffer: ensure non-zero sized
                        let meta_bytes: &[u8];
                        let tmp_meta: [ChunkMetaGpu; 1];

                        if chunks_meta.is_empty() {
                            tmp_meta = [dummy_chunk];
                            meta_bytes = bytemuck::cast_slice(&tmp_meta);
                        } else {
                            meta_bytes = bytemuck::cast_slice(&chunks_meta);
                        }

                        chunk_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("chunk_meta"),
                            contents: meta_bytes,
                            usage: wgpu::BufferUsages::STORAGE,
                        });

                        // Rebuild compute bind group because buffers changed
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
                        chunk_count: chunks_meta.len() as u32, // READY chunks packed (keep ring)
                        max_steps: 128,
                        _pad0: 0,

                        voxel_params: [VOXEL_SIZE_M, 0.0, 0.0, 0.0],
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

                    // Compute pass: ALWAYS dispatch (sky fallback covers chunk_count==0)
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
