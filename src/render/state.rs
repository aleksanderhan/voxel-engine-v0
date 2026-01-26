// src/render/state.rs
//
// Multi-pass compute renderer with persistent GPU buffers.
//
// Compute passes:
//   1) primary   (full-res): trace + fog -> writes color_tex (rgba16f) + depth_tex (r32f)
//   2) godray    (quarter):  shafts + temporal accumulation -> writes godray ping-pong (rgba16f)
//   3) composite (full-res): color + upsampled godray -> writes output (rgba16f)
//
// Render pass:
//   blit: samples output to the swapchain.
//
// IMPORTANT (wgpu rule):
//   A texture bound for STORAGE (write) is an exclusive usage within a dispatch.
//   Therefore godray pass must NOT bind primary_bg (which includes depth/color storage outputs),
//   otherwise depth_tex conflicts with being sampled.
//
// Bind groups:
//   group(0) primary_bg: camera + buffers + primary outputs (only for primary pass)
//   group(0) scene_bg  : camera + buffers only            (for godray pass)
//   group(1) godray_bg : depth sampled + history sampled + out storage
//   group(2) composite_bg: color sampled + godray sampled + output storage
//   group(0/1) empty_bg: empty bind groups to satisfy group indices for composite pipeline layout

use crate::{
    config,
    render::{
        gpu_types::{CameraGpu, ChunkMetaGpu, NodeGpu, OverlayGpu},
        resources::{create_output_texture, OutputTex},
        shaders,
    },
    streaming::ChunkUpload,
};

struct Tex2D {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
}

fn make_tex2d(
    device: &wgpu::Device,
    label: &str,
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> Tex2D {
    let w = w.max(1);
    let h = h.max(1);

    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    });

    let view = tex.create_view(&Default::default());
    Tex2D {
        tex,
        view,
        w,
        h,
        format,
    }
}

fn quarter_dim(x: u32) -> u32 {
    (x + 3) / 4
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,

    // final output texture used by composite (storage) and sampled by blit
    output: OutputTex,

    // intermediate textures
    color_tex: Tex2D, // full-res rgba16f
    depth_tex: Tex2D, // full-res r32f
    godray_a: Tex2D,  // quarter-res rgba16f
    godray_b: Tex2D,  // quarter-res rgba16f

    // pipelines
    primary_pipeline: wgpu::ComputePipeline,
    godray_pipeline: wgpu::ComputePipeline,
    composite_pipeline: wgpu::ComputePipeline,
    blit_pipeline: wgpu::RenderPipeline,

    // layouts
    primary_bgl: wgpu::BindGroupLayout,   // group(0) with storage outputs
    scene_bgl: wgpu::BindGroupLayout,     // group(0) without storage outputs
    godray_bgl: wgpu::BindGroupLayout,    // group(1)
    composite_bgl: wgpu::BindGroupLayout, // group(2)
    empty_bgl: wgpu::BindGroupLayout,     // empty for group(0/1) in composite pipeline layout
    blit_bgl: wgpu::BindGroupLayout,

    // resources
    sampler: wgpu::Sampler,

    camera_buf: wgpu::Buffer,
    overlay_buf: wgpu::Buffer,

    // persistent storage buffers
    node_buf: wgpu::Buffer,
    chunk_buf: wgpu::Buffer,
    chunk_grid_buf: wgpu::Buffer,

    // bind groups
    primary_bg: wgpu::BindGroup,     // group(0) for primary pass (includes storage outputs)
    scene_bg: wgpu::BindGroup,       // group(0) for godray pass (camera + buffers only)
    godray_bg_ab: wgpu::BindGroup,   // group(1) hist=A -> out=B
    godray_bg_ba: wgpu::BindGroup,   // group(1) hist=B -> out=A
    composite_bg_a: wgpu::BindGroup, // group(2) reads godray A
    composite_bg_b: wgpu::BindGroup, // group(2) reads godray B
    empty_bg: wgpu::BindGroup,       // empty bind group for composite group(0/1)
    blit_bg: wgpu::BindGroup,

    // ping-pong flag: true => AB this frame (write B), false => BA (write A)
    use_ab: bool,

    // capacity bookkeeping (in elements)
    node_capacity: u32,
    chunk_capacity: u32,
    grid_capacity: u32,

    // dummies (kept around)
    dummy_node: NodeGpu,
    dummy_chunk: ChunkMetaGpu,
}

impl Renderer {
    pub async fn new(
        adapter: &wgpu::Adapter,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        // keep adapter limits, but ensure max buffer sizes are set from adapter
        let adapter_limits = adapter.limits();
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

        // final output (composite writes here; blit samples it)
        let output = create_output_texture(&device, width.max(1), height.max(1));

        // intermediates
        let color_tex = make_tex2d(
            &device,
            "color_tex",
            width,
            height,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let depth_tex = make_tex2d(
            &device,
            "depth_tex",
            width,
            height,
            wgpu::TextureFormat::R32Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let qw = quarter_dim(width);
        let qh = quarter_dim(height);

        let godray_a = make_tex2d(
            &device,
            "godray_a",
            qw,
            qh,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let godray_b = make_tex2d(
            &device,
            "godray_b",
            qw,
            qh,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        // shaders
        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ray_cs"),
            source: wgpu::ShaderSource::Wgsl(shaders::ray_cs_wgsl().into()),
        });

        let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(shaders::blit_wgsl().into()),
        });

        // sampler for blit
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("nearest_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // uniforms
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

        // dummies
        let dummy_node = NodeGpu {
            child_base: 0xFFFF_FFFF,
            child_mask: 0,
            material: 0,
            _pad: 0,
        };
        let dummy_chunk = ChunkMetaGpu {
            origin: [0, 0, 0, 0],
            node_base: 0,
            node_count: 0,
            _pad0: 0,
            _pad1: 0,
        };

        // ---------------------------------------------------------------------
        // Persistent buffers
        // ---------------------------------------------------------------------
        let node_capacity =
            (config::NODE_BUDGET_BYTES / std::mem::size_of::<NodeGpu>()) as u32;
        let node_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("svo_nodes_arena"),
            size: (node_capacity as u64) * (std::mem::size_of::<NodeGpu>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Persistent chunk meta buffer capacity = max possible resident chunks in KEEP box
        let chunk_capacity =
            (2 * config::KEEP_RADIUS + 1) as u32 * 4u32 * (2 * config::KEEP_RADIUS + 1) as u32;

        let chunk_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk_meta_persistent"),
            size: (chunk_capacity as u64) * (std::mem::size_of::<ChunkMetaGpu>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Chunk grid buffer (fixed-capacity)
        let grid_capacity = chunk_capacity; // nx * ny * nz
        let chunk_grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk_grid_buf"),
            size: (grid_capacity as u64) * (std::mem::size_of::<u32>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---------------------------------------------------------------------
        // Bind group layouts
        // ---------------------------------------------------------------------

        // group(0) PRIMARY: camera + buffers + primary outputs (storage)
        let primary_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("primary_bgl"),
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
                // nodes arena
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
                // chunk grid
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // color_img storage texture (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // depth_img storage texture (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // group(0) SCENE: camera + buffers only (no storage textures!)
        let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene_bgl"),
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
                // nodes arena
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
                // chunk grid
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // group(1) GODRAY: depth sampled + hist sampled + out storage
        let godray_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("godray_bgl"),
            entries: &[
                // depth_tex sampled (r32f)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // godray_hist sampled (rgba16f)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // godray_out storage (rgba16f)
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

        // group(2) COMPOSITE: color sampled + godray sampled + output storage
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite_bgl"),
            entries: &[
                // color_tex sampled
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // godray_tex sampled
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // out_img storage
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

        // empty layout for group(0) and group(1) in composite pipeline layout
        let empty_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("empty_bgl"),
            entries: &[],
        });

        // blit layout
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

        // ---------------------------------------------------------------------
        // Pipelines
        // ---------------------------------------------------------------------

        let primary_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("primary_pl"),
            bind_group_layouts: &[&primary_bgl],
            push_constant_ranges: &[],
        });

        let primary_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("primary_pipeline"),
            layout: Some(&primary_pl),
            module: &cs_module,
            entry_point: "main_primary",
            compilation_options: Default::default(),
        });

        // Godray pass uses group(0)=scene_bgl and group(1)=godray_bgl
        let godray_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("godray_pl"),
            bind_group_layouts: &[&scene_bgl, &godray_bgl],
            push_constant_ranges: &[],
        });

        let godray_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("godray_pipeline"),
            layout: Some(&godray_pl),
            module: &cs_module,
            entry_point: "main_godray",
            compilation_options: Default::default(),
        });

        // Composite uses @group(2), so pipeline layout must include group(0), group(1), group(2).
        // We bind empty BGs for group(0/1).
        let composite_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite_pl"),
            bind_group_layouts: &[&empty_bgl, &empty_bgl, &composite_bgl],
            push_constant_ranges: &[],
        });

        let composite_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("composite_pipeline"),
            layout: Some(&composite_pl),
            module: &cs_module,
            entry_point: "main_composite",
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
                    format: surface_format,
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

        // ---------------------------------------------------------------------
        // Bind groups
        // ---------------------------------------------------------------------

        let primary_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("primary_bg"),
            layout: &primary_bgl,
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
                    resource: chunk_grid_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&color_tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&depth_tex.view),
                },
            ],
        });

        let scene_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_bg"),
            layout: &scene_bgl,
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
                    resource: chunk_grid_buf.as_entire_binding(),
                },
            ],
        });

        let godray_bg_ab = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("godray_bg_ab"),
            layout: &godray_bgl,
            entries: &[
                // depth sampled
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&depth_tex.view),
                },
                // hist = A
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&godray_a.view),
                },
                // out = B
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&godray_b.view),
                },
            ],
        });

        let godray_bg_ba = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("godray_bg_ba"),
            layout: &godray_bgl,
            entries: &[
                // depth sampled
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&depth_tex.view),
                },
                // hist = B
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&godray_b.view),
                },
                // out = A
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&godray_a.view),
                },
            ],
        });

        let composite_bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite_bg_a"),
            layout: &composite_bgl,
            entries: &[
                // color sampled
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&color_tex.view),
                },
                // godray sampled (A)
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&godray_a.view),
                },
                // output storage
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&output.view),
                },
            ],
        });

        let composite_bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite_bg_b"),
            layout: &composite_bgl,
            entries: &[
                // color sampled
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&color_tex.view),
                },
                // godray sampled (B)
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&godray_b.view),
                },
                // output storage
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&output.view),
                },
            ],
        });

        let empty_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("empty_bg"),
            layout: &empty_bgl,
            entries: &[],
        });

        let blit_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
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

        Self {
            device,
            queue,

            output,

            color_tex,
            depth_tex,
            godray_a,
            godray_b,

            primary_pipeline,
            godray_pipeline,
            composite_pipeline,
            blit_pipeline,

            primary_bgl,
            scene_bgl,
            godray_bgl,
            composite_bgl,
            empty_bgl,
            blit_bgl,

            sampler,

            camera_buf,
            overlay_buf,

            node_buf,
            chunk_buf,
            chunk_grid_buf,

            primary_bg,
            scene_bg,
            godray_bg_ab,
            godray_bg_ba,
            composite_bg_a,
            composite_bg_b,
            empty_bg,
            blit_bg,

            use_ab: true,

            node_capacity,
            chunk_capacity,
            grid_capacity,

            dummy_node,
            dummy_chunk,
        }
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn resize_output(&mut self, width: u32, height: u32) {
        // recreate textures
        self.output = create_output_texture(&self.device, width.max(1), height.max(1));

        self.color_tex = make_tex2d(
            &self.device,
            "color_tex",
            width,
            height,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        self.depth_tex = make_tex2d(
            &self.device,
            "depth_tex",
            width,
            height,
            wgpu::TextureFormat::R32Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let qw = quarter_dim(width);
        let qh = quarter_dim(height);

        self.godray_a = make_tex2d(
            &self.device,
            "godray_a",
            qw,
            qh,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        self.godray_b = make_tex2d(
            &self.device,
            "godray_b",
            qw,
            qh,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        // rebuild bind groups that reference texture views

        self.primary_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("primary_bg"),
            layout: &self.primary_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.chunk_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.node_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.chunk_grid_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.color_tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&self.depth_tex.view),
                },
            ],
        });

        // scene_bg doesn’t depend on texture views, but rebuilding is fine.
        self.scene_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_bg"),
            layout: &self.scene_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.chunk_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.node_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.chunk_grid_buf.as_entire_binding(),
                },
            ],
        });

        self.godray_bg_ab = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("godray_bg_ab"),
            layout: &self.godray_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.depth_tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.godray_a.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.godray_b.view),
                },
            ],
        });

        self.godray_bg_ba = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("godray_bg_ba"),
            layout: &self.godray_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.depth_tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.godray_b.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.godray_a.view),
                },
            ],
        });

        self.composite_bg_a = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite_bg_a"),
            layout: &self.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.color_tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.godray_a.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.output.view),
                },
            ],
        });

        self.composite_bg_b = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite_bg_b"),
            layout: &self.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.color_tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.godray_b.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.output.view),
                },
            ],
        });

        self.blit_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit_bg"),
            layout: &self.blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.output.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.overlay_buf.as_entire_binding(),
                },
            ],
        });

        // reset ping-pong after resize (optional but avoids weird history mix)
        self.use_ab = true;
    }

    pub fn write_chunk_grid(&self, grid: &[u32]) {
        let n = grid.len().min(self.grid_capacity as usize);
        self.queue
            .write_buffer(&self.chunk_grid_buf, 0, bytemuck::cast_slice(&grid[..n]));
    }

    pub fn write_camera(&self, cam: &CameraGpu) {
        self.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(cam));
    }

    pub fn write_overlay(&self, ov: &OverlayGpu) {
        self.queue
            .write_buffer(&self.overlay_buf, 0, bytemuck::bytes_of(ov));
    }

    pub fn apply_chunk_uploads(&self, uploads: Vec<ChunkUpload>) {
        let node_stride = std::mem::size_of::<NodeGpu>() as u64;
        let meta_stride = std::mem::size_of::<ChunkMetaGpu>() as u64;

        for u in uploads {
            // meta write (always)
            if u.slot < self.chunk_capacity {
                let meta_off = (u.slot as u64) * meta_stride;
                self.queue
                    .write_buffer(&self.chunk_buf, meta_off, bytemuck::bytes_of(&u.meta));
            }

            // nodes write (only if provided)
            if !u.nodes.is_empty() {
                let needed = u.nodes.len() as u32;
                if u.node_base <= self.node_capacity && u.node_base + needed <= self.node_capacity {
                    let node_off = (u.node_base as u64) * node_stride;
                    self.queue
                        .write_buffer(&self.node_buf, node_off, bytemuck::cast_slice(&u.nodes));
                }
            }
        }
    }

    // 3-pass compute encode: primary -> godray -> composite
    pub fn encode_compute(&mut self, encoder: &mut wgpu::CommandEncoder, width: u32, height: u32) {
        // pass 1: primary (full-res)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("primary_pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.primary_pipeline);
            cpass.set_bind_group(0, &self.primary_bg, &[]);

            let gx = (width + 7) / 8;
            let gy = (height + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        // pass 2: godray (quarter-res) — use scene_bg (no storage outputs bound)
        {
            let qw = quarter_dim(width);
            let qh = quarter_dim(height);

            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("godray_pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.godray_pipeline);

            // group(0): camera + world buffers only
            cpass.set_bind_group(0, &self.scene_bg, &[]);

            // group(1): depth + history + out (ping-pong)
            if self.use_ab {
                cpass.set_bind_group(1, &self.godray_bg_ab, &[]);
            } else {
                cpass.set_bind_group(1, &self.godray_bg_ba, &[]);
            }

            let gx = (qw + 7) / 8;
            let gy = (qh + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        // pass 3: composite (full-res) — bind empty for group(0/1), real for group(2)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("composite_pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.composite_pipeline);

            // satisfy group(0/1) in pipeline layout
            cpass.set_bind_group(0, &self.empty_bg, &[]);
            cpass.set_bind_group(1, &self.empty_bg, &[]);

            // group(2): read current godray buffer
            if self.use_ab {
                // AB writes B
                cpass.set_bind_group(2, &self.composite_bg_b, &[]);
            } else {
                // BA writes A
                cpass.set_bind_group(2, &self.composite_bg_a, &[]);
            }

            let gx = (width + 7) / 8;
            let gy = (height + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        // flip ping-pong for next frame
        self.use_ab = !self.use_ab;
    }

    pub fn encode_blit(&self, encoder: &mut wgpu::CommandEncoder, frame_view: &wgpu::TextureView) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("blit_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: frame_view,
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

        rpass.set_pipeline(&self.blit_pipeline);
        rpass.set_bind_group(0, &self.blit_bg, &[]);
        rpass.draw(0..3, 0..1);
    }
}
