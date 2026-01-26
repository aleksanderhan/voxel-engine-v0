use wgpu::util::DeviceExt;

use crate::{
    render::{
        gpu_types::{CameraGpu, ChunkMetaGpu, NodeGpu, OverlayGpu},
        resources::{create_output_texture, OutputTex},
        shaders,
    },
};

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,

    // output texture used by compute and sampled by blit
    output: OutputTex,

    // shaders/pipelines
    compute_pipeline: wgpu::ComputePipeline,
    blit_pipeline: wgpu::RenderPipeline,

    // layouts
    compute_bgl: wgpu::BindGroupLayout,
    blit_bgl: wgpu::BindGroupLayout,

    // resources
    sampler: wgpu::Sampler,

    camera_buf: wgpu::Buffer,
    overlay_buf: wgpu::Buffer,

    node_buf: wgpu::Buffer,
    chunk_buf: wgpu::Buffer,

    compute_bg: wgpu::BindGroup,
    blit_bg: wgpu::BindGroup,

    // dummies (kept so we can rebuild buffers with non-zero size)
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

        let output = create_output_texture(&device, width.max(1), height.max(1));

        // shaders
        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ray_cs"),
            source: wgpu::ShaderSource::Wgsl(shaders::ray_cs_wgsl().into()),
        });
        let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(shaders::blit_wgsl().into()),
        });

        // bind group layouts
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

        // pipelines
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

        // sampler
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

        // dummies + storage buffers
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

        let node_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("svo_nodes_dummy"),
            contents: bytemuck::bytes_of(&dummy_node),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let chunk_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk_meta_dummy"),
            contents: bytemuck::bytes_of(&dummy_chunk),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // bind groups
        let compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            compute_pipeline,
            blit_pipeline,
            compute_bgl,
            blit_bgl,
            sampler,
            camera_buf,
            overlay_buf,
            node_buf,
            chunk_buf,
            compute_bg,
            blit_bg,
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
        self.output = create_output_texture(&self.device, width.max(1), height.max(1));

        // rebuild bind groups that reference output texture
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

        self.compute_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute_bg"),
            layout: &self.compute_bgl,
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
                    resource: wgpu::BindingResource::TextureView(&self.output.view),
                },
            ],
        });
    }

    pub fn write_camera(&self, cam: &CameraGpu) {
        self.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(cam));
    }

    pub fn write_overlay(&self, ov: &OverlayGpu) {
        self.queue
            .write_buffer(&self.overlay_buf, 0, bytemuck::bytes_of(ov));
    }

    /// Rebuild storage buffers only when contents changed.
    pub fn update_scene_storage(&mut self, changed: bool, chunks_meta: &[ChunkMetaGpu], nodes: &[NodeGpu]) {
        if !changed {
            return;
        }

        // nodes (non-zero)
        let node_bytes: &[u8];
        let tmp_node: [NodeGpu; 1];
        if nodes.is_empty() {
            tmp_node = [self.dummy_node];
            node_bytes = bytemuck::cast_slice(&tmp_node);
        } else {
            node_bytes = bytemuck::cast_slice(nodes);
        }

        self.node_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("svo_nodes_packed"),
            contents: node_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

        // chunk meta (non-zero)
        let meta_bytes: &[u8];
        let tmp_meta: [ChunkMetaGpu; 1];
        if chunks_meta.is_empty() {
            tmp_meta = [self.dummy_chunk];
            meta_bytes = bytemuck::cast_slice(&tmp_meta);
        } else {
            meta_bytes = bytemuck::cast_slice(chunks_meta);
        }

        self.chunk_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk_meta"),
            contents: meta_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

        // compute bind group references the storage buffers, so rebuild it
        self.compute_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute_bg"),
            layout: &self.compute_bgl,
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
                    resource: wgpu::BindingResource::TextureView(&self.output.view),
                },
            ],
        });
    }

    pub fn encode_compute(&self, encoder: &mut wgpu::CommandEncoder, width: u32, height: u32) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ray_pass"),
            timestamp_writes: None,
        });

        cpass.set_pipeline(&self.compute_pipeline);
        cpass.set_bind_group(0, &self.compute_bg, &[]);
        let gx = (width + 7) / 8;
        let gy = (height + 7) / 8;
        cpass.dispatch_workgroups(gx, gy, 1);
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
