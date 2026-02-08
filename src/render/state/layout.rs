// src/render/state/layout.rs
//
// Bind group layouts and small helpers.

pub struct Layouts {
    pub primary: wgpu::BindGroupLayout,
    pub scene: wgpu::BindGroupLayout,
    pub godray: wgpu::BindGroupLayout,
    pub local_taa: wgpu::BindGroupLayout,
    pub composite: wgpu::BindGroupLayout,
    pub composite_taa: wgpu::BindGroupLayout,
    pub empty: wgpu::BindGroupLayout,
    pub blit: wgpu::BindGroupLayout,
}

fn bgl_sampler_non_filtering(
    binding: u32,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
        count: None,
    }
}

fn bgl_uniform(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_ro(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_rw(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_tex_sample_2d(
    binding: u32,
    visibility: wgpu::ShaderStages,
    sample_type: wgpu::TextureSampleType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn bgl_tex_sample_2d_array(
    binding: u32,
    visibility: wgpu::ShaderStages,
    sample_type: wgpu::TextureSampleType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    }
}

fn bgl_storage_tex_wo(
    binding: u32,
    visibility: wgpu::ShaderStages,
    format: wgpu::TextureFormat,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn bgl_storage_tex_array_wo(
    binding: u32,
    visibility: wgpu::ShaderStages,
    format: wgpu::TextureFormat,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2Array,
        },
        count: None,
    }
}

pub fn create_layouts(device: &wgpu::Device) -> Layouts {
    let cs_vis = wgpu::ShaderStages::COMPUTE;

    let scene_entries: [wgpu::BindGroupLayoutEntry; 6] = [
        bgl_uniform(0, cs_vis),
        bgl_storage_ro(1, cs_vis),
        bgl_storage_ro(2, cs_vis),
        bgl_storage_ro(3, cs_vis),
        bgl_storage_ro(9, cs_vis),
        bgl_storage_ro(10, cs_vis),
    ];

    let scene = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene_bgl"),
        entries: &scene_entries,
    });

    // PRIMARY bindings:
    // 0 camera (uniform)
    // 1 chunks (storage_ro)
    // 2 nodes (storage_ro)
    // 3 chunk_grid (storage_ro)
    // 4 primary outputs (storage array, layer 0 = color, layer 1 = local)
    // 5 depth (storage write)
    // 7 clipmap params (uniform)
    // 8 clipmap height texture array (sampled)
    // 9 macro/colinfo combined (storage_ro)
    // 10 node_ropes (storage_ro)
    // 12 primary hit history (sampled)
    // 13 primary hit history output (storage write)
    // 14 primary hit history sampler
    // 15 sun-shadow history (sampled)
    // 16 sun-shadow history output (storage buffer write)
    let primary_entries: [wgpu::BindGroupLayoutEntry; 15] = [
        bgl_uniform(0, cs_vis),
        bgl_storage_ro(1, cs_vis),
        bgl_storage_ro(2, cs_vis),
        bgl_storage_ro(3, cs_vis),

        bgl_storage_tex_array_wo(4, cs_vis, wgpu::TextureFormat::Rgba32Float),
        bgl_storage_tex_wo(5, cs_vis, wgpu::TextureFormat::R32Float),

        bgl_uniform(7, cs_vis),
        bgl_tex_sample_2d_array(
            8,
            cs_vis,
            wgpu::TextureSampleType::Float { filterable: false },
        ),

        bgl_storage_ro(9, cs_vis),
        bgl_storage_ro(10, cs_vis),

        bgl_tex_sample_2d(
            12,
            cs_vis,
            wgpu::TextureSampleType::Float { filterable: false },
        ),
        bgl_storage_tex_wo(13, cs_vis, wgpu::TextureFormat::Rgba32Float),
        bgl_sampler_non_filtering(14, cs_vis),
        bgl_tex_sample_2d(
            15,
            cs_vis,
            wgpu::TextureSampleType::Float { filterable: false },
        ),
        bgl_storage_rw(16, cs_vis),
    ];

    let primary = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("primary_bgl"),
        entries: &primary_entries,
    });


    let godray = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("godray_bgl"),
        entries: &[
            bgl_tex_sample_2d(
                0,
                cs_vis,
                wgpu::TextureSampleType::Float { filterable: false },
            ),
            bgl_tex_sample_2d(
                1,
                cs_vis,
                wgpu::TextureSampleType::Float { filterable: false },
            ),
            bgl_storage_tex_wo(2, cs_vis, wgpu::TextureFormat::Rgba32Float),
            bgl_sampler_non_filtering(3, cs_vis), // sampler for history
        ],
    });

    let local_taa = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("local_taa_bgl"),
        entries: &[
            bgl_tex_sample_2d(
                0,
                cs_vis,
                wgpu::TextureSampleType::Float { filterable: false },
            ),
            bgl_tex_sample_2d(
                1,
                cs_vis,
                wgpu::TextureSampleType::Float { filterable: false },
            ),
            bgl_storage_tex_wo(2, cs_vis, wgpu::TextureFormat::Rgba32Float),
            bgl_sampler_non_filtering(3, cs_vis),
        ],
    });

    let composite = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("composite_bgl"),
        entries: &[
            bgl_tex_sample_2d(
                0,
                cs_vis,
                wgpu::TextureSampleType::Float { filterable: false },
            ),
            bgl_tex_sample_2d(
                1,
                cs_vis,
                wgpu::TextureSampleType::Float { filterable: false },
            ),
            bgl_storage_tex_wo(2, cs_vis, wgpu::TextureFormat::Rgba32Float),

            // full-res depth for depth-aware upsample
            bgl_tex_sample_2d(
                3,
                cs_vis,
                wgpu::TextureSampleType::Float { filterable: false },
            ),

            // NEW: sampler for godray_tex (used by textureSampleLevel)
            bgl_sampler_non_filtering(4, cs_vis),

            // binding 5: local_hist_tex (sampled)
            bgl_tex_sample_2d(
                5,
                cs_vis,
                wgpu::TextureSampleType::Float { filterable: false },
            ),
            // binding 6: sampler (can reuse same sampler type)
            bgl_sampler_non_filtering(6, cs_vis),

        ],
    });

    let composite_taa = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("composite_taa_bgl"),
        entries: &[
            bgl_tex_sample_2d(
                0,
                cs_vis,
                wgpu::TextureSampleType::Float { filterable: false },
            ),
            bgl_tex_sample_2d(
                1,
                cs_vis,
                wgpu::TextureSampleType::Float { filterable: false },
            ),
            bgl_tex_sample_2d(
                2,
                cs_vis,
                wgpu::TextureSampleType::Float { filterable: false },
            ),
            bgl_sampler_non_filtering(3, cs_vis),
            bgl_storage_tex_wo(4, cs_vis, wgpu::TextureFormat::Rgba32Float),
            bgl_storage_tex_wo(5, cs_vis, wgpu::TextureFormat::Rgba32Float),
        ],
    });

    let empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("empty_bgl"),
        entries: &[],
    });

    let blit = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("blit_bgl"),
        entries: &[
            bgl_tex_sample_2d(
                0,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::TextureSampleType::Float { filterable: false },
            ),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
            bgl_uniform(2, wgpu::ShaderStages::FRAGMENT),
        ],
    });

    Layouts {
        primary,
        scene,
        godray,
        local_taa,
        composite,
        composite_taa,
        empty,
        blit,
    }
}
