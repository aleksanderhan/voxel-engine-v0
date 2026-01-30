// src/render/state/layout.rs
//
// Bind group layouts and small helpers.

pub struct Layouts {
    pub primary: wgpu::BindGroupLayout,
    pub scene: wgpu::BindGroupLayout,
    pub godray: wgpu::BindGroupLayout,
    pub composite: wgpu::BindGroupLayout,
    pub empty: wgpu::BindGroupLayout,
    pub blit: wgpu::BindGroupLayout,
}

fn bgl_sampler(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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

pub fn create_layouts(device: &wgpu::Device) -> Layouts {
    let cs_vis = wgpu::ShaderStages::COMPUTE;

    let scene_entries: [wgpu::BindGroupLayoutEntry; 7] = [
        bgl_uniform(0, cs_vis),
        bgl_storage_ro(1, cs_vis),
        bgl_storage_ro(2, cs_vis),
        bgl_storage_ro(3, cs_vis),
        bgl_storage_ro(8, cs_vis),
        bgl_storage_ro(9, cs_vis),
        bgl_storage_ro(10, cs_vis),
    ];

    let scene = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene_bgl"),
        entries: &scene_entries,
    });

    // PRIMARY: add clipmap uniform + clipmap height texture array
    // bindings:
    // 0 camera
    // 1 chunks
    // 2 nodes
    // 3 chunk_grid
    // 4 color storage
    // 5 depth storage
    // 6 clipmap uniform
    // 7 clipmap height texture array (R32Float)
    // 8 macro_occ
    let mut primary_entries = Vec::with_capacity(8);
    primary_entries.extend_from_slice(&scene_entries);

    primary_entries.push(bgl_storage_tex_wo(4, cs_vis, wgpu::TextureFormat::Rgba16Float));
    primary_entries.push(bgl_storage_tex_wo(5, cs_vis, wgpu::TextureFormat::R32Float));

    primary_entries.push(bgl_uniform(6, cs_vis));
    primary_entries.push(bgl_tex_sample_2d_array(
        7,
        cs_vis,
        wgpu::TextureSampleType::Float { filterable: false },
    ));

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
            bgl_storage_tex_wo(2, cs_vis, wgpu::TextureFormat::Rgba16Float),
        ],
    });

    let composite = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("composite_bgl"),
        entries: &[
            bgl_tex_sample_2d(
                0,
                cs_vis,
                wgpu::TextureSampleType::Float { filterable: true },
            ),
            bgl_tex_sample_2d(
                1,
                cs_vis,
                wgpu::TextureSampleType::Float { filterable: true },
            ),
            bgl_storage_tex_wo(2, cs_vis, wgpu::TextureFormat::Rgba16Float),

            // full-res depth for depth-aware upsample
            bgl_tex_sample_2d(
                3,
                cs_vis,
                wgpu::TextureSampleType::Float { filterable: false },
            ),

            // NEW: sampler for godray_tex (used by textureSampleLevel)
            bgl_sampler(4, cs_vis),
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
                wgpu::TextureSampleType::Float { filterable: true },
            ),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            bgl_uniform(2, wgpu::ShaderStages::FRAGMENT),
        ],
    });

    Layouts {
        primary,
        scene,
        godray,
        composite,
        empty,
        blit,
    }
}
