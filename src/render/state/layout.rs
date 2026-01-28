// src/render/state/layout.rs
// --------------------------

pub struct Layouts {
    pub primary: wgpu::BindGroupLayout,
    pub scene: wgpu::BindGroupLayout,
    pub godray: wgpu::BindGroupLayout,
    pub composite: wgpu::BindGroupLayout,
    pub empty: wgpu::BindGroupLayout,
    pub blit: wgpu::BindGroupLayout,
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

fn bgl_storage_buf(
    binding: u32,
    visibility: wgpu::ShaderStages,
    read_only: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_tex_sample(
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
    let cs = wgpu::ShaderStages::COMPUTE;

    let scene = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene_bgl"),
        entries: &[
            bgl_uniform(0, cs),
            bgl_storage_buf(1, cs, false),
            bgl_storage_buf(2, cs, false),
            bgl_uniform(3, cs),
            bgl_storage_buf(6, cs, true),
        ],
    });

    let primary = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("primary_bgl"),
        entries: &[
            bgl_uniform(0, cs),
            bgl_storage_buf(1, cs, false),
            bgl_storage_buf(2, cs, false),
            bgl_uniform(3, cs),
            bgl_storage_tex_wo(4, cs, wgpu::TextureFormat::Rgba16Float),
            bgl_storage_tex_wo(5, cs, wgpu::TextureFormat::R32Float),
            bgl_storage_buf(6, cs, true),
        ],
    });

    let godray = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("godray_bgl"),
        entries: &[
            bgl_tex_sample(
                0,
                cs,
                wgpu::TextureSampleType::Float { filterable: false },
            ),
            bgl_tex_sample(
                1,
                cs,
                wgpu::TextureSampleType::Float { filterable: false },
            ),
            bgl_storage_tex_wo(2, cs, wgpu::TextureFormat::Rgba16Float),
        ],
    });

    let composite = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("composite_bgl"),
        entries: &[
            bgl_tex_sample(
                0,
                cs,
                wgpu::TextureSampleType::Float { filterable: false },
            ),
            bgl_tex_sample(
                1,
                cs,
                wgpu::TextureSampleType::Float { filterable: false },
            ),
            bgl_storage_tex_wo(2, cs, wgpu::TextureFormat::Rgba16Float),
        ],
    });

    let empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("empty_bgl"),
        entries: &[],
    });

    let blit = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("blit_bgl"),
        entries: &[
            bgl_tex_sample(
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
