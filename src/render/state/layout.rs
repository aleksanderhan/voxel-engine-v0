// src/render/state/layout.rs
//
// Bind group layouts and small "entry constructors" to reduce repetition.
//
// This module encodes the contract between WGSL @group/@binding and Rust-side wgpu setup.

pub struct Layouts {
    pub primary: wgpu::BindGroupLayout,   // group(0) with storage outputs (primary pass only)
    pub scene: wgpu::BindGroupLayout,     // group(0) camera + buffers (godray pass)
    pub godray: wgpu::BindGroupLayout,    // group(1) depth + history + out
    pub composite: wgpu::BindGroupLayout, // group(2) color + godray + output
    pub empty: wgpu::BindGroupLayout,     // empty for group(0/1) placeholder in composite PL
    pub blit: wgpu::BindGroupLayout,      // render pass: output + sampler + overlay
}

// -----------------------------------------------------------------------------
// BindGroupLayoutEntry helpers
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Layout creation
// -----------------------------------------------------------------------------

pub fn create_layouts(device: &wgpu::Device) -> Layouts {
    let cs_vis = wgpu::ShaderStages::COMPUTE;

    // Shared scene entries (group(0)) used by both scene_bgl and primary_bgl.
    // These bindings must match your WGSL declarations.
    let scene_entries: [wgpu::BindGroupLayoutEntry; 4] = [
        bgl_uniform(0, cs_vis),    // camera uniform
        bgl_storage_ro(1, cs_vis), // chunks meta (ro)
        bgl_storage_ro(2, cs_vis), // nodes arena (ro)
        bgl_storage_ro(3, cs_vis), // chunk grid (ro)
    ];

    // group(0) SCENE: camera + buffers only (godray pass uses this to avoid storage conflicts)
    let scene = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene_bgl"),
        entries: &scene_entries,
    });

    // group(0) PRIMARY: scene + storage outputs for color/depth
    let mut primary_entries = Vec::with_capacity(6);
    primary_entries.extend_from_slice(&scene_entries);
    primary_entries.push(bgl_storage_tex_wo(
        4,
        cs_vis,
        wgpu::TextureFormat::Rgba16Float,
    )); // color write
    primary_entries.push(bgl_storage_tex_wo(5, cs_vis, wgpu::TextureFormat::R32Float)); // depth write

    let primary = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("primary_bgl"),
        entries: &primary_entries,
    });

    // group(1) GODRAY: depth sampled + history sampled + out storage
    let godray = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("godray_bgl"),
        entries: &[
            bgl_tex_sample(
                0,
                cs_vis,
                wgpu::TextureSampleType::Float { filterable: false },
            ), // depth_tex (r32f)
            bgl_tex_sample(
                1,
                cs_vis,
                wgpu::TextureSampleType::Float { filterable: false },
            ), // history (rgba16f)
            bgl_storage_tex_wo(2, cs_vis, wgpu::TextureFormat::Rgba16Float), // out (rgba16f)
        ],
    });

    // group(2) COMPOSITE: color sampled + godray sampled + output storage
    let composite = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("composite_bgl"),
        entries: &[
            bgl_tex_sample(
                0,
                cs_vis,
                wgpu::TextureSampleType::Float { filterable: false },
            ), // color_tex
            bgl_tex_sample(
                1,
                cs_vis,
                wgpu::TextureSampleType::Float { filterable: false },
            ), // godray_tex
            bgl_storage_tex_wo(2, cs_vis, wgpu::TextureFormat::Rgba16Float), // output
        ],
    });

    // Empty layouts used as placeholders for composite pipeline layout group(0/1).
    let empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("empty_bgl"),
        entries: &[],
    });

    // Blit layout: output texture + sampler + overlay uniform.
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
