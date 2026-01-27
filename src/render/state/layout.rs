// src/render/state/layout.rs
//
// Bind group layouts and small "entry constructors" to reduce repetition.
//
// A BindGroupLayout is the *schema* for a bind group: it defines which resources
// exist at which @group/@binding, and how shaders are allowed to access them.
//
// This module encodes the contract between WGSL `@group(n) @binding(m)`
// declarations and the Rust-side wgpu setup. If bindings or types mismatch,
// pipeline creation or bind group creation will fail (or validation will trip).

pub struct Layouts {
    /// group(0): scene inputs + storage outputs (color/depth). Used only by the primary pass.
    pub primary: wgpu::BindGroupLayout,

    /// group(0): scene inputs only (camera + buffers). Used by godray to avoid storage conflicts.
    pub scene: wgpu::BindGroupLayout,

    /// group(1): godray pass resources (depth sample + history sample + out storage).
    pub godray: wgpu::BindGroupLayout,

    /// group(2): composite pass resources (color sample + godray sample + output storage).
    pub composite: wgpu::BindGroupLayout,

    /// Empty layout: used as a placeholder when a pipeline layout expects group(0/1)
    /// but the composite pass only meaningfully uses group(2).
    pub empty: wgpu::BindGroupLayout,

    /// Blit render pass layout: sample final output texture + sampler + overlay uniform.
    pub blit: wgpu::BindGroupLayout,
}

// -----------------------------------------------------------------------------
// BindGroupLayoutEntry helpers
// -----------------------------------------------------------------------------
//
// These tiny constructors keep the layout definitions readable and ensure
// consistent settings (no dynamic offsets, no min_binding_size, etc.).

/// Convenience for a uniform-buffer entry.
///
/// - `binding`: WGSL @binding index
/// - `visibility`: which shader stages can read this binding
fn bgl_uniform(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            // No dynamic offsets; the entire buffer is bound as-is.
            has_dynamic_offset: false,
            // None => let wgpu infer/validate size at bind time.
            min_binding_size: None,
        },
        // Not an array binding.
        count: None,
    }
}

/// Convenience for a read-only storage-buffer entry.
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

/// Convenience for a sampled 2D texture entry.
///
/// `sample_type` must match the WGSL texture type (float/int/uint + filterable or not).
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

/// Convenience for a write-only storage texture entry.
///
/// Storage textures are typically used as compute outputs.
/// `format` must match the actual texture format and the WGSL storage texture declaration.
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

/// Create all bind group layouts used by the renderer.
///
/// The specific formats and binding indices here must match the WGSL shader code.
/// Notes:
/// - Most compute textures are declared non-filterable, since they are likely written
///   as storage textures and/or sampled as exact texel values.
/// - The final blit pass uses a filterable sampler + filterable texture sampling.
pub fn create_layouts(device: &wgpu::Device) -> Layouts {
    // Most passes in this file are compute passes; share the stage visibility constant.
    let cs_vis = wgpu::ShaderStages::COMPUTE;

    // Shared scene entries (group(0)) used by both `scene` and `primary`.
    //
    // These bindings must match your WGSL declarations, e.g.:
    //   @group(0) @binding(0) var<uniform> camera : ...
    //   @group(0) @binding(1) var<storage, read> chunks : ...
    // etc.
    let scene_entries: [wgpu::BindGroupLayoutEntry; 4] = [
        bgl_uniform(0, cs_vis),    // camera uniform
        bgl_storage_ro(1, cs_vis), // chunks meta (read-only)
        bgl_storage_ro(2, cs_vis), // nodes arena (read-only)
        bgl_storage_ro(3, cs_vis), // chunk grid (read-only)
    ];

    // group(0) SCENE:
    // Camera + buffers only.
    //
    // Godray pass uses this so it can bind depth/history textures in other groups
    // without also binding storage outputs that could cause read/write conflicts.
    let scene = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene_bgl"),
        entries: &scene_entries,
    });

    // group(0) PRIMARY:
    // Scene inputs + storage outputs for color/depth.
    //
    // This is the "main" compute pass that writes color and depth textures.
    let mut primary_entries = Vec::with_capacity(6);
    primary_entries.extend_from_slice(&scene_entries);

    // binding(4): color storage output (rgba16f)
    primary_entries.push(bgl_storage_tex_wo(
        4,
        cs_vis,
        wgpu::TextureFormat::Rgba16Float,
    ));

    // binding(5): depth storage output (r32f)
    primary_entries.push(bgl_storage_tex_wo(5, cs_vis, wgpu::TextureFormat::R32Float));

    let primary = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("primary_bgl"),
        entries: &primary_entries,
    });

    // group(1) GODRAY:
    // - depth sampled (r32f)
    // - history sampled (rgba16f)
    // - out storage (rgba16f)
    //
    // Sampling is marked filterable:false to match typical storage-written float textures.
    let godray = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("godray_bgl"),
        entries: &[
            // binding(0): depth texture (sampled)
            bgl_tex_sample(
                0,
                cs_vis,
                wgpu::TextureSampleType::Float { filterable: false },
            ),
            // binding(1): history texture (sampled)
            bgl_tex_sample(
                1,
                cs_vis,
                wgpu::TextureSampleType::Float { filterable: false },
            ),
            // binding(2): output storage texture (write-only)
            bgl_storage_tex_wo(2, cs_vis, wgpu::TextureFormat::Rgba16Float),
        ],
    });

    // group(2) COMPOSITE:
    // - color sampled
    // - godray sampled
    // - output storage (final offscreen output)
    let composite = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("composite_bgl"),
        entries: &[
            // binding(0): base color texture
            bgl_tex_sample(
                0,
                cs_vis,
                wgpu::TextureSampleType::Float { filterable: false },
            ),
            // binding(1): godray texture
            bgl_tex_sample(
                1,
                cs_vis,
                wgpu::TextureSampleType::Float { filterable: false },
            ),
            // binding(2): final output storage texture
            bgl_storage_tex_wo(2, cs_vis, wgpu::TextureFormat::Rgba16Float),
        ],
    });

    // Empty layout used as a placeholder bind group layout for group(0/1)
    // in the composite pipeline layout.
    let empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("empty_bgl"),
        entries: &[],
    });

    // Blit layout (render pass):
    // - sampled output texture (filterable, because we likely scale to the swapchain)
    // - filtering sampler
    // - overlay uniform (fragment stage)
    let blit = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("blit_bgl"),
        entries: &[
            // binding(0): output texture sampled in fragment shader
            bgl_tex_sample(
                0,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::TextureSampleType::Float { filterable: true },
            ),
            // binding(1): sampler for output texture
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // binding(2): overlay uniform (fps, dimensions, etc.)
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
