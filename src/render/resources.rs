// src/render/resources.rs
//
// Small GPU resource helpers that don't fit cleanly into the renderer "state" modules.
//
// Right now this file provides the final full-resolution output texture:
// - written as a STORAGE texture by the composite compute pass
// - sampled as a regular texture by the final blit render pass
//
// Keeping this as a tiny helper makes the main texture set code a bit cleaner.

/// Wrapper for the renderer's final output texture view.
///
/// The renderer stores only the TextureView; the view keeps the underlying texture alive
/// for as long as it exists (wgpu uses ref-counted internal ownership).
pub struct OutputTex {
    /// Texture view bound in bind groups (storage write in compute, sampled in blit).
    pub view: wgpu::TextureView,
}

/// Create the final output texture (full resolution).
///
/// Properties:
/// - Format: RGBA32F (high dynamic range, good for post-processing)
/// - Usage:
///   - STORAGE_BINDING: composite pass writes into it as a storage texture
///   - TEXTURE_BINDING: blit pass samples it in the fragment shader
///
/// Notes:
/// - wgpu forbids zero-sized textures, so we clamp `w`/`h` to at least 1.
pub fn create_output_texture(device: &wgpu::Device, w: u32, h: u32) -> OutputTex {
    // Avoid creating zero-sized textures (can happen during minimize/resizes).
    let w = w.max(1);
    let h = h.max(1);

    // Allocate the GPU texture backing store.
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
        format: wgpu::TextureFormat::Rgba32Float,
        // Must support both compute writes and render sampling.
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    // Default view covers the whole texture.
    let view = tex.create_view(&Default::default());

    OutputTex { view }
}
