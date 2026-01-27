// src/render/state/textures.rs
//
// Texture creation and sizing policy for the renderer.
//
// Everything in here is "size-dependent": if the window/output resolution changes,
// these textures must be recreated, and any bind groups that reference their views
// must be rebuilt.
//
// Texture roles in the frame graph:
// - color_tex (full-res, RGBA16F): written by primary pass, sampled by composite.
// - depth_tex (full-res, R32F)   : written by primary pass, sampled by godray.
// - godray[A/B] (quarter-res, RGBA16F): temporal accumulation ping-pong.
// - output (full-res)           : written by composite, sampled by blit.
//
// Usage policy:
// - Intermediates are both STORAGE_BINDING and TEXTURE_BINDING so they can be
//   written as storage textures in compute and later sampled as regular textures.

use crate::render::resources::{create_output_texture, OutputTex};

/// Minimal wrapper around a 2D texture view.
///
/// Note: we only store the TextureView, not the Texture handle itself.
/// In wgpu, the view keeps the underlying texture alive as long as the view exists.
pub struct Tex2D {
    /// View used for binding (storage or sampled).
    pub view: wgpu::TextureView,
}

/// Bundle of all size-dependent textures used by the renderer.
pub struct TextureSet {
    // -------------------------------------------------------------------------
    // Final output
    // -------------------------------------------------------------------------

    /// Final output texture:
    /// - composite pass writes here (storage)
    /// - blit pass samples from it (filterable sampling in fragment shader)
    pub output: OutputTex,

    // -------------------------------------------------------------------------
    // Full-resolution intermediates
    // -------------------------------------------------------------------------

    /// Primary color accumulation buffer (RGBA16F).
    /// Written by primary compute pass.
    pub color: Tex2D,

    /// Primary depth buffer (R32F).
    /// Written by primary compute pass and sampled by godray.
    pub depth: Tex2D,

    // -------------------------------------------------------------------------
    // Quarter-resolution intermediates (temporal ping-pong)
    // -------------------------------------------------------------------------

    /// Godray accumulation textures (RGBA16F), quarter resolution.
    ///
    /// Index meaning:
    /// - 0 = A
    /// - 1 = B
    ///
    /// Each frame:
    /// - one is sampled as history
    /// - the other is written as the new result
    pub godray: [Tex2D; 2],
}

/// Compute the quarter-resolution dimension for an axis.
///
/// This uses ceil(x/4) so small sizes don't collapse to zero and so coverage remains
/// conservative (quarter-res still covers the full-res domain when upsampled).
pub fn quarter_dim(x: u32) -> u32 {
    // ceil(x / 4)
    (x + 3) / 4
}

/// Create a 2D texture and return only its view wrapper.
///
/// `usage` is passed through so the same helper can build both sampled-only or
/// read/write compute textures.
///
/// Zero-sized textures are invalid in wgpu, so width/height are clamped to >= 1.
fn make_tex2d(
    device: &wgpu::Device,
    label: &str,
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> Tex2D {
    // Avoid creating zero-sized textures (wgpu disallows it).
    let w = w.max(1);
    let h = h.max(1);

    // Allocate the texture backing storage.
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

    // Default view (whole texture, base mip, 2D).
    let view = tex.create_view(&Default::default());

    Tex2D { view }
}

/// Create the full size-dependent texture set for the renderer.
///
/// Sizing policy:
/// - color/depth are full resolution (match output size)
/// - godray textures are quarter resolution (ceil(width/4), ceil(height/4))
/// - godray uses two textures for ping-pong temporal accumulation
///
/// Usage policy for intermediates:
/// - STORAGE_BINDING: compute passes write to them as storage textures
/// - TEXTURE_BINDING: later passes sample them as regular textures
pub fn create_textures(device: &wgpu::Device, width: u32, height: u32) -> TextureSet {
    // Clamp to avoid invalid zero-sized allocations during minimize/resizes.
    let width = width.max(1);
    let height = height.max(1);

    // Intermediates are written by compute (storage) and read by later passes (sampled).
    let rw_tex_usage =
        wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;

    // Final output texture used by composite -> blit.
    // (Implementation details live in render::resources.)
    let output = create_output_texture(device, width, height);

    // Full-res primary color buffer.
    let color = make_tex2d(
        device,
        "color_tex",
        width,
        height,
        wgpu::TextureFormat::Rgba16Float,
        rw_tex_usage,
    );

    // Full-res primary depth buffer.
    let depth = make_tex2d(
        device,
        "depth_tex",
        width,
        height,
        wgpu::TextureFormat::R32Float,
        rw_tex_usage,
    );

    // Quarter-res sizes for godrays.
    let qw = quarter_dim(width);
    let qh = quarter_dim(height);

    // Godray ping-pong textures (same format/usage; only roles swap per frame).
    let godray = [
        make_tex2d(
            device,
            "godray_a",
            qw,
            qh,
            wgpu::TextureFormat::Rgba16Float,
            rw_tex_usage,
        ),
        make_tex2d(
            device,
            "godray_b",
            qw,
            qh,
            wgpu::TextureFormat::Rgba16Float,
            rw_tex_usage,
        ),
    ];

    TextureSet {
        output,
        color,
        depth,
        godray,
    }
}
