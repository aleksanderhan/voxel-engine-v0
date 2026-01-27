// src/render/state/textures.rs
//
// Texture creation and sizing policy for the renderer.
// Kept separate so resize() is a simple "recreate textures + rebuild bind groups".

use crate::render::resources::{create_output_texture, OutputTex};

pub struct Tex2D {
    pub view: wgpu::TextureView,
}

/// Bundle of all size-dependent textures used by the renderer.
pub struct TextureSet {
    // Final output texture: composite writes (storage) here; blit samples it.
    pub output: OutputTex,

    // Full-res intermediates:
    pub color: Tex2D, // rgba16f
    pub depth: Tex2D, // r32f

    // Quarter-res intermediates (ping-pong for temporal accumulation):
    // 0 = A, 1 = B
    pub godray: [Tex2D; 2], // rgba16f
}

pub fn quarter_dim(x: u32) -> u32 {
    // ceil(x / 4)
    (x + 3) / 4
}

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
        view,
    }
}

/// Create the full size-dependent texture set for the renderer.
///
/// Policy:
/// - color/depth are full-res
/// - godray textures are quarter-res and ping-ponged each frame
/// - all intermediates are STORAGE_BINDING|TEXTURE_BINDING (write in compute, sample elsewhere)
pub fn create_textures(device: &wgpu::Device, width: u32, height: u32) -> TextureSet {
    let width = width.max(1);
    let height = height.max(1);

    let rw_tex_usage =
        wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;

    let output = create_output_texture(device, width, height);

    let color = make_tex2d(
        device,
        "color_tex",
        width,
        height,
        wgpu::TextureFormat::Rgba16Float,
        rw_tex_usage,
    );

    let depth = make_tex2d(
        device,
        "depth_tex",
        width,
        height,
        wgpu::TextureFormat::R32Float,
        rw_tex_usage,
    );

    let qw = quarter_dim(width);
    let qh = quarter_dim(height);

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
