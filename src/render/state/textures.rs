// src/render/state/textures.rs
// ----------------------------

use crate::app::config;
use crate::{
    render::resources::{create_output_texture, OutputTex},
};

pub struct Tex2D {
    pub view: wgpu::TextureView,
}

pub struct Tex2DArray {
    pub tex: wgpu::Texture,
    pub view: wgpu::TextureView,
}

pub struct TextureSet {
    pub output: OutputTex,
    pub color: Tex2D,
    pub depth: Tex2D,

    // per-pixel local lighting term (unfogged), written by primary
    pub local: Tex2D,

    pub primary_hit_hist: [Tex2D; 2],

    pub shadow_hist: [Tex2D; 2],
    
    pub godray: [Tex2D; 2],
    pub clip_height: Tex2DArray,
}


pub fn quarter_dim(x: u32) -> u32 {
    (x + 3) / 4
}

pub fn half_dim(x: u32) -> u32 {
    (x + 1) / 2
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
    Tex2D { view }
}

fn make_tex2d_array(
    device: &wgpu::Device,
    label: &str,
    w: u32,
    h: u32,
    layers: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> Tex2DArray {
    let w = w.max(1);
    let h = h.max(1);
    let layers = layers.max(1);

    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    });

    let view = tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some(&format!("{label}_view")),
        format: Some(format),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(layers),
    });

    Tex2DArray { tex, view }
}

pub fn create_textures(
    device: &wgpu::Device,
    out_w: u32,
    out_h: u32,
    internal_w: u32,
    internal_h: u32,
) -> TextureSet {

    let rw_tex_usage =
        wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;

    let output = create_output_texture(device, out_w, out_h);

    let color = make_tex2d(
        device,
        "color_tex",
        internal_w,
        internal_h,
        wgpu::TextureFormat::Rgba32Float,
        rw_tex_usage,
    );

    let depth = make_tex2d(
        device,
        "depth_tex",
        internal_w,
        internal_h,
        wgpu::TextureFormat::R32Float,
        rw_tex_usage,
    );

    let primary_hit_hist = [
        make_tex2d(
            device,
            "primary_hit_hist_a",
            internal_w,
            internal_h,
            wgpu::TextureFormat::Rgba32Float,
            rw_tex_usage,
        ),
        make_tex2d(
            device,
            "primary_hit_hist_b",
            internal_w,
            internal_h,
            wgpu::TextureFormat::Rgba32Float,
            rw_tex_usage,
        ),
    ];

    let shadow_hist = [
        make_tex2d(
            device,
            "shadow_hist_a",
            internal_w,
            internal_h,
            wgpu::TextureFormat::Rgba32Float,
            rw_tex_usage,
        ),
        make_tex2d(
            device,
            "shadow_hist_b",
            internal_w,
            internal_h,
            wgpu::TextureFormat::Rgba32Float,
            rw_tex_usage,
        ),
    ];

    let godray = [
        make_tex2d(
            device,
            "godray_a",
            internal_w,
            internal_h,
            wgpu::TextureFormat::Rgba32Float,
            rw_tex_usage,
        ),
        make_tex2d(
            device,
            "godray_b",
            internal_w,
            internal_h,
            wgpu::TextureFormat::Rgba32Float,
            rw_tex_usage,
        ),
    ];

    // Clipmap height: R32Float
    let clip_height = make_tex2d_array(
        device,
        "clip_height",
        config::CLIPMAP_RES,
        config::CLIPMAP_RES,
        config::CLIPMAP_LEVELS,
        wgpu::TextureFormat::R32Float,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );

    let local = make_tex2d(
        device,
        "local_tex",
        internal_w,
        internal_h,
        wgpu::TextureFormat::Rgba32Float,
        rw_tex_usage,
    );


    TextureSet {
        output,
        color,
        depth,
        local,
        primary_hit_hist,
        shadow_hist,
        godray,
        clip_height,
    }
}
