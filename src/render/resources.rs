// src/render/resources.rs

pub struct OutputTex {
    pub tex: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub w: u32,
    pub h: u32,
}

pub fn create_output_texture(device: &wgpu::Device, w: u32, h: u32) -> OutputTex {
    let w = w.max(1);
    let h = h.max(1);

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
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let view = tex.create_view(&Default::default());

    OutputTex { tex, view, w, h }
}
