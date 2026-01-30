// src/render/state/mod.rs
// -----------------------
mod bindgroups;
mod buffers;
mod layout;
mod pipelines;
pub mod textures;

use crate::{
    config,
    render::gpu_types::{CameraGpu, OverlayGpu},
    streaming::ChunkUpload,
};

use bindgroups::{create_bind_groups, BindGroups};
use buffers::{create_persistent_buffers, Buffers};
use layout::{create_layouts, Layouts};
use pipelines::{create_pipelines, Pipelines};
use textures::{create_textures, quarter_dim, TextureSet};

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,

    sampler: wgpu::Sampler,

    layouts: Layouts,
    pipelines: Pipelines,
    buffers: Buffers,
    textures: TextureSet,
    bind_groups: BindGroups,

    ping: usize,

    render_scale: f32,
    internal_w: u32,
    internal_h: u32,

    clip_scratch: Vec<u8>,
}

fn align_up(v: usize, a: usize) -> usize {
    (v + (a - 1)) & !(a - 1)
}

impl Renderer {
    pub async fn new(
        adapter: &wgpu::Adapter,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let adapter_limits = adapter.limits();
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_buffer_size: adapter_limits.max_buffer_size,
            ..wgpu::Limits::default()
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("device"),
                    required_features: wgpu::Features::empty(),
                    required_limits,
                },
                None,
            )
            .await
            .unwrap();

        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ray_cs"),
            source: wgpu::ShaderSource::Wgsl(crate::render::shaders::ray_cs_wgsl().into()),
        });

        let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(crate::render::shaders::blit_wgsl().into()),
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear_clamp_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest, // level 0 anyway
            ..Default::default()
        });

        let layouts = create_layouts(&device);
        let buffers = create_persistent_buffers(&device);

        let pipelines = create_pipelines(&device, &layouts, &cs_module, &fs_module, surface_format);

        let render_scale = config::RENDER_SCALE;
        let internal_w = ((width as f32) * render_scale).round() as u32;
        let internal_h = ((height as f32) * render_scale).round() as u32;
        
        let textures = create_textures(&device, width, height, internal_w, internal_h);
        let bind_groups = create_bind_groups(&device, &layouts, &buffers, &textures, &sampler);

        Self {
            device,
            queue,
            sampler,
            layouts,
            pipelines,
            buffers,
            textures,
            bind_groups,
            ping: 0,
            render_scale,
            internal_w,
            internal_h,
            clip_scratch: Vec::new(),
        }
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn resize_output(&mut self, width: u32, height: u32) {
        self.internal_w = ((width as f32) * self.render_scale).round() as u32;
        self.internal_h = ((height as f32) * self.render_scale).round() as u32;

        self.textures = create_textures(&self.device, width, height, self.internal_w, self.internal_h);

        self.bind_groups = create_bind_groups(
            &self.device, &self.layouts, &self.buffers, &self.textures, &self.sampler,
        );

        self.ping = 0;
    }

    pub fn write_chunk_grid(&self, grid: &[u32]) {
        let n = grid.len().min(self.buffers.grid_capacity as usize);
        self.queue.write_buffer(
            &self.buffers.chunk_grid,
            0,
            bytemuck::cast_slice(&grid[..n]),
        );
    }

    pub fn write_camera(&self, cam: &CameraGpu) {
        self.queue
            .write_buffer(&self.buffers.camera, 0, bytemuck::bytes_of(cam));
    }

    pub fn write_overlay(&self, ov: &OverlayGpu) {
        self.queue
            .write_buffer(&self.buffers.overlay, 0, bytemuck::bytes_of(ov));
    }

    pub fn encode_compute(&mut self, encoder: &mut wgpu::CommandEncoder, width: u32, height: u32) {
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("primary_pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.pipelines.primary);
            cpass.set_bind_group(0, &self.bind_groups.primary, &[]);

            let gx = (self.internal_w + 7) / 8;
            let gy = (self.internal_h + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);

        }

        let ping = self.ping;
        let pong = 1 - ping;

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("godray_pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.pipelines.godray);
            cpass.set_bind_group(0, &self.bind_groups.scene, &[]);
            cpass.set_bind_group(1, &self.bind_groups.godray[ping], &[]);

            let qw = quarter_dim(self.internal_w);
            let qh = quarter_dim(self.internal_h);

            let gx = (qw + 7) / 8;
            let gy = (qh + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);

        }

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("composite_pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.pipelines.composite);

            // group(0) must match layouts.scene now
            cpass.set_bind_group(0, &self.bind_groups.scene, &[]);

            // group(1) is still empty (or you can omit setting it)
            cpass.set_bind_group(1, &self.bind_groups.empty, &[]);

            // group(2) is your composite textures
            cpass.set_bind_group(2, &self.bind_groups.composite[pong], &[]);

            let gx = (width + 7) / 8;
            let gy = (height + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);

        }

        self.ping = pong;
    }

    pub fn encode_blit(&self, encoder: &mut wgpu::CommandEncoder, frame_view: &wgpu::TextureView) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("blit_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: frame_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        rpass.set_pipeline(&self.pipelines.blit);
        rpass.set_bind_group(0, &self.bind_groups.blit, &[]);
        rpass.draw(0..3, 0..1);
    }

    pub fn apply_chunk_uploads(&self, uploads: Vec<ChunkUpload>) {
        let node_stride = std::mem::size_of::<crate::render::gpu_types::NodeGpu>() as u64;
        let meta_stride = std::mem::size_of::<crate::render::gpu_types::ChunkMetaGpu>() as u64;
        let u32_stride  = std::mem::size_of::<u32>() as u64;
        let rope_stride = std::mem::size_of::<crate::render::gpu_types::NodeRopesGpu>() as u64;

        for u in uploads {
            // meta
            if u.slot < self.buffers.chunk_capacity {
                let meta_off = (u.slot as u64) * meta_stride;
                self.queue.write_buffer(&self.buffers.chunk, meta_off, bytemuck::bytes_of(&u.meta));
            }

            // nodes
            if !u.nodes.is_empty() {
                let needed = u.nodes.len() as u32;
                if u.node_base + needed <= self.buffers.node_capacity {
                    let node_off = (u.node_base as u64) * node_stride;
                    self.queue.write_buffer(&self.buffers.node, node_off, bytemuck::cast_slice(u.nodes.as_ref()));
                }
            }

            // macro occupancy
            if !u.macro_words.is_empty() {
                let needed = u.macro_words.len() as u32;
                if u.meta.macro_base + needed <= self.buffers.macro_capacity_u32 {
                    let off = (u.meta.macro_base as u64) * u32_stride;
                    self.queue.write_buffer(&self.buffers.macro_occ, off, bytemuck::cast_slice(u.macro_words.as_ref()));
                }
            }

            if !u.ropes.is_empty() {
                let needed = u.ropes.len() as u32;
                if u.node_base + needed <= self.buffers.rope_capacity {
                    let rope_off = (u.node_base as u64) * rope_stride;
                    self.queue.write_buffer(&self.buffers.node_ropes, rope_off, bytemuck::cast_slice(u.ropes.as_ref()));
                }
            }

            // colinfo (64*64 columns packed => 2048 u32 per chunk)
            if !u.colinfo_words.is_empty() {
                let needed = u.colinfo_words.len() as u32;
                if u.meta.colinfo_base + needed <= self.buffers.colinfo_capacity_u32 {
                    let off = (u.meta.colinfo_base as u64) * u32_stride;
                    self.queue.write_buffer(
                        &self.buffers.colinfo,
                        off,
                        bytemuck::cast_slice(u.colinfo_words.as_ref()),
                    );
                }
            }

        }
    }

    pub fn write_clipmap_updates(
        &mut self,
        clip: &crate::render::gpu_types::ClipmapGpu,
        uploads: &[crate::clipmap::ClipmapUpload],
    ) {
        let scratch = &mut self.clip_scratch;

        // 1) texture patches
        for u in uploads {
            let w = u.w as usize;
            let h = u.h as usize;
            if w == 0 || h == 0 { continue; }

            let row_bytes = w * 2;                 // R16Float => 2 bytes/texel
            let padded = align_up(row_bytes, 256); // required
            let needed = padded * h;

            scratch.clear();
            scratch.resize(needed, 0);

            // copy row-by-row into padded scratch
            let src: &[u8] = bytemuck::cast_slice(&u.data_f16);
            for row in 0..h {
                let s0 = row * row_bytes;
                let d0 = row * padded;
                scratch[d0..d0 + row_bytes].copy_from_slice(&src[s0..s0 + row_bytes]);
            }

            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.textures.clip_height.tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: u.x, y: u.y, z: u.level },
                    aspect: wgpu::TextureAspect::All,
                },
                &scratch,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded as u32),
                    rows_per_image: Some(u.h),
                },
                wgpu::Extent3d { width: u.w, height: u.h, depth_or_array_layers: 1 },
            );
        }

        // 2) uniform
        self.queue.write_buffer(&self.buffers.clipmap, 0, bytemuck::bytes_of(clip));
    }

    pub fn internal_dims(&self) -> (u32, u32) {
        (self.internal_w.max(1), self.internal_h.max(1))
    }

}
