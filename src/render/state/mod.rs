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

        let textures = create_textures(&device, width, height);

        let pipelines = create_pipelines(&device, &layouts, &cs_module, &fs_module, surface_format);

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
        }
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn resize_output(&mut self, width: u32, height: u32) {
        self.textures = create_textures(&self.device, width, height);
        self.bind_groups = create_bind_groups(
            &self.device,
            &self.layouts,
            &self.buffers,
            &self.textures,
            &self.sampler,
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

            let gx = (width + 7) / 8;
            let gy = (height + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        let ping = self.ping;
        let pong = 1 - ping;

        {
            let qw = quarter_dim(width);
            let qh = quarter_dim(height);

            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("godray_pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.pipelines.godray);
            cpass.set_bind_group(0, &self.bind_groups.scene, &[]);
            cpass.set_bind_group(1, &self.bind_groups.godray[ping], &[]);

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

    pub fn encode_clipmap_patch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        level: u32,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        data_f16: &[u16],
    ) {
        let res = config::CLIPMAP_RES;
        if level >= config::CLIPMAP_LEVELS { return; }
        if w == 0 || h == 0 { return; }
        if x + w > res || y + h > res { return; }

        let expected = (w as usize) * (h as usize);
        if data_f16.len() != expected { return; }

        // Tight row pitch in bytes (R16Float = 2 bytes/texel)
        let row_bytes = (w as usize) * 2;

        // WebGPU: bytes_per_row must be multiple of 256
        let padded_row_bytes = align_up(row_bytes, 256);

        // Prepare padded bytes (only when needed)
        let bytes: Vec<u8>;
        let bytes_ref: &[u8];

        if padded_row_bytes == row_bytes {
            bytes_ref = bytemuck::cast_slice(data_f16);
        } else {
            let src: &[u8] = bytemuck::cast_slice(data_f16);
            let mut out = vec![0u8; padded_row_bytes * (h as usize)];

            for row in 0..(h as usize) {
                let src_off = row * row_bytes;
                let dst_off = row * padded_row_bytes;
                out[dst_off..dst_off + row_bytes].copy_from_slice(&src[src_off..src_off + row_bytes]);
            }

            bytes = out;
            bytes_ref = &bytes;
        }

        // Staging buffer for this patch
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clipmap_patch_staging"),
            size: bytes_ref.len() as u64,
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });

        {
            let mut view = staging.slice(..).get_mapped_range_mut();
            view.copy_from_slice(bytes_ref);
        }
        staging.unmap();

        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: &staging,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes as u32),
                    rows_per_image: Some(h),
                },
            },
            wgpu::ImageCopyTexture {
                texture: &self.textures.clip_height.tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: level },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Encode clipmap uniform upload into the *current encoder* (no queue.write_buffer).
    pub fn encode_clipmap_uniform(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        clip: &crate::render::gpu_types::ClipmapGpu,
    ) {
        let bytes = bytemuck::bytes_of(clip);

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clipmap_uniform_staging"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });

        {
            let mut view = staging.slice(..).get_mapped_range_mut();
            view.copy_from_slice(bytes);
        }
        staging.unmap();

        encoder.copy_buffer_to_buffer(&staging, 0, &self.buffers.clipmap, 0, bytes.len() as u64);
    }

    /// Encode: (1) all patch uploads, then (2) uniform update — in the same encoder.
    pub fn encode_clipmap_updates(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        clip: &crate::render::gpu_types::ClipmapGpu,
        uploads: &[crate::clipmap::ClipmapUpload],
    ) {
        // 1) texture first
        for u in uploads {
            self.encode_clipmap_patch(encoder, u.level, u.x, u.y, u.w, u.h, &u.data_f16);
        }

        // 2) uniform second
        self.encode_clipmap_uniform(encoder, clip);
    }

}
