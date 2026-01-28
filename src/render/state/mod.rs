// src/render/state/mod.rs
// -----------------------
mod bindgroups;
mod buffers;
mod layout;
mod pipelines;
pub mod textures;

use crate::{
    config,
    render::gpu_types::{CameraGpu, ClipmapGpu, OverlayGpu},
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
            label: Some("nearest_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
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

    pub fn write_clipmap(&self, clip: &ClipmapGpu) {
        self.queue
            .write_buffer(&self.buffers.clipmap, 0, bytemuck::bytes_of(clip));
    }

    /// FP16 clipmap patch upload into a sub-rectangle of a level.
    ///
    /// `data_f16` is w*h u16 values (IEEE half bits), row-major for the patch.
    ///
    /// IMPORTANT: WebGPU requires `bytes_per_row` to be a multiple of 256 bytes.
    /// For narrow strips (especially columns), we pad each row on the CPU when needed.
    pub fn write_clipmap_patch(&self, level: u32, x: u32, y: u32, w: u32, h: u32, data_f16: &[u16]) {
        let res = config::CLIPMAP_RES;
        if level >= config::CLIPMAP_LEVELS {
            return;
        }
        if w == 0 || h == 0 {
            return;
        }
        if x + w > res || y + h > res {
            return;
        }

        let expected = (w as usize) * (h as usize);
        if data_f16.len() != expected {
            return;
        }

        // Source row pitch (tightly packed)
        let row_bytes = (w * 2) as usize; // R16Float => 2 bytes per texel

        // WebGPU alignment requirement
        let align = 256usize;
        let padded_row_bytes = ((row_bytes + align - 1) / align) * align;

        let bytes: Vec<u8>;
        let bytes_ref: &[u8];

        if padded_row_bytes == row_bytes {
            bytes_ref = bytemuck::cast_slice(data_f16);
        } else {
            // Pad each row up to padded_row_bytes
            bytes = {
                let mut out = vec![0u8; padded_row_bytes * (h as usize)];
                let src: &[u8] = bytemuck::cast_slice(data_f16);

                for row in 0..(h as usize) {
                    let src_off = row * row_bytes;
                    let dst_off = row * padded_row_bytes;
                    out[dst_off..dst_off + row_bytes].copy_from_slice(&src[src_off..src_off + row_bytes]);
                }
                out
            };
            bytes_ref = &bytes;
        }

        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.textures.clip_height.tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: level },
                aspect: wgpu::TextureAspect::All,
            },
            bytes_ref,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_row_bytes as u32),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Convenience full-level upload (kept for compatibility / debugging).
    pub fn write_clipmap_level(&self, level: u32, data_f16: &[u16]) {
        let res = config::CLIPMAP_RES as usize;
        let expected = res * res;
        if data_f16.len() != expected {
            return;
        }
        self.write_clipmap_patch(
            level,
            0,
            0,
            config::CLIPMAP_RES,
            config::CLIPMAP_RES,
            data_f16,
        );
    }

    pub fn apply_chunk_uploads(&self, uploads: Vec<ChunkUpload>) {
        let node_stride = std::mem::size_of::<crate::render::gpu_types::NodeGpu>() as u64;
        let meta_stride = std::mem::size_of::<crate::render::gpu_types::ChunkMetaGpu>() as u64;

        for u in uploads {
            if u.slot < self.buffers.chunk_capacity {
                let meta_off = (u.slot as u64) * meta_stride;
                self.queue
                    .write_buffer(&self.buffers.chunk, meta_off, bytemuck::bytes_of(&u.meta));
            }

            if !u.nodes.is_empty() {
                let needed = u.nodes.len() as u32;

                if u.node_base <= self.buffers.node_capacity
                    && u.node_base + needed <= self.buffers.node_capacity
                {
                    let node_off = (u.node_base as u64) * node_stride;
                    self.queue.write_buffer(
                        &self.buffers.node,
                        node_off,
                        bytemuck::cast_slice(u.nodes.as_ref()),
                    );
                }
            }
        }
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
}
