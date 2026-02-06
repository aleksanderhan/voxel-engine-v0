// src/render/state/mod.rs
// -----------------------
mod bindgroups;
mod buffers;
mod layout;
mod pipelines;
pub mod textures;

use crate::app::config;
use crate::{
    render::gpu_types::{CameraGpu, OverlayGpu},
    streaming::ChunkUpload,
};
use bytemuck::{Pod, cast_slice};

use bindgroups::{create_bind_groups, BindGroups};
use buffers::{create_persistent_buffers, Buffers};
use layout::{create_layouts, Layouts};
use pipelines::{create_pipelines, Pipelines};
use textures::{create_textures, TextureSet};

#[derive(Clone, Copy, Debug, Default)]
pub struct GpuTimingsMs {
    pub primary: f64,
    pub godray: f64,
    pub composite: f64,
    pub blit: f64,
    pub total: f64,
}


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

    // --- GPU timestamp profiling (optional) ---
    ts_enabled: bool,
    ts_period_ns: f64,              // ns per timestamp tick
    ts_qs: Option<wgpu::QuerySet>,
    ts_resolve: Option<wgpu::Buffer>,   // QUERY_RESOLVE | COPY_SRC
    ts_readback: Option<wgpu::Buffer>,  // COPY_DST | MAP_READ

}

#[derive(Clone)]
struct Region {
    off: u64,
    data: Vec<u8>,
}

fn merge_adjacent(mut regions: Vec<Region>) -> Vec<Region> {
    if regions.is_empty() { return regions; }
    regions.sort_by_key(|r| r.off);

    let mut out: Vec<Region> = Vec::with_capacity(regions.len());
    out.push(regions.remove(0));

    for r in regions {
        let last = out.last_mut().unwrap();
        let last_end = last.off + last.data.len() as u64;

        if r.off == last_end {
            last.data.extend_from_slice(&r.data);
        } else {
            out.push(r);
        }
    }
    out
}

// T must be Pod so we can cast to bytes safely.
fn push_typed_region<T: Pod>(regions: &mut Vec<Region>, off_elems: u32, stride: u64, slice: &[T]) {
    if slice.is_empty() { return; }
    let off = (off_elems as u64) * stride;
    let bytes: &[u8] = bytemuck::cast_slice(slice);
    regions.push(Region { off, data: bytes.to_vec() });
}


fn align_up(v: usize, a: usize) -> usize {
    (v + (a - 1)) & !(a - 1)
}

fn batch_write_meta(
    queue: &wgpu::Queue,
    dst: &wgpu::Buffer,
    meta_stride: u64,
    uploads: &[crate::streaming::ChunkUpload],
    chunk_capacity: u32,
) {
    // Collect (slot, meta)
    let mut items: Vec<(u32, crate::render::gpu_types::ChunkMetaGpu)> = Vec::with_capacity(uploads.len());
    for u in uploads {
        if u.slot < chunk_capacity {
            items.push((u.slot, u.meta));
        }
    }
    if items.is_empty() { return; }

    // Sort by slot so we can find contiguous runs
    items.sort_by_key(|(slot, _)| *slot);

    // Emit contiguous runs of slots as one write each
    let mut i = 0usize;
    while i < items.len() {
        let start_slot = items[i].0;
        let mut run_len = 1usize;

        while i + run_len < items.len() && items[i + run_len].0 == start_slot + run_len as u32 {
            run_len += 1;
        }

        // Build a contiguous vec of metas and write once
        let mut run: Vec<crate::render::gpu_types::ChunkMetaGpu> = Vec::with_capacity(run_len);
        for k in 0..run_len {
            run.push(items[i + k].1);
        }

        let dst_off = (start_slot as u64) * meta_stride;
        queue.write_buffer(dst, dst_off, cast_slice(&run));

        i += run_len;
    }
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

        let adapter_features = adapter.features();
        let want = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        let use_ts = adapter_features.contains(want);
        let required_features = if use_ts { want } else { wgpu::Features::empty() };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("device"),
                    required_features,
                    required_limits,
                },
                None,
            )
            .await
            .unwrap();

        const TS_COUNT: u32 = 8;
        let (ts_qs, ts_resolve, ts_readback, ts_period_ns) = if use_ts {
            let qs = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("ts_qs"),
                ty: wgpu::QueryType::Timestamp,
                count: TS_COUNT,
            });

            let bytes = (TS_COUNT as u64) * 8; // u64 timestamps
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ts_resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ts_readback"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let period = queue.get_timestamp_period() as f64; // ns per tick
            (Some(qs), Some(resolve), Some(readback), period)
        } else {
            (None, None, None, 0.0)
        };

        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ray_cs"),
            source: wgpu::ShaderSource::Wgsl(crate::render::shaders::ray_cs_wgsl().into()),
        });

        let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(crate::render::shaders::blit_wgsl().into()),
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("nearest_clamp_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest, // level 0 anyway
            ..Default::default()
        });

        let layouts = create_layouts(&device);
        let buffers = create_persistent_buffers(&device);

        let pipelines = create_pipelines(&device, &layouts, &cs_module, &fs_module, surface_format);

        let render_scale = config::RENDER_SCALE
            .clamp(config::PRIMARY_SCALE_MIN, config::PRIMARY_SCALE_MAX);
        let internal_w = ((width as f32) * render_scale).round().max(1.0) as u32;
        let internal_h = ((height as f32) * render_scale).round().max(1.0) as u32;
        
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
            ts_enabled: use_ts,
            ts_period_ns,
            ts_qs,
            ts_resolve,
            ts_readback,
        }
    }

    #[inline]
    fn ts_pair(&self, begin: u32, end: u32) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        if !self.ts_enabled { return None; }
        Some(wgpu::ComputePassTimestampWrites {
            query_set: self.ts_qs.as_ref().unwrap(),
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        })
    }

    #[inline]
    fn ts_pair_rp(&self, begin: u32, end: u32) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        if !self.ts_enabled { return None; }
        Some(wgpu::RenderPassTimestampWrites {
            query_set: self.ts_qs.as_ref().unwrap(),
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn resize_output(&mut self, width: u32, height: u32) {
        self.internal_w = ((width as f32) * self.render_scale).round().max(1.0) as u32;
        self.internal_h = ((height as f32) * self.render_scale).round().max(1.0) as u32;

        self.textures = create_textures(&self.device, width, height, self.internal_w, self.internal_h);

        self.bind_groups = create_bind_groups(
            &self.device, &self.layouts, &self.buffers, &self.textures, &self.sampler,
        );

        self.ping = 0;
    }

    pub fn update_primary_scale(&mut self, width: u32, height: u32, primary_ms: f64) -> bool {
        if primary_ms <= 0.0 {
            return false;
        }

        let current = self.render_scale;
        let target = config::PRIMARY_TARGET_MS.max(0.1);
        let ratio = (target / primary_ms as f32).sqrt();
        let desired = (current * ratio)
            .clamp(config::PRIMARY_SCALE_MIN, config::PRIMARY_SCALE_MAX);
        let next = current + (desired - current) * config::PRIMARY_SCALE_SMOOTH;

        if (next - current).abs() < config::PRIMARY_SCALE_EPS {
            return false;
        }

        self.render_scale = next;
        self.internal_w = ((width as f32) * self.render_scale).round().max(1.0) as u32;
        self.internal_h = ((height as f32) * self.render_scale).round().max(1.0) as u32;

        self.textures = create_textures(&self.device, width, height, self.internal_w, self.internal_h);
        self.bind_groups = create_bind_groups(
            &self.device, &self.layouts, &self.buffers, &self.textures, &self.sampler,
        );
        self.ping = 0;
        true
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
                timestamp_writes: self.ts_pair(0, 1),
            });

            cpass.set_pipeline(&self.pipelines.primary);
            cpass.set_bind_group(0, &self.bind_groups.primary, &[]);

            let gx = (self.internal_w + 7) / 8;
            let gy = (self.internal_h + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);

        }

        if self.internal_w != width || self.internal_h != height {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("depth_resolve_pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.pipelines.depth_resolve);
            cpass.set_bind_group(0, &self.bind_groups.empty, &[]);
            cpass.set_bind_group(1, &self.bind_groups.empty, &[]);
            cpass.set_bind_group(2, &self.bind_groups.empty, &[]);
            cpass.set_bind_group(3, &self.bind_groups.depth_resolve, &[]);

            let gx = (width + 7) / 8;
            let gy = (height + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        let ping = self.ping;
        let pong = 1 - ping;

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("godray_pass"),
                timestamp_writes: self.ts_pair(2, 3),
            });

            cpass.set_pipeline(&self.pipelines.godray);
            cpass.set_bind_group(0, &self.bind_groups.scene, &[]);
            cpass.set_bind_group(1, &self.bind_groups.godray[ping], &[]);

            let gx = (self.internal_w + 7) / 8;
            let gy = (self.internal_h + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("composite_pass"),
                timestamp_writes: self.ts_pair(4, 5),
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
            timestamp_writes: self.ts_pair_rp(6, 7),
            occlusion_query_set: None,
        });

        rpass.set_pipeline(&self.pipelines.blit);
        rpass.set_bind_group(0, &self.bind_groups.blit, &[]);
        rpass.draw(0..3, 0..1);
    }

    pub fn apply_chunk_uploads(&self, uploads: &[ChunkUpload]) {
        let node_stride = std::mem::size_of::<crate::render::gpu_types::NodeGpu>() as u64;
        let meta_stride = std::mem::size_of::<crate::render::gpu_types::ChunkMetaGpu>() as u64;
        let u32_stride  = std::mem::size_of::<u32>() as u64;
        let rope_stride = std::mem::size_of::<crate::render::gpu_types::NodeRopesGpu>() as u64;

        // 1) Meta in big runs
        batch_write_meta(
            self.queue(),
            &self.buffers.chunk,
            meta_stride,
            uploads,
            self.buffers.chunk_capacity,
        );

        // 2) Nodes / macro / ropes / colinfo as merged byte regions
        let mut node_regions: Vec<Region> = Vec::new();
        let mut macro_regions: Vec<Region> = Vec::new();
        let mut rope_regions: Vec<Region> = Vec::new();
        let mut colinfo_regions: Vec<Region> = Vec::new();

        for u in uploads {
            // nodes
            if !u.nodes.is_empty() {
                let needed = u.nodes.len() as u32;
                if u.node_base + needed <= self.buffers.node_capacity {
                    push_typed_region(&mut node_regions, u.node_base, node_stride, u.nodes.as_ref());
                }
            }

            // macro occupancy (u32)
            if !u.macro_words.is_empty() {
                let needed = u.macro_words.len() as u32;
                if u.meta.macro_base + needed <= self.buffers.macro_capacity_u32 {
                    push_typed_region(&mut macro_regions, u.meta.macro_base, u32_stride, u.macro_words.as_ref());
                }
            }

            // ropes
            if !u.ropes.is_empty() {
                let needed = u.ropes.len() as u32;
                if u.node_base + needed <= self.buffers.rope_capacity {
                    push_typed_region(&mut rope_regions, u.node_base, rope_stride, u.ropes.as_ref());
                }
            }

            // colinfo (u32)
            if !u.colinfo_words.is_empty() {
                let needed = u.colinfo_words.len() as u32;
                if u.meta.colinfo_base + needed <= self.buffers.colinfo_capacity_u32 {
                    push_typed_region(&mut colinfo_regions, u.meta.colinfo_base, u32_stride, u.colinfo_words.as_ref());
                }
            }
        }

        // Merge adjacent regions so we do far fewer write_buffer calls
        for r in merge_adjacent(node_regions) {
            self.queue.write_buffer(&self.buffers.node, r.off, &r.data);
        }
        for r in merge_adjacent(macro_regions) {
            self.queue.write_buffer(&self.buffers.macro_occ, r.off, &r.data);
        }
        for r in merge_adjacent(rope_regions) {
            self.queue.write_buffer(&self.buffers.node_ropes, r.off, &r.data);
        }
        for r in merge_adjacent(colinfo_regions) {
            self.queue.write_buffer(&self.buffers.colinfo, r.off, &r.data);
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

            let row_bytes = w * 4;                 // R32Float => 4 bytes/texel
            let padded = align_up(row_bytes, 256); // required
            let needed = padded * h;

            scratch.clear();
            scratch.resize(needed, 0);

            // copy row-by-row into padded scratch
            let src: &[u8] = bytemuck::cast_slice(&u.data_f32);
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

    pub fn encode_timestamp_resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if !self.ts_enabled { return; }
        let qs = self.ts_qs.as_ref().unwrap();
        let resolve = self.ts_resolve.as_ref().unwrap();
        let readback = self.ts_readback.as_ref().unwrap();
        encoder.resolve_query_set(qs, 0..8, resolve, 0);
        encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, 8 * 8);
    }

    pub fn read_gpu_timings_ms_blocking(&self) -> Option<GpuTimingsMs> {
        if !self.ts_enabled { return None; }

        // Kick off an async map on the readback buffer
        let readback = self.ts_readback.as_ref().unwrap();
        let slice = readback.slice(..);

        // one-shot channel to wait for map completion
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();

        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });

        // Ensure GPU finished the copy into readback AND drive mapping to completion
        self.device.poll(wgpu::Maintain::Wait);

        // Wait for the mapping callback
        let map_ok = pollster::block_on(async { rx.receive().await })
            .expect("map_async dropped")
            .expect("map_async failed");

        // Read 8 u64 timestamps
        let data = slice.get_mapped_range();
        let words: &[u64] = bytemuck::cast_slice(&data);
        if words.len() < 8 {
            drop(data);
            readback.unmap();
            return None;
        }

        let ts = [words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7]];

        drop(data);
        readback.unmap();

        // Convert ticks -> ms
        // ts_period_ns = ns per tick
        let ns = self.ts_period_ns;
        let to_ms = |a: u64, b: u64| -> f64 { (b.saturating_sub(a) as f64) * ns * 1e-6 };

        let primary   = to_ms(ts[0], ts[1]);
        let godray    = to_ms(ts[2], ts[3]);
        let composite = to_ms(ts[4], ts[5]);
        let blit      = to_ms(ts[6], ts[7]);

        Some(GpuTimingsMs {
            primary,
            godray,
            composite,
            blit,
            total: primary + godray + composite + blit,
        })
    }


}
