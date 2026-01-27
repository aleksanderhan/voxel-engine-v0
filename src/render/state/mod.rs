// src/render/state/mod.rs
//
// Multi-pass compute renderer with persistent GPU buffers.
//
// Passes (compute):
//   1) primary   (full-res): trace + fog -> writes color_tex (rgba16f) + depth_tex (r32f)
//   2) godray    (quarter):  shafts + temporal accumulation -> writes godray ping-pong (rgba16f)
//   3) composite (full-res): color + upsampled godray -> writes output (rgba16f)
//
// Pass (render):
//   blit: samples output to the swapchain.
//
// IMPORTANT (wgpu rule):
//   A texture bound for STORAGE (write) is an exclusive usage within a dispatch.
//   Therefore godray pass must NOT bind primary_bg (which includes depth/color storage outputs),
//   otherwise depth_tex conflicts with being sampled.
//
// Bind groups / shader group indices:
//   group(0) primary_bg  : camera + scene buffers + primary outputs (storage)     (primary pass only)
//   group(0) scene_bg    : camera + scene buffers (no storage outputs!)          (godray pass)
//   group(1) godray_bg   : depth sampled + history sampled + out storage         (godray pass)
//   group(2) composite_bg: color sampled + godray sampled + output storage       (composite pass)
//   group(0/1) empty_bg  : empty bind groups to satisfy pipeline layout indices  (composite pass)

mod bindgroups;
mod buffers;
mod layout;
mod pipelines;
pub mod textures;

use crate::{
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

    // Shared resources
    sampler: wgpu::Sampler,

    // Resource sets (split for clarity and easy resizing/rebuilds)
    layouts: Layouts,
    pipelines: Pipelines,
    buffers: Buffers,
    textures: TextureSet,
    bind_groups: BindGroups,

    // Ping-pong state:
    // ping is the "history" index this frame; pong is written this frame.
    ping: usize,
}

impl Renderer {
    pub async fn new(
        adapter: &wgpu::Adapter,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        // Keep adapter limits, but explicitly copy max sizes so storage buffers can be large.
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

        // Compute + blit shaders.
        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ray_cs"),
            source: wgpu::ShaderSource::Wgsl(crate::render::shaders::ray_cs_wgsl().into()),
        });

        let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(crate::render::shaders::blit_wgsl().into()),
        });

        // Sampler (blit). Compute passes use unfiltered sampling in their BGL sample_type.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("nearest_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // One-time, size-independent objects.
        let layouts = create_layouts(&device);
        let buffers = create_persistent_buffers(&device);

        // Size-dependent objects: textures + bind groups that reference their views.
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

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    // -------------------------------------------------------------------------
    // Resizing
    // -------------------------------------------------------------------------

    pub fn resize_output(&mut self, width: u32, height: u32) {
        // Recreate textures and all bind groups that depend on texture views.
        self.textures = create_textures(&self.device, width, height);
        self.bind_groups = create_bind_groups(
            &self.device,
            &self.layouts,
            &self.buffers,
            &self.textures,
            &self.sampler,
        );

        // Reset ping-pong after resize to avoid mixing mismatched history.
        self.ping = 0;
    }

    // -------------------------------------------------------------------------
    // Upload / write helpers
    // -------------------------------------------------------------------------

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

    pub fn apply_chunk_uploads(&self, uploads: Vec<ChunkUpload>) {
        let node_stride = std::mem::size_of::<crate::render::gpu_types::NodeGpu>() as u64;
        let meta_stride = std::mem::size_of::<crate::render::gpu_types::ChunkMetaGpu>() as u64;

        for u in uploads {
            // Meta write (always).
            if u.slot < self.buffers.chunk_capacity {
                let meta_off = (u.slot as u64) * meta_stride;
                self.queue
                    .write_buffer(&self.buffers.chunk, meta_off, bytemuck::bytes_of(&u.meta));
            }

            // Nodes write (only if provided).
            if !u.nodes.is_empty() {
                let needed = u.nodes.len() as u32;
                if u.node_base <= self.buffers.node_capacity
                    && u.node_base + needed <= self.buffers.node_capacity
                {
                    let node_off = (u.node_base as u64) * node_stride;
                    self.queue
                        .write_buffer(&self.buffers.node, node_off, bytemuck::cast_slice(&u.nodes));
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Encode passes
    // -------------------------------------------------------------------------

    /// Encode the 3 compute passes: primary -> godray -> composite.
    ///
    /// Notes:
    /// - width/height should match the current output size.
    /// - godray is quarter-res and uses ping-pong textures for temporal accumulation.
    pub fn encode_compute(&mut self, encoder: &mut wgpu::CommandEncoder, width: u32, height: u32) {
        // ---------------------------------------------------------------------
        // Pass 1: primary (full-res)
        // ---------------------------------------------------------------------
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("primary_pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.pipelines.primary);
            cpass.set_bind_group(0, &self.bind_groups.primary, &[]);

            // Workgroup sizing assumes 8x8 threads in the WGSL.
            let gx = (width + 7) / 8;
            let gy = (height + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        // ---------------------------------------------------------------------
        // Pass 2: godray (quarter-res)
        //
        // Uses:
        //   group(0) scene_bg   (camera + buffers ONLY)
        //   group(1) godray_bg  (depth sampled + hist sampled + out storage)
        // ---------------------------------------------------------------------
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

            // group(0): camera + world buffers only (no storage outputs bound)
            cpass.set_bind_group(0, &self.bind_groups.scene, &[]);

            // group(1): depth + history + out (ping-pong)
            // godray_bg[ping] reads history=godray[ping], writes out=godray[pong]
            cpass.set_bind_group(1, &self.bind_groups.godray[ping], &[]);

            let gx = (qw + 7) / 8;
            let gy = (qh + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        // ---------------------------------------------------------------------
        // Pass 3: composite (full-res)
        //
        // Pipeline layout includes group(0), group(1), group(2), but shader uses group(2),
        // so we bind empty BGs for group(0/1) and the real BG for group(2).
        // Composite reads the godray texture we *just wrote* (pong).
        // ---------------------------------------------------------------------
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("composite_pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.pipelines.composite);

            // satisfy group indices 0 and 1
            cpass.set_bind_group(0, &self.bind_groups.empty, &[]);
            cpass.set_bind_group(1, &self.bind_groups.empty, &[]);

            // group(2): color + godray[pong] + output
            cpass.set_bind_group(2, &self.bind_groups.composite[pong], &[]);

            let gx = (width + 7) / 8;
            let gy = (height + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        // Next frame, the newly written texture becomes history.
        self.ping = pong;
    }

    /// Blit pass: draws a full-screen triangle sampling `output` into the swapchain.
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
