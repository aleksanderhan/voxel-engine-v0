// src/render/state/mod.rs
//
// Multi-pass renderer built around compute shaders plus a final blit render pass.
//
// High-level data flow (textures):
//   primary   (full-res)   : scene tracing + fog
//     writes -> color_tex (RGBA16F) and depth_tex (R32F)
//
//   godray    (quarter-res): light shafts + temporal accumulation (ping-pong)
//     reads  -> depth_tex + godray_history
//     writes -> godray_out
//
//   composite (full-res)   : combine base color with upsampled godrays
//     reads  -> color_tex + godray_out
//     writes -> output_tex (RGBA16F)
//
//   blit      (render)     : present output_tex to the swapchain
//
// Why the bind groups are split the way they are (wgpu hazard rule):
//   In a single pass/dispatch, a texture cannot be bound for STORAGE write access
//   and also be bound for sampling (or any other conflicting usage).
//
//   The primary pass bind group (primary_bg) includes storage bindings for color/depth.
//   The godray pass must sample depth_tex, so it must *not* use primary_bg.
//   Instead, godray uses scene_bg (buffers only) + a separate godray_bg that samples depth.
//
// Bind groups and shader group indices (must match WGSL @group/@binding):
//   group(0) primary_bg   : camera + scene storage buffers + primary storage outputs (primary pass only)
//   group(0) scene_bg     : camera + scene storage buffers (NO storage textures)     (godray pass)
//   group(1) godray_bg    : depth sampled + history sampled + out storage           (godray pass)
//   group(2) composite_bg : color sampled + godray sampled + output storage         (composite pass)
//
// Composite special-case (pipeline layout vs shader usage):
//   The composite pipeline layout includes group(0), group(1), group(2),
//   but the composite shader only actually needs group(2).
//   wgpu still requires bind groups to be set for earlier groups, so we bind empty_bg
//   for group(0) and group(1) as placeholders.

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

/// Renderer owns the wgpu Device/Queue and all GPU-side resources needed to render frames.
///
/// Design goals:
/// - Keep per-frame work to "write updated buffers + encode passes".
/// - Keep persistent GPU buffers (nodes/chunks/grid) allocated once.
/// - Recreate only size-dependent resources (textures + bind groups) on resize.
/// - Encapsulate ping-pong state for temporal effects (godray history).
pub struct Renderer {
    /// Logical device used to create and encode GPU work.
    device: wgpu::Device,
    /// Submission queue for buffer uploads and command buffers.
    queue: wgpu::Queue,

    // -------------------------------------------------------------------------
    // Shared resources
    // -------------------------------------------------------------------------

    /// Sampler used by the final blit pass (and any other filterable sampling).
    sampler: wgpu::Sampler,

    // -------------------------------------------------------------------------
    // Resource sets (split for clarity and to make resize rebuilds local)
    // -------------------------------------------------------------------------

    /// Bind group layouts (the "ABI" between WGSL bindings and wgpu).
    layouts: Layouts,
    /// Pipelines for each pass (compute primary/godray/composite + render blit).
    pipelines: Pipelines,
    /// Persistent GPU buffers (camera, overlay, nodes, chunk meta, chunk grid).
    buffers: Buffers,
    /// Size-dependent textures (color/depth/godray ping-pong/output).
    textures: TextureSet,
    /// Concrete bind groups pointing at the current buffers + texture views.
    bind_groups: BindGroups,

    // -------------------------------------------------------------------------
    // Ping-pong state for temporal accumulation
    // -------------------------------------------------------------------------

    /// Index of the "history" godray texture for *this* frame.
    /// The other index is the "write target" for this frame.
    ping: usize,
}

impl Renderer {
    /// Create a renderer using an already-selected adapter and the chosen surface format.
    ///
    /// `width`/`height` are the initial output resolution; textures and bind groups are
    /// built for this size.
    pub async fn new(
        adapter: &wgpu::Adapter,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        // WGPU limits:
        // We mostly keep defaults, but explicitly preserve the adapter's max buffer sizes
        // so large STORAGE buffers (node arena, chunk meta) are allowed.
        let adapter_limits = adapter.limits();
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_buffer_size: adapter_limits.max_buffer_size,
            ..wgpu::Limits::default()
        };

        // Request a device/queue pair. No special features required here.
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

        // ---------------------------------------------------------------------
        // Shader modules
        // ---------------------------------------------------------------------

        // Single WGSL module that contains the compute entry points (primary/godray/composite).
        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ray_cs"),
            source: wgpu::ShaderSource::Wgsl(crate::render::shaders::ray_cs_wgsl().into()),
        });

        // WGSL module for the final fullscreen blit render pipeline.
        let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(crate::render::shaders::blit_wgsl().into()),
        });

        // Sampler used in the blit pass.
        // Compute passes intentionally use unfiltered sampling (declared via TextureSampleType).
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("nearest_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // ---------------------------------------------------------------------
        // One-time, size-independent resources
        // ---------------------------------------------------------------------

        // Layouts define the binding contract. Buffers are long-lived arenas.
        let layouts = create_layouts(&device);
        let buffers = create_persistent_buffers(&device);

        // ---------------------------------------------------------------------
        // Size-dependent resources
        // ---------------------------------------------------------------------

        // Textures depend on output size; bind groups depend on texture views.
        let textures = create_textures(&device, width, height);

        // Pipelines depend on layouts + shader modules (+ surface format for blit).
        let pipelines = create_pipelines(&device, &layouts, &cs_module, &fs_module, surface_format);

        // Bind groups connect layouts to the actual buffers/textures currently in use.
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

    /// Expose device so the app can configure the surface, create encoders, etc.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Expose queue for submissions and CPU->GPU buffer writes.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    // -------------------------------------------------------------------------
    // Resizing
    // -------------------------------------------------------------------------

    /// Handle output resize.
    ///
    /// What changes with size:
    /// - Textures (color/depth/godray/output) must be recreated.
    /// - Bind groups that reference those texture views must be recreated.
    ///
    /// What does *not* change with size:
    /// - Persistent buffers (nodes/chunks/grid) and pipelines/layouts.
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

        // Reset ping-pong after resize so we don't blend history from an incompatible size.
        self.ping = 0;
    }

    // -------------------------------------------------------------------------
    // Upload / write helpers
    // -------------------------------------------------------------------------

    /// Update the chunk grid indirection table on the GPU.
    ///
    /// The grid is clamped to the allocated capacity to avoid OOB writes.
    pub fn write_chunk_grid(&self, grid: &[u32]) {
        let n = grid.len().min(self.buffers.grid_capacity as usize);
        self.queue.write_buffer(
            &self.buffers.chunk_grid,
            0,
            bytemuck::cast_slice(&grid[..n]),
        );
    }

    /// Update camera uniforms (typically once per frame).
    pub fn write_camera(&self, cam: &CameraGpu) {
        self.queue
            .write_buffer(&self.buffers.camera, 0, bytemuck::bytes_of(cam));
    }

    /// Update overlay uniforms (FPS, dimensions, etc.).
    pub fn write_overlay(&self, ov: &OverlayGpu) {
        self.queue
            .write_buffer(&self.buffers.overlay, 0, bytemuck::bytes_of(ov));
    }

    /// Apply streamed chunk uploads into the persistent GPU arenas.
    ///
    /// Each upload can contain:
    /// - `meta` written into the chunk metadata array at `slot`
    /// - optional `nodes` written into the node arena starting at `node_base`
    ///
    /// Bounds checks prevent writing past allocated capacities (silently drops overflow).
    pub fn apply_chunk_uploads(&self, uploads: Vec<ChunkUpload>) {
        let node_stride = std::mem::size_of::<crate::render::gpu_types::NodeGpu>() as u64;
        let meta_stride = std::mem::size_of::<crate::render::gpu_types::ChunkMetaGpu>() as u64;

        for u in uploads {
            // Chunk metadata write (always present).
            if u.slot < self.buffers.chunk_capacity {
                let meta_off = (u.slot as u64) * meta_stride;
                self.queue
                    .write_buffer(&self.buffers.chunk, meta_off, bytemuck::bytes_of(&u.meta));
            }

            // Node payload write (only when nodes are included for this chunk).
            if !u.nodes.is_empty() {
                let needed = u.nodes.len() as u32;

                // Ensure [node_base, node_base + needed) fits within the node arena.
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

    /// Encode the compute pipeline sequence: primary -> godray -> composite.
    ///
    /// Expectations:
    /// - `width`/`height` match the current full-resolution output size.
    /// - Godray pass runs at quarter resolution (see `quarter_dim`).
    /// - Godray uses ping-pong textures:
    ///     ping = history (read)
    ///     pong = output  (write)
    pub fn encode_compute(&mut self, encoder: &mut wgpu::CommandEncoder, width: u32, height: u32) {
        // ---------------------------------------------------------------------
        // Pass 1: primary (full-res)
        //
        // Writes:
        //   color_tex (storage rgba16f)
        //   depth_tex (storage r32f)
        // ---------------------------------------------------------------------
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("primary_pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.pipelines.primary);
            cpass.set_bind_group(0, &self.bind_groups.primary, &[]);

            // Dispatch grid assumes 8x8 workgroup size in WGSL.
            let gx = (width + 7) / 8;
            let gy = (height + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        // ---------------------------------------------------------------------
        // Pass 2: godray (quarter-res)
        //
        // Reads:
        //   depth_tex (sampled)
        //   godray_history (sampled)
        // Writes:
        //   godray_out (storage)
        //
        // Binding split:
        //   group(0) = scene_bg (buffers only)
        //   group(1) = godray_bg[ping] (depth + history + out)
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

            // group(0): camera + world buffers (no storage textures)
            cpass.set_bind_group(0, &self.bind_groups.scene, &[]);

            // group(1): ping-pong bindings
            // `godray_bg[ping]` reads history=godray[ping], writes out=godray[pong].
            cpass.set_bind_group(1, &self.bind_groups.godray[ping], &[]);

            let gx = (qw + 7) / 8;
            let gy = (qh + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        // ---------------------------------------------------------------------
        // Pass 3: composite (full-res)
        //
        // Reads:
        //   color_tex (sampled)
        //   godray[pong] (sampled)  <- result written in the godray pass above
        // Writes:
        //   output_tex (storage)
        //
        // Pipeline layout expects group(0), group(1), group(2).
        // Shader only uses group(2), so we bind empty groups for 0 and 1.
        // ---------------------------------------------------------------------
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("composite_pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.pipelines.composite);

            // Satisfy pipeline layout slots even if unused by the shader.
            cpass.set_bind_group(0, &self.bind_groups.empty, &[]);
            cpass.set_bind_group(1, &self.bind_groups.empty, &[]);

            // group(2): color + godray[pong] + output
            cpass.set_bind_group(2, &self.bind_groups.composite[pong], &[]);

            let gx = (width + 7) / 8;
            let gy = (height + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        // Advance ping-pong:
        // The texture we wrote this frame becomes "history" next frame.
        self.ping = pong;
    }

    /// Encode the final blit render pass.
    ///
    /// This draws a fullscreen triangle and samples the renderer's `output` texture
    /// into the swapchain `frame_view`. Overlay uniforms are read in the fragment shader.
    pub fn encode_blit(&self, encoder: &mut wgpu::CommandEncoder, frame_view: &wgpu::TextureView) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("blit_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: frame_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Clear first (helps avoid undefined pixels if something goes wrong).
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
        // Fullscreen triangle (3 vertices), one instance.
        rpass.draw(0..3, 0..1);
    }
}
