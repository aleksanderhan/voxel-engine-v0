// src/render/state/mod.rs
// -----------------------

mod bindgroups;
mod buffers;
mod layout;
mod pipelines;
mod streaming;
pub mod textures;

use crate::config;
use crate::render::gpu_types::{CameraGpu, OverlayGpu};
use crate::world;

use bindgroups::{create_bind_groups, BindGroups};
use buffers::{create_persistent_buffers, Buffers};
use layout::{create_layouts, Layouts};
use pipelines::{create_pipelines, Pipelines};
use textures::{create_textures, quarter_dim, TextureSet};

use streaming::StreamingState;

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

    world_built: bool,
    streaming: StreamingState,

    // internal render size (scaled)
    render_w: u32,
    render_h: u32,
}

impl Renderer {
    pub async fn new(
        adapter: &wgpu::Adapter,
        surface_format: wgpu::TextureFormat,
        render_w: u32,
        render_h: u32,
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

        // Upscale looks much better with linear filtering; cost is tiny vs raymarching.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blit_sampler_linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let layouts = create_layouts(&device);
        let buffers = create_persistent_buffers(&device);

        let textures = create_textures(&device, render_w, render_h);
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
            world_built: false,
            streaming: StreamingState::new(),
            render_w: render_w.max(1),
            render_h: render_h.max(1),
        }
    }

    // -------------------------------------------------------------------------
    // Streaming
    // -------------------------------------------------------------------------

    pub fn update_center_chunk(&mut self, center: world::ChunkCoord) {
        let changed = self.streaming.update_center(center);
        if !changed {
            return;
        }

        self.queue.write_buffer(
            &self.buffers.chunk_meta,
            0,
            bytemuck::cast_slice(self.streaming.meta()),
        );

        self.queue.write_buffer(
            &self.buffers.dirty_slots,
            0,
            bytemuck::cast_slice(self.streaming.dirty_slots()),
        );

        let s = self.streaming.stream_gpu();
        self.queue.write_buffer(&self.buffers.stream, 0, bytemuck::bytes_of(&s));

        self.world_built = false;
    }

    pub fn request_world_rebuild(&mut self) {
        self.streaming.mark_all_dirty();

        self.queue.write_buffer(
            &self.buffers.chunk_meta,
            0,
            bytemuck::cast_slice(self.streaming.meta()),
        );
        self.queue.write_buffer(
            &self.buffers.dirty_slots,
            0,
            bytemuck::cast_slice(self.streaming.dirty_slots()),
        );

        let s = self.streaming.stream_gpu();
        self.queue.write_buffer(&self.buffers.stream, 0, bytemuck::bytes_of(&s));

        self.world_built = false;
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    pub fn device(&self) -> &wgpu::Device { &self.device }
    pub fn queue(&self) -> &wgpu::Queue { &self.queue }

    pub fn render_size(&self) -> (u32, u32) {
        (self.render_w, self.render_h)
    }

    // -------------------------------------------------------------------------
    // Resizing
    // -------------------------------------------------------------------------

    pub fn resize_output(&mut self, window_w: u32, window_h: u32) {
        let (rw, rh) = config::render_dims(window_w, window_h);
        self.render_w = rw;
        self.render_h = rh;

        self.textures = create_textures(&self.device, rw, rh);
        self.bind_groups = create_bind_groups(
            &self.device,
            &self.layouts,
            &self.buffers,
            &self.textures,
            &self.sampler,
        );
        self.ping = 0;
    }

    // -------------------------------------------------------------------------
    // Uniform writes
    // -------------------------------------------------------------------------

    pub fn write_camera(&self, cam: &CameraGpu) {
        self.queue.write_buffer(&self.buffers.camera, 0, bytemuck::bytes_of(cam));
    }

    pub fn write_overlay(&self, ov: &OverlayGpu) {
        self.queue.write_buffer(&self.buffers.overlay, 0, bytemuck::bytes_of(ov));
    }

    // -------------------------------------------------------------------------
    // Encode
    // -------------------------------------------------------------------------

    fn encode_world_build(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let dirty_count = self.streaming.stream_gpu().dirty_count;
        if dirty_count == 0 {
            return;
        }

        // build_leaves
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("build_leaves_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.build_leaves);
            cpass.set_bind_group(0, &self.bind_groups.scene, &[]);

            let gx = (32 + 7) / 8;
            let gy = (32 + 7) / 8;
            let gz = 32 * dirty_count;
            cpass.dispatch_workgroups(gx, gy, gz);
        }

        // build_L4
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("build_L4_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.build_l4);
            cpass.set_bind_group(0, &self.bind_groups.scene, &[]);

            let gx = (16 + 7) / 8;
            let gy = (16 + 7) / 8;
            let gz = 16 * dirty_count;
            cpass.dispatch_workgroups(gx, gy, gz);
        }

        // build_L3
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("build_L3_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.build_l3);
            cpass.set_bind_group(0, &self.bind_groups.scene, &[]);
            cpass.dispatch_workgroups(1, 1, 8 * dirty_count);
        }

        // build_L2
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("build_L2_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.build_l2);
            cpass.set_bind_group(0, &self.bind_groups.scene, &[]);
            cpass.dispatch_workgroups(1, 1, 4 * dirty_count);
        }

        // build_L1
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("build_L1_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.build_l1);
            cpass.set_bind_group(0, &self.bind_groups.scene, &[]);
            cpass.dispatch_workgroups(1, 1, 2 * dirty_count);
        }

        // build_L0
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("build_L0_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.build_l0);
            cpass.set_bind_group(0, &self.bind_groups.scene, &[]);
            cpass.dispatch_workgroups(dirty_count, 1, 1);
        }

        // clear_dirty
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clear_dirty_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.clear_dirty);
            cpass.set_bind_group(0, &self.bind_groups.scene, &[]);

            let gx = (dirty_count + 63) / 64;
            cpass.dispatch_workgroups(gx, 1, 1);
        }
    }

    pub fn encode_compute(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if !self.world_built {
            self.encode_world_build(encoder);
            self.world_built = true;
        }

        let width = self.render_w;
        let height = self.render_h;

        // primary
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

        // godray
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

        // composite
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("composite_pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.pipelines.composite);
            cpass.set_bind_group(0, &self.bind_groups.empty, &[]);
            cpass.set_bind_group(1, &self.bind_groups.empty, &[]);
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
