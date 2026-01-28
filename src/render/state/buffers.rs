// src/render/state/buffers.rs
// ---------------------------

use wgpu::util::DeviceExt;

use crate::{config, render::gpu_types};
use super::streaming::ChunkMetaGpu;

pub struct Buffers {
    pub camera: wgpu::Buffer,
    pub overlay: wgpu::Buffer,
    pub stream: wgpu::Buffer,

    pub nodes: wgpu::Buffer,
    pub chunk_meta: wgpu::Buffer,
    pub dirty_slots: wgpu::Buffer,

    // Height mip pyramid (min/max) per chunk.
    pub height_min: wgpu::Buffer,
    pub height_max: wgpu::Buffer,
}

fn make_uniform_buffer<T: Sized>(device: &wgpu::Device, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<T>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

pub fn create_persistent_buffers(device: &wgpu::Device) -> Buffers {
    let camera = make_uniform_buffer::<gpu_types::CameraGpu>(device, "camera_buf");
    let overlay = make_uniform_buffer::<gpu_types::OverlayGpu>(device, "overlay_buf");
    let stream = make_uniform_buffer::<gpu_types::StreamGpu>(device, "stream_buf");

    let node_count = config::TOTAL_NODES as usize;
    let node_bytes = node_count * std::mem::size_of::<u32>();

    // One-time zero-init at startup. (Large, but not per-frame.)
    let zero = vec![0u8; node_bytes];
    let nodes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("nodes_buf"),
        contents: &zero,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let meta = vec![ChunkMetaGpu::inactive(); config::MAX_CHUNKS as usize];
    let chunk_meta = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("chunk_meta_buf"),
        contents: bytemuck::cast_slice(&meta),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let dirty_zero = vec![0u32; config::MAX_CHUNKS as usize];
    let dirty_slots = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dirty_slots_buf"),
        contents: bytemuck::cast_slice(&dirty_zero),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Height mip: f32 texels
    let height_zero = vec![0.0f32; config::HEIGHT_MIP_TOTAL_TEXELS as usize];
    let height_min = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("height_min_buf"),
        contents: bytemuck::cast_slice(&height_zero),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let height_max = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("height_max_buf"),
        contents: bytemuck::cast_slice(&height_zero),
        usage: wgpu::BufferUsages::STORAGE,
    });

    Buffers {
        camera,
        overlay,
        stream,
        nodes,
        chunk_meta,
        dirty_slots,
        height_min,
        height_max,
    }
}
