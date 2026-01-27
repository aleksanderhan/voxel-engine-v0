// src/render/state/buffers.rs
//
// Persistent GPU buffers and capacities.
// Kept separate so the main renderer stays focused on passes and frame encoding.

use crate::{
    config,
    render::gpu_types::{ChunkMetaGpu, NodeGpu},
};

pub struct Buffers {
    // Uniforms
    pub camera: wgpu::Buffer,
    pub overlay: wgpu::Buffer,

    // Persistent storage buffers
    pub node: wgpu::Buffer,
    pub chunk: wgpu::Buffer,
    pub chunk_grid: wgpu::Buffer,

    // Capacities (in elements)
    pub node_capacity: u32,
    pub chunk_capacity: u32,
    pub grid_capacity: u32,

    // Dummies kept around (useful for debugging/validation, and occasionally for filler writes)
    pub dummy_node: NodeGpu,
    pub dummy_chunk: ChunkMetaGpu,
}

fn make_uniform_buffer<T: Sized>(device: &wgpu::Device, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<T>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn make_storage_buffer(device: &wgpu::Device, label: &str, size_bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

pub fn create_persistent_buffers(device: &wgpu::Device) -> Buffers {
    // Uniforms
    let camera = make_uniform_buffer::<crate::render::gpu_types::CameraGpu>(device, "camera_buf");
    let overlay =
        make_uniform_buffer::<crate::render::gpu_types::OverlayGpu>(device, "overlay_buf");

    // Dummy structs (mostly for debugging / optional fallback writes)
    let dummy_node = NodeGpu {
        child_base: 0xFFFF_FFFF,
        child_mask: 0,
        material: 0,
        _pad: 0,
    };
    let dummy_chunk = ChunkMetaGpu {
        origin: [0, 0, 0, 0],
        node_base: 0,
        node_count: 0,
        _pad0: 0,
        _pad1: 0,
    };

    // Node arena capacity derived from budget bytes.
    let node_capacity = (config::NODE_BUDGET_BYTES / std::mem::size_of::<NodeGpu>()) as u32;
    let node = make_storage_buffer(
        device,
        "svo_nodes_arena",
        (node_capacity as u64) * (std::mem::size_of::<NodeGpu>() as u64),
    );

    // Chunk meta capacity: max resident chunks in KEEP box (as defined by your streaming system).
    let chunk_capacity =
        (2 * config::KEEP_RADIUS + 1) as u32 * 4u32 * (2 * config::KEEP_RADIUS + 1) as u32;

    let chunk = make_storage_buffer(
        device,
        "chunk_meta_persistent",
        (chunk_capacity as u64) * (std::mem::size_of::<ChunkMetaGpu>() as u64),
    );

    // Chunk grid buffer (fixed capacity). Stored as u32 indices/handles.
    let grid_capacity = chunk_capacity;
    let chunk_grid = make_storage_buffer(
        device,
        "chunk_grid_buf",
        (grid_capacity as u64) * (std::mem::size_of::<u32>() as u64),
    );

    Buffers {
        camera,
        overlay,
        node,
        chunk,
        chunk_grid,
        node_capacity,
        chunk_capacity,
        grid_capacity,
        dummy_node,
        dummy_chunk,
    }
}
