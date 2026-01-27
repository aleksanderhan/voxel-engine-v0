// src/render/state/buffers.rs
//
// Persistent GPU buffers and capacities.
//
// This module owns the long-lived GPU buffers used across frames.
// It's separated out so the renderer code can focus on passes/encoding, while
// buffer sizing/allocation logic lives here.
//
// Buffer types used here:
// - UNIFORM buffers: small per-frame parameter blocks (camera, overlay).
// - STORAGE buffers: larger arenas updated incrementally (nodes, chunk metadata, chunk grid).

use crate::{
    config,
    render::gpu_types::{ChunkMetaGpu, NodeGpu},
};

/// Collection of GPU buffers that persist for the lifetime of the renderer.
///
/// These buffers are recreated only when the renderer is created (or when you
/// decide to change capacities). They are written to each frame via COPY_DST.
pub struct Buffers {
    // --- Uniforms (small, frequently updated) ---

    /// Camera uniforms (inverse matrices, camera position, grid params, etc.).
    /// Written each frame.
    pub camera: wgpu::Buffer,

    /// Overlay uniforms (FPS and screen size, etc.).
    /// Written each frame.
    pub overlay: wgpu::Buffer,

    // --- Persistent storage buffers (larger arenas) ---

    /// Node arena (likely an SVO / acceleration structure node pool).
    /// Capacity is derived from a fixed byte budget.
    pub node: wgpu::Buffer,

    /// Chunk metadata array for all resident chunks (as determined by streaming).
    pub chunk: wgpu::Buffer,

    /// Chunk grid indirection table (u32 handles/indices mapping 3D grid -> chunk slot).
    pub chunk_grid: wgpu::Buffer,

    // --- Capacities (element counts, not bytes) ---

    /// Number of NodeGpu elements the node arena can hold.
    pub node_capacity: u32,

    /// Number of ChunkMetaGpu elements the chunk metadata buffer can hold.
    pub chunk_capacity: u32,

    /// Number of u32 entries in the chunk grid buffer.
    pub grid_capacity: u32,
}

/// Helper to create a fixed-size uniform buffer for some POD-ish type `T`.
///
/// Usage flags:
/// - UNIFORM: bind as uniform buffer in shaders
/// - COPY_DST: allow updating via queue.write_buffer / copy operations
fn make_uniform_buffer<T: Sized>(device: &wgpu::Device, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        // Uniform holds exactly one T.
        size: std::mem::size_of::<T>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Helper to create a storage buffer of a specific byte size.
///
/// Usage flags:
/// - STORAGE: bind as storage buffer (read/write in compute, read in fragment/vertex if allowed)
/// - COPY_DST: allow CPU uploads
fn make_storage_buffer(device: &wgpu::Device, label: &str, size_bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Create all persistent buffers with capacities derived from config constants.
///
/// Important: this only allocates GPU memory. The actual contents are expected
/// to be filled/updated by streaming + renderer code at runtime.
pub fn create_persistent_buffers(device: &wgpu::Device) -> Buffers {
    // --- Uniform buffers ---

    // Camera uniform buffer (one CameraGpu struct).
    let camera = make_uniform_buffer::<crate::render::gpu_types::CameraGpu>(device, "camera_buf");

    // Overlay uniform buffer (one OverlayGpu struct).
    let overlay =
        make_uniform_buffer::<crate::render::gpu_types::OverlayGpu>(device, "overlay_buf");

    // --- Storage buffers ---

    // Node arena capacity derived from a fixed byte budget.
    //
    // NOTE: integer division truncates, which is fine: we allocate as many whole NodeGpu
    // elements as fit into NODE_BUDGET_BYTES.
    let node_capacity = (config::NODE_BUDGET_BYTES / std::mem::size_of::<NodeGpu>()) as u32;

    // Allocate node arena buffer sized to hold `node_capacity` NodeGpu elements.
    let node = make_storage_buffer(
        device,
        "svo_nodes_arena",
        (node_capacity as u64) * (std::mem::size_of::<NodeGpu>() as u64),
    );

    // Chunk meta capacity: max resident chunks in the KEEP box (streaming working set).
    //
    // The formula here implies:
    // - X dimension: (2*KEEP_RADIUS + 1)
    // - Z dimension: (2*KEEP_RADIUS + 1)
    // - Y dimension: 4 (hard-coded vertical span; likely number of chunk layers kept)
    //
    // So total = X * Y * Z.
    let chunk_capacity =
        (2 * config::KEEP_RADIUS + 1) as u32 * 4u32 * (2 * config::KEEP_RADIUS + 1) as u32;

    // Allocate chunk metadata buffer sized to hold `chunk_capacity` ChunkMetaGpu elements.
    let chunk = make_storage_buffer(
        device,
        "chunk_meta_persistent",
        (chunk_capacity as u64) * (std::mem::size_of::<ChunkMetaGpu>() as u64),
    );

    // Chunk grid buffer (fixed capacity).
    //
    // Stored as u32 indices/handles, likely mapping each grid cell to a slot in `chunk`
    // or a sentinel for "empty".
    // Here it's set equal to chunk_capacity, meaning "one u32 per chunk slot".
    let grid_capacity = chunk_capacity;

    let chunk_grid = make_storage_buffer(
        device,
        "chunk_grid_buf",
        (grid_capacity as u64) * (std::mem::size_of::<u32>() as u64),
    );

    // Return the assembled buffer set and their capacities.
    Buffers {
        camera,
        overlay,
        node,
        chunk,
        chunk_grid,
        node_capacity,
        chunk_capacity,
        grid_capacity,
    }
}
