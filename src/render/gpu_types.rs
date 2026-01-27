// src/render/gpu_types.rs
//
// GPU-facing data layouts shared between Rust and WGSL.
//
// These structs are written into uniform/storage buffers and read by shaders.
// Requirements for correctness:
// - `#[repr(C)]` fixes a predictable field order/layout.
// - `Pod` + `Zeroable` (bytemuck) guarantee the types can be safely cast to bytes.
// - Fields are sized/aligned to match WGSL expectations (notably 16-byte alignment rules).
//
// Practical note about padding:
// WGSL has strict alignment for uniform/storage layouts (e.g. vec3/vec4 alignment),
// so we often include explicit `_pad*` fields to keep everything 16-byte aligned.

use bytemuck::{Pod, Zeroable};

/// Node in the GPU-side node arena (likely an SVO / tree / BVH-like structure).
///
/// Expected usage:
/// - Stored in a STORAGE buffer (read-only from shaders in this codebase).
/// - Addressed by index; child pointers are expressed as a base index + mask.
///
/// Field ideas (based on naming):
/// - child_base: base index of this node's children in the global node arena.
/// - child_mask: bitmask indicating which child slots are present/valid.
/// - material: per-node material/voxel payload (id/flags/packed params).
/// - _pad: explicit padding to keep 16-byte stride (nice for storage access patterns).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct NodeGpu {
    pub child_base: u32,
    pub child_mask: u32,
    pub material: u32,
    pub _pad: u32,
}

/// Per-chunk metadata stored in a persistent STORAGE buffer.
///
/// Shaders use this to locate the chunk in world space and map the chunk to its nodes
/// within the global node arena.
///
/// - origin: chunk origin in chunk coordinates (xyz) with a spare lane for alignment.
/// - node_base/node_count: range into the node arena that belongs to this chunk.
/// - _pad*: explicit padding to preserve alignment / struct stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ChunkMetaGpu {
    /// Chunk origin expressed as i32 for easy chunk-space arithmetic in shaders.
    /// origin[3] is unused padding (keeps 16B alignment for the next fields).
    pub origin: [i32; 4],

    /// First node index in the node arena for this chunk.
    pub node_base: u32,

    /// Number of nodes belonging to this chunk.
    pub node_count: u32,

    pub _pad0: u32,
    pub _pad1: u32,
}

/// Camera + frame parameters packed for GPU consumption.
///
/// Stored as a UNIFORM buffer, updated once per frame.
///
/// Contents are tailored for screen-space ray generation / ray marching:
/// - Inverse matrices let the shader reconstruct world-space rays from pixel coords.
/// - cam_pos is vec4-aligned for uniform layout friendliness.
/// - Chunk/grid parameters tell the shader how to interpret streamed chunk data.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CameraGpu {
    /// Inverse view matrix (camera -> world transform).
    pub view_inv: [[f32; 4]; 4],

    /// Inverse projection matrix (clip -> view transform).
    pub proj_inv: [[f32; 4]; 4],

    /// Camera position in world space. Fourth component is padding (or could be 1.0).
    pub cam_pos: [f32; 4],

    /// Voxel chunk edge length (in voxels) for index math in shaders.
    pub chunk_size: u32,

    /// Number of chunk slots currently considered resident/valid.
    pub chunk_count: u32,

    /// Max raymarch/trace steps for the primary shader.
    pub max_steps: u32,

    pub _pad0: u32,

    /// Misc per-frame voxel/shader knobs.
    /// (Meaning is shader-defined; typical uses: voxel size, time, density, jitter, etc.)
    pub voxel_params: [f32; 4],

    /// Chunk grid origin in chunk coordinates (cx0, cy0, cz0, unused).
    /// This is the chunk-space coordinate of grid cell (0,0,0).
    pub grid_origin_chunk: [i32; 4],

    /// Chunk grid dimensions (nx, ny, nz, unused).
    /// Used to bounds-check and map 3D grid coords -> linear index into chunk_grid buffer.
    pub grid_dims: [u32; 4],
}

/// Tiny overlay uniform block for the final blit / UI overlay.
///
/// Stored as a UNIFORM buffer, updated periodically (FPS) and on resize (width/height).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct OverlayGpu {
    /// FPS value displayed in the overlay.
    pub fps: u32,

    /// Output width in pixels.
    pub width: u32,

    /// Output height in pixels.
    pub height: u32,

    pub _pad0: u32,
}
