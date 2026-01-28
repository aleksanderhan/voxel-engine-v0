// src/config.rs
// -------------
//
// World + voxel config shared across CPU and GPU assumptions.

pub const VOXEL_SIZE_M_F32: f32 = 0.10;

// Fixed chunk resolution for the SVO build:
// 32^3 voxels -> octree depth 5 (levels 0..5)
pub const CHUNK_VOXELS: u32 = 32;

// Compatibility constant used by CPU-side world code (voxel units).
pub const CHUNK_RES_I32: i32 = CHUNK_VOXELS as i32;

// Chunk world size in meters (must match shader logic)
pub const CHUNK_SIZE_M_F32: f32 = VOXEL_SIZE_M_F32 * CHUNK_VOXELS as f32;

// -----------------------------------------------------------------------------
// Render scale (internal resolution)
// -----------------------------------------------------------------------------

pub const RENDER_SCALE: f32 = 0.35;

pub fn render_dims(window_w: u32, window_h: u32) -> (u32, u32) {
    let w = (window_w.max(1) as f32 * RENDER_SCALE).round() as u32;
    let h = (window_h.max(1) as f32 * RENDER_SCALE).round() as u32;
    (w.max(1), h.max(1))
}

// -----------------------------------------------------------------------------
// Streaming policy (dense neighborhood grid)
// -----------------------------------------------------------------------------

pub const CHUNK_RADIUS_X: i32 = 8;
pub const CHUNK_RADIUS_Y: i32 = 2;
pub const CHUNK_RADIUS_Z: i32 = 8;

// With radius 8 => 2*8+1 = 17, radius 2 => 5
pub const GRID_X: u32 = (CHUNK_RADIUS_X as u32) * 2 + 1; // 17
pub const GRID_Y: u32 = (CHUNK_RADIUS_Y as u32) * 2 + 1; // 5
pub const GRID_Z: u32 = (CHUNK_RADIUS_Z as u32) * 2 + 1; // 17

pub const MAX_CHUNKS: u32 = GRID_X * GRID_Y * GRID_Z; // 17*5*17 = 1445

// -----------------------------------------------------------------------------
// Octree sizes
// -----------------------------------------------------------------------------

pub const NODES_PER_CHUNK: u32 = 37449;
pub const TOTAL_NODES: u32 = MAX_CHUNKS * NODES_PER_CHUNK;
