// src/config.rs
// -------------
// World + voxel config shared across CPU and GPU assumptions.

pub const VOXEL_SIZE_M_F32: f32 = 0.10;

// Fixed chunk resolution for the SVO build:
// 32^3 voxels -> octree depth 5 (levels 0..5)
pub const CHUNK_VOXELS: u32 = 64;

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

pub const CHUNK_RADIUS_X: i32 = 12;
pub const CHUNK_RADIUS_Y: i32 = 3;
pub const CHUNK_RADIUS_Z: i32 = 12;

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

// -----------------------------------------------------------------------------
// GPU build acceleration data
// -----------------------------------------------------------------------------
//
// Height min/max mip pyramid per chunk (2D):
// L0: 32x32, L1:16x16, L2:8x8, L3:4x4, L4:2x2, L5:1x1
// total texels = 1024+256+64+16+4+1 = 1365
pub const HEIGHT_MIP_TEXELS_PER_CHUNK: u32 = 1365;
pub const HEIGHT_MIP_TOTAL_TEXELS: u32 = MAX_CHUNKS * HEIGHT_MIP_TEXELS_PER_CHUNK;

// -----------------------------------------------------------------------------
// Terrain height (MUST MATCH shaders/world_svo.wgsl terrain_height_m)
// -----------------------------------------------------------------------------

#[inline]
pub fn terrain_height_m(x: f32, z: f32) -> f32 {
    let h0 = 2.0 * (x * 0.05).sin() + 2.0 * (z * 0.05).cos();
    let h1 = 1.2 * ((x + z) * 0.03).sin();
    h0 + h1
}
