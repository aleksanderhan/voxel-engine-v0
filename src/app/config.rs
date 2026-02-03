// src/config.rs
// -------------
// Global config knobs for the voxel/SVO renderer + streaming.

pub const CHUNK_SIZE: u32 = 64;

pub const ACTIVE_RADIUS: i32 = 14;
pub const KEEP_RADIUS: i32 = ACTIVE_RADIUS * 5;

pub const VOXEL_SIZE_M_F32: f32 = 0.10;
pub const VOXEL_SIZE_M_F64: f64 = 0.10;

// Keep this explicit so you can change voxel size later without hunting constants.
pub const VOXELS_PER_METER: i32 = 10; // 1.0 / 0.10

pub const RENDER_SCALE: f32 = 0.5;

pub const WORKER_THREADS: usize = 4;
pub const MAX_IN_FLIGHT: usize = 16;

// GPU node arena budget (storage buffer capacity).
pub const NODE_BUDGET_BYTES: usize = 1024 * 1024 * 1024; // 1 GB

// CPU chunk cache budget (SVO nodes stored on CPU so we don't rebuild chunks).
// This is the *total* bytes of cached NodeGpu arrays across all cached chunks.
pub const CHUNK_CACHE_BUDGET_BYTES: usize = 1024 * 1024 * 1024; // 1024 MB

// -----------------------------------------------------------------------------
// Clipmap (far terrain fallback)
// -----------------------------------------------------------------------------
//
// A set of nested 2D height textures around the camera (CPU updated).
// The primary compute shader samples this when the SVO grid doesn't cover the ray
// (or when no voxel hit occurs).
//
// Height units: meters (f32).
//
// NOTE: These constants MUST match shader-side constants in `shaders/clipmap.wgsl`.

pub const CLIPMAP_LEVELS: u32 = 16;
pub const CLIPMAP_LEVELS_USIZE: usize = CLIPMAP_LEVELS as usize;

// Texture resolution per level (square).
pub const CLIPMAP_RES: u32 = 256;

// Base cell size (meters) for level 0. Level i cell size = BASE * 2^i.
pub const CLIPMAP_BASE_CELL_M: f32 = 1.0;

// How often we allow a full refresh per level at most (seconds).
// (Prevents thrashing if you ever tie updates to very tiny camera jitter.)
pub const CLIPMAP_MIN_UPDATE_INTERVAL_S: f32 = 0.0;
