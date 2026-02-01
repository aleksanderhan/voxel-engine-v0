// Global config knobs for the voxel/SVO renderer + streaming.

pub const CHUNK_SIZE: u32 = 64;

pub const ACTIVE_RADIUS: i32 = 14;
pub const KEEP_RADIUS: i32 = ACTIVE_RADIUS * 5;

pub const VOXEL_SIZE_M_F32: f32 = 0.10;
pub const VOXEL_SIZE_M_F64: f64 = 0.10;

// Keep this explicit so you can change voxel size later without hunting constants.
pub const VOXELS_PER_METER: i32 = 10; // 1.0 / 0.10

pub const RENDER_SCALE: f32 = 0.5;

pub const WORKER_THREADS: usize = 8;
pub const MAX_IN_FLIGHT: usize = 16;

// GPU node arena budget (storage buffer capacity).
pub const NODE_BUDGET_BYTES: usize = 1024 * 1024 * 1024; // 1 GB

// CPU chunk cache budget (SVO nodes stored on CPU so we don't rebuild chunks).
pub const CHUNK_CACHE_BUDGET_BYTES: usize = 1024 * 1024 * 1024; // 1024 MB

// -----------------------------------------------------------------------------
// Clipmap (far terrain fallback)
// -----------------------------------------------------------------------------

pub const CLIPMAP_LEVELS: u32 = 16;
pub const CLIPMAP_LEVELS_USIZE: usize = CLIPMAP_LEVELS as usize;

// Texture resolution per level (square).
pub const CLIPMAP_RES: u32 = 256;

// Base cell size (meters) for level 0. Level i cell size = BASE * 2^i.
pub const CLIPMAP_BASE_CELL_M: f32 = 1.0;

// How often we allow a full refresh per level at most (seconds).
pub const CLIPMAP_MIN_UPDATE_INTERVAL_S: f32 = 0.0;

// Spawn a tiny bit in front of the camera (meters)
pub const VOXEL_SPAWN_NUDGE_M: f32 = 0.75;

// VOXEL speed (meters per second)
pub const VOXEL_SPEED_MPS: f32 = 35.0;

// Optional: lifetime (seconds)
pub const VOXEL_LIFETIME_S: f32 = 60.0;

pub const MAX_VOXELS: u32 = 1024;

// Voxel-ball (soft cluster) knobs
pub const BALL_RADIUS_VOX: i32 = 3;       // radius in *sub-voxels* (grid steps)
pub const BALL_COMPLIANCE: f32 = 1e-6;    // 0 = rigid; bigger = softer
