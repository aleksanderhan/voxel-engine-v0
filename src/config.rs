// src/config.rs
pub const CHUNK_SIZE: u32 = 128;

pub const ACTIVE_RADIUS: i32 = 3;
pub const KEEP_RADIUS: i32 = ACTIVE_RADIUS + 6;

pub const VOXEL_SIZE_M_F32: f32 = 0.10;
pub const VOXEL_SIZE_M_F64: f64 = 0.10;

// Keep this explicit so you can change voxel size later without hunting constants.
pub const VOXELS_PER_METER: i32 = 10; // 1.0 / 0.10

pub const WORKER_THREADS: usize = 4;
pub const MAX_IN_FLIGHT: usize = 8;

pub const NODE_BUDGET_BYTES: usize = 1024 * 1024 * 1024; // 1 GB
