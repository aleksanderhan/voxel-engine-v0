



pub const CHUNK_SIZE: u32 = 64;

pub const ACTIVE_RADIUS: i32 = 14;
pub const KEEP_RADIUS: i32 = ACTIVE_RADIUS * 2;

pub const VOXEL_SIZE_M_F32: f32 = 0.10;
pub const VOXEL_SIZE_M_F64: f64 = 0.10;


pub const VOXELS_PER_METER: i32 = 10; 

pub const RENDER_SCALE: f32 = 0.5;

pub const WORKER_THREADS: usize = 4;
pub const MAX_IN_FLIGHT: usize = 16;


pub const NODE_BUDGET_BYTES: usize = 1024 * 1024 * 1024; 



pub const CHUNK_CACHE_BUDGET_BYTES: usize = 1024 * 1024 * 1024; 













pub const CLIPMAP_LEVELS: u32 = 16;
pub const CLIPMAP_LEVELS_USIZE: usize = CLIPMAP_LEVELS as usize;


pub const CLIPMAP_RES: u32 = 256;


pub const CLIPMAP_BASE_CELL_M: f32 = 1.0;



pub const CLIPMAP_MIN_UPDATE_INTERVAL_S: f32 = 0.0;
