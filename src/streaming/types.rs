// src/streaming/types.rs
use std::sync::{Arc, atomic::AtomicBool};

use crate::world::edits::EditEntry;
use crate::render::gpu_types::{ChunkMetaGpu, NodeGpu, NodeRopesGpu};

pub const INVALID_U32: u32 = 0xFFFF_FFFF;

// Vertical band dy in [-1..=2]
pub const GRID_Y_MIN_DY: i32 = -2;
pub const GRID_Y_COUNT: u32 = 5;

pub const EVICT_ATTEMPTS: usize = 8;

// 8^3 bits = 512 bits = 16 u32
pub const MACRO_WORDS_PER_CHUNK: u32 = 16;
pub const MACRO_WORDS_PER_CHUNK_USIZE: usize = 16;

// 64x64 columns, packed 2x u16 per u32 => 2048 u32 per chunk
pub const COLINFO_WORDS_PER_CHUNK: u32 = 2048;
pub const COLINFO_WORDS_PER_CHUNK_USIZE: usize = 2048;

pub const MAX_UPLOADS_PER_FRAME: usize = 64;
pub const MAX_UPLOAD_BYTES_PER_FRAME: usize = 64 << 20;

pub const PRIORITY_RADIUS: i32 = 2; // => 5x5 in XZ, and with GRID_Y_* => 5 in Y


#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub struct ChunkKey {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[derive(Debug)]
pub enum ChunkState {
    Queued,
    Building,
    Uploading(Uploading),
    Resident(Resident),
}

#[derive(Clone, Debug)]
pub struct Uploading {
    pub slot: u32,
    pub node_base: u32,
    pub node_count: u32,
    pub uploaded: bool,
}

#[derive(Clone, Debug)]
pub struct Resident {
    pub slot: u32,
    pub node_base: u32,
    pub node_count: u32,
    pub rewrite_in_flight: bool,
}

#[derive(Clone, Debug)]
pub struct BuildJob {
    pub key: ChunkKey,
    pub cancel: Arc<AtomicBool>,
    pub edits: Arc<[EditEntry]>,
}

pub struct BuildDone {
    pub key: ChunkKey,
    pub cancel: Arc<AtomicBool>,
    pub canceled: bool,
    pub nodes: Vec<NodeGpu>,
    pub macro_words: Vec<u32>,
    pub ropes: Vec<NodeRopesGpu>,
    pub colinfo_words: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
pub enum UploadKind {
    PromoteToResident, // new chunk becoming resident
    RewriteResident,   // resident chunk being refreshed
}

pub struct ChunkUpload {
    pub key: ChunkKey,
    pub slot: u32,
    pub kind: UploadKind,
    pub meta: ChunkMetaGpu,

    pub node_base: u32,
    pub nodes: Arc<[NodeGpu]>,

    pub macro_words: Arc<[u32]>,

    pub ropes: Arc<[NodeRopesGpu]>,

    pub colinfo_words: Arc<[u32]>,

    pub completes_residency: bool,
}

#[inline(always)]
pub fn y_band_min() -> i32 {
    GRID_Y_MIN_DY
}

#[inline(always)]
pub fn y_band_max() -> i32 {
    GRID_Y_MIN_DY + GRID_Y_COUNT as i32 - 1
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StreamStats {
    pub center: (i32, i32, i32),

    pub resident_slots: u32,
    pub total_slots: u32,
    pub chunks_map: u32,

    pub st_queued: u32,
    pub st_building: u32,
    pub st_uploading: u32,
    pub st_resident: u32,

    pub in_flight: u32,
    pub done_backlog: u32,

    pub up_rewrite: u32,
    pub up_active: u32,
    pub up_other: u32,

    pub cache_bytes: u64,
    pub cache_entries: u32,
    pub cache_lru: u32,

    // NEW
    pub build_queue_len: u32,
    pub queued_set_len: u32,
    pub cancels_len: u32,

    pub orphan_queued: u32,
}
