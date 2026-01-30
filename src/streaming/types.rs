// src/streaming/types.rs
use std::sync::{Arc, atomic::AtomicBool};

use crate::render::gpu_types::{ChunkMetaGpu, NodeGpu, NodeRopesGpu};

pub const INVALID_U32: u32 = 0xFFFF_FFFF;

// Vertical band dy in [-1..=2]
pub const GRID_Y_MIN_DY: i32 = -1;
pub const GRID_Y_COUNT: u32 = 4;

pub const EVICT_ATTEMPTS: usize = 8;

// 8^3 bits = 512 bits = 16 u32
pub const MACRO_WORDS_PER_CHUNK: u32 = 16;
pub const MACRO_WORDS_PER_CHUNK_USIZE: usize = 16;

// 64x64 columns, packed 2x u16 per u32 => 2048 u32 per chunk
pub const COLINFO_WORDS_PER_CHUNK: u32 = 2048;
pub const COLINFO_WORDS_PER_CHUNK_USIZE: usize = 2048;

pub const MAX_UPLOADS_PER_FRAME: usize = 8;            // start 6–12
pub const MAX_UPLOAD_BYTES_PER_FRAME: usize = 4 << 20; // start 2–8 MB

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub struct ChunkKey {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

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
}

#[derive(Clone, Debug)]
pub struct BuildJob {
    pub key: ChunkKey,
    pub cancel: Arc<AtomicBool>,
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

pub struct ChunkUpload {
    pub key: ChunkKey,
    pub slot: u32,
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
