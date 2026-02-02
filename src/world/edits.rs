// src/world/edits.rs
// ------------------
//
// Stores per-voxel edits as CHUNK-LOCAL linear indices (0..CHUNK_SIZE^3-1).
// This matches builder.rs usage:
//   for e in edits { scratch.material[e.idx as usize] = e.mat; }

use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use rustc_hash::FxHashMap as HashMap;

use crate::app::config;
use crate::streaming::types::ChunkKey;

#[derive(Clone, Copy, Debug)]
pub struct EditEntry {
    /// Chunk-local linear index:
    /// idx = y*CHUNK_SIZE*CHUNK_SIZE + z*CHUNK_SIZE + x
    pub idx: u32,
    pub mat: u32,
}

#[derive(Default)]
struct ChunkEdits {
    // Always kept sorted by idx.
    list: Vec<EditEntry>,
    // Frozen snapshot for worker threads.
    frozen: Arc<[EditEntry]>,
    dirty: bool,
}

impl ChunkEdits {
    fn new() -> Self {
        Self {
            list: Vec::new(),
            frozen: Arc::from([]),
            dirty: false,
        }
    }

    #[inline]
    fn set_idx(&mut self, idx: u32, mat: u32) {
        match self.list.binary_search_by_key(&idx, |e| e.idx) {
            Ok(pos) => {
                if self.list[pos].mat != mat {
                    self.list[pos].mat = mat;
                    self.dirty = true;
                }
            }
            Err(pos) => {
                self.list.insert(pos, EditEntry { idx, mat });
                self.dirty = true;
            }
        }
    }

    #[inline]
    fn get_idx(&self, idx: u32) -> Option<u32> {
        match self.list.binary_search_by_key(&idx, |e| e.idx) {
            Ok(pos) => Some(self.list[pos].mat),
            Err(_) => None,
        }
    }

    #[allow(dead_code)]
    fn remove_idx(&mut self, idx: u32) {
        if let Ok(pos) = self.list.binary_search_by_key(&idx, |e| e.idx) {
            self.list.remove(pos);
            self.dirty = true;
        }
    }

    #[allow(dead_code)]
    fn clear(&mut self) {
        if !self.list.is_empty() {
            self.list.clear();
            self.dirty = true;
        }
    }

    fn snapshot(&mut self) -> Arc<[EditEntry]> {
        if self.dirty {
            // list is already sorted; just freeze.
            self.frozen = Arc::from(self.list.clone().into_boxed_slice());
            self.dirty = false;
        }
        self.frozen.clone()
    }

    fn is_empty(&self) -> bool {
        self.list.is_empty()
    }
}

pub struct EditStore {
    // Sparse per-chunk overrides.
    // Per-chunk list kept sorted by idx so lookups are fast.
    map: RwLock<HashMap<ChunkKey, ChunkEdits>>,
}

impl EditStore {
    pub fn new() -> Self {
        Self {
            map: RwLock::new(HashMap::default()),
        }
    }

    #[inline]
    fn map_read(&self) -> RwLockReadGuard<'_, HashMap<ChunkKey, ChunkEdits>> {
        self.map.read().unwrap_or_else(|e| e.into_inner())
    }

    #[inline]
    fn map_write(&self) -> RwLockWriteGuard<'_, HashMap<ChunkKey, ChunkEdits>> {
        self.map.write().unwrap_or_else(|e| e.into_inner())
    }

    #[inline]
    fn local_idx(lx: i32, ly: i32, lz: i32) -> Option<u32> {
        let cs = config::CHUNK_SIZE as i32;
        if lx < 0 || ly < 0 || lz < 0 || lx >= cs || ly >= cs || lz >= cs {
            return None;
        }

        let cs_u = config::CHUNK_SIZE as u32;
        let x = lx as u32;
        let y = ly as u32;
        let z = lz as u32;

        Some(y * cs_u * cs_u + z * cs_u + x)
    }

    /// Set/overwrite a single voxel override inside a chunk.
    /// Stores AIR too (so “dig” is a real override).
    pub fn apply_voxel(&self, key: ChunkKey, lx: i32, ly: i32, lz: i32, mat: u32) {
        let Some(idx) = Self::local_idx(lx, ly, lz) else { return; };

        let mut map = self.map_write();
        let chunk = map.entry(key).or_insert_with(ChunkEdits::new);
        chunk.set_idx(idx, mat);
    }

    /// Returns Some(mat) if there is an override for that voxel.
    pub fn get_override(&self, key: ChunkKey, lx: i32, ly: i32, lz: i32) -> Option<u32> {
        let idx = Self::local_idx(lx, ly, lz)?;
        let map = self.map_read();
        let chunk = map.get(&key)?;
        chunk.get_idx(idx)
    }

    /// Snapshot all edits for a chunk to send to a worker thread.
    /// (Your build code already calls this.)
    pub fn snapshot(&self, key: ChunkKey) -> Arc<[EditEntry]> {
        let mut map = self.map_write();
        match map.get_mut(&key) {
            Some(chunk) if !chunk.is_empty() => chunk.snapshot(),
            _ => Arc::from([]),
        }
    }

    /// Optional: clear all overrides in a chunk (handy for debugging).
    pub fn clear_chunk(&self, key: ChunkKey) {
        let mut map = self.map_write();
        map.remove(&key);
    }
}

/// Convert WORLD VOXEL coords -> (ChunkKey, chunk-local linear idx).
///
/// IMPORTANT:
/// - `div_euclid` + `rem_euclid` handle negative coordinates correctly.
/// - idx layout matches builder.rs: idx3(side, x, y, z) => (y*side*side) + (z*side) + x
#[inline]
pub fn chunk_key_and_local_idx_from_world_vox(wx: i32, wy: i32, wz: i32) -> (ChunkKey, u32) {
    let cs: i32 = config::CHUNK_SIZE as i32; // e.g. 64
    let cs_u: u32 = config::CHUNK_SIZE as u32;

    // Chunk coordinates in chunk units
    let ck = ChunkKey {
        x: wx.div_euclid(cs),
        y: wy.div_euclid(cs),
        z: wz.div_euclid(cs),
    };

    // Local voxel coordinates in [0..cs)
    let lx = wx.rem_euclid(cs) as u32;
    let ly = wy.rem_euclid(cs) as u32;
    let lz = wz.rem_euclid(cs) as u32;

    debug_assert!(lx < cs_u && ly < cs_u && lz < cs_u);

    let idx = ly * cs_u * cs_u + lz * cs_u + lx;

    debug_assert!(idx < cs_u * cs_u * cs_u);

    (ck, idx)
}
