






use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use rustc_hash::FxHashMap as HashMap;

use crate::app::config;
use crate::streaming::types::ChunkKey;
use crate::world::WorldGen;

#[inline(always)]
pub fn idx_xyz(lx: i32, ly: i32, lz: i32, cs: i32) -> u32 {
    
    
    (lx as u32) + (ly as u32) * (cs as u32) + (lz as u32) * (cs as u32) * (cs as u32)
}

#[inline]
pub fn voxel_to_chunk_local(
    _world: &WorldGen,
    wx: i32,
    wy: i32,
    wz: i32,
) -> (ChunkKey, i32, i32, i32) {
    let cs = config::CHUNK_SIZE as i32;

    
    let cx = wx.div_euclid(cs);
    let cy = wy.div_euclid(cs);
    let cz = wz.div_euclid(cs);

    
    let lx = wx.rem_euclid(cs);
    let ly = wy.rem_euclid(cs);
    let lz = wz.rem_euclid(cs);

    (ChunkKey { x: cx, y: cy, z: cz }, lx, ly, lz)
}


#[derive(Clone, Copy, Debug)]
pub struct EditEntry {
    
    
    pub idx: u32,
    pub mat: u32,
}

#[derive(Default)]
struct ChunkEdits {
    
    list: Vec<EditEntry>,
    
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

    
    
    pub fn apply_voxel(&self, key: ChunkKey, lx: i32, ly: i32, lz: i32, mat: u32) {
        let Some(idx) = Self::local_idx(lx, ly, lz) else { return; };

        let mut map = self.map_write();
        let chunk = map.entry(key).or_insert_with(ChunkEdits::new);
        chunk.set_idx(idx, mat);
    }

    
    pub fn get_override(&self, key: ChunkKey, lx: i32, ly: i32, lz: i32) -> Option<u32> {
        let idx = Self::local_idx(lx, ly, lz)?;
        let map = self.map_read();
        let chunk = map.get(&key)?;
        chunk.get_idx(idx)
    }

    
    
    pub fn snapshot(&self, key: ChunkKey) -> Arc<[EditEntry]> {
        let mut map = self.map_write();
        match map.get_mut(&key) {
            Some(chunk) if !chunk.is_empty() => chunk.snapshot(),
            _ => Arc::from([]),
        }
    }

}