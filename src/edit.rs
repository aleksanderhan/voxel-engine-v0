// src/edit.rs
use std::collections::HashMap;
use std::sync::{RwLock};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::world::materials::AIR;

pub struct VoxelEdits {
    map: RwLock<HashMap<u64, u32>>,
    version: AtomicU64,
}

impl Default for VoxelEdits {
    fn default() -> Self {
        Self {
            map: RwLock::new(HashMap::new()),
            version: AtomicU64::new(1),
        }
    }
}

impl VoxelEdits {
    #[inline]
    fn pack(x: i32, y: i32, z: i32) -> u64 {
        // pack signed i32 into u64 (bias to unsigned)
        let bx = (x as i64 - i32::MIN as i64) as u64;
        let by = (y as i64 - i32::MIN as i64) as u64;
        let bz = (z as i64 - i32::MIN as i64) as u64;
        (bx & 0x1F_FFFF) | ((by & 0x1F_FFFF) << 21) | ((bz & 0x1F_FFFF) << 42)
    }

    #[inline]
    pub fn get(&self, x: i32, y: i32, z: i32) -> Option<u32> {
        let k = Self::pack(x, y, z);
        self.map.read().unwrap().get(&k).copied()
    }

    /// Set a voxel material override. Use AIR to “dig”.
    pub fn set(&self, x: i32, y: i32, z: i32, material: u32) {
        let k = Self::pack(x, y, z);
        self.map.write().unwrap().insert(k, material);
        self.version.fetch_add(1, Ordering::Relaxed);
    }

    pub fn remove(&self, x: i32, y: i32, z: i32) {
        let k = Self::pack(x, y, z);
        self.map.write().unwrap().remove(&k);
        self.version.fetch_add(1, Ordering::Relaxed);
    }

    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }
}
