// src/svo/height_cache.rs
use crate::world::WorldGen;

/// Simple 2D height cache over an (x,z) rectangle in world-voxel coords.
/// Stores heights from gen.ground_height(x,z) to avoid recomputing.
pub struct HeightCache {
    ox: i32,
    oz: i32,
    w: i32,
    h: i32,
    data: Vec<i32>,
}

impl HeightCache {
    /// Empty cache (valid but has no data). Useful if you want to skip caching.
    pub fn new_empty() -> Self {
        Self {
            ox: 0,
            oz: 0,
            w: 0,
            h: 0,
            data: Vec::new(),
        }
    }

    /// Build and fill a cache for [wx0, wx1) × [wz0, wz1) in world-voxel coords.
    pub fn new(gen: &WorldGen, wx0: i32, wz0: i32, wx1: i32, wz1: i32) -> Self {
        let w = wx1 - wx0;
        let h = wz1 - wz0;
        debug_assert!(w > 0 && h > 0);

        let mut data = vec![0i32; (w * h) as usize];

        for z in 0..h {
            let wz = wz0 + z;
            for x in 0..w {
                let wx = wx0 + x;
                data[(z * w + x) as usize] = gen.ground_height(wx, wz);
            }
        }

        Self {
            ox: wx0,
            oz: wz0,
            w,
            h,
            data,
        }
    }

    #[inline]
    pub fn contains(&self, x: i32, z: i32) -> bool {
        let ix = x - self.ox;
        let iz = z - self.oz;
        ix >= 0 && ix < self.w && iz >= 0 && iz < self.h
    }

    #[inline]
    pub fn get(&self, x: i32, z: i32) -> i32 {
        let ix = x - self.ox;
        let iz = z - self.oz;
        debug_assert!(ix >= 0 && ix < self.w);
        debug_assert!(iz >= 0 && iz < self.h);
        self.data[(iz * self.w + ix) as usize]
    }
}
