use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::config;
use super::{
    materials::{AIR, DIRT, GRASS, STONE, WOOD},
    trees::TreeCache,
};

#[derive(Clone)]
pub struct WorldGen {
    pub seed: u32,
    height: Fbm<Perlin>,
    detail: Fbm<Perlin>,
}

impl WorldGen {
    pub fn new(seed: u32) -> Self {
        let height = Fbm::<Perlin>::new(seed).set_octaves(5).set_frequency(0.0025);
        let detail = Fbm::<Perlin>::new(seed ^ 0xA5A5_A5A5).set_octaves(3).set_frequency(0.02);
        Self { seed, height, detail }
    }

    pub fn ground_height(&self, x_vox: i32, z_vox: i32) -> i32 {
        let xm = x_vox as f64 * config::VOXEL_SIZE_M_F64;
        let zm = z_vox as f64 * config::VOXEL_SIZE_M_F64;

        let h0 = self.height.get([xm, zm]) as f32;
        let h1 = self.detail.get([xm, zm]) as f32;

        let base_m = 10.0;
        let amp_m = 18.0;
        let hills_m = h0 * amp_m + h1 * 3.0;

        let voxels_per_meter = (1.0 / config::VOXEL_SIZE_M_F64) as f32;
        ((base_m + hills_m) * voxels_per_meter).round() as i32
    }

    /// Keep your existing signature; builder will use cached height + TreeCache.
    pub fn material_at_world_cached_with_trees<F: Fn(i32, i32) -> i32>(
        &self,
        x: i32,
        y: i32,
        z: i32,
        height_at: &F,
        trees: &TreeCache,
    ) -> u32 {
        let ground = height_at(x, z);
        let voxels_per_meter = (1.0 / config::VOXEL_SIZE_M_F64) as i32;

        if y < ground {
            if y >= ground - 3 * voxels_per_meter {
                return DIRT;
            }
            return STONE;
        }

        if y == ground {
            let tm = self.trees_material_from_cache(x, y, z, height_at, trees);
            if tm == WOOD { return WOOD; }
            return GRASS;
        }

        let tm = self.trees_material_from_cache(x, y, z, height_at, trees);
        if tm != AIR { return tm; }
        AIR
    }
}
