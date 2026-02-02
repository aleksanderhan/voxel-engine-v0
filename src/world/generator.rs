// src/world/generator.rs

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::app::config;

#[derive(Clone)]
pub struct WorldGen {
    pub seed: u32,
    height: Fbm<Perlin>,
    detail: Fbm<Perlin>,
    voxels_per_meter: f32,
}

impl WorldGen {
    pub fn new(seed: u32) -> Self {
        let height = Fbm::<Perlin>::new(seed).set_octaves(7).set_frequency(0.010);
        let detail = Fbm::<Perlin>::new(seed ^ 0xA5A5_A5A5).set_octaves(3).set_frequency(0.02);

        let voxels_per_meter = (1.0 / config::VOXEL_SIZE_M_F64) as f32;

        Self { seed, height, detail, voxels_per_meter }
    }

    #[inline]
    pub fn ground_height(&self, x_vox: i32, z_vox: i32) -> i32 {
        let xm = (x_vox as f64) * config::VOXEL_SIZE_M_F64;
        let zm = (z_vox as f64) * config::VOXEL_SIZE_M_F64;

        let h0 = self.height.get([xm, zm]) as f32;
        let h1 = self.detail.get([xm, zm]) as f32;

        let base_m = 10.0;
        let amp_m  = 18.0;
        let hills_m = h0 * amp_m + h1 * 3.0;

        ((base_m + hills_m) * self.voxels_per_meter).round() as i32
    }

    pub fn material_at_voxel(&self, wx: i32, wy: i32, wz: i32) -> u32 {
        use crate::world::materials::{AIR, DIRT, GRASS, STONE};

        let g = self.ground_height(wx, wz);

        // match builder’s “dirt_depth = 3 * vpm”
        let dirt_depth = 3 * crate::app::config::VOXELS_PER_METER;

        if wy < g {
            if wy >= g - dirt_depth { DIRT } else { STONE }
        } else if wy == g {
            // If you want tree trunks to override ground at y==g,
            // you can add a trunk/canopy test here and return WOOD.
            GRASS
        } else {
            // Optional: approximate trees here if you want ray hits on leaves/wood.
            // For now: air above ground.
            AIR
        }
    }
}
