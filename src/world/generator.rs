// src/world/generator.rs
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::config;
use super::{
    materials::{AIR, DIRT, GRASS, STONE, WOOD},
    trees::TreeCache,
    tunnels::TunnelCache,
};

#[derive(Clone)]
pub struct WorldGen {
    pub seed: u32,
    height: Fbm<Perlin>,
    detail: Fbm<Perlin>,
    cave: Fbm<Perlin>,
    cave_warp: Fbm<Perlin>,
}

impl WorldGen {
    pub fn new(seed: u32) -> Self {
        let height = Fbm::<Perlin>::new(seed).set_octaves(8).set_frequency(0.025);
        let detail = Fbm::<Perlin>::new(seed ^ 0xA5A5_A5A5).set_octaves(3).set_frequency(0.02);
        
        // 3D caves: tuned in meters (not voxels)
        let cave = Fbm::<Perlin>::new(seed ^ 0xC0FF_EE00).set_octaves(3).set_frequency(0.030);
        let cave_warp = Fbm::<Perlin>::new(seed ^ 0xF00D_BAAD).set_octaves(2).set_frequency(0.010);

        Self { seed, height, detail, cave, cave_warp }
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

    pub fn material_at_world_cached_with_features<F: Fn(i32, i32) -> i32>(
        &self,
        x: i32,
        y: i32,
        z: i32,
        height_at: &F,
        trees: &super::trees::TreeCache,
        tunnels: &TunnelCache,
    ) -> u32 {
        let ground = height_at(x, z);
        let vpm = (1.0 / config::VOXEL_SIZE_M_F64) as i32;

        // ---- base terrain + trees (your old logic) ----
        let mut m = if y < ground {
            if y >= ground - 3 * vpm { super::materials::DIRT } else { super::materials::STONE }
        } else if y == ground {
            let tm = self.trees_material_from_cache(x, y, z, height_at, trees);
            if tm == super::materials::WOOD { super::materials::WOOD } else { super::materials::GRASS }
        } else {
            let tm = self.trees_material_from_cache(x, y, z, height_at, trees);
            if tm != super::materials::AIR { tm } else { super::materials::AIR }
        };

        // ---- carve: only bother below ground-ish ----
        let ground = height_at(x, z);
        let vpm = config::VOXELS_PER_METER;

        // carve only in a band below ground
        let carve_y0 = ground - 80 * vpm;   // bottom: 80m
        let carve_y1 = ground - 1 * vpm;    // top: 1m below ground (was ~8m)

        if m != AIR && y >= carve_y0 && y <= carve_y1 {
            if self.cave_density(x, y, z) < 0.0 { return AIR; }
            if tunnels.contains_point(x, y, z) { return AIR; }
        }

        m
    }

    #[inline]
    pub fn cave_density(&self, x_vox: i32, y_vox: i32, z_vox: i32) -> f32 {
        // world coords in meters
        let xm = x_vox as f64 * config::VOXEL_SIZE_M_F64;
        let ym = y_vox as f64 * config::VOXEL_SIZE_M_F64;
        let zm = z_vox as f64 * config::VOXEL_SIZE_M_F64;

        // domain warp (meters)
        let w = self.cave_warp.get([xm, ym, zm]) as f32;     // ~[-1,1]
        let wx = xm + (w as f64) * 12.0;
        let wy = ym + (w as f64) * 8.0;
        let wz = zm + (w as f64) * 12.0;

        let n = self.cave.get([wx, wy, wz]) as f32;          // ~[-1,1]
        // wormy: abs() creates ridges; threshold controls openness
        n.abs() - 0.32
    }
}
