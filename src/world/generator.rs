// src/world/generator.rs

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::app::config;
use crate::world::materials::{AIR, DIRT, GRASS, STONE, WOOD, LEAF};
use crate::world::hash::{hash2, hash_u32, u01};
use crate::world::edits::{EditStore, voxel_to_chunk_local};


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
        // --- terrain ---
        let g = self.ground_height(wx, wz);

        // match builder’s “dirt_depth = 3 * vpm”
        let vpm = config::VOXELS_PER_METER;
        let dirt_depth = 3 * vpm;

        let mut m = if wy < g {
            if wy >= g - dirt_depth { DIRT } else { STONE }
        } else if wy == g {
            GRASS
        } else {
            AIR
        };

        // --- trees overlay (so rays can hit them) ---
        if m == AIR {
            if let Some(tm) = self.tree_material_at_voxel(wx, wy, wz) {
                m = tm;
            }
        }

        m
    }

    /// Query world material with edits taking priority (edits can “delete” tree voxels).
    /// This is the one you want for picking / interaction.
    pub fn material_at_voxel_with_edits(
        &self,
        edits: &EditStore,
        wx: i32,
        wy: i32,
        wz: i32,
    ) -> u32 {
        // 1) edits override everything
        let (ck, lx, ly, lz) = voxel_to_chunk_local(&self, wx, wy, wz);
        if let Some(mat) = edits.get_override(ck, lx, ly, lz) {
            return mat;
        }

        // 2) otherwise query generated world (terrain + trees)
        self.material_at_voxel(wx, wy, wz)
    }

    /// Returns Some(WOOD/LEAF) if (wx,wy,wz) is inside an approximate tree volume.
    #[inline]
    fn tree_material_at_voxel(&self, wx: i32, wy: i32, wz: i32) -> Option<u32> {
        let vpm = config::VOXELS_PER_METER;

        // Your trees are placed on a meter grid cell. Reconstruct trunk position.
        let xm = wx.div_euclid(vpm);
        let zm = wz.div_euclid(vpm);

        // Match trees.rs placement: hash2(seed, xm, zm) and TREE_CELL_MOD gating.
        // NOTE: keep TREE_CELL_MOD in sync with trees.rs (currently 256).
        const TREE_CELL_MOD: u32 = 256;

        let seed = hash2(self.seed, xm, zm);
        if (seed % TREE_CELL_MOD) != 0 {
            return None;
        }

        // Trunk position in voxel coords:
        let tx = xm * vpm;
        let tz = zm * vpm;

        // Ground at trunk
        let ground = self.ground_height(tx, tz);

        // Match trees.rs: trunk base is ground + vpm
        let base_y = ground + vpm;

        // Match trees.rs: trunk_h_m = 5..10m, crown_r_m = 3..6m
        let trunk_h_m = 5 + (hash_u32(seed) % 6) as i32;
        let crown_r_m = 3 + (hash_u32(seed ^ 0xBEEF) % 4) as i32;

        let trunk_h = trunk_h_m * vpm;
        let crown_r = crown_r_m * vpm;

        let top_y = base_y + trunk_h;

        // Outside tree vertical span (with a bit of canopy headroom)
        if wy < base_y - 2 || wy > top_y + 8 * vpm {
            return None;
        }

        // Approx trunk radius (match trees.rs)
        let vpm_f = vpm as f32;
        let r0 = 0.45 * vpm_f + u01(hash_u32(seed ^ 0x1111)) * (0.30 * vpm_f);
        let r1 = 0.18 * vpm_f + u01(hash_u32(seed ^ 0x2222)) * (0.12 * vpm_f);

        // Linear taper along trunk
        if wy >= base_y && wy <= top_y {
            let t = ((wy - base_y) as f32 / (trunk_h as f32)).clamp(0.0, 1.0);
            let rr = (r0 + (r1 - r0) * t).max(1.0);

            let dx = (wx - tx) as f32;
            let dz = (wz - tz) as f32;

            if dx * dx + dz * dz <= rr * rr {
                return Some(WOOD);
            }
        }

        // Approx canopy: a soft sphere-ish region around trunk tip.
        // This is intentionally generous so leaves become “hittable”.
        let cy = top_y + 5 * vpm; // similar to builder’s canopy_h_vox
        let dx = (wx - tx) as f32;
        let dy = (wy - cy) as f32;
        let dz = (wz - tz) as f32;

        let cr = (crown_r + 2 * vpm) as f32;
        if dx * dx + (dy * 0.85) * (dy * 0.85) + dz * dz <= cr * cr {
            return Some(LEAF);
        }

        None
    }
}
