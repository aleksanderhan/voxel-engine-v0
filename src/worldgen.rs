// worldgen.rs
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

/// Material IDs (keep these in sync with your WGSL palette).
pub const AIR: u32 = 0;
pub const GRASS: u32 = 1;
pub const DIRT: u32 = 2;
pub const STONE: u32 = 3;
pub const WOOD: u32 = 4;
pub const LEAF: u32 = 5;

pub const VOXEL_SIZE_M: f64 = 0.10;          // 10 cm
pub const VOXELS_PER_METER: i32 = 10;        // 1.0 / 0.10


#[inline]
fn hash_u32(mut v: u32) -> u32 {
    v ^= v >> 16;
    v = v.wrapping_mul(0x7feb_352d);
    v ^= v >> 15;
    v = v.wrapping_mul(0x846c_a68b);
    v ^= v >> 16;
    v
}

#[inline]
fn hash2(seed: u32, x: i32, z: i32) -> u32 {
    let a = (x as u32).wrapping_mul(0x9e37_79b1);
    let b = (z as u32).wrapping_mul(0x85eb_ca6b);
    hash_u32(seed ^ a ^ b)
}

pub struct WorldGen {
    pub seed: u32,
    height: Fbm<Perlin>,
    detail: Fbm<Perlin>,
}

impl WorldGen {
    pub fn new(seed: u32) -> Self {
        let height = Fbm::<Perlin>::new(seed)
            .set_octaves(5)
            .set_frequency(0.0025);

        let detail = Fbm::<Perlin>::new(seed ^ 0xA5A5_A5A5)
            .set_octaves(3)
            .set_frequency(0.02);

        Self { seed, height, detail }
    }

    /// Uncached height query (fine for gameplay; chunk builder should cache).
    pub fn ground_height(&self, x_vox: i32, z_vox: i32) -> i32 {
        let xm = x_vox as f64 * VOXEL_SIZE_M;
        let zm = z_vox as f64 * VOXEL_SIZE_M;

        let h0 = self.height.get([xm, zm]) as f32;
        let h1 = self.detail.get([xm, zm]) as f32;

        // These are METERS now:
        let base_m = 10.0;
        let amp_m  = 18.0;
        let hills_m = h0 * amp_m + h1 * 3.0;

        // Convert meters -> voxels:
        ((base_m + hills_m) * (VOXELS_PER_METER as f32)).round() as i32
    }


    #[inline]
    fn tree_params(&self, x_vox: i32, z_vox: i32) -> Option<(i32, i32, u32)> {
        let xm = x_vox.div_euclid(VOXELS_PER_METER);
        let zm = z_vox.div_euclid(VOXELS_PER_METER);

        let r = hash2(self.seed, xm, zm);

        // density per 1m cell
        if (r % 96) != 0 {
            return None;
        }

        // meter-scale dimensions (same “shape” as before)
        let trunk_h_m = 4 + (hash_u32(r) % 4) as i32;        // 4..7 meters
        let crown_r_m = 2 + (hash_u32(r ^ 0xBEEF) % 2) as i32; // 2..3 meters

        // scale to voxels (more detail)
        let trunk_h_vox = trunk_h_m * VOXELS_PER_METER;
        let crown_r_vox = crown_r_m * VOXELS_PER_METER;

        Some((trunk_h_vox, crown_r_vox, r))
    }


    /// Cached version: pass a height function (typically a chunk-local cache).
    pub fn material_at_world_cached<F: Fn(i32, i32) -> i32>(
        &self,
        x: i32,
        y: i32,
        z: i32,
        height_at: &F,
    ) -> u32 {
        // Trees first so they occupy air above ground.
        let tm = self.trees_material_at_cached(x, y, z, height_at);
        if tm != AIR {
            return tm;
        }

        let ground = height_at(x, z);

        if y > ground {
            return AIR;
        }

        if y == ground {
            return GRASS;
        }
        if y >= ground - 3 * VOXELS_PER_METER {
            return DIRT;
        }

        STONE
    }

    /// Returns WOOD/LEAF/AIR. Uses neighborhood search so leaves extend across columns.
    pub fn trees_material_at_cached<F: Fn(i32, i32) -> i32>(
        &self,
        x: i32,
        y: i32,
        z: i32,
        height_at: &F,
    ) -> u32 {
        let xm = x.div_euclid(VOXELS_PER_METER);
        let zm = z.div_euclid(VOXELS_PER_METER);

        // max crown radius is 3m => search +/-3 meter cells
        for tz_m in (zm - 3)..=(zm + 3) {
            for tx_m in (xm - 3)..=(xm + 3) {
                let tx = tx_m * VOXELS_PER_METER;
                let tz = tz_m * VOXELS_PER_METER;

                let Some((trunk_h, crown_r, _r)) = self.tree_params(tx, tz) else { continue; };

                let ground = height_at(tx, tz);
                let trunk_base_y = ground + VOXELS_PER_METER; // +1m above ground (matches old “+1 voxel” intent)
                let trunk_top_y = trunk_base_y + trunk_h;

                if x == tx && z == tz && y >= trunk_base_y && y <= trunk_top_y {
                    return WOOD;
                }

                let dx = x - tx;
                let dy = y - trunk_top_y;
                let dz = z - tz;
                if dx*dx + dy*dy + dz*dz <= crown_r*crown_r {
                    if !(x == tx && z == tz && y >= trunk_base_y && y <= trunk_top_y) {
                        return LEAF;
                    }
                }
            }
        }

        AIR
    }

    pub fn tree_instance_at_meter(&self, xm: i32, zm: i32) -> Option<(i32, i32)> {
        let r = hash2(self.seed, xm, zm);
        if (r % 96) != 0 { return None; }

        let trunk_h_m = 4 + (hash_u32(r) % 4) as i32;          // 4..7 m
        let crown_r_m = 2 + (hash_u32(r ^ 0xBEEF) % 2) as i32; // 2..3 m

        let vpm = VOXELS_PER_METER;
        Some((trunk_h_m * vpm, crown_r_m * vpm))
    }

}
