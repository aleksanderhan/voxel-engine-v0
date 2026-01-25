use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

/// Material IDs (keep these in sync with your WGSL palette).
pub const AIR: u32 = 0;
pub const GRASS: u32 = 1;
pub const DIRT: u32 = 2;
pub const STONE: u32 = 3;
pub const WOOD: u32 = 4;
pub const LEAF: u32 = 5;

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
    pub fn ground_height(&self, x: i32, z: i32) -> i32 {
        let xf = x as f64;
        let zf = z as f64;

        let h0 = self.height.get([xf, zf]) as f32;
        let h1 = self.detail.get([xf, zf]) as f32;

        let base = 10.0;
        let amp = 18.0;
        let hills = h0 * amp + h1 * 3.0;

        (base + hills).round() as i32
    }

    #[inline]
    fn tree_params(&self, x: i32, z: i32) -> Option<(i32, i32, u32)> {
        let r = hash2(self.seed, x, z);

        // Density (tune): ~1 in 24 columns become trees.
        if (r % 96) != 0 {
            return None;
        }

        let trunk_h = 4 + (hash_u32(r) % 4) as i32; // 4..7
        let crown_r = 2 + (hash_u32(r ^ 0xBEEF) % 2) as i32; // 2..3
        Some((trunk_h, crown_r, r))
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
        if y >= ground - 3 {
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
        // Max crown radius = 3 in our params, so search +/-3.
        for tz in (z - 3)..=(z + 3) {
            for tx in (x - 3)..=(x + 3) {
                let Some((trunk_h, crown_r, _r)) = self.tree_params(tx, tz) else { continue; };

                let ground = height_at(tx, tz);
                let trunk_base_y = ground + 1;
                let trunk_top_y = trunk_base_y + trunk_h;

                // Trunk column at (tx,tz)
                if x == tx && z == tz && y >= trunk_base_y && y <= trunk_top_y {
                    return WOOD;
                }

                // Leaves: sphere-ish around (tx, trunk_top_y, tz)
                let dx = x - tx;
                let dy = y - trunk_top_y;
                let dz = z - tz;
                if dx * dx + dy * dy + dz * dz <= crown_r * crown_r {
                    // Don't overwrite trunk
                    if !(x == tx && z == tz && y >= trunk_base_y && y <= trunk_top_y) {
                        return LEAF;
                    }
                }
            }
        }

        AIR
    }
}
