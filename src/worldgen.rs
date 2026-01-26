// worldgen.rs
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

/// Material IDs (keep these in sync with your WGSL palette).
pub const AIR: u32 = 0;
pub const GRASS: u32 = 1;
pub const DIRT: u32 = 2;
pub const STONE: u32 = 3;
pub const WOOD: u32 = 4;
pub const LEAF: u32 = 5;

pub const VOXEL_SIZE_M: f64 = 0.10; // 10 cm
pub const VOXELS_PER_METER: i32 = 10; // 1.0 / 0.10

// -----------------------------------------------------------------------------
// Hash helpers
// -----------------------------------------------------------------------------

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

#[inline]
fn hash3(seed: u32, x: i32, y: i32, z: i32) -> u32 {
    let a = (x as u32).wrapping_mul(0x9e37_79b1);
    let b = (y as u32).wrapping_mul(0x85eb_ca6b);
    let c = (z as u32).wrapping_mul(0xc2b2_ae35);
    hash_u32(seed ^ a ^ b ^ c)
}

#[inline]
fn u01(v: u32) -> f32 {
    (v as f32) * (1.0 / 4294967296.0) // [0,1)
}

// -----------------------------------------------------------------------------
// Trees
// -----------------------------------------------------------------------------

/// Procedural tree instance (voxel units for positions/sizes).
#[derive(Clone, Copy)]
struct Tree {
    tx: i32,
    tz: i32,
    base_y: i32,   // touches ground
    trunk_h: i32,  // voxels
    crown_r: i32,  // voxels (horizontal extent)
    canopy_h: i32, // voxels (vertical extent)
    trunk_r0: f32, // voxels radius at base
    trunk_r1: f32, // voxels radius near top
    seed: u32,
}

/// Per-chunk cache of trees that can affect material queries.
/// Speed path: per-voxel checks loop over a few cached trees.
pub struct TreeCache {
    trees: Vec<Tree>,
}

impl TreeCache {
    pub fn empty() -> Self {
        Self { trees: Vec::new() }
    }
}

// -----------------------------------------------------------------------------
// WorldGen
// -----------------------------------------------------------------------------

pub struct WorldGen {
    pub seed: u32,
    height: Fbm<Perlin>,
    detail: Fbm<Perlin>,
}

impl WorldGen {
    pub fn new(seed: u32) -> Self {
        let height = Fbm::<Perlin>::new(seed).set_octaves(5).set_frequency(0.0025);
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

        // meters
        let base_m = 10.0;
        let amp_m = 18.0;
        let hills_m = h0 * amp_m + h1 * 3.0;

        // meters -> voxels
        ((base_m + hills_m) * (VOXELS_PER_METER as f32)).round() as i32
    }

    // -------------------------------------------------------------------------
    // Tree placement + params (1m grid anchors, voxel-detail geometry)
    // -------------------------------------------------------------------------

    #[inline]
    fn tree_at_meter_cell(
        &self,
        xm: i32,
        zm: i32,
    ) -> Option<(u32 /*seed*/, i32 /*trunk_h_vox*/, i32 /*crown_r_vox*/)> {
        let r = hash2(self.seed, xm, zm);

        // density per 1m cell
        if (r % 256) != 0 {
            return None;
        }

        // trunk height in meters: 5..10m
        let trunk_h_m = 5 + (hash_u32(r) % 6) as i32;

        // crown radius in meters: 2..4m
        let crown_r_m = 2 + (hash_u32(r ^ 0xBEEF) % 3) as i32;

        let vpm = VOXELS_PER_METER;
        Some((r, trunk_h_m * vpm, crown_r_m * vpm))
    }

    #[inline]
    fn tree_params_at_trunk(&self, tx: i32, tz: i32, ground: i32) -> Option<Tree> {
        let xm = tx.div_euclid(VOXELS_PER_METER);
        let zm = tz.div_euclid(VOXELS_PER_METER);
        let (seed, trunk_h, crown_r) = self.tree_at_meter_cell(xm, zm)?;

        let vpm = VOXELS_PER_METER as f32;

        // trunk radii (voxels): base ~0.45..0.75m, top ~0.18..0.30m
        let r0 = 0.45 * vpm + u01(hash_u32(seed ^ 0x1111)) * (0.30 * vpm);
        let r1 = 0.18 * vpm + u01(hash_u32(seed ^ 0x2222)) * (0.12 * vpm);

        // canopy height: ~2.5..4.5m
        let canopy_h = ((2.5 * vpm) + u01(hash_u32(seed ^ 0x3333)) * (2.0 * vpm)).round() as i32;

        Some(Tree {
            tx,
            tz,
            base_y: ground, // touches ground
            trunk_h,
            crown_r,
            canopy_h,
            trunk_r0: r0.max(1.0),
            trunk_r1: r1.max(1.0),
            seed,
        })
    }

    /// Build a per-chunk tree cache. Call once per chunk build (in `svo.rs`).
    pub fn build_tree_cache<F: Fn(i32, i32) -> i32>(
        &self,
        chunk_ox: i32,
        chunk_oz: i32,
        chunk_size: i32,
        height_at: &F,
    ) -> TreeCache {
        let vpm = VOXELS_PER_METER;

        // crown up to ~4m + slack (branches / wobble / canopy clumps)
        let pad_m = 6;

        let xm0 = chunk_ox.div_euclid(vpm) - pad_m;
        let xm1 = (chunk_ox + chunk_size).div_euclid(vpm) + pad_m;
        let zm0 = chunk_oz.div_euclid(vpm) - pad_m;
        let zm1 = (chunk_oz + chunk_size).div_euclid(vpm) + pad_m;

        let mut trees = Vec::new();

        for zm in zm0..=zm1 {
            for xm in xm0..=xm1 {
                let Some((_seed, _th, _cr)) = self.tree_at_meter_cell(xm, zm) else { continue; };

                let tx = xm * vpm;
                let tz = zm * vpm;
                let ground = height_at(tx, tz);

                let Some(t) = self.tree_params_at_trunk(tx, tz, ground) else { continue; };

                // Conservative XZ AABB reject vs chunk footprint
                let r = t.crown_r + 2 * vpm;
                let x0 = t.tx - r;
                let x1 = t.tx + r;
                let z0 = t.tz - r;
                let z1 = t.tz + r;

                let cx0 = chunk_ox;
                let cx1 = chunk_ox + chunk_size;
                let cz0 = chunk_oz;
                let cz1 = chunk_oz + chunk_size;

                if x1 < cx0 || x0 > cx1 || z1 < cz0 || z0 > cz1 {
                    continue;
                }

                trees.push(t);
            }
        }

        TreeCache { trees }
    }

    // -------------------------------------------------------------------------
    // Terrain + trees (cached) materials
    // -------------------------------------------------------------------------

    /// Use this in the SVO builder: avoids per-voxel neighborhood scans.
    pub fn material_at_world_cached_with_trees<F: Fn(i32, i32) -> i32>(
        &self,
        x: i32,
        y: i32,
        z: i32,
        height_at: &F,
        trees: &TreeCache,
    ) -> u32 {
        let ground = height_at(x, z);

        if y < ground {
            if y >= ground - 3 * VOXELS_PER_METER {
                return DIRT;
            }
            return STONE;
        }

        if y == ground {
            // allow trunk base to replace grass
            let tm = self.trees_material_from_cache(x, y, z, height_at, trees);
            if tm == WOOD {
                return WOOD;
            }
            return GRASS;
        }

        let tm = self.trees_material_from_cache(x, y, z, height_at, trees);
        if tm != AIR {
            return tm;
        }
        AIR
    }

    // -------------------------------------------------------------------------
    // Organic tree geometry: trunk + branches + sparse canopy (and sparse branch tufts)
    // -------------------------------------------------------------------------

    #[inline]
    fn trunk_radius_at(tree: &Tree, y: i32) -> f32 {
        let t = ((y - tree.base_y) as f32 / (tree.trunk_h as f32)).clamp(0.0, 1.0);
        tree.trunk_r0 + (tree.trunk_r1 - tree.trunk_r0) * t
    }

    #[inline]
    fn trunk_wobble(&self, tree: &Tree, y: i32) -> (f32, f32) {
        let t = ((y - tree.base_y) as f32 / (tree.trunk_h as f32)).clamp(0.0, 1.0);

        let a = u01(hash_u32(tree.seed ^ 0xA001 ^ (y as u32).wrapping_mul(97)));
        let b = u01(hash_u32(tree.seed ^ 0xA002 ^ (y as u32).wrapping_mul(193)));

        // max ~0.25m drift near the top
        let amp = 0.25 * (VOXELS_PER_METER as f32) * t;
        ((a - 0.5) * 2.0 * amp, (b - 0.5) * 2.0 * amp)
    }

    #[inline]
    fn dist2_point_segment(
        px: f32,
        py: f32,
        pz: f32,
        ax: f32,
        ay: f32,
        az: f32,
        bx: f32,
        by: f32,
        bz: f32,
    ) -> (f32 /*d2*/, f32 /*t*/) {
        let abx = bx - ax;
        let aby = by - ay;
        let abz = bz - az;

        let apx = px - ax;
        let apy = py - ay;
        let apz = pz - az;

        let ab2 = abx * abx + aby * aby + abz * abz;
        if ab2 <= 1e-8 {
            return (apx * apx + apy * apy + apz * apz, 0.0);
        }

        let t = ((apx * abx + apy * aby + apz * abz) / ab2).clamp(0.0, 1.0);
        let cx = ax + t * abx;
        let cy = ay + t * aby;
        let cz = az + t * abz;

        let dx = px - cx;
        let dy = py - cy;
        let dz = pz - cz;

        (dx * dx + dy * dy + dz * dz, t)
    }

    #[inline]
    fn is_branch_wood(&self, tree: &Tree, x: i32, y: i32, z: i32) -> bool {
        let px = x as f32;
        let py = y as f32;
        let pz = z as f32;

        let top_y = tree.base_y + tree.trunk_h;
        let starts = [
            tree.base_y + (tree.trunk_h * 5) / 10,
            tree.base_y + (tree.trunk_h * 6) / 10,
            tree.base_y + (tree.trunk_h * 7) / 10,
            tree.base_y + (tree.trunk_h * 8) / 10,
        ];

        for (i, sy) in starts.iter().copied().enumerate() {
            if y < sy - 2 || y > top_y + tree.canopy_h {
                continue;
            }

            let r = hash_u32(tree.seed ^ 0xB000 ^ (i as u32).wrapping_mul(0x9E37_79B9));
            let ang = u01(r) * std::f32::consts::TAU;

            let len = (1.8 * VOXELS_PER_METER as f32)
                + u01(hash_u32(r ^ 0x1111)) * (1.7 * VOXELS_PER_METER as f32);
            let pitch = 0.15 + u01(hash_u32(r ^ 0x2222)) * 0.30;

            let (wx, wz) = self.trunk_wobble(tree, sy);
            let ax = tree.tx as f32 + wx;
            let ay = sy as f32;
            let az = tree.tz as f32 + wz;

            let bx = ax + ang.cos() * len;
            let by = ay + pitch * len;
            let bz = az + ang.sin() * len;

            let br0 = (0.12 * VOXELS_PER_METER as f32)
                + u01(hash_u32(r ^ 0x3333)) * (0.10 * VOXELS_PER_METER as f32);
            let br1 = br0 * 0.45;

            let (d2, t) = Self::dist2_point_segment(px, py, pz, ax, ay, az, bx, by, bz);
            let br = br0 + (br1 - br0) * t;

            if d2 <= br * br {
                return true;
            }
        }

        false
    }

    #[inline]
    fn is_branch_leaf(&self, tree: &Tree, x: i32, y: i32, z: i32) -> bool {
        let px = x as f32;
        let py = y as f32;
        let pz = z as f32;

        let vpm_f = VOXELS_PER_METER as f32;

        // 4 branches in upper trunk
        let starts = [
            tree.base_y + (tree.trunk_h * 5) / 10,
            tree.base_y + (tree.trunk_h * 6) / 10,
            tree.base_y + (tree.trunk_h * 7) / 10,
            tree.base_y + (tree.trunk_h * 8) / 10,
        ];

        // Coarse cell size for clump noise (~0.4m)
        let cell: i32 = ((0.4 * vpm_f).round() as i32).max(2);

        for (i, sy) in starts.iter().copied().enumerate() {
            // Per-branch leaf presence (rarely a naked branch)
            let br_seed = hash_u32(tree.seed ^ 0xB000 ^ (i as u32).wrapping_mul(0x9E37_79B9));
            if (hash_u32(br_seed ^ 0x55AA_1234) & 31) == 0 {
                continue; // ~1/32 branches have no leaves
            }

            let ang = u01(br_seed) * std::f32::consts::TAU;

            // length 1.8..3.5m, pitch 0.15..0.45
            let len = (1.8 * vpm_f) + u01(hash_u32(br_seed ^ 0x1111)) * (1.7 * vpm_f);
            let pitch = 0.15 + u01(hash_u32(br_seed ^ 0x2222)) * 0.30;

            // start at trunk center (with wobble)
            let (wx, wz) = self.trunk_wobble(tree, sy);
            let ax = tree.tx as f32 + wx;
            let ay = sy as f32;
            let az = tree.tz as f32 + wz;

            let bx = ax + ang.cos() * len;
            let by = ay + pitch * len;
            let bz = az + ang.sin() * len;

            // Distance to branch centerline + parameter t along branch
            let (d2, t) = Self::dist2_point_segment(px, py, pz, ax, ay, az, bx, by, bz);

            // Leaves mainly on the outer half of the branch (prevents trunk-area fuzz)
            if t < 0.55 {
                continue;
            }

            // --- Guaranteed attachment sleeve near the tip (keeps "connected canopy") ---
            // No randomness here, so every branch can visibly have leaves.
            if t > 0.78 {
                let sleeve_r = (0.20 * vpm_f).max(1.0);
                if d2 <= sleeve_r * sleeve_r {
                    return true;
                }
            }

            // Smaller tufts: ~0.55..0.90m
            let tuft_r = (0.55 * vpm_f) + u01(hash_u32(br_seed ^ 0x4444)) * (0.35 * vpm_f);

            // Two clump centers: tip + outer-mid
            let centers = [
                (bx, by, bz, tuft_r),
                (
                    ax + 0.72 * (bx - ax),
                    ay + 0.72 * (by - ay),
                    az + 0.72 * (bz - az),
                    tuft_r * 0.85,
                ),
            ];

            for (cx, cy, cz, r) in centers {
                // Ellipsoid-ish (flatter vertically)
                let dx = px - cx;
                let dy = (py - cy) * 1.25;
                let dz = pz - cz;

                let dd2 = dx * dx + dy * dy + dz * dz;
                if dd2 > r * r {
                    continue;
                }

                // Coarse clump noise (connected pockets)
                let gx = (x).div_euclid(cell);
                let gy = (y).div_euclid(cell);
                let gz = (z).div_euclid(cell);
                let n = hash3(self.seed ^ br_seed ^ 0x1EE7_1EAF, gx, gy, gz);

                // Hollow-ish clumps: keep mostly shell
                let nd = (dd2.sqrt() / r).clamp(0.0, 1.0);
                if nd < 0.72 {
                    if (n & 63) != 0 {
                        continue; // keep ~1/64 interior cells
                    }
                } else {
                    if (n & 7) == 0 {
                        continue; // drop 1/8 of shell cells
                    }
                }

                // Final: stay near branch line so tufts don't float
                let max_line_r = (0.40 * vpm_f).max(1.0);
                if d2 <= max_line_r * max_line_r {
                    return true;
                }
            }
        }

        false
    }



    #[inline]
    fn is_canopy_leaf(&self, tree: &Tree, x: i32, y: i32, z: i32) -> bool {
        // Sparser, thinner canopy:
        // - tighter y-range
        // - smaller radius
        // - budget gate (1/4)
        // - only keep outer shell (nd >= 0.80) except tiny interior fraction
        // - extra porosity on shell
        let vpm = VOXELS_PER_METER;
        let top_y = tree.base_y + tree.trunk_h;

        let canopy_y0 = top_y - (vpm / 3);
        let canopy_y1 = top_y + (tree.canopy_h * 3) / 4;
        if y < canopy_y0 || y > canopy_y1 {
            return false;
        }

        let (wx, wz) = self.trunk_wobble(tree, top_y);
        let cx0 = tree.tx as f32 + wx;
        let cz0 = tree.tz as f32 + wz;
        let cy0 = top_y as f32 + 0.30 * (tree.canopy_h as f32);

        let base_r = (tree.crown_r as f32) * 0.90;

        let px = x as f32;
        let pz = z as f32;
        let dx0 = px - cx0;
        let dz0 = pz - cz0;

        let max_r = base_r * 1.15;
        if dx0 * dx0 + dz0 * dz0 > max_r * max_r {
            return false;
        }

        // Budget gate (canopy)
        let gate = hash3(self.seed ^ 0xA11C_E5ED, x, y, z) & 3; // keep 1/4
        if gate != 0 {
            return false;
        }

        let py = y as f32;
        let mut best_nd: f32 = 2.0;
        let mut hit = false;

        // fewer blobs (4)
        for i in 0..4u32 {
            let rr = hash_u32(tree.seed ^ 0xC100 ^ i.wrapping_mul(0x9E37_79B9));
            let ang = u01(rr) * std::f32::consts::TAU;

            let rad = u01(hash_u32(rr ^ 0x10)) * (0.60 * base_r);
            let cx = cx0 + ang.cos() * rad;
            let cz = cz0 + ang.sin() * rad;
            let cy = cy0 + (u01(hash_u32(rr ^ 0x20)) - 0.5) * (0.50 * tree.canopy_h as f32);

            let cr = (0.48 + 0.28 * u01(hash_u32(rr ^ 0x30))) * base_r;

            let dx = px - cx;
            let dy = (py - cy) * 1.45;
            let dz = pz - cz;

            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 <= cr * cr {
                hit = true;
                let nd = d2.sqrt() / cr;
                if nd < best_nd {
                    best_nd = nd;
                }
            }
        }

        if !hit {
            return false;
        }

        // thin shell
        if best_nd < 0.80 {
            let n = hash3(self.seed ^ 0xCAFE_BABE, x, y, z);
            return (n & 63) == 0; // ~1/64 interior
        }

        // shell porosity
        let n = hash3(self.seed ^ 0xD00D_F00D, x, y, z);
        if (n & 7) == 0 {
            return false; // drop 1/8 on shell
        }

        // reduce leaves near trunk centerline
        let trunk_clear = 0.45 * (vpm as f32);
        let dtr2 = dx0 * dx0 + dz0 * dz0;
        if dtr2 < trunk_clear * trunk_clear {
            let n2 = hash3(self.seed ^ 0x5151_5151, x, y, z);
            if (n2 & 15) != 0 {
                return false; // keep only 1/16 near trunk
            }
        }

        true
    }

    // -------------------------------------------------------------------------
    // Fast tree material query: uses TreeCache
    // -------------------------------------------------------------------------

    pub fn trees_material_from_cache<F: Fn(i32, i32) -> i32>(
        &self,
        x: i32,
        y: i32,
        z: i32,
        height_at: &F,
        cache: &TreeCache,
    ) -> u32 {
        if y < height_at(x, z) {
            return AIR;
        }

        for tree in &cache.trees {
            let top_y = tree.base_y + tree.trunk_h;

            // conservative rejects
            let r = tree.crown_r + 2 * VOXELS_PER_METER;
            if x < tree.tx - r || x > tree.tx + r || z < tree.tz - r || z > tree.tz + r {
                continue;
            }
            if y < tree.base_y || y > top_y + tree.canopy_h + VOXELS_PER_METER {
                continue;
            }

            // trunk
            if y >= tree.base_y && y <= top_y {
                let rr = Self::trunk_radius_at(tree, y);
                let (wx, wz) = self.trunk_wobble(tree, y);
                let cx = tree.tx as f32 + wx;
                let cz = tree.tz as f32 + wz;
                let dx = x as f32 - cx;
                let dz = z as f32 - cz;
                if dx * dx + dz * dz <= rr * rr {
                    return WOOD;
                }
            }

            // branch wood
            if self.is_branch_wood(tree, x, y, z) {
                return WOOD;
            }

            // branch leaves (now sparse/gated)
            if self.is_branch_leaf(tree, x, y, z) {
                return LEAF;
            }

            // canopy leaves (already gated inside is_canopy_leaf)
            if y > top_y && self.is_canopy_leaf(tree, x, y, z) {
                return LEAF;
            }
        }

        AIR
    }

    /// Used by SVO builder to stamp tree-top bounds. Keep conservative.
    pub fn tree_instance_at_meter(&self, xm: i32, zm: i32) -> Option<(i32, i32)> {
        let (_seed, trunk_h_vox, crown_r_vox) = self.tree_at_meter_cell(xm, zm)?;
        Some((trunk_h_vox, crown_r_vox))
    }
}
