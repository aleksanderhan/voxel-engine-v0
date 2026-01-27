// src/world/trees.rs
use crate::config;

use super::{
    generator::WorldGen,
    hash::{hash2, hash3, hash_u32, u01},
    materials::{AIR, LEAF, WOOD},
};

// ================================================================================================
// TREE TUNING KNOBS (adjust these)
// ================================================================================================
//
// Units:
// - many are in "meters", internally multiplied by VOXELS_PER_METER.
// - probabilities are via bitmasks (fast): (hash & mask) == 0  => true with prob 1/(mask+1).

const TREE_CELL_MOD: u32 = 96; // tree density: 1 per N meter-cells (lower => more trees)

// Primary branches
const PRIMARY_MIN: usize = 10;
const PRIMARY_MAX: usize = 26;
const PRIMARY_MAX_CLAMP: usize = 32;
const PRIMARY_START_Y_LO_FRAC: i32 = 32; // % of trunk height
const PRIMARY_START_Y_HI_FRAC: i32 = 93; // % of trunk height
const PRIMARY_START_JITTER_FRAC: f32 = 0.08; // fraction of trunk height

const PRIMARY_ANG_JITTER: f32 = 0.78; // radians-ish scaled from [-1..1]
const PRIMARY_LEN_M_MIN: f32 = 2.6;
const PRIMARY_LEN_M_RAND: f32 = 3.8;

const PRIMARY_PITCH_BASE: f32 = 0.10; // + h*...
const PRIMARY_PITCH_H_SCALE: f32 = 0.20;
const PRIMARY_PITCH_QUAD_BASE: f32 = 0.56;
const PRIMARY_PITCH_QUAD_H_SCALE: f32 = 0.28;

const PRIMARY_R0_M_MIN: f32 = 0.13;
const PRIMARY_R0_M_RAND: f32 = 0.22;
const PRIMARY_R1_SCALE_MIN: f32 = 0.28;
const PRIMARY_R1_SCALE_RAND: f32 = 0.22;

const PRIMARY_BEND_M_MIN: f32 = 0.75;
const PRIMARY_BEND_M_RAND: f32 = 1.05;
const PRIMARY_C1_T: f32 = 0.33;
const PRIMARY_C2_T: f32 = 0.70;
const PRIMARY_C1_BEND_SCALE: f32 = 0.95;
const PRIMARY_C2_BEND_SCALE: f32 = 0.70;
const PRIMARY_C1_Y_BEND_SCALE: f32 = 0.32;
const PRIMARY_C2_Y_BEND_SCALE: f32 = 0.24;

// Recursion
const MAX_BRANCH_DEPTH: u8 = 4;
const BRANCH_CAP: usize = 520;

const CHILD_COUNT_D0_MIN: usize = 4;
const CHILD_COUNT_D0_RAND: usize = 4; // 4..7
const CHILD_COUNT_D1_MIN: usize = 4;
const CHILD_COUNT_D1_RAND: usize = 5; // 4..8
const CHILD_COUNT_D2_MIN: usize = 3;
const CHILD_COUNT_D2_RAND: usize = 5; // 3..7
const CHILD_COUNT_D3_MIN: usize = 2;
const CHILD_COUNT_D3_RAND: usize = 4; // 2..5

const CHILD_SPAWN_T_MIN: f32 = 0.20;
const CHILD_SPAWN_T_RAND: f32 = 0.70;

const CHILD_LEN_M_MIN: f32 = 0.75;
const CHILD_LEN_M_RAND: f32 = 2.70;
const CHILD_LEN_M_FLOOR: f32 = 0.60;

// parent.len * (A - B*depth)
const CHILD_MAXLEN_A: f32 = 0.78;
const CHILD_MAXLEN_B: f32 = 0.10;

const CHILD_FAN_AMP_BASE: f32 = 0.75;
const CHILD_FAN_AMP_DEPTH: f32 = 0.25;

const CHILD_UP_BASE: f32 = 0.16;
const CHILD_UP_DEPTH: f32 = 0.14;
const CHILD_UP_RAND: f32 = 0.32;

const CHILD_SHRINK_D1: f32 = 0.58;
const CHILD_SHRINK_D2: f32 = 0.52;
const CHILD_SHRINK_D3: f32 = 0.46;
const CHILD_SHRINK_D4: f32 = 0.40;
const CHILD_R0_MIN_VOX: f32 = 0.70;
const CHILD_R1_MIN_VOX: f32 = 0.50;
const CHILD_R1_SCALE_BASE: f32 = 0.34;
const CHILD_R1_SCALE_RAND: f32 = 0.16;

const CHILD_BEND_M_MIN: f32 = 0.26;
const CHILD_BEND_M_RAND: f32 = 0.78;
const CHILD_C1_T: f32 = 0.33;
const CHILD_C2_T: f32 = 0.70;
const CHILD_C1_BEND_SCALE: f32 = 0.85;
const CHILD_C2_BEND_SCALE: f32 = 0.60;
const CHILD_C1_Y_BEND_SCALE: f32 = 0.22;
const CHILD_C2_Y_BEND_SCALE: f32 = 0.18;

// Branch rasterization
const BRANCH_STEPS_MIN: i32 = 8;
const BRANCH_STEPS_MAX: i32 = 30;
const BRANCH_STEPS_BASE: f32 = 10.0;
const BRANCH_STEPS_LEN_DIV_M: f32 = 0.30; // more => fewer steps
const BRANCH_RADIUS_MIN_VOX: f32 = 0.55;

// Leaves (tufts on branches; no canopy blob)
const LEAF_MIN_DEPTH: u8 = 2;

// Sparser canopy: increase these masks and/or decrease radii.
// (hash & mask)==0 => probability 1/(mask+1). Larger mask => rarer.
const LEAF_RARE_SKIP_MASK: u32 = 1023; // 1/1024 branches become bare (very rare)

const LEAF_SLEEVE_N_D2: usize = 1;
const LEAF_SLEEVE_N_D3: usize = 2;
const LEAF_SLEEVE_N_D4: usize = 3;
const LEAF_SLEEVE_T_MIN: f32 = 0.62;
const LEAF_SLEEVE_T_RAND: f32 = 0.30;

const LEAF_OFFSET_TIP_MASK: u32 = 7; // tip offset tuft occurs when (hash & mask)!=0 => ~7/8
const LEAF_OFFSET_M_XZ: f32 = 0.30;
const LEAF_OFFSET_M_Y: f32 = 0.22;

const LEAF_SLEEVE_OFFSET_M_XZ: f32 = 0.22;
const LEAF_SLEEVE_OFFSET_M_Y: f32 = 0.18;

// Tuft radii (meters)
const TUFT_R_M_D2: f32 = 0.24;
const TUFT_R_M_D2_RAND: f32 = 0.34;
const TUFT_R_M_D3: f32 = 0.25;
const TUFT_R_M_D3_RAND: f32 = 0.38;
const TUFT_R_M_D4: f32 = 0.26;
const TUFT_R_M_D4_RAND: f32 = 0.42;

// Leaf voxel fill inside tuft:
// gate keeps 1/(gate_mask+1) of candidates; interior keeps 1/(interior_keep_mask+1)
const LEAF_GATE_MASK: u32 = 3; // 1/4 kept (sparser). was 1 (1/2)
const LEAF_INTERIOR_KEEP_MASK: u32 = 63; // 1/64 kept interior (sparser). was 31 (1/32)
const LEAF_SHELL_DROP_MASK: u32 = 15; // drop 1/16 near shell (same)
const LEAF_SHELL_THRESH: f32 = 0.80;
const LEAF_SPHERE_Y_SCALE: f32 = 1.16;

// Trunk (unchanged)
const TRUNK_WOBBLE_M_MAX: f32 = 0.25;

// ================================================================================================

/// Per-chunk cache of trees that can affect material queries.
pub struct TreeCache {
    trees: Vec<Tree>,
}

/// Chunk-local voxel mask for trees (fast O(1) queries in the chunk build).
/// mask codes: 0 = none, 1 = wood, 2 = leaf
pub struct TreeMaskCache {
    pub origin: [i32; 3], // chunk origin in world-voxel coords
    pub size: i32,        // chunk side in voxels
    mask: Vec<u8>,

    // precomputed strides for indexing
    stride_z: usize,
    stride_y: usize,
}

/// Procedural tree instance (voxel units for positions/sizes).
#[derive(Clone, Copy)]
struct Tree {
    tx: i32,
    tz: i32,
    base_y: i32,   // touches ground
    trunk_h: i32,  // voxels
    crown_r: i32,  // voxels (horizontal extent)
    canopy_h: i32, // voxels (kept for bounds/variety; no canopy-blob raster)
    trunk_r0: f32, // voxels radius at base
    trunk_r1: f32, // voxels radius near top
    seed: u32,
}

/// A single curved branch segment (cubic bezier), plus metadata for recursion.
#[derive(Clone, Copy)]
struct Branch {
    ax: f32,
    ay: f32,
    az: f32,
    c1x: f32,
    c1y: f32,
    c1z: f32,
    c2x: f32,
    c2y: f32,
    c2z: f32,
    bx: f32,
    by: f32,
    bz: f32,
    r0: f32,
    r1: f32,
    len: f32,
    depth: u8,
    seed: u32,
}

#[inline(always)]
fn idx3_strided(stride_z: usize, stride_y: usize, x: i32, y: i32, z: i32) -> usize {
    (y as usize) * stride_y + (z as usize) * stride_z + (x as usize)
}

impl TreeMaskCache {
    #[inline(always)]
    pub fn contains_point_fast(&self, x: i32, y: i32, z: i32) -> u8 {
        let lx = x - self.origin[0];
        let ly = y - self.origin[1];
        let lz = z - self.origin[2];

        // single bounds check
        if (lx | ly | lz) < 0 || lx >= self.size || ly >= self.size || lz >= self.size {
            return 0;
        }
        let i = idx3_strided(self.stride_z, self.stride_y, lx, ly, lz);
        self.mask[i]
    }

    #[inline(always)]
    pub fn material_fast(&self, x: i32, y: i32, z: i32) -> u32 {
        match self.contains_point_fast(x, y, z) {
            1 => WOOD,
            2 => LEAF,
            _ => AIR,
        }
    }

    #[inline(always)]
    fn write_leaf(&mut self, lx: i32, ly: i32, lz: i32) {
        if (lx | ly | lz) < 0 || lx >= self.size || ly >= self.size || lz >= self.size {
            return;
        }
        let i = idx3_strided(self.stride_z, self.stride_y, lx, ly, lz);
        // don't overwrite wood
        if self.mask[i] != 1 {
            self.mask[i] = 2;
        }
    }

    #[inline(always)]
    fn write_wood(&mut self, lx: i32, ly: i32, lz: i32) {
        if (lx | ly | lz) < 0 || lx >= self.size || ly >= self.size || lz >= self.size {
            return;
        }
        let i = idx3_strided(self.stride_z, self.stride_y, lx, ly, lz);
        self.mask[i] = 1;
    }
}

impl WorldGen {
    // -------------------------------------------------------------------------
    // Small math helpers
    // -------------------------------------------------------------------------

    #[inline(always)]
    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }

    // Hash -> [-1, +1]
    #[inline(always)]
    fn s11(n: u32) -> f32 {
        (u01(n) - 0.5) * 2.0
    }

    #[inline(always)]
    fn bezier3(a: f32, b: f32, c: f32, d: f32, t: f32) -> f32 {
        let u = 1.0 - t;
        u * u * u * a + 3.0 * u * u * t * b + 3.0 * u * t * t * c + t * t * t * d
    }

    #[inline(always)]
    fn bezier3_tangent(a: f32, b: f32, c: f32, d: f32, t: f32) -> f32 {
        let u = 1.0 - t;
        3.0 * u * u * (b - a) + 6.0 * u * t * (c - b) + 3.0 * t * t * (d - c)
    }

    #[inline(always)]
    fn bez_point(b: &Branch, t: f32) -> (f32, f32, f32) {
        (
            Self::bezier3(b.ax, b.c1x, b.c2x, b.bx, t),
            Self::bezier3(b.ay, b.c1y, b.c2y, b.by, t),
            Self::bezier3(b.az, b.c1z, b.c2z, b.bz, t),
        )
    }

    #[inline(always)]
    fn bez_tangent(b: &Branch, t: f32) -> (f32, f32, f32) {
        (
            Self::bezier3_tangent(b.ax, b.c1x, b.c2x, b.bx, t),
            Self::bezier3_tangent(b.ay, b.c1y, b.c2y, b.by, t),
            Self::bezier3_tangent(b.az, b.c1z, b.c2z, b.bz, t),
        )
    }

    #[inline(always)]
    fn norm3(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        let inv = 1.0 / (x * x + y * y + z * z).sqrt().max(1e-6);
        (x * inv, y * inv, z * inv)
    }

    #[inline(always)]
    fn cross3(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32) -> (f32, f32, f32) {
        (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)
    }

    /// Orthonormal basis around direction `w` (unit). Returns (u, v, w).
    #[inline(always)]
    fn basis_from_w(w: (f32, f32, f32)) -> ((f32, f32, f32), (f32, f32, f32), (f32, f32, f32)) {
        let (wx, wy, wz) = w;
        let (hx, hy, hz) = if wy.abs() < 0.95 { (0.0, 1.0, 0.0) } else { (1.0, 0.0, 0.0) };
        let (ux, uy, uz) = Self::cross3(hx, hy, hz, wx, wy, wz);
        let (ux, uy, uz) = Self::norm3(ux, uy, uz);
        let (vx, vy, vz) = Self::cross3(wx, wy, wz, ux, uy, uz);
        ((ux, uy, uz), (vx, vy, vz), (wx, wy, wz))
    }

    // -------------------------------------------------------------------------
    // Tree placement + params
    // -------------------------------------------------------------------------

    #[inline]
    fn tree_at_meter_cell(
        &self,
        xm: i32,
        zm: i32,
    ) -> Option<(u32 /*seed*/, i32 /*trunk_h_vox*/, i32 /*crown_r_vox*/)> {
        let r = hash2(self.seed, xm, zm);

        // density per 1m cell
        if (r % TREE_CELL_MOD) != 0 {
            return None;
        }

        // trunk height in meters: 5..10m
        let trunk_h_m = 5 + (hash_u32(r) % 6) as i32;

        // crown radius in meters: 2..4m
        let crown_r_m = 2 + (hash_u32(r ^ 0xBEEF) % 3) as i32;

        let vpm = config::VOXELS_PER_METER;
        Some((r, trunk_h_m * vpm, crown_r_m * vpm))
    }

    #[inline]
    fn tree_params_at_trunk(&self, tx: i32, tz: i32, ground: i32) -> Option<Tree> {
        let xm = tx.div_euclid(config::VOXELS_PER_METER);
        let zm = tz.div_euclid(config::VOXELS_PER_METER);
        let (seed, trunk_h, crown_r) = self.tree_at_meter_cell(xm, zm)?;

        let vpm = config::VOXELS_PER_METER as f32;

        // trunk radii (voxels): base ~0.45..0.75m, top ~0.18..0.30m
        let r0 = 0.45 * vpm + u01(hash_u32(seed ^ 0x1111)) * (0.30 * vpm);
        let r1 = 0.18 * vpm + u01(hash_u32(seed ^ 0x2222)) * (0.12 * vpm);

        // kept for conservative bounds / variety
        let canopy_h = ((2.5 * vpm) + u01(hash_u32(seed ^ 0x3333)) * (2.0 * vpm)).round() as i32;

        Some(Tree {
            tx,
            tz,
            base_y: ground,
            trunk_h,
            crown_r,
            canopy_h,
            trunk_r0: r0.max(1.0),
            trunk_r1: r1.max(1.0),
            seed,
        })
    }

    /// Build a per-chunk tree cache. Call once per chunk build.
    pub fn build_tree_cache<F: Fn(i32, i32) -> i32>(
        &self,
        chunk_ox: i32,
        chunk_oz: i32,
        chunk_size: i32,
        height_at: &F,
    ) -> TreeCache {
        let vpm = config::VOXELS_PER_METER;

        // crown up to ~4m + slack
        let pad_m = 6;

        let xm0 = chunk_ox.div_euclid(vpm) - pad_m;
        let xm1 = (chunk_ox + chunk_size).div_euclid(vpm) + pad_m;
        let zm0 = chunk_oz.div_euclid(vpm) - pad_m;
        let zm1 = (chunk_oz + chunk_size).div_euclid(vpm) + pad_m;

        let mut trees = Vec::new();

        for zm in zm0..=zm1 {
            for xm in xm0..=xm1 {
                let Some((_seed, _th, _cr)) = self.tree_at_meter_cell(xm, zm) else {
                    continue;
                };

                let tx = xm * vpm;
                let tz = zm * vpm;
                let ground = height_at(tx, tz);

                let Some(t) = self.tree_params_at_trunk(tx, tz, ground) else {
                    continue;
                };

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

    /// Build a tree cache + chunk-local voxel mask (fast O(1) tree material queries).
    pub fn build_tree_cache_with_mask<F: Fn(i32, i32) -> i32>(
        &self,
        chunk_ox: i32,
        chunk_oy: i32,
        chunk_oz: i32,
        chunk_size: i32,
        height_at: &F,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> (TreeCache, TreeMaskCache) {
        let cache = self.build_tree_cache(chunk_ox, chunk_oz, chunk_size, height_at);

        let side = chunk_size.max(1) as i32;
        let side_u = side as usize;

        let mask = vec![0u8; side_u * side_u * side_u];

        let mut out = TreeMaskCache {
            origin: [chunk_ox, chunk_oy, chunk_oz],
            size: side,
            mask,
            stride_z: side_u,
            stride_y: side_u * side_u,
        };

        for t in &cache.trees {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            self.raster_tree_into_mask(t, &mut out);
        }

        (cache, out)
    }

    // -------------------------------------------------------------------------
    // Geometry helpers
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

        // max drift near the top
        let amp = TRUNK_WOBBLE_M_MAX * (config::VOXELS_PER_METER as f32) * t;
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
    fn primary_count(&self, tree: &Tree) -> usize {
        let r = hash_u32(tree.seed ^ 0xBABA_1234);
        let span = (PRIMARY_MAX - PRIMARY_MIN + 1) as u32;
        let mut n = PRIMARY_MIN + (r % span) as usize;

        if (r & 31) == 0 {
            n = PRIMARY_MIN;
        } else if (r & 63) == 1 {
            n = PRIMARY_MAX;
        }
        n.min(PRIMARY_MAX_CLAMP)
    }

    fn primary_starts(&self, tree: &Tree, count: usize) -> [i32; 32] {
        let mut out = [tree.base_y; 32];

        let y_lo = tree.base_y + (tree.trunk_h * PRIMARY_START_Y_LO_FRAC) / 100;
        let y_hi = tree.base_y + (tree.trunk_h * PRIMARY_START_Y_HI_FRAC) / 100;

        for i in 0..count.min(32) {
            let t = if count <= 1 { 0.5 } else { (i as f32) / ((count - 1) as f32) };
            let y_base = (Self::lerp(y_lo as f32, y_hi as f32, t)) as i32;

            let r = hash_u32(tree.seed ^ 0xC0DE_0001 ^ (i as u32).wrapping_mul(0x9E37_79B9));
            let jitter = (Self::s11(r ^ 0x1111) * PRIMARY_START_JITTER_FRAC * (tree.trunk_h as f32))
                as i32;
            out[i] = (y_base + jitter).clamp(y_lo, y_hi);
        }

        // insertion sort ascending
        for i in 1..count.min(32) {
            let mut j = i;
            while j > 0 && out[j - 1] > out[j] {
                out.swap(j - 1, j);
                j -= 1;
            }
        }

        out
    }

    fn raster_bezier_wood(&self, out: &mut TreeMaskCache, b: &Branch, steps: i32) {
        let steps = steps.max(2);
        let inv = 1.0 / (steps as f32);

        let mut px = b.ax;
        let mut py = b.ay;
        let mut pz = b.az;

        for i in 1..=steps {
            let t = (i as f32) * inv;

            let qx = Self::bezier3(b.ax, b.c1x, b.c2x, b.bx, t);
            let qy = Self::bezier3(b.ay, b.c1y, b.c2y, b.by, t);
            let qz = Self::bezier3(b.az, b.c1z, b.c2z, b.bz, t);

            let rr0 = b.r0 + (b.r1 - b.r0) * (((i - 1) as f32) * inv);
            let rr1 = b.r0 + (b.r1 - b.r0) * t;

            self.raster_segment_wood(out, px, py, pz, qx, qy, qz, rr0.max(BRANCH_RADIUS_MIN_VOX), rr1.max(BRANCH_RADIUS_MIN_VOX));

            px = qx;
            py = qy;
            pz = qz;
        }
    }

    fn child_count_for_depth(&self, depth: u8, base: u32) -> usize {
        match depth {
            0 => CHILD_COUNT_D0_MIN + (base % (CHILD_COUNT_D0_RAND + 1) as u32) as usize,
            1 => CHILD_COUNT_D1_MIN + (base % (CHILD_COUNT_D1_RAND + 1) as u32) as usize,
            2 => CHILD_COUNT_D2_MIN + (base % (CHILD_COUNT_D2_RAND + 1) as u32) as usize,
            _ => CHILD_COUNT_D3_MIN + (base % (CHILD_COUNT_D3_RAND + 1) as u32) as usize,
        }
    }

    fn shrink_for_depth(depth: u8) -> f32 {
        match depth {
            1 => CHILD_SHRINK_D1,
            2 => CHILD_SHRINK_D2,
            3 => CHILD_SHRINK_D3,
            _ => CHILD_SHRINK_D4,
        }
    }

    /// Balanced children via azimuthal rotation around parent tangent.
    fn spawn_child_branches(&self, parent: &Branch, tree_seed: u32, out: &mut Vec<Branch>) {
        if parent.depth >= MAX_BRANCH_DEPTH {
            return;
        }

        let vpm = config::VOXELS_PER_METER as f32;
        let base = hash_u32(parent.seed ^ 0xCC11_0000);

        let child_count = self.child_count_for_depth(parent.depth, base);

        for j in 0..child_count {
            let sj = hash_u32(base ^ (j as u32).wrapping_mul(0x9E37_79B9));

            let t = (CHILD_SPAWN_T_MIN + CHILD_SPAWN_T_RAND * u01(sj ^ 0x1111)).clamp(0.0, 1.0);

            let (ax, ay, az) = Self::bez_point(parent, t);
            let (tx, ty, tz) = Self::bez_tangent(parent, t);
            let w = Self::norm3(tx, ty, tz);
            let (u, v, w) = Self::basis_from_w(w);

            let phi = u01(sj ^ 0xFACE_B00C) * std::f32::consts::TAU;
            let (cs, sn) = (phi.cos(), phi.sin());
            let sideways = (u.0 * cs + v.0 * sn, u.1 * cs + v.1 * sn, u.2 * cs + v.2 * sn);

            let depth = parent.depth + 1;

            let max_from_parent = parent.len * (CHILD_MAXLEN_A - CHILD_MAXLEN_B * (depth as f32));
            let mut len = (CHILD_LEN_M_MIN * vpm + u01(sj ^ 0x2222) * (CHILD_LEN_M_RAND * vpm)).min(max_from_parent);
            len = len.max(CHILD_LEN_M_FLOOR * vpm);

            let fan_amp = CHILD_FAN_AMP_BASE + CHILD_FAN_AMP_DEPTH * (depth as f32);
            let fan = Self::s11(sj ^ 0x3333) * fan_amp;

            let up = (CHILD_UP_BASE + CHILD_UP_DEPTH * (depth as f32)) + CHILD_UP_RAND * u01(sj ^ 0x4444);

            let dirx = (w.0 + sideways.0 * fan).clamp(-2.0, 2.0);
            let diry = (w.1 + sideways.1 * fan + up).clamp(-0.2, 2.0);
            let dirz = (w.2 + sideways.2 * fan).clamp(-2.0, 2.0);
            let (dirx, diry, dirz) = Self::norm3(dirx, diry, dirz);

            let pr = parent.r0 + (parent.r1 - parent.r0) * t;

            let shrink = Self::shrink_for_depth(depth);
            let r0 = (pr * shrink).max(CHILD_R0_MIN_VOX);
            let r1 = (r0 * (CHILD_R1_SCALE_BASE + CHILD_R1_SCALE_RAND * u01(sj ^ 0x5555))).max(CHILD_R1_MIN_VOX);

            let bx = ax + dirx * len;
            let by = ay + diry * len;
            let bz = az + dirz * len;

            let bend = (CHILD_BEND_M_MIN * vpm) + (CHILD_BEND_M_RAND * vpm) * u01(sj ^ 0x6666);

            let k1 = Self::s11(sj ^ 0x7777);
            let k2 = Self::s11(sj ^ 0x8888);
            let u1 = Self::s11(sj ^ 0x9999);
            let u2 = Self::s11(sj ^ 0xAAAA);

            let c1x = ax + dirx * (CHILD_C1_T * len) + sideways.0 * (bend * CHILD_C1_BEND_SCALE * k1);
            let c1y = ay + diry * (CHILD_C1_T * len) + (bend * CHILD_C1_Y_BEND_SCALE * u1);
            let c1z = az + dirz * (CHILD_C1_T * len) + sideways.2 * (bend * CHILD_C1_BEND_SCALE * k1);

            let c2x = ax + dirx * (CHILD_C2_T * len) + sideways.0 * (bend * CHILD_C2_BEND_SCALE * k2);
            let c2y = ay + diry * (CHILD_C2_T * len) + (bend * CHILD_C2_Y_BEND_SCALE * u2);
            let c2z = az + dirz * (CHILD_C2_T * len) + sideways.2 * (bend * CHILD_C2_BEND_SCALE * k2);

            out.push(Branch {
                ax,
                ay,
                az,
                c1x,
                c1y,
                c1z,
                c2x,
                c2y,
                c2z,
                bx,
                by,
                bz,
                r0,
                r1,
                len,
                depth,
                seed: hash_u32(tree_seed ^ sj ^ 0xBEEF_1234),
            });
        }
    }

    // -------------------------------------------------------------------------
    // Rasterization into chunk-local mask
    // -------------------------------------------------------------------------

    fn raster_tree_into_mask(&self, tree: &Tree, out: &mut TreeMaskCache) {
        let side = out.size;

        // --- trunk ---
        let top_y = tree.base_y + tree.trunk_h;
        let y0 = tree.base_y.max(out.origin[1]);
        let y1 = top_y.min(out.origin[1] + side - 1);

        for wy in y0..=y1 {
            let (wx, wz) = self.trunk_wobble(tree, wy);
            let cx = tree.tx as f32 + wx;
            let cz = tree.tz as f32 + wz;
            let rr = Self::trunk_radius_at(tree, wy).max(1.0);
            let r = rr.ceil() as i32;
            let rr2 = rr * rr;

            let minx = (cx.floor() as i32 - r).max(out.origin[0]);
            let maxx = (cx.ceil() as i32 + r).min(out.origin[0] + side - 1);
            let minz = (cz.floor() as i32 - r).max(out.origin[2]);
            let maxz = (cz.ceil() as i32 + r).min(out.origin[2] + side - 1);

            for x in minx..=maxx {
                let dx = (x as f32) - cx;
                let dx2 = dx * dx;
                for z in minz..=maxz {
                    let dz = (z as f32) - cz;
                    if dx2 + dz * dz <= rr2 {
                        out.write_wood(x - out.origin[0], wy - out.origin[1], z - out.origin[2]);
                    }
                }
            }
        }

        // --- branches (recursive), no canopy blob ---
        let vpm_f = config::VOXELS_PER_METER as f32;

        let primary_count = self.primary_count(tree);
        let starts = self.primary_starts(tree, primary_count);

        let ang0 = u01(hash_u32(tree.seed ^ 0xA0A0_0001)) * std::f32::consts::TAU;

        let mut branches: Vec<Branch> = Vec::with_capacity(BRANCH_CAP.min(1024));

        for i in 0..primary_count.min(32) {
            let sy = starts[i];
            let br_seed = hash_u32(tree.seed ^ 0xB000 ^ (i as u32).wrapping_mul(0x9E37_79B9));

            let ang_j = Self::s11(hash_u32(br_seed ^ 0xABCD_0001)) * PRIMARY_ANG_JITTER;
            let ang = ang0 + (i as f32) * (std::f32::consts::TAU / (primary_count as f32)) + ang_j;

            let dirx = ang.cos();
            let dirz = ang.sin();

            let len = (PRIMARY_LEN_M_MIN * vpm_f) + u01(hash_u32(br_seed ^ 0x1111)) * (PRIMARY_LEN_M_RAND * vpm_f);

            let h = ((sy - tree.base_y) as f32 / (tree.trunk_h as f32)).clamp(0.0, 1.0);
            let p = u01(hash_u32(br_seed ^ 0x2222));
            let pitch = (PRIMARY_PITCH_BASE + PRIMARY_PITCH_H_SCALE * h)
                + (p * p) * (PRIMARY_PITCH_QUAD_BASE + PRIMARY_PITCH_QUAD_H_SCALE * h);

            let br0 = (PRIMARY_R0_M_MIN * vpm_f) + u01(hash_u32(br_seed ^ 0x3333)) * (PRIMARY_R0_M_RAND * vpm_f);
            let br1 = br0 * (PRIMARY_R1_SCALE_MIN + PRIMARY_R1_SCALE_RAND * u01(hash_u32(br_seed ^ 0x3334)));

            let (twx, twz) = self.trunk_wobble(tree, sy);
            let ax = tree.tx as f32 + twx;
            let ay = sy as f32;
            let az = tree.tz as f32 + twz;

            let bx = ax + dirx * len;
            let by = ay + pitch * len;
            let bz = az + dirz * len;

            let bend = (PRIMARY_BEND_M_MIN * vpm_f) + (PRIMARY_BEND_M_RAND * vpm_f) * u01(hash_u32(br_seed ^ 0x9000));
            let k1 = Self::s11(hash_u32(br_seed ^ 0x9001));
            let k2 = Self::s11(hash_u32(br_seed ^ 0x9002));
            let u1 = Self::s11(hash_u32(br_seed ^ 0x9003));
            let u2 = Self::s11(hash_u32(br_seed ^ 0x9004));

            let perpx = -dirz;
            let perpz = dirx;

            let c1x = ax + dirx * (PRIMARY_C1_T * len) + perpx * (bend * (PRIMARY_C1_BEND_SCALE * k1));
            let c1z = az + dirz * (PRIMARY_C1_T * len) + perpz * (bend * (PRIMARY_C1_BEND_SCALE * k1));
            let c1y = ay + (pitch * len) * PRIMARY_C1_T + (PRIMARY_C1_Y_BEND_SCALE * bend * u1);

            let c2x = ax + dirx * (PRIMARY_C2_T * len) + perpx * (bend * (PRIMARY_C2_BEND_SCALE * k2));
            let c2z = az + dirz * (PRIMARY_C2_T * len) + perpz * (bend * (PRIMARY_C2_BEND_SCALE * k2));
            let c2y = ay + (pitch * len) * PRIMARY_C2_T + (PRIMARY_C2_Y_BEND_SCALE * bend * u2);

            branches.push(Branch {
                ax,
                ay,
                az,
                c1x,
                c1y,
                c1z,
                c2x,
                c2y,
                c2z,
                bx,
                by,
                bz,
                r0: br0.max(1.0),
                r1: br1.max(0.80),
                len,
                depth: 0,
                seed: hash_u32(tree.seed ^ br_seed ^ 0x1A2B_3C4D),
            });
        }

        // BFS recursion
        let mut k = 0usize;
        while k < branches.len() && branches.len() < BRANCH_CAP {
            let b = branches[k];
            self.spawn_child_branches(&b, tree.seed, &mut branches);
            k += 1;
        }

        // wood
        for b in &branches {
            let len_m = b.len / vpm_f;
            let steps = (BRANCH_STEPS_BASE + (len_m / BRANCH_STEPS_LEN_DIV_M)).round() as i32;
            let steps = steps.clamp(BRANCH_STEPS_MIN, BRANCH_STEPS_MAX);
            self.raster_bezier_wood(out, b, steps);
        }

        // leaves: sparser overall, but still distributed (no bare “canopy branches”)
        for b in &branches {
            if b.depth < LEAF_MIN_DEPTH {
                continue;
            }

            // extremely rare bare branch
            if (hash_u32(b.seed ^ 0x1357_2468) & LEAF_RARE_SKIP_MASK) == 0 {
                continue;
            }

            let tuft_r_m = match b.depth {
                4 => TUFT_R_M_D4 + TUFT_R_M_D4_RAND * u01(hash_u32(b.seed ^ 0x4444)),
                3 => TUFT_R_M_D3 + TUFT_R_M_D3_RAND * u01(hash_u32(b.seed ^ 0x4445)),
                _ => TUFT_R_M_D2 + TUFT_R_M_D2_RAND * u01(hash_u32(b.seed ^ 0x4446)),
            };
            let tuft_r = tuft_r_m * vpm_f;

            // tip tuft
            self.raster_sphere_leaf_tuft(
                out,
                b.bx,
                b.by,
                b.bz,
                tuft_r,
                b.seed ^ 0x1EE7_1EAF,
            );

            // sleeve tufts near tip
            let sleeve_n = match b.depth {
                4 => LEAF_SLEEVE_N_D4,
                3 => LEAF_SLEEVE_N_D3,
                _ => LEAF_SLEEVE_N_D2,
            };

            for si in 0..sleeve_n {
                let rr = hash_u32(b.seed ^ 0x9000_1000 ^ (si as u32).wrapping_mul(0x9E37_79B9));
                let t = LEAF_SLEEVE_T_MIN + LEAF_SLEEVE_T_RAND * u01(rr ^ 0x0101);
                let (mx, my, mz) = Self::bez_point(b, t);

                let ox = Self::s11(rr ^ 0x0202) * (LEAF_SLEEVE_OFFSET_M_XZ * vpm_f);
                let oy = u01(rr ^ 0x0303) * (LEAF_SLEEVE_OFFSET_M_Y * vpm_f);
                let oz = Self::s11(rr ^ 0x0404) * (LEAF_SLEEVE_OFFSET_M_XZ * vpm_f);

                self.raster_sphere_leaf_tuft(
                    out,
                    mx + ox,
                    my + oy,
                    mz + oz,
                    tuft_r * (0.58 + 0.10 * (si as f32)),
                    b.seed ^ rr ^ 0x2AA2_2AA2,
                );
            }

            // offset tip tuft (usually)
            let h = hash_u32(b.seed ^ 0xABC0_0001);
            if (h & LEAF_OFFSET_TIP_MASK) != 0 {
                let rr = hash_u32(b.seed ^ 0xABC0_0002);
                let ox = Self::s11(rr ^ 0xABC1) * (LEAF_OFFSET_M_XZ * vpm_f);
                let oy = u01(rr ^ 0xABC2) * (LEAF_OFFSET_M_Y * vpm_f);
                let oz = Self::s11(rr ^ 0xABC3) * (LEAF_OFFSET_M_XZ * vpm_f);
                self.raster_sphere_leaf_tuft(
                    out,
                    b.bx + ox,
                    b.by + oy,
                    b.bz + oz,
                    tuft_r * 0.80,
                    b.seed ^ 0x600D_600D,
                );
            }
        }
    }

    fn raster_segment_wood(
        &self,
        out: &mut TreeMaskCache,
        ax: f32,
        ay: f32,
        az: f32,
        bx: f32,
        by: f32,
        bz: f32,
        r0: f32,
        r1: f32,
    ) {
        let rr = r0.max(r1);
        let r = rr.ceil() as i32;

        let minx = (ax.min(bx).floor() as i32 - r).max(out.origin[0]);
        let maxx = (ax.max(bx).ceil() as i32 + r).min(out.origin[0] + out.size - 1);
        let miny = (ay.min(by).floor() as i32 - r).max(out.origin[1]);
        let maxy = (ay.max(by).ceil() as i32 + r).min(out.origin[1] + out.size - 1);
        let minz = (az.min(bz).floor() as i32 - r).max(out.origin[2]);
        let maxz = (az.max(bz).ceil() as i32 + r).min(out.origin[2] + out.size - 1);

        for y in miny..=maxy {
            let py = y as f32;
            for x in minx..=maxx {
                let px = x as f32;
                for z in minz..=maxz {
                    let pz = z as f32;
                    let (d2, t) = Self::dist2_point_segment(px, py, pz, ax, ay, az, bx, by, bz);
                    let rr = r0 + (r1 - r0) * t;
                    if d2 <= rr * rr {
                        out.write_wood(x - out.origin[0], y - out.origin[1], z - out.origin[2]);
                    }
                }
            }
        }
    }

    // Leaf tufts: now sparser (via LEAF_GATE_MASK / LEAF_INTERIOR_KEEP_MASK).
    fn raster_sphere_leaf_tuft(
        &self,
        out: &mut TreeMaskCache,
        cx: f32,
        cy: f32,
        cz: f32,
        r: f32,
        seed: u32,
    ) {
        let rr = r.max(1.0);
        let ir = rr.ceil() as i32;

        let minx = (cx.floor() as i32 - ir).max(out.origin[0]);
        let maxx = (cx.ceil() as i32 + ir).min(out.origin[0] + out.size - 1);
        let miny = (cy.floor() as i32 - ir).max(out.origin[1]);
        let maxy = (cy.ceil() as i32 + ir).min(out.origin[1] + out.size - 1);
        let minz = (cz.floor() as i32 - ir).max(out.origin[2]);
        let maxz = (cz.ceil() as i32 + ir).min(out.origin[2] + out.size - 1);

        let r2 = rr * rr;

        for y in miny..=maxy {
            let py = y as f32;
            let dy = (py - cy) * LEAF_SPHERE_Y_SCALE;
            let dy2 = dy * dy;

            for x in minx..=maxx {
                let px = x as f32;
                let dx = px - cx;
                let dx2 = dx * dx;

                for z in minz..=maxz {
                    // gate
                    let gate = hash3(seed ^ 0xA11C_E5ED, x, y, z) & LEAF_GATE_MASK;
                    if gate != 0 {
                        continue;
                    }

                    let pz = z as f32;
                    let dz = pz - cz;
                    let d2 = dx2 + dy2 + dz * dz;
                    if d2 > r2 {
                        continue;
                    }

                    let nd = (d2.sqrt() / rr).clamp(0.0, 1.0);
                    if nd < LEAF_SHELL_THRESH {
                        let n = hash3(seed ^ 0xCAFE_BABE, x, y, z);
                        if (n & LEAF_INTERIOR_KEEP_MASK) != 0 {
                            continue;
                        }
                    } else {
                        let n = hash3(seed ^ 0xD00D_F00D, x, y, z);
                        if (n & LEAF_SHELL_DROP_MASK) == 0 {
                            continue;
                        }
                    }

                    out.write_leaf(x - out.origin[0], y - out.origin[1], z - out.origin[2]);
                }
            }
        }
    }

    /// Used by the SVO builder to stamp conservative tree-top bounds.
    pub fn tree_instance_at_meter(&self, xm: i32, zm: i32) -> Option<(i32, i32)> {
        let (_seed, trunk_h_vox, crown_r_vox) = self.tree_at_meter_cell(xm, zm)?;
        Some((trunk_h_vox, crown_r_vox))
    }
}
