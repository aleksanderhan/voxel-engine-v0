// src/world/trees.rs
use crate::config;

use super::{
    generator::WorldGen,
    hash::{hash2, hash3, hash_u32, u01},
    materials::{AIR, LEAF, WOOD},
};

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
    /// This removes the “mostly one plane / one side” bias by allowing azimuthal rotation.
    #[inline(always)]
    fn basis_from_w(w: (f32, f32, f32)) -> ((f32, f32, f32), (f32, f32, f32), (f32, f32, f32)) {
        let (wx, wy, wz) = w;
        // pick a helper not parallel to w
        let (hx, hy, hz) = if wy.abs() < 0.95 { (0.0, 1.0, 0.0) } else { (1.0, 0.0, 0.0) };
        let (ux, uy, uz) = Self::cross3(hx, hy, hz, wx, wy, wz);
        let (ux, uy, uz) = Self::norm3(ux, uy, uz);
        let (vx, vy, vz) = Self::cross3(wx, wy, wz, ux, uy, uz);
        ( (ux, uy, uz), (vx, vy, vz), (wx, wy, wz) )
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
        if (r % 96) != 0 {
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

        // kept for conservative bounds / variety (not used to make a big leaf blob)
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
    // Organic tree geometry helpers
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
        let amp = 0.25 * (config::VOXELS_PER_METER as f32) * t;
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
    fn branch_count_for_tree(&self, tree: &Tree) -> usize {
        let r = hash_u32(tree.seed ^ 0xBABA_1234);
        // 10..26 (more primaries)
        let mut n = 10 + (r % 17) as usize;
        if (r & 31) == 0 {
            n = 10;
        } else if (r & 63) == 1 {
            n = 26;
        }
        n
    }

    fn branch_starts_for_tree(&self, tree: &Tree, count: usize) -> [i32; 32] {
        let mut out = [tree.base_y; 32];

        let y_lo = tree.base_y + (tree.trunk_h * 32) / 100;
        let y_hi = tree.base_y + (tree.trunk_h * 93) / 100;

        for i in 0..count.min(32) {
            let t = if count <= 1 { 0.5 } else { (i as f32) / ((count - 1) as f32) };
            let y_base = (Self::lerp(y_lo as f32, y_hi as f32, t)) as i32;

            let r = hash_u32(tree.seed ^ 0xC0DE_0001 ^ (i as u32).wrapping_mul(0x9E37_79B9));
            let jitter = (Self::s11(r ^ 0x1111) * 0.08 * (tree.trunk_h as f32)) as i32;
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

            self.raster_segment_wood(out, px, py, pz, qx, qy, qz, rr0.max(0.55), rr1.max(0.55));

            px = qx;
            py = qy;
            pz = qz;
        }
    }

    /// More children, and (critically) azimuthal rotation around parent tangent to avoid planar / one-side bias.
    fn spawn_child_branches(&self, parent: &Branch, tree_seed: u32, out: &mut Vec<Branch>) {
        if parent.depth >= 4 {
            return; // total levels: 0..4
        }

        let vpm = config::VOXELS_PER_METER as f32;

        let base = hash_u32(parent.seed ^ 0xCC11_0000);

        // higher fan-out mid-depths
        let child_count = match parent.depth {
            0 => 4 + (base % 4) as usize, // 4..7
            1 => 4 + (base % 5) as usize, // 4..8
            2 => 3 + (base % 5) as usize, // 3..7
            _ => 2 + (base % 4) as usize, // 2..5
        };

        for j in 0..child_count {
            let sj = hash_u32(base ^ (j as u32).wrapping_mul(0x9E37_79B9));

            // spawn along curve (skip very base)
            let t = (0.20 + 0.70 * u01(sj ^ 0x1111)).clamp(0.0, 1.0);

            let (ax, ay, az) = Self::bez_point(parent, t);
            let (tx, ty, tz) = Self::bez_tangent(parent, t);
            let w = Self::norm3(tx, ty, tz);
            let (u, v, w) = Self::basis_from_w(w);

            // azimuth around tangent (this is the balance fix)
            let phi = u01(sj ^ 0xFACE_B00C) * std::f32::consts::TAU;
            let (cs, sn) = (phi.cos(), phi.sin());
            let sideways = (
                u.0 * cs + v.0 * sn,
                u.1 * cs + v.1 * sn,
                u.2 * cs + v.2 * sn,
            );

            let depth = parent.depth + 1;

            // length shrinks with depth but keeps many visible twigs
            let max_from_parent = parent.len * (0.78 - 0.10 * (depth as f32));
            let mut len = (0.75 * vpm + u01(sj ^ 0x2222) * (2.70 * vpm)).min(max_from_parent);
            len = len.max(0.60 * vpm);

            // fan amount is symmetric (mean 0), upward bias is symmetric across azimuth
            let fan_amp = 0.75 + 0.25 * (depth as f32);
            let fan = Self::s11(sj ^ 0x3333) * fan_amp;

            let up = (0.16 + 0.14 * (depth as f32)) + 0.32 * u01(sj ^ 0x4444);

            // direction = tangent + sideways + up
            let dirx = (w.0 + sideways.0 * fan).clamp(-2.0, 2.0);
            let diry = (w.1 + sideways.1 * fan + up).clamp(-0.2, 2.0);
            let dirz = (w.2 + sideways.2 * fan).clamp(-2.0, 2.0);
            let (dirx, diry, dirz) = Self::norm3(dirx, diry, dirz);

            // parent radius at spawn point
            let pr = parent.r0 + (parent.r1 - parent.r0) * t;

            // keep children visible, but taper
            let shrink = match depth {
                1 => 0.58,
                2 => 0.52,
                3 => 0.46,
                _ => 0.40,
            };
            let r0 = (pr * shrink).max(0.70);
            let r1 = (r0 * (0.34 + 0.16 * u01(sj ^ 0x5555))).max(0.50);

            let bx = ax + dirx * len;
            let by = ay + diry * len;
            let bz = az + dirz * len;

            // curvature: bend in the sideways direction (still randomized by azimuth)
            let bend = (0.26 + 0.78 * u01(sj ^ 0x6666)) * vpm;

            let k1 = Self::s11(sj ^ 0x7777);
            let k2 = Self::s11(sj ^ 0x8888);
            let u1 = Self::s11(sj ^ 0x9999);
            let u2 = Self::s11(sj ^ 0xAAAA);

            let c1x = ax + dirx * (0.33 * len) + sideways.0 * (bend * 0.85 * k1);
            let c1y = ay + diry * (0.33 * len) + (0.22 * bend * u1);
            let c1z = az + dirz * (0.33 * len) + sideways.2 * (bend * 0.85 * k1);

            let c2x = ax + dirx * (0.70 * len) + sideways.0 * (bend * 0.60 * k2);
            let c2y = ay + diry * (0.70 * len) + (0.18 * bend * u2);
            let c2z = az + dirz * (0.70 * len) + sideways.2 * (bend * 0.60 * k2);

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

        // --- branches: lots of recursion, no canopy blob ---
        let vpm_f = config::VOXELS_PER_METER as f32;

        let primary_count = self.branch_count_for_tree(tree).min(32);
        let starts = self.branch_starts_for_tree(tree, primary_count);

        let ang0 = u01(hash_u32(tree.seed ^ 0xA0A0_0001)) * std::f32::consts::TAU;

        let mut branches: Vec<Branch> = Vec::with_capacity(512);

        for i in 0..primary_count {
            let sy = starts[i];
            let br_seed = hash_u32(tree.seed ^ 0xB000 ^ (i as u32).wrapping_mul(0x9E37_79B9));

            let ang_j = Self::s11(hash_u32(br_seed ^ 0xABCD_0001)) * 0.78;
            let ang = ang0 + (i as f32) * (std::f32::consts::TAU / (primary_count as f32)) + ang_j;

            let dirx = ang.cos();
            let dirz = ang.sin();

            // Primary length: longer for more children
            let len = (2.6 * vpm_f) + u01(hash_u32(br_seed ^ 0x1111)) * (3.8 * vpm_f);

            let h = ((sy - tree.base_y) as f32 / (tree.trunk_h as f32)).clamp(0.0, 1.0);
            let p = u01(hash_u32(br_seed ^ 0x2222));
            let pitch = (0.10 + 0.20 * h) + (p * p) * (0.56 + 0.28 * h);

            // slightly thicker to keep small branches visible
            let br0 = (0.13 * vpm_f) + u01(hash_u32(br_seed ^ 0x3333)) * (0.22 * vpm_f);
            let br1 = br0 * (0.28 + 0.22 * u01(hash_u32(br_seed ^ 0x3334)));

            let (twx, twz) = self.trunk_wobble(tree, sy);
            let ax = tree.tx as f32 + twx;
            let ay = sy as f32;
            let az = tree.tz as f32 + twz;

            let bx = ax + dirx * len;
            let by = ay + pitch * len;
            let bz = az + dirz * len;

            // curve a bit, but not “spiral”
            let bend = (0.75 + 1.05 * u01(hash_u32(br_seed ^ 0x9000))) * vpm_f;
            let k1 = Self::s11(hash_u32(br_seed ^ 0x9001));
            let k2 = Self::s11(hash_u32(br_seed ^ 0x9002));
            let u1 = Self::s11(hash_u32(br_seed ^ 0x9003));
            let u2 = Self::s11(hash_u32(br_seed ^ 0x9004));

            // use world-up perpendicular here (primaries), but children are fully azimuthal
            let perpx = -dirz;
            let perpz = dirx;

            let c1x = ax + dirx * (0.33 * len) + perpx * (bend * (0.95 * k1));
            let c1z = az + dirz * (0.33 * len) + perpz * (bend * (0.95 * k1));
            let c1y = ay + (pitch * len) * 0.33 + (0.32 * bend * u1);

            let c2x = ax + dirx * (0.70 * len) + perpx * (bend * (0.70 * k2));
            let c2z = az + dirz * (0.70 * len) + perpz * (bend * (0.70 * k2));
            let c2y = ay + (pitch * len) * 0.70 + (0.24 * bend * u2);

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

        // BFS recursion to generate many endpoints.
        let mut k = 0usize;
        let cap = 520usize; // more branches
        while k < branches.len() && branches.len() < cap {
            let b = branches[k];
            self.spawn_child_branches(&b, tree.seed, &mut branches);
            k += 1;
        }

        // Raster all branches as wood
        for b in &branches {
            let steps = (10.0 + (b.len / (0.30 * vpm_f))).round() as i32;
            let steps = steps.clamp(8, 30);
            self.raster_bezier_wood(out, b, steps);
        }

        // Leaves:
        // - Put leaves on (almost) every branch that is depth>=2.
        // - Also add 1–3 extra tufts along the last part of the branch so you don't get “bare sticks”.
        for b in &branches {
            if b.depth < 2 {
                continue;
            }

            // very rare bare branch for variety (was too common before)
            if (hash_u32(b.seed ^ 0x1357_2468) & 255) == 0 {
                continue;
            }

            let tuft_r = match b.depth {
                4 => (0.26 * vpm_f) + u01(hash_u32(b.seed ^ 0x4444)) * (0.50 * vpm_f),
                3 => (0.25 * vpm_f) + u01(hash_u32(b.seed ^ 0x4445)) * (0.46 * vpm_f),
                _ => (0.24 * vpm_f) + u01(hash_u32(b.seed ^ 0x4446)) * (0.42 * vpm_f),
            };

            // Always a tip tuft
            self.raster_sphere_leaf_tuft_dense(out, b.bx, b.by, b.bz, tuft_r, b.seed ^ 0x1EE7_1EAF);

            // Sleeve tufts along last ~35% of branch (prevents bare branches)
            let sleeve_n = match b.depth {
                4 => 3,
                3 => 2,
                _ => 1,
            };

            for si in 0..sleeve_n {
                let rr = hash_u32(b.seed ^ 0x9000_1000 ^ (si as u32).wrapping_mul(0x9E37_79B9));
                let t = 0.62 + 0.30 * u01(rr ^ 0x0101); // 0.62..0.92
                let (mx, my, mz) = Self::bez_point(b, t);

                // small random offset around the branch (keeps it from looking like a ball-chain)
                let ox = Self::s11(rr ^ 0x0202) * (0.22 * vpm_f);
                let oy = u01(rr ^ 0x0303) * (0.18 * vpm_f);
                let oz = Self::s11(rr ^ 0x0404) * (0.22 * vpm_f);

                self.raster_sphere_leaf_tuft_dense(
                    out,
                    mx + ox,
                    my + oy,
                    mz + oz,
                    tuft_r * (0.60 + 0.12 * (si as f32)),
                    b.seed ^ rr ^ 0x2AA2_2AA2,
                );
            }

            // Extra offset tip tuft quite often to break symmetry
            if (hash_u32(b.seed ^ 0xABC0_0001) & 3) != 0 {
                let rr = hash_u32(b.seed ^ 0xABC0_0002);
                let ox = Self::s11(rr ^ 0xABC1) * (0.30 * vpm_f);
                let oy = u01(rr ^ 0xABC2) * (0.22 * vpm_f);
                let oz = Self::s11(rr ^ 0xABC3) * (0.30 * vpm_f);
                self.raster_sphere_leaf_tuft_dense(
                    out,
                    b.bx + ox,
                    b.by + oy,
                    b.bz + oz,
                    tuft_r * 0.80,
                    b.seed ^ 0x600D_600D,
                );
            }
        }

        // NOTE: no canopy raster pass at all.
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

    // Denser leaf tufts (still irregular). Designed to be used many times on branch tips/sleeves.
    fn raster_sphere_leaf_tuft_dense(
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
            let dy = (py - cy) * 1.16;
            let dy2 = dy * dy;

            for x in minx..=maxx {
                let px = x as f32;
                let dx = px - cx;
                let dx2 = dx * dx;

                for z in minz..=maxz {
                    // keep ~1/2 (denser than earlier)
                    let gate = hash3(seed ^ 0xA11C_E5ED, x, y, z) & 1;
                    if gate != 0 {
                        continue;
                    }

                    let pz = z as f32;
                    let dz = pz - cz;
                    let d2 = dx2 + dy2 + dz * dz;
                    if d2 > r2 {
                        continue;
                    }

                    // denser interior; still a bit hollow-ish
                    let nd = (d2.sqrt() / rr).clamp(0.0, 1.0);
                    if nd < 0.80 {
                        let n = hash3(seed ^ 0xCAFE_BABE, x, y, z);
                        if (n & 31) != 0 {
                            continue; // keep 1/32 interior
                        }
                    } else {
                        let n = hash3(seed ^ 0xD00D_F00D, x, y, z);
                        if (n & 15) == 0 {
                            continue; // drop 1/16 near shell
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
