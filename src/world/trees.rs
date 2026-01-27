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
    canopy_h: i32, // voxels (vertical extent)
    trunk_r0: f32, // voxels radius at base
    trunk_r1: f32, // voxels radius near top
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
        // cubic bezier
        let u = 1.0 - t;
        u * u * u * a + 3.0 * u * u * t * b + 3.0 * u * t * t * c + t * t * t * d
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

        // canopy height: ~2.5..4.5m
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
    /// Use this in the SVO builder to avoid per-voxel geometry tests.
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

        // Rasterize each tree into the mask.
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
        // 2..7
        let mut n = 2 + (r % 6) as usize;
        // occasional extremes
        if (r & 7) == 0 {
            n = 2;
        } else if (r & 7) == 1 {
            n = 7;
        }
        n
    }

    fn branch_starts_for_tree(&self, tree: &Tree, count: usize) -> [i32; 8] {
        let mut out = [tree.base_y; 8];

        let y_lo = tree.base_y + (tree.trunk_h * 45) / 100;
        let y_hi = tree.base_y + (tree.trunk_h * 85) / 100;

        for i in 0..count.min(8) {
            let t = if count <= 1 {
                0.5
            } else {
                (i as f32) / ((count - 1) as f32)
            };
            let y_base = (Self::lerp(y_lo as f32, y_hi as f32, t)) as i32;

            let r = hash_u32(tree.seed ^ 0xC0DE_0001 ^ (i as u32).wrapping_mul(0x9E37_79B9));
            let jitter = (Self::s11(r ^ 0x1111) * 0.06 * (tree.trunk_h as f32)) as i32;
            out[i] = (y_base + jitter).clamp(y_lo, y_hi);
        }

        // insertion sort ascending
        for i in 1..count.min(8) {
            let mut j = i;
            while j > 0 && out[j - 1] > out[j] {
                out.swap(j - 1, j);
                j -= 1;
            }
        }

        out
    }

    fn raster_bezier_wood(
        &self,
        out: &mut TreeMaskCache,
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
        steps: i32,
    ) {
        let steps = steps.max(2);
        let inv = 1.0 / (steps as f32);

        let mut px = ax;
        let mut py = ay;
        let mut pz = az;

        for i in 1..=steps {
            let t = (i as f32) * inv;

            let qx = Self::bezier3(ax, c1x, c2x, bx, t);
            let qy = Self::bezier3(ay, c1y, c2y, by, t);
            let qz = Self::bezier3(az, c1z, c2z, bz, t);

            let rr0 = r0 + (r1 - r0) * (((i - 1) as f32) * inv);
            let rr1 = r0 + (r1 - r0) * t;

            self.raster_segment_wood(out, px, py, pz, qx, qy, qz, rr0.max(1.0), rr1.max(1.0));

            px = qx;
            py = qy;
            pz = qz;
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

        // --- branches + branch leaves (variable count, truly curved via bezier tessellation) ---
        let vpm_f = config::VOXELS_PER_METER as f32;

        let bcount = self.branch_count_for_tree(tree).min(8);
        let starts = self.branch_starts_for_tree(tree, bcount);

        let ang0 = u01(hash_u32(tree.seed ^ 0xA0A0_0001)) * std::f32::consts::TAU;

        for i in 0..bcount {
            let sy = starts[i];
            let br_seed = hash_u32(tree.seed ^ 0xB000 ^ (i as u32).wrapping_mul(0x9E37_79B9));

            // Angle around trunk
            let ang_j = Self::s11(hash_u32(br_seed ^ 0xABCD_0001)) * 0.55;
            let ang = ang0 + (i as f32) * (std::f32::consts::TAU / (bcount as f32)) + ang_j;

            let dirx = ang.cos();
            let dirz = ang.sin();
            let perpx = -dirz;
            let perpz = dirx;

            // Branch length
            let len = (1.6 * vpm_f) + u01(hash_u32(br_seed ^ 0x1111)) * (2.4 * vpm_f);

            // Height fraction on trunk
            let h = ((sy - tree.base_y) as f32 / (tree.trunk_h as f32)).clamp(0.0, 1.0);

            // Upward pitch (biased upward, more for higher branches)
            let p = u01(hash_u32(br_seed ^ 0x2222));
            let pitch = (0.22 + 0.30 * h) + (p * p) * (0.40 + 0.30 * h); // ~0.22..0.92

            // Thickness
            let br0 = (0.11 * vpm_f) + u01(hash_u32(br_seed ^ 0x3333)) * (0.13 * vpm_f);
            let br1 = br0 * (0.34 + 0.22 * u01(hash_u32(br_seed ^ 0x3334)));

            // Start on trunk (wobble)
            let (twx, twz) = self.trunk_wobble(tree, sy);
            let ax = tree.tx as f32 + twx;
            let ay = sy as f32;
            let az = tree.tz as f32 + twz;

            // End point (ideal)
            let bx = ax + dirx * len;
            let by = ay + pitch * len;
            let bz = az + dirz * len;

            // Curvature strength in voxels (make it big enough to show at your resolution)
            let bend = (0.85 + 0.85 * u01(hash_u32(br_seed ^ 0x9000))) * vpm_f; // ~0.85..1.70m
            let k1 = Self::s11(hash_u32(br_seed ^ 0x9001));
            let k2 = Self::s11(hash_u32(br_seed ^ 0x9002));
            let u1 = Self::s11(hash_u32(br_seed ^ 0x9003));
            let u2 = Self::s11(hash_u32(br_seed ^ 0x9004));

            // Control points: sideways S-curve + slight vertical kinks
            let c1x = ax + dirx * (0.33 * len) + perpx * (bend * (0.95 * k1));
            let c1z = az + dirz * (0.33 * len) + perpz * (bend * (0.95 * k1));
            let c1y = ay + (pitch * len) * 0.33 + (0.35 * bend * u1);

            let c2x = ax + dirx * (0.70 * len) + perpx * (bend * (0.70 * k2));
            let c2z = az + dirz * (0.70 * len) + perpz * (bend * (0.70 * k2));
            let c2y = ay + (pitch * len) * 0.70 + (0.25 * bend * u2);

            // Tessellate enough so it stops reading as "straight"
            let steps = 22 + (hash_u32(br_seed ^ 0xDEAD) % 12) as i32; // 22..33
            self.raster_bezier_wood(
                out,
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
                br0.max(1.0),
                br1.max(1.0),
                steps,
            );

            // Choose two points along the curve for leaf placement (mid + tip)
            let tmid = 0.70;
            let midx = Self::bezier3(ax, c1x, c2x, bx, tmid);
            let midy = Self::bezier3(ay, c1y, c2y, by, tmid);
            let midz = Self::bezier3(az, c1z, c2z, bz, tmid);

            let tipx = bx;
            let tipy = by;
            let tipz = bz;

            // Some branches are bare
            if (hash_u32(br_seed ^ 0x55AA_1234) & 31) == 0 {
                continue;
            }

            // Sleeve near tip
            let sleeve_r = (0.18 * vpm_f).max(1.0);
            self.raster_sphere_leaf_canopy_style(
                out,
                tipx,
                tipy,
                tipz,
                sleeve_r,
                tree.seed ^ br_seed ^ 0x600D_600D,
            );

            // Tufts
            let tuft_r = (0.55 * vpm_f) + u01(hash_u32(br_seed ^ 0x4444)) * (0.40 * vpm_f);
            self.raster_sphere_leaf_canopy_style(
                out,
                tipx,
                tipy,
                tipz,
                tuft_r,
                tree.seed ^ br_seed ^ 0x1EE7_1EAF,
            );
            self.raster_sphere_leaf_canopy_style(
                out,
                midx,
                midy,
                midz,
                tuft_r * 0.80,
                tree.seed ^ br_seed ^ 0x2AA2_2AA2,
            );
        }

        // --- canopy leaves (4 lumpy spheres) ---
        self.raster_canopy(out, tree);
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

    fn raster_sphere_leaf_canopy_style(
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
            let dy = (py - cy) * 1.25;
            let dy2 = dy * dy;

            for x in minx..=maxx {
                let px = x as f32;
                let dx = px - cx;
                let dx2 = dx * dx;

                for z in minz..=maxz {
                    // cheap gate (same as canopy): keep ~1/4
                    let gate = hash3(seed ^ 0xA11C_E5ED, x, y, z) & 3;
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

                    if nd < 0.80 {
                        let n = hash3(seed ^ 0xCAFE_BABE, x, y, z);
                        if (n & 63) != 0 {
                            continue; // keep 1/64 in interior
                        }
                    } else {
                        let n = hash3(seed ^ 0xD00D_F00D, x, y, z);
                        if (n & 7) == 0 {
                            continue; // drop 1/8 near shell
                        }
                    }

                    out.write_leaf(x - out.origin[0], y - out.origin[1], z - out.origin[2]);
                }
            }
        }
    }

    fn raster_canopy(&self, out: &mut TreeMaskCache, tree: &Tree) {
        let vpm = config::VOXELS_PER_METER;
        let top_y = tree.base_y + tree.trunk_h;

        let canopy_y0 = top_y - (vpm / 3);
        let canopy_y1 = top_y + (tree.canopy_h * 3) / 4;

        let y0 = canopy_y0.max(out.origin[1]);
        let y1 = canopy_y1.min(out.origin[1] + out.size - 1);
        if y0 > y1 {
            return;
        }

        let (wx, wz) = self.trunk_wobble(tree, top_y);
        let cx0 = tree.tx as f32 + wx;
        let cz0 = tree.tz as f32 + wz;
        let cy0 = top_y as f32 + 0.30 * (tree.canopy_h as f32);

        let base_r = (tree.crown_r as f32) * 0.90;

        // Conservative AABB in XZ
        let max_r = base_r * 1.20;
        let minx = (cx0 - max_r).floor() as i32;
        let maxx = (cx0 + max_r).ceil() as i32;
        let minz = (cz0 - max_r).floor() as i32;
        let maxz = (cz0 + max_r).ceil() as i32;

        let x0 = minx.max(out.origin[0]);
        let x1 = maxx.min(out.origin[0] + out.size - 1);
        let z0 = minz.max(out.origin[2]);
        let z1 = maxz.min(out.origin[2] + out.size - 1);
        if x0 > x1 || z0 > z1 {
            return;
        }

        // 4 lumpy spheres
        let mut spheres = [(0.0f32, 0.0f32, 0.0f32, 0.0f32); 4];
        for i in 0..4u32 {
            let rr = hash_u32(tree.seed ^ 0xC100 ^ i.wrapping_mul(0x9E37_79B9));
            let ang = u01(rr) * std::f32::consts::TAU;

            let rad = u01(hash_u32(rr ^ 0x10)) * (0.60 * base_r);
            let cx = cx0 + ang.cos() * rad;
            let cz = cz0 + ang.sin() * rad;
            let cy = cy0 + (u01(hash_u32(rr ^ 0x20)) - 0.5) * (0.50 * tree.canopy_h as f32);

            let cr = (0.48 + 0.28 * u01(hash_u32(rr ^ 0x30))) * base_r;
            spheres[i as usize] = (cx, cy, cz, cr);
        }

        let max_r2 = max_r * max_r;
        let trunk_clear = 0.45 * (vpm as f32);
        let trunk_clear2 = trunk_clear * trunk_clear;

        for y in y0..=y1 {
            let py = y as f32;

            for x in x0..=x1 {
                let px = x as f32;
                let dx0 = px - cx0;
                let dx02 = dx0 * dx0;

                for z in z0..=z1 {
                    // cheap “gate” (cuts fill-rate)
                    let gate = hash3(self.seed ^ 0xA11C_E5ED, x, y, z) & 3;
                    if gate != 0 {
                        continue;
                    }

                    let pz = z as f32;
                    let dz0 = pz - cz0;
                    let d0 = dx02 + dz0 * dz0;

                    if d0 > max_r2 {
                        continue;
                    }

                    let mut best_nd: f32 = 2.0;
                    let mut hit = false;

                    for (cx, cy, cz, cr) in spheres {
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
                        continue;
                    }

                    if best_nd < 0.80 {
                        let n = hash3(self.seed ^ 0xCAFE_BABE, x, y, z);
                        if (n & 63) != 0 {
                            continue;
                        }
                    } else {
                        let n = hash3(self.seed ^ 0xD00D_F00D, x, y, z);
                        if (n & 7) == 0 {
                            continue;
                        }
                    }

                    // trunk clear-ish
                    if d0 < trunk_clear2 {
                        let n2 = hash3(self.seed ^ 0x5151_5151, x, y, z);
                        if (n2 & 15) != 0 {
                            continue;
                        }
                    }

                    out.write_leaf(x - out.origin[0], y - out.origin[1], z - out.origin[2]);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Old per-point query API (kept for compatibility)
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

            let r = tree.crown_r + 2 * config::VOXELS_PER_METER;
            if x < tree.tx - r || x > tree.tx + r || z < tree.tz - r || z > tree.tz + r {
                continue;
            }
            if y < tree.base_y || y > top_y + tree.canopy_h + config::VOXELS_PER_METER {
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
        }

        AIR
    }

    /// Used by the SVO builder to stamp conservative tree-top bounds.
    pub fn tree_instance_at_meter(&self, xm: i32, zm: i32) -> Option<(i32, i32)> {
        let (_seed, trunk_h_vox, crown_r_vox) = self.tree_at_meter_cell(xm, zm)?;
        Some((trunk_h_vox, crown_r_vox))
    }
}
