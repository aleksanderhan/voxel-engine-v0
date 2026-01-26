// src/world/tunnels.rs

use std::sync::atomic::{AtomicBool, Ordering};

use crate::config;
use super::{
    generator::WorldGen,
    hash::{hash2, hash_u32, u01},
};

#[derive(Clone, Copy, Debug)]
struct TunnelSeg {
    ax: f32, ay: f32, az: f32,
    bx: f32, by: f32, bz: f32,
    r0: f32,
    r1: f32,
}

/// Simple chunk-local bitmask (voxel resolution).
/// Stores occupancy for a single chunk volume [ox..ox+side), [oy..oy+side), [oz..oz+side).
#[derive(Clone, Debug)]
struct ChunkMask {
    ox: i32,
    oy: i32,
    oz: i32,
    side: i32,
    bits: Vec<u32>, // bitset: 1 = tunnel-carved
}

impl ChunkMask {
    fn new(ox: i32, oy: i32, oz: i32, side: i32) -> Self {
        let n = (side as usize) * (side as usize) * (side as usize);
        let words = (n + 31) / 32;
        Self { ox, oy, oz, side, bits: vec![0u32; words] }
    }

    #[inline]
    fn idx(&self, x: i32, y: i32, z: i32) -> Option<usize> {
        let lx = x - self.ox;
        let ly = y - self.oy;
        let lz = z - self.oz;
        if lx < 0 || ly < 0 || lz < 0 || lx >= self.side || ly >= self.side || lz >= self.side {
            return None;
        }
        let side = self.side as usize;
        let i = (ly as usize) * side * side + (lz as usize) * side + (lx as usize);
        Some(i)
    }

    #[inline]
    fn set(&mut self, x: i32, y: i32, z: i32) {
        if let Some(i) = self.idx(x, y, z) {
            let w = i >> 5;
            let b = i & 31;
            self.bits[w] |= 1u32 << b;
        }
    }

    #[inline]
    fn get(&self, x: i32, y: i32, z: i32) -> bool {
        let Some(i) = self.idx(x, y, z) else { return false; };
        let w = i >> 5;
        let b = i & 31;
        (self.bits[w] >> b) & 1u32 == 1u32
    }
}

/// Per-chunk cache of tunnel segments that might affect material queries.
pub struct TunnelCache {
    segs: Vec<TunnelSeg>,

    // Spatial bins for fast point queries
    grid_origin: [i32; 3],   // world vox
    cell: i32,               // world vox per cell
    dims: [i32; 3],          // number of cells in each axis
    bins: Vec<Vec<u16>>,     // bins[i] holds indices into segs

    // Optional chunk-local mask for O(1) contains tests
    mask: Option<ChunkMask>,
}

impl TunnelCache {
    /// Conservative (fast) stamp into (x,z) arrays storing carve bottom/top in world-voxel Y.
    /// Arrays are chunk-footprint indexed by local (lx,lz).
    pub fn stamp_carve_bounds(
        &self,
        chunk_ox: i32,
        chunk_oz: i32,
        chunk_size: i32,
        carve_bottom: &mut [i32],
        carve_top: &mut [i32],
    ) {
        let side = chunk_size as usize;
        debug_assert_eq!(carve_bottom.len(), side * side);
        debug_assert_eq!(carve_top.len(), side * side);

        for s in &self.segs {
            let r = s.r0.max(s.r1).ceil() as i32;

            let minx = s.ax.min(s.bx).floor() as i32 - r;
            let maxx = s.ax.max(s.bx).ceil() as i32 + r;
            let minz = s.az.min(s.bz).floor() as i32 - r;
            let maxz = s.az.max(s.bz).ceil() as i32 + r;

            // Clamp to chunk footprint.
            let x0 = minx.max(chunk_ox);
            let x1 = maxx.min(chunk_ox + chunk_size - 1);
            let z0 = minz.max(chunk_oz);
            let z1 = maxz.min(chunk_oz + chunk_size - 1);
            if x0 > x1 || z0 > z1 { continue; }

            let rr = s.r0.max(s.r1);
            let ylo = (s.ay.min(s.by) - rr).floor() as i32;
            let yhi = (s.ay.max(s.by) + rr).ceil() as i32;

            for wz in z0..=z1 {
                let lz = (wz - chunk_oz) as usize;
                for wx in x0..=x1 {
                    let lx = (wx - chunk_ox) as usize;
                    let idx = lz * side + lx;
                    carve_bottom[idx] = carve_bottom[idx].min(ylo);
                    carve_top[idx] = carve_top[idx].max(yhi);
                }
            }
        }
    }

    /// Exact test using spatial bins (your old behavior).
    #[inline]
    pub fn contains_point(&self, x: i32, y: i32, z: i32) -> bool {
        let [ox, oy, oz] = self.grid_origin;
        let [dx, dy, dz] = self.dims;
        let c = self.cell;

        let ix = (x - ox).div_euclid(c);
        let iy = (y - oy).div_euclid(c);
        let iz = (z - oz).div_euclid(c);

        if ix < 0 || iy < 0 || iz < 0 || ix >= dx || iy >= dy || iz >= dz {
            return false;
        }

        let idx = (iz * dy * dx + iy * dx + ix) as usize;
        let px = x as f32 + 0.5;
        let py = y as f32 + 0.5;
        let pz = z as f32 + 0.5;

        for &si in &self.bins[idx] {
            let s = &self.segs[si as usize];

            let (d2, t) = dist2_point_segment(px, py, pz, s.ax, s.ay, s.az, s.bx, s.by, s.bz);
            let rr = s.r0 + (s.r1 - s.r0) * t;
            if d2 <= rr * rr {
                return true;
            }
        }

        false
    }

    /// Fast test: if we have a chunk-local mask, use it; else fallback to exact bins.
    #[inline]
    pub fn contains_point_fast(&self, x: i32, y: i32, z: i32) -> bool {
        if let Some(m) = &self.mask {
            return m.get(x, y, z);
        }
        self.contains_point(x, y, z)
    }

    /// Build a chunk-local voxel mask for tunnels (expensive once, cheap queries later).
    fn build_mask_for_chunk(&self, chunk_ox: i32, chunk_oy: i32, chunk_oz: i32, chunk_size: i32, cancel: Option<&AtomicBool>) -> ChunkMask {
        let mut mask = ChunkMask::new(chunk_ox, chunk_oy, chunk_oz, chunk_size);

        let cx0 = chunk_ox;
        let cy0 = chunk_oy;
        let cz0 = chunk_oz;
        let cx1 = chunk_ox + chunk_size - 1;
        let cy1 = chunk_oy + chunk_size - 1;
        let cz1 = chunk_oz + chunk_size - 1;

        for (si, s) in self.segs.iter().enumerate() {
            if let Some(c) = cancel {
                if (si & 7) == 0 && c.load(Ordering::Relaxed) {
                    break;
                }
            }

            let rr = s.r0.max(s.r1);

            // segment AABB expanded by radius (in voxel coords)
            let minx = (s.ax.min(s.bx) - rr).floor() as i32;
            let maxx = (s.ax.max(s.bx) + rr).ceil() as i32;
            let miny = (s.ay.min(s.by) - rr).floor() as i32;
            let maxy = (s.ay.max(s.by) + rr).ceil() as i32;
            let minz = (s.az.min(s.bz) - rr).floor() as i32;
            let maxz = (s.az.max(s.bz) + rr).ceil() as i32;

            let x0 = minx.max(cx0);
            let x1 = maxx.min(cx1);
            let y0 = miny.max(cy0);
            let y1 = maxy.min(cy1);
            let z0 = minz.max(cz0);
            let z1 = maxz.min(cz1);

            if x0 > x1 || y0 > y1 || z0 > z1 {
                continue;
            }

            for z in z0..=z1 {
                if let Some(c) = cancel {
                    if (z & 15) == 0 && c.load(Ordering::Relaxed) {
                        return mask;
                    }
                }
                let pz = z as f32 + 0.5;
                for y in y0..=y1 {
                    let py = y as f32 + 0.5;
                    for x in x0..=x1 {
                        let px = x as f32 + 0.5;

                        let (d2, t) = dist2_point_segment(px, py, pz, s.ax, s.ay, s.az, s.bx, s.by, s.bz);
                        let rad = s.r0 + (s.r1 - s.r0) * t;

                        if d2 <= rad * rad {
                            mask.set(x, y, z);
                        }
                    }
                }
            }
        }

        mask
    }
}

#[inline]
fn dist2_point_segment(
    px: f32, py: f32, pz: f32,
    ax: f32, ay: f32, az: f32,
    bx: f32, by: f32, bz: f32,
) -> (f32, f32) {
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

impl WorldGen {
    /// Old API: builds segs + bins only (no voxel mask).
    pub fn build_tunnel_cache<F: Fn(i32, i32) -> i32>(
        &self,
        chunk_ox: i32,
        chunk_oz: i32,
        chunk_size: i32,
        height_at: &F,
    ) -> TunnelCache {
        self.build_tunnel_cache_impl(chunk_ox, 0, chunk_oz, chunk_size, height_at, None, false)
    }

    /// New API: builds segs + bins + chunk-local voxel mask (fast queries).
    pub fn build_tunnel_cache_with_mask<F: Fn(i32, i32) -> i32>(
        &self,
        chunk_ox: i32,
        chunk_oy: i32,
        chunk_oz: i32,
        chunk_size: i32,
        height_at: &F,
        cancel: &AtomicBool,
    ) -> TunnelCache {
        self.build_tunnel_cache_impl(chunk_ox, chunk_oy, chunk_oz, chunk_size, height_at, Some(cancel), true)
    }

    fn build_tunnel_cache_impl<F: Fn(i32, i32) -> i32>(
        &self,
        chunk_ox: i32,
        chunk_oy: i32,
        chunk_oz: i32,
        chunk_size: i32,
        height_at: &F,
        cancel: Option<&AtomicBool>,
        want_mask: bool,
    ) -> TunnelCache {
        let vpm = config::VOXELS_PER_METER;
        let pad_m = 10; // include starts nearby

        let xm0 = chunk_ox.div_euclid(vpm) - pad_m;
        let xm1 = (chunk_ox + chunk_size).div_euclid(vpm) + pad_m;
        let zm0 = chunk_oz.div_euclid(vpm) - pad_m;
        let zm1 = (chunk_oz + chunk_size).div_euclid(vpm) + pad_m;

        let cx0 = chunk_ox;
        let cx1 = chunk_ox + chunk_size - 1;
        let cz0 = chunk_oz;
        let cz1 = chunk_oz + chunk_size - 1;

        // Keep tunnel walk from roaming too far (avoids lots of slow height fallback)
        let roam = chunk_size * 4;
        let roam_x0 = chunk_ox - roam;
        let roam_x1 = chunk_ox + chunk_size + roam;
        let roam_z0 = chunk_oz - roam;
        let roam_z1 = chunk_oz + chunk_size + roam;

        let mut segs: Vec<TunnelSeg> = Vec::new();

        for zm in zm0..=zm1 {
            if let Some(c) = cancel {
                if ((zm - zm0) & 7) == 0 && c.load(Ordering::Relaxed) {
                    break;
                }
            }

            for xm in xm0..=xm1 {
                let r = hash2(self.seed ^ 0x6A09_E667, xm, zm);

                // density per 1m cell
                if (r % 96) != 0 {
                    continue;
                }

                let base = height_at(xm * vpm, zm * vpm);

                // target depth: 12..45m below ground
                let depth_m = 12 + (hash_u32(r ^ 0xBEEF) % 34) as i32;
                let mut y = base - depth_m * vpm;

                let mut x = xm * vpm;
                let mut z = zm * vpm;

                let steps = 8 + (hash_u32(r ^ 0x1234) % 9) as i32;

                let mut yaw = u01(hash_u32(r ^ 0x2222)) * std::f32::consts::TAU;

                for i in 0..steps {
                    let s0 = hash_u32(r ^ 0x9E37_79B9 ^ (i as u32).wrapping_mul(97));
                    let s1 = hash_u32(r ^ 0x85EB_CA6B ^ (i as u32).wrapping_mul(193));

                    let dy_m = (u01(s0) - 0.5) * 1.2;  // meters
                    let turn = (u01(s1) - 0.5) * 0.9;
                    yaw += turn;

                    // 8..18m segment length
                    let len_m = 8.0 + u01(hash_u32(s0 ^ 0x1111)) * 10.0;
                    let len = len_m * vpm as f32;

                    let dx = yaw.cos() * len;
                    let dz = yaw.sin() * len;
                    let dy = dy_m * vpm as f32;

                    let ax = x as f32;
                    let ay = y as f32;
                    let az = z as f32;

                    let bx = ax + dx;
                    let by = ay + dy;
                    let bz = az + dz;

                    // radius 0.9..2.4m
                    let r_m = 0.9 + u01(hash_u32(s1 ^ 0x3333)) * 1.5;
                    let rad = (r_m * vpm as f32).max(2.0);

                    // keep underground bias: clamp to [ground-70m .. ground-0.5m] (or 4m if not open)
                    let ground_here = height_at(x, z);
                    let open = (hash_u32(s0 ^ 0x0FEC_1A7E) & 3) == 0;

                    let ceiling = if open { ground_here - (vpm / 2) } else { ground_here - 4 * vpm };
                    let floor   = ground_here - 70 * vpm;
                    let clamped_by = by.clamp(floor as f32, ceiling as f32);

                    let seg = TunnelSeg {
                        ax, ay, az,
                        bx, by: clamped_by, bz,
                        r0: rad,
                        r1: (rad * (0.85 + 0.30 * u01(hash_u32(s0 ^ 0x4444)))).max(2.0),
                    };

                    // AABB intersect chunk footprint expanded by radius
                    let rr = seg.r0.max(seg.r1).ceil() as i32;
                    let minx = seg.ax.min(seg.bx).floor() as i32 - rr;
                    let maxx = seg.ax.max(seg.bx).ceil() as i32 + rr;
                    let minz = seg.az.min(seg.bz).floor() as i32 - rr;
                    let maxz = seg.az.max(seg.bz).ceil() as i32 + rr;

                    if !(maxx < cx0 || minx > cx1 || maxz < cz0 || minz > cz1) {
                        segs.push(seg);
                    }

                    // advance head (clamp roam in XZ)
                    x = seg.bx.round() as i32;
                    y = seg.by.round() as i32;
                    z = seg.bz.round() as i32;

                    x = x.clamp(roam_x0, roam_x1);
                    z = z.clamp(roam_z0, roam_z1);
                }
            }
        }

        // -------------------------
        // Build spatial bins
        // -------------------------
        // Cell size: 16 vox = 1.6m
        let cell: i32 = 16;

        // Use the same roam AABB for bins (XZ)
        let gx0 = roam_x0;
        let gx1 = roam_x1;
        let gz0 = roam_z0;
        let gz1 = roam_z1;

        // Vertical band for tunnels
        let gy0: i32 = -2048;
        let gy1: i32 =  2048;

        let dim_x = ((gx1 - gx0) + cell - 1).div_euclid(cell).max(1);
        let dim_y = ((gy1 - gy0) + cell - 1).div_euclid(cell).max(1);
        let dim_z = ((gz1 - gz0) + cell - 1).div_euclid(cell).max(1);

        let total = (dim_x * dim_y * dim_z) as usize;
        let mut bins: Vec<Vec<u16>> = vec![Vec::new(); total];

        let bin_index = |ix: i32, iy: i32, iz: i32| -> usize {
            (iz * dim_y * dim_x + iy * dim_x + ix) as usize
        };

        for (si, s) in segs.iter().enumerate() {
            let rr = s.r0.max(s.r1).ceil() as i32;
            let minx = s.ax.min(s.bx).floor() as i32 - rr;
            let maxx = s.ax.max(s.bx).ceil() as i32 + rr;
            let miny = s.ay.min(s.by).floor() as i32 - rr;
            let maxy = s.ay.max(s.by).ceil() as i32 + rr;
            let minz = s.az.min(s.bz).floor() as i32 - rr;
            let maxz = s.az.max(s.bz).ceil() as i32 + rr;

            let ix0 = ((minx - gx0).div_euclid(cell)).clamp(0, dim_x - 1);
            let ix1 = ((maxx - gx0).div_euclid(cell)).clamp(0, dim_x - 1);
            let iy0 = ((miny - gy0).div_euclid(cell)).clamp(0, dim_y - 1);
            let iy1 = ((maxy - gy0).div_euclid(cell)).clamp(0, dim_y - 1);
            let iz0 = ((minz - gz0).div_euclid(cell)).clamp(0, dim_z - 1);
            let iz1 = ((maxz - gz0).div_euclid(cell)).clamp(0, dim_z - 1);

            for iz in iz0..=iz1 {
                for iy in iy0..=iy1 {
                    for ix in ix0..=ix1 {
                        bins[bin_index(ix, iy, iz)].push(si as u16);
                    }
                }
            }
        }

        let mut cache = TunnelCache {
            segs,
            grid_origin: [gx0, gy0, gz0],
            cell,
            dims: [dim_x, dim_y, dim_z],
            bins,
            mask: None,
        };

        if want_mask {
            // Build voxel mask for this chunk volume.
            let m = cache.build_mask_for_chunk(chunk_ox, chunk_oy, chunk_oz, chunk_size, cancel);
            cache.mask = Some(m);
        }

        cache
    }
}
