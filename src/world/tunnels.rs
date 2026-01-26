// src/world/tunnels.rs
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

/// Per-chunk cache of tunnel segments that might affect material queries.
pub struct TunnelCache {
    segs: Vec<TunnelSeg>,
}

impl TunnelCache {
    pub fn segs(&self) -> &[TunnelSeg] { &self.segs }

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

            let ylo = (s.ay.min(s.by) - s.r0.max(s.r1)).floor() as i32;
            let yhi = (s.ay.max(s.by) + s.r0.max(s.r1)).ceil() as i32;

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

    #[inline]
    pub fn contains_point(&self, x: i32, y: i32, z: i32) -> bool {
        let px = x as f32;
        let py = y as f32;
        let pz = z as f32;

        for s in &self.segs {
            // Quick AABB reject in world voxels
            let r = s.r0.max(s.r1) + 1.0;
            let minx = s.ax.min(s.bx) - r;
            let maxx = s.ax.max(s.bx) + r;
            let miny = s.ay.min(s.by) - r;
            let maxy = s.ay.max(s.by) + r;
            let minz = s.az.min(s.bz) - r;
            let maxz = s.az.max(s.bz) + r;
            if px < minx || px > maxx || py < miny || py > maxy || pz < minz || pz > maxz {
                continue;
            }

            let (d2, t) = dist2_point_segment(px, py, pz, s.ax, s.ay, s.az, s.bx, s.by, s.bz);
            let rr = s.r0 + (s.r1 - s.r0) * t;
            if d2 <= rr * rr {
                return true;
            }
        }
        false
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
    /// Build a per-chunk tunnel cache (like TreeCache): deterministic segments that may intersect chunk.
    pub fn build_tunnel_cache<F: Fn(i32, i32) -> i32>(
        &self,
        chunk_ox: i32,
        chunk_oz: i32,
        chunk_size: i32,
        height_at: &F,
    ) -> TunnelCache {
        let vpm = config::VOXELS_PER_METER;
        let pad_m = 10; // include segments starting nearby

        let xm0 = chunk_ox.div_euclid(vpm) - pad_m;
        let xm1 = (chunk_ox + chunk_size).div_euclid(vpm) + pad_m;
        let zm0 = chunk_oz.div_euclid(vpm) - pad_m;
        let zm1 = (chunk_oz + chunk_size).div_euclid(vpm) + pad_m;

        let cx0 = chunk_ox;
        let cx1 = chunk_ox + chunk_size - 1;
        let cz0 = chunk_oz;
        let cz1 = chunk_oz + chunk_size - 1;

        let mut segs = Vec::new();

        for zm in zm0..=zm1 {
            for xm in xm0..=xm1 {
                let r = hash2(self.seed ^ 0x6A09_E667, xm, zm);

                // density per 1m cell
                if (r % 252) != 0 {
                    continue;
                }

                let base = height_at(xm * vpm, zm * vpm);

                // target depth: 10..35m below ground
                let depth_m = 12 + (hash_u32(r ^ 0xBEEF) % 34) as i32;
                let mut y = base - depth_m * vpm;

                let mut x = xm * vpm;
                let mut z = zm * vpm;

                // segments
                let steps = 8 + (hash_u32(r ^ 0x1234) % 9) as i32;

                // initial yaw
                let mut yaw = u01(hash_u32(r ^ 0x2222)) * std::f32::consts::TAU;

                for i in 0..steps {
                    // deterministic per-step variation
                    let s0 = hash_u32(r ^ 0x9E37_79B9 ^ (i as u32).wrapping_mul(97));
                    let s1 = hash_u32(r ^ 0x85EB_CA6B ^ (i as u32).wrapping_mul(193));

                    let dy_m = (u01(s0) - 0.5) * 1.2;        // small vertical drift (meters)
                    let turn = (u01(s1) - 0.5) * 0.9;        // yaw turn
                    yaw += turn;

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

                    // radius 1.2..3.5m, slowly varying
                    let r_m = 0.9 + u01(hash_u32(s1 ^ 0x3333)) * 1.5;
                    let rad = (r_m * vpm as f32).max(2.0);

                    // keep underground bias: if too high, nudge down
                    let ground_here = height_at(x, z);
                    let ceiling = ground_here - 4 * vpm;
                    let floor = ground_here - 55 * vpm;
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

                    // advance head
                    x = seg.bx.round() as i32;
                    y = seg.by.round() as i32;
                    z = seg.bz.round() as i32;
                }
            }
        }

        TunnelCache { segs }
    }
}
