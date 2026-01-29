// src/svo/raycast.rs
use glam::Vec3;
use crate::{config, render::NodeGpu, world::materials::AIR};

const LEAF: u32 = 0xFFFF_FFFF;

#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub t_m: f32,
    pub voxel: [i32; 3],
    pub normal: [i32; 3],
    pub material: u32,
    pub leaf_min: [i32; 3],
    pub leaf_size: i32,
}


#[inline]
fn popcount(x: u32) -> u32 {
    x.count_ones()
}

#[inline]
fn child_index(child_mask: u32, child_id: u32) -> Option<u32> {
    if (child_mask & (1u32 << child_id)) == 0 {
        return None;
    }
    let lower = child_mask & ((1u32 << child_id) - 1);
    Some(popcount(lower))
}

#[inline]
fn safe_inv(d: f32) -> f32 {
    let eps = 1e-12;
    if d.abs() < eps {
        1.0 / (eps * d.signum().max(1.0)) // treat 0 as +eps
    } else {
        1.0 / d
    }
}

#[inline]
fn ray_aabb(origin: Vec3, dir: Vec3, bmin: Vec3, bmax: Vec3) -> Option<(f32, f32)> {
    let inv = Vec3::new(safe_inv(dir.x), safe_inv(dir.y), safe_inv(dir.z));

    let t0 = (bmin - origin) * inv;
    let t1 = (bmax - origin) * inv;

    let tmin = Vec3::new(t0.x.min(t1.x), t0.y.min(t1.y), t0.z.min(t1.z));
    let tmax = Vec3::new(t0.x.max(t1.x), t0.y.max(t1.y), t0.z.max(t1.z));

    let entry = tmin.x.max(tmin.y).max(tmin.z);
    let exit  = tmax.x.min(tmax.y).min(tmax.z);

    if exit >= entry.max(0.0) { Some((entry, exit)) } else { None }
}


// Traverse one chunk SVO. All coordinates here are in *voxel units*.
pub fn raycast_chunk_svo_vox(
    nodes: &[NodeGpu],
    chunk_origin_vox: [i32; 3],
    ray_o_vox: Vec3,
    ray_d_vox: Vec3,
) -> Option<(f32, [i32;3], [i32;3], u32, [i32;3], i32)> {
    let cs = config::CHUNK_SIZE as i32;
    let bmin = Vec3::new(chunk_origin_vox[0] as f32, chunk_origin_vox[1] as f32, chunk_origin_vox[2] as f32);
    let bmax = bmin + Vec3::splat(cs as f32);

    let (t_entry, t_exit) = ray_aabb(ray_o_vox, ray_d_vox, bmin, bmax)?;
    if nodes.is_empty() {
        return None;
    }

    // stack of (node_index, node_min, size, t_entry, t_exit)
    let mut stack: Vec<(u32, Vec3, i32, f32, f32)> = Vec::with_capacity(64);
    stack.push((0, bmin, cs, t_entry, t_exit));

    let mut best: Option<(f32, [i32; 3], [i32; 3], u32, [i32; 3], i32)> = None;

    while let Some((ni, nmin, size, te, tx)) = stack.pop() {
        if let Some((bt, _, _, _, _, _)) = best {
            if te >= bt {
                continue;
            }
        }

        let n = nodes[ni as usize];

        // leaf?
        if n.child_base == LEAF {
            let mat = n.material;
            if mat == AIR {
                continue;
            }

            // Determine a “hit voxel” and normal from entry point.
            let p = ray_o_vox + ray_d_vox * te;
            let eps = 1e-4;

            let nmax = nmin + Vec3::splat(size as f32);

            // pick normal by which face was entered
            let mut normal = [0, 0, 0];
            if (p.x - nmin.x).abs() < eps { normal = [-1, 0, 0]; }
            else if (p.x - nmax.x).abs() < eps { normal = [ 1, 0, 0]; }
            else if (p.y - nmin.y).abs() < eps { normal = [ 0,-1, 0]; }
            else if (p.y - nmax.y).abs() < eps { normal = [ 0, 1, 0]; }
            else if (p.z - nmin.z).abs() < eps { normal = [ 0, 0,-1]; }
            else if (p.z - nmax.z).abs() < eps { normal = [ 0, 0, 1]; }

            let vx = p.x.floor() as i32;
            let vy = p.y.floor() as i32;
            let vz = p.z.floor() as i32;

            let leaf_min = [nmin.x as i32, nmin.y as i32, nmin.z as i32];
            let leaf_size = size;
            best = Some((te, [vx, vy, vz], normal, mat, leaf_min, leaf_size));
            continue;
        }

        // internal: push children near-first
        let half = size / 2;
        let base = n.child_base;
        let mask = n.child_mask;

        // We collect candidates then sort by entry t (small vector: up to 8)
        let mut kids: Vec<(f32, u32, Vec3, i32, f32, f32)> = Vec::with_capacity(8);

        for cid in 0..8u32 {
            let Some(koff) = child_index(mask, cid) else { continue; };
            let kidx = base + koff;

            let dx = if (cid & 1) != 0 { half } else { 0 };
            let dy = if (cid & 2) != 0 { half } else { 0 };
            let dz = if (cid & 4) != 0 { half } else { 0 };

            let kmin = nmin + Vec3::new(dx as f32, dy as f32, dz as f32);
            let kmax = kmin + Vec3::splat(half as f32);

            if let Some((ke, kx)) = ray_aabb(ray_o_vox, ray_d_vox, kmin, kmax) {
                // also clamp to parent interval
                let ke = ke.max(te);
                let kx = kx.min(tx);
                if kx >= ke.max(0.0) {
                    kids.push((ke, kidx, kmin, half, ke, kx));
                }
            }
        }

        kids.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // push reverse so nearest is popped first
        for (_, kidx, kmin, ksize, ke, kx) in kids.into_iter().rev() {
            stack.push((kidx, kmin, ksize, ke, kx));
        }
    }

    best
}
