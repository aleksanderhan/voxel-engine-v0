use std::collections::HashMap;
use glam::{IVec3, Vec3};

use crate::app::config;

use crate::physics::collision::{
    sphere_voxels::{sweep_dynamic_voxel_vs_static_voxels, resolve_dynamic_voxel_vs_static_voxels},
    WorldQuery
};


#[derive(Clone, Copy, Debug)]
pub struct DynamicVoxel {
    pub pos: Vec3,      // meters
    pub vel: Vec3,      // m/s
    pub radius: f32,    // meters
    pub age: f32,       // seconds
    pub alive: bool,
}

// Minimal tuning for now.
#[derive(Clone, Copy, Debug)]
pub struct DynamicVoxelTuning {
    pub speed_mps: f32,
    pub gravity_mps2: f32,
    pub restitution: f32,        // dynamic voxel <-> static voxels
    pub solver_iters: u32,       // dynamic voxel <-> static voxels (max impacts for sweep)
    pub lifetime_s: f32,

    // voxel <-> voxel
    pub voxel_voxel_restitution: f32,
    pub voxel_voxel_iters: u32,
    pub voxel_mass: f32,

    pub ccd_min_dist_m: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct DistanceConstraint {
    pub a: usize,
    pub b: usize,
    pub rest_len: f32,
    pub compliance: f32, // 0 = rigid, >0 = soft (XPBD)
    pub lambda: f32,     // XPBD accumulator
}

pub struct VoxelCluster {
    pub voxels: Vec<DynamicVoxel>,
    pub constraints: Vec<DistanceConstraint>,
    pub alive: bool,
}



impl Default for DynamicVoxelTuning {
    fn default() -> Self {
        Self {
            speed_mps: 22.0,
            gravity_mps2: -9.81,
            restitution: 0.0,
            solver_iters: 4,
            lifetime_s: 8.0,
            voxel_voxel_restitution: 0.6,
            voxel_voxel_iters: 48,
            voxel_mass: 1.0,
            ccd_min_dist_m: 0.10,
        }
    }
}

pub fn step_voxels<W: WorldQuery>(voxels: &mut [DynamicVoxel], world: &W, tuning: DynamicVoxelTuning, dt: f32) {
    let max_impacts = tuning.solver_iters.max(1);

    // Keep your original sleep threshold
    let sleep_speed2 = 0.01f32 * 0.01f32;

    // pass 1: integrate + collide vs voxels
    for b in voxels.iter_mut() {
        if !b.alive {
            continue;
        }

        b.age += dt;
        if b.age > tuning.lifetime_s {
            b.alive = false;
            continue;
        }

        // gravity
        b.vel.y += tuning.gravity_mps2 * dt;

        // choose CCD only when needed
        let travel = b.vel.length() * dt;
        if travel >= tuning.ccd_min_dist_m {
            let (p2, v2, _on_ground) = sweep_dynamic_voxel_vs_static_voxels(
                world,
                b.pos,
                b.vel,
                b.radius,
                dt,
                max_impacts,
                tuning.restitution,
            );
            b.pos = p2;
            b.vel = v2;
        } else {
            // cheap discrete path
            let pos_pred = b.pos + b.vel * dt;
            let (p2, v2, _on_ground) = resolve_dynamic_voxel_vs_static_voxels(
                world,
                pos_pred,
                b.vel,
                b.radius,
                2,
                tuning.restitution,
            );
            b.pos = p2;
            b.vel = v2;
        }
    }

    // pass 2: voxel <-> voxel (broadphase grid version)
    solve_voxel_voxel_grid(
        voxels,
        tuning.voxel_mass,
        tuning.voxel_voxel_restitution,
        tuning.voxel_voxel_iters.max(1),
    );

    // pass 3: sleep/kill
    for b in voxels.iter_mut() {
        if !b.alive {
            continue;
        }
        if b.vel.length_squared() < sleep_speed2 {
            b.alive = false;
        }
    }
}


#[inline]
fn cell_key(p: Vec3, cell: f32) -> (i32, i32, i32) {
    let inv = 1.0 / cell;
    ((p.x * inv).floor() as i32, (p.y * inv).floor() as i32, (p.z * inv).floor() as i32)
}

fn solve_voxel_voxel_grid(voxels: &mut [DynamicVoxel], mass: f32, restitution: f32, iters: u32) {
    if voxels.len() < 2 { return; }

    let slop = 0.001;
    let percent = 0.8;
    let inv_m = if mass > 0.0 { 1.0 / mass } else { 0.0 };

    // Pick a cell size: diameter of the *largest* voxel (all yours are same radius)
    let cell = (voxels[0].radius * 2.0).max(1e-3);

    // Bucket indices
    let mut buckets: HashMap<(i32,i32,i32), Vec<usize>> = HashMap::new();
    buckets.reserve(voxels.len() * 2);

    for (idx, b) in voxels.iter().enumerate() {
        if !b.alive { continue; }
        buckets.entry(cell_key(b.pos, cell)).or_default().push(idx);
    }

    // Helper: narrowphase+impulse for a pair
    let mut resolve_pair = |i: usize, j: usize, voxels: &mut [DynamicVoxel]| {
        if !voxels[i].alive || !voxels[j].alive { return; }

        let pi = voxels[i].pos;
        let pj = voxels[j].pos;
        let rij = voxels[i].radius + voxels[j].radius;

        let d = pj - pi;
        let dist2 = d.length_squared();
        if dist2 >= rij * rij { return; }

        let (n, dist) = if dist2 > 1e-12 {
            let dist = dist2.sqrt();
            (d / dist, dist)
        } else {
            (Vec3::Y, 0.0)
        };

        // positional correction
        let pen = rij - dist;
        let corr_mag = ((pen - slop).max(0.0)) * percent;
        if corr_mag > 0.0 {
            let corr = n * (corr_mag * 0.5);
            voxels[i].pos -= corr;
            voxels[j].pos += corr;
        }

        // impulse
        let rel = voxels[j].vel - voxels[i].vel;
        let vn = rel.dot(n);
        if vn < 0.0 {
            let denom = inv_m + inv_m;
            if denom > 0.0 {
                let j_imp = -(1.0 + restitution) * vn / denom;
                let impulse = n * j_imp;
                voxels[i].vel -= impulse * inv_m;
                voxels[j].vel += impulse * inv_m;
            }
        }
    };

    // Iterate a few times for stability
    for _ in 0..iters {
        // For each occupied cell, test against itself + 26 neighbors
        for (&(cx,cy,cz), ids) in buckets.iter() {
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let nk = (cx+dx, cy+dy, cz+dz);
                        let Some(other) = buckets.get(&nk) else { continue; };

                        // pair indices from ids x other, but avoid duplicates
                        for &i in ids {
                            for &j in other {
                                if j <= i { continue; }
                                resolve_pair(i, j, voxels);
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn solve_distance_constraints_xpbd(
    voxels: &mut [DynamicVoxel],
    constraints: &mut [DistanceConstraint],
    inv_mass: f32,
    dt: f32,
    iters: u32,
) {
    if inv_mass <= 0.0 || dt <= 0.0 { return; }

    // XPBD: alpha = compliance / dt^2
    let alpha = |c: f32| c / (dt * dt);

    for _ in 0..iters {
        for c in constraints.iter_mut() {
            if !voxels[c.a].alive || !voxels[c.b].alive { continue; }

            let pa = voxels[c.a].pos;
            let pb = voxels[c.b].pos;

            let d = pb - pa;
            let len2 = d.length_squared();
            if len2 < 1e-12 { continue; }

            let len = len2.sqrt();
            let n = d / len;

            // C = |pb-pa| - rest
            let C = len - c.rest_len;

            // w = inv_ma + inv_mb  (same mass)
            let w = inv_mass + inv_mass;

            let a = alpha(c.compliance);
            let dl = (-C - a * c.lambda) / (w + a);
            c.lambda += dl;

            // Δx_i = -w_i * dl * ∇C ; ∇C = n
            let corr = n * dl;

            voxels[c.a].pos -= corr * inv_mass;
            voxels[c.b].pos += corr * inv_mass;
        }
    }
}



pub fn spawn_voxel_ball(
    eye: Vec3,
    forward: Vec3,
    voxel_size_m: f32,   // spacing between sub-voxels (also rest length for 6-neighbor links)
    speed_mps: f32,
    radius_vox: i32,
) -> VoxelCluster {
    let f = forward.normalize_or_zero();
    let sub_r = 0.5 * voxel_size_m;

    let center = eye + f * (sub_r + config::VOXEL_SPAWN_NUDGE_M);
    let vel = f * speed_mps;

    // 1) filled sphere offsets in voxel grid
    let mut offsets: Vec<IVec3> = Vec::new();
    let r2 = radius_vox * radius_vox;

    for z in -radius_vox..=radius_vox {
        for y in -radius_vox..=radius_vox {
            for x in -radius_vox..=radius_vox {
                if x*x + y*y + z*z <= r2 {
                    offsets.push(IVec3::new(x, y, z));
                }
            }
        }
    }

    // offset -> index
    let mut idx_of: HashMap<IVec3, usize> = HashMap::with_capacity(offsets.len());

    let mut voxels: Vec<DynamicVoxel> = Vec::with_capacity(offsets.len());
    for (i, off) in offsets.iter().enumerate() {
        idx_of.insert(*off, i);
        voxels.push(DynamicVoxel {
            pos: center + off.as_vec3() * voxel_size_m,
            vel,
            radius: sub_r,
            age: 0.0,
            alive: true,
        });
    }

    // 2) constraints: 6-neighbor links
    let mut constraints = Vec::new();
    let compliance = config::BALL_COMPLIANCE;

    let edge = voxel_size_m;
    let diag = voxel_size_m * 2.0_f32.sqrt();
    let body = voxel_size_m * 3.0_f32.sqrt();

    let dirs: &[(IVec3, f32)] = &[
        // 6 edges
        (IVec3::new( 1, 0, 0), edge),
        (IVec3::new(-1, 0, 0), edge),
        (IVec3::new( 0, 1, 0), edge),
        (IVec3::new( 0,-1, 0), edge),
        (IVec3::new( 0, 0, 1), edge),
        (IVec3::new( 0, 0,-1), edge),

        // 12 face diagonals
        (IVec3::new( 1, 1, 0), diag),
        (IVec3::new( 1,-1, 0), diag),
        (IVec3::new(-1, 1, 0), diag),
        (IVec3::new(-1,-1, 0), diag),
        (IVec3::new( 1, 0, 1), diag),
        (IVec3::new( 1, 0,-1), diag),
        (IVec3::new(-1, 0, 1), diag),
        (IVec3::new(-1, 0,-1), diag),
        (IVec3::new( 0, 1, 1), diag),
        (IVec3::new( 0, 1,-1), diag),
        (IVec3::new( 0,-1, 1), diag),
        (IVec3::new( 0,-1,-1), diag),

        // 8 body diagonals  ✅
        (IVec3::new( 1, 1, 1), body),
        (IVec3::new( 1, 1,-1), body),
        (IVec3::new( 1,-1, 1), body),
        (IVec3::new( 1,-1,-1), body),
        (IVec3::new(-1, 1, 1), body),
        (IVec3::new(-1, 1,-1), body),
        (IVec3::new(-1,-1, 1), body),
        (IVec3::new(-1,-1,-1), body),
    ];


    for off in &offsets {
        let a = idx_of[off];
        for (d, rest_len) in dirs {
            let nb = *off + *d;
            if let Some(&b) = idx_of.get(&nb) {
                if b > a {
                    constraints.push(DistanceConstraint {
                        a, b,
                        rest_len: *rest_len,
                        compliance,
                        lambda: 0.0,
                    });
                }
            }
        }
    }

    VoxelCluster { voxels, constraints, alive: true }
}



pub fn step_cluster<W: WorldQuery>(
    cluster: &mut VoxelCluster,
    world: &W,
    tuning: DynamicVoxelTuning,
    dt: f32,
) {
    if !cluster.alive { return; }

    // snapshot BEFORE any integration/constraints
    let mut old = Vec::with_capacity(cluster.voxels.len());
    old.extend(cluster.voxels.iter().map(|v| v.pos));

    // 0) integrate positions (semi-implicit Euler)
    for v in cluster.voxels.iter_mut() {
        if !v.alive { continue; }

        v.age += dt;
        if v.age > tuning.lifetime_s {
            v.alive = false;
            continue;
        }

        // gravity
        v.vel.y += tuning.gravity_mps2 * dt;

        // integrate (NO collision here)
        v.pos += v.vel * dt;
    }

    // 1) XPBD + world contact projection in the SAME loop
    let inv_m = if tuning.voxel_mass > 0.0 { 1.0 / tuning.voxel_mass } else { 0.0 };
    let iters = tuning.voxel_voxel_iters.max(1);

    for _ in 0..iters {
        // (a) distance constraints: do ONE iteration per outer loop
        solve_distance_constraints_xpbd(
            &mut cluster.voxels,
            &mut cluster.constraints,
            inv_m,
            dt,
            1,
        );

        // (b) project out of the world (position-only)
        // Use your existing resolver with vel=0 and restitution=0 so it only pushes out.
        for v in cluster.voxels.iter_mut() {
            if !v.alive { continue; }

            let (p2, _v2, _on_ground) = resolve_dynamic_voxel_vs_static_voxels(
                world,
                v.pos,
                Vec3::ZERO,   // important: position projection only
                v.radius,
                2,            // small push-out iterations per projection
                0.0,          // no bounce during projection
            );
            v.pos = p2;
        }
    }

    // 2) recompute velocity from corrected positions
    if dt > 0.0 {
        for (i, v) in cluster.voxels.iter_mut().enumerate() {
            if !v.alive { continue; }
            v.vel = (v.pos - old[i]) / dt;
        }
    }

    if cluster.voxels.iter().all(|v| !v.alive) {
        cluster.alive = false;
    }
}
