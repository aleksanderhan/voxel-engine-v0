use glam::Vec3;
use std::collections::HashMap;

use crate::physics::collision::{
    sphere_voxels::{sweep_sphere_vs_voxels, resolve_sphere_vs_voxels},
    WorldQuery
};

#[derive(Clone, Copy, Debug)]
pub struct Ball {
    pub pos: Vec3,      // meters
    pub vel: Vec3,      // m/s
    pub radius: f32,    // meters
    pub age: f32,       // seconds
    pub alive: bool,
}

// Minimal tuning for now.
#[derive(Clone, Copy, Debug)]
pub struct BallTuning {
    pub speed_mps: f32,
    pub gravity_mps2: f32,
    pub restitution: f32,        // ball <-> voxels
    pub solver_iters: u32,       // ball <-> voxels (max impacts for sweep)
    pub lifetime_s: f32,

    // ball <-> ball
    pub ball_ball_restitution: f32,
    pub ball_ball_iters: u32,
    pub ball_mass: f32,

    pub ccd_min_dist_m: f32,
}


impl Default for BallTuning {
    fn default() -> Self {
        Self {
            speed_mps: 22.0,
            gravity_mps2: -9.81,
            restitution: 0.25,
            solver_iters: 4,
            lifetime_s: 8.0,
            ball_ball_restitution: 0.6,
            ball_ball_iters: 3,
            ball_mass: 1.0,
            ccd_min_dist_m: 0.10,
        }
    }
}

pub fn step_balls<W: WorldQuery>(balls: &mut [Ball], world: &W, tuning: BallTuning, dt: f32) {
    let max_impacts = tuning.solver_iters.max(1);

    // Keep your original sleep threshold
    let sleep_speed2 = 0.01f32 * 0.01f32;

    // pass 1: integrate + collide vs voxels
    for b in balls.iter_mut() {
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
            let (p2, v2, _on_ground) = sweep_sphere_vs_voxels(
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
            let (p2, v2, _on_ground) = crate::physics::collision::sphere_voxels::resolve_sphere_vs_voxels(
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

    // pass 2: ball <-> ball (broadphase grid version)
    solve_ball_ball_grid(
        balls,
        tuning.ball_mass,
        tuning.ball_ball_restitution,
        tuning.ball_ball_iters.max(1),
    );

    // pass 3: sleep/kill
    for b in balls.iter_mut() {
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

fn solve_ball_ball_grid(balls: &mut [Ball], mass: f32, restitution: f32, iters: u32) {
    if balls.len() < 2 { return; }

    let slop = 0.001;
    let percent = 0.8;
    let inv_m = if mass > 0.0 { 1.0 / mass } else { 0.0 };

    // Pick a cell size: diameter of the *largest* ball (all yours are same radius)
    let cell = (balls[0].radius * 2.0).max(1e-3);

    // Bucket indices
    let mut buckets: HashMap<(i32,i32,i32), Vec<usize>> = HashMap::new();
    buckets.reserve(balls.len() * 2);

    for (idx, b) in balls.iter().enumerate() {
        if !b.alive { continue; }
        buckets.entry(cell_key(b.pos, cell)).or_default().push(idx);
    }

    // Helper: narrowphase+impulse for a pair
    let mut resolve_pair = |i: usize, j: usize, balls: &mut [Ball]| {
        if !balls[i].alive || !balls[j].alive { return; }

        let pi = balls[i].pos;
        let pj = balls[j].pos;
        let rij = balls[i].radius + balls[j].radius;

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
            balls[i].pos -= corr;
            balls[j].pos += corr;
        }

        // impulse
        let rel = balls[j].vel - balls[i].vel;
        let vn = rel.dot(n);
        if vn < 0.0 {
            let denom = inv_m + inv_m;
            if denom > 0.0 {
                let j_imp = -(1.0 + restitution) * vn / denom;
                let impulse = n * j_imp;
                balls[i].vel -= impulse * inv_m;
                balls[j].vel += impulse * inv_m;
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
                                resolve_pair(i, j, balls);
                            }
                        }
                    }
                }
            }
        }
    }
}



