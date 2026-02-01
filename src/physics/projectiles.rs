use glam::Vec3;

use crate::physics::collision::{sphere_voxels::sweep_sphere_vs_voxels, WorldQuery};

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
    pub gravity_mps2: f32,      // negative for downward if you follow your player tuning convention
    pub restitution: f32,
    pub solver_iters: u32,
    pub lifetime_s: f32,
}

impl Default for BallTuning {
    fn default() -> Self {
        Self {
            speed_mps: 22.0,
            gravity_mps2: -9.81,
            restitution: 0.25,
            solver_iters: 4,
            lifetime_s: 8.0,
        }
    }
}

pub fn step_balls<W: WorldQuery>(balls: &mut [Ball], world: &W, tuning: BallTuning, dt: f32) {
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

        // integrate
        let pos_pred = b.pos + b.vel * dt;

        // collide
        // CCD collide (sweep)
        let (p2, v2, _on_ground) = sweep_sphere_vs_voxels(
            world,
            b.pos,
            b.vel,
            b.radius,
            dt,
            /*max_impacts=*/ 4,
            tuning.restitution,
        );

        b.pos = p2;
        b.vel = v2;


        // cheap “sleep” kill if it basically stopped
        if b.vel.length_squared() < 0.01 * 0.01 {
            b.alive = false;
        }
    }
}
