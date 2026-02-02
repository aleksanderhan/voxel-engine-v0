// src/physics/player.rs
use glam::Vec3;
use crate::physics::collision::WorldContact;

/// A simple dynamic body for now (you'll later replace/extend this with an articulated body).
#[derive(Clone, Copy, Debug)]
pub struct PlayerBody {
    pub pos: Vec3,      // meters
    pub vel: Vec3,      // m/s
    pub radius: f32,    // meters (sphere collider)
    pub on_ground: bool,
    pub jump_queued: bool,
    pub contact: WorldContact,
}

impl PlayerBody {
    pub fn new(pos: Vec3) -> Self {
        Self {
            pos,
            vel: Vec3::ZERO,
            radius: 0.40,
            on_ground: false,
            jump_queued: false,
            contact: WorldContact::default(),
        }
    }
}

/// Movement/physics tuning parameters.
#[derive(Clone, Copy, Debug)]
pub struct PlayerTuning {
    pub speed_mps: f32,
    pub sprint_mul: f32,

    pub gravity_mps2: f32,

    pub accel_ground: f32,
    pub accel_air: f32,
    pub friction_ground: f32,

    pub jump_speed_mps: f32,

    /// How much velocity to keep after collision along the normal (0 = fully remove into-surface component).
    pub normal_restitution: f32,

    /// How many push-out iterations we do per step (more = more stable in corners, but slower).
    pub solver_iters: u32,
}

impl Default for PlayerTuning {
    fn default() -> Self {
        Self {
            speed_mps: 2.5,
            sprint_mul: 3.0,
            gravity_mps2: -9.81,
            accel_ground: 45.0,
            accel_air: 12.0,
            friction_ground: 14.0,
            jump_speed_mps: 5.5,
            normal_restitution: 0.0,
            solver_iters: 6,
        }
    }
}