// src/physics/step.rs
use glam::Vec3;

use crate::app::InputState;

use crate::app::config;
use super::{
    collision::{sphere_voxels::resolve_dynamic_voxel_vs_static_voxels, WorldQuery},
    player::{PlayerBody, PlayerTuning},
};
use super::projectiles::{DynamicVoxel, DynamicVoxelTuning, step_voxels};


/// Fixed-step physics driver.
/// Owns the player body for now; later this owns articulations too.
pub struct Physics {
    pub fixed_dt: f32,
    accum: f32,

    pub player: PlayerBody,
    pub tuning: PlayerTuning,

    // previous state for interpolation
    prev_player_pos: Vec3,
    prev_player_vel: Vec3,

    // view orientation (keep in physics so camera becomes a pure view-projection thing)
    pub yaw: f32,
    pub pitch: f32,

    pub mouse_sens: f32,
    pub eye_offset: Vec3,

    pub voxels: Vec<DynamicVoxel>,
    pub voxel_tuning: DynamicVoxelTuning,
}

impl Physics {
    pub fn new(player_start: Vec3) -> Self {
        let player = PlayerBody::new(player_start);
        Self {
            fixed_dt: 1.0 / 120.0,
            accum: 0.0,
            player: PlayerBody::new(player_start),
            tuning: PlayerTuning::default(),
            prev_player_pos: player.pos,
            prev_player_vel: player.vel,
            yaw: 0.0,
            pitch: 0.15,
            mouse_sens: 0.0025,
            eye_offset: Vec3::new(0.0, 1.6, 0.0),
            voxels: Vec::new(),  
            voxel_tuning: DynamicVoxelTuning::default(),
        }
    }

    pub fn step_frame_player_only<W: WorldQuery>(&mut self, dt_frame: f32, world: &W) {
        self.accum += dt_frame.min(0.05);

        while self.accum >= self.fixed_dt {
            self.prev_player_pos = self.player.pos;
            self.prev_player_vel = self.player.vel;

            self.step_fixed_with_desired(Vec3::ZERO, world, self.fixed_dt);
            self.accum -= self.fixed_dt;
        }
    }

    /// Call once per rendered frame; internally advances fixed steps.
    ///
    /// Returns the recommended camera "eye" position in world meters.
    pub fn step_frame<W: WorldQuery>(
        &mut self,
        input: &mut InputState,
        dt_frame: f32,
        world: &W,
    ) -> Vec3 {
        // 1) mouse look -> yaw/pitch (once per rendered frame)
        if input.focused {
            let (dx, dy) = input.take_mouse_delta();
            self.yaw -= dx * self.mouse_sens;
            self.pitch = (self.pitch - dy * self.mouse_sens).clamp(-1.55, 1.55);
        } else {
            let _ = input.take_mouse_delta();
        }

        // 2) cache frame intent ONCE (deterministic across fixed ticks)
        let desired = self.desired_move_velocity(input);

        // 3) queue jump intent (still “hold to try jump”, but sampled once per frame)
        if input.keys.space {
            self.player.jump_queued = true;
        }

        // 4) fixed-step loop
        self.accum += dt_frame.min(0.05);

        while self.accum >= self.fixed_dt {
            // snapshot BEFORE advancing one tick (for interpolation)
            self.prev_player_pos = self.player.pos;
            self.prev_player_vel = self.player.vel;

            self.step_fixed_with_desired(desired, world, self.fixed_dt);
            self.accum -= self.fixed_dt;
        }

        // 5) return interpolated camera eye for rendering
        self.interpolated_eye()
    }


    fn desired_move_velocity(&self, input: &InputState) -> Vec3 {
        let forward = Vec3::new(self.yaw.sin(), 0.0, self.yaw.cos()).normalize();
        let right = forward.cross(Vec3::Y).normalize();

        let k = input.keys;
        let mut wish = Vec3::ZERO;
        if k.w { wish += forward; }
        if k.s { wish -= forward; }
        if k.d { wish += right; }
        if k.a { wish -= right; }

        if wish.length_squared() > 0.0 {
            let speed = if k.shift {
                self.tuning.speed_mps * self.tuning.sprint_mul
            } else {
                self.tuning.speed_mps
            };
            wish.normalize() * speed
        } else {
            Vec3::ZERO
        }
    }


    fn step_fixed<W: WorldQuery>(&mut self, input: &InputState, world: &W, dt: f32) {
        let desired = self.desired_move_velocity(input);

        let mut vel = self.player.vel;

        // Split horizontal vs vertical
        let hvel = Vec3::new(vel.x, 0.0, vel.z);

        // Accel toward desired
        let accel = if self.player.on_ground {
            self.tuning.accel_ground
        } else {
            self.tuning.accel_air
        };

        let dv = desired - hvel;
        let max_dv = accel * dt;
        let dv_len = dv.length();
        let dv_clamped = if dv_len > max_dv && dv_len > 1e-6 {
            dv * (max_dv / dv_len)
        } else {
            dv
        };

        vel.x += dv_clamped.x;
        vel.z += dv_clamped.z;

        // Ground friction when no input
        if self.player.on_ground && desired.length_squared() == 0.0 {
            let drop = self.tuning.friction_ground * dt;
            let sp = hvel.length();
            let newsp = (sp - drop).max(0.0);
            if sp > 1e-5 {
                let scale = newsp / sp;
                vel.x *= scale;
                vel.z *= scale;
            }
        }

        // Jump
        if self.player.on_ground && self.player.jump_queued {
            vel.y = self.tuning.jump_speed_mps;
            self.player.jump_queued = false;
            self.player.on_ground = false;
        }

        // Gravity
        vel.y += self.tuning.gravity_mps2 * dt;

        // Integrate
        let pos_pred = self.player.pos + vel * dt;

        // Collide
        let (pos2, vel2, on_ground) = resolve_dynamic_voxel_vs_static_voxels(
            world,
            pos_pred,
            vel,
            self.player.radius,
            self.tuning.solver_iters,
            self.tuning.normal_restitution,
        );

        self.player.pos = pos2;
        self.player.vel = vel2;
        self.player.on_ground = on_ground;

        // If we landed, clear queued jump (prevents “auto-bunnyhop” from holding space)
        if self.player.on_ground {
            self.player.jump_queued = false;
        }

        // step voxels after player
        step_voxels(&mut self.voxels, world, self.voxel_tuning, dt);

        // optional: compact dead voxels occasionally
        if self.voxels.len() > 256 {
            self.voxels.retain(|b| b.alive);
        }

    }

    fn step_fixed_with_desired<W: WorldQuery>(
        &mut self,
        desired: Vec3,
        world: &W,
        dt: f32,
    ) {
        let mut vel = self.player.vel;

        let hvel = Vec3::new(vel.x, 0.0, vel.z);

        let accel = if self.player.on_ground {
            self.tuning.accel_ground
        } else {
            self.tuning.accel_air
        };

        let dv = desired - hvel;
        let max_dv = accel * dt;
        let dv_len = dv.length();
        let dv_clamped = if dv_len > max_dv && dv_len > 1e-6 {
            dv * (max_dv / dv_len)
        } else {
            dv
        };

        vel.x += dv_clamped.x;
        vel.z += dv_clamped.z;

        if self.player.on_ground && desired.length_squared() == 0.0 {
            let drop = self.tuning.friction_ground * dt;
            let sp = hvel.length();
            let newsp = (sp - drop).max(0.0);
            if sp > 1e-5 {
                let scale = newsp / sp;
                vel.x *= scale;
                vel.z *= scale;
            }
        }

        if self.player.on_ground && self.player.jump_queued {
            vel.y = self.tuning.jump_speed_mps;
            self.player.jump_queued = false;
            self.player.on_ground = false;
        }

        vel.y += self.tuning.gravity_mps2 * dt;

        let pos_pred = self.player.pos + vel * dt;

        let (pos2, vel2, on_ground) = resolve_dynamic_voxel_vs_static_voxels(
            world,
            pos_pred,
            vel,
            self.player.radius,
            self.tuning.solver_iters,
            self.tuning.normal_restitution,
        );

        self.player.pos = pos2;
        self.player.vel = vel2;
        self.player.on_ground = on_ground;

        if self.player.on_ground {
            self.player.jump_queued = false;
        }

        // step voxels after player
        step_voxels(&mut self.voxels, world, self.voxel_tuning, dt);

        // optional: compact dead voxels occasionally
        if self.voxels.len() > 256 {
            self.voxels.retain(|b| b.alive);
        }
    }

    pub fn spawn_voxel(&mut self, eye: Vec3, forward: Vec3) {
        let f = if forward.length_squared() > 1e-6 { forward.normalize() } else { Vec3::Z };

        let r = config::VOXEL_SIZE_M_F32 / 2.0;

        let pos = eye + f * (r + config::VOXEL_SPAWN_NUDGE_M);

        self.voxel_tuning.speed_mps = config::VOXEL_SPEED_MPS;
        self.voxel_tuning.lifetime_s = config::VOXEL_LIFETIME_S;

        self.voxels.push(DynamicVoxel {
            pos,
            vel: f * self.voxel_tuning.speed_mps,
            radius: r,
            age: 0.0,
            alive: true,
        });
    }

    #[inline]
    pub fn voxels_iter(&self) -> impl Iterator<Item = &super::projectiles::DynamicVoxel> {
        self.voxels.iter().filter(|b| b.alive)
    }

    /// Iterate alive voxels (mutable), if you ever need it.
    #[inline]
    pub fn voxels_iter_mut(&mut self) -> impl Iterator<Item = &mut super::projectiles::DynamicVoxel> {
        self.voxels.iter_mut().filter(|b| b.alive)
    }

    /// Optional helper: alive count for GPU uniforms etc.
    #[inline]
    pub fn voxels_alive_count(&self) -> u32 {
        self.voxels_iter().count() as u32
    }

    #[inline]
    pub fn render_alpha(&self) -> f32 {
        (self.accum / self.fixed_dt).clamp(0.0, 1.0)
    }

    #[inline]
    pub fn interpolated_eye(&self) -> Vec3 {
        let a = self.render_alpha();
        let p = self.prev_player_pos.lerp(self.player.pos, a);
        p + self.eye_offset
    }

}
