// src/physics/step.rs
use glam::Vec3;

use crate::app::InputState;

use crate::app::config;
use super::{
    collision::{sphere_voxels::resolve_sphere_vs_voxels, WorldQuery},
    player::{PlayerBody, PlayerTuning},
};
use super::projectiles::{Ball, BallTuning, step_balls};


/// Fixed-step physics driver.
/// Owns the player body for now; later this owns articulations too.
pub struct Physics {
    pub fixed_dt: f32,
    accum: f32,

    pub player: PlayerBody,
    pub tuning: PlayerTuning,

    // view orientation (keep in physics so camera becomes a pure view-projection thing)
    pub yaw: f32,
    pub pitch: f32,

    pub mouse_sens: f32,
    pub eye_offset: Vec3,

    pub balls: Vec<Ball>,
    pub ball_tuning: BallTuning,
}

impl Physics {
    pub fn new(player_start: Vec3) -> Self {
        Self {
            fixed_dt: 1.0 / 120.0,
            accum: 0.0,
            player: PlayerBody::new(player_start),
            tuning: PlayerTuning::default(),
            yaw: 0.0,
            pitch: 0.15,
            mouse_sens: 0.0025,
            eye_offset: Vec3::new(0.0, 1.6, 0.0),
            balls: Vec::new(),  
            ball_tuning: BallTuning::default(),
        }
    }

    pub fn step_frame_player_only<W: WorldQuery>(&mut self, dt_frame: f32, world: &W) {
        self.accum += dt_frame.min(0.05);
        while self.accum >= self.fixed_dt {
            self.step_fixed_with_desired(Vec3::ZERO, false, world, self.fixed_dt);
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
        // 1) mouse look -> yaw/pitch
        if input.focused {
            let (dx, dy) = input.take_mouse_delta();
            self.yaw -= dx * self.mouse_sens;
            self.pitch = (self.pitch - dy * self.mouse_sens).clamp(-1.55, 1.55);
        } else {
            let _ = input.take_mouse_delta();
        }

        // 2) queue jump (space) on press; for now: space being held means "try jump"
        // If you want true edge-triggered jump, track previous key state.
        if input.keys.space {
            self.player.jump_queued = true;
        }

        // 3) fixed-step loop
        self.accum += dt_frame.min(0.05);
        while self.accum >= self.fixed_dt {
            self.step_fixed(input, world, self.fixed_dt);
            self.accum -= self.fixed_dt;
        }

        // camera eye position derived from player
        self.player.pos + self.eye_offset
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
        let (pos2, vel2, on_ground) = resolve_sphere_vs_voxels(
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

        // step balls after player
        step_balls(&mut self.balls, world, self.ball_tuning, dt);

        // optional: compact dead balls occasionally
        if self.balls.len() > 256 {
            self.balls.retain(|b| b.alive);
        }

    }

    fn step_fixed_with_desired<W: WorldQuery>(
        &mut self,
        desired: Vec3,
        _jump_pressed: bool,
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

        let (pos2, vel2, on_ground) = resolve_sphere_vs_voxels(
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

        // step balls after player
        step_balls(&mut self.balls, world, self.ball_tuning, dt);

        // optional: compact dead balls occasionally
        if self.balls.len() > 256 {
            self.balls.retain(|b| b.alive);
        }
    }

    pub fn spawn_ball(&mut self, eye: Vec3, forward: Vec3) {
        let f = if forward.length_squared() > 1e-6 { forward.normalize() } else { Vec3::Z };

        let r = config::BALL_RADIUS as f32 * config::VOXEL_SIZE_M_F32;

        let pos = eye + f * (r + config::BALL_SPAWN_NUDGE_M);

        self.ball_tuning.speed_mps = config::BALL_SPEED_MPS;
        self.ball_tuning.lifetime_s = config::BALL_LIFETIME_S;

        self.balls.push(Ball {
            pos,
            vel: f * self.ball_tuning.speed_mps,
            radius: r,
            age: 0.0,
            alive: true,
        });
    }

    #[inline]
    pub fn balls_iter(&self) -> impl Iterator<Item = &super::projectiles::Ball> {
        self.balls.iter().filter(|b| b.alive)
    }

    /// Iterate alive balls (mutable), if you ever need it.
    #[inline]
    pub fn balls_iter_mut(&mut self) -> impl Iterator<Item = &mut super::projectiles::Ball> {
        self.balls.iter_mut().filter(|b| b.alive)
    }

    /// Optional helper: alive count for GPU uniforms etc.
    #[inline]
    pub fn balls_alive_count(&self) -> u32 {
        self.balls_iter().count() as u32
    }

}
