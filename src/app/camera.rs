// src/camera.rs
use glam::{Mat4, Vec3};

use crate::app::config;
use crate::app::input::InputState;

pub struct Camera {
    pos: Vec3,
    yaw: f32,
    pitch: f32,
    fovy_rad: f32,
    z_near: f32,
    z_far: f32,
    // movement tuning
    speed_mps: f32,
    mouse_sens: f32,
    sprint_mul: f32,
}

pub struct CameraFrame {
    pub view_inv: Mat4,
    pub proj_inv: Mat4,
    pub pos: Vec3,
}

impl Camera {
    pub fn new(aspect: f32) -> Self {
        let _ = aspect; // kept for future (if you want aspect-dependent params)
        Self {
            pos: Vec3::new((config::CHUNK_SIZE as f32 * config::VOXEL_SIZE_M_F32) * 0.5, 20.0, -20.0),
            yaw: 0.0,
            pitch: 0.15,
            fovy_rad: 60.0_f32.to_radians(),
            z_near: 0.1,
            z_far: 1000.0,
            speed_mps: 2.5,
            mouse_sens: 0.0025,
            sprint_mul: 3.0
        }
    }

    pub fn position(&self) -> Vec3 {
        self.pos
    }

    pub fn forward(&self) -> glam::Vec3 {
        let (yaw, pitch) = (self.yaw, self.pitch);
        glam::Vec3::new(
            yaw.sin() * pitch.cos(),
            pitch.sin(),
            yaw.cos() * pitch.cos(),
        )
        .normalize()
    }

    pub fn integrate_input(&mut self, input: &mut InputState, dt: f32) {
        // mouse look
        if input.focused {
            let (dx, dy) = input.take_mouse_delta();
            self.yaw -= dx * self.mouse_sens;
            self.pitch = (self.pitch - dy * self.mouse_sens).clamp(-1.55, 1.55);
        } else {
            let _ = input.take_mouse_delta();
        }

        // basis
        let forward = Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        )
        .normalize();

        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward).normalize();

        // movement intent
        let k = input.keys;
        let mut vel = Vec3::ZERO;
        if k.w { vel += forward; }
        if k.s { vel -= forward; }
        if k.d { vel += right; }
        if k.a { vel -= right; }
        if k.space { vel += up; }
        if k.alt { vel -= up; }

        if vel.length_squared() > 0.0 {
            let speed = if k.shift { self.speed_mps * self.sprint_mul } else { self.speed_mps };
            self.pos += vel.normalize() * (speed * dt);
        }

    }


    pub fn frame_matrices(&self, aspect: f32) -> CameraFrame {
        let forward = Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        )
        .normalize();

        let view = Mat4::look_at_rh(self.pos, self.pos + forward, Vec3::Y);
        let proj = Mat4::perspective_rh(self.fovy_rad, aspect, self.z_near, self.z_far);

        CameraFrame {
            view_inv: view.inverse(),
            proj_inv: proj.inverse(),
            pos: self.pos,
        }
    }

    pub fn set_position(&mut self, p: glam::Vec3) { self.pos = p; }
    pub fn set_yaw_pitch(&mut self, yaw: f32, pitch: f32) { self.yaw = yaw; self.pitch = pitch; }
    pub fn yaw_pitch(&self) -> (f32, f32) { (self.yaw, self.pitch) }
}
