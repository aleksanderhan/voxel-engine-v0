// src/camera.rs
// -------------
//
// dt-based camera movement (meters/second), not meters/frame.

use glam::{Mat4, Vec3};

use crate::input::InputState;

pub struct Camera {
    pos: Vec3,
    yaw: f32,
    pitch: f32,
    fovy_rad: f32,
    z_near: f32,
    z_far: f32,

    speed_mps: f32,
    mouse_sens: f32,
}

pub struct CameraFrame {
    pub view_inv: Mat4,
    pub proj_inv: Mat4,
    pub pos: Vec3,
}

impl Camera {
    pub fn new(_aspect: f32) -> Self {
        // NOTE:
        // Your streamed neighborhood is centered on the camera chunk.
        // Terrain heights are around ~[-10..+6] meters, so starting at y=20
        // streams only high-altitude chunks that are completely empty.
        // Start near the surface so initial chunks contain voxels.
        Self {
            pos: Vec3::new(0.0, 6.0, -10.0),
            yaw: 0.0,
            pitch: -0.20, // look slightly down so you see terrain immediately
            fovy_rad: 60.0_f32.to_radians(),
            z_near: 0.1,
            z_far: 1000.0,
            speed_mps: 12.0,
            mouse_sens: 0.0025,
        }
    }

    pub fn position(&self) -> Vec3 {
        self.pos
    }

    pub fn forward(&self) -> Vec3 {
        let (yaw, pitch) = (self.yaw, self.pitch);
        Vec3::new(
            yaw.sin() * pitch.cos(),
            pitch.sin(),
            yaw.cos() * pitch.cos(),
        )
        .normalize()
    }

    pub fn integrate_input(&mut self, input: &mut InputState, dt: f32) {
        if input.focused {
            let (dx, dy) = input.take_mouse_delta();
            self.yaw -= dx * self.mouse_sens;
            self.pitch = (self.pitch - dy * self.mouse_sens).clamp(-1.55, 1.55);
        } else {
            let _ = input.take_mouse_delta();
        }

        let forward = self.forward();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward).normalize();

        let k = input.keys;
        let mut vel = Vec3::ZERO;
        if k.w { vel += forward; }
        if k.s { vel -= forward; }
        if k.d { vel += right; }
        if k.a { vel -= right; }
        if k.space { vel += up; }
        if k.alt { vel -= up; }

        if vel.length_squared() > 0.0 {
            let dt = dt.clamp(0.0, 0.05);
            self.pos += vel.normalize() * self.speed_mps * dt;
        }
    }

    pub fn frame_matrices(&self, aspect: f32) -> CameraFrame {
        let forward = self.forward();
        let view = Mat4::look_at_rh(self.pos, self.pos + forward, Vec3::Y);
        let proj = Mat4::perspective_rh(self.fovy_rad, aspect, self.z_near, self.z_far);

        CameraFrame {
            view_inv: view.inverse(),
            proj_inv: proj.inverse(),
            pos: self.pos,
        }
    }
}
