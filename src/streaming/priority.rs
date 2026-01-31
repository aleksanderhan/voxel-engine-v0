// src/streaming/priority.rs
use glam::{Vec2, Vec3};
use super::types::ChunkKey;

pub fn priority_score(k: ChunkKey, center: ChunkKey, cam_fwd: Vec3) -> f32 {
    let mut f = Vec2::new(cam_fwd.x, cam_fwd.z);
    if f.length_squared() > 1e-6 {
        f = f.normalize();
    } else {
        f = Vec2::ZERO;
    }
    chunk_priority_score(k, center, f)
}

// keep this function as-is, but make it visible inside the module file
fn chunk_priority_score(k: ChunkKey, c: ChunkKey, fwd_xz: Vec2) -> f32 {
    let dx = (k.x - c.x) as f32;
    let dz = (k.z - c.z) as f32;
    let dy = (k.y - c.y) as f32;

    let base = dx.abs() + dz.abs() + 2.0 * dy.abs();
    let dir = dx * fwd_xz.x + dz * fwd_xz.y;

    let front_bonus = 0.75;
    let behind_penalty = 0.25;

    let bias = if dir >= 0.0 { -front_bonus * dir } else { -behind_penalty * dir };

    base + bias
}
