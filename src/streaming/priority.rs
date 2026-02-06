// src/streaming/priority.rs
use glam::{Vec2, Vec3};
use super::types::ChunkKey;

use crate::streaming::ChunkManager;
use crate::streaming::manager::ground;

#[inline]
pub fn priority_score_streaming(mgr: &ChunkManager, k: ChunkKey, center: ChunkKey, cam_fwd: Vec3) -> f32 {
    // 1) primary: XZ distance only (don’t penalize elevation differences between columns)
    let dx = (k.x - center.x) as f32;
    let dz = (k.z - center.z) as f32;
    let dist2_xz = dx * dx + dz * dz;

    // 2) secondary: local vertical offset within the column band (usually small: -2..2 or 0..2)
    let ground_cy = ground::ground_cy_for_column(mgr, k.x, k.z).unwrap_or(center.y);
    let local_dy = (k.y - ground_cy) as f32;

    // Keep this small so it never beats a 1-step XZ shell difference.
    let y_term = 0.25 * local_dy * local_dy;

    // 3) tiny directional tie-break within same XZ shell
    let mut f = Vec2::new(cam_fwd.x, cam_fwd.z);
    if f.length_squared() > 1e-6 {
        f = f.normalize();
    } else {
        f = Vec2::ZERO;
    }

    let dist = dist2_xz.sqrt().max(1e-3);
    let dir = dx * f.x + dz * f.y;
    let dir_norm = (dir / dist).clamp(-1.0, 1.0);

    // keep < 0.5 so it can’t jump shells
    let dir_bias = -0.49 * dir_norm;

    dist2_xz + y_term + dir_bias
}
