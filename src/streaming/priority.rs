// src/streaming/priority.rs
use glam::{Vec2, Vec3};
use super::types::ChunkKey;

/// Priority score where *smaller is better*.
/// This version is **concentric-shell first**:
/// - primary key: squared distance (XZ + weighted Y)
/// - secondary: tiny camera-forward bias that cannot override a 1-step shell difference
pub fn priority_score(k: ChunkKey, center: ChunkKey, cam_fwd: Vec3) -> f32 {
    let mut f = Vec2::new(cam_fwd.x, cam_fwd.z);
    if f.length_squared() > 1e-6 {
        f = f.normalize();
    } else {
        f = Vec2::ZERO;
    }
    chunk_priority_score_concentric(k, center, f)
}

/// Concentric distance in 3D-ish (XZ + heavier Y), with a tiny directional tie-break.
/// IMPORTANT: the directional term is clamped to < 0.5 so it can never “jump shells”
/// when the base distance changes by >= 1.
fn chunk_priority_score_concentric(k: ChunkKey, c: ChunkKey, fwd_xz: Vec2) -> f32 {
    let dx = (k.x - c.x) as f32;
    let dz = (k.z - c.z) as f32;
    let dy = (k.y - c.y) as f32;

    // Primary: squared distance. Weight Y more so vertical offsets don’t “skip” nearby XZ.
    // (You already weight Y in other places; this keeps behavior consistent.)
    let dist2 = dx * dx + dz * dz + 4.0 * dy * dy;

    // Secondary: prefer chunks in front *within the same shell*.
    // Normalize by distance so this stays in [-1..1], then scale to < 0.5.
    let dist = dist2.sqrt();
    let dir = dx * fwd_xz.x + dz * fwd_xz.y; // [-|xz| .. |xz|]
    let dir_norm = if dist > 1e-3 { (dir / dist).clamp(-1.0, 1.0) } else { 0.0 };

    // Max magnitude 0.49 => cannot outrank a chunk that is 1 unit closer in dist2.
    let dir_bias = -0.49 * dir_norm;

    dist2 + dir_bias
}
