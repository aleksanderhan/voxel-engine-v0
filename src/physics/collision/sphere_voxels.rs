// src/physics/collision/sphere_voxels.rs
use glam::Vec3;

use super::{IVec3i, WorldQuery};

#[inline]
fn closest_point_aabb(p: Vec3, bmin: Vec3, bmax: Vec3) -> Vec3 {
    Vec3::new(
        p.x.clamp(bmin.x, bmax.x),
        p.y.clamp(bmin.y, bmax.y),
        p.z.clamp(bmin.z, bmax.z),
    )
}

#[inline]
fn sphere_voxel_aabb(pos: Vec3, r: f32, voxel_size: f32) -> (IVec3i, IVec3i) {
    let minp = pos - Vec3::splat(r);
    let maxp = pos + Vec3::splat(r);

    let to_v = |x: f32| (x / voxel_size).floor() as i32;

    (
        IVec3i { x: to_v(minp.x), y: to_v(minp.y), z: to_v(minp.z) },
        IVec3i { x: to_v(maxp.x), y: to_v(maxp.y), z: to_v(maxp.z) },
    )
}

/// Resolve a moving sphere against solid voxel AABBs.
/// Returns (new_pos, new_vel, on_ground).
///
/// Notes:
/// - Iterative "push-out" for stability.
/// - Removes velocity component into the surface normal.
/// - on_ground when contact normal has strong +Y component.
pub fn resolve_sphere_vs_voxels<W: WorldQuery>(
    world: &W,
    mut pos: Vec3,
    mut vel: Vec3,
    radius: f32,
    solver_iters: u32,
    normal_restitution: f32,
) -> (Vec3, Vec3, bool) {
    let mut on_ground = false;
    let s = world.voxel_size_m();

    for _ in 0..solver_iters {
        let (min_v, max_v) = sphere_voxel_aabb(pos, radius, s);

        let mut worst_pen = 0.0f32;
        let mut worst_n = Vec3::ZERO;

        // Scan nearby voxels.
        for vz in min_v.z..=max_v.z {
            for vy in min_v.y..=max_v.y {
                for vx in min_v.x..=max_v.x {
                    if !world.solid_voxel_at(vx, vy, vz) {
                        continue;
                    }

                    let (bmin, bmax) = world.voxel_aabb_world(vx, vy, vz);

                    let c = closest_point_aabb(pos, bmin, bmax);
                    let d = pos - c;
                    let dist2 = d.length_squared();

                    if dist2 < radius * radius {
                        let dist = dist2.sqrt();
                        let pen = radius - dist;

                        // Choose a stable normal.
                        let n = if dist > 1e-6 {
                            d / dist
                        } else {
                            // Very rare: sphere center exactly equals closest point.
                            // Pick axis based on position relative to voxel center.
                            let center = 0.5 * (bmin + bmax);
                            let a = pos - center;
                            let ax = a.x.abs();
                            let ay = a.y.abs();
                            let az = a.z.abs();
                            if ay >= ax && ay >= az {
                                Vec3::new(0.0, a.y.signum(), 0.0)
                            } else if ax >= az {
                                Vec3::new(a.x.signum(), 0.0, 0.0)
                            } else {
                                Vec3::new(0.0, 0.0, a.z.signum())
                            }
                        };

                        if pen > worst_pen {
                            worst_pen = pen;
                            worst_n = n;
                        }
                    }
                }
            }
        }

        // No overlaps found.
        if worst_pen <= 0.0 {
            break;
        }

        // Push out of penetration.
        pos += worst_n * worst_pen;

        // Velocity response: remove component into surface normal, with optional restitution.
        let vn = vel.dot(worst_n);
        if vn < 0.0 {
            // normal restitution: 0 => cancel vn, 1 => bounce back fully
            let keep = -vn * normal_restitution;
            vel -= worst_n * (vn - keep);
        }

        // Ground check (normal has strong +Y).
        if worst_n.y > 0.6 {
            on_ground = true;
        }
    }

    (pos, vel, on_ground)
}
