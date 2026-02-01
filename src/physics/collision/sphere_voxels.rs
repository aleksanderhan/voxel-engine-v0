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

#[inline]
fn signum_nonzero(x: f32) -> f32 {
    if x >= 0.0 { 1.0 } else { -1.0 }
}

fn segment_aabb_hit(p0: Vec3, p1: Vec3, bmin: Vec3, bmax: Vec3) -> Option<(f32, Vec3)> {
    let d = p1 - p0;

    let mut tmin = 0.0f32;
    let mut tmax = 1.0f32;

    let mut best_axis = 0usize;
    let mut best_sign = 1.0f32;

    for axis in 0..3 {
        let (p, dir, minv, maxv) = match axis {
            0 => (p0.x, d.x, bmin.x, bmax.x),
            1 => (p0.y, d.y, bmin.y, bmax.y),
            _ => (p0.z, d.z, bmin.z, bmax.z),
        };

        if dir.abs() < 1e-9 {
            // Parallel: must be inside slab
            if p < minv || p > maxv { return None; }
            continue;
        }

        let inv = 1.0 / dir;

        // Pick near/far so that t_near <= t_far ALWAYS, and compute correct normal sign.
        // If inv >= 0, near plane is min slab, outward normal is -axis.
        // If inv < 0, near plane is max slab, outward normal is +axis.
        let (t_near, t_far, nsign) = if inv >= 0.0 {
            ((minv - p) * inv, (maxv - p) * inv, -1.0)
        } else {
            ((maxv - p) * inv, (minv - p) * inv,  1.0)
        };

        if t_near > tmin {
            tmin = t_near;
            best_axis = axis;
            best_sign = nsign;
        }

        tmax = tmax.min(t_far);
        if tmin > tmax { return None; }
    }

    if tmin < 0.0 || tmin > 1.0 { return None; }

    let n = match best_axis {
        0 => Vec3::new(best_sign, 0.0, 0.0),
        1 => Vec3::new(0.0, best_sign, 0.0),
        _ => Vec3::new(0.0, 0.0, best_sign),
    };

    Some((tmin, n))
}



/// 3D DDA traversal over the voxel grid for a segment.
/// At each visited cell, we test a small neighbor cube of voxels for intersection with the *expanded* AABB.
/// Returns earliest hit fraction t in [0,1] and normal.
fn raycast_sphere_grid<W: WorldQuery>(
    world: &W,
    p0: Vec3,
    p1: Vec3,
    radius: f32,
) -> Option<(f32, Vec3)> {
    let s = world.voxel_size_m();
    let d = p1 - p0;
    let len2 = d.length_squared();
    if len2 < 1e-12 {
        return None;
    }

    // Convert position -> voxel coords
    let to_v = |x: f32| (x / s).floor() as i32;

    // Start cell
    let mut vx = to_v(p0.x);
    let mut vy = to_v(p0.y);
    let mut vz = to_v(p0.z);

    // DDA setup
    let dir = d; // not normalized; we step in param t in [0,1]
    let step_x = if dir.x > 0.0 { 1 } else if dir.x < 0.0 { -1 } else { 0 };
    let step_y = if dir.y > 0.0 { 1 } else if dir.y < 0.0 { -1 } else { 0 };
    let step_z = if dir.z > 0.0 { 1 } else if dir.z < 0.0 { -1 } else { 0 };

    // next boundary in world coordinates
    let next_boundary = |v: i32, step: i32| -> f32 {
        if step > 0 {
            (v as f32 + 1.0) * s
        } else {
            (v as f32) * s
        }
    };

    // tMax = param t at which we cross the first boundary on each axis
    // tDelta = how far in t between crossings
    let mut tmax_x = if step_x != 0 {
        let bx = next_boundary(vx, step_x);
        (bx - p0.x) / dir.x
    } else {
        f32::INFINITY
    };
    let mut tmax_y = if step_y != 0 {
        let by = next_boundary(vy, step_y);
        (by - p0.y) / dir.y
    } else {
        f32::INFINITY
    };
    let mut tmax_z = if step_z != 0 {
        let bz = next_boundary(vz, step_z);
        (bz - p0.z) / dir.z
    } else {
        f32::INFINITY
    };

    let tdelta_x = if step_x != 0 { s / dir.x.abs() } else { f32::INFINITY };
    let tdelta_y = if step_y != 0 { s / dir.y.abs() } else { f32::INFINITY };
    let tdelta_z = if step_z != 0 { s / dir.z.abs() } else { f32::INFINITY };

    // How many neighbor voxels around current cell we need to test due to expansion.
    // Usually radius ~= 1-2 voxels, so this is small.
    let k = (radius / s).ceil() as i32;

    // Safety cap on traversal steps (prevents pathological loops)
    let max_steps = ((d.length() / s).ceil() as i32 + 8).clamp(8, 2048);

    let mut best_t = 2.0f32;
    let mut best_n = Vec3::ZERO;

    // Traverse cells along the segment
    let mut t = 0.0f32;

    for _ in 0..max_steps {
        // Stop once we've passed segment end
        if t > 1.0 {
            break;
        }

        // Test voxels in a small neighborhood around current cell:
        // For each solid voxel, build its world AABB, expand by radius, and segment test.
        for oz in -k..=k {
            for oy in -k..=k {
                for ox in -k..=k {
                    let cx = vx + ox;
                    let cy = vy + oy;
                    let cz = vz + oz;

                    if !world.solid_voxel_at(cx, cy, cz) {
                        continue;
                    }

                    let (bmin, bmax) = world.voxel_aabb_world(cx, cy, cz);

                    // Expand by radius (swept sphere -> point vs expanded box)
                    let ebmin = bmin - Vec3::splat(radius);
                    let ebmax = bmax + Vec3::splat(radius);

                    if let Some((thit, nhit)) = segment_aabb_hit(p0, p1, ebmin, ebmax) {
                        if thit < best_t {
                            best_t = thit;
                            best_n = nhit;
                        }
                    }
                }
            }
        }

        // Early out if we found a hit extremely close
        if best_t <= t + 1e-5 {
            break;
        }

        // Step to next voxel boundary along smallest tmax
        if tmax_x < tmax_y && tmax_x < tmax_z {
            vx += step_x;
            t = tmax_x;
            tmax_x += tdelta_x;
        } else if tmax_y < tmax_z {
            vy += step_y;
            t = tmax_y;
            tmax_y += tdelta_y;
        } else {
            vz += step_z;
            t = tmax_z;
            tmax_z += tdelta_z;
        }
    }

    if best_t <= 1.0 {
        Some((best_t, best_n))
    } else {
        None
    }
}

/// CCD sweep for a moving sphere over dt.
/// - Finds earliest hit against voxels (as expanded AABBs)
/// - Applies bounce/slide response
/// - Supports a few impacts per step
pub fn sweep_sphere_vs_voxels<W: WorldQuery>(
    world: &W,
    mut pos: Vec3,
    mut vel: Vec3,
    radius: f32,
    dt: f32,
    max_impacts: u32,
    restitution: f32,
) -> (Vec3, Vec3, bool) {
    let mut on_ground = false;

    let mut remaining = dt;
    let skin = 1e-4f32; // tiny push to avoid immediate re-hit

    for _ in 0..max_impacts {
        if remaining <= 0.0 {
            break;
        }

        let p0 = pos;
        let p1 = pos + vel * remaining;

        if let Some((t, n)) = raycast_sphere_grid(world, p0, p1, radius) {
            // Move to contact point
            pos = p0 + (p1 - p0) * t;

            // Nudge out
            pos += n * skin;

            // Velocity response (same idea as your old resolver)
            let vn = vel.dot(n);
            if vn < 0.0 {
                // reflect with restitution on normal component
                vel -= n * (vn * (1.0 + restitution));
            }

            if n.y > 0.6 {
                on_ground = true;
            }

            // Consume time
            remaining *= (1.0 - t).max(0.0);

            // If we are barely moving, stop
            if vel.length_squared() < 1e-10 {
                break;
            }
        } else {
            // No hit: move full remaining
            pos = p1;
            break;
        }
    }

    (pos, vel, on_ground)
}
