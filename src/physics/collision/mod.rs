// src/physics/collision/mod.rs

use glam::Vec3;

/// Minimal query interface from physics -> your voxel world.
///
/// All coordinates for `solid_voxel_at` are in **voxel units** (integer grid).
/// World-space meters are used by the player body and camera.
pub trait WorldQuery {
    /// Size of one voxel edge in meters.
    fn voxel_size_m(&self) -> f32;

    /// Return true if that voxel is solid.
    fn solid_voxel_at(&self, vx: i32, vy: i32, vz: i32) -> bool;

    /// World-space AABB of the voxel (meters).
    #[inline]
    fn voxel_aabb_world(&self, vx: i32, vy: i32, vz: i32) -> (Vec3, Vec3) {
        let s = self.voxel_size_m();
        let bmin = Vec3::new(vx as f32 * s, vy as f32 * s, vz as f32 * s);
        let bmax = bmin + Vec3::splat(s);
        (bmin, bmax)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct WorldContact {
    pub cell: (i32, i32, i32), // static voxel coordinate (vx,vy,vz)
    pub n: Vec3,               // contact normal (world space)
    pub valid: bool,
}

impl Default for WorldContact {
    fn default() -> Self {
        Self { cell: (0,0,0), n: Vec3::ZERO, valid: false }
    }
}

const EPS: f32 = 1e-5;

/// Project point `p` onto AABB [bmin,bmax].
#[inline]
fn closest_point_on_aabb(p: Vec3, bmin: Vec3, bmax: Vec3) -> Vec3 {
    Vec3::new(
        p.x.clamp(bmin.x, bmax.x),
        p.y.clamp(bmin.y, bmax.y),
        p.z.clamp(bmin.z, bmax.z),
    )
}

/// Compute penetration normal + depth between sphere(center,r) and AABB.
/// Returns Some((normal, depth)) if overlapping.
fn sphere_aabb_contact(center: Vec3, r: f32, bmin: Vec3, bmax: Vec3) -> Option<(Vec3, f32)> {
    let q = closest_point_on_aabb(center, bmin, bmax);
    let d = center - q;
    let d2 = d.length_squared();

    if d2 > r * r {
        return None;
    }

    // If center is outside AABB, normal = (center - closest).normalized.
    if d2 > EPS * EPS {
        let dist = d2.sqrt();
        let n = d / dist;
        let depth = r - dist;
        return Some((n, depth));
    }

    // If center is inside AABB (or extremely close), pick the smallest push axis.
    // Distances to each face:
    let to_min = center - bmin;
    let to_max = bmax - center;

    // pick axis with minimal distance to face
    let (n, depth) = {
        // x-
        let mut best_n = -Vec3::X;
        let mut best_d = to_min.x;

        // x+
        if to_max.x < best_d { best_d = to_max.x; best_n = Vec3::X; }

        // y-
        if to_min.y < best_d { best_d = to_min.y; best_n = -Vec3::Y; }

        // y+
        if to_max.y < best_d { best_d = to_max.y; best_n = Vec3::Y; }

        // z-
        if to_min.z < best_d { best_d = to_min.z; best_n = -Vec3::Z; }

        // z+
        if to_max.z < best_d { best_d = to_max.z; best_n = Vec3::Z; }

        // When inside, we want to push sphere center out of AABB by (r + best_d).
        // best_d is distance from center to that face.
        (best_n, r + best_d)
    };

    Some((n, depth))
}

/// Resolve sphere collisions against solid voxels around `pos`.
/// Returns (new_pos, new_vel, on_ground, contact_normal, contact_cell_opt).
pub fn resolve_sphere_world<W: WorldQuery>(
    world: &W,
    mut pos: Vec3,
    mut vel: Vec3,
    radius: f32,
    solver_iters: u32,
    normal_restitution: f32,
) -> (Vec3, Vec3, bool, Vec3, Option<(i32, i32, i32)>) {
    let s = world.voxel_size_m();

    // Broadphase voxel range from sphere AABB.
    let bmin = pos - Vec3::splat(radius);
    let bmax = pos + Vec3::splat(radius);

    let vx0 = ((bmin.x / s).floor() as i32) - 1;
    let vy0 = ((bmin.y / s).floor() as i32) - 1;
    let vz0 = ((bmin.z / s).floor() as i32) - 1;

    let vx1 = ((bmax.x / s).floor() as i32) + 1;
    let vy1 = ((bmax.y / s).floor() as i32) + 1;
    let vz1 = ((bmax.z / s).floor() as i32) + 1;

    let mut on_ground = false;
    let mut best_n = Vec3::ZERO;
    let mut best_cell: Option<(i32, i32, i32)> = None;

    // Sequential impulse / push-out style solver.
    for _ in 0..solver_iters {
        let mut any = false;

        for vz in vz0..=vz1 {
            for vy in vy0..=vy1 {
                for vx in vx0..=vx1 {
                    if !world.solid_voxel_at(vx, vy, vz) {
                        continue;
                    }

                    let (cell_min, cell_max) = world.voxel_aabb_world(vx, vy, vz);

                    if let Some((n, depth)) = sphere_aabb_contact(pos, radius, cell_min, cell_max) {
                        any = true;

                        // Push out
                        pos += n * depth;

                        // Remove into-surface velocity component
                        let vn = vel.dot(n);
                        if vn < 0.0 {
                            // keep vn * restitution (0 = stick)
                            vel -= n * vn * (1.0 - normal_restitution);
                        }

                        // Grounding if the normal points up enough
                        if n.y > 0.7 {
                            on_ground = true;
                        }

                        // Track a representative contact
                        if depth > 0.0 && (best_n == Vec3::ZERO || n.y > best_n.y) {
                            best_n = n;
                            best_cell = Some((vx, vy, vz));
                        }
                    }
                }
            }
        }

        if !any {
            break;
        }
    }

    (pos, vel, on_ground, best_n, best_cell)
}
