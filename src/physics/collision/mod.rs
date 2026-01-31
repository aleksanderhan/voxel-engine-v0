// src/physics/collision/mod.rs
pub mod sphere_voxels;

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

/// Small integer 3D coordinate helper (avoids pulling in glam::IVec3).
#[derive(Clone, Copy, Debug)]
pub struct IVec3i {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}
    