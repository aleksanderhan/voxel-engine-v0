use std::collections::HashMap;

use crate::config;

// Use whatever math type you already use. glam is common in wgpu projects.
use glam::{IVec3, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct VoxelBall {
    pub pos_m: Vec3,   // world position in meters
    pub vel_mps: Vec3, // velocity in meters/sec
    pub age_s: f32,
}

pub struct VoxelBallSystem {
    pub balls: Vec<VoxelBall>,

    // Sparse voxel overlay: world-voxel -> material
    // (Only stores the voxels that should be forced to MAT_BALL.)
    pub overlay: HashMap<IVec3, u32>,

    // Cache so we can clear/rebuild only when center voxel changes
    last_centers: Vec<IVec3>,
}

impl VoxelBallSystem {
    pub fn new() -> Self {
        Self {
            balls: Vec::new(),
            overlay: HashMap::new(),
            last_centers: Vec::new(),
        }
    }

    pub fn spawn(&mut self, cam_pos_m: Vec3, ray_dir: Vec3, voxel_size_m: f32) {
        let dir = ray_dir.normalize_or_zero();
        let pos = cam_pos_m + dir * config::BALL_SPAWN_NUDGE_M;
        let vel = dir * config::BALL_SPEED_MPS;

        self.balls.push(Ball { pos_m: pos, vel_mps: vel, age_s: 0.0 });
        self.last_centers.push(Self::world_to_vox(pos, voxel_size_m));
        // Force a rebuild next update
        self.rebuild_overlay(voxel_size_m);
    }

    pub fn update(&mut self, dt_s: f32, voxel_size_m: f32) {
        // Integrate + cull by lifetime
        for b in &mut self.balls {
            b.pos_m += b.vel_mps * dt_s;
            b.age_s += dt_s;
        }

        let max_age = config::BALL_LIFETIME_S;
        if max_age > 0.0 {
            let mut i = 0usize;
            while i < self.balls.len() {
                if self.balls[i].age_s >= max_age {
                    self.balls.swap_remove(i);
                    self.last_centers.swap_remove(i);
                } else {
                    i += 1;
                }
            }
        }

        // Only rebuild voxels if any ball center crossed into a new voxel cell
        let mut changed = false;
        for (i, b) in self.balls.iter().enumerate() {
            let c = Self::world_to_vox(b.pos_m, voxel_size_m);
            if c != self.last_centers[i] {
                self.last_centers[i] = c;
                changed = true;
            }
        }

        if changed {
            self.rebuild_overlay(voxel_size_m);
        }
    }

    #[inline]
    fn world_to_vox(p_m: Vec3, voxel_size_m: f32) -> IVec3 {
        // world-voxel coords (same space your chunk origins use: integer voxels)
        (p_m / voxel_size_m).floor().as_ivec3()
    }

    fn rebuild_overlay(&mut self, voxel_size_m: f32) {
        self.overlay.clear();

        // One ball => a solid voxel sphere of radius BALL_RADIUS (voxels)
        let r = config::BALL_RADIUS.max(1);
        let r2 = (r * r) as i32;

        for b in &self.balls {
            let c = Self::world_to_vox(b.pos_m, voxel_size_m);

            // voxel sphere fill
            for dz in -r..=r {
                for dy in -r..=r {
                    for dx in -r..=r {
                        let d2 = dx*dx + dy*dy + dz*dz;
                        if d2 <= r2 {
                            let v = c + IVec3::new(dx, dy, dz);
                            self.overlay.insert(v, 6u32); // MAT_BALL == 6
                        }
                    }
                }
            }
        }
    }
}
