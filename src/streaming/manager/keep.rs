use glam::Vec3;
use crate::{world::WorldGen};
use crate::app::config;
use crate::streaming::types::*;

use super::ChunkManager;



#[inline]
pub fn build_offsets(radius: i32) -> Vec<(i32, i32, i32)> {
    let mut v = Vec::new();

    let r2 = radius * radius;
    for dy in GRID_Y_MIN_DY..=(GRID_Y_MIN_DY + GRID_Y_COUNT as i32 - 1) {
        for dz in -radius..=radius {
            for dx in -radius..=radius {
                if dx*dx + dz*dz <= r2 {
                    v.push((dx, dy, dz));
                }
            }
        }
    }

    
    v.sort_by_key(|(dx, dy, dz)| (dx*dx + dz*dz, *dy, *dz, *dx));
    v
}

#[inline(always)]
pub fn in_active_xz(center: ChunkKey, k: ChunkKey) -> bool {
    let ar = config::ACTIVE_RADIUS;
    let dx = k.x - center.x;
    let dz = k.z - center.z;
    dx*dx + dz*dz <= ar*ar
}

pub fn keep_origin_for(center: ChunkKey) -> [i32; 3] {
    let ox = center.x - config::KEEP_RADIUS;
    let oz = center.z - config::KEEP_RADIUS;
    let oy = center.y + GRID_Y_MIN_DY;
    [ox, oy, oz]
}



#[inline]
pub fn compute_center(world: &WorldGen, cam_pos_m: Vec3) -> ChunkKey {
    let cam_vx = (cam_pos_m.x / config::VOXEL_SIZE_M_F32).floor() as i32;
    let cam_vz = (cam_pos_m.z / config::VOXEL_SIZE_M_F32).floor() as i32;

    let cs = config::CHUNK_SIZE as i32;
    let half = cs / 2;

    let ccx = cam_vx.div_euclid(cs);
    let ccz = cam_vz.div_euclid(cs);

    let wx = ccx * cs + half;
    let wz = ccz * cs + half;
    let ground_y_vox = world.ground_height(wx, wz);
    let ground_cy = ground_y_vox.div_euclid(cs);

    ChunkKey { x: ccx, y: ground_cy, z: ccz }
}


pub fn publish_center_and_rebucket(mgr: &mut ChunkManager, center: ChunkKey) -> bool {
    let changed = mgr.build.last_center.map_or(true, |c| c != center);
    if changed {
        mgr.build.last_center = Some(center);
        super::uploads::rebucket_for_center(mgr, center);
    }
    changed
}
