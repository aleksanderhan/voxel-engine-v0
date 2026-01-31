use crate::streaming::types::*;
use crate::{config, world::WorldGen};
use super::{ChunkManager};
use super::keep;

#[inline]
fn compute_ground_cy_at_column(world: &WorldGen, cx: i32, cz: i32) -> i32 {
    let cs = config::CHUNK_SIZE as i32;
    let half = cs / 2;
    let wx = cx * cs + half;
    let wz = cz * cs + half;
    let ground_y_vox = world.ground_height(wx, wz);
    ground_y_vox.div_euclid(cs)
}

pub fn ensure_column_cache(mgr: &mut ChunkManager, world: &WorldGen, center: ChunkKey) {
    let new_origin = keep::keep_origin_for(center);
    let origin_changed = mgr.ground.col_ground_cy.is_empty() || new_origin != mgr.grid.grid_origin_chunk;

    if !origin_changed {
        return;
    }

    let old_origin = mgr.grid.grid_origin_chunk;

    // publish new origin first (so indexing uses it)
    mgr.grid.grid_origin_chunk = new_origin;

    update_column_ground_cache(mgr, world, old_origin, new_origin);
    mgr.grid.grid_dirty = true;
}

pub fn ground_cy_for_column(mgr: &ChunkManager, cx: i32, cz: i32) -> Option<i32> {
    let [ox, _, oz] = mgr.grid.grid_origin_chunk;

    let nx = (2 * config::KEEP_RADIUS + 1) as i32;
    let nz = nx;

    let ix = cx - ox;
    let iz = cz - oz;
    if ix < 0 || iz < 0 || ix >= nx || iz >= nz { return None; }

    let idx = (iz * nx + ix) as usize;
    mgr.ground.col_ground_cy.get(idx).copied()
}

fn update_column_ground_cache(
    mgr: &mut ChunkManager,
    world: &WorldGen,
    old_origin: [i32; 3],
    new_origin: [i32; 3],
) {
    let nx = (2 * config::KEEP_RADIUS + 1) as i32;
    let nz = nx;
    let len = (nx * nz) as usize;

    // first time / resize => full rebuild
    if mgr.ground.col_ground_cy.len() != len || mgr.ground.col_ground_cy.is_empty() {
        mgr.ground.col_ground_cy.resize(len, 0);
        let ox = new_origin[0];
        let oz = new_origin[2];
        for dz in 0..nz {
            for dx in 0..nx {
                let cx = ox + dx;
                let cz = oz + dz;
                mgr.ground.col_ground_cy[(dz * nx + dx) as usize] =
                    compute_ground_cy_at_column(world, cx, cz);
            }
        }
        return;
    }

    let dx_chunks = new_origin[0] - old_origin[0];
    let dz_chunks = new_origin[2] - old_origin[2];

    if dx_chunks == 0 && dz_chunks == 0 {
        return;
    }

    // teleport => rebuild
    if dx_chunks.abs() >= nx || dz_chunks.abs() >= nz {
        let ox = new_origin[0];
        let oz = new_origin[2];
        for dz in 0..nz {
            for dx in 0..nx {
                let cx = ox + dx;
                let cz = oz + dz;
                mgr.ground.col_ground_cy[(dz * nx + dx) as usize] =
                    compute_ground_cy_at_column(world, cx, cz);
            }
        }
        return;
    }

    let old = std::mem::take(&mut mgr.ground.col_ground_cy);
    let mut newv = vec![0i32; len];

    let ox_new = new_origin[0];
    let oz_new = new_origin[2];

    for iz in 0..nz {
        for ix in 0..nx {
            let sx = ix - dx_chunks;
            let sz = iz - dz_chunks;

            let dst_idx = (iz * nx + ix) as usize;

            if sx >= 0 && sx < nx && sz >= 0 && sz < nz {
                let src_idx = (sz * nx + sx) as usize;
                newv[dst_idx] = old[src_idx];
            } else {
                let cx = ox_new + ix;
                let cz = oz_new + iz;
                newv[dst_idx] = compute_ground_cy_at_column(world, cx, cz);
            }
        }
    }

    mgr.ground.col_ground_cy = newv;
}
