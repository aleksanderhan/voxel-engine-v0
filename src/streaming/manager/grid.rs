use crate::{streaming::types::*};
use crate::app::config;

use super::ChunkManager;

pub fn rebuild_if_dirty(mgr: &mut ChunkManager, center: ChunkKey) -> bool {
    let changed = mgr.grid.grid_dirty;
    if mgr.grid.grid_dirty {
        rebuild_grid(mgr, center);
        mgr.grid.grid_dirty = false;
    }
    changed
}

pub fn rebuild_grid(mgr: &mut ChunkManager, center: ChunkKey) {
    let nx = (2 * config::KEEP_RADIUS + 1) as u32;
    let nz = nx;
    let ny = GRID_Y_COUNT;

    mgr.grid.grid_dims = [nx, ny, nz];
    mgr.grid.grid_origin_chunk = super::keep::keep_origin_for(center);
    mgr.grid.grid_dims_coarse = [
        (nx + 1) / 2,
        (ny + 1) / 2,
        (nz + 1) / 2,
    ];

    let needed = (nx * ny * nz) as usize;
    if mgr.grid.chunk_grid.len() != needed {
        mgr.grid.chunk_grid.resize(needed, INVALID_U32);
    }
    mgr.grid.chunk_grid.fill(INVALID_U32);

    // Include any slot that is actually usable by the GPU.
    for slot in 0..mgr.slots.slot_to_key.len() {
        let k = mgr.slots.slot_to_key[slot];

        let ready = match mgr.build.chunks.get(&k) {
            Some(ChunkState::Resident(_)) => true,
            Some(ChunkState::Uploading(up)) => up.uploaded, // uploaded => meta+macro/colinfo+nodes are on GPU
            _ => false,
        };

        if !ready {
            continue;
        }

        if let Some(idx) = grid_index_for_chunk(mgr, k) {
            mgr.grid.chunk_grid[idx] = slot as u32;
        }
    }

    let [cnx, cny, cnz] = mgr.grid.grid_dims_coarse;
    let coarse_needed = (cnx * cny * cnz) as usize;
    if mgr.grid.chunk_grid_coarse.len() != coarse_needed {
        mgr.grid.chunk_grid_coarse.resize(coarse_needed, 0);
    }
    mgr.grid.chunk_grid_coarse.fill(0);

    for cz in 0..cnz {
        for cy in 0..cny {
            for cx in 0..cnx {
                let mut occupied = false;
                for dz in 0..2 {
                    let z = cz * 2 + dz;
                    if z >= nz { continue; }
                    for dy in 0..2 {
                        let y = cy * 2 + dy;
                        if y >= ny { continue; }
                        for dx in 0..2 {
                            let x = cx * 2 + dx;
                            if x >= nx { continue; }
                            let idx = (z * ny * nx) + (y * nx) + x;
                            let slot = mgr.grid.chunk_grid[idx as usize];
                            if slot == INVALID_U32 { continue; }
                            let slot = slot as usize;
                            if slot >= mgr.slots.chunk_meta.len() { continue; }
                            if mgr.slots.chunk_meta[slot].macro_empty != 0 {
                                continue;
                            }
                            occupied = true;
                            break;
                        }
                        if occupied { break; }
                    }
                    if occupied { break; }
                }
                let cidx = (cz * cny * cnx) + (cy * cnx) + cx;
                mgr.grid.chunk_grid_coarse[cidx as usize] = u32::from(occupied);
            }
        }
    }
}

#[inline]
fn grid_index_for_chunk(mgr: &ChunkManager, k: ChunkKey) -> Option<usize> {
    let [ox, oy, oz] = mgr.grid.grid_origin_chunk;
    let [nx, ny, nz] = mgr.grid.grid_dims;

    let ix = k.x - ox;
    let iy = k.y - oy;
    let iz = k.z - oz;

    if ix < 0 || iy < 0 || iz < 0 {
        return None;
    }

    let ix = ix as u32;
    let iy = iy as u32;
    let iz = iz as u32;

    if ix >= nx || iy >= ny || iz >= nz {
        return None;
    }

    let idx = (iz * ny * nx) + (iy * nx) + ix;
    Some(idx as usize)
}
