use crate::{config, streaming::types::*};

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
