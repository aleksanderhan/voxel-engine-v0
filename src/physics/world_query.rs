// src/physics/world_query.rs
use crate::{
    streaming::{types::ChunkKey, ChunkManager},
    world::WorldGen,
};
use crate::app::config;

use super::collision::WorldQuery;

/// Adapter: physics queries solidity through ChunkManager's resident/uploading slots.
/// Falls back to WorldGen heightfield if the chunk isn't available yet.
pub struct ChunkManagerQuery<'a> {
    pub mgr: &'a ChunkManager,
    pub world: Option<&'a WorldGen>,
}

impl<'a> ChunkManagerQuery<'a> {
    #[inline]
    fn chunk_key_for_voxel(vx: i32, vy: i32, vz: i32) -> (ChunkKey, i32, i32, i32) {
        let cs = config::CHUNK_SIZE as i32;

        let cx = vx.div_euclid(cs);
        let cy = vy.div_euclid(cs);
        let cz = vz.div_euclid(cs);

        let lx = vx.rem_euclid(cs);
        let ly = vy.rem_euclid(cs);
        let lz = vz.rem_euclid(cs);

        (ChunkKey { x: cx, y: cy, z: cz }, lx, ly, lz)
    }

    /// Read packed u16 column value from `colinfo_words`.
    /// Layout assumption we *can* rely on from your comment:
    /// - 64x64 columns
    /// - packed 2x u16 per u32
    #[inline]
    fn col_u16(colinfo_words: &[u32], lx: i32, lz: i32) -> u16 {
        // chunk is assumed to be CHUNK_SIZE=64 in XZ for colinfo.
        // If you ever change CHUNK_SIZE, update this mapping.
        let x = lx as usize;
        let z = lz as usize;
        let col = z * 64 + x; // 0..4095
        let w = colinfo_words[col >> 1];
        if (col & 1) == 0 {
            (w & 0xFFFF) as u16
        } else {
            (w >> 16) as u16
        }
    }

    /// Interpret the u16 column info as a solidity test at local y.
    ///
    /// The *exact* encoding of your u16 isn't shown yet, so this is a robust heuristic:
    /// - If it looks like (min_y, max_y) packed in bytes: solid when y in [min..=max]
    /// - Else treat the low byte as a "surface height": solid when y <= surface
    ///
    /// This is enough to get grounded movement + no falling-through while you paste the real encoding later.
    #[inline]
    fn solid_from_col_u16(v: u16, ly: i32) -> bool {
        let top_y = (v & 0x00FF) as i32;   // low byte
        // let mat = (v >> 8) as u8;       // high byte (unused for solidity)

        if top_y == 255 {
            return false;
        }

        // "filled below top" convention
        ly <= top_y
    }
}

impl<'a> WorldQuery for ChunkManagerQuery<'a> {
    #[inline]
    fn voxel_size_m(&self) -> f32 {
        config::VOXEL_SIZE_M_F32
    }

    #[inline]
    fn solid_voxel_at(&self, vx: i32, vy: i32, vz: i32) -> bool {
        let (k, lx, ly, lz) = Self::chunk_key_for_voxel(vx, vy, vz);

        // Try loaded chunks first (Resident or Uploading with a slot).
        let slot_opt = match self.mgr.build.chunks.get(&k) {
            Some(crate::streaming::types::ChunkState::Resident(r)) => Some(r.slot as usize),
            Some(crate::streaming::types::ChunkState::Uploading(u)) => Some(u.slot as usize),
            _ => None,
        };

        if let Some(slot) = slot_opt {
            // slot_colinfo exists for both uploading and resident slots
            if slot < self.mgr.slots.slot_colinfo.len() {
                let colinfo = &self.mgr.slots.slot_colinfo[slot];
                if colinfo.len() == crate::streaming::types::COLINFO_WORDS_PER_CHUNK_USIZE {
                    // Safety: colinfo assumes XZ=64x64
                    if lx >= 0 && lx < 64 && lz >= 0 && lz < 64 {
                        let v = Self::col_u16(colinfo, lx, lz);
                        return Self::solid_from_col_u16(v, ly);
                    }
                }
            }
        }

        // Fallback: if chunk isn't loaded, approximate solidity from the heightfield.
        // This prevents the player from falling through while streaming catches up.
        if let Some(world) = self.world {
            let ground_y_vox = world.ground_height(vx, vz);
            return vy <= ground_y_vox;
        }

        false
    }
}