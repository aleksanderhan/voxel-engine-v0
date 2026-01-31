use glam::Vec3;

use crate::{config, streaming::{ChunkManager, types::{ChunkKey, ChunkState}}, world::WorldGen};

/// VoxelQuery: minimal interface physics needs.
/// Positions are in meters (same as your camera).
pub trait VoxelQuery {
    /// Returns approximate ground height (meters) at x,z.
    /// If unknown, returns None.
    fn ground_height_m(&self, x_m: f32, z_m: f32) -> Option<f32>;

    /// Is the voxel at world position solid?
    /// For now we implement "filled column below top", which matches your current terrain gen.
    fn solid_at_m(&self, p_m: Vec3) -> bool;
}

/// Query implementation backed by ChunkManager + WorldGen.
/// Uses colinfo_words when chunks are available, falls back to WorldGen::ground_height.
pub struct VoxelWorldQuery<'a> {
    pub world: &'a WorldGen,
    pub chunks: &'a ChunkManager,
}

impl<'a> VoxelWorldQuery<'a> {
    #[inline]
    fn m_to_vox(x_m: f32) -> i32 {
        (x_m / config::VOXEL_SIZE_M_F32).floor() as i32
    }

    #[inline]
    fn vox_to_m(y_vox: i32) -> f32 {
        (y_vox as f32) * config::VOXEL_SIZE_M_F32
    }

    #[inline]
    fn chunk_key_from_vox(vx: i32, vy: i32, vz: i32) -> (ChunkKey, i32, i32, i32) {
        let cs = config::CHUNK_SIZE as i32;

        let cx = vx.div_euclid(cs);
        let cy = vy.div_euclid(cs);
        let cz = vz.div_euclid(cs);

        let lx = vx.rem_euclid(cs);
        let ly = vy.rem_euclid(cs);
        let lz = vz.rem_euclid(cs);

        (ChunkKey { x: cx, y: cy, z: cz }, lx, ly, lz)
    }

    /// Fetch slot for a chunk if it is GPU-ready (Resident or Uploading(uploaded)).
    fn slot_for_chunk_if_ready(&self, k: ChunkKey) -> Option<usize> {
        match self.chunks.build.chunks.get(&k) {
            Some(ChunkState::Resident(r)) => Some(r.slot as usize),
            Some(ChunkState::Uploading(u)) if u.uploaded => Some(u.slot as usize),
            _ => None,
        }
    }

    /// Decode the packed u16 column entry from colinfo_words.
    /// Returns (top_y_local_vox, mat8). If empty, top_y_local=255.
    fn colinfo_entry(colinfo_words: &[u32], lx: i32, lz: i32) -> (u8, u8) {
        let idx = (lz as u32) * 64 + (lx as u32); // 0..4095
        let w = (idx >> 1) as usize;              // 0..2047
        let hi = (idx & 1) != 0;

        let word = colinfo_words[w];
        let entry16 = if !hi { word & 0xFFFF } else { (word >> 16) & 0xFFFF };

        let y8 = (entry16 & 0xFF) as u8;
        let mat8 = ((entry16 >> 8) & 0xFF) as u8;
        (y8, mat8)
    }

    fn ground_height_from_chunk_column(&self, k: ChunkKey, lx: i32, lz: i32) -> Option<i32> {
        let slot = self.slot_for_chunk_if_ready(k)?;
        let colinfo = self.chunks.slots.slot_colinfo.get(slot)?.as_ref();
        if colinfo.len() != crate::streaming::types::COLINFO_WORDS_PER_CHUNK_USIZE {
            return None;
        }

        let (y8, _mat8) = Self::colinfo_entry(colinfo, lx, lz);
        if y8 == 255 {
            return None;
        }

        let cs = config::CHUNK_SIZE as i32;
        let chunk_oy = k.y * cs;
        Some(chunk_oy + (y8 as i32))
    }
}

impl<'a> VoxelQuery for VoxelWorldQuery<'a> {
    fn ground_height_m(&self, x_m: f32, z_m: f32) -> Option<f32> {
        let vx = Self::m_to_vox(x_m);
        let vz = Self::m_to_vox(z_m);

        // Use vy=0 just to pick a chunk y via div_euclid; we only need x/z column anyway.
        let (k, lx, _ly, lz) = Self::chunk_key_from_vox(vx, 0, vz);

        // Try streamed chunk column
        if let Some(top_y_vox) = self.ground_height_from_chunk_column(k, lx, lz) {
            // height of the *surface* is one voxel above top solid
            return Some(Self::vox_to_m(top_y_vox + 1));
        }

        // Fallback to procedural height (in voxel coords)
        let gy_vox = self.world.ground_height(vx, vz);
        Some(Self::vox_to_m(gy_vox + 1))
    }

    fn solid_at_m(&self, p_m: Vec3) -> bool {
        let vx = Self::m_to_vox(p_m.x);
        let vy = Self::m_to_vox(p_m.y);
        let vz = Self::m_to_vox(p_m.z);

        let (k, lx, ly, lz) = Self::chunk_key_from_vox(vx, vy, vz);

        // If we have colinfo, use “filled below top” test.
        if let Some(top_y_vox) = self.ground_height_from_chunk_column(k, lx, lz) {
            return (k.y * config::CHUNK_SIZE as i32 + ly) <= top_y_vox;
        }

        // Fallback: use procedural ground height
        let gy = self.world.ground_height(vx, vz);
        vy <= gy
    }
}
