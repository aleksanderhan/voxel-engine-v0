
use crate::{
    streaming::{types::ChunkKey, ChunkManager},
    world::WorldGen,
};
use crate::app::config;

use super::collision::WorldQuery;



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

    
    
    
    
    #[inline]
    fn col_u16(colinfo_words: &[u32], lx: i32, lz: i32) -> u16 {
        
        
        let x = lx as usize;
        let z = lz as usize;
        let col = z * 64 + x; 
        let w = colinfo_words[col >> 1];
        if (col & 1) == 0 {
            (w & 0xFFFF) as u16
        } else {
            (w >> 16) as u16
        }
    }

    
    
    
    
    
    
    
    #[inline]
    fn solid_from_col_u16(v: u16, ly: i32) -> bool {
        let top_y = (v & 0x00FF) as i32;   
        

        if top_y == 255 {
            return false;
        }

        
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

        
        let slot_opt = match self.mgr.build.chunks.get(&k) {
            Some(crate::streaming::types::ChunkState::Resident(r)) => Some(r.slot as usize),
            Some(crate::streaming::types::ChunkState::Uploading(u)) => Some(u.slot as usize),
            _ => None,
        };

        if let Some(slot) = slot_opt {
            
            if slot < self.mgr.slots.slot_colinfo.len() {
                let colinfo = &self.mgr.slots.slot_colinfo[slot];
                if colinfo.len() == crate::streaming::types::COLINFO_WORDS_PER_CHUNK_USIZE {
                    
                    if lx >= 0 && lx < 64 && lz >= 0 && lz < 64 {
                        let v = Self::col_u16(colinfo, lx, lz);
                        return Self::solid_from_col_u16(v, ly);
                    }
                }
            }
        }

        
        
        if let Some(world) = self.world {
            let ground_y_vox = world.ground_height(vx, vz);
            return vy <= ground_y_vox;
        }

        false
    }
}