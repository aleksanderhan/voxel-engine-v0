// src/world.rs
// ------------
//
// CPU-side chunk neighborhood helper.

use crate::config;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkCoord {
    pub fn origin_vox_i32(self) -> [i32; 4] {
        [
            self.x * config::CHUNK_RES_I32,
            self.y * config::CHUNK_RES_I32,
            self.z * config::CHUNK_RES_I32,
            0,
        ]
    }

    pub fn from_world_pos_m(p: glam::Vec3) -> Self {
        let cs = config::CHUNK_SIZE_M_F32;
        Self {
            x: (p.x / cs).floor() as i32,
            y: (p.y / cs).floor() as i32,
            z: (p.z / cs).floor() as i32,
        }
    }
}

pub fn neighborhood(center: ChunkCoord) -> Vec<[i32; 4]> {
    let mut out = Vec::with_capacity(config::MAX_CHUNKS as usize);

    for dz in -config::CHUNK_RADIUS_Z..=config::CHUNK_RADIUS_Z {
        for dx in -config::CHUNK_RADIUS_X..=config::CHUNK_RADIUS_X {
            for dy in -config::CHUNK_RADIUS_Y..=config::CHUNK_RADIUS_Y {
                let c = ChunkCoord {
                    x: center.x + dx,
                    y: center.y + dy,
                    z: center.z + dz,
                };
                out.push(c.origin_vox_i32());
            }
        }
    }

    out.truncate(config::MAX_CHUNKS as usize);
    out
}
