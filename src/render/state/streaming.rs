// src/render/state/streaming.rs
// -----------------------------

use bytemuck::{Pod, Zeroable};

use crate::config;
use crate::render::gpu_types::StreamGpu;
use crate::world::ChunkCoord;

pub const FLAG_ACTIVE: u32 = 1;
pub const FLAG_DIRTY: u32 = 2;
pub const FLAG_READY: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ChunkMetaGpu {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub flags: u32,
}

impl ChunkMetaGpu {
    pub fn inactive() -> Self {
        Self { x: 0, y: 0, z: 0, flags: 0 }
    }
}

pub struct StreamingState {
    meta: Vec<ChunkMetaGpu>,
    origin: ChunkCoord,
    ox: u32,
    oy: u32,
    oz: u32,
    center: Option<ChunkCoord>,
    dirty_slots: Vec<u32>,
}

impl StreamingState {
    pub fn new() -> Self {
        Self {
            meta: vec![ChunkMetaGpu::inactive(); config::MAX_CHUNKS as usize],
            origin: ChunkCoord { x: 0, y: 0, z: 0 },
            ox: 0,
            oy: 0,
            oz: 0,
            center: None,
            dirty_slots: Vec::with_capacity(config::MAX_CHUNKS as usize),
        }
    }

    pub fn meta(&self) -> &[ChunkMetaGpu] { &self.meta }
    pub fn dirty_slots(&self) -> &[u32] { &self.dirty_slots }

    pub fn stream_gpu(&self) -> StreamGpu {
        StreamGpu {
            origin_x: self.origin.x,
            origin_y: self.origin.y,
            origin_z: self.origin.z,
            _pad0: 0,

            ox: self.ox,
            oy: self.oy,
            oz: self.oz,
            dirty_count: self.dirty_slots.len() as u32,

            dirty_offset: 0,
            build_count: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }

    /// Called by Renderer after it has *built* dirty_slots[start..start+count] this frame.
    /// Mirrors the GPU's clear_dirty behavior into CPU meta so future uploads don't wipe READY.
    pub fn mark_built_range(&mut self, start: u32, count: u32) {
        if count == 0 { return; }
        let end = start.saturating_add(count).min(self.dirty_slots.len() as u32);

        for i in start..end {
            let slot = self.dirty_slots[i as usize] as usize;
            if slot >= self.meta.len() { continue; }

            let f = self.meta[slot].flags;
            self.meta[slot].flags = (f & !FLAG_DIRTY) | FLAG_READY;
        }
    }

    fn dx() -> i32 { config::GRID_X as i32 }
    fn dy() -> i32 { config::GRID_Y as i32 }
    fn dz() -> i32 { config::GRID_Z as i32 }

    fn slot_from_phys(px: i32, py: i32, pz: i32) -> usize {
        (pz as usize * config::GRID_X as usize + px as usize) * config::GRID_Y as usize + py as usize
    }

    fn wrap_i32(v: i32, m: i32) -> i32 { v.rem_euclid(m) }

    fn phys_from_rel(&self, ix: i32, iy: i32, iz: i32) -> (i32, i32, i32) {
        let px = Self::wrap_i32(ix + self.ox as i32, Self::dx());
        let py = Self::wrap_i32(iy + self.oy as i32, Self::dy());
        let pz = Self::wrap_i32(iz + self.oz as i32, Self::dz());
        (px, py, pz)
    }

    fn set_slot(&mut self, slot: usize, cc: ChunkCoord, flags: u32) {
        self.meta[slot] = ChunkMetaGpu { x: cc.x, y: cc.y, z: cc.z, flags };
    }

    /// Rebuild dirty_slots from meta flags.
    /// This is the key change:
    /// - we can drop old batching progress safely,
    /// - without losing pending chunks that are still DIRTY (often not READY yet).
    fn refresh_dirty_slots_from_meta(&mut self) {
        self.dirty_slots.clear();
        for (i, cm) in self.meta.iter().enumerate() {
            if (cm.flags & FLAG_ACTIVE) == 0 { continue; }
            if (cm.flags & FLAG_DIRTY) == 0 { continue; }
            self.dirty_slots.push(i as u32);
        }
        self.dirty_slots.sort_unstable();
        self.dirty_slots.dedup();
    }

    fn clear_and_rebuild_all(&mut self, center: ChunkCoord) {
        self.center = Some(center);
        self.origin = ChunkCoord {
            x: center.x - config::CHUNK_RADIUS_X,
            y: center.y - config::CHUNK_RADIUS_Y,
            z: center.z - config::CHUNK_RADIUS_Z,
        };

        self.ox = 0;
        self.oy = 0;
        self.oz = 0;

        for i in 0..config::MAX_CHUNKS as usize {
            self.meta[i] = ChunkMetaGpu::inactive();
        }

        for iz in 0..Self::dz() {
            for ix in 0..Self::dx() {
                for iy in 0..Self::dy() {
                    let cc = ChunkCoord {
                        x: self.origin.x + ix,
                        y: self.origin.y + iy,
                        z: self.origin.z + iz,
                    };
                    let (px, py, pz) = self.phys_from_rel(ix, iy, iz);
                    let slot = Self::slot_from_phys(px, py, pz);

                    // fresh coords: NOT READY yet, must build
                    self.set_slot(slot, cc, FLAG_ACTIVE | FLAG_DIRTY);
                }
            }
        }

        self.refresh_dirty_slots_from_meta();
    }

    fn shift_x_pos(&mut self) {
        self.origin.x += 1;
        self.ox = (self.ox + 1) % config::GRID_X;

        let ix_enter = Self::dx() - 1;
        for iz in 0..Self::dz() {
            for iy in 0..Self::dy() {
                let cc = ChunkCoord { x: self.origin.x + ix_enter, y: self.origin.y + iy, z: self.origin.z + iz };
                let (px, py, pz) = self.phys_from_rel(ix_enter, iy, iz);
                let slot = Self::slot_from_phys(px, py, pz);

                // entering strip: new coord => clear READY
                self.set_slot(slot, cc, FLAG_ACTIVE | FLAG_DIRTY);
            }
        }
    }

    fn shift_x_neg(&mut self) {
        self.origin.x -= 1;
        self.ox = (self.ox + config::GRID_X - 1) % config::GRID_X;

        let ix_enter = 0;
        for iz in 0..Self::dz() {
            for iy in 0..Self::dy() {
                let cc = ChunkCoord { x: self.origin.x + ix_enter, y: self.origin.y + iy, z: self.origin.z + iz };
                let (px, py, pz) = self.phys_from_rel(ix_enter, iy, iz);
                let slot = Self::slot_from_phys(px, py, pz);

                self.set_slot(slot, cc, FLAG_ACTIVE | FLAG_DIRTY);
            }
        }
    }

    fn shift_y_pos(&mut self) {
        self.origin.y += 1;
        self.oy = (self.oy + 1) % config::GRID_Y;

        let iy_enter = Self::dy() - 1;
        for iz in 0..Self::dz() {
            for ix in 0..Self::dx() {
                let cc = ChunkCoord { x: self.origin.x + ix, y: self.origin.y + iy_enter, z: self.origin.z + iz };
                let (px, py, pz) = self.phys_from_rel(ix, iy_enter, iz);
                let slot = Self::slot_from_phys(px, py, pz);

                self.set_slot(slot, cc, FLAG_ACTIVE | FLAG_DIRTY);
            }
        }
    }

    fn shift_y_neg(&mut self) {
        self.origin.y -= 1;
        self.oy = (self.oy + config::GRID_Y - 1) % config::GRID_Y;

        let iy_enter = 0;
        for iz in 0..Self::dz() {
            for ix in 0..Self::dx() {
                let cc = ChunkCoord { x: self.origin.x + ix, y: self.origin.y + iy_enter, z: self.origin.z + iz };
                let (px, py, pz) = self.phys_from_rel(ix, iy_enter, iz);
                let slot = Self::slot_from_phys(px, py, pz);

                self.set_slot(slot, cc, FLAG_ACTIVE | FLAG_DIRTY);
            }
        }
    }

    fn shift_z_pos(&mut self) {
        self.origin.z += 1;
        self.oz = (self.oz + 1) % config::GRID_Z;

        let iz_enter = Self::dz() - 1;
        for ix in 0..Self::dx() {
            for iy in 0..Self::dy() {
                let cc = ChunkCoord { x: self.origin.x + ix, y: self.origin.y + iy, z: self.origin.z + iz_enter };
                let (px, py, pz) = self.phys_from_rel(ix, iy, iz_enter);
                let slot = Self::slot_from_phys(px, py, pz);

                self.set_slot(slot, cc, FLAG_ACTIVE | FLAG_DIRTY);
            }
        }
    }

    fn shift_z_neg(&mut self) {
        self.origin.z -= 1;
        self.oz = (self.oz + config::GRID_Z - 1) % config::GRID_Z;

        let iz_enter = 0;
        for ix in 0..Self::dx() {
            for iy in 0..Self::dy() {
                let cc = ChunkCoord { x: self.origin.x + ix, y: self.origin.y + iy, z: self.origin.z + iz_enter };
                let (px, py, pz) = self.phys_from_rel(ix, iy, iz_enter);
                let slot = Self::slot_from_phys(px, py, pz);

                self.set_slot(slot, cc, FLAG_ACTIVE | FLAG_DIRTY);
            }
        }
    }

    pub fn update_center(&mut self, center: ChunkCoord) -> bool {
        if self.center.is_none() {
            self.clear_and_rebuild_all(center);
            return !self.dirty_slots.is_empty();
        }

        let old = self.center.unwrap();
        if old == center {
            // keep whatever dirty set is already pending
            return !self.dirty_slots.is_empty();
        }

        let dx = center.x - old.x;
        let dy = center.y - old.y;
        let dz = center.z - old.z;

        self.center = Some(center);

        if dx.abs() > 1 || dy.abs() > 1 || dz.abs() > 1 {
            self.clear_and_rebuild_all(center);
            return !self.dirty_slots.is_empty();
        }

        // Apply 1-step torus shifts. These mark entering strips DIRTY (and clear READY).
        if dx == 1 { self.shift_x_pos(); }
        if dx == -1 { self.shift_x_neg(); }
        if dy == 1 { self.shift_y_pos(); }
        if dy == -1 { self.shift_y_neg(); }
        if dz == 1 { self.shift_z_pos(); }
        if dz == -1 { self.shift_z_neg(); }

        // CRITICAL: rebuild dirty_slots from flags so we don't lose pending "not READY yet" chunks
        // when the camera moves again before prior work completes.
        self.refresh_dirty_slots_from_meta();

        !self.dirty_slots.is_empty()
    }

    pub fn mark_all_dirty(&mut self) {
        for i in 0..config::MAX_CHUNKS as usize {
            if (self.meta[i].flags & FLAG_ACTIVE) != 0 {
                // keep READY so old data can render while rebuilding
                self.meta[i].flags |= FLAG_DIRTY;
            }
        }
        self.refresh_dirty_slots_from_meta();
    }
}
