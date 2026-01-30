// src/streaming/manager.rs
// Chunk streaming + background SVO building + CPU cache.

use std::{
    collections::VecDeque,
    mem::size_of,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use glam::Vec3;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::{
    config,
    render::gpu_types::{ChunkMetaGpu, NodeGpu, NodeRopesGpu},
    world::WorldGen,
};

use super::{
    NodeArena,
    cache::ChunkCache,
    priority::sort_queue_near_first,
    types::{
        BuildDone, BuildJob, ChunkKey, ChunkState, ChunkUpload, Resident, Uploading,
        COLINFO_WORDS_PER_CHUNK, COLINFO_WORDS_PER_CHUNK_USIZE,
        EVICT_ATTEMPTS, GRID_Y_COUNT, GRID_Y_MIN_DY,
        INVALID_U32,
        MACRO_WORDS_PER_CHUNK, MACRO_WORDS_PER_CHUNK_USIZE,
        MAX_UPLOAD_BYTES_PER_FRAME, MAX_UPLOADS_PER_FRAME,
        y_band_max, y_band_min,
    },
    workers::spawn_workers,
};

pub struct ChunkManager {
    chunks: HashMap<ChunkKey, ChunkState>,
    build_queue: VecDeque<ChunkKey>,
    queued_set: HashSet<ChunkKey>,
    last_center: Option<ChunkKey>,

    cancels: HashMap<ChunkKey, Arc<AtomicBool>>,

    tx_job: Sender<BuildJob>,
    rx_done: Receiver<BuildDone>,
    in_flight: usize,

    // Slot-backed arrays: [0 .. resident_slots) are guaranteed resident.
    slot_to_key: Vec<ChunkKey>,
    chunk_meta: Vec<ChunkMetaGpu>,
    slot_macro: Vec<Arc<[u32]>>,
    slot_colinfo: Vec<Arc<[u32]>>,
    resident_slots: usize,

    uploads: VecDeque<ChunkUpload>,

    grid_dirty: bool,
    arena: NodeArena,

    grid_origin_chunk: [i32; 3],
    grid_dims: [u32; 3],
    chunk_grid: Vec<u32>,

    // Column ground cache for keep-area (x,z) -> ground chunk-y.
    col_ground_cy: Vec<i32>,

    cache: ChunkCache,

    active_offsets: Vec<(i32, i32, i32)>,
    to_unload: Vec<ChunkKey>,

    uploads_rewrite: VecDeque<ChunkUpload>, // slot rewrites (completes_residency=false)
    uploads_active:  VecDeque<ChunkUpload>, // completes residency, inside ACTIVE xz
    uploads_other:   VecDeque<ChunkUpload>, // completes residency, outside ACTIVE xz
}

impl ChunkManager {
    pub fn new(gen: Arc<WorldGen>) -> Self {
        let cap = (config::MAX_IN_FLIGHT * 2).max(8);

        let (tx_job, rx_job) = bounded::<BuildJob>(cap);
        let (tx_done, rx_done) = bounded::<BuildDone>(cap);

        spawn_workers(gen, rx_job, tx_done);

        let node_capacity = (config::NODE_BUDGET_BYTES / size_of::<NodeGpu>()) as u32;

        let nx = (2 * config::KEEP_RADIUS + 1) as u32;
        let nz = nx;
        let ny = GRID_Y_COUNT;

        let grid_len = (nx * ny * nz) as usize;
        let active_offsets = Self::build_offsets(config::ACTIVE_RADIUS);

        Self {
            chunks: HashMap::default(),
            build_queue: VecDeque::new(),
            queued_set: HashSet::default(),
            last_center: None,

            cancels: HashMap::default(),

            tx_job,
            rx_done,
            in_flight: 0,

            slot_to_key: Vec::new(),
            chunk_meta: Vec::new(),
            slot_macro: Vec::new(),
            slot_colinfo: Vec::new(),
            resident_slots: 0,

            uploads: VecDeque::new(),

            grid_dirty: true,
            arena: NodeArena::new(node_capacity),

            grid_origin_chunk: [0, 0, 0],
            grid_dims: [nx, ny, nz],
            chunk_grid: vec![INVALID_U32; grid_len],

            col_ground_cy: Vec::new(),
            cache: ChunkCache::new(),

            active_offsets,
            to_unload: Vec::new(),
            
            uploads_rewrite: VecDeque::new(),
            uploads_active:  VecDeque::new(),
            uploads_other:   VecDeque::new(),
        }
    }

    #[inline]
    fn build_offsets(radius: i32) -> Vec<(i32, i32, i32)> {
        let mut v = Vec::new();
        v.reserve(
            (GRID_Y_COUNT as usize)
                * ((2 * radius + 1) as usize)
                * ((2 * radius + 1) as usize),
        );

        for dy in GRID_Y_MIN_DY..=(GRID_Y_MIN_DY + GRID_Y_COUNT as i32 - 1) {
            for dz in -radius..=radius {
                for dx in -radius..=radius {
                    v.push((dx, dy, dz));
                }
            }
        }
        v
    }

    #[inline]
    fn keep_origin_for(&self, center: ChunkKey) -> [i32; 3] {
        let ox = center.x - config::KEEP_RADIUS;
        let oz = center.z - config::KEEP_RADIUS;
        let oy = center.y + GRID_Y_MIN_DY;
        [ox, oy, oz]
    }

    #[inline]
    fn cancel_token(&mut self, key: ChunkKey) -> Arc<AtomicBool> {
        self.cancels
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicBool::new(false)))
            .clone()
    }

    #[inline]
    fn ground_cy_for_column(&self, cx: i32, cz: i32) -> Option<i32> {
        let [ox, _, oz] = self.grid_origin_chunk;

        let nx = (2 * config::KEEP_RADIUS + 1) as i32;
        let nz = nx;

        let ix = cx - ox;
        let iz = cz - oz;
        if ix < 0 || iz < 0 || ix >= nx || iz >= nz {
            return None;
        }

        let idx = (iz * nx + ix) as usize;
        self.col_ground_cy.get(idx).copied()
    }

    #[inline(always)]
    fn in_keep(&self, center: ChunkKey, k: ChunkKey) -> bool {
        let dx = k.x - center.x;
        let dz = k.z - center.z;

        if dx < -config::KEEP_RADIUS || dx > config::KEEP_RADIUS { return false; }
        if dz < -config::KEEP_RADIUS || dz > config::KEEP_RADIUS { return false; }

        let Some(ground_cy) = self.ground_cy_for_column(k.x, k.z) else {
            return false; // outside cached area
        };

        let dy = k.y - ground_cy;
        dy >= y_band_min() && dy <= y_band_max()
    }

    pub fn update(&mut self, world: &Arc<WorldGen>, cam_pos_m: Vec3, cam_fwd: Vec3) -> bool {
        // --- camera -> chunk center (x,z) and ground-based center.y ---
        let cam_vx = (cam_pos_m.x / config::VOXEL_SIZE_M_F32).floor() as i32;
        let cam_vz = (cam_pos_m.z / config::VOXEL_SIZE_M_F32).floor() as i32;

        let ccx = cam_vx.div_euclid(config::CHUNK_SIZE as i32);
        let ccz = cam_vz.div_euclid(config::CHUNK_SIZE as i32);

        let ground_y_vox = world.ground_height(cam_vx, cam_vz);
        let ground_cy = ground_y_vox.div_euclid(config::CHUNK_SIZE as i32);

        let center = ChunkKey { x: ccx, y: ground_cy, z: ccz };

        // --- ensure column-height cache is built BEFORE anyone indexes it ---
        let new_origin = self.keep_origin_for(center);
        let origin_changed = self.col_ground_cy.is_empty() || new_origin != self.grid_origin_chunk;

        if origin_changed {
            let old_origin = self.grid_origin_chunk;

            // Publish new origin first so ground_cy_for_column() indexes using new origin…
            self.grid_origin_chunk = new_origin;

            // …but slide using the explicit old_origin -> new_origin transform.
            self.update_column_ground_cache(world.as_ref(), old_origin, new_origin);

            self.grid_dirty = true;
        }



        // --- ACTIVE: enqueue/restore chunks around camera ---
        let n_active = self.active_offsets.len();
        for i in 0..n_active {
            let (dx, dy, dz) = self.active_offsets[i];
            let x = center.x + dx;
            let z = center.z + dz;

            let Some(ground_cy_col) = self.ground_cy_for_column(x, z) else {
                // If the column cache can’t answer, skip this offset for now.
                continue;
            };
            let k = ChunkKey { x, y: ground_cy_col + dy, z };


            match self.chunks.get(&k) {
                Some(ChunkState::Resident(_))
                | Some(ChunkState::Uploading(_))
                | Some(ChunkState::Queued)
                | Some(ChunkState::Building) => {}
                None => {
                    if self.cache.get(&k).is_some() {
                        let _ = self.try_promote_from_cache(center, k);
                        continue;
                    }

                    let c = self.cancel_token(k);
                    c.store(false, Ordering::Relaxed);

                    self.chunks.insert(k, ChunkState::Queued);

                    if self.queued_set.insert(k) {
                        self.build_queue.push_back(k);
                    }
                }
            }
        }

        // --- Unload outside KEEP ---
        self.to_unload.clear();
        for &k in self.chunks.keys() {
            // Pin active area: do not unload these (prevents pop-thrash).
            if self.in_active_xz(center, k) {
                continue;
            }

            if !self.in_keep(center, k) {
                self.to_unload.push(k);
            }
        }

        let unload = std::mem::take(&mut self.to_unload);
        for k in unload {
            self.unload_chunk(k);
        }

        // --- handle center change (queue cleanup + resort) ---
        let center_changed = self.last_center.map_or(true, |c| c != center);
        if center_changed {
            self.last_center = Some(center);
            self.rebucket_uploads_for_center(center);

            // CANCEL stale builds: worker threads are still finishing jobs from the old center.
            // If they're not in/near ACTIVE now, stop them so CPU time goes to new nearby chunks.
            let mut stale_building = Vec::new();
            for (&k, st) in self.chunks.iter() {
                if matches!(st, ChunkState::Building) && !self.in_active_xz(center, k) {
                    stale_building.push(k);
                }
            }
            for k in stale_building {
                self.cancel_token(k).store(true, Ordering::Relaxed);
                // Keep state as Building for now; when the canceled completion arrives,
                // your harvest loop will drop it and remove it from self.chunks.
            }
            // Also cancel queued items that are already outside KEEP for the new center.
            // (They won't be dispatched, but this keeps tokens consistent and avoids waste.)
            let mut stale_queued = Vec::new();
            for (&k, st) in self.chunks.iter() {
                if matches!(st, ChunkState::Queued) && !self.in_keep(center, k) {
                    stale_queued.push(k);
                }
            }
            for k in stale_queued {
                self.cancel_token(k).store(true, Ordering::Relaxed);
                self.chunks.remove(&k);
                self.queued_set.remove(&k);
                // build_queue cleanup happens below via retain()
            }


            // retain only queued + still in keep, but do it safely (no panics)
            let origin = self.grid_origin_chunk;
            let nx = (2 * config::KEEP_RADIUS + 1) as i32;

            self.build_queue.retain(|k| {
                if !matches!(self.chunks.get(k), Some(ChunkState::Queued)) {
                    return false;
                }

                let dx = k.x - center.x;
                let dz = k.z - center.z;
                if dx < -config::KEEP_RADIUS || dx > config::KEEP_RADIUS { return false; }
                if dz < -config::KEEP_RADIUS || dz > config::KEEP_RADIUS { return false; }

                let ix = k.x - origin[0];
                let iz = k.z - origin[2];
                if ix < 0 || iz < 0 || ix >= nx || iz >= nx { return false; }

                let idx = (iz * nx + ix) as usize;
                let Some(ground_cy) = self.col_ground_cy.get(idx).copied() else { return false; };

                let dy = k.y - ground_cy;
                dy >= y_band_min() && dy <= y_band_max()
            });

            self.queued_set.clear();
            self.queued_set.extend(self.build_queue.iter().copied());

            sort_queue_near_first(&mut self.build_queue, center, cam_fwd);
        }

        // --- Dispatch builds ---
        while self.in_flight < config::MAX_IN_FLIGHT {
            let Some(k) = self.build_queue.pop_front() else { break; };
            self.queued_set.remove(&k);

            if !self.in_keep(center, k) {
                self.cancel_token(k).store(true, Ordering::Relaxed);
                self.chunks.remove(&k);
                continue;
            }

            if self.cache.get(&k).is_some() {
                self.chunks.remove(&k);
                let _ = self.try_promote_from_cache(center, k);
                continue;
            }

            if matches!(self.chunks.get(&k), Some(ChunkState::Queued)) {
                self.chunks.insert(k, ChunkState::Building);

                let cancel = self.cancel_token(k);
                cancel.store(false, Ordering::Relaxed);

                match self.tx_job.try_send(BuildJob { key: k, cancel: cancel.clone() }) {
                    Ok(()) => self.in_flight += 1,
                    Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                        self.chunks.insert(k, ChunkState::Queued);
                        self.build_queue.push_front(k);
                        self.queued_set.insert(k);
                        break;
                    }
                }
            }
        }

        // --- Harvest done builds (bounded per frame) ---
        let done_backlog = self.rx_done.len(); // crossbeam_channel Receiver::len() works for bounded channels
        let max_done = (16 + done_backlog / 2).clamp(16, 64);

        let mut done_count = 0usize;
        while done_count < max_done {
            let Ok(done) = self.rx_done.try_recv() else { break; };
            done_count += 1;

            if self.in_flight > 0 { self.in_flight -= 1; }

            // Drop stale completions (job from an old cancel token)
            let Some(cur_cancel) = self.cancels.get(&done.key) else { continue; };
            if !Arc::ptr_eq(cur_cancel, &done.cancel) { continue; }

            if done.canceled || done.cancel.load(Ordering::Relaxed) {
                self.chunks.remove(&done.key);
                continue;
            }

            if !self.in_keep(center, done.key) {
                self.cancel_token(done.key).store(true, Ordering::Relaxed);
                self.chunks.remove(&done.key);
                continue;
            }

            self.on_build_done(center, done.key, done.nodes, done.macro_words, done.ropes, done.colinfo_words);
        }

        // --- Grid rebuild if needed ---
        let grid_changed = self.grid_dirty;
        if self.grid_dirty {
            self.rebuild_grid(center);
            self.grid_dirty = false;
        }

        grid_changed
    }

    fn evict_one_farthest(&mut self, center: ChunkKey, protect: ChunkKey) -> bool {
        if self.slot_to_key.is_empty() {
            return false;
        }

        // Pass 1: evict farthest chunk that is OUTSIDE active xz.
        let mut best_outside: Option<(f32, ChunkKey)> = None;

        for &k in &self.slot_to_key {
            if k == protect { continue; }
            if self.in_active_xz(center, k) { continue; }

            let dx = (k.x - center.x) as f32;
            let dz = (k.z - center.z) as f32;
            let dy = (k.y - center.y) as f32;
            let d = dx * dx + dz * dz + 4.0 * dy * dy;

            if best_outside.map_or(true, |(bd, _)| d > bd) {
                best_outside = Some((d, k));
            }
        }

        if let Some((_, k)) = best_outside {
            self.unload_chunk(k);
            return true;
        }

        // Pass 2: if everything is inside ACTIVE, evict farthest overall (last resort).
        let mut best_any: Option<(f32, ChunkKey)> = None;

        for &k in &self.slot_to_key {
            if k == protect { continue; }

            let dx = (k.x - center.x) as f32;
            let dz = (k.z - center.z) as f32;
            let dy = (k.y - center.y) as f32;
            let d = dx * dx + dz * dz + 4.0 * dy * dy;

            if best_any.map_or(true, |(bd, _)| d > bd) {
                best_any = Some((d, k));
            }
        }

        if let Some((_, k)) = best_any {
            self.unload_chunk(k);
            return true;
        }

        false
    }


    fn try_promote_from_cache(&mut self, center: ChunkKey, key: ChunkKey) -> bool {
        let Some(e) = self.cache.get(&key) else { return false; };

        let nodes = e.nodes.clone();
        let macro_words = e.macro_words.clone();
        let ropes = e.ropes.clone();
        let colinfo_words = e.colinfo_words.clone();

        self.cache.touch(key);
        self.try_make_uploading(center, key, nodes, macro_words, ropes, colinfo_words)
    }

    fn on_build_done(
        &mut self,
        center: ChunkKey,
        key: ChunkKey,
        nodes: Vec<NodeGpu>,
        macro_words: Vec<u32>,
        ropes: Vec<NodeRopesGpu>,
        colinfo_words: Vec<u32>,
    ) {
        if let Some(c) = self.cancels.get(&key) {
            if c.load(Ordering::Relaxed) {
                self.chunks.remove(&key);
                return;
            }
        }

        let nodes_arc: Arc<[NodeGpu]> = nodes.into();
        let macro_arc: Arc<[u32]> = macro_words.into();
        let ropes_arc: Arc<[NodeRopesGpu]> = ropes.into();
        let colinfo_arc: Arc<[u32]> = colinfo_words.into();

        self.cache.put(key, nodes_arc.clone(), macro_arc.clone(), ropes_arc.clone(), colinfo_arc.clone());

        if matches!(self.chunks.get(&key), Some(ChunkState::Resident(_))) {
            return;
        }

        let ok = self.try_make_uploading(center, key, nodes_arc, macro_arc, ropes_arc, colinfo_arc);
        if !ok {
            self.chunks.remove(&key);
        }
    }

    fn try_make_uploading(
        &mut self,
        center: ChunkKey,
        key: ChunkKey,
        nodes: Arc<[NodeGpu]>,
        macro_words: Arc<[u32]>,
        ropes: Arc<[NodeRopesGpu]>,
        colinfo_words: Arc<[u32]>,
    ) -> bool {
        if matches!(self.chunks.get(&key), Some(ChunkState::Resident(_))) {
            return true;
        }

        let need = nodes.len() as u32;
        if need == 0 { self.chunks.remove(&key); return false; }
        if macro_words.len() != MACRO_WORDS_PER_CHUNK_USIZE { self.chunks.remove(&key); return false; }
        if ropes.len() != nodes.len() { self.chunks.remove(&key); return false; }
        if colinfo_words.len() != COLINFO_WORDS_PER_CHUNK_USIZE { self.chunks.remove(&key); return false; }

        let mut node_base = self.arena.alloc(need);
        if node_base.is_none() {
            for _ in 0..EVICT_ATTEMPTS {
                if !self.evict_one_farthest(center, key) { break; }
                node_base = self.arena.alloc(need);
                if node_base.is_some() { break; }
            }
        }

        let Some(node_base) = node_base else {
            self.chunks.remove(&key);
            return false;
        };

        let slot = self.slot_to_key.len() as u32;
        self.slot_to_key.push(key);

        self.slot_macro.push(macro_words.clone());
        self.slot_colinfo.push(colinfo_words.clone());

        let macro_base = slot * MACRO_WORDS_PER_CHUNK;
        let colinfo_base = slot * COLINFO_WORDS_PER_CHUNK;

        let origin_vox = [
            key.x * config::CHUNK_SIZE as i32,
            key.y * config::CHUNK_SIZE as i32,
            key.z * config::CHUNK_SIZE as i32,
        ];

        let meta = ChunkMetaGpu {
            origin: [origin_vox[0], origin_vox[1], origin_vox[2], 0],
            node_base,
            node_count: need,
            macro_base,
            colinfo_base,
        };

        self.chunk_meta.push(meta);

        self.chunks.insert(
            key,
            ChunkState::Uploading(Uploading { slot, node_base, node_count: need, uploaded: false }),
        );

        self.enqueue_upload(ChunkUpload {
            key,
            slot,
            meta,
            node_base,
            nodes,
            macro_words,
            ropes,
            colinfo_words,
            completes_residency: true,
        });


        true
    }

    #[inline]
    fn enqueue_slot_rewrite(&mut self, slot: usize) {
        let key = self.slot_to_key[slot];
        let slot_u32 = slot as u32;

        self.enqueue_upload(ChunkUpload {
            key,
            slot: slot_u32,
            meta: self.chunk_meta[slot],
            node_base: 0,
            nodes: Arc::<[NodeGpu]>::from(Vec::<NodeGpu>::new()),
            macro_words: self.slot_macro[slot].clone(),
            ropes: Arc::<[NodeRopesGpu]>::from(Vec::<NodeRopesGpu>::new()),
            colinfo_words: self.slot_colinfo[slot].clone(),
            completes_residency: false,
        });

    }

    #[inline]
    fn swap_slots(&mut self, a: usize, b: usize) {
        if a == b { return; }

        let ka = self.slot_to_key[a];
        let kb = self.slot_to_key[b];

        self.slot_to_key.swap(a, b);
        self.chunk_meta.swap(a, b);
        self.slot_macro.swap(a, b);
        self.slot_colinfo.swap(a, b);

        self.chunk_meta[a].macro_base = (a as u32) * MACRO_WORDS_PER_CHUNK;
        self.chunk_meta[a].colinfo_base = (a as u32) * COLINFO_WORDS_PER_CHUNK;
        self.chunk_meta[b].macro_base = (b as u32) * MACRO_WORDS_PER_CHUNK;
        self.chunk_meta[b].colinfo_base = (b as u32) * COLINFO_WORDS_PER_CHUNK;

        if let Some(st) = self.chunks.get_mut(&ka) {
            match st {
                ChunkState::Resident(r) => r.slot = b as u32,
                ChunkState::Uploading(u) => u.slot = b as u32,
                _ => {}
            }
        }
        if let Some(st) = self.chunks.get_mut(&kb) {
            match st {
                ChunkState::Resident(r) => r.slot = a as u32,
                ChunkState::Uploading(u) => u.slot = a as u32,
                _ => {}
            }
        }
    }

    fn unload_chunk(&mut self, key: ChunkKey) {
        let Some(state) = self.chunks.remove(&key) else { return; };

        match state {
            ChunkState::Resident(res) => {
                self.arena.free(res.node_base, res.node_count);
                let dead = res.slot as usize;

                let last_res = self.resident_slots.saturating_sub(1);
                debug_assert!(dead < self.resident_slots, "resident slot out of prefix");

                if dead != last_res {
                    self.swap_slots(dead, last_res);
                    self.enqueue_slot_rewrite(dead);
                }

                self.resident_slots = last_res;

                let remove_idx = last_res;
                let last_slot = self.slot_to_key.len().saturating_sub(1);

                if remove_idx != last_slot {
                    self.swap_slots(remove_idx, last_slot);
                    self.enqueue_slot_rewrite(remove_idx);
                }

                self.slot_to_key.pop();
                self.chunk_meta.pop();
                self.slot_macro.pop();
                self.slot_colinfo.pop();

                self.grid_dirty = true;
            }

            ChunkState::Uploading(up) => {
                self.arena.free(up.node_base, up.node_count);
                let dead = up.slot as usize;

                debug_assert!(dead >= self.resident_slots, "uploading slot inside resident prefix");

                let last_slot = self.slot_to_key.len().saturating_sub(1);
                if dead != last_slot {
                    self.swap_slots(dead, last_slot);
                    self.enqueue_slot_rewrite(dead);
                }

                self.slot_to_key.pop();
                self.chunk_meta.pop();
                self.slot_macro.pop();
                self.slot_colinfo.pop();
            }

            ChunkState::Queued | ChunkState::Building => {
                self.cancel_token(key).store(true, Ordering::Relaxed);
                self.queued_set.remove(&key);
                self.grid_dirty = true;
            }
        }

        self.cancels.remove(&key);
    }

    fn rebuild_grid(&mut self, center: ChunkKey) {
        let nx = (2 * config::KEEP_RADIUS + 1) as u32;
        let nz = nx;
        let ny = GRID_Y_COUNT;

        self.grid_dims = [nx, ny, nz];
        self.grid_origin_chunk = self.keep_origin_for(center);

        let needed = (nx * ny * nz) as usize;
        if self.chunk_grid.len() != needed {
            self.chunk_grid.resize(needed, INVALID_U32);
        }
        self.chunk_grid.fill(INVALID_U32);

        // IMPORTANT: include any slot that is actually usable by the GPU.
        for slot in 0..self.slot_to_key.len() {
            let k = self.slot_to_key[slot];

            let ready = match self.chunks.get(&k) {
                Some(ChunkState::Resident(_)) => true,
                Some(ChunkState::Uploading(up)) => up.uploaded, // uploaded => meta + macro/colinfo + nodes are on GPU
                _ => false,
            };

            if !ready { continue; }

            if let Some(idx) = self.grid_index_for_chunk(k) {
                self.chunk_grid[idx] = slot as u32;
            }
        }
    }


    #[inline]
    fn grid_index_for_chunk(&self, k: ChunkKey) -> Option<usize> {
        let [ox, oy, oz] = self.grid_origin_chunk;
        let [nx, ny, nz] = self.grid_dims;

        let ix = k.x - ox;
        let iy = k.y - oy;
        let iz = k.z - oz;

        if ix < 0 || iy < 0 || iz < 0 { return None; }

        let ix = ix as u32;
        let iy = iy as u32;
        let iz = iz as u32;

        if ix >= nx || iy >= ny || iz >= nz { return None; }

        let idx = (iz * ny * nx) + (iy * nx) + ix;
        Some(idx as usize)
    }

    fn rebuild_column_ground(&mut self, world: &WorldGen, center: ChunkKey) {
        let [ox, _, oz] = self.keep_origin_for(center);

        let nx = (2 * config::KEEP_RADIUS + 1) as i32;
        let nz = nx;

        self.col_ground_cy.resize((nx * nz) as usize, 0);

        let cs = config::CHUNK_SIZE as i32;
        let half = cs / 2;

        for dz in 0..nz {
            for dx in 0..nx {
                let cx = ox + dx;
                let cz = oz + dz;

                let wx = cx * cs + half;
                let wz = cz * cs + half;

                let ground_y_vox = world.ground_height(wx, wz);
                self.col_ground_cy[(dz * nx + dx) as usize] = ground_y_vox.div_euclid(cs);
            }
        }
    }

    // -------------------------
    // Public API (unchanged)
    // -------------------------

    pub fn chunk_count(&self) -> u32 {
        self.resident_slots as u32
    }

    pub fn grid_origin(&self) -> [i32; 3] {
        self.grid_origin_chunk
    }

    pub fn grid_dims(&self) -> [u32; 3] {
        self.grid_dims
    }

    pub fn chunk_grid(&self) -> &[u32] {
        &self.chunk_grid
    }

    #[inline]
    fn upload_bytes(u: &ChunkUpload) -> usize {
        let mut b = size_of::<ChunkMetaGpu>();
        b += u.nodes.len() * size_of::<NodeGpu>();
        b += u.macro_words.len() * size_of::<u32>();
        b += u.ropes.len() * size_of::<NodeRopesGpu>();
        b += u.colinfo_words.len() * size_of::<u32>();
        b
    }

    pub fn take_uploads(&mut self) -> Vec<ChunkUpload> {
        self.uploads.drain(..).collect()
    }

    pub fn take_uploads_budgeted(&mut self) -> Vec<ChunkUpload> {
        let backlog = self.uploads_len_total();

        let max_uploads =
            (MAX_UPLOADS_PER_FRAME + backlog / 4).clamp(MAX_UPLOADS_PER_FRAME, 32);
        let max_bytes =
            (MAX_UPLOAD_BYTES_PER_FRAME + backlog * (256 << 10)).clamp(MAX_UPLOAD_BYTES_PER_FRAME, 16 << 20);

        let mut out = Vec::new();
        let mut bytes = 0usize;

        // Helper: pop next upload in priority order (rewrites -> active -> other)
        let mut pop_next = |mgr: &mut ChunkManager| -> Option<(u8, ChunkUpload)> {
            if let Some(u) = mgr.uploads_rewrite.pop_front() { return Some((0, u)); }
            if let Some(u) = mgr.uploads_active.pop_front()  { return Some((1, u)); }
            if let Some(u) = mgr.uploads_other.pop_front()   { return Some((2, u)); }
            None
        };

        // Helper: push back to the front of the same queue we popped from (so we don't reorder)
        let mut push_front_same = |mgr: &mut ChunkManager, which: u8, u: ChunkUpload| {
            match which {
                0 => mgr.uploads_rewrite.push_front(u),
                1 => mgr.uploads_active.push_front(u),
                _ => mgr.uploads_other.push_front(u),
            }
        };

        while let Some((which, mut u)) = pop_next(self) {
            if out.len() >= max_uploads {
                push_front_same(self, which, u);
                break;
            }

            // Validate slot + update meta bases (same behavior as before)
            let slot = match self.chunks.get(&u.key) {
                Some(ChunkState::Resident(r)) => r.slot,
                Some(ChunkState::Uploading(up)) => up.slot,
                _ => continue, // stale
            };

            u.slot = slot;
            u.meta.macro_base = slot * MACRO_WORDS_PER_CHUNK;
            u.meta.colinfo_base = slot * COLINFO_WORDS_PER_CHUNK;

            let ub = Self::upload_bytes(&u);

            // Respect byte budget: if adding this would exceed and we already have something, stop.
            if bytes + ub > max_bytes && !out.is_empty() {
                push_front_same(self, which, u);
                break;
            }

            bytes += ub;
            out.push(u);
        }

        out
    }


    #[inline(always)]
    fn upload_dist_score(k: ChunkKey, c: ChunkKey) -> f32 {
        let dx = (k.x - c.x) as f32;
        let dz = (k.z - c.z) as f32;
        let dy = (k.y - c.y) as f32;
        dx.abs() + dz.abs() + 2.0 * dy.abs()
    }



    pub fn commit_uploads_applied(&mut self, applied: &[ChunkUpload]) -> bool {
        let mut any_promoted = false;

        // Mark uploads as applied.
        for u in applied {
            if !u.completes_residency { continue; }
            if let Some(ChunkState::Uploading(up)) = self.chunks.get_mut(&u.key) {
                up.uploaded = true;
            }
        }

        // We’ll score candidates by distance to current center; ACTIVE gets a big bonus.
        let center_opt = self.last_center;

        loop {
            if self.resident_slots >= self.slot_to_key.len() { break; }

            // If the next slot is ready, promote it directly.
            let next_key = self.slot_to_key[self.resident_slots];
            let next_ready = matches!(
                self.chunks.get(&next_key),
                Some(ChunkState::Uploading(Uploading { uploaded: true, .. }))
            );

            if !next_ready {
                // Find the best READY uploading chunk in [resident_slots..].
                let mut best: Option<(f32, usize)> = None;

                for s in self.resident_slots..self.slot_to_key.len() {
                    let k = self.slot_to_key[s];

                    let Some(ChunkState::Uploading(up)) = self.chunks.get(&k) else { continue; };
                    if !up.uploaded { continue; }

                    let mut score = if let Some(c) = center_opt {
                        Self::upload_dist_score(k, c)
                    } else {
                        s as f32
                    };

                    if let Some(c) = center_opt {
                        if self.in_active_xz(c, k) {
                            score -= 10_000.0; // huge preference for ACTIVE
                        }
                    }

                    if best.map_or(true, |(bs, _)| score < bs) {
                        best = Some((score, s));
                    }
                }

                let Some((_, best_slot)) = best else {
                    // No ready uploads to promote this frame.
                    break;
                };

                if best_slot != self.resident_slots {
                    // Swap the ready chunk into the promotion frontier slot.
                    self.swap_slots(best_slot, self.resident_slots);

                    // IMPORTANT:
                    // The chunk we swapped in was already uploaded, but its slot changed,
                    // so macro/colinfo/meta bases changed => rewrite this slot ASAP.
                    self.enqueue_slot_rewrite(self.resident_slots);

                    // If the chunk swapped out was ALSO already uploaded, it now lives at best_slot
                    // and needs a rewrite too (avoid visual corruption).
                    let swapped_out_key = self.slot_to_key[best_slot];
                    let swapped_out_uploaded = matches!(
                        self.chunks.get(&swapped_out_key),
                        Some(ChunkState::Uploading(Uploading { uploaded: true, .. }))
                    );
                    if swapped_out_uploaded {
                        self.enqueue_slot_rewrite(best_slot);
                    }
                }
            }

            // Now the frontier slot should be ready; promote it.
            let slot = self.resident_slots;
            let key = self.slot_to_key[slot];

            let ready = matches!(
                self.chunks.get(&key),
                Some(ChunkState::Uploading(Uploading { uploaded: true, .. }))
            );
            if !ready { break; }

            if let Some(st) = self.chunks.get_mut(&key) {
                if let ChunkState::Uploading(up) = st {
                    let slot_u32 = up.slot;
                    let node_base = up.node_base;
                    let node_count = up.node_count;
                    *st = ChunkState::Resident(Resident { slot: slot_u32, node_base, node_count });
                    self.resident_slots += 1;
                    any_promoted = true;
                    continue;
                }
            }

            break;
        }

        if any_promoted {
            self.grid_dirty = true;
            if let Some(center) = self.last_center {
                self.rebuild_grid(center);
                self.grid_dirty = false;
            }
            return true;
        }

        false
    }


    #[inline]
    fn compute_ground_cy_at_column(&self, world: &WorldGen, cx: i32, cz: i32) -> i32 {
        let cs = config::CHUNK_SIZE as i32;
        let half = cs / 2;
        let wx = cx * cs + half;
        let wz = cz * cs + half;
        let ground_y_vox = world.ground_height(wx, wz);
        ground_y_vox.div_euclid(cs)
    }

    fn update_column_ground_cache(
        &mut self,
        world: &WorldGen,
        old_origin: [i32; 3],
        new_origin: [i32; 3],
    ) {
        let nx = (2 * config::KEEP_RADIUS + 1) as i32;
        let nz = nx;
        let len = (nx * nz) as usize;

        // First-time or size mismatch => full rebuild.
        if self.col_ground_cy.len() != len || self.col_ground_cy.is_empty() {
            self.col_ground_cy.resize(len, 0);
            let ox = new_origin[0];
            let oz = new_origin[2];
            for dz in 0..nz {
                for dx in 0..nx {
                    let cx = ox + dx;
                    let cz = oz + dz;
                    self.col_ground_cy[(dz * nx + dx) as usize] =
                        self.compute_ground_cy_at_column(world, cx, cz);
                }
            }
            return;
        }

        let dx_chunks = new_origin[0] - old_origin[0];
        let dz_chunks = new_origin[2] - old_origin[2];

        if dx_chunks == 0 && dz_chunks == 0 {
            return;
        }

        // Teleport / huge jump => full rebuild.
        if dx_chunks.abs() >= nx || dz_chunks.abs() >= nz {
            let ox = new_origin[0];
            let oz = new_origin[2];
            for dz in 0..nz {
                for dx in 0..nx {
                    let cx = ox + dx;
                    let cz = oz + dz;
                    self.col_ground_cy[(dz * nx + dx) as usize] =
                        self.compute_ground_cy_at_column(world, cx, cz);
                }
            }
            return;
        }

        let old = std::mem::take(&mut self.col_ground_cy);
        let mut newv = vec![0i32; len];

        let ox_new = new_origin[0];
        let oz_new = new_origin[2];

        for iz in 0..nz {
            for ix in 0..nx {
                // old index = new index minus movement
                let sx = ix - dx_chunks;
                let sz = iz - dz_chunks;

                let dst_idx = (iz * nx + ix) as usize;

                if sx >= 0 && sx < nx && sz >= 0 && sz < nz {
                    let src_idx = (sz * nx + sx) as usize;
                    newv[dst_idx] = old[src_idx];
                } else {
                    let cx = ox_new + ix;
                    let cz = oz_new + iz;
                    newv[dst_idx] = self.compute_ground_cy_at_column(world, cx, cz);
                }
            }
        }

        self.col_ground_cy = newv;
    }

    #[inline]
    fn prioritize_residency_frontier_uploads(&mut self, window: usize) {
        if self.resident_slots >= self.slot_to_key.len() {
            return;
        }

        let end = (self.resident_slots + window).min(self.slot_to_key.len());

        // For each upcoming slot, ensure its upload (if any) is near the front.
        for s in self.resident_slots..end {
            let key = self.slot_to_key[s];

            // Find first pending upload for that key.
            if let Some(pos) = self.uploads.iter().position(|u| u.key == key) {
                // Already front-ish? keep it simple.
                if pos == 0 {
                    continue;
                }

                // Rotate so that the found element moves to the front, preserving relative order.
                // (VecDeque has rotate_left / rotate_right stable in std.)
                self.uploads.rotate_left(pos);
            }
        }
    }

    #[inline(always)]
    fn in_active_xz(&self, center: ChunkKey, k: ChunkKey) -> bool {
        let dx = (k.x - center.x).abs();
        let dz = (k.z - center.z).abs();
        dx <= config::ACTIVE_RADIUS && dz <= config::ACTIVE_RADIUS
    }

    #[inline]
    fn uploads_len_total(&self) -> usize {
        self.uploads_rewrite.len() + self.uploads_active.len() + self.uploads_other.len()
    }

    #[inline]
    fn enqueue_upload(&mut self, u: ChunkUpload) {
        if !u.completes_residency {
            // rewrites should be ASAP; preserve old behavior (push_front)
            self.uploads_rewrite.push_front(u);
            return;
        }

        // If we don't know center yet, treat as "other"
        let Some(center) = self.last_center else {
            self.uploads_other.push_back(u);
            return;
        };

        if self.in_active_xz(center, u.key) {
            self.uploads_active.push_back(u);
        } else {
            self.uploads_other.push_back(u);
        }
    }

    /// When the center changes, ACTIVE membership changes too.
    /// Rebucket only active/other; rewrites remain rewrites.
    fn rebucket_uploads_for_center(&mut self, center: ChunkKey) {
        let mut new_active = VecDeque::with_capacity(self.uploads_active.len());
        let mut new_other  = VecDeque::with_capacity(self.uploads_other.len());

        let ar = config::ACTIVE_RADIUS;

        let mut is_active = |k: ChunkKey| {
            let dx = (k.x - center.x).abs();
            let dz = (k.z - center.z).abs();
            dx <= ar && dz <= ar
        };

        for u in self.uploads_active.drain(..) {
            if is_active(u.key) { new_active.push_back(u); }
            else { new_other.push_back(u); }
        }

        for u in self.uploads_other.drain(..) {
            if is_active(u.key) { new_active.push_back(u); }
            else { new_other.push_back(u); }
        }

        self.uploads_active = new_active;
        self.uploads_other  = new_other;
    }





}
