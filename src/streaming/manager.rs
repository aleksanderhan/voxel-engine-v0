// src/streaming/manager.rs
// ------------------------
// Chunk streaming + background SVO building + CPU cache.

use std::collections::VecDeque;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use std::mem::size_of;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use crossbeam_channel::{unbounded, Receiver, Sender};
use glam::{Vec2, Vec3};

use crate::{
    config,
    render::gpu_types::{ChunkMetaGpu, NodeGpu, NodeRopesGpu},
    svo::{build_chunk_svo_sparse_cancelable_with_scratch, BuildScratch},
    world::WorldGen,
};

use super::NodeArena;

const INVALID_U32: u32 = 0xFFFF_FFFF;

// Vertical band dy in [-1..=2]
const GRID_Y_MIN_DY: i32 = -1;
const GRID_Y_COUNT: u32 = 4;

const EVICT_ATTEMPTS: usize = 8;

// 8^3 bits = 512 bits = 16 u32
const MACRO_WORDS_PER_CHUNK: u32 = 16;
const MACRO_WORDS_PER_CHUNK_USIZE: usize = 16;

// 64x64 columns, packed 2x u16 per u32 => 2048 u32 per chunk
const COLINFO_WORDS_PER_CHUNK: u32 = 2048;
const COLINFO_WORDS_PER_CHUNK_USIZE: usize = 2048;

const MAX_UPLOADS_PER_FRAME: usize = 8;            // start 6–12
const MAX_UPLOAD_BYTES_PER_FRAME: usize = 4 << 20; // start 2–8 MB


#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
struct ChunkKey {
    x: i32,
    y: i32,
    z: i32,
}

enum ChunkState {
    Queued,
    Building,
    Uploading(Uploading),
    Resident(Resident),
}

#[derive(Clone, Debug)]
struct Uploading {
    slot: u32,
    node_base: u32,
    node_count: u32,
}


#[derive(Clone, Debug)]
struct Resident {
    slot: u32,
    node_base: u32,
    node_count: u32,
}

#[derive(Clone, Debug)]
struct BuildJob {
    key: ChunkKey,
    cancel: Arc<AtomicBool>,
}

struct BuildDone {
    key: ChunkKey,
    cancel: Arc<AtomicBool>,
    canceled: bool,
    nodes: Vec<NodeGpu>,
    macro_words: Vec<u32>,
    ropes: Vec<NodeRopesGpu>,
    colinfo_words: Vec<u32>,
}

pub struct ChunkUpload {
    pub slot: u32,
    pub meta: ChunkMetaGpu,

    pub node_base: u32,
    pub nodes: Arc<[NodeGpu]>,

    pub macro_words: Arc<[u32]>,

    pub ropes: Arc<[NodeRopesGpu]>,

    pub colinfo_words: Arc<[u32]>,

    pub completes_residency: bool,
}

// -----------------------------
// CPU cache (budgeted, LRU-ish)
// -----------------------------

#[derive(Clone)]
struct CachedChunk {
    nodes: Arc<[NodeGpu]>,
    ropes: Arc<[NodeRopesGpu]>,
    macro_words: Arc<[u32]>,
    colinfo_words: Arc<[u32]>,
    bytes: usize,
    stamp: u64,
}

fn spawn_workers(gen: Arc<WorldGen>, rx_job: Receiver<BuildJob>, tx_done: Sender<BuildDone>) {
    for _ in 0..config::WORKER_THREADS {
        let gen = gen.clone();
        let rx_job = rx_job.clone();
        let tx_done = tx_done.clone();

        std::thread::spawn(move || {
            let mut scratch = BuildScratch::new();

            while let Ok(job) = rx_job.recv() {
                let k = job.key;

                if job.cancel.load(Ordering::Relaxed) {
                    let _ = tx_done.send(BuildDone {
                        key: k,
                        cancel: job.cancel,
                        canceled: true,
                        nodes: Vec::new(),
                        macro_words: Vec::new(),
                        ropes: Vec::new(),
                        colinfo_words: Vec::new()
                    });
                    continue;
                }

                let origin = [
                    k.x * config::CHUNK_SIZE as i32,
                    k.y * config::CHUNK_SIZE as i32,
                    k.z * config::CHUNK_SIZE as i32,
                    0,
                ];

                let (nodes, macro_words, ropes, colinfo_words) = build_chunk_svo_sparse_cancelable_with_scratch(
                    &gen,
                    [origin[0], origin[1], origin[2]],
                    config::CHUNK_SIZE,
                    job.cancel.as_ref(),
                    &mut scratch,
                );

                let canceled = job.cancel.load(Ordering::Relaxed);
                let (nodes, macro_words, ropes) = if canceled {
                    (Vec::new(), Vec::new(), Vec::new())
                } else {
                    (nodes, macro_words, ropes)
                };

                let _ = tx_done.send(BuildDone {
                    key: k,
                    cancel: job.cancel,
                    canceled,
                    nodes,
                    macro_words,
                    ropes,
                    colinfo_words
                });
            }
        });
    }
}

fn sort_queue_near_first(queue: &mut VecDeque<ChunkKey>, center: ChunkKey, cam_fwd: Vec3) {
    let mut v: Vec<ChunkKey> = queue.drain(..).collect();

    let mut f = Vec2::new(cam_fwd.x, cam_fwd.z);
    if f.length_squared() > 1e-6 {
        f = f.normalize();
    } else {
        f = Vec2::ZERO;
    }

    v.sort_by(|a, b| {
        let sa = chunk_priority_score(*a, center, f);
        let sb = chunk_priority_score(*b, center, f);
        sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
    });

    queue.extend(v);
}

fn chunk_priority_score(k: ChunkKey, c: ChunkKey, fwd_xz: Vec2) -> f32 {
    let dx = (k.x - c.x) as f32;
    let dz = (k.z - c.z) as f32;
    let dy = (k.y - c.y) as f32;

    let base = dx.abs() + dz.abs() + 2.0 * dy.abs();
    let dir = dx * fwd_xz.x + dz * fwd_xz.y;

    let front_bonus = 0.75;
    let behind_penalty = 0.25;

    let bias = if dir >= 0.0 {
        -front_bonus * dir
    } else {
        -behind_penalty * dir
    };

    base + bias
}

pub struct ChunkManager {
    chunks: HashMap<ChunkKey, ChunkState>,
    build_queue: VecDeque<ChunkKey>,
    queued_set: HashSet<ChunkKey>,
    last_center: Option<ChunkKey>,

    cancels: HashMap<ChunkKey, Arc<AtomicBool>>,

    tx_job: Sender<BuildJob>,
    rx_done: Receiver<BuildDone>,
    in_flight: usize,

    slot_to_key: Vec<ChunkKey>,
    chunk_meta: Vec<ChunkMetaGpu>,
    uploads: VecDeque<ChunkUpload>,

    changed: bool,
    grid_dirty: bool,

    arena: NodeArena,

    grid_origin_chunk: [i32; 3],
    grid_dims: [u32; 3],
    chunk_grid: Vec<u32>,

    cache: HashMap<ChunkKey, CachedChunk>,
    cache_lru: VecDeque<(ChunkKey, u64)>,
    cache_stamp: u64,
    cache_bytes: usize,

    active_offsets: Vec<(i32, i32, i32)>,
    to_unload: Vec<ChunkKey>,

    slot_macro: Vec<Arc<[u32]>>,
    slot_colinfo: Vec<Arc<[u32]>>,

    resident_slots: usize,
}

impl ChunkManager {
    pub fn new(gen: Arc<WorldGen>) -> Self {
        let (tx_job, rx_job) = unbounded::<BuildJob>();
        let (tx_done, rx_done) = unbounded::<BuildDone>();
        spawn_workers(gen.clone(), rx_job, tx_done);

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
            uploads: VecDeque::new(),

            changed: false,
            grid_dirty: true,

            arena: NodeArena::new(node_capacity),

            grid_origin_chunk: [0, 0, 0],
            grid_dims: [nx, ny, nz],
            chunk_grid: vec![INVALID_U32; grid_len],

            cache: HashMap::default(),
            cache_lru: VecDeque::default(),
            cache_stamp: 1,
            cache_bytes: 0,

            active_offsets,
            to_unload: Vec::new(),

            slot_macro: Vec::new(),
            slot_colinfo: Vec::new(),

            resident_slots: 0,

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

    #[inline(always)]
    fn y_band_min() -> i32 {
        GRID_Y_MIN_DY
    }

    #[inline(always)]
    fn y_band_max() -> i32 {
        GRID_Y_MIN_DY + GRID_Y_COUNT as i32 - 1
    }

    #[inline(always)]
    fn in_keep(world: &WorldGen, center: ChunkKey, k: ChunkKey) -> bool {
        let dx = k.x - center.x;
        let dz = k.z - center.z;

        if dx < -config::KEEP_RADIUS || dx > config::KEEP_RADIUS {
            return false;
        }
        if dz < -config::KEEP_RADIUS || dz > config::KEEP_RADIUS {
            return false;
        }

        // Compute ground chunk-Y for THIS column (k.x,k.z), not the camera column.
        let cs = config::CHUNK_SIZE as i32;
        let half = cs / 2;
        let wx = k.x * cs + half;
        let wz = k.z * cs + half;
        let ground_y_vox = world.ground_height(wx, wz);
        let ground_cy = ground_y_vox.div_euclid(cs);

        let dy = k.y - ground_cy;
        dy >= Self::y_band_min() && dy <= Self::y_band_max()
    }

    pub fn update(&mut self, world: &Arc<WorldGen>, cam_pos_m: Vec3, cam_fwd: Vec3) -> bool {
        let cam_vx = (cam_pos_m.x / config::VOXEL_SIZE_M_F32).floor() as i32;
        let cam_vz = (cam_pos_m.z / config::VOXEL_SIZE_M_F32).floor() as i32;

        let ccx = cam_vx.div_euclid(config::CHUNK_SIZE as i32);
        let ccz = cam_vz.div_euclid(config::CHUNK_SIZE as i32);

        let ground_y_vox = world.ground_height(cam_vx, cam_vz);
        let ground_cy = ground_y_vox.div_euclid(config::CHUNK_SIZE as i32);

        let center = ChunkKey { x: ccx, y: ground_cy, z: ccz };

        // ACTIVE
        let n_active = self.active_offsets.len();
        for i in 0..n_active {
            let (dx, dy, dz) = self.active_offsets[i];

            let cs = config::CHUNK_SIZE as i32;
            let half = cs / 2;

            let x = center.x + dx;
            let z = center.z + dz;

            // ground chunk-Y for THIS (x,z) column
            let wx = x * cs + half;
            let wz = z * cs + half;
            let ground_y_vox = world.ground_height(wx, wz);
            let ground_cy = ground_y_vox.div_euclid(cs);

            let k = ChunkKey {
                x,
                y: ground_cy + dy, // dy is still your vertical band offset
                z,
            };


            match self.chunks.get(&k) {
                Some(ChunkState::Resident(_))
                | Some(ChunkState::Uploading(_))
                | Some(ChunkState::Queued)
                | Some(ChunkState::Building) => {
                    // already exists / in progress, do nothing
                }

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

        // Unload outside KEEP
        self.to_unload.clear();
        for &k in self.chunks.keys() {
            if !Self::in_keep(world.as_ref(), center, k) {
                self.to_unload.push(k);
            }
        }


        let mut unload_list = std::mem::take(&mut self.to_unload);
        for k in unload_list.drain(..) {
            self.unload_chunk(k);
        }
        self.to_unload = unload_list;

        let center_changed = self.last_center.map_or(true, |c| c != center);
        if center_changed {
            self.last_center = Some(center);

            self.build_queue.retain(|k| {
                Self::in_keep(world.as_ref(), center, *k) && matches!(self.chunks.get(k), Some(ChunkState::Queued))
            });

            self.queued_set.clear();
            self.queued_set.extend(self.build_queue.iter().copied());

            sort_queue_near_first(&mut self.build_queue, center, cam_fwd);
        }

        // Dispatch builds
        while self.in_flight < config::MAX_IN_FLIGHT {
            let Some(k) = self.build_queue.pop_front() else { break; };
            self.queued_set.remove(&k);

            if !Self::in_keep(world.as_ref(), center, k) {
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

                if self.tx_job.send(BuildJob { key: k, cancel: cancel.clone() }).is_ok() {
                    self.in_flight += 1;
                } else {
                    self.chunks.insert(k, ChunkState::Queued);
                    if self.queued_set.insert(k) {
                        self.build_queue.push_back(k);
                    }
                    break;
                }
            }
        }

        // Harvest done builds
        while let Ok(done) = self.rx_done.try_recv() {
            if self.in_flight > 0 {
                self.in_flight -= 1;
            }

            if done.canceled || done.cancel.load(Ordering::Relaxed) {
                self.chunks.remove(&done.key);
                continue;
            }

            if !Self::in_keep(world.as_ref(), center, done.key) {
                self.cancel_token(done.key).store(true, Ordering::Relaxed);
                self.chunks.remove(&done.key);
                continue;
            }

            self.on_build_done(center, done.key, done.nodes, done.macro_words, done.ropes, done.colinfo_words);
        }

        if self.keep_origin_for(center) != self.grid_origin_chunk {
            self.grid_dirty = true;
        }

        let grid_changed = self.grid_dirty;
        if self.grid_dirty {
            self.rebuild_grid(center);
            self.grid_dirty = false;
        }

        grid_changed
    }

    fn cancel_token(&mut self, key: ChunkKey) -> Arc<AtomicBool> {
        self.cancels
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicBool::new(false)))
            .clone()
    }

    fn evict_one_farthest(&mut self, center: ChunkKey, protect: ChunkKey) -> bool {
        if self.slot_to_key.is_empty() {
            return false;
        }

        let mut best: Option<(f32, ChunkKey)> = None;
        for &k in &self.slot_to_key {
            if k == protect {
                continue;
            }
            let dx = (k.x - center.x) as f32;
            let dz = (k.z - center.z) as f32;
            let dy = (k.y - center.y) as f32;

            let d = dx * dx + dz * dz + 4.0 * dy * dy;

            if best.map_or(true, |(bd, _)| d > bd) {
                best = Some((d, k));
            }
        }

        if let Some((_, k)) = best {
            self.unload_chunk(k);
            return true;
        }

        false
    }

    // ----------------
    // Cache helpers
    // ----------------

    fn cache_touch(&mut self, key: ChunkKey) {
        if let Some(e) = self.cache.get_mut(&key) {
            self.cache_stamp = self.cache_stamp.wrapping_add(1).max(1);
            e.stamp = self.cache_stamp;
            self.cache_lru.push_back((key, e.stamp));
        }
    }

    fn cache_put(
        &mut self,
        key: ChunkKey,
        nodes: Arc<[NodeGpu]>,
        macro_words: Arc<[u32]>,
        ropes: Arc<[NodeRopesGpu]>,
        colinfo_words: Arc<[u32]>,
    ) {
        if let Some(old) = self.cache.remove(&key) {
            self.cache_bytes = self.cache_bytes.saturating_sub(old.bytes);
        }

        self.cache_stamp = self.cache_stamp.wrapping_add(1).max(1);
        let stamp = self.cache_stamp;

        let bytes =
            nodes.len() * size_of::<NodeGpu>()
            + ropes.len() * size_of::<NodeRopesGpu>()
            + macro_words.len() * size_of::<u32>()
            + colinfo_words.len() * size_of::<u32>();

        self.cache.insert(
            key,
            CachedChunk {
                nodes,
                ropes,
                macro_words,
                colinfo_words,
                bytes,
                stamp,
            },
        );

        self.cache_bytes = self.cache_bytes.saturating_add(bytes);
        self.cache_lru.push_back((key, stamp));

        self.evict_cache_as_needed();
    }

    fn evict_cache_as_needed(&mut self) {
        let budget = config::CHUNK_CACHE_BUDGET_BYTES;

        while self.cache_bytes > budget {
            let Some((k, stamp)) = self.cache_lru.pop_front() else { break; };

            let should_evict = self
                .cache
                .get(&k)
                .map(|e| e.stamp == stamp)
                .unwrap_or(false);

            if !should_evict {
                continue;
            }

            if let Some(ev) = self.cache.remove(&k) {
                self.cache_bytes = self.cache_bytes.saturating_sub(ev.bytes);
            }
        }
    }

    // -------------------------------
    // Resident creation / promotion
    // -------------------------------

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
        if need == 0 {
            self.chunks.remove(&key);
            return false;
        }
        if macro_words.len() != MACRO_WORDS_PER_CHUNK_USIZE {
            self.chunks.remove(&key);
            return false;
        }
        if ropes.len() != nodes.len() {
            self.chunks.remove(&key);
            return false;
        }
        if colinfo_words.len() != COLINFO_WORDS_PER_CHUNK_USIZE {
            self.chunks.remove(&key);
            return false;
        }

        let mut node_base = self.arena.alloc(need);
        if node_base.is_none() {
            for _ in 0..EVICT_ATTEMPTS {
                if !self.evict_one_farthest(center, key) {
                    break;
                }
                node_base = self.arena.alloc(need);
                if node_base.is_some() {
                    break;
                }
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

        let origin_vox = [
            key.x * config::CHUNK_SIZE as i32,
            key.y * config::CHUNK_SIZE as i32,
            key.z * config::CHUNK_SIZE as i32,
        ];

        let colinfo_base = slot * COLINFO_WORDS_PER_CHUNK;

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
            ChunkState::Uploading(Uploading { slot, node_base, node_count: need }),
        );

        self.uploads.push_back(ChunkUpload {
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

    fn try_promote_from_cache(&mut self, center: ChunkKey, key: ChunkKey) -> bool {
        let (nodes, macro_words, ropes) = match self.cache.get(&key) {
            Some(e) => (e.nodes.clone(), e.macro_words.clone(), e.ropes.clone()),
            None => return false,
        };
        let (nodes, macro_words, ropes, colinfo_words) = match self.cache.get(&key) {
            Some(e) => (e.nodes.clone(), e.macro_words.clone(), e.ropes.clone(), e.colinfo_words.clone()),
            None => return false,
        };

        self.cache_touch(key);
        self.try_make_uploading(center, key, nodes, macro_words, ropes, colinfo_words)
    }

    fn on_build_done(
        &mut self,
        center: ChunkKey,
        key: ChunkKey,
        nodes: Vec<NodeGpu>,
        macro_words: Vec<u32>,
        ropes: Vec<NodeRopesGpu>,
        colinfo_words: Vec<u32>
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

        self.cache_put(key, nodes_arc.clone(), macro_arc.clone(), ropes_arc.clone(), colinfo_arc.clone());

        if matches!(self.chunks.get(&key), Some(ChunkState::Resident(_))) {
            return;
        }

        let ok = self.try_make_uploading(center, key, nodes_arc, macro_arc, ropes_arc, colinfo_arc);
        if !ok {
            self.chunks.remove(&key);
        }
    }

    #[inline]
    fn enqueue_slot_rewrite(&mut self, slot: usize) {
        // Rewrite meta + per-slot macro/colinfo at the NEW slot offsets.
        // High priority: push_front so budgeting doesn't delay correctness-critical remaps.
        let slot_u32 = slot as u32;

        self.uploads.push_front(ChunkUpload {
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
    fn swap_pending_upload_slots(&mut self, a: u32, b: u32) {
        if a == b {
            return;
        }

        // Any queued uploads targeting old slot indices must follow the chunk
        // after we swap the slot contents.
        for u in self.uploads.iter_mut() {
            if u.slot == a {
                u.slot = b;
                u.meta.macro_base = b * MACRO_WORDS_PER_CHUNK;
                u.meta.colinfo_base = b * COLINFO_WORDS_PER_CHUNK;
            } else if u.slot == b {
                u.slot = a;
                u.meta.macro_base = a * MACRO_WORDS_PER_CHUNK;
                u.meta.colinfo_base = a * COLINFO_WORDS_PER_CHUNK;
            }
        }
    }

    #[inline]
    fn swap_slots(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }

        let ka = self.slot_to_key[a];
        let kb = self.slot_to_key[b];

        // Swap CPU slot ownership
        self.slot_to_key.swap(a, b);
        self.chunk_meta.swap(a, b);
        self.slot_macro.swap(a, b);
        self.slot_colinfo.swap(a, b);

        // Fix per-slot bases in meta for BOTH slots
        self.chunk_meta[a].macro_base = (a as u32) * MACRO_WORDS_PER_CHUNK;
        self.chunk_meta[a].colinfo_base = (a as u32) * COLINFO_WORDS_PER_CHUNK;
        self.chunk_meta[b].macro_base = (b as u32) * MACRO_WORDS_PER_CHUNK;
        self.chunk_meta[b].colinfo_base = (b as u32) * COLINFO_WORDS_PER_CHUNK;

        // Update state.slot for BOTH chunks
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

        // Retarget any queued uploads that referenced these slot indices
        self.swap_pending_upload_slots(a as u32, b as u32);
    }

    fn unload_chunk(&mut self, key: ChunkKey) {
        let Some(state) = self.chunks.remove(&key) else { return; };

        match state {
            ChunkState::Resident(res) => {
                // Free node arena allocation
                self.arena.free(res.node_base, res.node_count);

                let dead = res.slot as usize;

                // --- Step 1: keep resident prefix dense ---
                // Swap dead resident slot with last resident slot.
                let last_res = self.resident_slots.saturating_sub(1);
                debug_assert!(dead < self.resident_slots, "resident slot out of prefix");

                if dead != last_res {
                    self.swap_slots(dead, last_res);

                    // dead now holds a DIFFERENT resident chunk; rewrite it immediately
                    self.enqueue_slot_rewrite(dead);
                }

                // Removed resident is now sitting at `last_res` (which is about to become uploading boundary)
                self.resident_slots = last_res;

                // --- Step 2: remove physical slot entry (compact arrays) ---
                let remove_idx = last_res;
                let last_slot = self.slot_to_key.len().saturating_sub(1);

                if remove_idx != last_slot {
                    // This may swap with an Uploading slot. That's OK because swap_slots()
                    // also retargets any pending uploads.
                    self.swap_slots(remove_idx, last_slot);

                    // remove_idx now holds the moved chunk (resident or uploading); if it's resident, it
                    // must be correct immediately; if it's uploading, rewrite is still fine (and cheap).
                    self.enqueue_slot_rewrite(remove_idx);
                }

                // Drop the last slot
                self.slot_to_key.pop();
                self.chunk_meta.pop();
                self.slot_macro.pop();
                self.slot_colinfo.pop();

                self.grid_dirty = true;
                self.changed = true;
            }

            ChunkState::Uploading(up) => {
                // Uploading chunks also allocated arena + slots. They may still have pending uploads.
                self.arena.free(up.node_base, up.node_count);

                let dead = up.slot as usize;

                // Uploading slots should be outside the resident prefix
                debug_assert!(dead >= self.resident_slots, "uploading slot inside resident prefix");

                let last_slot = self.slot_to_key.len().saturating_sub(1);

                if dead != last_slot {
                    self.swap_slots(dead, last_slot);

                    // dead now holds moved uploading chunk; keep its slot data consistent
                    self.enqueue_slot_rewrite(dead);
                }

                self.slot_to_key.pop();
                self.chunk_meta.pop();
                self.slot_macro.pop();
                self.slot_colinfo.pop();

                // grid/chunk_count exclude uploading, so typically no grid change
                self.changed = true;
            }

            ChunkState::Queued | ChunkState::Building => {
                self.cancel_token(key).store(true, Ordering::Relaxed);
                self.queued_set.remove(&key);

                self.grid_dirty = true;
                self.changed = true;
            }
        }
    }


    #[inline]
    fn keep_origin_for(&self, center: ChunkKey) -> [i32; 3] {
        let ox = center.x - config::KEEP_RADIUS;
        let oz = center.z - config::KEEP_RADIUS;
        let oy = center.y + GRID_Y_MIN_DY;
        [ox, oy, oz]
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

        // IMPORTANT: only residents live in [0 .. resident_slots)
        let n = self.resident_slots.min(self.slot_to_key.len());
        for slot in 0..n {
            let k = self.slot_to_key[slot];
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

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

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
        // meta always written (one struct)
        let mut b = size_of::<ChunkMetaGpu>();

        b += u.nodes.len() * size_of::<NodeGpu>();
        b += u.macro_words.len() * size_of::<u32>();
        b += u.ropes.len() * size_of::<NodeRopesGpu>();
        b += u.colinfo_words.len() * size_of::<u32>();
        b
    }

    pub fn take_uploads(&mut self) -> Vec<ChunkUpload> {
        // drain everything (old behavior, but VecDeque-compatible)
        self.uploads.drain(..).collect()
    }

    pub fn take_uploads_budgeted(&mut self) -> Vec<ChunkUpload> {
        const MAX_UPLOADS_PER_FRAME: usize = 8;            // tune: 4–12
        const MAX_UPLOAD_BYTES_PER_FRAME: usize = 4 << 20; // tune: 2–8 MB

        #[inline]
        fn upload_bytes(u: &ChunkUpload) -> usize {
            use std::mem::size_of;
            size_of::<ChunkMetaGpu>()
                + u.nodes.len() * size_of::<NodeGpu>()
                + u.macro_words.len() * size_of::<u32>()
                + u.ropes.len() * size_of::<NodeRopesGpu>()
                + u.colinfo_words.len() * size_of::<u32>()
        }

        let mut out = Vec::new();
        let mut bytes = 0usize;

        while let Some(u) = self.uploads.front() {
            if out.len() >= MAX_UPLOADS_PER_FRAME {
                break;
            }

            let ub = upload_bytes(u);
            if bytes + ub > MAX_UPLOAD_BYTES_PER_FRAME && !out.is_empty() {
                break;
            }

            bytes += ub;
            out.push(self.uploads.pop_front().unwrap());
        }

        out
    }

    pub fn commit_uploads_applied(&mut self, applied: &[ChunkUpload]) -> bool {
    let mut any_promoted = false;

    for u in applied {
        if !u.completes_residency {
            continue;
        }

        let slot = u.slot as usize;

        // With FIFO uploads, this should always be true.
        if slot != self.resident_slots {
            // If this ever triggers, you have out-of-order promotion.
            // For now: skip promotion to keep invariants safe.
            continue;
        }

        let key = self.slot_to_key[slot];

        if let Some(st) = self.chunks.get_mut(&key) {
            if let ChunkState::Uploading(up) = st {
                *st = ChunkState::Resident(Resident {
                    slot: up.slot,
                    node_base: up.node_base,
                    node_count: up.node_count,
                });

                self.resident_slots += 1;
                any_promoted = true;
            }
        }
    }

    if any_promoted {
        self.grid_dirty = true;
        self.changed = true;

        // Rebuild now (we have last_center from update)
        if let Some(center) = self.last_center {
            self.rebuild_grid(center);
            self.grid_dirty = false;
        }

        return true; // grid changed
    }

    false
}


}
