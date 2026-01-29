// --- src/streaming/manager.rs ---
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
    render::gpu_types::{ChunkMetaGpu, NodeGpu},
    svo::{build_chunk_svo_sparse_cancelable_with_scratch, BuildScratch},
    world::WorldGen,
};

use super::NodeArena;

const INVALID_U32: u32 = 0xFFFF_FFFF;

// Vertical band dy in [-1..=2]
const GRID_Y_MIN_DY: i32 = -1;
const GRID_Y_COUNT: u32 = 4;

// How many eviction attempts to make when we can't fit a chunk's nodes contiguously.
const EVICT_ATTEMPTS: usize = 8;

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
struct ChunkKey {
    x: i32,
    y: i32,
    z: i32,
}

enum ChunkState {
    Missing,
    Queued,
    Building,
    Resident(Resident),
}

#[derive(Clone, Copy, Debug)]
struct Resident {
    slot: u32,      // index into chunk_meta (dense)
    node_base: u32, // base index into global node arena
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
}

pub struct ChunkUpload {
    pub slot: u32,
    pub meta: ChunkMetaGpu,

    pub node_base: u32,
    pub nodes: Arc<[NodeGpu]>,
}

// -----------------------------
// CPU cache (budgeted, LRU-ish)
// -----------------------------

#[derive(Clone)]
struct CachedChunk {
    nodes: Arc<[NodeGpu]>,
    bytes: usize,
    stamp: u64,
}

fn spawn_workers(gen: Arc<WorldGen>, rx_job: Receiver<BuildJob>, tx_done: Sender<BuildDone>) {
    for _ in 0..config::WORKER_THREADS {
        let gen = gen.clone();
        let rx_job = rx_job.clone();
        let tx_done = tx_done.clone();

        std::thread::spawn(move || {
            // One reusable scratch per worker thread: removes most per-chunk allocations.
            let mut scratch = BuildScratch::new();

            while let Ok(job) = rx_job.recv() {
                let k = job.key;

                if job.cancel.load(Ordering::Relaxed) {
                    if tx_done
                        .send(BuildDone {
                            key: k,
                            cancel: job.cancel,
                            canceled: true,
                            nodes: Vec::new(),
                        })
                        .is_err()
                    {
                        break;
                    }
                    continue;
                }

                let origin = [
                    k.x * config::CHUNK_SIZE as i32,
                    k.y * config::CHUNK_SIZE as i32,
                    k.z * config::CHUNK_SIZE as i32,
                ];

                let nodes = build_chunk_svo_sparse_cancelable_with_scratch(
                    &gen,
                    origin,
                    config::CHUNK_SIZE,
                    job.cancel.as_ref(),
                    &mut scratch,
                );

                let canceled = job.cancel.load(Ordering::Relaxed);
                let nodes = if canceled { Vec::new() } else { nodes };

                if tx_done
                    .send(BuildDone {
                        key: k,
                        cancel: job.cancel,
                        canceled,
                        nodes,
                    })
                    .is_err()
                {
                    break;
                }
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
    gen: Arc<WorldGen>,

    chunks: HashMap<ChunkKey, ChunkState>,
    build_queue: VecDeque<ChunkKey>,

    // Deduplicate queued keys (prevents unbounded queue growth).
    queued_set: std::collections::HashSet<ChunkKey>, // <-- unchanged here; this is NOT per-frame.
                                                 // keep it if you still want it.
    // Only sort/purge when center chunk changes.
    last_center: Option<ChunkKey>,

    // Per-chunk cancel tokens
    cancels: HashMap<ChunkKey, Arc<AtomicBool>>,

    tx_job: Sender<BuildJob>,
    rx_done: Receiver<BuildDone>,
    in_flight: usize,

    // Dense slots for resident chunks
    slot_to_key: Vec<ChunkKey>,    // slot -> key
    chunk_meta: Vec<ChunkMetaGpu>, // slot -> meta
    uploads: Vec<ChunkUpload>,     // pending GPU writes this frame

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

        Self {
            gen,
            chunks: HashMap::default(),
            build_queue: VecDeque::new(),
            queued_set: std::collections::HashSet::default(),
            last_center: None,

            cancels: HashMap::default(),
            tx_job,
            rx_done,
            in_flight: 0,

            slot_to_key: Vec::new(),
            chunk_meta: Vec::new(),
            uploads: Vec::new(),
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
        }
    }

    // -------------------------------------------------------------------------
    // NEW: Cheap region checks (replaces per-frame keep_set HashSet)
    // -------------------------------------------------------------------------

    #[inline(always)]
    fn y_band_min() -> i32 {
        GRID_Y_MIN_DY
    }

    #[inline(always)]
    fn y_band_max() -> i32 {
        GRID_Y_MIN_DY + GRID_Y_COUNT as i32 - 1
    }

    #[inline(always)]
    fn in_keep(center: ChunkKey, k: ChunkKey) -> bool {
        let dx = k.x - center.x;
        let dz = k.z - center.z;
        let dy = k.y - center.y;

        dx >= -config::KEEP_RADIUS
            && dx <= config::KEEP_RADIUS
            && dz >= -config::KEEP_RADIUS
            && dz <= config::KEEP_RADIUS
            && dy >= Self::y_band_min()
            && dy <= Self::y_band_max()
    }

    #[inline(always)]
    fn in_active(center: ChunkKey, k: ChunkKey) -> bool {
        let dx = k.x - center.x;
        let dz = k.z - center.z;
        let dy = k.y - center.y;

        dx >= -config::ACTIVE_RADIUS
            && dx <= config::ACTIVE_RADIUS
            && dz >= -config::ACTIVE_RADIUS
            && dz <= config::ACTIVE_RADIUS
            && dy >= Self::y_band_min()
            && dy <= Self::y_band_max()
    }

    // -------------------------------------------------------------------------
    // Update (rewritten parts only)
    // -------------------------------------------------------------------------

    pub fn update(&mut self, world: &Arc<WorldGen>, cam_pos_m: Vec3, cam_fwd: Vec3) -> bool {
        self.uploads.clear();

        let cam_vx = (cam_pos_m.x / config::VOXEL_SIZE_M_F32).floor() as i32;
        let cam_vz = (cam_pos_m.z / config::VOXEL_SIZE_M_F32).floor() as i32;

        let ccx = cam_vx.div_euclid(config::CHUNK_SIZE as i32);
        let ccz = cam_vz.div_euclid(config::CHUNK_SIZE as i32);

        let ground_y_vox = world.ground_height(cam_vx, cam_vz);
        let ground_cy = ground_y_vox.div_euclid(config::CHUNK_SIZE as i32);

        let center = ChunkKey {
            x: ccx,
            y: ground_cy,
            z: ccz,
        };

        // ---------------------------------------------------------------------
        // Desired (ACTIVE) handling: no keep_set HashSet; just in_keep/in_active
        // ---------------------------------------------------------------------

        // Instead of `let desired = desired_chunks(...); for k in &desired { ... }`,
        // we iterate directly to avoid allocating `desired` too (optional),
        // while still preserving behavior.

        for dy in GRID_Y_MIN_DY..=(GRID_Y_MIN_DY + GRID_Y_COUNT as i32 - 1) {
            for dz in -config::ACTIVE_RADIUS..=config::ACTIVE_RADIUS {
                for dx in -config::ACTIVE_RADIUS..=config::ACTIVE_RADIUS {
                    let k = ChunkKey {
                        x: center.x + dx,
                        y: center.y + dy,
                        z: center.z + dz,
                    };

                    match self.chunks.get(&k) {
                        Some(ChunkState::Resident(_))
                        | Some(ChunkState::Queued)
                        | Some(ChunkState::Building) => {}
                        None | Some(ChunkState::Missing) => {
                            // Cache hit -> promote, no build queue.
                            if self.cache.contains_key(&k) {
                                let _ = self.try_promote_from_cache(center, k);
                                continue;
                            }

                            // Cache miss -> queue build.
                            let c = self.cancel_token(k);
                            c.store(false, Ordering::Relaxed);

                            self.chunks.insert(k, ChunkState::Queued);

                            if self.queued_set.insert(k) {
                                self.build_queue.push_back(k);
                            }
                        }
                    }
                }
            }
        }

        // ---------------------------------------------------------------------
        // Unload outside KEEP: replaced keep_set.contains with in_keep()
        // ---------------------------------------------------------------------
        {
            let keys_snapshot: Vec<ChunkKey> = self.chunks.keys().copied().collect();
            for k in keys_snapshot {
                if !Self::in_keep(center, k) {
                    self.unload_chunk(k);
                }
            }
        }

        // Only purge + sort when the center chunk changes.
        let center_changed = self.last_center.map_or(true, |c| c != center);
        if center_changed {
            self.last_center = Some(center);

            // Purge stale keys aggressively:
            self.build_queue.retain(|k| {
                Self::in_keep(center, *k) && matches!(self.chunks.get(k), Some(ChunkState::Queued))
            });

            self.queued_set.clear();
            self.queued_set.extend(self.build_queue.iter().copied());

            sort_queue_near_first(&mut self.build_queue, center, cam_fwd);
        }

        // Dispatch builds
        while self.in_flight < config::MAX_IN_FLIGHT {
            let Some(k) = self.build_queue.pop_front() else { break; };
            self.queued_set.remove(&k);

            if !Self::in_keep(center, k) {
                self.cancel_token(k).store(true, Ordering::Relaxed);
                self.chunks.insert(k, ChunkState::Missing);
                continue;
            }

            if self.cache.contains_key(&k) {
                self.chunks.insert(k, ChunkState::Missing);
                let _ = self.try_promote_from_cache(center, k);
                continue;
            }

            if matches!(self.chunks.get(&k), Some(ChunkState::Queued)) {
                self.chunks.insert(k, ChunkState::Building);

                let cancel = self.cancel_token(k);
                cancel.store(false, Ordering::Relaxed);

                if self
                    .tx_job
                    .send(BuildJob {
                        key: k,
                        cancel: cancel.clone(),
                    })
                    .is_ok()
                {
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
                if self.chunks.get(&done.key).is_some() {
                    self.chunks.insert(done.key, ChunkState::Missing);
                }
                continue;
            }

            // Replaced keep_set.contains(&done.key)
            if !Self::in_keep(center, done.key) {
                self.cancel_token(done.key).store(true, Ordering::Relaxed);
                self.chunks.insert(done.key, ChunkState::Missing);
                continue;
            }

            self.on_build_done(center, done.key, done.nodes);
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

    // -------------------------------------------------------------------------
    // Internals
    // -------------------------------------------------------------------------

    fn cancel_token(&mut self, key: ChunkKey) -> Arc<AtomicBool> {
        self.cancels
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicBool::new(false)))
            .clone()
    }

    fn desired_chunks(center: ChunkKey, radius: i32) -> Vec<ChunkKey> {
        let mut out = Vec::new();
        for dy in GRID_Y_MIN_DY..=(GRID_Y_MIN_DY + GRID_Y_COUNT as i32 - 1) {
            for dz in -radius..=radius {
                for dx in -radius..=radius {
                    out.push(ChunkKey {
                        x: center.x + dx,
                        y: center.y + dy,
                        z: center.z + dz,
                    });
                }
            }
        }
        out
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

            // Weighted distance (favor keeping vertical neighbors).
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

    fn cache_put(&mut self, key: ChunkKey, nodes: Arc<[NodeGpu]>) {
        let bytes = nodes.len() * size_of::<NodeGpu>();

        // If replacing an existing entry, subtract its bytes first.
        if let Some(old) = self.cache.remove(&key) {
            self.cache_bytes = self.cache_bytes.saturating_sub(old.bytes);
        }

        self.cache_stamp = self.cache_stamp.wrapping_add(1).max(1);
        let stamp = self.cache_stamp;

        self.cache.insert(
            key,
            CachedChunk {
                nodes,
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

            // Only evict if this LRU record matches the current entry stamp.
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

    fn try_make_resident(
        &mut self,
        center: ChunkKey,
        key: ChunkKey,
        nodes: Arc<[NodeGpu]>,
    ) -> bool {
        // If already resident, nothing to do.
        if matches!(self.chunks.get(&key), Some(ChunkState::Resident(_))) {
            return true;
        }

        let need = nodes.len() as u32;

        // If we have nothing (shouldn't happen for non-air chunks, but be safe).
        if need == 0 {
            self.chunks.insert(key, ChunkState::Missing);
            return false;
        }

        // Try allocate; if fails, evict farthest chunks and retry a few times.
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
            // Can't fit right now; keep cache so we can promote later.
            self.chunks.insert(key, ChunkState::Missing);
            return false;
        };

        // Allocate slot (dense).
        let slot = self.slot_to_key.len() as u32;
        self.slot_to_key.push(key);

        let origin_vox = [
            key.x * config::CHUNK_SIZE as i32,
            key.y * config::CHUNK_SIZE as i32,
            key.z * config::CHUNK_SIZE as i32,
        ];

        let meta = ChunkMetaGpu {
            origin: [origin_vox[0], origin_vox[1], origin_vox[2], 0],
            node_base,
            node_count: need,
            _pad0: 0,
            _pad1: 0,
        };

        self.chunk_meta.push(meta);

        // Mark resident.
        self.chunks.insert(
            key,
            ChunkState::Resident(Resident {
                slot,
                node_base,
                node_count: need,
            }),
        );

        // Schedule GPU upload (nodes + meta).
        self.uploads.push(ChunkUpload {
            slot,
            meta,
            node_base,
            nodes,
        });

        // Resident set changed => grid mapping may change.
        self.grid_dirty = true;
        self.changed = true;

        true
    }

    fn try_promote_from_cache(&mut self, center: ChunkKey, key: ChunkKey) -> bool {
        let Some(entry) = self.cache.get(&key).cloned() else {
            return false;
        };

        // Touch LRU.
        self.cache_touch(key);

        // Try to make resident from cached nodes.
        self.try_make_resident(center, key, entry.nodes)
    }

    fn on_build_done(&mut self, center: ChunkKey, key: ChunkKey, nodes: Vec<NodeGpu>) {
        // If this chunk got cancelled while the result was in flight, drop it.
        if let Some(c) = self.cancels.get(&key) {
            if c.load(Ordering::Relaxed) {
                self.chunks.insert(key, ChunkState::Missing);
                return;
            }
        }

        // Convert to Arc slice once (cheap clones thereafter).
        let nodes_arc: Arc<[NodeGpu]> = nodes.into();

        // Cache it (so we never rebuild this chunk again unless evicted).
        self.cache_put(key, nodes_arc.clone());

        // If already resident (should be rare), don't allocate/upload again.
        if matches!(self.chunks.get(&key), Some(ChunkState::Resident(_))) {
            return;
        }

        // Try to allocate + upload now; if we can't fit, keep it cached and mark missing.
        let ok = self.try_make_resident(center, key, nodes_arc);
        if !ok {
            self.chunks.insert(key, ChunkState::Missing);
        }
    }

    fn unload_chunk(&mut self, key: ChunkKey) {
        let Some(state) = self.chunks.get(&key) else { return; };

        match *state {
            ChunkState::Resident(res) => {
                // Free node arena range.
                self.arena.free(res.node_base, res.node_count);

                // Remove slot densely by swap-remove.
                let dead_slot = res.slot as usize;
                let last_slot = self.slot_to_key.len().saturating_sub(1);

                if dead_slot != last_slot {
                    let moved_key = self.slot_to_key[last_slot];
                    self.slot_to_key[dead_slot] = moved_key;

                    // Move meta.
                    let moved_meta = self.chunk_meta[last_slot];
                    self.chunk_meta[dead_slot] = moved_meta;

                    // Update moved chunk's Resident.slot.
                    if let Some(state) = self.chunks.get_mut(&moved_key) {
                        if let ChunkState::Resident(mr) = state {
                            mr.slot = dead_slot as u32;
                        }
                    }

                    // Schedule meta rewrite for moved slot (GPU needs updated slot meta).
                    self.uploads.push(ChunkUpload {
                        slot: dead_slot as u32,
                        meta: self.chunk_meta[dead_slot],
                        node_base: 0,
                        nodes: Arc::<[NodeGpu]>::from(Vec::<NodeGpu>::new()),
                    });
                }

                self.slot_to_key.pop();
                self.chunk_meta.pop();

                self.chunks.insert(key, ChunkState::Missing);

                // Slot mapping changed => grid mapping changed.
                self.grid_dirty = true;
                self.changed = true;
            }

            ChunkState::Queued | ChunkState::Building => {
                // Cancel work in-flight.
                self.cancel_token(key).store(true, Ordering::Relaxed);

                // If it was queued, prevent duplicate re-adds.
                self.queued_set.remove(&key);

                self.chunks.insert(key, ChunkState::Missing);

                // Conservative dirty.
                self.grid_dirty = true;
                self.changed = true;
            }

            _ => {}
        }
    }

    /// Compute the KEEP-grid origin for a given center (helper so we can detect origin shifts).
    #[inline]
    fn keep_origin_for(&self, center: ChunkKey) -> [i32; 3] {
        let ox = center.x - config::KEEP_RADIUS;
        let oz = center.z - config::KEEP_RADIUS;
        let oy = center.y + GRID_Y_MIN_DY;
        [ox, oy, oz]
    }

    /// Rebuild the chunk_grid mapping for the current KEEP volume.
    ///
    /// This is intentionally called only when `grid_dirty` is set, because it is O(ncells + nchunks)
    /// and it forces a full GPU upload if you do it every frame.
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

        // Fill grid from resident chunks (slot -> key).
        for (slot, &k) in self.slot_to_key.iter().enumerate() {
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

    pub fn chunk_count(&self) -> u32 {
        self.slot_to_key.len() as u32
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

    pub fn take_uploads(&mut self) -> Vec<ChunkUpload> {
        std::mem::take(&mut self.uploads)
    }
}
