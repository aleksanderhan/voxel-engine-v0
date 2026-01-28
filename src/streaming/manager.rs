// src/streaming/manager.rs
// ------------------------
// Editing-aware chunk streaming.
//
// Key fixes vs your current version:
// 1) App edits must go through ChunkManager::apply_edits(), which updates per-chunk edit maps.
// 2) Raycast must use *resident* nodes, not cache (cache can be evicted, causing misses).
//
// Abbreviations:
// - SVO = Sparse Voxel Octree
// - LRU = Least Recently Used
// - AABB = Axis-Aligned Bounding Box

use std::collections::{HashMap, HashSet, VecDeque};
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

// How many eviction attempts when we can't fit a chunk's nodes contiguously.
const EVICT_ATTEMPTS: usize = 8;

// Local coord packing for CHUNK_SIZE=64 (6 bits each axis).
#[inline(always)]
fn pack_local_64(lx: i32, ly: i32, lz: i32) -> u32 {
    (lx as u32) | ((lz as u32) << 6) | ((ly as u32) << 12)
}

#[inline(always)]
fn world_to_chunk_key(wx: i32, wy: i32, wz: i32) -> ChunkKey {
    ChunkKey {
        x: wx.div_euclid(config::CHUNK_SIZE as i32),
        y: wy.div_euclid(config::CHUNK_SIZE as i32),
        z: wz.div_euclid(config::CHUNK_SIZE as i32),
    }
}

#[inline(always)]
fn world_to_local(wx: i32, wy: i32, wz: i32, ck: ChunkKey) -> (i32, i32, i32) {
    let ox = ck.x * config::CHUNK_SIZE as i32;
    let oy = ck.y * config::CHUNK_SIZE as i32;
    let oz = ck.z * config::CHUNK_SIZE as i32;
    (wx - ox, wy - oy, wz - oz)
}

/// A single voxel edit in world-voxel coordinates.
#[derive(Clone, Copy, Debug)]
pub struct VoxelEdit {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub material: u32,
}

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

    // Immutable snapshot of this chunk's edit overrides (local packed -> material).
    edit_stamp: u64,
    edit_map: Option<crate::svo::builder::ChunkEditMap>,
}

struct BuildDone {
    key: ChunkKey,
    cancel: Arc<AtomicBool>,
    canceled: bool,

    edit_stamp: u64,
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

    // Must match the chunk's edit stamp to be usable.
    edit_stamp: u64,
}

// -----------------------------
// Edit layer (per-chunk overrides)
// -----------------------------

#[derive(Default)]
struct ChunkEdits {
    stamp: u64,
    map: HashMap<u32, u32>, // packed local -> material (presence = override)
}

pub struct ChunkManager {
    gen: Arc<WorldGen>,

    chunks: HashMap<ChunkKey, ChunkState>,
    build_queue: VecDeque<ChunkKey>,

    // Deduplicate queued keys (prevents unbounded queue growth).
    queued_set: HashSet<ChunkKey>,

    // Only sort/purge when center chunk changes.
    last_center: Option<ChunkKey>,

    // Per-chunk cancel tokens
    cancels: HashMap<ChunkKey, Arc<AtomicBool>>,

    tx_job: Sender<BuildJob>,
    rx_done: Receiver<BuildDone>,
    in_flight: usize,

    // Dense slots for resident chunks
    slot_to_key: Vec<ChunkKey>,        // slot -> key
    resident_nodes: Vec<Arc<[NodeGpu]>>, // slot -> nodes (authoritative for raycast)
    chunk_meta: Vec<ChunkMetaGpu>,     // slot -> meta
    uploads: Vec<ChunkUpload>,         // pending GPU writes this frame

    // General “something changed” flag (kept for other systems).
    changed: bool,

    // Grid dirty flag:
    // - true when the GPU lookup grid needs rebuilding (origin shift or resident slot mapping changed)
    grid_dirty: bool,

    // Node arena (in units of NodeGpu elements)
    arena: NodeArena,

    // Chunk grid for GPU lookup: maps grid cell -> resident slot index (or INVALID_U32).
    grid_origin_chunk: [i32; 3],
    grid_dims: [u32; 3],
    chunk_grid: Vec<u32>,

    // ----------------
    // CPU chunk cache
    // ----------------
    cache: HashMap<ChunkKey, CachedChunk>,
    cache_lru: VecDeque<(ChunkKey, u64)>, // (key, stamp) entries; duplicates are allowed
    cache_stamp: u64,
    cache_bytes: usize,

    // ----------------
    // Editing
    // ----------------
    edits: HashMap<ChunkKey, ChunkEdits>,
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

                // If already cancelled before starting, still notify main thread.
                if job.cancel.load(Ordering::Relaxed) {
                    let _ = tx_done.send(BuildDone {
                        key: k,
                        cancel: job.cancel,
                        canceled: true,
                        edit_stamp: job.edit_stamp,
                        nodes: Vec::new(),
                    });
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
                    job.edit_map.as_ref(),
                );

                let canceled = job.cancel.load(Ordering::Relaxed);
                let nodes = if canceled { Vec::new() } else { nodes };

                let _ = tx_done.send(BuildDone {
                    key: k,
                    cancel: job.cancel,
                    canceled,
                    edit_stamp: job.edit_stamp,
                    nodes,
                });
            }
        });
    }
}

fn sort_queue_near_first(queue: &mut VecDeque<ChunkKey>, center: ChunkKey, cam_fwd: Vec3) {
    let mut v: Vec<ChunkKey> = queue.drain(..).collect();

    // Horizontal forward (XZ) for “look direction”.
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

    // Lower score = higher priority.
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

impl ChunkManager {
    pub fn new(gen: Arc<WorldGen>) -> Self {
        let (tx_job, rx_job) = unbounded::<BuildJob>();
        let (tx_done, rx_done) = unbounded::<BuildDone>();
        spawn_workers(gen.clone(), rx_job, tx_done);

        // Arena capacity in NodeGpu elements.
        let node_capacity = (config::NODE_BUDGET_BYTES / size_of::<NodeGpu>()) as u32;

        // Grid size (KEEP box).
        let nx = (2 * config::KEEP_RADIUS + 1) as u32;
        let nz = nx;
        let ny = GRID_Y_COUNT;
        let grid_len = (nx * ny * nz) as usize;

        Self {
            gen,

            chunks: HashMap::new(),
            build_queue: VecDeque::new(),
            queued_set: HashSet::new(),
            last_center: None,

            cancels: HashMap::new(),
            tx_job,
            rx_done,
            in_flight: 0,

            slot_to_key: Vec::new(),
            resident_nodes: Vec::new(),
            chunk_meta: Vec::new(),
            uploads: Vec::new(),
            changed: false,

            grid_dirty: true,
            arena: NodeArena::new(node_capacity),

            grid_origin_chunk: [0, 0, 0],
            grid_dims: [nx, ny, nz],
            chunk_grid: vec![INVALID_U32; grid_len],

            cache: HashMap::new(),
            cache_lru: VecDeque::new(),
            cache_stamp: 1,
            cache_bytes: 0,

            edits: HashMap::new(),
        }
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

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

    /// Apply a batch of voxel edits (world-voxel coords).
    /// This marks affected chunks dirty so they get rebuilt and uploaded.
    pub fn apply_edits(&mut self, edits: &[VoxelEdit]) {
        let mut touched: HashSet<ChunkKey> = HashSet::new();

        for e in edits {
            let ck = world_to_chunk_key(e.x, e.y, e.z);
            let (lx, ly, lz) = world_to_local(e.x, e.y, e.z, ck);

            // Ignore edits outside the chunk bounds (defensive).
            let cs = config::CHUNK_SIZE as i32;
            if (lx | ly | lz) < 0 || lx >= cs || ly >= cs || lz >= cs {
                continue;
            }

            let entry = self.edits.entry(ck).or_insert_with(ChunkEdits::default);
            entry.stamp = entry.stamp.wrapping_add(1).max(1);
            entry.map.insert(pack_local_64(lx, ly, lz), e.material);

            touched.insert(ck);
        }

        // Any touched chunk: cancel in-flight, invalidate cache, and force rebuild.
        for ck in touched {
            self.invalidate_and_rebuild_chunk(ck);
        }
    }

    // -------------------------------------------------------------------------
    // Streaming update
    // -------------------------------------------------------------------------

    /// Returns `true` if the GPU lookup grid changed and needs upload.
    pub fn update(&mut self, world: &Arc<WorldGen>, cam_pos_m: Vec3, cam_fwd: Vec3) -> bool {
        // Center chunk (ground-anchored).
        let cam_vx = (cam_pos_m.x / config::VOXEL_SIZE_M_F32).floor() as i32;
        let cam_vz = (cam_pos_m.z / config::VOXEL_SIZE_M_F32).floor() as i32;

        let ccx = cam_vx.div_euclid(config::CHUNK_SIZE as i32);
        let ccz = cam_vz.div_euclid(config::CHUNK_SIZE as i32);

        let ground_y_vox = world.ground_height(cam_vx, cam_vz);
        let ground_cy = ground_y_vox.div_euclid(config::CHUNK_SIZE as i32);

        let center = ChunkKey { x: ccx, y: ground_cy, z: ccz };

        let desired = Self::desired_chunks(center, config::ACTIVE_RADIUS);
        let keep = Self::desired_chunks(center, config::KEEP_RADIUS);
        let keep_set: HashSet<ChunkKey> = keep.iter().copied().collect();

        // Promote cached desired chunks immediately (no rebuild).
        // For missing desired chunks that are not cached, queue a build.
        for k in &desired {
            match self.chunks.get(k) {
                Some(ChunkState::Resident(_)) | Some(ChunkState::Queued) | Some(ChunkState::Building) => {}
                None | Some(ChunkState::Missing) => {
                    if self.cache_contains_valid(*k) {
                        let _ = self.try_promote_from_cache(center, *k);
                        continue;
                    }

                    let c = self.cancel_token(*k);
                    c.store(false, Ordering::Relaxed);

                    self.chunks.insert(*k, ChunkState::Queued);

                    if self.queued_set.insert(*k) {
                        self.build_queue.push_back(*k);
                    }
                }
            }
        }

        // Unload outside keep (also cancel queued/building).
        {
            let keys_snapshot: Vec<ChunkKey> = self.chunks.keys().copied().collect();
            for k in keys_snapshot {
                if !keep_set.contains(&k) {
                    self.unload_chunk(k);
                }
            }
        }

        // Only purge + sort when center chunk changes.
        let center_changed = self.last_center.map_or(true, |c| c != center);
        if center_changed {
            self.last_center = Some(center);

            self.build_queue.retain(|k| {
                keep_set.contains(k) && matches!(self.chunks.get(k), Some(ChunkState::Queued))
            });

            self.queued_set.clear();
            self.queued_set.extend(self.build_queue.iter().copied());

            sort_queue_near_first(&mut self.build_queue, center, cam_fwd);
        }

        // Dispatch builds (only for cache misses).
        while self.in_flight < config::MAX_IN_FLIGHT {
            let Some(k) = self.build_queue.pop_front() else { break; };
            self.queued_set.remove(&k);

            if !keep_set.contains(&k) {
                self.cancel_token(k).store(true, Ordering::Relaxed);
                self.chunks.insert(k, ChunkState::Missing);
                continue;
            }

            // If it became valid-cached since it was queued, don't rebuild.
            if self.cache_contains_valid(k) {
                self.chunks.insert(k, ChunkState::Missing);
                let _ = self.try_promote_from_cache(center, k);
                continue;
            }

            if matches!(self.chunks.get(&k), Some(ChunkState::Queued)) {
                self.chunks.insert(k, ChunkState::Building);

                let cancel = self.cancel_token(k);
                cancel.store(false, Ordering::Relaxed);

                let (edit_stamp, edit_map) = self.snapshot_chunk_edit_map(k);

                if self.tx_job.send(BuildJob { key: k, cancel: cancel.clone(), edit_stamp, edit_map }).is_ok() {
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

        // Harvest done builds.
        while let Ok(done) = self.rx_done.try_recv() {
            if self.in_flight > 0 {
                self.in_flight -= 1;
            }

            // If cancelled (before start or mid-build), don't keep it Building.
            if done.canceled || done.cancel.load(Ordering::Relaxed) {
                if self.chunks.get(&done.key).is_some() {
                    self.chunks.insert(done.key, ChunkState::Missing);
                }
                continue;
            }

            // If finished but no longer in KEEP, drop it.
            if !keep_set.contains(&done.key) {
                self.cancel_token(done.key).store(true, Ordering::Relaxed);
                self.chunks.insert(done.key, ChunkState::Missing);
                continue;
            }

            // If edits changed since the job was queued, discard result.
            let cur_stamp = self.chunk_edit_stamp(done.key);
            if cur_stamp != done.edit_stamp {
                self.chunks.insert(done.key, ChunkState::Missing);
                continue;
            }

            self.on_build_done(center, done.key, done.nodes, done.edit_stamp);
        }

        // If keep-grid origin would shift, mapping changes.
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
    // Internals: cancels + desired set
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
                    out.push(ChunkKey { x: center.x + dx, y: center.y + dy, z: center.z + dz });
                }
            }
        }
        out
    }

    // -------------------------------------------------------------------------
    // Editing helpers
    // -------------------------------------------------------------------------

    fn chunk_edit_stamp(&self, key: ChunkKey) -> u64 {
        self.edits.get(&key).map(|e| e.stamp).unwrap_or(0)
    }

    fn snapshot_chunk_edit_map(&self, key: ChunkKey) -> (u64, Option<crate::svo::builder::ChunkEditMap>) {
        let Some(ed) = self.edits.get(&key) else {
            return (0, None);
        };
        if ed.map.is_empty() {
            return (ed.stamp, None);
        }
        (ed.stamp, Some(Arc::new(ed.map.clone())))
    }

    fn invalidate_and_rebuild_chunk(&mut self, key: ChunkKey) {
        // Invalidate cache entry (edits mean old cached nodes are wrong).
        self.cache_remove(key);

        // Cancel in-flight if queued/building.
        if matches!(self.chunks.get(&key), Some(ChunkState::Queued | ChunkState::Building)) {
            self.cancel_token(key).store(true, Ordering::Relaxed);
            self.queued_set.remove(&key);
            self.chunks.insert(key, ChunkState::Missing);
        }

        // If resident, unload (frees node arena, updates slots, dirties grid).
        if matches!(self.chunks.get(&key), Some(ChunkState::Resident(_))) {
            self.unload_chunk(key);
        }

        // Re-queue immediately (high priority).
        self.chunks.insert(key, ChunkState::Queued);
        if self.queued_set.insert(key) {
            self.build_queue.push_front(key);
        }

        self.changed = true;
    }

    // -------------------------------------------------------------------------
    // Cache helpers
    // -------------------------------------------------------------------------

    fn cache_touch(&mut self, key: ChunkKey) {
        if let Some(e) = self.cache.get_mut(&key) {
            self.cache_stamp = self.cache_stamp.wrapping_add(1).max(1);
            e.stamp = self.cache_stamp;
            self.cache_lru.push_back((key, e.stamp));
        }
    }

    fn cache_put(&mut self, key: ChunkKey, nodes: Arc<[NodeGpu]>, edit_stamp: u64) {
        let bytes = nodes.len() * size_of::<NodeGpu>();

        if let Some(old) = self.cache.remove(&key) {
            self.cache_bytes = self.cache_bytes.saturating_sub(old.bytes);
        }

        self.cache_stamp = self.cache_stamp.wrapping_add(1).max(1);
        let stamp = self.cache_stamp;

        self.cache.insert(key, CachedChunk { nodes, bytes, stamp, edit_stamp });
        self.cache_bytes = self.cache_bytes.saturating_add(bytes);
        self.cache_lru.push_back((key, stamp));

        self.evict_cache_as_needed();
    }

    fn cache_remove(&mut self, key: ChunkKey) {
        if let Some(old) = self.cache.remove(&key) {
            self.cache_bytes = self.cache_bytes.saturating_sub(old.bytes);
        }
    }

    fn evict_cache_as_needed(&mut self) {
        let budget = config::CHUNK_CACHE_BUDGET_BYTES;

        while self.cache_bytes > budget {
            let Some((k, stamp)) = self.cache_lru.pop_front() else { break; };

            let should_evict = self.cache.get(&k).map(|e| e.stamp == stamp).unwrap_or(false);
            if !should_evict {
                continue;
            }

            if let Some(ev) = self.cache.remove(&k) {
                self.cache_bytes = self.cache_bytes.saturating_sub(ev.bytes);
            }
        }
    }

    fn cache_contains_valid(&self, key: ChunkKey) -> bool {
        let Some(c) = self.cache.get(&key) else { return false; };
        c.edit_stamp == self.chunk_edit_stamp(key)
    }

    // -------------------------------------------------------------------------
    // Resident creation / promotion
    // -------------------------------------------------------------------------

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

    fn try_make_resident(&mut self, center: ChunkKey, key: ChunkKey, nodes: Arc<[NodeGpu]>) -> bool {
        if matches!(self.chunks.get(&key), Some(ChunkState::Resident(_))) {
            return true;
        }

        let need = nodes.len() as u32;
        if need == 0 {
            self.chunks.insert(key, ChunkState::Missing);
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
            self.chunks.insert(key, ChunkState::Missing);
            return false;
        };

        let slot = self.slot_to_key.len() as u32;
        self.slot_to_key.push(key);
        self.resident_nodes.push(nodes.clone());

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

        self.chunks.insert(
            key,
            ChunkState::Resident(Resident { slot, node_base, node_count: need }),
        );

        self.uploads.push(ChunkUpload { slot, meta, node_base, nodes });

        self.grid_dirty = true;
        self.changed = true;

        true
    }

    fn try_promote_from_cache(&mut self, center: ChunkKey, key: ChunkKey) -> bool {
        let Some(entry) = self.cache.get(&key).cloned() else { return false; };

        if entry.edit_stamp != self.chunk_edit_stamp(key) {
            return false;
        }

        self.cache_touch(key);
        self.try_make_resident(center, key, entry.nodes)
    }

    fn on_build_done(&mut self, center: ChunkKey, key: ChunkKey, nodes: Vec<NodeGpu>, edit_stamp: u64) {
        if let Some(c) = self.cancels.get(&key) {
            if c.load(Ordering::Relaxed) {
                self.chunks.insert(key, ChunkState::Missing);
                return;
            }
        }

        let nodes_arc: Arc<[NodeGpu]> = nodes.into();

        // Cache (with edit stamp).
        self.cache_put(key, nodes_arc.clone(), edit_stamp);

        if matches!(self.chunks.get(&key), Some(ChunkState::Resident(_))) {
            return;
        }

        let ok = self.try_make_resident(center, key, nodes_arc);
        if !ok {
            self.chunks.insert(key, ChunkState::Missing);
        }
    }

    fn unload_chunk(&mut self, key: ChunkKey) {
        let Some(state) = self.chunks.get(&key) else { return; };

        match *state {
            ChunkState::Resident(res) => {
                self.arena.free(res.node_base, res.node_count);

                let dead_slot = res.slot as usize;
                let last_slot = self.slot_to_key.len().saturating_sub(1);

                if dead_slot != last_slot {
                    let moved_key = self.slot_to_key[last_slot];
                    self.slot_to_key[dead_slot] = moved_key;

                    let moved_nodes = self.resident_nodes[last_slot].clone();
                    self.resident_nodes[dead_slot] = moved_nodes;

                    let moved_meta = self.chunk_meta[last_slot];
                    self.chunk_meta[dead_slot] = moved_meta;

                    if let Some(st) = self.chunks.get_mut(&moved_key) {
                        if let ChunkState::Resident(mr) = st {
                            mr.slot = dead_slot as u32;
                        }
                    }

                    // Schedule meta rewrite for moved slot (nodes empty => meta-only update on GPU).
                    self.uploads.push(ChunkUpload {
                        slot: dead_slot as u32,
                        meta: self.chunk_meta[dead_slot],
                        node_base: 0,
                        nodes: Arc::<[NodeGpu]>::from(Vec::<NodeGpu>::new()),
                    });
                }

                self.slot_to_key.pop();
                self.resident_nodes.pop();
                self.chunk_meta.pop();

                self.chunks.insert(key, ChunkState::Missing);

                self.grid_dirty = true;
                self.changed = true;
            }

            ChunkState::Queued | ChunkState::Building => {
                self.cancel_token(key).store(true, Ordering::Relaxed);
                self.queued_set.remove(&key);
                self.chunks.insert(key, ChunkState::Missing);

                self.grid_dirty = true;
                self.changed = true;
            }

            _ => {}
        }
    }

    // -------------------------------------------------------------------------
    // Grid mapping
    // -------------------------------------------------------------------------

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

    // -------------------------------------------------------------------------
    // Raycast support (must be resident nodes, NOT cache)
    // -------------------------------------------------------------------------

    pub fn resident_chunks_for_raycast(&self) -> Vec<([i32; 3], Arc<[NodeGpu]>)> {
        let mut out = Vec::with_capacity(self.slot_to_key.len());
        for (slot, &k) in self.slot_to_key.iter().enumerate() {
            let origin = [
                k.x * config::CHUNK_SIZE as i32,
                k.y * config::CHUNK_SIZE as i32,
                k.z * config::CHUNK_SIZE as i32,
            ];
            out.push((origin, self.resident_nodes[slot].clone()));
        }
        out
    }
}
