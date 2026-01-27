// src/streaming/manager.rs
//
// Chunk streaming + background SVO (Sparse Voxel Octree) building.
//
// Key perf change (for the renderer optimization pass):
// - update(...) now returns `bool grid_changed` so the app can skip uploading `chunk_grid`
//   every frame. We track a `grid_dirty` flag and only rebuild the GPU (Graphics Processing Unit)
//   lookup grid when something that affects it actually changed (origin shift or resident set / slot moves).
//
// Important correctness fix:
// - Always send a BuildDone for every BuildJob that is received by a worker, even if it was canceled.
//   Otherwise `in_flight` can get "stuck" at MAX_IN_FLIGHT and chunk streaming will halt.
//
// Additional perf / stall fixes:
// - Deduplicate and purge stale entries in build_queue.
// - Only (purge + sort) the queue when the center chunk changes.
// - Never leave chunks stuck in Building when a job is canceled.
// - On node arena pressure, evict farthest resident chunks and avoid infinite rebuild/requeue loops.

use std::collections::{HashMap, HashSet, VecDeque};
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
    pub nodes: Vec<NodeGpu>,
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

                // If we were already cancelled before starting, still notify the main thread
                // so it can decrement `in_flight`.
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

                // If we got cancelled mid-build, still notify the main thread,
                // but drop nodes to save main-thread work + upload pressure.
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

    // Horizontal forward (XZ) for “look direction”.
    let mut f = Vec2::new(cam_fwd.x, cam_fwd.z);
    if f.length_squared() > 1e-6 {
        f = f.normalize();
    }

    v.sort_by(|a, b| {
        let sa = chunk_priority_score(*a, center, f);
        let sb = chunk_priority_score(*b, center, f);
        sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
    });

    queue.extend(v);
}

fn chunk_priority_score(k: ChunkKey, c: ChunkKey, _fwd_xz: Vec2) -> f32 {
    let dx = (k.x - c.x) as f32;
    let dz = (k.z - c.z) as f32;
    let dy = (k.y - c.y) as f32;

    // Lower score = higher priority.
    // Base distance (prefer close). Penalize vertical moves more.
    dx.abs() + dz.abs() + 2.0 * dy.abs()
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
    slot_to_key: Vec<ChunkKey>,    // slot -> key
    chunk_meta: Vec<ChunkMetaGpu>, // slot -> meta
    uploads: Vec<ChunkUpload>,     // pending GPU writes this frame

    // General “something changed” flag (kept for other systems).
    changed: bool,

    // Grid dirty flag:
    // - true when the GPU lookup grid needs rebuilding (origin shift or resident slot mapping changed)
    // - update() returns this so the renderer can skip `write_chunk_grid()` on most frames.
    grid_dirty: bool,

    // Node arena (in units of NodeGpu elements)
    arena: NodeArena,

    // Chunk grid for GPU lookup: maps grid cell -> resident slot index (or INVALID_U32).
    grid_origin_chunk: [i32; 3],
    grid_dims: [u32; 3],
    chunk_grid: Vec<u32>,
}

impl ChunkManager {
    pub fn new(gen: Arc<WorldGen>) -> Self {
        let (tx_job, rx_job) = unbounded::<BuildJob>();
        let (tx_done, rx_done) = unbounded::<BuildDone>();
        spawn_workers(gen.clone(), rx_job, tx_done);

        // Arena capacity in NodeGpu elements.
        let node_capacity = (config::NODE_BUDGET_BYTES / std::mem::size_of::<NodeGpu>()) as u32;

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
            chunk_meta: Vec::new(),
            uploads: Vec::new(),
            changed: false,

            grid_dirty: true, // first update should build + upload the grid
            arena: NodeArena::new(node_capacity),

            grid_origin_chunk: [0, 0, 0],
            grid_dims: [nx, ny, nz],
            chunk_grid: vec![INVALID_U32; grid_len],
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

    // -------------------------------------------------------------------------
    // Streaming update
    // -------------------------------------------------------------------------
    //
    // Returns:
    // - true  if the chunk_grid mapping changed (origin shift and/or resident slot mapping changed)
    // - false if chunk_grid is identical to last frame (safe to skip GPU upload)

    pub fn update(&mut self, world: &Arc<WorldGen>, cam_pos_m: Vec3, cam_fwd: Vec3) -> bool {
        self.uploads.clear();

        // Center chunk (ground-anchored).
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

        // Desired vs keep sets.
        let desired = Self::desired_chunks(center, config::ACTIVE_RADIUS);
        let keep = Self::desired_chunks(center, config::KEEP_RADIUS);
        let keep_set: HashSet<ChunkKey> = keep.iter().copied().collect();

        // Queue missing desired (reset cancel token to false).
        for k in &desired {
            match self.chunks.get(k) {
                None | Some(ChunkState::Missing) => {
                    let c = self.cancel_token(*k);
                    c.store(false, Ordering::Relaxed);

                    self.chunks.insert(*k, ChunkState::Queued);

                    // Dedupe queue entries.
                    if self.queued_set.insert(*k) {
                        self.build_queue.push_back(*k);
                    }
                }
                _ => {}
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

        // Only purge + sort when the center chunk changes.
        let center_changed = self.last_center.map_or(true, |c| c != center);
        if center_changed {
            self.last_center = Some(center);

            // Purge stale keys aggressively: keep only keys that are still queued and still in KEEP.
            self.build_queue.retain(|k| {
                keep_set.contains(k) && matches!(self.chunks.get(k), Some(ChunkState::Queued))
            });

            // Rebuild queued_set from the queue so it matches reality.
            self.queued_set.clear();
            self.queued_set.extend(self.build_queue.iter().copied());

            // Sort once per center change.
            sort_queue_near_first(&mut self.build_queue, center, cam_fwd);
        }

        // Dispatch builds.
        while self.in_flight < config::MAX_IN_FLIGHT {
            let Some(k) = self.build_queue.pop_front() else { break; };

            // Popped => no longer queued.
            self.queued_set.remove(&k);

            if !keep_set.contains(&k) {
                // Cancel if it was pending.
                self.cancel_token(k).store(true, Ordering::Relaxed);
                self.chunks.insert(k, ChunkState::Missing);
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
                    // Channel closed; keep it queued.
                    self.chunks.insert(k, ChunkState::Queued);

                    // Put it back (dedup-safe).
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

            // If the job was canceled (either before start or mid-build),
            // the chunk MUST NOT stay in Building forever.
            if done.canceled || done.cancel.load(Ordering::Relaxed) {
                if matches!(self.chunks.get(&done.key), Some(ChunkState::Building)) {
                    self.chunks.insert(done.key, ChunkState::Missing);
                } else if self.chunks.get(&done.key).is_some() {
                    self.chunks.insert(done.key, ChunkState::Missing);
                }
                continue;
            }

            // If it finished but is no longer in KEEP, drop it and mark Missing.
            if !keep_set.contains(&done.key) {
                self.cancel_token(done.key).store(true, Ordering::Relaxed);
                self.chunks.insert(done.key, ChunkState::Missing);
                continue;
            }

            // Still relevant.
            self.on_build_done(center, done.key, done.nodes);
        }

        // If the keep-grid origin would shift, the lookup mapping changes.
        if self.keep_origin_for(center) != self.grid_origin_chunk {
            self.grid_dirty = true;
        }

        // Rebuild grid mapping for current KEEP region only when needed.
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

    fn on_build_done(&mut self, center: ChunkKey, key: ChunkKey, nodes: Vec<NodeGpu>) {
        // If this chunk got cancelled while the result was in flight, drop it.
        if let Some(c) = self.cancels.get(&key) {
            if c.load(Ordering::Relaxed) {
                self.chunks.insert(key, ChunkState::Missing);
                return;
            }
        }

        let need = nodes.len() as u32;

        // Try allocate; if fails, evict farthest chunks and retry a few times.
        let mut node_base = self.arena.alloc(need);
        if node_base.is_none() {
            for _ in 0..EVICT_ATTEMPTS {
                // If we can't evict anything, stop.
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
            // Important: do NOT requeue and spin forever rebuilding.
            // Mark Missing; desired() will re-queue naturally once the system has room.
            self.chunks.insert(key, ChunkState::Missing);
            return;
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
                        nodes: Vec::new(), // meta-only update
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
}
