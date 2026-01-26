// src/streaming/manager.rs

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
    svo::build_chunk_svo_sparse_cancelable,
    world::WorldGen,
};

use super::NodeArena;

const INVALID_U32: u32 = 0xFFFF_FFFF;

// Vertical band dy in [-1..=2]
const GRID_Y_MIN_DY: i32 = -1;
const GRID_Y_COUNT: u32 = 4;

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
            while let Ok(job) = rx_job.recv() {
                if job.cancel.load(Ordering::Relaxed) {
                    continue;
                }

                let k = job.key;
                let origin = [
                    k.x * config::CHUNK_SIZE as i32,
                    k.y * config::CHUNK_SIZE as i32,
                    k.z * config::CHUNK_SIZE as i32,
                ];

                let nodes =
                    build_chunk_svo_sparse_cancelable(&gen, origin, config::CHUNK_SIZE, &job.cancel);

                // If we got cancelled mid-build, drop it (don’t send, saves main-thread work).
                if job.cancel.load(Ordering::Relaxed) {
                    continue;
                }

                if tx_done
                    .send(BuildDone {
                        key: k,
                        cancel: job.cancel,
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

    // Horizontal forward (XZ) for “look direction”
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

fn chunk_priority_score(k: ChunkKey, c: ChunkKey, fwd_xz: Vec2) -> f32 {
    let dx = (k.x - c.x) as f32;
    let dz = (k.z - c.z) as f32;
    let dy = (k.y - c.y) as f32;

    // base distance (prefer close)
    let dist = dx.abs() + dz.abs() + 2.0 * dy.abs();

    // prefer in front (ahead in XZ)
    let dir = Vec2::new(dx, dz);
    let ahead = if dir.length_squared() > 1e-6 {
        dir.normalize().dot(fwd_xz) // [-1..1]
    } else {
        1.0
    };

    // lower score = higher priority
    dist - 1.75 * ahead
}

pub struct ChunkManager {
    gen: Arc<WorldGen>,

    chunks: HashMap<ChunkKey, ChunkState>,
    build_queue: VecDeque<ChunkKey>,

    // Per-chunk cancel tokens (A1)
    cancels: HashMap<ChunkKey, Arc<AtomicBool>>,

    tx_job: Sender<BuildJob>,
    rx_done: Receiver<BuildDone>,
    in_flight: usize,

    // Dense slots for resident chunks
    slot_to_key: Vec<ChunkKey>,    // slot -> key
    chunk_meta: Vec<ChunkMetaGpu>, // slot -> meta
    uploads: Vec<ChunkUpload>,     // pending GPU writes this frame
    changed: bool,                 // still useful for other systems

    // Node arena (in units of NodeGpu elements)
    arena: NodeArena,

    // Chunk grid for GPU lookup (same as before)
    grid_origin_chunk: [i32; 3],
    grid_dims: [u32; 3],
    chunk_grid: Vec<u32>,
}

impl ChunkManager {
    pub fn new(gen: Arc<WorldGen>) -> Self {
        let (tx_job, rx_job) = unbounded::<BuildJob>();
        let (tx_done, rx_done) = unbounded::<BuildDone>();
        spawn_workers(gen.clone(), rx_job, tx_done);

        // Arena capacity in NodeGpu elements
        let node_capacity = (config::NODE_BUDGET_BYTES / std::mem::size_of::<NodeGpu>()) as u32;

        // Grid size (KEEP box)
        let nx = (2 * config::KEEP_RADIUS + 1) as u32;
        let nz = nx;
        let ny = GRID_Y_COUNT;
        let grid_len = (nx * ny * nz) as usize;

        Self {
            gen,
            chunks: HashMap::new(),
            build_queue: VecDeque::new(),
            cancels: HashMap::new(),
            tx_job,
            rx_done,
            in_flight: 0,

            slot_to_key: Vec::new(),
            chunk_meta: Vec::new(),
            uploads: Vec::new(),
            changed: false,

            arena: NodeArena::new(node_capacity),

            grid_origin_chunk: [0, 0, 0],
            grid_dims: [nx, ny, nz],
            chunk_grid: vec![INVALID_U32; grid_len],
        }
    }

    // -------- public API --------

    pub fn chunk_count(&self) -> u32 {
        self.slot_to_key.len() as u32
    }

    pub fn chunk_meta(&self) -> &[ChunkMetaGpu] {
        &self.chunk_meta
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

    pub fn changed(&mut self) -> bool {
        let c = self.changed;
        self.changed = false;
        c
    }

    // -------- update --------

    pub fn update(&mut self, world: &Arc<WorldGen>, cam_pos_m: Vec3, cam_fwd: Vec3) {
        self.uploads.clear();

        // center chunk (ground-anchored)
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

        let desired = Self::desired_chunks(center, config::ACTIVE_RADIUS);
        let keep = Self::desired_chunks(center, config::KEEP_RADIUS);
        let keep_set: HashSet<ChunkKey> = keep.iter().copied().collect();

        // queue missing desired (reset cancel token to false)
        for k in &desired {
            match self.chunks.get(k) {
                None | Some(ChunkState::Missing) => {
                    let c = self.cancel_token(*k);
                    c.store(false, Ordering::Relaxed);

                    self.chunks.insert(*k, ChunkState::Queued);
                    self.build_queue.push_back(*k);
                }
                _ => {}
            }
        }

        // unload outside keep (also cancel queued/building)
        {
            let keys_snapshot: Vec<ChunkKey> = self.chunks.keys().copied().collect();
            for k in keys_snapshot {
                if !keep_set.contains(&k) {
                    self.unload_chunk(k);
                }
            }
        }

        // dispatch builds (A3: forward-cone priority)
        sort_queue_near_first(&mut self.build_queue, center, cam_fwd);

        while self.in_flight < config::MAX_IN_FLIGHT {
            let Some(k) = self.build_queue.pop_front() else { break; };

            if !keep_set.contains(&k) {
                // cancel if it was pending
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
                    self.chunks.insert(k, ChunkState::Queued);
                    break;
                }
            }
        }

        // harvest done
        while let Ok(done) = self.rx_done.try_recv() {
            if self.in_flight > 0 {
                self.in_flight -= 1;
            }

            // if cancelled after completion, drop
            if done.cancel.load(Ordering::Relaxed) {
                continue;
            }

            if !keep_set.contains(&done.key) {
                // drop build result, also mark cancelled to avoid late reuse
                self.cancel_token(done.key).store(true, Ordering::Relaxed);
                self.chunks.insert(done.key, ChunkState::Missing);
                continue;
            }

            // still relevant
            self.on_build_done(done.key, done.nodes);
        }

        // rebuild grid mapping for current KEEP region
        self.rebuild_grid(center);
    }

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

    fn on_build_done(&mut self, key: ChunkKey, nodes: Vec<NodeGpu>) {
        // if this chunk got cancelled while the result was in flight, drop it
        if let Some(c) = self.cancels.get(&key) {
            if c.load(Ordering::Relaxed) {
                self.chunks.insert(key, ChunkState::Missing);
                return;
            }
        }

        // allocate node range
        let need = nodes.len() as u32;
        let Some(node_base) = self.arena.alloc(need) else {
            // out of arena budget; requeue (make sure it isn't cancelled)
            self.cancel_token(key).store(false, Ordering::Relaxed);

            self.chunks.insert(key, ChunkState::Queued);
            self.build_queue.push_back(key);
            return;
        };

        // allocate slot (dense)
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

        // mark resident
        self.chunks.insert(
            key,
            ChunkState::Resident(Resident {
                slot,
                node_base,
                node_count: need,
            }),
        );

        // schedule GPU upload (nodes + meta)
        self.uploads.push(ChunkUpload {
            slot,
            meta,
            node_base,
            nodes,
        });

        self.changed = true;
    }

    fn unload_chunk(&mut self, key: ChunkKey) {
        let Some(state) = self.chunks.get(&key) else { return; };

        match *state {
            ChunkState::Resident(res) => {
                // free node arena range
                self.arena.free(res.node_base, res.node_count);

                // remove slot densely by swap-remove
                let dead_slot = res.slot as usize;
                let last_slot = self.slot_to_key.len().saturating_sub(1);

                if dead_slot != last_slot {
                    let moved_key = self.slot_to_key[last_slot];
                    self.slot_to_key[dead_slot] = moved_key;

                    // move meta
                    let moved_meta = self.chunk_meta[last_slot];
                    self.chunk_meta[dead_slot] = moved_meta;

                    // update moved chunk's Resident.slot
                    if let Some(state) = self.chunks.get_mut(&moved_key) {
                        if let ChunkState::Resident(mr) = state {
                            mr.slot = dead_slot as u32;
                        }
                    }

                    // schedule meta rewrite for moved slot
                    self.uploads.push(ChunkUpload {
                        slot: dead_slot as u32,
                        meta: self.chunk_meta[dead_slot],
                        node_base: 0,
                        nodes: Vec::new(), // meta-only update (Renderer will ignore empty nodes)
                    });
                }

                self.slot_to_key.pop();
                self.chunk_meta.pop();

                self.chunks.insert(key, ChunkState::Missing);
                self.changed = true;
            }

            ChunkState::Queued | ChunkState::Building => {
                // cancel work in-flight (A1)
                self.cancel_token(key).store(true, Ordering::Relaxed);

                self.chunks.insert(key, ChunkState::Missing);
                self.changed = true;
            }

            _ => {}
        }
    }

    fn rebuild_grid(&mut self, center: ChunkKey) {
        let nx = (2 * config::KEEP_RADIUS + 1) as u32;
        let nz = nx;
        let ny = GRID_Y_COUNT;

        self.grid_dims = [nx, ny, nz];

        let ox = center.x - config::KEEP_RADIUS;
        let oz = center.z - config::KEEP_RADIUS;
        let oy = center.y + GRID_Y_MIN_DY;

        self.grid_origin_chunk = [ox, oy, oz];

        let needed = (nx * ny * nz) as usize;
        if self.chunk_grid.len() != needed {
            self.chunk_grid.resize(needed, INVALID_U32);
        }
        self.chunk_grid.fill(INVALID_U32);

        // Fill grid from resident chunks (slot -> key)
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
