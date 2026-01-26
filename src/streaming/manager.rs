use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use crossbeam_channel::{unbounded, Receiver, Sender};
use glam::Vec3;

use crate::{
    config,
    render::{ChunkMetaGpu, NodeGpu},
    world::WorldGen,
};

use super::{
    chunk::{ChunkCpu, ChunkKey, ChunkState},
    workers::{spawn_workers, BuildDone, BuildJob},
};

pub struct ChunkManager {
    gen: Arc<WorldGen>,

    chunks: HashMap<ChunkKey, ChunkState>,
    build_queue: VecDeque<ChunkKey>,

    tx_job: Sender<BuildJob>,
    rx_done: Receiver<BuildDone>,
    in_flight: usize,

    nodes_packed: Vec<NodeGpu>,
    chunks_meta: Vec<ChunkMetaGpu>,

    changed: bool,
}

impl ChunkManager {
    pub fn new(gen: Arc<WorldGen>) -> Self {
        let (tx_job, rx_job) = unbounded::<BuildJob>();
        let (tx_done, rx_done) = unbounded::<BuildDone>();
        spawn_workers(gen.clone(), rx_job, tx_done);

        Self {
            gen,
            chunks: HashMap::new(),
            build_queue: VecDeque::new(),
            tx_job,
            rx_done,
            in_flight: 0,
            nodes_packed: Vec::new(),
            chunks_meta: Vec::new(),
            changed: true,
        }
    }

    pub fn changed(&self) -> bool {
        self.changed
    }

    pub fn packed_nodes(&self) -> &[NodeGpu] {
        &self.nodes_packed
    }

    pub fn packed_meta(&self) -> &[ChunkMetaGpu] {
        &self.chunks_meta
    }

    pub fn chunk_count(&self) -> u32 {
        self.chunks_meta.len() as u32
    }

    pub fn update(&mut self, world: &WorldGen, camera_pos_m: Vec3) {
        self.changed = false;

        // --- ground-anchored center chunk (matches your original) ---
        let cam_vx = (camera_pos_m.x / config::VOXEL_SIZE_M_F32).floor() as i32;
        let cam_vz = (camera_pos_m.z / config::VOXEL_SIZE_M_F32).floor() as i32;

        let ccx = cam_vx.div_euclid(config::CHUNK_SIZE as i32);
        let ccz = cam_vz.div_euclid(config::CHUNK_SIZE as i32);

        let ground_y_vox = world.ground_height(cam_vx, cam_vz);
        let ground_cy = ground_y_vox.div_euclid(config::CHUNK_SIZE as i32);

        let center = ChunkKey {
            x: ccx,
            y: ground_cy,
            z: ccz,
        };

        let desired = desired_chunks(center, config::ACTIVE_RADIUS);
        let keep = desired_chunks(center, config::KEEP_RADIUS);
        let keep_set: HashSet<ChunkKey> = keep.iter().copied().collect();

        // --- queue missing desired chunks ---
        for k in &desired {
            match self.chunks.get(k) {
                None | Some(ChunkState::Missing) => {
                    self.chunks.insert(*k, ChunkState::Queued);
                    self.build_queue.push_back(*k);
                }
                _ => {}
            }
        }

        // --- unload outside keep ---
        let keys_snapshot: Vec<ChunkKey> = self.chunks.keys().copied().collect();
        for k in keys_snapshot {
            if !keep_set.contains(&k) {
                match self.chunks.get(&k) {
                    Some(ChunkState::Ready(_)) => {
                        self.chunks.insert(k, ChunkState::Missing);
                        self.changed = true;
                    }
                    Some(ChunkState::Queued) | Some(ChunkState::Building) => {
                        self.chunks.insert(k, ChunkState::Missing);
                    }
                    _ => {}
                }
            }
        }

        // --- worker dispatch ---
        sort_queue_near_first(&mut self.build_queue, center);

        while self.in_flight < config::MAX_IN_FLIGHT {
            let Some(k) = self.build_queue.pop_front() else { break; };

            if !keep_set.contains(&k) {
                self.chunks.insert(k, ChunkState::Missing);
                continue;
            }

            if matches!(self.chunks.get(&k), Some(ChunkState::Queued)) {
                self.chunks.insert(k, ChunkState::Building);
                if self.tx_job.send(BuildJob { key: k }).is_ok() {
                    self.in_flight += 1;
                } else {
                    self.chunks.insert(k, ChunkState::Queued);
                    break;
                }
            }
        }

        // --- harvest done ---
        while let Ok(done) = self.rx_done.try_recv() {
            if self.in_flight > 0 {
                self.in_flight -= 1;
            }

            if !keep_set.contains(&done.key) {
                self.chunks.insert(done.key, ChunkState::Missing);
                continue;
            }

            self.chunks.insert(
                done.key,
                ChunkState::Ready(ChunkCpu { nodes: done.nodes }),
            );
            self.changed = true;
        }

        // --- pack KEEP chunks every frame (so chunk_count is always correct) ---
        self.pack_keep(&keep);
    }

    fn pack_keep(&mut self, keep: &[ChunkKey]) {
        self.nodes_packed.clear();
        self.chunks_meta.clear();

        let node_stride = std::mem::size_of::<NodeGpu>();
        let mut used_bytes: usize = 0;

        for k in keep {
            let Some(ChunkState::Ready(cpu)) = self.chunks.get(k) else { continue; };

            let chunk_bytes = cpu.nodes.len() * node_stride;
            if used_bytes + chunk_bytes > config::NODE_BUDGET_BYTES {
                continue;
            }

            let origin = [
                k.x * config::CHUNK_SIZE as i32,
                k.y * config::CHUNK_SIZE as i32,
                k.z * config::CHUNK_SIZE as i32,
            ];

            let node_base = self.nodes_packed.len() as u32;
            self.nodes_packed.extend_from_slice(&cpu.nodes);
            used_bytes += chunk_bytes;

            self.chunks_meta.push(ChunkMetaGpu {
                origin: [origin[0], origin[1], origin[2], 0],
                node_base,
                node_count: cpu.nodes.len() as u32,
                _pad0: 0,
                _pad1: 0,
            });
        }
    }
}

// --------------------- helpers ---------------------

fn desired_chunks(center: ChunkKey, radius: i32) -> Vec<ChunkKey> {
    let mut out = Vec::new();
    for dy in -1..=2 {
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

fn sort_queue_near_first(queue: &mut VecDeque<ChunkKey>, center: ChunkKey) {
    let mut v: Vec<ChunkKey> = queue.drain(..).collect();
    v.sort_by_key(|k| (k.x - center.x).abs() + (k.z - center.z).abs() + 2 * (k.y - center.y).abs());
    queue.extend(v);
}
