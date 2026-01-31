
mod build;
mod grid;
mod slots;
mod stats;

mod ground;
mod keep;
mod uploads;


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
    streaming::types::StreamStats,
};

use crate::streaming::{
    NodeArena,
    cache::ChunkCache,
    priority::sort_queue_near_first,
    types::*,
    workers::spawn_workers,
};

/// Build-related state bucket.
pub(crate) struct BuildState {
    pub chunks: HashMap<ChunkKey, ChunkState>,

    pub build_queue: VecDeque<ChunkKey>,
    pub queued_set: HashSet<ChunkKey>,
    pub cancels: HashMap<ChunkKey, Arc<AtomicBool>>,

    pub tx_job: Sender<BuildJob>,
    pub rx_done: Receiver<BuildDone>,
    pub in_flight: usize,

    pub last_center: Option<ChunkKey>,
    pub to_unload: Vec<ChunkKey>,
}

/// Slot-residency bucket.
pub(crate) struct SlotState {
    pub slot_to_key: Vec<ChunkKey>,
    pub chunk_meta: Vec<ChunkMetaGpu>,
    pub slot_macro: Vec<Arc<[u32]>>,
    pub slot_colinfo: Vec<Arc<[u32]>>,
    pub resident_slots: usize,
}

/// Upload bucket.
pub(crate) struct UploadState {
    pub uploads_rewrite: VecDeque<ChunkUpload>,
    pub uploads_active:  VecDeque<ChunkUpload>,
    pub uploads_other:   VecDeque<ChunkUpload>,
}

/// Grid bucket.
pub(crate) struct GridState {
    pub grid_dirty: bool,
    pub grid_origin_chunk: [i32; 3],
    pub grid_dims: [u32; 3],
    pub chunk_grid: Vec<u32>,
}

/// Column ground cache bucket.
pub(crate) struct GroundState {
    pub col_ground_cy: Vec<i32>,
}

/// Offset precomputes bucket.
pub(crate) struct Offsets {
    pub active_offsets: Vec<(i32, i32, i32)>,
    pub priority_offsets: Vec<(i32, i32, i32)>,
}

pub struct ChunkManager {
    pub(crate) build: BuildState,
    pub(crate) slots: SlotState,
    pub(crate) uploads: UploadState,
    pub(crate) grid: GridState,
    pub(crate) ground: GroundState,
    pub(crate) offsets: Offsets,

    pub(crate) arena: NodeArena,
    pub(crate) cache: ChunkCache,
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

        let active_offsets = keep::build_offsets(config::ACTIVE_RADIUS);
        let priority_offsets = keep::build_offsets(PRIORITY_RADIUS);

        Self {
            build: BuildState {
                chunks: HashMap::default(),
                build_queue: VecDeque::new(),
                queued_set: HashSet::default(),
                cancels: HashMap::default(),
                tx_job,
                rx_done,
                in_flight: 0,
                last_center: None,
                to_unload: Vec::new(),
            },
            slots: SlotState {
                slot_to_key: Vec::new(),
                chunk_meta: Vec::new(),
                slot_macro: Vec::new(),
                slot_colinfo: Vec::new(),
                resident_slots: 0,
            },
            uploads: UploadState {
                uploads_rewrite: VecDeque::new(),
                uploads_active:  VecDeque::new(),
                uploads_other:   VecDeque::new(),
            },
            grid: GridState {
                grid_dirty: true,
                grid_origin_chunk: [0, 0, 0],
                grid_dims: [nx, ny, nz],
                chunk_grid: vec![INVALID_U32; grid_len],
            },
            ground: GroundState { col_ground_cy: Vec::new() },
            offsets: Offsets { active_offsets, priority_offsets },

            arena: NodeArena::new(node_capacity),
            cache: ChunkCache::new(),
        }
    }

    /// Main frame update (same logic as before; now calls into submodules).
    pub fn update(&mut self, world: &Arc<WorldGen>, cam_pos_m: Vec3, cam_fwd: Vec3) -> bool {
        // 1) compute center
        let center = keep::compute_center(world.as_ref(), cam_pos_m);

        // 2) ensure ground cache + publish origin
        ground::ensure_column_cache(self, world.as_ref(), center);

        // 3) publish center (rebucket uploads on center change)
        if keep::publish_center_and_rebucket(self, center) {
            // keep::publish_center_and_rebucket already rebuckets uploads
        }

        // 4) ensure priority box builds/promotions
        build::ensure_priority_box(self, center);

        // 5) enqueue active offsets
        build::enqueue_active_ring(self, center);

        // 6) unload outside keep
        build::unload_outside_keep(self, center);

        // 7) handle center-change cleanup/sort
        build::on_center_change_resort(self, center, cam_fwd);

        // 8) dispatch builds
        build::dispatch_builds(self, center);

        // 9) harvest completions
        build::harvest_done_builds(self, center);

        // 10) rebuild grid if dirty
        grid::rebuild_if_dirty(self, center)
    }

    // --- Public API (same signatures; implemented in submodules via impl blocks) ---

    pub fn chunk_count(&self) -> u32 { self.slots.resident_slots as u32 }
    pub fn grid_origin(&self) -> [i32; 3] { self.grid.grid_origin_chunk }
    pub fn grid_dims(&self) -> [u32; 3] { self.grid.grid_dims }
    pub fn chunk_grid(&self) -> &[u32] { &self.grid.chunk_grid }

    pub fn take_uploads(&mut self) -> Vec<ChunkUpload> {
        uploads::take_all(self)
    }

    pub fn take_uploads_budgeted(&mut self) -> Vec<ChunkUpload> {
        uploads::take_budgeted(self)
    }

    pub fn commit_uploads_applied(&mut self, applied: &[ChunkUpload]) -> bool {
        slots::commit_uploads_applied(self, applied)
    }

    pub fn stats(&self) -> Option<StreamStats> {
        stats::stats(self)
    }
}
