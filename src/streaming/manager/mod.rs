
// src/streaming/manager/mod.rs
pub mod build;
mod grid;
mod slots;
mod stats;
mod visibility;
pub mod ground;
mod keep;
mod uploads;


use std::{
    collections::{VecDeque, BinaryHeap},
    mem::size_of,
    sync::{
        atomic::AtomicBool,
        Arc,
    },
};
use crate::svo::builder::BuildTimingsMs;

use crossbeam_channel::{bounded, Receiver, Sender};
use glam::Vec3;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::app::config;
use crate::{
    render::gpu_types::{ChunkMetaGpu, NodeGpu},
    world::WorldGen,
    streaming::types::StreamStats,
};

use crate::streaming::{
    NodeArena,
    cache::ChunkCache,
    types::*,
    workers::spawn_workers,
};

/// Column visibility (PVS) bucket.
pub(crate) struct VisibilityState {
    pub col_visible: Vec<u8>,
}


/// Build-related state bucket.
pub(crate) struct BuildState {
    pub chunks: HashMap<ChunkKey, ChunkState>,

    pub build_queue: VecDeque<ChunkKey>,
    pub build_heap: BinaryHeap<build::HeapItem>,
    pub queued_set: HashSet<ChunkKey>,
    pub cancels: HashMap<ChunkKey, Arc<AtomicBool>>,

    pub tx_job: Sender<BuildJob>,
    pub rx_done: Receiver<BuildDone>,
    pub in_flight: usize,

    pub last_center: Option<ChunkKey>,
    pub to_unload: Vec<ChunkKey>,

    pub rebuild_queue: VecDeque<ChunkKey>,
    pub rebuild_set: HashSet<ChunkKey>,
    // needed to score heap inserts when center doesn't change
    pub last_cam_fwd: Vec3,
    // monotonically increasing tie-breaker so newer bumps win
    pub heap_tie: u32,

    // if true, allow negative dy (we are in caves / below terrain surface)
    pub cam_below_ground: bool,
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

    pub slot_rewrite_q: VecDeque<u32>,
    pub slot_rewrite_set: HashSet<u32>,
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
    pub col_ground_y_vox: Vec<i32>,
}

/// Offset precomputes bucket.
pub(crate) struct Offsets {
    pub active_offsets: Vec<(i32, i32, i32)>,
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

    pub(crate) pinned: HashSet<ChunkKey>,
    pub(crate) edits: Arc<crate::world::edits::EditStore>,

    pub(crate) vis: VisibilityState,

    // Build timing window (drained on stats() print cadence)
    pub timing: StreamTimingWindow,
}

impl ChunkManager {
    pub fn new(gen: Arc<WorldGen>) -> Self {
        let cap = (config::MAX_IN_FLIGHT * 8).max(64);

        let (tx_job, rx_job) = bounded::<BuildJob>(cap);
        let (tx_done, rx_done) = bounded::<BuildDone>(cap);

        let edits = Arc::new(crate::world::edits::EditStore::new());
        spawn_workers(gen,  rx_job, tx_done);

        let node_capacity = (config::NODE_BUDGET_BYTES / size_of::<NodeGpu>()) as u32;

        let nx = (2 * config::KEEP_RADIUS + 1) as u32;
        let nz = nx;
        let ny = GRID_Y_COUNT;

        let grid_len = (nx * ny * nz) as usize;

        let active_offsets = keep::build_offsets(config::ACTIVE_RADIUS);

        Self {
            build: BuildState {
                chunks: HashMap::default(),
                build_queue: VecDeque::new(),
                build_heap: BinaryHeap::new(),
                queued_set: HashSet::default(),
                cancels: HashMap::default(),
                tx_job,
                rx_done,
                in_flight: 0,
                last_center: None,
                to_unload: Vec::new(),
                rebuild_queue: VecDeque::new(),
                rebuild_set: HashSet::default(),
                last_cam_fwd: Vec3::Z, // or Vec3::ZERO
                heap_tie: 0,
                cam_below_ground: false,
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
                slot_rewrite_q: VecDeque::new(),
                slot_rewrite_set: HashSet::default(),
            },
            grid: GridState {
                grid_dirty: true,
                grid_origin_chunk: [0, 0, 0],
                grid_dims: [nx, ny, nz],
                chunk_grid: vec![INVALID_U32; grid_len],
            },
            ground: GroundState { col_ground_y_vox: Vec::new() },
            offsets: Offsets { active_offsets },

            arena: NodeArena::new(node_capacity),
            cache: ChunkCache::new(),

            pinned: HashSet::default(),
            edits,

            vis: VisibilityState { col_visible: Vec::new() },

            timing: StreamTimingWindow::default(),
        }
    }

    /// Main frame update (same logic as before; now calls into submodules).
    pub fn update(&mut self, world: &Arc<WorldGen>, cam_pos_m: Vec3, cam_fwd: Vec3) -> bool {
        // 1) compute center
        let center = {
            let cam_vx = (cam_pos_m.x / config::VOXEL_SIZE_M_F32).floor() as i32;
            let cam_vz = (cam_pos_m.z / config::VOXEL_SIZE_M_F32).floor() as i32;

            let cs = config::CHUNK_SIZE as i32;
            let half = cs / 2;

            let ccx = cam_vx.div_euclid(cs);
            let ccz = cam_vz.div_euclid(cs);

            // Fast path: use cached column ground if available
            if let Some(ground_cy) = ground::ground_cy_for_column(self, ccx, ccz) {
                ChunkKey { x: ccx, y: ground_cy, z: ccz }
            } else {
                // Slow fallback (only when outside cache / first frame)
                keep::compute_center( world.as_ref(), cam_pos_m)
            }
        };

        self.build.last_cam_fwd = cam_fwd;

        let prev_cam_below_ground = self.build.cam_below_ground;
        let prev_cam_fwd = self.build.last_cam_fwd;
        
        // 2) ensure ground cache (only does real work when origin changes)
        ground::ensure_column_cache(self, world.as_ref(), center);

        // determine if camera is below ground at the current center column
        let cam_y_vox = cam_pos_m.y / config::VOXEL_SIZE_M_F32;
        let ground_y_vox = ground::ground_y_vox_for_column(self, center.x, center.z)
            .unwrap_or_else(|| {
                // fallback (should be rare): compute at center column
                let cs = config::CHUNK_SIZE as i32;
                let half = cs / 2;
                let wx = center.x * cs + half;
                let wz = center.z * cs + half;
                world.ground_height(wx, wz)
            }) as f32;

        // small margin to avoid flicker when standing exactly on the surface
        self.build.cam_below_ground = cam_y_vox < (ground_y_vox - 1.0);

        let cam_below_ground_changed = self.build.cam_below_ground != prev_cam_below_ground;

        // rebuild heap when turning a lot (avoids “I turned to look at it” starvation)
        let cam_fwd_changed = {
            let a = glam::Vec2::new(prev_cam_fwd.x, prev_cam_fwd.z);
            let b = glam::Vec2::new(cam_fwd.x, cam_fwd.z);
            if a.length_squared() < 1e-6 || b.length_squared() < 1e-6 {
                false
            } else {
                let da = a.normalize();
                let db = b.normalize();
                da.dot(db) < 0.90 // ~25° turn threshold; tune
            }
        };



        // 3) publish center (rebuckets uploads only if changed)
        let center_changed = keep::publish_center_and_rebucket(self, center);

        let planning_changed = center_changed || cam_below_ground_changed || cam_fwd_changed;

        if planning_changed {
            visibility::ensure_visible_columns(self, center, cam_pos_m);
        }

        // Always do cheap “keep things moving”
        build::harvest_done_builds(self, center);
        build::dispatch_builds(self, center);

        // Do planning when needed (not only on center change)
        if planning_changed {
            build::enqueue_active_ring(self, center);
            build::unload_outside_keep(self, center);

            // heap rebuild is the expensive part; only when we detect a meaningful change
            build::rebuild_build_heap(self, center, cam_fwd);
        }


        let changed = grid::rebuild_if_dirty(self, center);

        #[cfg(debug_assertions)]
        slots::assert_slot_invariants(self);

        changed
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

    pub fn stats(&mut self) -> Option<StreamStats> {
        stats::stats(self)
    }


    /// Cheap per-frame maintenance. MUST NOT do expensive planning.
    /// - harvest worker completions
    /// - keep workers fed using the existing build_heap
    /// - rebuild grid if dirty
    pub fn pump_completed(&mut self) -> bool {
        let Some(center) = self.build.last_center else {
            // No center published yet => nothing meaningful to pump.
            return false;
        };

        // 1) Drain completed builds (non-blocking)
        build::harvest_done_builds(self, center);
        #[cfg(debug_assertions)]
        slots::assert_slot_invariants(self);

        // 2) Keep workers busy (uses existing heap; no heap rebuild here)
        build::dispatch_builds(self, center);

        // 3) Rebuild grid if something changed (dirty flag set by slot ops)
        let changed = grid::rebuild_if_dirty(self, center);

        #[cfg(debug_assertions)]
        slots::assert_slot_invariants(self);

        changed
    }
}


#[derive(Clone, Debug, Default)]
pub struct StreamTimingWindow {
    pub builds_done: u32,
    pub builds_canceled: u32,

    pub queue_ms_sum: f64,
    pub queue_ms_max: f64,

    pub build_ms_sum: f64,
    pub build_ms_max: f64,

    pub nodes_sum: u64,
    pub nodes_max: u32,

    pub bt_sum: BuildTimingsMs, // sums (times)
    pub bt_max: BuildTimingsMs, // max  (times + counters as max signal)
}


impl StreamTimingWindow {
    #[inline]
    pub fn record_build(
        &mut self,
        queue_ms: f64,
        build_ms: f64,
        nodes: u32,
        canceled: bool,
        tim: &BuildTimingsMs, // NEW
    ) {
        if canceled {
            self.builds_canceled += 1;
            return;
        }

        self.builds_done += 1;

        self.queue_ms_sum += queue_ms;
        self.queue_ms_max = self.queue_ms_max.max(queue_ms);

        self.build_ms_sum += build_ms;
        self.build_ms_max = self.build_ms_max.max(build_ms);

        self.nodes_sum += nodes as u64;
        self.nodes_max = self.nodes_max.max(nodes);

        // ---- per-stage sums ----
        self.bt_sum.total         += tim.total;
        self.bt_sum.height_cache  += tim.height_cache;
        self.bt_sum.tree_mask     += tim.tree_mask;
        self.bt_sum.ground_2d     += tim.ground_2d;
        self.bt_sum.ground_mip    += tim.ground_mip;
        self.bt_sum.tree_top      += tim.tree_top;
        self.bt_sum.tree_mip      += tim.tree_mip;
        self.bt_sum.material_fill += tim.material_fill;
        self.bt_sum.cave_mask     += tim.cave_mask;
        self.bt_sum.colinfo       += tim.colinfo;
        self.bt_sum.colinfo_pack  += tim.colinfo_pack;
        self.bt_sum.prefix_x      += tim.prefix_x;
        self.bt_sum.prefix_y      += tim.prefix_y;
        self.bt_sum.prefix_z      += tim.prefix_z;
        self.bt_sum.macro_occ     += tim.macro_occ;
        self.bt_sum.svo_build     += tim.svo_build;
        self.bt_sum.ropes         += tim.ropes;

        // ---- per-stage maxima ----
        self.bt_max.total         = self.bt_max.total.max(tim.total);
        self.bt_max.height_cache  = self.bt_max.height_cache.max(tim.height_cache);
        self.bt_max.tree_mask     = self.bt_max.tree_mask.max(tim.tree_mask);
        self.bt_max.ground_2d     = self.bt_max.ground_2d.max(tim.ground_2d);
        self.bt_max.ground_mip    = self.bt_max.ground_mip.max(tim.ground_mip);
        self.bt_max.tree_top      = self.bt_max.tree_top.max(tim.tree_top);
        self.bt_max.tree_mip      = self.bt_max.tree_mip.max(tim.tree_mip);
        self.bt_max.material_fill = self.bt_max.material_fill.max(tim.material_fill);
        self.bt_max.cave_mask     = self.bt_max.cave_mask.max(tim.cave_mask);
        self.bt_max.colinfo       = self.bt_max.colinfo.max(tim.colinfo);
        self.bt_max.colinfo_pack  = self.bt_max.colinfo_pack.max(tim.colinfo_pack);
        self.bt_max.prefix_x      = self.bt_max.prefix_x.max(tim.prefix_x);
        self.bt_max.prefix_y      = self.bt_max.prefix_y.max(tim.prefix_y);
        self.bt_max.prefix_z      = self.bt_max.prefix_z.max(tim.prefix_z);
        self.bt_max.macro_occ     = self.bt_max.macro_occ.max(tim.macro_occ);
        self.bt_max.svo_build     = self.bt_max.svo_build.max(tim.svo_build);
        self.bt_max.ropes         = self.bt_max.ropes.max(tim.ropes);

        // counters as maxima signal (optional but useful)
        self.bt_max.cache_w           = self.bt_max.cache_w.max(tim.cache_w);
        self.bt_max.cache_h           = self.bt_max.cache_h.max(tim.cache_h);
        self.bt_max.tree_cells_tested = self.bt_max.tree_cells_tested.max(tim.tree_cells_tested);
        self.bt_max.tree_instances    = self.bt_max.tree_instances.max(tim.tree_instances);
        self.bt_max.solid_voxels      = self.bt_max.solid_voxels.max(tim.solid_voxels);
        self.bt_max.nodes             = self.bt_max.nodes.max(tim.nodes);
    }

    #[inline]
    pub fn drain(&mut self) -> Self {
        std::mem::take(self)
    }
}

