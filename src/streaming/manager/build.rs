use std::sync::{
    atomic::{AtomicBool, Ordering as AtomicOrdering},
    Arc,
};

use crossbeam_channel::TrySendError;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use glam::Vec3;

use crate::streaming::priority::priority_score;
use crate::streaming::types::*;
use crate::{config, render::gpu_types::{NodeGpu, NodeRopesGpu}};

use super::{ground, keep, slots};
use super::ChunkManager;

#[inline(always)]
fn in_keep(mgr: &ChunkManager, center: ChunkKey, k: ChunkKey) -> bool {
    let dx = k.x - center.x;
    let dz = k.z - center.z;

    if dx < -config::KEEP_RADIUS || dx > config::KEEP_RADIUS { return false; }
    if dz < -config::KEEP_RADIUS || dz > config::KEEP_RADIUS { return false; }

    let Some(ground_cy) = ground::ground_cy_for_column(mgr, k.x, k.z) else {
        return false;
    };

    let dy = k.y - ground_cy;
    dy >= y_band_min() && dy <= y_band_max()
}

#[inline]
fn cancel_token(mgr: &mut ChunkManager, key: ChunkKey) -> Arc<AtomicBool> {
    mgr.build
        .cancels
        .entry(key)
        .or_insert_with(|| Arc::new(AtomicBool::new(false)))
        .clone()
}

#[inline]
fn queue_build_front(mgr: &mut ChunkManager, k: ChunkKey) {
    if mgr.build.queued_set.contains(&k) {
        if let Some(pos) = mgr.build.build_queue.iter().position(|x| *x == k) {
            mgr.build.build_queue.remove(pos);
            mgr.build.build_queue.push_front(k);
        }
        return;
    }

    mgr.build.chunks.insert(k, ChunkState::Queued);
    mgr.build.queued_set.insert(k);
    mgr.build.build_queue.push_front(k);
}

pub fn ensure_priority_box(mgr: &mut ChunkManager, center: ChunkKey) {
    let n = mgr.offsets.priority_offsets.len();

    for i in 0..n {
        // copy the tuple out; immutable borrow ends right here
        let (dx, dy, dz) = mgr.offsets.priority_offsets[i];

        let x = center.x + dx;
        let z = center.z + dz;

        let Some(ground_cy_col) = ground::ground_cy_for_column(mgr, x, z) else { continue; };
        let k = ChunkKey { x, y: ground_cy_col + dy, z };

        match mgr.build.chunks.get(&k) {
            Some(ChunkState::Resident(_))
            | Some(ChunkState::Uploading(_))
            | Some(ChunkState::Queued)
            | Some(ChunkState::Building) => {
                if matches!(mgr.build.chunks.get(&k), Some(ChunkState::Queued)) {
                    queue_build_front(mgr, k);
                }
            }
            None => {
                if mgr.cache.get(&k).is_some() {
                    let _ = try_promote_from_cache(mgr, center, k);
                    continue;
                }

                let c = cancel_token(mgr, k);
                c.store(false, AtomicOrdering::Relaxed);
                queue_build_front(mgr, k);
            }
        }
    }
}


pub fn enqueue_active_ring(mgr: &mut ChunkManager, center: ChunkKey) {
    let n_active = mgr.offsets.active_offsets.len();
    for i in 0..n_active {
        let (dx, dy, dz) = mgr.offsets.active_offsets[i];
        let x = center.x + dx;
        let z = center.z + dz;

        let Some(ground_cy_col) = ground::ground_cy_for_column(mgr, x, z) else {
            continue;
        };
        let k = ChunkKey { x, y: ground_cy_col + dy, z };

        match mgr.build.chunks.get(&k) {
            Some(ChunkState::Resident(_))
            | Some(ChunkState::Uploading(_))
            | Some(ChunkState::Queued)
            | Some(ChunkState::Building) => {}
            None => {
                if mgr.cache.get(&k).is_some() {
                    let _ = try_promote_from_cache(mgr, center, k);
                    continue;
                }

                let c = cancel_token(mgr, k);
                c.store(false, AtomicOrdering::Relaxed);

                mgr.build.chunks.insert(k, ChunkState::Queued);
                if mgr.build.queued_set.insert(k) {
                    mgr.build.build_queue.push_back(k);
                }
            }
        }
    }
}

pub fn unload_outside_keep(mgr: &mut ChunkManager, center: ChunkKey) {
    mgr.build.to_unload.clear();

    for &k in mgr.build.chunks.keys() {
        if slots::in_priority_box(mgr, center, k) { continue; }

        if keep::in_active_xz(center, k) {
            match mgr.build.chunks.get(&k) {
                Some(ChunkState::Resident(_)) | Some(ChunkState::Uploading(_)) => continue,
                _ => {}
            }
        }

        if !in_keep(mgr, center, k) {
            mgr.build.to_unload.push(k);
        }
    }

    let unload = std::mem::take(&mut mgr.build.to_unload);
    for k in unload {
        slots::unload_chunk(mgr, center, k);
    }
}

pub fn dispatch_builds(mgr: &mut ChunkManager, center: ChunkKey) {
    let mut attempts = 0usize;
    let max_attempts = mgr.build.build_heap.len().max(1);

    while mgr.build.in_flight < config::MAX_IN_FLIGHT && attempts < max_attempts {
        attempts += 1;

        let Some(item) = mgr.build.build_heap.pop() else { break; };
        let k = item.key;
        mgr.build.queued_set.remove(&k);

        if !in_keep(mgr, center, k) {
            cancel_token(mgr, k).store(true, AtomicOrdering::Relaxed);
            mgr.build.chunks.remove(&k);
            mgr.build.cancels.remove(&k);
            continue;
        }

        if mgr.cache.get(&k).is_some() {
            mgr.build.chunks.remove(&k);
            mgr.build.cancels.remove(&k);
            let _ = try_promote_from_cache(mgr, center, k);
            continue;
        }

        if matches!(mgr.build.chunks.get(&k), Some(ChunkState::Queued)) {
            mgr.build.chunks.insert(k, ChunkState::Building);

            let cancel = cancel_token(mgr, k);
            cancel.store(false, AtomicOrdering::Relaxed);

            match mgr.build.tx_job.try_send(BuildJob { key: k, cancel: cancel.clone() }) {
                Ok(()) => mgr.build.in_flight += 1,
                Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                    mgr.build.chunks.insert(k, ChunkState::Queued);
                    mgr.build.build_queue.push_front(k);
                    mgr.build.queued_set.insert(k);
                    break;
                }
            }
        }
    }
}

pub fn harvest_done_builds(mgr: &mut ChunkManager, center: ChunkKey) {
    let done_backlog = mgr.build.rx_done.len();
    let max_done = (16 + done_backlog / 2).clamp(16, 64);

    let mut done_count = 0usize;
    while done_count < max_done {
        let Ok(done) = mgr.build.rx_done.try_recv() else { break; };
        done_count += 1;

        if mgr.build.in_flight > 0 {
            mgr.build.in_flight -= 1;
        }

        // Drop stale completions (job from an old cancel token)
        let Some(cur_cancel) = mgr.build.cancels.get(&done.key) else { continue; };
        if !Arc::ptr_eq(cur_cancel, &done.cancel) { continue; }

        if done.canceled || done.cancel.load(AtomicOrdering::Relaxed) {
            mgr.build.chunks.remove(&done.key);
            mgr.build.cancels.remove(&done.key);
            continue;
        }

        if !in_keep(mgr, center, done.key) {
            cancel_token(mgr, done.key).store(true, AtomicOrdering::Relaxed);
            mgr.build.chunks.remove(&done.key);
            continue;
        }

        on_build_done(
            mgr,
            center,
            done.key,
            done.nodes,
            done.macro_words,
            done.ropes,
            done.colinfo_words,
        );
    }
}

fn try_promote_from_cache(mgr: &mut ChunkManager, center: ChunkKey, key: ChunkKey) -> bool {
    // If priority box isn't ready, defer non-priority promotions.
    if !slots::priority_box_ready(mgr, center) && !slots::in_priority_box(mgr, center, key) {
        mgr.build.chunks.insert(key, ChunkState::Queued);
        if mgr.build.queued_set.insert(key) {
            mgr.build.build_queue.push_back(key);
        }
        return false;
    }

    let Some(e) = mgr.cache.get(&key) else { return false; };

    let nodes = e.nodes.clone();
    let macro_words = e.macro_words.clone();
    let ropes = e.ropes.clone();
    let colinfo_words = e.colinfo_words.clone();

    mgr.cache.touch(key);
    slots::try_make_uploading(mgr, center, key, nodes, macro_words, ropes, colinfo_words)
}

fn on_build_done(
    mgr: &mut ChunkManager,
    center: ChunkKey,
    key: ChunkKey,
    nodes: Vec<NodeGpu>,
    macro_words: Vec<u32>,
    ropes: Vec<NodeRopesGpu>,
    colinfo_words: Vec<u32>,
) {
    if let Some(c) = mgr.build.cancels.get(&key) {
        if c.load(AtomicOrdering::Relaxed) {
            mgr.build.chunks.remove(&key);
            return;
        }
    }

    let nodes_arc: Arc<[NodeGpu]> = nodes.into();
    let macro_arc: Arc<[u32]> = macro_words.into();
    let ropes_arc: Arc<[NodeRopesGpu]> = ropes.into();
    let colinfo_arc: Arc<[u32]> = colinfo_words.into();

    mgr.cache.put(key, nodes_arc.clone(), macro_arc.clone(), ropes_arc.clone(), colinfo_arc.clone());

    if matches!(mgr.build.chunks.get(&key), Some(ChunkState::Resident(_))) {
        return;
    }

    // Defer GPU upload for non-priority until priority box is GPU-ready.
    if !slots::priority_box_ready(mgr, center) && !slots::in_priority_box(mgr, center, key) {
        mgr.build.chunks.insert(key, ChunkState::Queued);
        if mgr.build.queued_set.insert(key) {
            mgr.build.build_queue.push_back(key);
        }
        return;
    }

    let ok = slots::try_make_uploading(mgr, center, key, nodes_arc, macro_arc, ropes_arc, colinfo_arc);
    if !ok {
        mgr.build.chunks.remove(&key);
    }
}

#[derive(Clone, Copy)]
pub struct HeapItem {
    // Larger is better (because BinaryHeap pops max)
    pub prio: u32,
    pub tie: u32,
    pub key: ChunkKey,
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.prio.cmp(&other.prio).then_with(|| self.tie.cmp(&other.tie))
    }
}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool { self.prio == other.prio && self.tie == other.tie && self.key == other.key }
}
impl Eq for HeapItem {}

#[inline]
fn float_to_ordered_u32(x: f32) -> u32 {
    // Total order trick for IEEE floats (excluding NaN oddities):
    // negative => flip all bits, positive => flip sign bit
    let b = x.to_bits();
    if (b & 0x8000_0000) != 0 { !b } else { b ^ 0x8000_0000 }
}

#[inline]
fn make_prio(score: f32) -> u32 {
    // Want smaller score => higher priority in a max-heap
    let ord = if score.is_nan() { u32::MAX } else { float_to_ordered_u32(score) };
    u32::MAX - ord
}

pub fn rebuild_build_heap(mgr: &mut ChunkManager, center: ChunkKey, cam_fwd: Vec3) {
    // 1) Gather all queued keys: existing heap + staged build_queue
    let mut keys: Vec<ChunkKey> = Vec::with_capacity(mgr.build.build_heap.len() + mgr.build.build_queue.len());

    keys.extend(mgr.build.build_heap.iter().map(|it| it.key));
    keys.extend(mgr.build.build_queue.drain(..)); // staging consumed

    mgr.build.build_heap.clear();

    // 2) Filter invalid keys (copying your keep/sanity checks)
    let origin = mgr.grid.grid_origin_chunk;
    let nx = (2 * config::KEEP_RADIUS + 1) as i32;

    let mut kept: Vec<ChunkKey> = Vec::with_capacity(keys.len());
    let mut drop_keys: Vec<ChunkKey> = Vec::new();

    for k in keys {
        let keep_it =
            matches!(mgr.build.chunks.get(&k), Some(ChunkState::Queued)) &&
            {
                let dx = k.x - center.x;
                let dz = k.z - center.z;
                !(dx < -config::KEEP_RADIUS || dx > config::KEEP_RADIUS ||
                  dz < -config::KEEP_RADIUS || dz > config::KEEP_RADIUS)
            } &&
            {
                let ix = k.x - origin[0];
                let iz = k.z - origin[2];
                !(ix < 0 || iz < 0 || ix >= nx || iz >= nx)
            } &&
            {
                let idx = ((k.z - origin[2]) * nx + (k.x - origin[0])) as usize;
                mgr.ground.col_ground_cy.get(idx).is_some()
            } &&
            {
                let idx = ((k.z - origin[2]) * nx + (k.x - origin[0])) as usize;
                let ground_cy = mgr.ground.col_ground_cy[idx];
                let dy = k.y - ground_cy;
                dy >= y_band_min() && dy <= y_band_max()
            };

        if keep_it { kept.push(k); } else { drop_keys.push(k); }
    }

    // 3) Clean state for dropped keys (prevents orphan growth)
    for k in drop_keys {
        if let Some(c) = mgr.build.cancels.get(&k) {
            c.store(true, AtomicOrdering::Relaxed);
        }
        mgr.build.chunks.remove(&k);
        mgr.build.cancels.remove(&k);
        mgr.build.queued_set.remove(&k);
    }

    // 4) Rebuild queued_set to match kept
    mgr.build.queued_set.clear();
    mgr.build.queued_set.extend(kept.iter().copied());

    // 5) Score + heapify
    let mut items: Vec<HeapItem> = Vec::with_capacity(kept.len());
    for (i, k) in kept.into_iter().enumerate() {
        let mut s = priority_score(k, center, cam_fwd);

        // Preserve your “priority box first” behavior without special-casing structure:
        if super::slots::in_priority_box(mgr, center, k) {
            s -= 20_000.0;
        } else if super::keep::in_active_xz(center, k) {
            s -= 10_000.0;
        }

        items.push(HeapItem { prio: make_prio(s), tie: i as u32, key: k });
    }

    mgr.build.build_heap = BinaryHeap::from(items);
}
