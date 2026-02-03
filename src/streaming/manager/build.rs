// src/streaming/manager/build.rs
use std::sync::{
    atomic::{AtomicBool, Ordering as AtomicOrdering},
    Arc,
};
use std::time::Instant;


use crossbeam_channel::TrySendError;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use glam::Vec3;

use crate::streaming::types::*;
use crate::{render::gpu_types::{NodeGpu, NodeRopesGpu}};
use crate::app::config;

use super::{ground, keep, slots};
use super::ChunkManager;
use crate::streaming::priority::priority_score;

// src/streaming/manager/build.rs


#[inline(always)]
fn y_band_min_dyn(mgr: &ChunkManager) -> i32 {
    if mgr.build.cam_below_ground { y_band_min() } else { 0 }
}

#[inline(always)]
fn y_band_max_dyn(_mgr: &ChunkManager) -> i32 {
    y_band_max()
}

#[inline(always)]
fn in_keep(mgr: &ChunkManager, center: ChunkKey, k: ChunkKey) -> bool {
    if mgr.pinned.contains(&k) {
        return true;
    }

    let dx = k.x - center.x;
    let dz = k.z - center.z;

    // NEW: circular keep
    let r = config::KEEP_RADIUS;
    if dx*dx + dz*dz > r*r {
        return false;
    }

    let ar = config::ACTIVE_RADIUS;
    if dx*dx + dz*dz > ar*ar {
        if !super::visibility::column_visible(mgr, k.x, k.z) {
            return false;
        }
    }


    let Some(ground_cy) = ground::ground_cy_for_column(mgr, k.x, k.z) else {
        return false;
    };

    let dy = k.y - ground_cy;
    dy >= y_band_min_dyn(mgr) && dy <= y_band_max_dyn(mgr)

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
fn queue_build_front(mgr: &mut ChunkManager, center: ChunkKey, k: ChunkKey) {
    // Ensure it is tracked as queued
    match mgr.build.chunks.get(&k) {
        Some(ChunkState::Queued) | Some(ChunkState::Building) | Some(ChunkState::Uploading(_)) | Some(ChunkState::Resident(_)) => {}
        None => {
            mgr.build.chunks.insert(k, ChunkState::Queued);
            mgr.build.queued_set.insert(k);
        }
    }

    // Big boost so it jumps ahead in the heap
    heap_push(mgr, center, k, 100_000.0);
}


pub fn enqueue_active_ring(mgr: &mut ChunkManager, center: ChunkKey) {
    let n_active = mgr.offsets.active_offsets.len();
    for i in 0..n_active {
        let (dx, dy, dz) = mgr.offsets.active_offsets[i];

        if dy < y_band_min_dyn(mgr) || dy > y_band_max_dyn(mgr) {
            continue;
        }

        let x = center.x + dx;
        let z = center.z + dz;

        // Outside priority radius: only consider columns that are in the 360° PVS.
        let ddx = x - center.x;
        let ddz = z - center.z;
        let ar = config::ACTIVE_RADIUS;
        if dx*dx + dz*dz > ar*ar {
            if !super::visibility::column_visible(mgr, x, z) {
                continue;
            }
        }



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
                    heap_push(mgr, center, k, 0.0);
                }

            }
        }
    }
}

pub fn unload_outside_keep(mgr: &mut ChunkManager, center: ChunkKey) {
    mgr.build.to_unload.clear();

    for &k in mgr.build.chunks.keys() {
        if mgr.pinned.contains(&k) { continue; }

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

    // Reserve some capacity so edits (rebuilds) don’t sit behind streaming.
    let reserve_for_rebuilds = 2usize;
    let rebuild_pending = !mgr.build.rebuild_queue.is_empty();
    let max_normal_in_flight = if rebuild_pending {
        config::MAX_IN_FLIGHT.saturating_sub(reserve_for_rebuilds).max(1)
    } else {
        config::MAX_IN_FLIGHT
    };


    // Also dispatch rebuilds for already-slotted chunks (no state changes).
    while mgr.build.in_flight < config::MAX_IN_FLIGHT {
        let Some(k) = mgr.build.rebuild_queue.pop_front() else { break; };
        mgr.build.rebuild_set.remove(&k);

        // Must still exist and still be slotted.
        let Some(st) = mgr.build.chunks.get(&k) else { continue; };
        if !matches!(st, ChunkState::Resident(_) | ChunkState::Uploading(_)) {
            continue;
        }

        // If it’s outside keep, skip (optional but consistent with your other logic).
        if !in_keep(mgr, center, k) {
            continue;
        }

        // Make a fresh cancel token and replace the stored one,
        // so stale completions get dropped by Arc::ptr_eq in harvest_done_builds.
        let cancel = Arc::new(AtomicBool::new(false));
        mgr.build.cancels.insert(k, cancel.clone());

        let edits = mgr.edits.snapshot(k);

        match mgr.build.tx_job.try_send(BuildJob { key: k, cancel, edits, enqueued_at: Instant::now(), }) {
            Ok(()) => mgr.build.in_flight += 1,
            Err(TrySendError::Full(job)) | Err(TrySendError::Disconnected(job)) => {
                // Put it back and stop trying this frame.
                mgr.build.rebuild_set.insert(job.key);
                mgr.build.rebuild_queue.push_front(job.key);
                break;
            }
        }
    }


    while mgr.build.in_flight < max_normal_in_flight && attempts < max_attempts {
        attempts += 1;

        let Some(item) = mgr.build.build_heap.pop() else { break; };
        let k = item.key;

        // Heap contains stale entries; only queued chunks may be dispatched/canceled here.
        let Some(st) = mgr.build.chunks.get(&k) else {
            // Not tracked anymore.
            continue;
        };

        // Ignore anything that isn't currently queued.
        if !matches!(st, ChunkState::Queued) {
            continue;
        }

        // Now it's safe to update queued bookkeeping.
        mgr.build.queued_set.remove(&k);

        // If it's outside keep, just cancel/remove the queued entry.
        if !in_keep(mgr, center, k) {
            cancel_token(mgr, k).store(true, AtomicOrdering::Relaxed);
            mgr.build.chunks.remove(&k);
            mgr.build.cancels.remove(&k);
            continue;
        }

        // If cached, drop queued tracking and try promote.
        if mgr.cache.get(&k).is_some() {
            mgr.build.chunks.remove(&k);
            mgr.build.cancels.remove(&k);
            let _ = try_promote_from_cache(mgr, center, k);
            continue;
        }

        // Dispatch the build job.
        mgr.build.chunks.insert(k, ChunkState::Building);

        let cancel = cancel_token(mgr, k);
        cancel.store(false, AtomicOrdering::Relaxed);

        let edits = mgr.edits.snapshot(k);
        match mgr.build.tx_job.try_send(BuildJob { key: k, cancel: cancel.clone(), edits, enqueued_at: Instant::now(), }) {
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

        let canceled = done.canceled || done.cancel.load(AtomicOrdering::Relaxed);
        let nodes_n = done.tim.nodes.max(done.nodes.len() as u32);

        // Record timing window FIRST (even if we'll discard the result due to keep/cancel)
        mgr.timing.record_build(
            done.queue_ms,
            done.build_ms,
            nodes_n,
            canceled,
            &done.tim,
        );

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
    let Some(e) = mgr.cache.get(&key) else { return false; };

    let nodes = e.nodes.clone();
    let macro_words = e.macro_words.clone();
    let ropes = e.ropes.clone();
    let colinfo_words = e.colinfo_words.clone();

    mgr.cache.touch(key);

    let ok = slots::try_make_uploading(mgr, center, key, nodes, macro_words, ropes, colinfo_words);
    if ok {
        // purge stale queued bookkeeping
        if mgr.build.queued_set.remove(&key) {
            if let Some(pos) = mgr.build.build_queue.iter().position(|x| *x == key) {
                mgr.build.build_queue.remove(pos);
            }
        }
    }
    ok
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

    // If we already have a slot, replace contents in-place (enqueue rewrite upload).
    match mgr.build.chunks.get(&key) {
        Some(ChunkState::Resident(_)) | Some(ChunkState::Uploading(_)) => {
            let ok = super::slots::replace_chunk_contents(
                mgr,
                center,
                key,
                nodes_arc.clone(),
                macro_arc.clone(),
                ropes_arc.clone(),
                colinfo_arc.clone(),
            );

            if ok {
                // cache the new version once
                mgr.cache.put(key, nodes_arc, macro_arc, ropes_arc, colinfo_arc);
            }
            return;
        }
        _ => {}
    }

    // Cache it even if we’re going to defer upload (so promote-from-cache works).
    mgr.cache.put(
        key,
        nodes_arc.clone(),
        macro_arc.clone(),
        ropes_arc.clone(),
        colinfo_arc.clone(),
    );

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

#[inline]
fn heap_push(mgr: &mut ChunkManager, center: ChunkKey, k: ChunkKey, boost: f32) {
    let cam_fwd = mgr.build.last_cam_fwd;
    let mut s = priority_score(k, center, cam_fwd) - boost;

    if super::keep::in_active_xz(center, k) {
        s -= 10_000.0;
    }


    mgr.build.heap_tie = mgr.build.heap_tie.wrapping_add(1).max(1);

    mgr.build.build_heap.push(HeapItem {
        prio: make_prio(s),
        tie: mgr.build.heap_tie,
        key: k,
    });
}


pub fn rebuild_build_heap(mgr: &mut ChunkManager, center: ChunkKey, cam_fwd: Vec3) {
    // Gather keys
    let mut keys: Vec<ChunkKey> =
        Vec::with_capacity(mgr.build.build_heap.len() + mgr.build.build_queue.len());

    // Consume heap safely (avoid leaving stale items around)
    let old_heap = std::mem::take(&mut mgr.build.build_heap);
    keys.extend(old_heap.into_iter().map(|it| it.key));
    keys.extend(mgr.build.build_queue.drain(..));

    // We will rebuild the heap from scratch
    mgr.build.build_heap = BinaryHeap::new();

    let origin = mgr.grid.grid_origin_chunk;
    let nx = (2 * config::KEEP_RADIUS + 1) as i32;

    let mut kept: Vec<ChunkKey> = Vec::new();

    for k in keys {
        // If it's no longer queued, it must NOT be in queue bookkeeping,
        // but we must NOT remove its state from build.chunks.
        if !matches!(mgr.build.chunks.get(&k), Some(ChunkState::Queued)) {
            mgr.build.queued_set.remove(&k);
            continue;
        }

        // Your existing keep checks
        let keep_it =
            {
                let dx = k.x - center.x;
                let dz = k.z - center.z;
                let r = config::KEEP_RADIUS;
                dx*dx + dz*dz <= r*r
            } &&
            {
                let ix = k.x - origin[0];
                let iz = k.z - origin[2];
                !(ix < 0 || iz < 0 || ix >= nx || iz >= nx)
            } &&
            {
                let idx = ((k.z - origin[2]) * nx + (k.x - origin[0])) as usize;
                // if idx is in-bounds, we have a cached value
                idx < mgr.ground.col_ground_y_vox.len()
            } &&
            {
                let idx = ((k.z - origin[2]) * nx + (k.x - origin[0])) as usize;
                let ground_y_vox = mgr.ground.col_ground_y_vox[idx];
                let ground_cy = ground_y_vox.div_euclid(config::CHUNK_SIZE as i32);
                let dy = k.y - ground_cy;
                dy >= y_band_min() && dy <= y_band_max()
            };


        if keep_it {
            kept.push(k);
        } else {
            // Only safe to cancel/remove because it's STILL queued (no slots/arena allocated)
            if let Some(c) = mgr.build.cancels.get(&k) {
                c.store(true, AtomicOrdering::Relaxed);
            }
            mgr.build.chunks.remove(&k);
            mgr.build.cancels.remove(&k);
            mgr.build.queued_set.remove(&k);
        }
    }

    // Rebuild queued_set to match kept
    mgr.build.queued_set.clear();
    mgr.build.queued_set.extend(kept.iter().copied());

    // Score + heapify
    let mut items: Vec<HeapItem> = Vec::with_capacity(kept.len());
    for (i, k) in kept.into_iter().enumerate() {
        let mut s = priority_score(k, center, cam_fwd);

        if super::keep::in_active_xz(center, k) {
            s -= 10_000.0;
        }

        items.push(HeapItem { prio: make_prio(s), tie: i as u32, key: k });
    }

    mgr.build.build_heap = BinaryHeap::from(items);
}

pub fn request_rebuild(mgr: &mut ChunkManager, key: ChunkKey) {
    // Don’t enqueue rebuilds for chunks we don’t track.
    let Some(st) = mgr.build.chunks.get(&key) else { return; };

    // Only meaningful if it already has a slot (Uploading or Resident).
    match st {
        ChunkState::Resident(_) | ChunkState::Uploading(_) => {}
        _ => return,
    }

    // Dedup
    if mgr.build.rebuild_set.insert(key) {
        mgr.build.rebuild_queue.push_back(key);
    }
}

pub fn request_edit_refresh(mgr: &mut ChunkManager, key: ChunkKey) {
    // (A) Always invalidate cached CPU chunk contents when we know edits changed it.
    // Otherwise, future promote-from-cache can resurrect stale pre-edit data.
    mgr.cache.remove(&key);

    // We might be called before the first update() publishes a center.
    // Use last_center if available; otherwise fall back to the edited chunk itself.
    let center = mgr.build.last_center.unwrap_or(key);

    match mgr.build.chunks.get(&key) {
        // already has a slot -> rebuild in place
        Some(ChunkState::Resident(_)) | Some(ChunkState::Uploading(_)) => {
            request_rebuild(mgr, key);
        }

        // already queued -> boost priority (front / heap bump)
        Some(ChunkState::Queued) => {
            queue_build_front(mgr, center, key);
        }

        // already building -> do nothing; completion will apply edits
        Some(ChunkState::Building) => {}

        // not tracked -> enqueue a build (safe; does NOT touch slots)
        None => {
            let c = cancel_token(mgr, key);
            c.store(false, AtomicOrdering::Relaxed);
            queue_build_front(mgr, center, key);
        }
    }
}

