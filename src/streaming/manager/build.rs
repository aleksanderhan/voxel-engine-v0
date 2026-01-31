use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use crossbeam_channel::TrySendError;
use glam::Vec3;

use crate::streaming::priority::sort_queue_near_first;
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
    let offsets = mgr.offsets.priority_offsets.clone();

    for (dx, dy, dz) in offsets {
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
                c.store(false, Ordering::Relaxed);
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
                c.store(false, Ordering::Relaxed);

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

    // gather keys first (avoid borrow issues)
    let keys: Vec<ChunkKey> = mgr.build.chunks.keys().copied().collect();

    for k in keys {
        if slots::in_priority_box(mgr, center, k) {
            continue;
        }
        // Pin active area: do not unload these (prevents pop-thrash).
        if keep::in_active_xz(center, k) {
            continue;
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

/// This is safe to run every frame; it just keeps the queue sane + near-sorted.
/// (Your old “center changed” special-case becomes “always maintain invariants”.)
pub fn on_center_change_resort(mgr: &mut ChunkManager, center: ChunkKey, cam_fwd: Vec3) {
    // Retain only queued + still in keep (safe, no panics).
    let origin = mgr.grid.grid_origin_chunk;
    let nx = (2 * config::KEEP_RADIUS + 1) as i32;

    mgr.build.build_queue.retain(|k| {
        if !matches!(mgr.build.chunks.get(k), Some(ChunkState::Queued)) {
            return false;
        }

        let dx = k.x - center.x;
        let dz = k.z - center.z;
        if dx < -config::KEEP_RADIUS || dx > config::KEEP_RADIUS { return false; }
        if dz < -config::KEEP_RADIUS || dz > config::KEEP_RADIUS { return false; }

        let ix = k.x - origin[0];
        let iz = k.z - origin[2];
        if ix < 0 || iz < 0 || ix >= nx || iz >= nx { return false; }

        let idx = (iz * nx + ix) as usize;
        let Some(ground_cy) = mgr.ground.col_ground_cy.get(idx).copied() else { return false; };

        let dy = k.y - ground_cy;
        dy >= y_band_min() && dy <= y_band_max()
    });

    mgr.build.queued_set.clear();
    mgr.build.queued_set.extend(mgr.build.build_queue.iter().copied());

    // Priority box always first, then sort the rest near-first.
    let mut pri = std::collections::VecDeque::new();
    let mut rest = std::collections::VecDeque::new();

    while let Some(k) = mgr.build.build_queue.pop_front() {
        if slots::in_priority_box(mgr, center, k) {
            pri.push_back(k);
        } else {
            rest.push_back(k);
        }
    }

    sort_queue_near_first(&mut rest, center, cam_fwd);

    mgr.build.build_queue = pri;
    mgr.build.build_queue.extend(rest);
}

pub fn dispatch_builds(mgr: &mut ChunkManager, center: ChunkKey) {
    let mut attempts = 0usize;
    let max_attempts = mgr.build.build_queue.len().max(1);

    while mgr.build.in_flight < config::MAX_IN_FLIGHT && attempts < max_attempts {
        attempts += 1;

        let Some(k) = mgr.build.build_queue.pop_front() else { break; };
        mgr.build.queued_set.remove(&k);

        if !in_keep(mgr, center, k) {
            cancel_token(mgr, k).store(true, Ordering::Relaxed);
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
            cancel.store(false, Ordering::Relaxed);

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

        if done.canceled || done.cancel.load(Ordering::Relaxed) {
            mgr.build.chunks.remove(&done.key);
            mgr.build.cancels.remove(&done.key);
            continue;
        }

        if !in_keep(mgr, center, done.key) {
            cancel_token(mgr, done.key).store(true, Ordering::Relaxed);
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
        if c.load(Ordering::Relaxed) {
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
