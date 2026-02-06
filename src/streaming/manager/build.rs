
use std::sync::{
    atomic::{AtomicBool, Ordering as AtomicOrdering},
    Arc,
};
use std::time::Instant;


use crossbeam_channel::TrySendError;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use glam::{Vec2, Vec3};

use crate::streaming::types::*;
use crate::{render::gpu_types::{NodeGpu, NodeRopesGpu}};
use crate::app::config;

use super::{ground, keep, slots};
use super::ChunkManager;
use crate::streaming::priority::{priority_score_streaming};




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
    
    match mgr.build.chunks.get(&k) {
        Some(ChunkState::Queued) | Some(ChunkState::Building) | Some(ChunkState::Uploading(_)) | Some(ChunkState::Resident(_)) => {}
        None => {
            mgr.build.chunks.insert(k, ChunkState::Queued);
            mgr.build.queued_set.insert(k);
        }
    }

    
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
    
    
    
    
    
    
    let reinject_cap = 64usize.min(mgr.build.build_queue.len());
    for _ in 0..reinject_cap {
        let Some(k) = mgr.build.build_queue.pop_front() else { break; };

        
        if !matches!(mgr.build.chunks.get(&k), Some(ChunkState::Queued)) {
            continue;
        }

        
        heap_push(mgr, center, k, 0.0);
    }

    
    let reserve_for_rebuilds = 2usize;
    let rebuild_pending = !mgr.build.rebuild_queue.is_empty();
    let max_normal_in_flight = if rebuild_pending {
        config::MAX_IN_FLIGHT.saturating_sub(reserve_for_rebuilds).max(1)
    } else {
        config::MAX_IN_FLIGHT
    };

    
    
    
    while mgr.build.in_flight < config::MAX_IN_FLIGHT {
        let Some(k) = mgr.build.rebuild_queue.pop_front() else { break; };
        mgr.build.rebuild_set.remove(&k);

        let Some(st) = mgr.build.chunks.get(&k) else { continue; };
        if !matches!(st, ChunkState::Resident(_) | ChunkState::Uploading(_)) {
            continue;
        }

        if !in_keep(mgr, center, k) {
            continue;
        }

        
        let cancel = Arc::new(AtomicBool::new(false));
        mgr.build.cancels.insert(k, cancel.clone());

        let edits = mgr.edits.snapshot(k);

        match mgr.build.tx_job.try_send(BuildJob { key: k, cancel, edits, enqueued_at: Instant::now() }) {
            Ok(()) => mgr.build.in_flight += 1,
            Err(TrySendError::Full(job)) | Err(TrySendError::Disconnected(job)) => {
                mgr.build.rebuild_set.insert(job.key);
                mgr.build.rebuild_queue.push_front(job.key);
                break;
            }
        }
    }

    
    
    
    let mut attempts = 0usize;
    let max_attempts = mgr.build.build_heap.len().max(1);

    while mgr.build.in_flight < max_normal_in_flight && attempts < max_attempts {
        attempts += 1;

        let Some(item) = mgr.build.build_heap.pop() else { break; };
        let k = item.key;

        let Some(st) = mgr.build.chunks.get(&k) else {
            continue; 
        };

        if !matches!(st, ChunkState::Queued) {
            continue; 
        }

        
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

        
        mgr.build.chunks.insert(k, ChunkState::Building);

        let cancel = cancel_token(mgr, k);
        cancel.store(false, AtomicOrdering::Relaxed);

        let edits = mgr.edits.snapshot(k);
        match mgr.build.tx_job.try_send(BuildJob { key: k, cancel: cancel.clone(), edits, enqueued_at: Instant::now() }) {
            Ok(()) => mgr.build.in_flight += 1,
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                
                mgr.build.chunks.insert(k, ChunkState::Queued);
                if mgr.build.queued_set.insert(k) {
                    mgr.build.build_queue.push_front(k);
                }
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

        
        let Some(cur_cancel) = mgr.build.cancels.get(&done.key) else { continue; };
        if !Arc::ptr_eq(cur_cancel, &done.cancel) { continue; }

        let canceled = done.canceled || done.cancel.load(AtomicOrdering::Relaxed);
        let nodes_n = done.tim.nodes.max(done.nodes.len() as u32);

        
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
            mgr.build.cancels.remove(&key);
            return;
        }
    }

    
    if nodes.is_empty()
        || macro_words.len() != MACRO_WORDS_PER_CHUNK_USIZE
        || ropes.len() != nodes.len()
        || colinfo_words.len() != COLINFO_WORDS_PER_CHUNK_USIZE
    {
        mgr.build.chunks.remove(&key);
        mgr.build.cancels.remove(&key);
        return;
    }

    let nodes_arc: Arc<[NodeGpu]> = nodes.into();
    let macro_arc: Arc<[u32]> = macro_words.into();
    let ropes_arc: Arc<[NodeRopesGpu]> = ropes.into();
    let colinfo_arc: Arc<[u32]> = colinfo_words.into();

    
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
                mgr.cache.put(key, nodes_arc, macro_arc, ropes_arc, colinfo_arc);
            }
            return;
        }
        _ => {}
    }

    
    mgr.cache.put(
        key,
        nodes_arc.clone(),
        macro_arc.clone(),
        ropes_arc.clone(),
        colinfo_arc.clone(),
    );

    let ok = slots::try_make_uploading(mgr, center, key, nodes_arc, macro_arc, ropes_arc, colinfo_arc);

    if !ok {
        
        
        
        
        
        
        if mgr.build.chunks.get(&key).is_none() {
            mgr.build.cancels.remove(&key);
        }
    }
}



#[derive(Clone, Copy)]
pub struct HeapItem {
    
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
    
    
    let b = x.to_bits();
    if (b & 0x8000_0000) != 0 { !b } else { b ^ 0x8000_0000 }
}

#[inline]
fn make_prio(score: f32) -> u32 {
    
    let ord = if score.is_nan() { u32::MAX } else { float_to_ordered_u32(score) };
    u32::MAX - ord
}

#[inline]
fn heap_push(mgr: &mut ChunkManager, center: ChunkKey, k: ChunkKey, boost: f32) {
    let cam_fwd = mgr.build.last_cam_fwd;
    let mut s = priority_score_streaming(mgr, k, center, cam_fwd) - boost;

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
    
    let mut keys: Vec<ChunkKey> =
        Vec::with_capacity(mgr.build.build_heap.len() + mgr.build.build_queue.len());

    
    let old_heap = std::mem::take(&mut mgr.build.build_heap);
    keys.extend(old_heap.into_iter().map(|it| it.key));
    keys.extend(mgr.build.build_queue.drain(..));

    
    mgr.build.build_heap = BinaryHeap::new();

    let origin = mgr.grid.grid_origin_chunk;
    let nx = (2 * config::KEEP_RADIUS + 1) as i32;

    let mut kept: Vec<ChunkKey> = Vec::new();

    for k in keys {
        
        
        if !matches!(mgr.build.chunks.get(&k), Some(ChunkState::Queued)) {
            mgr.build.queued_set.remove(&k);
            continue;
        }

        
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
            
            if let Some(c) = mgr.build.cancels.get(&k) {
                c.store(true, AtomicOrdering::Relaxed);
            }
            mgr.build.chunks.remove(&k);
            mgr.build.cancels.remove(&k);
            mgr.build.queued_set.remove(&k);
        }
    }

    
    mgr.build.queued_set.clear();
    mgr.build.queued_set.extend(kept.iter().copied());

    
    let mut items: Vec<HeapItem> = Vec::with_capacity(kept.len());
    for (i, k) in kept.into_iter().enumerate() {
        let mut s = priority_score_streaming(mgr, k, center, cam_fwd);

        if super::keep::in_active_xz(center, k) {
            s -= 10_000.0;
        }

        items.push(HeapItem { prio: make_prio(s), tie: i as u32, key: k });
    }

    mgr.build.build_heap = BinaryHeap::from(items);
}

pub fn request_rebuild(mgr: &mut ChunkManager, key: ChunkKey) {
    
    let Some(st) = mgr.build.chunks.get(&key) else { return; };

    
    match st {
        ChunkState::Resident(_) | ChunkState::Uploading(_) => {}
        _ => return,
    }

    
    if mgr.build.rebuild_set.insert(key) {
        mgr.build.rebuild_queue.push_back(key);
    }
}

pub fn request_edit_refresh(mgr: &mut ChunkManager, key: ChunkKey) {
    
    
    mgr.cache.remove(&key);

    
    
    let center = mgr.build.last_center.unwrap_or(key);

    match mgr.build.chunks.get(&key) {
        
        Some(ChunkState::Resident(_)) | Some(ChunkState::Uploading(_)) => {
            request_rebuild(mgr, key);
        }

        
        Some(ChunkState::Queued) => {
            queue_build_front(mgr, center, key);
        }

        
        Some(ChunkState::Building) => {}

        
        None => {
            let c = cancel_token(mgr, key);
            c.store(false, AtomicOrdering::Relaxed);
            queue_build_front(mgr, center, key);
        }
    }
}

