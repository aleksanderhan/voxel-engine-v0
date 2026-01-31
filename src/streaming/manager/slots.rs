use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use crate::{
    config,
    render::gpu_types::{ChunkMetaGpu, NodeGpu, NodeRopesGpu},
    streaming::types::*,
};

use super::{ground, keep};
use super::ChunkManager;

#[inline(always)]
pub fn in_priority_box(mgr: &ChunkManager, center: ChunkKey, k: ChunkKey) -> bool {
    if !keep::in_priority_xz(center, k) {
        return false;
    }
    let Some(ground_cy) = ground::ground_cy_for_column(mgr, k.x, k.z) else {
        return false;
    };
    let dy = k.y - ground_cy;
    dy >= GRID_Y_MIN_DY && dy <= (GRID_Y_MIN_DY + GRID_Y_COUNT as i32 - 1)
}

#[inline(always)]
fn in_keep(mgr: &ChunkManager, center: ChunkKey, k: ChunkKey) -> bool {
    let dx = k.x - center.x;
    let dz = k.z - center.z;

    if dx < -config::KEEP_RADIUS || dx > config::KEEP_RADIUS {
        return false;
    }
    if dz < -config::KEEP_RADIUS || dz > config::KEEP_RADIUS {
        return false;
    }

    let Some(ground_cy) = ground::ground_cy_for_column(mgr, k.x, k.z) else {
        return false;
    };

    let dy = k.y - ground_cy;
    dy >= y_band_min() && dy <= y_band_max()
}

#[inline(always)]
pub fn is_gpu_ready(mgr: &ChunkManager, k: ChunkKey) -> bool {
    match mgr.build.chunks.get(&k) {
        Some(ChunkState::Resident(_)) => true,
        Some(ChunkState::Uploading(up)) => up.uploaded,
        _ => false,
    }
}

pub fn priority_box_ready(mgr: &ChunkManager, center: ChunkKey) -> bool {
    for &(dx, dy, dz) in &mgr.offsets.priority_offsets {
        let x = center.x + dx;
        let z = center.z + dz;

        let Some(ground_cy) = ground::ground_cy_for_column(mgr, x, z) else {
            continue;
        };
        let k = ChunkKey { x, y: ground_cy + dy, z };

        if !is_gpu_ready(mgr, k) {
            return false;
        }
    }
    true
}

#[inline]
fn cancel_token(mgr: &mut ChunkManager, key: ChunkKey) -> Arc<AtomicBool> {
    mgr.build
        .cancels
        .entry(key)
        .or_insert_with(|| Arc::new(AtomicBool::new(false)))
        .clone()
}

pub fn commit_uploads_applied(mgr: &mut ChunkManager, applied: &[ChunkUpload]) -> bool {
    let mut any_promoted = false;
    let mut any_uploaded_flip = false;

    // Mark uploads as applied; detect first-time flip.
    for u in applied {
        if !u.completes_residency {
            continue;
        }
        if let Some(ChunkState::Uploading(up)) = mgr.build.chunks.get_mut(&u.key) {
            if !up.uploaded {
                up.uploaded = true;
                any_uploaded_flip = true;
            }
        }
    }

    let center_opt = mgr.build.last_center;
    let priority_gate = center_opt
        .map(|c| !priority_box_ready(mgr, c))
        .unwrap_or(false);

    loop {
        if mgr.slots.resident_slots >= mgr.slots.slot_to_key.len() {
            break;
        }

        // If the next slot is ready, promote it directly.
        let next_key = mgr.slots.slot_to_key[mgr.slots.resident_slots];
        let next_ready = matches!(
            mgr.build.chunks.get(&next_key),
            Some(ChunkState::Uploading(Uploading { uploaded: true, .. }))
        );

        if !next_ready {
            // Find the best READY uploading chunk in [resident_slots..].
            let mut best: Option<(f32, usize)> = None;

            for s in mgr.slots.resident_slots..mgr.slots.slot_to_key.len() {
                let k = mgr.slots.slot_to_key[s];

                if priority_gate {
                    let c = center_opt.unwrap();
                    if !in_priority_box(mgr, c, k) {
                        continue;
                    }
                }

                let Some(ChunkState::Uploading(up)) = mgr.build.chunks.get(&k) else { continue; };
                if !up.uploaded {
                    continue;
                }

                let mut score = if let Some(c) = center_opt {
                    super::uploads::upload_dist_score(k, c)
                } else {
                    s as f32
                };

                if let Some(c) = center_opt {
                    if keep::in_active_xz(c, k) {
                        score -= 10_000.0;
                    }
                    if in_priority_box(mgr, c, k) {
                        score -= 20_000.0;
                    }
                }

                if best.map_or(true, |(bs, _)| score < bs) {
                    best = Some((score, s));
                }
            }

            let Some((_, best_slot)) = best else {
                break;
            };

            if best_slot != mgr.slots.resident_slots {
                swap_slots(mgr, best_slot, mgr.slots.resident_slots);

                // Slot changed => rewrite that slot ASAP.
                enqueue_slot_rewrite(mgr, mgr.slots.resident_slots);

                // If swapped-out chunk was uploaded too, rewrite it as well.
                let swapped_out_key = mgr.slots.slot_to_key[best_slot];
                let swapped_out_uploaded = matches!(
                    mgr.build.chunks.get(&swapped_out_key),
                    Some(ChunkState::Uploading(Uploading { uploaded: true, .. }))
                );
                if swapped_out_uploaded {
                    enqueue_slot_rewrite(mgr, best_slot);
                }
            }
        }

        // Now the frontier slot should be ready; promote it.
        let slot = mgr.slots.resident_slots;
        let key = mgr.slots.slot_to_key[slot];

        if priority_gate {
            let c = center_opt.unwrap();
            if !in_priority_box(mgr, c, key) {
                break;
            }
        }

        let ready = matches!(
            mgr.build.chunks.get(&key),
            Some(ChunkState::Uploading(Uploading { uploaded: true, .. }))
        );
        if !ready {
            break;
        }

        if let Some(st) = mgr.build.chunks.get_mut(&key) {
            if let ChunkState::Uploading(up) = st {
                let slot_u32 = slot as u32;
                let node_base = up.node_base;
                let node_count = up.node_count;
                *st = ChunkState::Resident(Resident { slot: slot_u32, node_base, node_count });
                mgr.slots.resident_slots += 1;
                any_promoted = true;
                continue;
            }
        }

        break;
    }

    if any_promoted || any_uploaded_flip {
        mgr.grid.grid_dirty = true;
        if let Some(center) = mgr.build.last_center {
            super::grid::rebuild_grid(mgr, center);
            mgr.grid.grid_dirty = false;
        }
        return true;
    }

    false
}

pub fn try_make_uploading(
    mgr: &mut ChunkManager,
    center: ChunkKey,
    key: ChunkKey,
    nodes: Arc<[NodeGpu]>,
    macro_words: Arc<[u32]>,
    ropes: Arc<[NodeRopesGpu]>,
    colinfo_words: Arc<[u32]>,
) -> bool {
    if matches!(mgr.build.chunks.get(&key), Some(ChunkState::Resident(_))) {
        return true;
    }

    let need = nodes.len() as u32;
    if need == 0 {
        mgr.build.chunks.remove(&key);
        return false;
    }
    if macro_words.len() != MACRO_WORDS_PER_CHUNK_USIZE {
        mgr.build.chunks.remove(&key);
        return false;
    }
    if ropes.len() != nodes.len() {
        mgr.build.chunks.remove(&key);
        return false;
    }
    if colinfo_words.len() != COLINFO_WORDS_PER_CHUNK_USIZE {
        mgr.build.chunks.remove(&key);
        return false;
    }

    let mut node_base = mgr.arena.alloc(need);
    if node_base.is_none() {
        for _ in 0..EVICT_ATTEMPTS {
            if !evict_one_farthest(mgr, center, key) {
                break;
            }
            node_base = mgr.arena.alloc(need);
            if node_base.is_some() {
                break;
            }
        }
    }

    let Some(node_base) = node_base else {
        // Defer rather than drop on the floor.
        mgr.build.chunks.insert(key, ChunkState::Queued);
        if mgr.build.queued_set.insert(key) {
            mgr.build.build_queue.push_back(key);
        }
        return false;
    };

    let slot = mgr.slots.slot_to_key.len() as u32;
    mgr.slots.slot_to_key.push(key);

    mgr.slots.slot_macro.push(macro_words.clone());
    mgr.slots.slot_colinfo.push(colinfo_words.clone());

    let macro_base = slot * MACRO_WORDS_PER_CHUNK;
    let colinfo_base = slot * COLINFO_WORDS_PER_CHUNK;

    let origin_vox = [
        key.x * config::CHUNK_SIZE as i32,
        key.y * config::CHUNK_SIZE as i32,
        key.z * config::CHUNK_SIZE as i32,
    ];

    let meta = ChunkMetaGpu {
        origin: [origin_vox[0], origin_vox[1], origin_vox[2], 0],
        node_base,
        node_count: need,
        macro_base,
        colinfo_base,
    };

    mgr.slots.chunk_meta.push(meta);

    mgr.build.chunks.insert(
        key,
        ChunkState::Uploading(Uploading { slot, node_base, node_count: need, uploaded: false }),
    );

    super::uploads::enqueue(mgr, ChunkUpload {
        key,
        slot,
        meta,
        node_base,
        nodes,
        macro_words,
        ropes,
        colinfo_words,
        completes_residency: true,
    });

    true
}

pub fn unload_chunk(mgr: &mut ChunkManager, center: ChunkKey, key: ChunkKey) {
    let Some(state) = mgr.build.chunks.remove(&key) else {
        return;
    };

    match state {
        ChunkState::Resident(res) => {
            mgr.arena.free(res.node_base, res.node_count);
            let dead = res.slot as usize;

            let last_res = mgr.slots.resident_slots.saturating_sub(1);
            debug_assert!(dead < mgr.slots.resident_slots, "resident slot out of prefix");

            if dead != last_res {
                swap_slots(mgr, dead, last_res);
                enqueue_slot_rewrite(mgr, dead);
            }

            mgr.slots.resident_slots = last_res;

            let remove_idx = last_res;
            let last_slot = mgr.slots.slot_to_key.len().saturating_sub(1);

            if remove_idx != last_slot {
                swap_slots(mgr, remove_idx, last_slot);
                enqueue_slot_rewrite(mgr, remove_idx);
            }

            mgr.slots.slot_to_key.pop();
            mgr.slots.chunk_meta.pop();
            mgr.slots.slot_macro.pop();
            mgr.slots.slot_colinfo.pop();

            mgr.grid.grid_dirty = true;
        }

        ChunkState::Uploading(up) => {
            mgr.arena.free(up.node_base, up.node_count);
            let dead = up.slot as usize;

            debug_assert!(
                dead >= mgr.slots.resident_slots,
                "uploading slot inside resident prefix"
            );

            let last_slot = mgr.slots.slot_to_key.len().saturating_sub(1);
            if dead != last_slot {
                swap_slots(mgr, dead, last_slot);
                enqueue_slot_rewrite(mgr, dead);
            }

            mgr.slots.slot_to_key.pop();
            mgr.slots.chunk_meta.pop();
            mgr.slots.slot_macro.pop();
            mgr.slots.slot_colinfo.pop();
        }

        ChunkState::Queued | ChunkState::Building => {
            cancel_token(mgr, key).store(true, Ordering::Relaxed);
            mgr.build.queued_set.remove(&key);
            mgr.grid.grid_dirty = true;
        }
    }

    mgr.build.cancels.remove(&key);

    // If we removed something that was outside keep anyway, fine.
    // If you want to be aggressive about cleaning cache too:
    // mgr.cache.remove(&key);
    let _ = center; // keep signature aligned with callers; can remove if unused.
}

fn enqueue_slot_rewrite(mgr: &mut ChunkManager, slot: usize) {
    let key = mgr.slots.slot_to_key[slot];
    let slot_u32 = slot as u32;

    super::uploads::enqueue(mgr, ChunkUpload {
        key,
        slot: slot_u32,
        meta: mgr.slots.chunk_meta[slot],
        node_base: 0,
        nodes: Arc::<[NodeGpu]>::from(Vec::<NodeGpu>::new()),
        macro_words: mgr.slots.slot_macro[slot].clone(),
        ropes: Arc::<[NodeRopesGpu]>::from(Vec::<NodeRopesGpu>::new()),
        colinfo_words: mgr.slots.slot_colinfo[slot].clone(),
        completes_residency: false,
    });
}

fn swap_slots(mgr: &mut ChunkManager, a: usize, b: usize) {
    if a == b {
        return;
    }

    let ka = mgr.slots.slot_to_key[a];
    let kb = mgr.slots.slot_to_key[b];

    mgr.slots.slot_to_key.swap(a, b);
    mgr.slots.chunk_meta.swap(a, b);
    mgr.slots.slot_macro.swap(a, b);
    mgr.slots.slot_colinfo.swap(a, b);

    mgr.slots.chunk_meta[a].macro_base = (a as u32) * MACRO_WORDS_PER_CHUNK;
    mgr.slots.chunk_meta[a].colinfo_base = (a as u32) * COLINFO_WORDS_PER_CHUNK;
    mgr.slots.chunk_meta[b].macro_base = (b as u32) * MACRO_WORDS_PER_CHUNK;
    mgr.slots.chunk_meta[b].colinfo_base = (b as u32) * COLINFO_WORDS_PER_CHUNK;

    if let Some(st) = mgr.build.chunks.get_mut(&ka) {
        match st {
            ChunkState::Resident(r) => r.slot = b as u32,
            ChunkState::Uploading(u) => u.slot = b as u32,
            _ => {}
        }
    }
    if let Some(st) = mgr.build.chunks.get_mut(&kb) {
        match st {
            ChunkState::Resident(r) => r.slot = a as u32,
            ChunkState::Uploading(u) => u.slot = a as u32,
            _ => {}
        }
    }
}

fn evict_one_farthest(mgr: &mut ChunkManager, center: ChunkKey, protect: ChunkKey) -> bool {
    if mgr.slots.slot_to_key.is_empty() {
        return false;
    }

    // Pass 1: evict farthest chunk that is OUTSIDE active xz.
    let mut best_outside: Option<(f32, ChunkKey)> = None;

    for &k in &mgr.slots.slot_to_key {
        if k == protect {
            continue;
        }
        if in_priority_box(mgr, center, k) {
            continue;
        }
        if keep::in_active_xz(center, k) {
            continue;
        }

        let dx = (k.x - center.x) as f32;
        let dz = (k.z - center.z) as f32;
        let dy = (k.y - center.y) as f32;
        let d = dx * dx + dz * dz + 4.0 * dy * dy;

        if best_outside.map_or(true, |(bd, _)| d > bd) {
            best_outside = Some((d, k));
        }
    }

    if let Some((_, k)) = best_outside {
        unload_chunk(mgr, center, k);
        return true;
    }

    // Pass 2: if everything is inside ACTIVE, evict farthest overall.
    let mut best_any: Option<(f32, ChunkKey)> = None;

    for &k in &mgr.slots.slot_to_key {
        if k == protect {
            continue;
        }
        if in_priority_box(mgr, center, k) {
            continue;
        }
        if keep::in_active_xz(center, k) { continue; }

        let dx = (k.x - center.x) as f32;
        let dz = (k.z - center.z) as f32;
        let dy = (k.y - center.y) as f32;
        let d = dx * dx + dz * dz + 4.0 * dy * dy;

        if best_any.map_or(true, |(bd, _)| d > bd) {
            best_any = Some((d, k));
        }
    }

    if let Some((_, k)) = best_any {
        unload_chunk(mgr, center, k);
        return true;
    }

    false
}

