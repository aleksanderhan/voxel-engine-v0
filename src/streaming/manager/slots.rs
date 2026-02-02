use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use crate::app::config;
use crate::{
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
    let mut any_rewrite_cleared = false;

    // 1) Mark uploads as applied.
    for u in applied {
        match u.kind {
            UploadKind::PromoteToResident => {
                if let Some(ChunkState::Uploading(up)) = mgr.build.chunks.get_mut(&u.key) {
                    if !up.uploaded {
                        up.uploaded = true;
                        any_uploaded_flip = true;
                    }
                }
            }
            UploadKind::RewriteResident => {
                if let Some(ChunkState::Resident(r)) = mgr.build.chunks.get_mut(&u.key) {
                    if r.rewrite_in_flight {
                        r.rewrite_in_flight = false;
                        any_rewrite_cleared = true;
                    }
                }
            }
        }
    }

    let center_opt = mgr.build.last_center;
    let priority_gate = center_opt
        .map(|c| !priority_box_ready(mgr, c))
        .unwrap_or(false);

    // 2) Promote ready Uploading chunks into the resident prefix.
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

                *st = ChunkState::Resident(Resident {
                    slot: slot_u32,
                    node_base,
                    node_count,
                    rewrite_in_flight: false, // IMPORTANT
                });

                mgr.slots.resident_slots += 1;
                any_promoted = true;
                continue;
            }
        }

        break;
    }

    repair_slot_prefix(mgr);

    if any_promoted || any_uploaded_flip || any_rewrite_cleared {
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
    match mgr.build.chunks.get(&key) {
        Some(ChunkState::Resident(_)) => return true,
        Some(ChunkState::Uploading(_)) => {
            // Already has a slot + arena allocation. Don’t allocate a new slot.
            return true;
        }
        _ => {}
    }
    
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
        kind: UploadKind::PromoteToResident,
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
    // Copy the info we need WITHOUT removing the map entry yet.
    // This avoids swapping slots while `slot_to_key` still references a key missing from `build.chunks`.
    enum Kind {
        Resident { dead: usize, node_base: u32, node_count: u32 },
        Uploading { dead: usize, node_base: u32, node_count: u32 },
        QueuedOrBuilding,
    }

    let kind = match mgr.build.chunks.get(&key) {
        None => return,
        Some(ChunkState::Resident(r)) => Kind::Resident {
            dead: r.slot as usize,
            node_base: r.node_base,
            node_count: r.node_count,
        },
        Some(ChunkState::Uploading(u)) => Kind::Uploading {
            dead: u.slot as usize,
            node_base: u.node_base,
            node_count: u.node_count,
        },
        Some(ChunkState::Queued) | Some(ChunkState::Building) => Kind::QueuedOrBuilding,
    };

    match kind {
        Kind::Resident { dead, node_base, node_count } => {
            mgr.arena.free(node_base, node_count);

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

        Kind::Uploading { dead, node_base, node_count } => {
            mgr.arena.free(node_base, node_count);

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

            // IMPORTANT: removing a slot can affect the grid even if it was Uploading+uploaded.
            mgr.grid.grid_dirty = true;
        }

        Kind::QueuedOrBuilding => {
            cancel_token(mgr, key).store(true, Ordering::Relaxed);
            mgr.build.queued_set.remove(&key);
            mgr.grid.grid_dirty = true;
        }
    }

    // Now remove bookkeeping. At this point the slot arrays no longer reference `key`.
    mgr.build.chunks.remove(&key);
    mgr.build.cancels.remove(&key);

    repair_slot_prefix(mgr);

    let _ = center;
}


fn enqueue_slot_rewrite(mgr: &mut ChunkManager, slot: usize) {
    let key = mgr.slots.slot_to_key[slot];
    let slot_u32 = slot as u32;

    // If this is a resident chunk, track that a rewrite upload is in flight.
    if let Some(ChunkState::Resident(r)) = mgr.build.chunks.get_mut(&key) {
        r.rewrite_in_flight = true;
    }

    super::uploads::enqueue(mgr, ChunkUpload {
        key,
        slot: slot_u32,
        kind: UploadKind::RewriteResident,
        meta: mgr.slots.chunk_meta[slot],
        node_base: 0,
        nodes: Arc::<[NodeGpu]>::from(Vec::<NodeGpu>::new()),
        macro_words: mgr.slots.slot_macro[slot].clone(),
        ropes: Arc::<[NodeRopesGpu]>::from(Vec::<NodeRopesGpu>::new()),
        colinfo_words: mgr.slots.slot_colinfo[slot].clone(),
        completes_residency: false,
    });
}


fn swap_slots_impl(mgr: &mut ChunkManager, a: usize, b: usize) {
    if a == b { return; }

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

fn swap_slots(mgr: &mut ChunkManager, a: usize, b: usize) {
    #[cfg(debug_assertions)]
    {
        if a != b {
            let ka = mgr.slots.slot_to_key[a];
            let kb = mgr.slots.slot_to_key[b];

            assert!(
                mgr.build.chunks.contains_key(&ka),
                "swap_slots: slot_to_key[a] missing in build.chunks: a={} ka={:?}",
                a, ka
            );
            assert!(
                mgr.build.chunks.contains_key(&kb),
                "swap_slots: slot_to_key[b] missing in build.chunks: b={} kb={:?}",
                b, kb
            );

            let a_is_res = matches!(mgr.build.chunks.get(&ka), Some(ChunkState::Resident(_)));
            let b_is_res = matches!(mgr.build.chunks.get(&kb), Some(ChunkState::Resident(_)));

            if a_is_res != b_is_res {
                let a_in_prefix = a < mgr.slots.resident_slots;
                let b_in_prefix = b < mgr.slots.resident_slots;
                assert!(
                    a_in_prefix == b_in_prefix,
                    "swap_slots crosses prefix boundary: a={} b={} resident_slots={} ka={:?} kb={:?} ka_state={:?} kb_state={:?}",
                    a, b, mgr.slots.resident_slots, ka, kb,
                    mgr.build.chunks.get(&ka),
                    mgr.build.chunks.get(&kb),
                );
            }
        }
    }

    swap_slots_impl(mgr, a, b);
}



fn evict_one_farthest(mgr: &mut ChunkManager, center: ChunkKey, protect: ChunkKey) -> bool {
    if mgr.slots.slot_to_key.is_empty() {
        return false;
    }

    #[cfg(debug_assertions)]
    {
        if let Some(st) = mgr.build.chunks.get(&protect) {
            let ps = match st {
                ChunkState::Resident(r) => r.slot as usize,
                ChunkState::Uploading(u) => u.slot as usize,
                _ => usize::MAX,
            };
            // protect must still map back correctly
            if ps != usize::MAX {
                debug_assert_eq!(mgr.slots.slot_to_key[ps], protect);
            }
        }
    }


    // Pass 1: evict farthest chunk that is OUTSIDE active xz.
    let mut best_outside: Option<(f32, ChunkKey)> = None;

    for &k in &mgr.slots.slot_to_key {
        if mgr.pinned.contains(&k) { continue; }
        if k == protect {
            continue;
        }
        if let Some(ChunkState::Resident(r)) = mgr.build.chunks.get(&k) {
            if r.rewrite_in_flight {
                continue;
            }
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
        if let Some(ChunkState::Resident(r)) = mgr.build.chunks.get(&k) {
            if r.rewrite_in_flight {
                continue;
            }
        }

        if in_priority_box(mgr, center, k) {
            continue;
        }
        // REMOVE this line for pass 2:
        // if keep::in_active_xz(center, k) { continue; }

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

#[cfg(debug_assertions)]
pub fn assert_slot_invariants(mgr: &ChunkManager) {
    use rustc_hash::FxHashSet;

    // slot_to_key must not contain duplicates
    let mut seen: FxHashSet<ChunkKey> = FxHashSet::default();
    for (i, &k) in mgr.slots.slot_to_key.iter().enumerate() {
        assert!(
            seen.insert(k),
            "DUPLICATE key in slot_to_key: i={} k={:?} resident_slots={} total_slots={}",
            i, k, mgr.slots.resident_slots, mgr.slots.slot_to_key.len()
        );
    }

    // Prefix rule: [0 .. resident_slots) must be Resident, rest must be Uploading.
    for (i, &k) in mgr.slots.slot_to_key.iter().enumerate() {
        match mgr.build.chunks.get(&k) {
            Some(ChunkState::Resident(r)) => {
                assert!(i < mgr.slots.resident_slots, "Resident beyond prefix: i={i} resident_slots={} k={:?}", mgr.slots.resident_slots, k);
                assert!(r.slot as usize == i, "Resident slot mismatch: i={i} r.slot={} k={:?}", r.slot, k);
            }
            Some(ChunkState::Uploading(u)) => {
                assert!(i >= mgr.slots.resident_slots, "Uploading inside prefix: i={i} resident_slots={} k={:?}", mgr.slots.resident_slots, k);
                assert!(u.slot as usize == i, "Uploading slot mismatch: i={i} u.slot={} k={:?}", u.slot, k);
            }
            other => {
                panic!("slot_to_key contains key with non-slot state: i={i} k={:?} state={:?}", k, other);
            }
        }
    }

    assert!(
        mgr.slots.resident_slots <= mgr.slots.slot_to_key.len(),
        "resident_slots {} > slot_to_key.len {}",
        mgr.slots.resident_slots,
        mgr.slots.slot_to_key.len()
    );
}


pub fn replace_chunk_contents(
    mgr: &mut ChunkManager,
    center: ChunkKey,
    key: ChunkKey,
    mut nodes: Arc<[NodeGpu]>,
    macro_words: Arc<[u32]>,
    mut ropes: Arc<[NodeRopesGpu]>,
    colinfo_words: Arc<[u32]>,
) -> bool {
    // (node_base, allocated_capacity, is_resident_now)
    let (old_base, old_cap, is_resident_now) = match mgr.build.chunks.get(&key) {
        Some(ChunkState::Resident(r))  => (r.node_base, r.node_count, true),
        Some(ChunkState::Uploading(u)) => (u.node_base, u.node_count, false),
        _ => return false,
    };

    let need = nodes.len() as u32;
    if need == 0 { return false; }

    // Re-fetch CURRENT slot.
    let slot = match mgr.build.chunks.get(&key) {
        Some(ChunkState::Resident(r))  => r.slot,
        Some(ChunkState::Uploading(u)) => u.slot,
        _ => return false,
    };
    let s = slot as usize;

    debug_assert!(
        mgr.slots.slot_to_key.get(s).copied() == Some(key),
        "replace_chunk_contents: slot_to_key mismatch: key={:?} slot={} slot_key={:?}",
        key, slot, mgr.slots.slot_to_key.get(s).copied()
    );

    // Decide allocation strategy FIRST (so we don't partially mutate state on failure).
    let (node_base, alloc_cap, pad_to_cap) = if need <= old_cap {
        // Reuse old allocation. Keep capacity as old_cap to avoid leaking arena space.
        (old_base, old_cap, true)
    } else {
        // Need bigger: allocate new range, then free old.
        let mut new_base = mgr.arena.alloc(need);
        if new_base.is_none() {
            for _ in 0..EVICT_ATTEMPTS {
                if !evict_one_farthest(mgr, center, key) { break; }
                new_base = mgr.arena.alloc(need);
                if new_base.is_some() { break; }
            }
        }
        let Some(new_base) = new_base else {
            return false; // keep old chunk intact
        };
        mgr.arena.free(old_base, old_cap);
        (new_base, need, false)
    };

    // Now it's safe to update slot-owned CPU payloads.
    mgr.slots.slot_macro[s]   = macro_words.clone();
    mgr.slots.slot_colinfo[s] = colinfo_words.clone();

    // Update meta. IMPORTANT: meta.node_count must match the uploaded buffer length and arena allocation capacity.
    let mut meta = mgr.slots.chunk_meta[s];
    meta.node_base = node_base;
    meta.node_count = need;
    meta.macro_base = slot * MACRO_WORDS_PER_CHUNK;
    meta.colinfo_base = slot * COLINFO_WORDS_PER_CHUNK;
    mgr.slots.chunk_meta[s] = meta;

    // Update state. IMPORTANT: keep node_count as allocation capacity (alloc_cap).
    if let Some(st) = mgr.build.chunks.get_mut(&key) {
        match st {
            ChunkState::Resident(r) => {
                r.slot = slot;
                r.node_base = node_base;
                r.node_count = alloc_cap;
                r.rewrite_in_flight = true;
            }
            ChunkState::Uploading(u) => {
                u.slot = slot;
                u.node_base = node_base;
                u.node_count = alloc_cap;
                u.uploaded = false;
            }
            _ => {}
        }
    }

    let (kind, completes_residency) = if is_resident_now {
        (UploadKind::RewriteResident, false)
    } else {
        (UploadKind::PromoteToResident, true)
    };

    super::uploads::enqueue(mgr, ChunkUpload {
        key,
        slot,
        kind,
        meta,
        node_base,
        nodes,
        macro_words,
        ropes,
        colinfo_words,
        completes_residency,
    });

    mgr.grid.grid_dirty = true;
    true
}



#[inline(always)]
fn is_resident(mgr: &ChunkManager, k: ChunkKey) -> bool {
    matches!(mgr.build.chunks.get(&k), Some(ChunkState::Resident(_)))
}

#[inline(always)]
fn is_gpu_ready_key(mgr: &ChunkManager, k: ChunkKey) -> bool {
    match mgr.build.chunks.get(&k) {
        Some(ChunkState::Resident(_)) => true,
        Some(ChunkState::Uploading(up)) => up.uploaded,
        _ => false,
    }
}

/// Enforce the invariant:
/// - all Resident chunks are packed into the prefix [0..resident_slots)
/// - all Uploading chunks are in [resident_slots..)
fn repair_slot_prefix(mgr: &mut ChunkManager) {
    // Count how many residents we actually have among slotted chunks.
    let mut resident_count = 0usize;
    for &k in &mgr.slots.slot_to_key {
        if is_resident(mgr, k) {
            resident_count += 1;
        }
    }

    // Two-pointer partition by swapping.
    let mut i = 0usize;
    let mut j = mgr.slots.slot_to_key.len();

    while i < resident_count {
        let ki = mgr.slots.slot_to_key[i];
        if is_resident(mgr, ki) {
            i += 1;
            continue;
        }

        // Find a resident from the end to swap into i.
        while j > i {
            j -= 1;
            let kj = mgr.slots.slot_to_key[j];
            if is_resident(mgr, kj) {
                break;
            }
        }

        if j <= i {
            break;
        }

        // Before swap: record whether these slots need GPU rewrites.
        let need_rewrite_i = is_gpu_ready_key(mgr, mgr.slots.slot_to_key[i]);
        let need_rewrite_j = is_gpu_ready_key(mgr, mgr.slots.slot_to_key[j]);

        swap_slots(mgr, i, j);

        // After swap: if either slot is GPU-relevant, enqueue rewrites to fix bases/meta.
        if need_rewrite_i || is_gpu_ready_key(mgr, mgr.slots.slot_to_key[i]) {
            enqueue_slot_rewrite(mgr, i);
        }
        if need_rewrite_j || is_gpu_ready_key(mgr, mgr.slots.slot_to_key[j]) {
            enqueue_slot_rewrite(mgr, j);
        }

        i += 1;
    }

    mgr.slots.resident_slots = resident_count;
}
