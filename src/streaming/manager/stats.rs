use crate::streaming::types::*;
use super::ChunkManager;

pub fn stats(mgr: &ChunkManager) -> Option<StreamStats> {
    let mut s = StreamStats::default();

    if let Some(c) = mgr.build.last_center {
        s.center = (c.x, c.y, c.z);
    }

    s.resident_slots = mgr.slots.resident_slots as u32;
    s.total_slots    = mgr.slots.slot_to_key.len() as u32;
    s.chunks_map     = mgr.build.chunks.len() as u32;

    for st in mgr.build.chunks.values() {
        match st {
            ChunkState::Queued        => s.st_queued += 1,
            ChunkState::Building      => s.st_building += 1,
            ChunkState::Uploading(_)  => s.st_uploading += 1,
            ChunkState::Resident(_)   => s.st_resident += 1,
        }
    }

    s.in_flight = mgr.build.in_flight as u32;
    s.done_backlog = mgr.build.rx_done.len() as u32;

    s.up_rewrite = mgr.uploads.uploads_rewrite.len() as u32;
    s.up_active  = mgr.uploads.uploads_active.len() as u32;
    s.up_other   = mgr.uploads.uploads_other.len() as u32;

    let (cb, ce, cl) = mgr.cache.stats();
    s.cache_bytes   = cb as u64;
    s.cache_entries = ce as u32;
    s.cache_lru     = cl as u32;

    s.build_queue_len = mgr.build.build_queue.len() as u32;
    s.queued_set_len  = mgr.build.queued_set.len() as u32;
    s.cancels_len     = mgr.build.cancels.len() as u32;

    let mut orphan = 0u32;
    for (k, st) in mgr.build.chunks.iter() {
        if matches!(st, ChunkState::Queued) && !mgr.build.queued_set.contains(k) {
            orphan += 1;
        }
    }
    s.orphan_queued = orphan;

    Some(s)
}
