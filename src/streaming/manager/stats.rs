// src/streaming/manager/stats.rs
use crate::streaming::types::*;
use super::ChunkManager;
use crate::svo::builder::BuildTimingsMs;

pub fn stats(mgr: &mut ChunkManager) -> Option<StreamStats> {
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

    // ---------------------------------------------------------------------
    // Drain timing window (only called on profiler cadence)
    // ---------------------------------------------------------------------
    let w = mgr.timing.drain();

    s.builds_done     = w.builds_done;
    s.builds_canceled = w.builds_canceled;

    if w.builds_done > 0 {
        let n = w.builds_done as f64;

        s.queue_ms_avg = w.queue_ms_sum / n;
        s.queue_ms_max = w.queue_ms_max;

        s.build_ms_avg = w.build_ms_sum / n;
        s.build_ms_max = w.build_ms_max;

        s.nodes_avg = (w.nodes_sum as f64) / n;
        s.nodes_max = w.nodes_max;

        let mut bt_avg = w.bt_sum;
        bt_avg.total         /= n;
        bt_avg.height_cache  /= n;
        bt_avg.tree_mask     /= n;
        bt_avg.ground_2d     /= n;
        bt_avg.ground_mip    /= n;
        bt_avg.tree_top      /= n;
        bt_avg.tree_mip      /= n;
        bt_avg.material_fill /= n;
        bt_avg.colinfo       /= n;
        bt_avg.prefix_x      /= n;
        bt_avg.prefix_y      /= n;
        bt_avg.prefix_z      /= n;
        bt_avg.macro_occ     /= n;
        bt_avg.svo_build     /= n;
        bt_avg.ropes         /= n;

        s.bt_avg = bt_avg;
        s.bt_max = w.bt_max;
    } else {
        s.queue_ms_avg = 0.0;
        s.queue_ms_max = 0.0;
        s.build_ms_avg = 0.0;
        s.build_ms_max = 0.0;
        s.nodes_avg = 0.0;
        s.nodes_max = 0;

        s.bt_avg = BuildTimingsMs::default();
        s.bt_max = BuildTimingsMs::default();
    }


    Some(s)
}
