use std::{collections::VecDeque, mem::size_of};
use crate::{render::gpu_types::{ChunkMetaGpu, NodeGpu, NodeRopesGpu}};
use crate::app::config;
use crate::streaming::types::*;
use super::{ChunkManager};
use super::keep;

#[inline]
fn upload_bytes(u: &ChunkUpload) -> usize {
    let mut b = size_of::<ChunkMetaGpu>();
    b += u.nodes.len() * size_of::<NodeGpu>();
    b += u.macro_words.len() * size_of::<u32>();
    b += u.ropes.len() * size_of::<NodeRopesGpu>();
    b += u.colinfo_words.len() * size_of::<u32>();
    b
}

#[inline(always)]
pub fn upload_dist_score(k: ChunkKey, c: ChunkKey) -> f32 {
    let dx = (k.x - c.x) as f32;
    let dz = (k.z - c.z) as f32;
    let dy = (k.y - c.y) as f32;
    dx.abs() + dz.abs() + 2.0 * dy.abs()
}

pub fn enqueue(mgr: &mut ChunkManager, u: ChunkUpload) {
    #[cfg(debug_assertions)]
    {
        let expect = matches!(u.kind, UploadKind::PromoteToResident);
        debug_assert_eq!(
            u.completes_residency, expect,
            "ChunkUpload kind/completes_residency mismatch: key={:?} kind={:?} completes_residency={}",
            u.key, u.kind, u.completes_residency
        );
    }


    if !u.completes_residency {
        mgr.uploads.uploads_rewrite.push_front(u);
        return;
    }

    let Some(center) = mgr.build.last_center else {
        mgr.uploads.uploads_other.push_back(u);
        return;
    };

    if keep::in_active_xz(center, u.key) {
        insert_sorted_by_center(&mut mgr.uploads.uploads_active, u, center);
    } else {
        mgr.uploads.uploads_other.push_back(u);
    }
}

pub fn rebucket_for_center(mgr: &mut ChunkManager, center: ChunkKey) {
    let mut new_active = VecDeque::with_capacity(mgr.uploads.uploads_active.len());
    let mut new_other  = VecDeque::with_capacity(mgr.uploads.uploads_other.len());

    let ar = config::ACTIVE_RADIUS.max(PRIORITY_RADIUS);

    let is_active = |k: ChunkKey| {
        let dx = (k.x - center.x).abs();
        let dz = (k.z - center.z).abs();
        dx <= ar && dz <= ar
    };

    for u in mgr.uploads.uploads_active.drain(..) {
        if is_active(u.key) { new_active.push_back(u); }
        else { new_other.push_back(u); }
    }

    for u in mgr.uploads.uploads_other.drain(..) {
        if is_active(u.key) { new_active.push_back(u); }
        else { new_other.push_back(u); }
    }

    mgr.uploads.uploads_active = new_active;
    mgr.uploads.uploads_other  = new_other;
}

#[inline]
fn insert_sorted_by_center(q: &mut VecDeque<ChunkUpload>, u: ChunkUpload, center: ChunkKey) {
    let us = upload_dist_score(u.key, center);
    let pos = q.iter()
        .position(|e| upload_dist_score(e.key, center) > us)
        .unwrap_or(q.len());
    q.insert(pos, u);
}

#[inline]
fn uploads_len_total(mgr: &ChunkManager) -> usize {
    mgr.uploads.uploads_rewrite.len() + mgr.uploads.uploads_active.len() + mgr.uploads.uploads_other.len()
}

pub fn take_all(mgr: &mut ChunkManager) -> Vec<ChunkUpload> {
    let mut out = Vec::new();
    out.extend(mgr.uploads.uploads_rewrite.drain(..));
    out.extend(mgr.uploads.uploads_active.drain(..));
    out.extend(mgr.uploads.uploads_other.drain(..));
    out
}

pub fn take_budgeted(mgr: &mut ChunkManager) -> Vec<ChunkUpload> {
    let backlog = uploads_len_total(mgr);

    let max_uploads = (MAX_UPLOADS_PER_FRAME + backlog / 4).clamp(MAX_UPLOADS_PER_FRAME, 32);
    let max_bytes   = (MAX_UPLOAD_BYTES_PER_FRAME + backlog * (256 << 10)).clamp(MAX_UPLOAD_BYTES_PER_FRAME, 16 << 20);

    let mut out = Vec::new();
    let mut bytes = 0usize;

    let mut rewrites_taken = 0usize;
    let rewrite_cap = 4;

    let mut pop_next = |mgr: &mut ChunkManager| -> Option<(u8, ChunkUpload)> {
        if rewrites_taken < rewrite_cap {
            if let Some(u) = mgr.uploads.uploads_rewrite.pop_front() {
                rewrites_taken += 1;
                return Some((0, u));
            }
        }
        if let Some(u) = mgr.uploads.uploads_active.pop_front() { return Some((1, u)); }
        if let Some(u) = mgr.uploads.uploads_other.pop_front()  { return Some((2, u)); }
        None
    };

    // NOTE: push *back* so we don't immediately re-pop the same blocked item.
    let push_back_same = |mgr: &mut ChunkManager, which: u8, u: ChunkUpload| {
        match which {
            0 => mgr.uploads.uploads_rewrite.push_back(u),
            1 => mgr.uploads.uploads_active.push_back(u),
            _ => mgr.uploads.uploads_other.push_back(u),
        }
    };

    // Prevent infinite scan when gate is closed and almost everything is non-priority.
    let mut deferred = 0usize;
    let defer_cap = uploads_len_total(mgr).max(1);

    while let Some((which, mut u)) = pop_next(mgr) {
        if out.len() >= max_uploads {
            push_back_same(mgr, which, u);
            break;
        }

        // priority gate
        if u.completes_residency {
            if let Some(center) = mgr.build.last_center {
                let gate = !super::slots::priority_box_ready(mgr, center);
                if gate && !super::slots::in_priority_box(mgr, center, u.key) {
                    push_back_same(mgr, which, u);
                    deferred += 1;
                    if deferred >= defer_cap {
                        break;
                    }
                    continue;
                }
            }
        }

        // validate slot + update bases
        let slot = match mgr.build.chunks.get(&u.key) {
            Some(ChunkState::Resident(r)) => r.slot,
            Some(ChunkState::Uploading(up)) => up.slot,
            _ => continue,
        };

        u.slot = slot;
        u.meta.macro_base = slot * MACRO_WORDS_PER_CHUNK;
        u.meta.colinfo_base = slot * COLINFO_WORDS_PER_CHUNK;

        let ub = upload_bytes(&u);

        if bytes + ub > max_bytes && !out.is_empty() {
            push_back_same(mgr, which, u);
            break;
        }

        bytes += ub;
        out.push(u);
    }

    out
}
