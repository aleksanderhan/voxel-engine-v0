// src/svo/builder_ropes.rs
//
// Rope building on CPU for packed SVO nodes.
// Ropes provide neighbor pointers for traversal.

use crate::render::gpu_types::{NodeGpu, NodeRopesGpu};

const INVALID: u32 = u32::MAX;
const LEAF: u32 = INVALID;

#[inline]
fn is_leaf(n: &NodeGpu) -> bool {
    n.child_base == LEAF
}

#[inline]
fn ropes_invalid() -> NodeRopesGpu {
    NodeRopesGpu {
        px: INVALID,
        nx: INVALID,
        py: INVALID,
        ny: INVALID,
        pz: INVALID,
        nz: INVALID,
        _pad0: 0,
        _pad1: 0,
    }
}

#[inline]
fn child_idx(nodes: &[NodeGpu], parent_idx: u32, ci: u32) -> u32 {
    let p = &nodes[parent_idx as usize];
    debug_assert!(!is_leaf(p));

    let mask = p.child_mask;
    let bit = 1u32 << ci;

    if (mask & bit) == 0 {
        return INVALID;
    }

    // number of children before ci in the packed array
    let before = mask & (bit - 1);
    let rank = before.count_ones() as u32;

    p.child_base + rank
}

#[inline]
fn descend_one(nodes: &[NodeGpu], nei: u32, hx: u32, hy: u32, hz: u32) -> u32 {
    if nei == INVALID {
        return INVALID;
    }
    let n = &nodes[nei as usize];
    if is_leaf(n) {
        return nei;
    }
    let ci = hx | (hy << 1) | (hz << 2);
    child_idx(nodes, nei, ci)
}

fn compute_child_of_ci(p: &NodeGpu) -> [u32; 8] {
    let mut child_of_ci = [INVALID; 8];
    let mask = p.child_mask;

    let mut rank = 0u32;
    for ci in 0u32..8u32 {
        if (mask & (1u32 << ci)) != 0 {
            child_of_ci[ci as usize] = p.child_base + rank;
            rank += 1;
        }
    }
    child_of_ci
}

fn build_ropes_rec(nodes: &[NodeGpu], ropes: &mut [NodeRopesGpu], idx: u32) {
    if is_leaf(&nodes[idx as usize]) {
        return;
    }

    let p = &nodes[idx as usize];
    let pr = ropes[idx as usize];
    let mask = p.child_mask;

    let child_of_ci = compute_child_of_ci(p);

    // For each existing child, compute its ropes.
    for ci in 0u32..8u32 {
        if (mask & (1u32 << ci)) == 0 {
            continue;
        }

        let hx = ci & 1;
        let hy = (ci >> 1) & 1;
        let hz = (ci >> 2) & 1;

        let self_child = child_of_ci[ci as usize];
        debug_assert_ne!(self_child, INVALID);

        let sib_x = ci ^ 1;
        let sib_y = ci ^ 2;
        let sib_z = ci ^ 4;

        let sib = |sci: u32| -> u32 { child_of_ci[sci as usize] };

        // +X / -X
        let px = if hx == 0 {
            let s = sib(sib_x);
            if s != INVALID {
                s
            } else {
                descend_one(nodes, pr.px, 0, hy, hz)
            }
        } else {
            descend_one(nodes, pr.px, 0, hy, hz)
        };

        let nx = if hx == 1 {
            let s = sib(sib_x);
            if s != INVALID {
                s
            } else {
                descend_one(nodes, pr.nx, 1, hy, hz)
            }
        } else {
            descend_one(nodes, pr.nx, 1, hy, hz)
        };

        // +Y / -Y
        let py = if hy == 0 {
            let s = sib(sib_y);
            if s != INVALID {
                s
            } else {
                descend_one(nodes, pr.py, hx, 0, hz)
            }
        } else {
            descend_one(nodes, pr.py, hx, 0, hz)
        };

        let ny = if hy == 1 {
            let s = sib(sib_y);
            if s != INVALID {
                s
            } else {
                descend_one(nodes, pr.ny, hx, 1, hz)
            }
        } else {
            descend_one(nodes, pr.ny, hx, 1, hz)
        };

        // +Z / -Z
        let pz = if hz == 0 {
            let s = sib(sib_z);
            if s != INVALID {
                s
            } else {
                descend_one(nodes, pr.pz, hx, hy, 0)
            }
        } else {
            descend_one(nodes, pr.pz, hx, hy, 0)
        };

        let nz = if hz == 1 {
            let s = sib(sib_z);
            if s != INVALID {
                s
            } else {
                descend_one(nodes, pr.nz, hx, hy, 1)
            }
        } else {
            descend_one(nodes, pr.nz, hx, hy, 1)
        };

        ropes[self_child as usize] = NodeRopesGpu { px, nx, py, ny, pz, nz, _pad0: 0, _pad1: 0 };
    }

    // Recurse into existing children
    for ci in 0u32..8u32 {
        if (mask & (1u32 << ci)) == 0 {
            continue;
        }
        let c = child_of_ci[ci as usize];
        if c != INVALID {
            build_ropes_rec(nodes, ropes, c);
        }
    }
}

pub fn build_ropes(nodes: &[NodeGpu]) -> Vec<NodeRopesGpu> {
    let mut ropes = vec![ropes_invalid(); nodes.len()];
    // Root external ropes are invalid => stay INVALID.
    build_ropes_rec(nodes, &mut ropes, 0);
    ropes
}
