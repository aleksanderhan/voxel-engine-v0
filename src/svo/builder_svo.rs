// src/svo/builder_svo.rs
//
// Bottom-up SVO builder.
// Builds leaf nodes for solid voxels, then packs parents upward level-by-level,
// and finally flattens into a single Vec<NodeGpu> with fixed child_base pointers.

use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;

use crate::render::gpu_types::NodeGpu;
use crate::world::materials::{AIR, STONE};

use super::builder_prefix::prefix_sum_cube;
use super::mips::MinMaxMipView;

const INVALID: u32 = u32::MAX;
const LEAF: u32 = INVALID;

/// Pack (x,y,z,level) into NodeGpu.key.
/// - chunk_size = 64 => max coord at finest level is 63, fits 6 bits each.
/// - level: 0=root(size 64), 6=voxel(size 1)
#[inline]
pub(crate) fn pack_key(chunk_size: u32, ox: i32, oy: i32, oz: i32, size: i32) -> u32 {
    let cs = chunk_size;
    debug_assert!(cs.is_power_of_two());
    let size_u = size as u32;
    debug_assert!(size_u.is_power_of_two());

    let lvl = cs.trailing_zeros() - size_u.trailing_zeros(); // 0..=log2(cs)
    let sh = size_u.trailing_zeros();
    let cx = (ox as u32) >> sh;
    let cy = (oy as u32) >> sh;
    let cz = (oz as u32) >> sh;

    (cx & 63) | ((cy & 63) << 6) | ((cz & 63) << 12) | ((lvl & 7) << 18)
}

#[inline]
fn should_cancel(cancel: &AtomicBool) -> bool {
    cancel.load(Ordering::Relaxed)
}

#[inline(always)]
fn make_leaf(chunk_size: u32, ox: i32, oy: i32, oz: i32, size: i32, mat: u32) -> NodeGpu {
    NodeGpu {
        child_base: LEAF,
        child_mask: 0,
        material: mat,
        key: pack_key(chunk_size, ox, oy, oz, size),
    }
}

#[inline(always)]
fn idx_xyz(side: usize, x: usize, y: usize, z: usize) -> usize {
    (y * side * side) + (z * side) + x
}

#[inline(always)]
fn coord_from_linear(i: usize, side: usize) -> (usize, usize, usize) {
    debug_assert!(side.is_power_of_two());
    let bits = side.trailing_zeros() as usize;
    let mask = side - 1;

    let x = i & mask;
    let z = (i >> bits) & mask;
    let y = i >> (2 * bits);
    (x, y, z)
}

fn build_level_bottom_up(
    chunk_size: u32,
    chunk_oy: i32,
    parent_size: i32,          // size of parent cell in voxels
    parent_side: usize,        // number of parent cells per axis
    child_size: i32,           // size of child cell in voxels (= parent_size/2)
    child_side: usize,         // number of child cells per axis (= parent_side*2)
    child_idx_grid: &[u32],    // dense grid: child cell -> index in child_nodes (or INVALID)
    child_nodes: &[NodeGpu],   // compact list of existing child nodes
    prefix: &[u32],
    side_vox: usize,           // == chunk_size as usize (64)
    ground_mip: &MinMaxMipView<'_>,
    dirt_depth: i32,
    cancel: &AtomicBool,
) -> (Vec<NodeGpu>, Vec<u32>, Vec<NodeGpu>) {
    debug_assert_eq!(child_side, parent_side * 2);
    debug_assert_eq!(child_idx_grid.len(), child_side * child_side * child_side);

    let parent_cells = parent_side * parent_side * parent_side;

    // Per parent-cell metadata (dense, deterministic scan order).
    // mask == 0 => empty => parent doesn't exist in compacted parent_nodes.
    let mut mask_dense = vec![0u8; parent_cells];
    let mut collapse_stone_dense = vec![0u8; parent_cells]; // 1 => collapse to STONE leaf

    // 1) Compute child masks (parallel-friendly, deterministic storage)
    mask_dense
        .par_iter_mut()
        .enumerate()
        .for_each(|(pi, out_mask)| {
            if (pi & 4095) == 0 && should_cancel(cancel) {
                return;
            }

            let (px, py, pz) = coord_from_linear(pi, parent_side);

            let cx0 = px * 2;
            let cy0 = py * 2;
            let cz0 = pz * 2;

            let mut m: u8 = 0;

            // ci bits: x=1, y=2, z=4
            for ci in 0u32..8u32 {
                let dx = (ci & 1) as usize;
                let dy = ((ci >> 1) & 1) as usize;
                let dz = ((ci >> 2) & 1) as usize;

                let cx = cx0 + dx;
                let cy = cy0 + dy;
                let cz = cz0 + dz;

                let c_lin = idx_xyz(child_side, cx, cy, cz);
                let c_idx = unsafe { *child_idx_grid.get_unchecked(c_lin) };
                if c_idx != INVALID {
                    m |= 1u8 << ci;
                }
            }

            *out_mask = m;
        });

    if should_cancel(cancel) {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    // 2) Optional collapse-to-stone decision (only where it can trigger)
    //    We avoid prefix_sum_cube() unless y1 < gmin - dirt_depth.
    collapse_stone_dense
        .par_iter_mut()
        .enumerate()
        .for_each(|(pi, out_flag)| {
            let m = mask_dense[pi];
            if m == 0 {
                *out_flag = 0;
                return;
            }

            if (pi & 4095) == 0 && should_cancel(cancel) {
                return;
            }

            let (px, py, pz) = coord_from_linear(pi, parent_side);

            let ox = (px as i32) * parent_size;
            let oy = (py as i32) * parent_size;
            let oz = (pz as i32) * parent_size;

            let (gmin, _gmax) = ground_mip.query(ox, oz, parent_size as u32);
            let y0 = chunk_oy + oy;
            let y1 = y0 + parent_size - 1;

            if y1 >= gmin - dirt_depth {
                *out_flag = 0;
                return;
            }

            // Only here do we pay prefix_sum_cube().
            let sx = ox as usize;
            let sy = oy as usize;
            let sz = oz as usize;
            let s = parent_size as usize;

            let sum = prefix_sum_cube(prefix, side_vox, sx, sy, sz, s);
            let full = (s * s * s) as u32;

            *out_flag = if sum == full { 1 } else { 0 };
        });

    if should_cancel(cancel) {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    // 3) Compact parent nodes (prefix-sum over "exists") in deterministic order.
    let mut parent_idx_grid = vec![INVALID; parent_cells];
    let mut parent_count: u32 = 0;

    for pi in 0..parent_cells {
        if mask_dense[pi] != 0 {
            parent_idx_grid[pi] = parent_count;
            parent_count += 1;
        }
        if (pi & 8191) == 0 && should_cancel(cancel) {
            return (Vec::new(), Vec::new(), Vec::new());
        }
    }

    let mut parent_nodes = vec![make_leaf(chunk_size, 0, 0, 0, parent_size, AIR); parent_count as usize];
    let mut packed_children = Vec::<NodeGpu>::new();
    packed_children.reserve((parent_count as usize) * 4); // rough; grows if needed

    // 4) Fill parent_nodes and produce packed children (sequential, cache-friendly).
    //    Ordering:
    //      - parents in (pz,py,px) scan order (via linear index)
    //      - children in ci=0..7 order
    for pi in 0..parent_cells {
        let m = mask_dense[pi];
        if m == 0 {
            continue;
        }

        if (pi & 4095) == 0 && should_cancel(cancel) {
            return (Vec::new(), Vec::new(), Vec::new());
        }

        let p_out = parent_idx_grid[pi];
        debug_assert!(p_out != INVALID);

        let (px, py, pz) = coord_from_linear(pi, parent_side);

        let ox = (px as i32) * parent_size;
        let oy = (py as i32) * parent_size;
        let oz = (pz as i32) * parent_size;

        // Collapse to STONE leaf if flagged.
        if collapse_stone_dense[pi] != 0 {
            parent_nodes[p_out as usize] = make_leaf(chunk_size, ox, oy, oz, parent_size, STONE);
            continue;
        }

        let base = packed_children.len() as u32;

        let cx0 = px * 2;
        let cy0 = py * 2;
        let cz0 = pz * 2;

        let mut child_mask_u32: u32 = 0;

        for ci in 0u32..8u32 {
            if (m & (1u8 << ci)) == 0 {
                continue;
            }

            let dx = (ci & 1) as usize;
            let dy = ((ci >> 1) & 1) as usize;
            let dz = ((ci >> 2) & 1) as usize;

            let cx = cx0 + dx;
            let cy = cy0 + dy;
            let cz = cz0 + dz;

            let c_lin = idx_xyz(child_side, cx, cy, cz);
            let c_idx = unsafe { *child_idx_grid.get_unchecked(c_lin) };
            debug_assert!(c_idx != INVALID);

            packed_children.push(child_nodes[c_idx as usize]);
            child_mask_u32 |= 1u32 << ci;
        }

        parent_nodes[p_out as usize] = NodeGpu {
            child_base: base,
            child_mask: child_mask_u32,
            material: 0,
            key: pack_key(chunk_size, ox, oy, oz, parent_size),
        };
    }

    (parent_nodes, parent_idx_grid, packed_children)
}

pub fn build_svo_bottom_up(
    chunk_size: u32,
    chunk_oy: i32,
    material: &[u32],
    prefix: &[u32],
    ground_mip: &MinMaxMipView<'_>,
    dirt_depth: i32,
    cancel: &AtomicBool,
) -> Vec<NodeGpu> {
    let side_vox = chunk_size as usize;
    debug_assert!(chunk_size.is_power_of_two());
    debug_assert_eq!(material.len(), side_vox * side_vox * side_vox);

    let max_lvl = chunk_size.trailing_zeros() as usize; // 64 -> 6

    // ---------------------------
    // Level max_lvl (leaf voxels)
    // ---------------------------
    let n = material.len();

    // occ[i]=1 if solid, else 0 (parallel)
    let mut occ = vec![0u8; n];
    occ.par_iter_mut()
        .enumerate()
        .for_each(|(i, o)| *o = (material[i] != AIR) as u8);

    if should_cancel(cancel) {
        return Vec::new();
    }

    // positions via prefix sum (sequential, deterministic)
    let mut pos = vec![INVALID; n];
    let mut total: u32 = 0;
    for i in 0..n {
        if occ[i] != 0 {
            pos[i] = total;
            total += 1;
        }
        if (i & 65535) == 0 && should_cancel(cancel) {
            return Vec::new();
        }
    }

    let mut inv = vec![0u32; total as usize];
    for i in 0..n {
        if occ[i] != 0 {
            inv[pos[i] as usize] = i as u32;
        }
    }

    // leaf nodes + dense idx grid
    let mut leaf_nodes = vec![make_leaf(chunk_size, 0, 0, 0, 1, AIR); total as usize];
    let mut leaf_idx_grid = vec![INVALID; n];

    leaf_nodes
        .par_iter_mut()
        .enumerate()
        .for_each(|(p, out)| {
            let i = inv[p] as usize;
            let (x, y, z) = coord_from_linear(i, side_vox);
            let m = material[i];
            *out = make_leaf(chunk_size, x as i32, y as i32, z as i32, 1, m);
        });

    leaf_idx_grid
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, out)| {
            if occ[i] != 0 {
                *out = pos[i];
            }
        });

    if should_cancel(cancel) {
        return Vec::new();
    }

    // packed_children_levels[l] stores nodes at level l, packed by parents.
    // We'll fill levels 1..=max_lvl. Level 0 is the root node.
    let mut packed_children_levels: Vec<Vec<NodeGpu>> = vec![Vec::new(); max_lvl + 1];

    // Current child representation used to compute the next level up.
    let mut child_nodes = leaf_nodes;
    let mut child_idx_grid = leaf_idx_grid;
    let mut child_side = side_vox;
    let mut child_size: i32 = 1;

    // Build from level (max_lvl-1) down to 0
    for _lvl in (0..max_lvl).rev() {
        if should_cancel(cancel) {
            return Vec::new();
        }

        let parent_size = child_size * 2;
        let parent_side = child_side / 2;

        let (parent_nodes, parent_idx_grid, packed_children) = build_level_bottom_up(
            chunk_size,
            chunk_oy,
            parent_size,
            parent_side,
            child_size,
            child_side,
            &child_idx_grid,
            &child_nodes,
            prefix,
            side_vox,
            ground_mip,
            dirt_depth,
            cancel,
        );

        if should_cancel(cancel) {
            return Vec::new();
        }

        // packed children correspond to the current child_size
        let child_level = (chunk_size / (child_size as u32)).trailing_zeros() as usize;
        debug_assert!(child_level <= max_lvl);
        packed_children_levels[child_level] = packed_children;

        // Move up one level
        child_nodes = parent_nodes;
        child_idx_grid = parent_idx_grid;
        child_side = parent_side;
        child_size = parent_size;
    }

    // After loop, child_nodes is the root-level compact list (either empty or 1 node)
    let root = if child_nodes.is_empty() {
        make_leaf(chunk_size, 0, 0, 0, chunk_size as i32, AIR)
    } else {
        debug_assert_eq!(child_nodes.len(), 1);
        child_nodes[0]
    };

    // -----------------------------------------
    // Flatten into one Vec<NodeGpu> + fix child_base
    // Layout: [root] + level1 + level2 + ... + level6
    // -----------------------------------------
    let mut bases = vec![0u32; max_lvl + 1];
    let mut cur: u32 = 1;
    for lvl in 1..=max_lvl {
        bases[lvl] = cur;
        cur += packed_children_levels[lvl].len() as u32;
    }

    // Fix-up child_base pointers (only for internal nodes)
    let mut root_fixed = root;
    if root_fixed.child_base != LEAF {
        root_fixed.child_base += bases[1];
    }

    for lvl in 1..max_lvl {
        let add = bases[lvl + 1];
        for n in &mut packed_children_levels[lvl] {
            if n.child_base != LEAF {
                n.child_base += add;
            }
        }
    }

    // Concatenate
    let total_nodes = cur as usize;
    let mut out = Vec::with_capacity(total_nodes);
    out.push(root_fixed);
    for lvl in 1..=max_lvl {
        out.extend_from_slice(&packed_children_levels[lvl]);
    }

    out
}
