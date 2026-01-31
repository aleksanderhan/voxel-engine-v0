// src/svo/builder.rs

use std::sync::atomic::{AtomicBool, Ordering};

use crate::{
    config,
    render::gpu_types::{NodeGpu, NodeRopesGpu},
    world::{
        materials::{AIR, DIRT, GRASS, STONE, WOOD},
        WorldGen,
    },
};

use super::mips::{build_max_mip_inplace, build_minmax_mip_inplace, MaxMipView, MinMaxMipView};

const LEAF: u32 = 0xFFFF_FFFF;
const INVALID_U32: u32 = 0xFFFF_FFFF;

// 8^3 bits = 512 bits = 16 u32
const MACRO_WORDS_PER_CHUNK_USIZE: usize = 16;

#[inline]
fn should_cancel(cancel: &AtomicBool) -> bool {
    cancel.load(Ordering::Relaxed)
}

/// Pack (x,y,z,level) into NodeGpu.key.
/// - chunk_size = 64 => max coord at finest level is 63, fits 6 bits each.
/// - level: 0=root(size 64), 6=voxel(size 1)
#[inline]
fn pack_key(chunk_size: u32, ox: i32, oy: i32, oz: i32, size: i32) -> u32 {
    let cs = chunk_size;
    debug_assert!(cs.is_power_of_two());
    let size_u = size as u32;
    debug_assert!(size_u.is_power_of_two());

    let lvl = cs.trailing_zeros() - size_u.trailing_zeros(); // 0..=log2(cs)
    let cx = (ox as u32) / size_u;
    let cy = (oy as u32) / size_u;
    let cz = (oz as u32) / size_u;

    (cx & 63)
        | ((cy & 63) << 6)
        | ((cz & 63) << 12)
        | ((lvl & 7) << 18)
}

#[inline]
fn idx2(side: usize, x: usize, z: usize) -> usize {
    z * side + x
}

#[inline]
fn idx3(side: usize, x: usize, y: usize, z: usize) -> usize {
    (y * side * side) + (z * side) + x
}

#[inline]
fn pidx(dim: usize, x: usize, y: usize, z: usize) -> usize {
    (z * dim * dim) + (y * dim) + x
}

#[inline]
fn prefix_sum_cube(prefix: &[u32], side: usize, x0: usize, y0: usize, z0: usize, size: usize) -> u32 {
    let dim = side + 1;
    let x1 = x0 + size;
    let y1 = y0 + size;
    let z1 = z0 + size;

    let a = prefix[pidx(dim, x1, y1, z1)] as i64;
    let b = prefix[pidx(dim, x0, y1, z1)] as i64;
    let c = prefix[pidx(dim, x1, y0, z1)] as i64;
    let d = prefix[pidx(dim, x1, y1, z0)] as i64;
    let e = prefix[pidx(dim, x0, y0, z1)] as i64;
    let f = prefix[pidx(dim, x0, y1, z0)] as i64;
    let g = prefix[pidx(dim, x1, y0, z0)] as i64;
    let h = prefix[pidx(dim, x0, y0, z0)] as i64;

    let s = a - b - c - d + e + f + g - h;
    debug_assert!(s >= 0);
    s as u32
}

/// Reusable scratch buffers for chunk building (reduces allocations & improves locality).
pub struct BuildScratch {
    // 2D (side*side)
    ground: Vec<i32>,
    tree_top: Vec<i32>,

    // height cache
    height_cache: Vec<i32>,
    height_cache_w: usize,
    height_cache_h: usize,

    // 3D (side^3)
    material: Vec<u32>,
    prefix: Vec<u32>,

    // mip storage
    ground_min_levels: Vec<Vec<i32>>,
    ground_max_levels: Vec<Vec<i32>>,
    tree_levels: Vec<Vec<i32>>,

    // per-column top-most non-air (y, mat)
    col_top_y: Vec<u8>,   // 255 = empty
    col_top_mat: Vec<u8>, // low 8 bits of material id
}

impl BuildScratch {
    pub fn new() -> Self {
        Self {
            ground: Vec::new(),
            tree_top: Vec::new(),
            height_cache: Vec::new(),
            height_cache_w: 0,
            height_cache_h: 0,
            material: Vec::new(),
            prefix: Vec::new(),
            ground_min_levels: Vec::new(),
            ground_max_levels: Vec::new(),
            tree_levels: Vec::new(),
            col_top_y: Vec::new(),
            col_top_mat: Vec::new(),
        }
    }

    #[inline]
    fn ensure_coltop(col_top_y: &mut Vec<u8>, col_top_mat: &mut Vec<u8>, side: usize) {
        let need = side * side;
        if col_top_y.len() != need {
            col_top_y.resize(need, 255);
            col_top_mat.resize(need, 0);
        } else {
            col_top_y.fill(255);
            col_top_mat.fill(0);
        }
    }


    #[inline]
    fn ensure_height_cache(&mut self, w: usize, h: usize) {
        let need = w * h;
        if self.height_cache.len() != need {
            self.height_cache.resize(need, 0);
        } else {
            self.height_cache.fill(0);
        }
        self.height_cache_w = w;
        self.height_cache_h = h;
    }

    #[inline]
    fn ensure_2d(v: &mut Vec<i32>, side: usize, fill: i32) {
        let need = side * side;
        if v.len() != need {
            v.resize(need, fill);
        } else {
            v.fill(fill);
        }
    }

    #[inline]
    fn ensure_3d_u32(v: &mut Vec<u32>, side: usize, fill: u32) {
        let need = side * side * side;
        if v.len() != need {
            v.resize(need, fill);
        } else {
            v.fill(fill);
        }
    }

    #[inline]
    fn ensure_prefix(v: &mut Vec<u32>, side: usize) {
        let dim = side + 1;
        let need = dim * dim * dim;
        if v.len() != need {
            v.resize(need, 0);
        } else {
            v.fill(0);
        }
    }
}

// -----------------------------
// Rope building (CPU)
// -----------------------------

#[inline]
fn is_leaf(n: &NodeGpu) -> bool {
    n.child_base == LEAF
}

#[inline]
fn ropes_invalid() -> NodeRopesGpu {
    NodeRopesGpu {
        px: INVALID_U32,
        nx: INVALID_U32,
        py: INVALID_U32,
        ny: INVALID_U32,
        pz: INVALID_U32,
        nz: INVALID_U32,
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
        return INVALID_U32;
    }

    // number of children before ci in the packed array
    let before = mask & (bit - 1);
    let rank = before.count_ones() as u32;

    p.child_base + rank
}


#[inline]
fn descend_one(nodes: &[NodeGpu], nei: u32, hx: u32, hy: u32, hz: u32) -> u32 {
    if nei == INVALID_U32 {
        return INVALID_U32;
    }
    let n = &nodes[nei as usize];
    if is_leaf(n) {
        return nei;
    }
    let ci = hx | (hy << 1) | (hz << 2);
    child_idx(nodes, nei, ci)
}


fn build_ropes_rec(nodes: &[NodeGpu], ropes: &mut [NodeRopesGpu], idx: u32) {
    if is_leaf(&nodes[idx as usize]) {
        return;
    }

    let p = &nodes[idx as usize];
    let pr = ropes[idx as usize];
    let mask = p.child_mask;

    // For each *existing* child, compute its ropes.
    for ci in 0u32..8u32 {
        if (mask & (1u32 << ci)) == 0 {
            continue;
        }

        let hx = ci & 1;
        let hy = (ci >> 1) & 1;
        let hz = (ci >> 2) & 1;

        let self_child = child_idx(nodes, idx, ci);
        debug_assert_ne!(self_child, INVALID_U32);

        let sib_x = ci ^ 1;
        let sib_y = ci ^ 2;
        let sib_z = ci ^ 4;

        // helper: sibling root if it exists
        let sib = |sci: u32| -> u32 {
            if (mask & (1u32 << sci)) != 0 {
                child_idx(nodes, idx, sci)
            } else {
                INVALID_U32
            }
        };

        // +X / -X
        let px = if hx == 0 {
            let s = sib(sib_x);
            if s != INVALID_U32 { s } else { descend_one(nodes, pr.px, 0, hy, hz) }
        } else {
            descend_one(nodes, pr.px, 0, hy, hz)
        };

        let nx = if hx == 1 {
            let s = sib(sib_x);
            if s != INVALID_U32 { s } else { descend_one(nodes, pr.nx, 1, hy, hz) }
        } else {
            descend_one(nodes, pr.nx, 1, hy, hz)
        };

        // +Y / -Y
        let py = if hy == 0 {
            let s = sib(sib_y);
            if s != INVALID_U32 { s } else { descend_one(nodes, pr.py, hx, 0, hz) }
        } else {
            descend_one(nodes, pr.py, hx, 0, hz)
        };

        let ny = if hy == 1 {
            let s = sib(sib_y);
            if s != INVALID_U32 { s } else { descend_one(nodes, pr.ny, hx, 1, hz) }
        } else {
            descend_one(nodes, pr.ny, hx, 1, hz)
        };

        // +Z / -Z
        let pz = if hz == 0 {
            let s = sib(sib_z);
            if s != INVALID_U32 { s } else { descend_one(nodes, pr.pz, hx, hy, 0) }
        } else {
            descend_one(nodes, pr.pz, hx, hy, 0)
        };

        let nz = if hz == 1 {
            let s = sib(sib_z);
            if s != INVALID_U32 { s } else { descend_one(nodes, pr.nz, hx, hy, 1) }
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
        let c = child_idx(nodes, idx, ci);
        if c != INVALID_U32 {
            build_ropes_rec(nodes, ropes, c);
        }
    }
}

fn build_ropes(nodes: &[NodeGpu]) -> Vec<NodeRopesGpu> {
    let mut ropes = vec![ropes_invalid(); nodes.len()];
    // Root external ropes are invalid => stay INVALID_U32.
    build_ropes_rec(nodes, &mut ropes, 0);
    ropes
}

// -----------------------------------------------------------------------------
// Cancelable build with reusable scratch (fast path).
// NOW RETURNS: (nodes, macro_words, ropes)
// -----------------------------------------------------------------------------
pub fn build_chunk_svo_sparse_cancelable_with_scratch(
    gen: &WorldGen,
    chunk_origin: [i32; 3],
    chunk_size: u32,
    cancel: &AtomicBool,
    scratch: &mut BuildScratch,
) -> (Vec<NodeGpu>, Vec<u32>, Vec<NodeRopesGpu>, Vec<u32>) {
    if should_cancel(cancel) {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    let chunk_ox = chunk_origin[0];
    let chunk_oy = chunk_origin[1];
    let chunk_oz = chunk_origin[2];

    let cs_u = chunk_size;
    let cs_i = chunk_size as i32;
    debug_assert!(cs_u.is_power_of_two());

    let side = cs_u as usize;
    let vpm: i32 = config::VOXELS_PER_METER as i32;
    debug_assert!(vpm > 0);

    // -------------------------------------------------------------------------
    // Height cache
    // -------------------------------------------------------------------------
    let margin_m: i32 = 6;
    let margin: i32 = margin_m * vpm + (vpm - 1);

    let cache_x0 = chunk_ox - margin;
    let cache_z0 = chunk_oz - margin;
    let cache_x1 = chunk_ox + cs_i + margin; // inclusive
    let cache_z1 = chunk_oz + cs_i + margin; // inclusive

    let cache_w = (cache_x1 - cache_x0 + 1) as usize;
    let cache_h = (cache_z1 - cache_z0 + 1) as usize;

    scratch.ensure_height_cache(cache_w, cache_h);
    for z in 0..cache_h {
        if (z & 15) == 0 && should_cancel(cancel) {
            return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        }
        let wz = cache_z0 + z as i32;
        let row = z * cache_w;
        for x in 0..cache_w {
            let wx = cache_x0 + x as i32;
            scratch.height_cache[row + x] = gen.ground_height(wx, wz);
        }
    }

    let height_at = |wx: i32, wz: i32| -> i32 {
        if wx < cache_x0 || wx > cache_x1 || wz < cache_z0 || wz > cache_z1 {
            gen.ground_height(wx, wz)
        } else {
            let ix = (wx - cache_x0) as usize;
            let iz = (wz - cache_z0) as usize;
            scratch.height_cache[iz * scratch.height_cache_w + ix]
        }
    };

    // -------------------------------------------------------------------------
    // Tree cache/mask
    // -------------------------------------------------------------------------
    let (_tree_cache_unused, tree_mask) = gen.build_tree_cache_with_mask(
        chunk_ox,
        chunk_oy,
        chunk_oz,
        cs_i,
        &height_at,
        cancel,
    );

    // -------------------------------------------------------------------------
    // 2D maps (ground)
    // -------------------------------------------------------------------------
    BuildScratch::ensure_2d(&mut scratch.ground, side, 0);

    for lz in 0..cs_i {
        if (lz & 15) == 0 && should_cancel(cancel) {
            return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        }
        for lx in 0..cs_i {
            let wx = chunk_ox + lx;
            let wz = chunk_oz + lz;
            let g = height_at(wx, wz);

            let i = idx2(side, lx as usize, lz as usize);
            scratch.ground[i] = g;
        }
    }

    let ground_mip: MinMaxMipView<'_> = build_minmax_mip_inplace(
        &scratch.ground,
        cs_u,
        &mut scratch.ground_min_levels,
        &mut scratch.ground_max_levels,
    );

    // -------------------------------------------------------------------------
    // Tree top stamp (2D)
    // -------------------------------------------------------------------------
    BuildScratch::ensure_2d(&mut scratch.tree_top, side, -1);

    let pad_m = 4;
    let xm0 = (chunk_ox.div_euclid(vpm)) - pad_m;
    let xm1 = ((chunk_ox + cs_i).div_euclid(vpm)) + pad_m;
    let zm0 = (chunk_oz.div_euclid(vpm)) - pad_m;
    let zm1 = ((chunk_oz + cs_i).div_euclid(vpm)) + pad_m;

    for zm in zm0..=zm1 {
        if ((zm - zm0) & 3) == 0 && should_cancel(cancel) {
            return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        }

        for xm in xm0..=xm1 {
            let Some((trunk_h_vox, crown_r_vox)) = gen.tree_instance_at_meter(xm, zm) else {
                continue;
            };

            let tx = xm * vpm;
            let tz = zm * vpm;

            let g = height_at(tx, tz);
            let trunk_base = g + vpm;
            let trunk_top = trunk_base + trunk_h_vox;

            let canopy_h_vox = 5 * vpm;
            let top_y = trunk_top + canopy_h_vox + 2 * vpm;

            let r = crown_r_vox + 2 * vpm;

            for dz in -r..=r {
                for dx in -r..=r {
                    if dx * dx + dz * dz > r * r {
                        continue;
                    }

                    let wx = tx + dx;
                    let wz = tz + dz;

                    let lx = wx - chunk_ox;
                    let lz = wz - chunk_oz;
                    if lx >= 0 && lx < cs_i && lz >= 0 && lz < cs_i {
                        let i = idx2(side, lx as usize, lz as usize);
                        scratch.tree_top[i] = scratch.tree_top[i].max(top_y);
                    }
                }
            }

            // ensure trunk column included
            let lx = tx - chunk_ox;
            let lz = tz - chunk_oz;
            if lx >= 0 && lx < cs_i && lz >= 0 && lz < cs_i {
                let i = idx2(side, lx as usize, lz as usize);
                scratch.tree_top[i] = scratch.tree_top[i].max(trunk_top);
            }
        }
    }

    let tree_mip: MaxMipView<'_> =
        build_max_mip_inplace(&scratch.tree_top, cs_u, &mut scratch.tree_levels);

    // -------------------------------------------------------------------------
    // Precompute per-voxel material
    // -------------------------------------------------------------------------
    BuildScratch::ensure_3d_u32(&mut scratch.material, side, AIR);

    BuildScratch::ensure_coltop(&mut scratch.col_top_y, &mut scratch.col_top_mat, side);


    let dirt_depth = 3 * vpm;

    for ly in 0..cs_i {
        if (ly & 7) == 0 && should_cancel(cancel) {
            return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        }

        let wy = chunk_oy + ly;

        for lz in 0..cs_i {
            for lx in 0..cs_i {
                let col = idx2(side, lx as usize, lz as usize);
                let g = scratch.ground[col];

                let m: u32 = if wy < g {
                    if wy >= g - dirt_depth { DIRT } else { STONE }
                } else if wy == g {
                    let tm = tree_mask.material_local(lx as usize, ly as usize, lz as usize);
                    if tm == WOOD { WOOD } else { GRASS }
                } else {
                    let tm = tree_mask.material_local(lx as usize, ly as usize, lz as usize);
                    if tm != AIR { tm } else { AIR }
                };

                let i3 = idx3(side, lx as usize, ly as usize, lz as usize);
                scratch.material[i3] = m;

                //track top-most non-air for this (x,z) column
                if m != AIR {
                    scratch.col_top_y[col] = ly as u8;            // last non-air wins => highest y
                    scratch.col_top_mat[col] = (m & 0xFF) as u8;  // pack only low 8 bits
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Column top map (64x64): per (x,z), store top-most non-air voxel (y, mat)
    // packed u16: (mat8<<8) | y8, y8=255 means empty column
    // 2 entries per u32 => 2048 u32 words
    // -------------------------------------------------------------------------
    let side_u = cs_u as usize;
    debug_assert_eq!(side_u, 64, "colinfo packing assumes chunk_size=64");

    let mut colinfo_words = vec![0u32; 2048];

    for lz in 0..64usize {
        for lx in 0..64usize {
            let col = idx2(64, lx, lz);

            // computed during material fill
            let y8: u32 = scratch.col_top_y[col] as u32;      // 255 means empty
            let mat8: u32 = scratch.col_top_mat[col] as u32;  // 0 means empty

            let entry16: u32 = (y8 & 0xFF) | ((mat8 & 0xFF) << 8);

            let idx = (lz * 64 + lx) as u32; // 0..4095
            let w = (idx >> 1) as usize;     // 0..2047
            let hi = (idx & 1) != 0;

            if !hi {
                colinfo_words[w] = (colinfo_words[w] & 0xFFFF_0000) | entry16;
            } else {
                colinfo_words[w] = (colinfo_words[w] & 0x0000_FFFF) | (entry16 << 16);
            }
        }
    }


    // -------------------------------------------------------------------------
    // Prefix sum over solid occupancy
    // -------------------------------------------------------------------------
    BuildScratch::ensure_prefix(&mut scratch.prefix, side);
    let dim = side + 1;

    for z in 1..=side {
        if (z & 7) == 0 && should_cancel(cancel) {
            return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        }
        for y in 1..=side {
            let mut run: u32 = 0;
            for x in 1..=side {
                let v = (scratch.material[idx3(side, x - 1, y - 1, z - 1)] != AIR) as u32;
                run += v;

                let a = scratch.prefix[pidx(dim, x, y, z - 1)] as i64;
                let b = scratch.prefix[pidx(dim, x, y - 1, z)] as i64;
                let c = scratch.prefix[pidx(dim, x, y - 1, z - 1)] as i64;

                let p = a + b - c + (run as i64);
                debug_assert!(p >= 0);
                scratch.prefix[pidx(dim, x, y, z)] = p as u32;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Macro occupancy bitset (8x8x8 => 512 bits => 16 u32 words)
    // -------------------------------------------------------------------------
    let macro_dim: usize = 8;
    debug_assert_eq!(side % macro_dim, 0);

    let cell: usize = side / macro_dim;

    let mut macro_words = vec![0u32; MACRO_WORDS_PER_CHUNK_USIZE];

    for mz in 0..macro_dim {
        if should_cancel(cancel) {
            return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        }
        for my in 0..macro_dim {
            for mx in 0..macro_dim {
                let x0 = mx * cell;
                let y0 = my * cell;
                let z0 = mz * cell;

                let sum = prefix_sum_cube(&scratch.prefix, side, x0, y0, z0, cell);
                if sum > 0 {
                    let bit = mx + macro_dim * (my + macro_dim * mz); // 0..511
                    let w = bit >> 5;
                    let b = bit & 31;
                    macro_words[w] |= 1u32 << b;
                }
            }
        }
    }

    fn make_leaf(chunk_size: u32, ox: i32, oy: i32, oz: i32, size: i32, mat: u32) -> NodeGpu {
        NodeGpu {
            child_base: LEAF,
            child_mask: 0,
            material: mat,
            key: pack_key(chunk_size, ox, oy, oz, size),
        }
    }

    fn build_node(
        nodes: &mut Vec<NodeGpu>,
        chunk_size: u32,
        ox: i32,
        oy: i32,
        oz: i32,
        size: i32,
        chunk_oy: i32,
        material: &[u32],
        prefix: &[u32],
        side: usize,
        ground_mip: &MinMaxMipView<'_>,
        tree_mip: &MaxMipView<'_>,
        dirt_depth: i32,
        cancel: &AtomicBool,
    ) -> NodeGpu {
        if should_cancel(cancel) {
            return make_leaf(chunk_size, ox, oy, oz, size, AIR);
        }

        let size_u = size as u32;

        let (gmin, gmax) = ground_mip.query(ox, oz, size_u);
        let tmax = tree_mip.query_max(ox, oz, size_u);

        let y0 = chunk_oy + oy;
        let y1 = y0 + size - 1;

        // above everything
        let top_solid = gmax.max(tmax);
        if y0 > top_solid {
            return make_leaf(chunk_size, ox, oy, oz, size, AIR);
        }

        // deep solid stone
        if y1 < gmin - dirt_depth {
            return make_leaf(chunk_size, ox, oy, oz, size, STONE);
        }

        // empty check via prefix
        let sx = ox as usize;
        let sy = oy as usize;
        let sz = oz as usize;
        let s = size as usize;

        let sum = prefix_sum_cube(prefix, side, sx, sy, sz, s);
        if sum == 0 {
            return make_leaf(chunk_size, ox, oy, oz, size, AIR);
        }

        if size == 1 {
            let m = material[idx3(side, sx, sy, sz)];
            return make_leaf(chunk_size, ox, oy, oz, size, m);
        }

        let half = size / 2;

        // Collect child roots without pushing them yet (so they end up contiguous)
        let mut child_roots: [Option<NodeGpu>; 8] = [None; 8];
        let mut mask: u32 = 0;

        for ci in 0u32..8u32 {
            if ((ci as i32) & 3) == 0 && should_cancel(cancel) {
                return make_leaf(chunk_size, ox, oy, oz, size, AIR);
            }

            let dx = if (ci & 1) != 0 { half } else { 0 };
            let dy = if (ci & 2) != 0 { half } else { 0 };
            let dz = if (ci & 4) != 0 { half } else { 0 };

            let csx = (ox + dx) as usize;
            let csy = (oy + dy) as usize;
            let csz = (oz + dz) as usize;

            // fast empty test for the child region
            if prefix_sum_cube(prefix, side, csx, csy, csz, half as usize) == 0 {
                continue;
            }

            let child = build_node(
                nodes,
                chunk_size,
                ox + dx,
                oy + dy,
                oz + dz,
                half,
                chunk_oy,
                material,
                prefix,
                side,
                ground_mip,
                tree_mip,
                dirt_depth,
                cancel,
            );

            child_roots[ci as usize] = Some(child);
            mask |= 1u32 << ci;
        }

        if mask == 0 {
            return make_leaf(chunk_size, ox, oy, oz, size, AIR);
        }

        // NOW push roots contiguously
        let base = nodes.len() as u32;
        for ci in 0..8 {
            if let Some(ch) = child_roots[ci] {
                nodes.push(ch);
            }
        }

        NodeGpu {
            child_base: base,
            child_mask: mask,
            material: 0,
            key: pack_key(chunk_size, ox, oy, oz, size),
        }


    }

    // Root must be at index 0 for GPU.
    let mut nodes = vec![make_leaf(cs_u, 0, 0, 0, cs_i, AIR)];

    let root = build_node(
        &mut nodes,
        cs_u,
        0,
        0,
        0,
        cs_i,
        chunk_oy,
        &scratch.material,
        &scratch.prefix,
        side,
        &ground_mip,
        &tree_mip,
        dirt_depth,
        cancel,
    );

    if should_cancel(cancel) {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    nodes[0] = root;

    // Build ropes AFTER nodes are finalized.
    let ropes = build_ropes(&nodes);

    (nodes, macro_words, ropes, colinfo_words)
}
