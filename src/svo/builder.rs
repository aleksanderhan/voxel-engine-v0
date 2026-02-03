// src/svo/builder.rs

use std::sync::atomic::{AtomicBool, Ordering};
use rayon::prelude::*;

use crate::app::config;
use crate::{
    render::gpu_types::{NodeGpu, NodeRopesGpu},
    world::{
        materials::{AIR, DIRT, GRASS, STONE, WOOD},
        WorldGen,
    },
};
use crate::world::edits::EditEntry;

use super::mips::{build_max_mip_inplace, build_minmax_mip_inplace, MaxMipView, MinMaxMipView};

const LEAF: u32 = 0xFFFF_FFFF;
const INVALID_U32: u32 = 0xFFFF_FFFF;

// 8^3 bits = 512 bits = 16 u32
const MACRO_WORDS_PER_CHUNK_USIZE: usize = 16;

// --- Worldgen build profiling ------------------------------------------------

#[derive(Clone, Copy, Default, Debug)]
pub struct BuildTimingsMs {
    pub total: f64,

    pub height_cache: f64,
    pub tree_mask: f64,
    pub ground_2d: f64,
    pub ground_mip: f64,
    pub tree_top: f64,
    pub tree_mip: f64,
    pub material_fill: f64,
    pub colinfo: f64,
    pub prefix_x: f64,
    pub prefix_y: f64,
    pub prefix_z: f64,
    pub macro_occ: f64,
    pub svo_build: f64,
    pub ropes: f64,

    // useful counters (not time)
    pub cache_w: u32,
    pub cache_h: u32,
    pub tree_cells_tested: u32,
    pub tree_instances: u32,
    pub solid_voxels: u32,
    pub nodes: u32,
}

#[inline(always)]
fn ms_since(t0: std::time::Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1000.0
}

macro_rules! time_it {
    ($tim:expr, $field:ident, $body:block) => {{
        let _t0 = std::time::Instant::now();
        let _r = { $body };
        $tim.$field += ms_since(_t0);
        _r
    }};
}



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
    let sh = size_u.trailing_zeros();
    let cx = (ox as u32) >> sh;
    let cy = (oy as u32) >> sh;
    let cz = (oz as u32) >> sh;


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

    // Precompute child index for each ci in one pass (no count_ones in hot loop).
    let mut child_of_ci = [INVALID_U32; 8];
    let mut rank = 0u32;
    for ci in 0u32..8u32 {
        if (mask & (1u32 << ci)) != 0 {
            child_of_ci[ci as usize] = p.child_base + rank;
            rank += 1;
        }
    }


    // For each *existing* child, compute its ropes.
    for ci in 0u32..8u32 {
        if (mask & (1u32 << ci)) == 0 {
            continue;
        }

        let hx = ci & 1;
        let hy = (ci >> 1) & 1;
        let hz = (ci >> 2) & 1;

        let self_child = child_of_ci[ci as usize];
        debug_assert_ne!(self_child, INVALID_U32);

        let sib_x = ci ^ 1;
        let sib_y = ci ^ 2;
        let sib_z = ci ^ 4;

        // helper: sibling root if it exists
        let sib = |sci: u32| -> u32 {
            child_of_ci[sci as usize]
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
        let c = child_of_ci[ci as usize];
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
    edits: &[EditEntry],
) -> (Vec<NodeGpu>, Vec<u32>, Vec<NodeRopesGpu>, Vec<u32>, BuildTimingsMs) {
    let mut tim = BuildTimingsMs::default();
    let t_total = std::time::Instant::now();

    if should_cancel(cancel) {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new(), tim);
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
    let dirt_depth: i32 = 3 * vpm;

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

    time_it!(tim, height_cache, {
        scratch.ensure_height_cache(cache_w, cache_h);
        tim.cache_w = cache_w as u32;
        tim.cache_h = cache_h as u32;

        scratch
            .height_cache
            .par_chunks_mut(cache_w)
            .enumerate()
            .for_each(|(z, row)| {
                if (z & 15) == 0 && should_cancel(cancel) { return; }
                let wz = cache_z0 + z as i32;
                for x in 0..cache_w {
                    let wx = cache_x0 + x as i32;
                    row[x] = gen.ground_height(wx, wz);
                }
            });
    });


    if should_cancel(cancel) {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new(), tim);
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
    let (_tree_cache_unused, tree_mask) = time_it!(tim, tree_mask, {
        gen.build_tree_cache_with_mask(
            chunk_ox,
            chunk_oy,
            chunk_oz,
            cs_i,
            &height_at,
            cancel,
        )
    });


    // -------------------------------------------------------------------------
    // 2D maps (ground)
    // -------------------------------------------------------------------------
    time_it!(tim, ground_2d, {
        BuildScratch::ensure_2d(&mut scratch.ground, side, 0);

        scratch
            .ground
            .par_chunks_mut(side)
            .enumerate()
            .for_each(|(lz, row)| {
                if (lz & 15) == 0 && should_cancel(cancel) { return; }
                let wz = chunk_oz + lz as i32;
                for lx in 0..side {
                    let wx = chunk_ox + lx as i32;
                    row[lx] = height_at(wx, wz);
                }
            });
    });


    if should_cancel(cancel) {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new(), tim);
    }

    let ground_mip = time_it!(tim, ground_mip, {
        build_minmax_mip_inplace(
            &scratch.ground,
            cs_u,
            &mut scratch.ground_min_levels,
            &mut scratch.ground_max_levels,
        )
    });


    // -------------------------------------------------------------------------
    // Tree top stamp (2D)
    // -------------------------------------------------------------------------
    let mut cancelled = false;

    time_it!(tim, tree_top, {
        BuildScratch::ensure_2d(&mut scratch.tree_top, side, -1);

        let pad_m = 4;
        let xm0 = (chunk_ox.div_euclid(vpm)) - pad_m;
        let xm1 = ((chunk_ox + cs_i).div_euclid(vpm)) + pad_m;
        let zm0 = (chunk_oz.div_euclid(vpm)) - pad_m;
        let zm1 = ((chunk_oz + cs_i).div_euclid(vpm)) + pad_m;

        tim.tree_cells_tested = ((xm1 - xm0 + 1) * (zm1 - zm0 + 1)) as u32;

        'zm_loop: for zm in zm0..=zm1 {
            if ((zm - zm0) & 3) == 0 && should_cancel(cancel) {
                cancelled = true;
                break 'zm_loop;
            }

            for xm in xm0..=xm1 {
                let Some((trunk_h_vox, crown_r_vox)) = gen.tree_instance_at_meter(xm, zm) else {
                    continue;
                };

                // ... (rest unchanged)
            }
        }
    });

    if cancelled {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new(), tim);
    }


    let tree_mip = time_it!(tim, tree_mip, {
        build_max_mip_inplace(&scratch.tree_top, cs_u, &mut scratch.tree_levels)
    });


    // -------------------------------------------------------------------------
    // Precompute per-voxel material
    // -------------------------------------------------------------------------
    time_it!(tim, material_fill, {
        BuildScratch::ensure_3d_u32(&mut scratch.material, side, AIR);

        // NOTE: we will rebuild col_top_* AFTER edits are applied
        BuildScratch::ensure_coltop(&mut scratch.col_top_y, &mut scratch.col_top_mat, side);

        let vpm: i32 = config::VOXELS_PER_METER as i32;
        debug_assert!(vpm > 0);
        let dirt_depth: i32 = 3 * vpm;


        let side2 = side * side;

        // We compute per-thread column tops (packed u16) while filling material,
        // then reduce with elementwise max.
        // packed = ((y+1) << 8) | mat8, so higher y wins; 0 means empty.
        let coltops_packed: Vec<u16> = scratch
            .material
            .par_chunks_mut(side2)
            .enumerate()
            .fold(
                || vec![0u16; side * side],
                |mut local_tops, (ly, slab)| {
                    if (ly & 7) == 0 && should_cancel(cancel) {
                        return local_tops;
                    }

                    let wy = chunk_oy + ly as i32;

                    for lz in 0..side {
                        let wz = chunk_oz + lz as i32;
                        let row_off = lz * side;

                        for lx in 0..side {
                            let wx = chunk_ox + lx as i32;

                            let col = idx2(side, lx, lz);
                            let g = scratch.ground[col];

                            // --- 1) terrain ---
                            let mut m: u32 = if wy < g {
                                if wy >= g - dirt_depth { DIRT } else { STONE }
                            } else if wy == g {
                                GRASS
                            } else {
                                AIR
                            };

                            // --- 2) caves (with depth gate) ---
                            let max_depth_vox: i32 = (48.0_f32 * (vpm as f32)).round() as i32;
                            if m != AIR {
                                let depth_vox = g - wy;
                                if depth_vox > 0 && depth_vox <= max_depth_vox {
                                    if gen.carve_cave(wx, wy, wz, g) {
                                        m = AIR;
                                    }
                                }
                            }

                            // --- 3) trees overlay ---
                            if m == AIR {
                                let tm = tree_mask.material_local(lx, ly, lz);
                                if tm != AIR {
                                    m = tm;
                                }
                            }

                            slab[row_off + lx] = m;

                            // update local col top
                            if m != AIR {
                                let mat8 = (m & 0xFF) as u16;
                                let y8p1 = (ly as u16).wrapping_add(1); // 1..64, 0 reserved for empty
                                let packed = (y8p1 << 8) | mat8;
                                let cur = unsafe { *local_tops.get_unchecked(col) };
                                if packed > cur {
                                    unsafe { *local_tops.get_unchecked_mut(col) = packed; }
                                }
                            }
                        }
                    }

                    local_tops
                },
            )
            .reduce(
                || vec![0u16; side * side],
                |mut a, b| {
                    for i in 0..a.len() {
                        let bi = unsafe { *b.get_unchecked(i) };
                        let ai = unsafe { *a.get_unchecked(i) };
                        if bi > ai {
                            unsafe { *a.get_unchecked_mut(i) = bi; }
                        }
                    }
                    a
                },
            );

        // Decode into scratch.col_top_y / col_top_mat
        for i in 0..(side * side) {
            let p = coltops_packed[i];
            if p == 0 {
                scratch.col_top_y[i] = 255;
                scratch.col_top_mat[i] = 0;
            } else {
                let y = ((p >> 8) as u8).wrapping_sub(1);
                let m8 = (p & 0xFF) as u8;
                scratch.col_top_y[i] = y;
                scratch.col_top_mat[i] = m8;
            }
        }
    });


    // -------------------------------------------------------------------------
    // Column top map (64x64): per (x,z), store top-most non-air voxel (y, mat)
    // packed u16: (mat8<<8) | y8, y8=255 means empty column
    // 2 entries per u32 => 2048 u32 words
    // -------------------------------------------------------------------------
    let side_u = cs_u as usize;
    debug_assert_eq!(side_u, 64, "colinfo packing assumes chunk_size=64");

    let colinfo_words = time_it!(tim, colinfo, {
        let mut colinfo_words = vec![0u32; 2048];

        for lz in 0..64usize {
            for lx in 0..64usize {
                let col = idx2(64, lx, lz);

                let y8: u32 = scratch.col_top_y[col] as u32;
                let mat8: u32 = scratch.col_top_mat[col] as u32;

                let entry16: u32 = (y8 & 0xFF) | ((mat8 & 0xFF) << 8);

                let idx = (lz * 64 + lx) as u32;
                let w = (idx >> 1) as usize;
                let hi = (idx & 1) != 0;

                if !hi {
                    colinfo_words[w] = (colinfo_words[w] & 0xFFFF_0000) | entry16;
                } else {
                    colinfo_words[w] = (colinfo_words[w] & 0x0000_FFFF) | (entry16 << 16);
                }
            }
        }

        colinfo_words
    });



    // -------------------------------------------------------------------------
    // Prefix sum (summed-volume table) over solid occupancy
    // prefix[x,y,z] = sum of occupancy in [0..x)×[0..y)×[0..z)
    // -------------------------------------------------------------------------
    BuildScratch::ensure_prefix(&mut scratch.prefix, side);
    let dim = side + 1;
    let side_us = side;

    // material is read-only here (Sync), prefix is written via disjoint chunks
    let material: &[u32] = &scratch.material;

    // ---- Pass 1: write occupancy and prefix along X for each (y,z) row ----
    // We parallelize over x-rows of the 3D prefix buffer.
    // Layout: pidx(dim,x,y,z) = z*dim*dim + y*dim + x, so x is contiguous.
    time_it!(tim, prefix_x, {
        scratch
            .prefix
            .par_chunks_mut(dim)
            .enumerate()
            .for_each(|(row_idx, row)| {
                // row_idx corresponds to (z,y):
                let z = row_idx / dim;
                let y = row_idx % dim;

                // keep z=0 or y=0 boundary planes as 0
                if z == 0 || y == 0 || z > side_us || y > side_us {
                    return;
                }

                // optional cancel check: cheap + not too frequent
                if (z & 7) == 0 && should_cancel(cancel) {
                    return;
                }

                // row[0] stays 0
                let base_m = idx3(side_us, 0, y - 1, z - 1);

                let mut run: u32 = 0;
                for x in 1..=side_us {
                    // material index advances by +1 in x
                    let m = unsafe { *material.get_unchecked(base_m + (x - 1)) };
                    run += (m != AIR) as u32;
                    row[x] = run;
                }
            });
    });

    if should_cancel(cancel) {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new(), tim);
    }

    // ---- Pass 2: prefix along Y within each Z-plane ----
    // Each z-plane is a contiguous slab of size dim*dim.
    let plane = dim * dim;
    time_it!(tim, prefix_y, {
        scratch
            .prefix
            .par_chunks_mut(plane)
            .enumerate()
            .for_each(|(z, slab)| {
                if z == 0 || z > side_us {
                    return; // boundary plane remains 0
                }
                if (z & 7) == 0 && should_cancel(cancel) {
                    return;
                }

                // For each x, scan y = 1..=side; index in slab is (y*dim + x)
                for x in 1..=side_us {
                    let mut run: u32 = 0;
                    for y in 1..=side_us {
                        let idx = y * dim + x;
                        run += slab[idx];
                        slab[idx] = run;
                    }
                }
            });
    });

    if should_cancel(cancel) {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new(), tim);
    }

    // ---- Pass 3: prefix along Z ----
    // prefix[x,y,z] += prefix[x,y,z-1]
    let plane = dim * dim;

    // z=0 is the boundary plane (all zeros). Accumulate into planes 1..=side.
    let mut cancelled_prefix_z = false;

    time_it!(tim, prefix_z, {
        for z in 1..=side_us {
            if (z & 7) == 0 && should_cancel(cancel) {
                cancelled_prefix_z = true;
                break;
            }

            let (head, tail) = scratch.prefix.split_at_mut(z * plane);
            let prev = &head[(z - 1) * plane .. z * plane];
            let cur  = &mut tail[..plane];

            cur.par_iter_mut()
                .zip(prev.par_iter())
                .for_each(|(c, p)| {
                    *c += *p;
                });
        }
    });

    if cancelled_prefix_z {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new(), tim);
    }



    // -------------------------------------------------------------------------
    // Macro occupancy bitset (8x8x8 => 512 bits => 16 u32 words)
    // -------------------------------------------------------------------------
    let macro_dim: usize = 8;
    debug_assert_eq!(side % macro_dim, 0);

    let cell: usize = side / macro_dim;

    let mut cancelled_macro = false;

    let macro_words = time_it!(tim, macro_occ, {
        let mut macro_words = vec![0u32; MACRO_WORDS_PER_CHUNK_USIZE];

        'mz: for mz in 0..macro_dim {
            if should_cancel(cancel) {
                cancelled_macro = true;
                break 'mz;
            }
            for my in 0..macro_dim {
                for mx in 0..macro_dim {
                    let x0 = mx * cell;
                    let y0 = my * cell;
                    let z0 = mz * cell;

                    let sum = prefix_sum_cube(&scratch.prefix, side, x0, y0, z0, cell);
                    if sum > 0 {
                        let bit = mx + macro_dim * (my + macro_dim * mz);
                        let w = bit >> 5;
                        let b = bit & 31;
                        macro_words[w] |= 1u32 << b;
                    }
                }
            }
        }

        macro_words
    });

    if cancelled_macro {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new(), tim);
    }


    fn make_leaf(chunk_size: u32, ox: i32, oy: i32, oz: i32, size: i32, mat: u32) -> NodeGpu {
        NodeGpu {
            child_base: LEAF,
            child_mask: 0,
            material: mat,
            key: pack_key(chunk_size, ox, oy, oz, size),
        }
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
        child_idx_grid: &[u32],    // dense grid: child cell -> index in child_nodes (or INVALID_U32)
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

                    let c_lin = idx3(child_side, cx, cy, cz);
                    let c_idx = unsafe { *child_idx_grid.get_unchecked(c_lin) };
                    if c_idx != INVALID_U32 {
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
        let mut parent_idx_grid = vec![INVALID_U32; parent_cells];
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
            debug_assert!(p_out != INVALID_U32);

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

                let c_lin = idx3(child_side, cx, cy, cz);
                let c_idx = unsafe { *child_idx_grid.get_unchecked(c_lin) };
                debug_assert!(c_idx != INVALID_U32);

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

    fn build_svo_bottom_up(
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
        let mut pos = vec![INVALID_U32; n];
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
        let mut leaf_idx_grid = vec![INVALID_U32; n];

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


        // packed_children_levels[l] stores nodes at level l (children of level l-1), packed by parents.
        // We'll fill levels 1..=max_lvl. Level 0 is the root node (separate).
        let mut packed_children_levels: Vec<Vec<NodeGpu>> = vec![Vec::new(); max_lvl + 1];

        // Current child representation used to *compute* the next level up:
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

            // packed children are the nodes at the *child level* of these parents:
            // i.e. parent level is one above current child level, so store in slot corresponding
            // to the current child level size.
            //
            // child_size: 1 -> this packed is level 6 (voxels)
            // child_size: 2 -> packed is level 5
            // ...
            // We can derive level index by: level = (chunk_size / child_size).trailing_zeros()
            let child_level = (chunk_size / (child_size as u32)).trailing_zeros() as usize;
            debug_assert!(child_level <= max_lvl);
            packed_children_levels[child_level] = packed_children;

            // Move up one level
            child_nodes = parent_nodes;
            child_idx_grid = parent_idx_grid;
            child_side = parent_side;
            child_size = parent_size;
        }

        // After the loop, child_nodes is the root-level compact list (either empty or 1 node)
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

    // Bottom-up build (levels) => nodes[0] is root, children packed per parent.
    let nodes = time_it!(tim, svo_build, {
        build_svo_bottom_up(
            cs_u,
            chunk_oy,
            &scratch.material,
            &scratch.prefix,
            &ground_mip,
            dirt_depth,
            cancel,
        )
    });

    let ropes = time_it!(tim, ropes, {
        build_ropes(&nodes)
    });

    tim.nodes = nodes.len() as u32;
    tim.total = ms_since(t_total);

    return (nodes, macro_words, ropes, colinfo_words, tim);


}
