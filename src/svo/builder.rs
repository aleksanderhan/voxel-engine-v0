// src/svo/builder.rs

use std::sync::atomic::{AtomicBool, Ordering};

use crate::{
    config,
    render::NodeGpu,
    world::{
        materials::{AIR, DIRT, GRASS, STONE, WOOD},
        WorldGen,
    },
};

use super::mips::{build_max_mip_inplace, build_minmax_mip_inplace, MaxMipView, MinMaxMipView};

const LEAF: u32 = 0xFFFF_FFFF;

const MACRO_WORDS_PER_CHUNK_USIZE: usize = 16;


#[inline]
fn is_empty_leaf(n: &NodeGpu) -> bool {
    n.child_base == LEAF && n.material == AIR
}

#[inline]
fn should_cancel(cancel: &AtomicBool) -> bool {
    cancel.load(Ordering::Relaxed)
}


/// Reusable scratch buffers for chunk building (reduces allocations & improves locality).
pub struct BuildScratch {
    // 2D (side*side)
    ground: Vec<i32>,
    tree_top: Vec<i32>,

    // --- add this ---
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

/// Cancelable build with reusable scratch (fast path).
pub fn build_chunk_svo_sparse_cancelable_with_scratch(
    gen: &WorldGen,
    chunk_origin: [i32; 3],
    chunk_size: u32,
    cancel: &AtomicBool,
    scratch: &mut BuildScratch,
) -> (Vec<NodeGpu>, Vec<u32>) {
    if should_cancel(cancel) {
        return (Vec::new(), Vec::new());
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
    // Height cache (local array)
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
            return (Vec::new(), Vec::new());
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
    // Tree cache/mask (fast O(1) material lookup)
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
            return (Vec::new(), Vec::new());
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
    // Tree top stamp (2D) for fast “above everything” pruning
    // -------------------------------------------------------------------------
    BuildScratch::ensure_2d(&mut scratch.tree_top, side, -1);

    let pad_m = 4;
    let xm0 = (chunk_ox.div_euclid(vpm)) - pad_m;
    let xm1 = ((chunk_ox + cs_i).div_euclid(vpm)) + pad_m;
    let zm0 = (chunk_oz.div_euclid(vpm)) - pad_m;
    let zm1 = ((chunk_oz + cs_i).div_euclid(vpm)) + pad_m;

    for zm in zm0..=zm1 {
        if ((zm - zm0) & 3) == 0 && should_cancel(cancel) {
            return (Vec::new(), Vec::new());
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
    // Precompute per-voxel material + occupancy (terrain + trees only)
    // -------------------------------------------------------------------------
    BuildScratch::ensure_3d_u32(&mut scratch.material, side, AIR);

    let dirt_depth = 3 * vpm;

    for ly in 0..cs_i {
        if (ly & 7) == 0 && should_cancel(cancel) {
            return (Vec::new(), Vec::new());
        }

        let wy = chunk_oy + ly;

        for lz in 0..cs_i {
            for lx in 0..cs_i {
                let col = idx2(side, lx as usize, lz as usize);
                let g = scratch.ground[col];

                // base terrain + trees (FAST)
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
            return (Vec::new(), Vec::new());
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
        if should_cancel(cancel) { return (Vec::new(), Vec::new()); }
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


    fn build_node(
        nodes: &mut Vec<NodeGpu>,
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
            return NodeGpu { child_base: LEAF, child_mask: 0, material: AIR, _pad: 0 };
        }

        let size_u = size as u32;

        let (gmin, gmax) = ground_mip.query(ox, oz, size_u);
        let tmax = tree_mip.query_max(ox, oz, size_u);

        let y0 = chunk_oy + oy;
        let y1 = y0 + size - 1;

        // above everything (ground or tree top)
        let top_solid = gmax.max(tmax);
        if y0 > top_solid {
            return NodeGpu { child_base: LEAF, child_mask: 0, material: AIR, _pad: 0 };
        }

        // deep solid stone (below dirt band everywhere in this node footprint)
        if y1 < gmin - dirt_depth {
            return NodeGpu { child_base: LEAF, child_mask: 0, material: STONE, _pad: 0 };
        }

        // empty check via prefix
        let sx = ox as usize;
        let sy = oy as usize;
        let sz = oz as usize;
        let s = size as usize;

        let sum = prefix_sum_cube(prefix, side, sx, sy, sz, s);
        if sum == 0 {
            return NodeGpu { child_base: LEAF, child_mask: 0, material: AIR, _pad: 0 };
        }

        if size == 1 {
            let m = material[idx3(side, sx, sy, sz)];
            return NodeGpu { child_base: LEAF, child_mask: 0, material: m, _pad: 0 };
        }

        let half = size / 2;
        let mut child_roots: [NodeGpu; 8] =
            [NodeGpu { child_base: LEAF, child_mask: 0, material: AIR, _pad: 0 }; 8];

        for ci in 0..8 {
            if (ci & 3) == 0 && should_cancel(cancel) {
                return NodeGpu { child_base: LEAF, child_mask: 0, material: AIR, _pad: 0 };
            }

            let dx = if (ci & 1) != 0 { half } else { 0 };
            let dy = if (ci & 2) != 0 { half } else { 0 };
            let dz = if (ci & 4) != 0 { half } else { 0 };

            child_roots[ci] = build_node(
                nodes,
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
        }

        let base = nodes.len() as u32;
        let mut mask: u32 = 0;

        for ci in 0..8 {
            if !is_empty_leaf(&child_roots[ci]) {
                mask |= 1u32 << ci;
                nodes.push(child_roots[ci]);
            }
        }

        if mask == 0 {
            return NodeGpu { child_base: LEAF, child_mask: 0, material: AIR, _pad: 0 };
        }

        NodeGpu { child_base: base, child_mask: mask, material: 0, _pad: 0 }
    }

    // Root must be at index 0 for GPU.
    let mut nodes = vec![NodeGpu { child_base: LEAF, child_mask: 0, material: AIR, _pad: 0 }];

    let root = build_node(
        &mut nodes,
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
        return (Vec::new(), Vec::new());
    }

    nodes[0] = root;
    (nodes, macro_words)
}
