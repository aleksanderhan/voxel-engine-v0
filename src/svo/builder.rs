use crate::{
    config,
    render::NodeGpu,
    world::{
        materials::{AIR, STONE},
        WorldGen,
    },
};

use super::{
    height_cache::HeightCache,
    mips::{build_max_mip, build_minmax_mip, MaxMip, MinMaxMip},
};

const LEAF: u32 = 0xFFFF_FFFF;

#[inline]
fn is_empty_leaf(n: &NodeGpu) -> bool {
    n.child_base == LEAF && n.material == AIR
}

/// Builds a single chunk SVO with sparse (compact) child allocation.
///
/// - `chunk_origin`: world voxel coordinates of the chunk min corner.
/// - `chunk_size`: must be power of two (balanced octree).
pub fn build_chunk_svo_sparse(gen: &WorldGen, chunk_origin: [i32; 3], chunk_size: u32) -> Vec<NodeGpu> {
    let chunk_ox = chunk_origin[0];
    let chunk_oy = chunk_origin[1];
    let chunk_oz = chunk_origin[2];

    let cs_u = chunk_size;
    let cs_i = chunk_size as i32;

    let vpm = config::VOXELS_PER_METER;
    debug_assert!(vpm > 0);

    let margin_m: i32 = 6;
    let margin = margin_m * vpm + (vpm - 1);

    let cache = HeightCache::new(
        gen,
        chunk_ox - margin,
        chunk_oz - margin,
        chunk_ox + cs_i + margin + 1,
        chunk_oz + cs_i + margin + 1,
    );
    let height_at = |wx: i32, wz: i32| -> i32 { cache.get(wx, wz) };

    let tree_cache = gen.build_tree_cache(chunk_ox, chunk_oz, cs_i, &height_at);

    let mat_at_local = |lx: i32, ly: i32, lz: i32| -> u32 {
        let wx = chunk_ox + lx;
        let wy = chunk_oy + ly;
        let wz = chunk_oz + lz;
        gen.material_at_world_cached_with_trees(wx, wy, wz, &height_at, &tree_cache)
    };

    // Ground height map over chunk footprint
    let side = cs_u as usize;
    let mut ground = vec![0i32; side * side];
    for lz in 0..cs_i {
        for lx in 0..cs_i {
            let wx = chunk_ox + lx;
            let wz = chunk_oz + lz;
            ground[(lz as usize) * side + (lx as usize)] = height_at(wx, wz);
        }
    }
    let ground_mip = build_minmax_mip(&ground, cs_u);

    // Tree top height map over chunk footprint
    let mut tree_top = vec![-1i32; side * side];

    let pad_m = 4;
    let xm0 = (chunk_ox.div_euclid(vpm)) - pad_m;
    let xm1 = ((chunk_ox + cs_i).div_euclid(vpm)) + pad_m;
    let zm0 = (chunk_oz.div_euclid(vpm)) - pad_m;
    let zm1 = ((chunk_oz + cs_i).div_euclid(vpm)) + pad_m;

    for zm in zm0..=zm1 {
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
                        let idx = (lz as usize) * side + (lx as usize);
                        tree_top[idx] = tree_top[idx].max(top_y);
                    }
                }
            }

            let lx = tx - chunk_ox;
            let lz = tz - chunk_oz;
            if lx >= 0 && lx < cs_i && lz >= 0 && lz < cs_i {
                let idx = (lz as usize) * side + (lx as usize);
                tree_top[idx] = tree_top[idx].max(trunk_top);
            }
        }
    }

    let tree_mip = build_max_mip(&tree_top, cs_u);

    let dirt_depth = 3 * vpm;

    fn build_node(
        nodes: &mut Vec<NodeGpu>,
        ox: i32,
        oy: i32,
        oz: i32,
        size: i32,
        chunk_oy: i32,
        mat_fn: &dyn Fn(i32, i32, i32) -> u32,
        ground_mip: &MinMaxMip,
        tree_mip: &MaxMip,
        dirt_depth: i32,
    ) -> NodeGpu {
        debug_assert!(size > 0);
        let size_u = size as u32;

        let (gmin, gmax) = ground_mip.query(ox, oz, size_u);
        let tmax = tree_mip.query_max(ox, oz, size_u);

        let y0 = chunk_oy + oy;
        let y1 = y0 + size - 1;

        let top_solid = gmax.max(tmax);
        if y0 > top_solid {
            return NodeGpu { child_base: LEAF, child_mask: 0, material: AIR, _pad: 0 };
        }

        if y1 < gmin - dirt_depth {
            return NodeGpu { child_base: LEAF, child_mask: 0, material: STONE, _pad: 0 };
        }

        if size == 1 {
            let m = mat_fn(ox, oy, oz);
            return NodeGpu { child_base: LEAF, child_mask: 0, material: m, _pad: 0 };
        }

        let half = size / 2;
        let mut child_roots: [NodeGpu; 8] =
            [NodeGpu { child_base: LEAF, child_mask: 0, material: AIR, _pad: 0 }; 8];

        for ci in 0..8 {
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
                mat_fn,
                ground_mip,
                tree_mip,
                dirt_depth,
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

    let mut nodes = vec![NodeGpu { child_base: LEAF, child_mask: 0, material: AIR, _pad: 0 }];

    let root = build_node(
        &mut nodes,
        0,
        0,
        0,
        cs_i,
        chunk_oy,
        &mat_at_local,
        &ground_mip,
        &tree_mip,
        dirt_depth,
    );
    nodes[0] = root;

    nodes
}
