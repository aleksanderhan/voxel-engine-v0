// svo.rs
//
// Fast chunk SVO (Sparse Voxel Octree) builder for heightfield + trees worldgen.
//
// Key change vs a naive builder:
// - NO 3D “uniformity scan” over every voxel in a node.
// - Uses 2D min/max mips of ground height + a max mip of “tree-top height” to early-out:
//   - Entire node above (terrain OR trees) => AIR leaf
//   - Entire node deep below terrain everywhere => STONE leaf
// - Only does exact material queries at voxel resolution (size == 1).
//

use bytemuck::{Pod, Zeroable};

use crate::worldgen::{WorldGen, AIR, STONE};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct NodeGpu {
    /// If internal: index of first child in compact list. If leaf: 0xFFFF_FFFF.
    pub child_base: u32,
    /// Bitmask for which children exist (bits 0..7).
    pub child_mask: u32,
    /// Material id for leaf nodes (0 = empty / air).
    pub material: u32,
    /// Padding for 16-byte alignment.
    pub _pad: u32,
}

const LEAF: u32 = 0xFFFF_FFFF;

#[inline]
fn is_empty_leaf(n: &NodeGpu) -> bool {
    n.child_base == LEAF && n.material == AIR
}

// ------------------------------------------------------------
// Height cache (world-space y heights by xz)
// ------------------------------------------------------------

struct HeightCache {
    ox: i32,
    oz: i32,
    w: i32,
    h: i32,
    data: Vec<i32>,
}

impl HeightCache {
    /// Builds cache over [wx0..wx1) x [wz0..wz1) in WORLD voxel coords.
    fn new(gen: &WorldGen, wx0: i32, wz0: i32, wx1: i32, wz1: i32) -> Self {
        let w = wx1 - wx0;
        let h = wz1 - wz0;
        debug_assert!(w > 0 && h > 0);

        let mut data = vec![0i32; (w * h) as usize];

        for z in 0..h {
            let wz = wz0 + z;
            for x in 0..w {
                let wx = wx0 + x;
                data[(z * w + x) as usize] = gen.ground_height(wx, wz);
            }
        }

        Self { ox: wx0, oz: wz0, w, h, data }
    }

    #[inline]
    fn get(&self, x: i32, z: i32) -> i32 {
        let ix = x - self.ox;
        let iz = z - self.oz;
        debug_assert!(ix >= 0 && ix < self.w);
        debug_assert!(iz >= 0 && iz < self.h);
        self.data[(iz * self.w + ix) as usize]
    }
}

// ------------------------------------------------------------
// 2D mips for fast min/max queries over XZ tiles
// ------------------------------------------------------------

struct MinMaxMip {
    root_side: u32, // e.g. 128
    min_levels: Vec<Vec<i32>>,
    max_levels: Vec<Vec<i32>>,
}

fn build_minmax_mip(base: &[i32], side: u32) -> MinMaxMip {
    debug_assert!(side.is_power_of_two());
    debug_assert_eq!(base.len(), (side * side) as usize);

    let mut min_levels = Vec::new();
    let mut max_levels = Vec::new();

    min_levels.push(base.to_vec());
    max_levels.push(base.to_vec());

    let mut cur_side = side;
    while cur_side > 1 {
        let next_side = cur_side / 2;
        let mut mn = vec![0i32; (next_side * next_side) as usize];
        let mut mx = vec![0i32; (next_side * next_side) as usize];

        let cur_min = min_levels.last().unwrap();
        let cur_max = max_levels.last().unwrap();

        for z in 0..next_side {
            for x in 0..next_side {
                let i00 = ((2 * z) * cur_side + (2 * x)) as usize;
                let i10 = ((2 * z) * cur_side + (2 * x + 1)) as usize;
                let i01 = ((2 * z + 1) * cur_side + (2 * x)) as usize;
                let i11 = ((2 * z + 1) * cur_side + (2 * x + 1)) as usize;

                let o = (z * next_side + x) as usize;

                let a0 = cur_min[i00];
                let a1 = cur_min[i10];
                let a2 = cur_min[i01];
                let a3 = cur_min[i11];

                let b0 = cur_max[i00];
                let b1 = cur_max[i10];
                let b2 = cur_max[i01];
                let b3 = cur_max[i11];

                mn[o] = a0.min(a1).min(a2).min(a3);
                mx[o] = b0.max(b1).max(b2).max(b3);
            }
        }

        min_levels.push(mn);
        max_levels.push(mx);
        cur_side = next_side;
    }

    MinMaxMip { root_side: side, min_levels, max_levels }
}

impl MinMaxMip {
    /// Query min/max over aligned tile [x0..x0+size) x [z0..z0+size)
    /// where `size` is power-of-two voxel width (1,2,...,root_side).
    #[inline]
    fn query(&self, x0: i32, z0: i32, size: u32) -> (i32, i32) {
        debug_assert!(size.is_power_of_two());
        debug_assert!(size <= self.root_side);
        debug_assert!(x0 >= 0 && z0 >= 0);

        // level: size=1 -> 0, 2 -> 1, ..., root_side -> log2(root_side)
        let level = size.trailing_zeros() as usize;
        debug_assert!(level < self.min_levels.len());

        let side = self.root_side >> level; // number of tiles along an axis
        let x = (x0 as u32) / size;
        let z = (z0 as u32) / size;
        let idx = (z * side + x) as usize;

        (self.min_levels[level][idx], self.max_levels[level][idx])
    }
}

struct MaxMip {
    root_side: u32,
    levels: Vec<Vec<i32>>,
}

fn build_max_mip(base: &[i32], side: u32) -> MaxMip {
    debug_assert!(side.is_power_of_two());
    debug_assert_eq!(base.len(), (side * side) as usize);

    let mut levels = Vec::new();
    levels.push(base.to_vec());

    let mut cur_side = side;
    while cur_side > 1 {
        let next_side = cur_side / 2;
        let mut mx = vec![0i32; (next_side * next_side) as usize];

        let cur = levels.last().unwrap();
        for z in 0..next_side {
            for x in 0..next_side {
                let i00 = ((2 * z) * cur_side + (2 * x)) as usize;
                let i10 = ((2 * z) * cur_side + (2 * x + 1)) as usize;
                let i01 = ((2 * z + 1) * cur_side + (2 * x)) as usize;
                let i11 = ((2 * z + 1) * cur_side + (2 * x + 1)) as usize;

                let o = (z * next_side + x) as usize;
                mx[o] = cur[i00].max(cur[i10]).max(cur[i01]).max(cur[i11]);
            }
        }

        levels.push(mx);
        cur_side = next_side;
    }

    MaxMip { root_side: side, levels }
}

impl MaxMip {
    /// Query max over aligned tile [x0..x0+size) x [z0..z0+size)
    #[inline]
    fn query_max(&self, x0: i32, z0: i32, size: u32) -> i32 {
        debug_assert!(size.is_power_of_two());
        debug_assert!(size <= self.root_side);
        debug_assert!(x0 >= 0 && z0 >= 0);

        let level = size.trailing_zeros() as usize;
        debug_assert!(level < self.levels.len());

        let side = self.root_side >> level;
        let x = (x0 as u32) / size;
        let z = (z0 as u32) / size;
        let idx = (z * side + x) as usize;

        self.levels[level][idx]
    }
}

// ------------------------------------------------------------
// SVO builder
// ------------------------------------------------------------

/// Builds a single chunk SVO with sparse (compact) child allocation.
///
/// - `chunk_origin`: world voxel coordinates of the chunk min corner.
/// - `chunk_size`: must be power of two (balanced octree).
pub fn build_chunk_svo_sparse(
    gen: &WorldGen,
    chunk_origin: [i32; 3],
    chunk_size: u32,
) -> Vec<NodeGpu> {
    let chunk_ox = chunk_origin[0];
    let chunk_oy = chunk_origin[1];
    let chunk_oz = chunk_origin[2];

    let cs_u = chunk_size;
    let cs_i = chunk_size as i32;

    // Height cache must cover tree queries in a neighborhood.
    // With trees defined on a 1m grid and crown radius up to ~3m,
    // margin 4m worth of voxels is a safe bound.
    let vpm = crate::worldgen::VOXELS_PER_METER;
    debug_assert!(vpm > 0);

    // Worldgen scans +/-4 meter cells and meter-grid alignment can add up to (vpm-1) extra.
    // Add a little headroom for branches/canopy.
    let margin_m: i32 = 6; // meters
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

    // Exact material query (only used at voxel resolution).
    let mat_at_local = |lx: i32, ly: i32, lz: i32| -> u32 {
        let wx = chunk_ox + lx;
        let wy = chunk_oy + ly;
        let wz = chunk_oz + lz;
        gen.material_at_world_cached_with_trees(wx, wy, wz, &height_at, &tree_cache)
    };

    // --------------------------------------------------------
    // Ground height map over chunk footprint
    // --------------------------------------------------------
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

    // --------------------------------------------------------
    // Tree top height map over chunk footprint
    // Value is max world-y (in voxels) occupied by any tree in that column.
    // -1 means no tree influence.
    // --------------------------------------------------------
    let mut tree_top = vec![-1i32; side * side];

    // Iterate meter-cells around this chunk (+/-4m).
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

            let tx = xm * vpm; // world voxel x of trunk
            let tz = zm * vpm; // world voxel z of trunk

            let g = height_at(tx, tz);
            let trunk_base = g + vpm; // 1m above ground
            let trunk_top = trunk_base + trunk_h_vox;
            // Must be CONSERVATIVE for SVO early-outs, otherwise leaves flicker/missing.
            // canopy_h in your worldgen is ~2.5..4.5m, so approximate conservatively here.
            let canopy_h_vox = (5 * vpm); // ~5m conservative cap
            let top_y = trunk_top + canopy_h_vox + 2 * vpm;

            // canopy clumps can push a bit wider than crown_r; add margin
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

            // Ensure trunk column is included even if crown is tiny.
            let lx = tx - chunk_ox;
            let lz = tz - chunk_oz;
            if lx >= 0 && lx < cs_i && lz >= 0 && lz < cs_i {
                let idx = (lz as usize) * side + (lx as usize);
                tree_top[idx] = tree_top[idx].max(trunk_top);
            }
        }
    }

    let tree_mip = build_max_mip(&tree_top, cs_u);

    // Dirt depth in voxels (top 3m)
    let dirt_depth = 3 * vpm;

    // --------------------------------------------------------
    // Recursive build with fast early-outs
    // --------------------------------------------------------
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

        // Fast XZ min/max queries for this node footprint.
        let (gmin, gmax) = ground_mip.query(ox, oz, size_u);
        let tmax = tree_mip.query_max(ox, oz, size_u); // -1 if no trees

        // Node y-range in WORLD voxel coords
        let y0 = chunk_oy + oy;
        let y1 = y0 + size - 1;

        // 1) Entirely above both terrain and trees => AIR
        let top_solid = gmax.max(tmax);
        if y0 > top_solid {
            return NodeGpu { child_base: LEAF, child_mask: 0, material: AIR, _pad: 0 };
        }

        // 2) Entirely deep underground everywhere => STONE
        if y1 < gmin - dirt_depth {
            return NodeGpu { child_base: LEAF, child_mask: 0, material: STONE, _pad: 0 };
        }

        // 3) At voxel resolution: exact material
        if size == 1 {
            let m = mat_fn(ox, oy, oz);
            return NodeGpu { child_base: LEAF, child_mask: 0, material: m, _pad: 0 };
        }

        // Subdivide
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

        // Compact children: append only non-empty child roots contiguously in increasing `ci`.
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
        &mat_at_local,
        &ground_mip,
        &tree_mip,
        dirt_depth,
    );
    nodes[0] = root;

    nodes
}
