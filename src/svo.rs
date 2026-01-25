// svo.rs
use bytemuck::{Pod, Zeroable};

use crate::worldgen::WorldGen;

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

struct HeightCache {
    ox: i32,
    oz: i32,
    w: i32,
    data: Vec<i32>, // row-major: z * w + x
}

impl HeightCache {
    fn new(gen: &WorldGen, wx0: i32, wz0: i32, wx1: i32, wz1: i32) -> Self {
        let w = wx1 - wx0;
        let h = wz1 - wz0;
        let mut data = vec![0i32; (w * h) as usize];

        for z in 0..h {
            let wz = wz0 + z;
            for x in 0..w {
                let wx = wx0 + x;
                data[(z * w + x) as usize] = gen.ground_height(wx, wz);
            }
        }

        Self { ox: wx0, oz: wz0, w, data }
    }

    #[inline]
    fn get(&self, x: i32, z: i32) -> i32 {
        let ix = x - self.ox;
        let iz = z - self.oz;
        debug_assert!(ix >= 0 && ix < self.w);
        debug_assert!(iz >= 0);
        self.data[(iz * self.w + ix) as usize]
    }
}

/// Builds a single chunk SVO with sparse (compact) child allocation.
///
/// - `chunk_origin`: world voxel coordinates of the chunk min corner.
/// - `chunk_size`: must be power of two (balanced octree).
pub fn build_chunk_svo_sparse(
    gen: &WorldGen,
    chunk_origin: [i32; 3],
    chunk_size: u32,
) -> Vec<NodeGpu> {
    let ox = chunk_origin[0];
    let oy = chunk_origin[1];
    let oz = chunk_origin[2];
    let cs = chunk_size as i32;

    // Trees need neighborhood height lookups up to +/-3 in x/z.
    let margin = 3;
    let cache = HeightCache::new(gen, ox - margin, oz - margin, ox + cs + margin, oz + cs + margin);
    let height_at = |x: i32, z: i32| -> i32 { cache.get(x, z) };

    let mat_at_local = |lx: i32, ly: i32, lz: i32| -> u32 {
        let wx = ox + lx;
        let wy = oy + ly;
        let wz = oz + lz;
        gen.material_at_world_cached(wx, wy, wz, &height_at)
    };

    fn is_empty_leaf(n: &NodeGpu) -> bool {
        n.child_base == LEAF && n.material == 0
    }

    fn build_node(
        nodes: &mut Vec<NodeGpu>,
        ox: i32,
        oy: i32,
        oz: i32,
        size: i32,
        mat_fn: &dyn Fn(i32, i32, i32) -> u32,
    ) -> NodeGpu {
        // Uniformity test: if all voxels in this region share the same material, make a leaf.
        let first = mat_fn(ox, oy, oz);
        let mut uniform = true;

        'outer: for z in oz..(oz + size) {
            for y in oy..(oy + size) {
                for x in ox..(ox + size) {
                    if mat_fn(x, y, z) != first {
                        uniform = false;
                        break 'outer;
                    }
                }
            }
        }

        if uniform || size == 1 {
            return NodeGpu { child_base: LEAF, child_mask: 0, material: first, _pad: 0 };
        }

        let half = size / 2;

        // Build 8 child roots (their subtrees get appended during recursion).
        let mut child_roots: [NodeGpu; 8] = [NodeGpu { child_base: LEAF, child_mask: 0, material: 0, _pad: 0 }; 8];

        for ci in 0..8 {
            let dx = if (ci & 1) != 0 { half } else { 0 };
            let dy = if (ci & 2) != 0 { half } else { 0 };
            let dz = if (ci & 4) != 0 { half } else { 0 };

            child_roots[ci] = build_node(nodes, ox + dx, oy + dy, oz + dz, half, mat_fn);
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

        // If everything collapsed to empty, return empty leaf.
        if mask == 0 {
            return NodeGpu { child_base: LEAF, child_mask: 0, material: 0, _pad: 0 };
        }

        NodeGpu { child_base: base, child_mask: mask, material: 0, _pad: 0 }
    }

    // Root must be at index 0 for GPU.
    let mut nodes = vec![NodeGpu { child_base: LEAF, child_mask: 0, material: 0, _pad: 0 }];

    let root = build_node(&mut nodes, 0, 0, 0, cs, &mat_at_local);
    nodes[0] = root;
    nodes
}
