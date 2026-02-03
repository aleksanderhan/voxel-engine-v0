// src/svo/builder.rs
//
// SVO = Sparse Voxel Octree.
// mip = mip levels / mipmaps (downsampled pyramid).
// CPU = central processing unit.
// VPM = voxels per meter.

use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;

use crate::app::config;
use crate::{
    render::gpu_types::{NodeGpu, NodeRopesGpu},
    world::{
        edits::EditEntry,
        materials::{AIR, DIRT, GRASS, STONE},
        WorldGen,
    },
};

use super::mips::{build_max_mip_inplace, build_minmax_mip_inplace};

use super::builder_prefix as prefix;
use super::builder_ropes as ropes;
use super::builder_svo as svo;

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

// -----------------------------------------------------------------------------
// Cancellation helpers
// -----------------------------------------------------------------------------

#[inline]
fn should_cancel(cancel: &AtomicBool) -> bool {
    cancel.load(Ordering::Relaxed)
}

macro_rules! cancel_if {
    ($cancel:expr, $tim:expr) => {{
        if should_cancel($cancel) {
            return BuildOutput::cancelled($tim);
        }
    }};
}

// -----------------------------------------------------------------------------
// Output type (replaces the 5-tuple)
// -----------------------------------------------------------------------------

pub struct BuildOutput {
    pub nodes: Vec<NodeGpu>,
    pub macro_words: Vec<u32>,
    pub ropes: Vec<NodeRopesGpu>,
    pub colinfo_words: Vec<u32>,
    pub timings: BuildTimingsMs,
}

impl BuildOutput {
    #[inline]
    fn cancelled(timings: BuildTimingsMs) -> Self {
        Self {
            nodes: Vec::new(),
            macro_words: Vec::new(),
            ropes: Vec::new(),
            colinfo_words: Vec::new(),
            timings,
        }
    }
}

// -----------------------------------------------------------------------------
// Indexing helpers (clear naming)
// -----------------------------------------------------------------------------

#[inline]
fn idx_xz(side: usize, x: usize, z: usize) -> usize {
    z * side + x
}

// -----------------------------------------------------------------------------
// Chunk context (single source of truth)
// -----------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct ChunkCtx {
    ox: i32,
    oy: i32,
    oz: i32,
    size_u: u32,
    size_i: i32,
    side: usize,
    vpm: i32,        // voxels per meter (VPM)
    dirt_depth: i32,
}

impl ChunkCtx {
    #[inline]
    fn new(chunk_origin: [i32; 3], chunk_size: u32) -> Self {
        let ox = chunk_origin[0];
        let oy = chunk_origin[1];
        let oz = chunk_origin[2];

        let size_u = chunk_size;
        let size_i = chunk_size as i32;
        let side = size_u as usize;

        debug_assert!(size_u.is_power_of_two());

        let vpm = config::VOXELS_PER_METER as i32;
        debug_assert!(vpm > 0);
        let dirt_depth = 3 * vpm;

        Self {
            ox,
            oy,
            oz,
            size_u,
            size_i,
            side,
            vpm,
            dirt_depth,
        }
    }
}

// -----------------------------------------------------------------------------
// Height sampling (cache-backed)
// -----------------------------------------------------------------------------

struct HeightSampler<'a> {
    gen: &'a WorldGen,
    cache_ptr: *const i32,
    cache_w: usize,
    x0: i32,
    x1: i32,
    z0: i32,
    z1: i32,
}

unsafe impl<'a> Sync for HeightSampler<'a> {}

impl<'a> HeightSampler<'a> {
    #[inline]
    fn at(&self, wx: i32, wz: i32) -> i32 {
        if wx < self.x0 || wx > self.x1 || wz < self.z0 || wz > self.z1 {
            self.gen.ground_height(wx, wz)
        } else {
            let ix = (wx - self.x0) as usize;
            let iz = (wz - self.z0) as usize;
            let idx = iz * self.cache_w + ix;
            unsafe { *self.cache_ptr.add(idx) }
        }
    }
}


// -----------------------------------------------------------------------------
// Reusable scratch buffers for chunk building (reduces allocations & improves locality).
// -----------------------------------------------------------------------------

pub struct BuildScratch {
    // 2D (side*side)
    ground: Vec<i32>,
    tree_top: Vec<i32>,

    // height cache
    height_cache: Vec<i32>,
    height_cache_w: usize,
    height_cache_h: usize,

    // 3D (side^3)
    material: Vec<u8>,
    prefix: Vec<u32>,

    // mip storage
    ground_min_levels: Vec<Vec<i32>>,
    ground_max_levels: Vec<Vec<i32>>,
    tree_levels: Vec<Vec<i32>>,

    // per-column top-most non-air (y, mat)
    col_top_y: Vec<u8>,   // 255 = empty
    col_top_mat: Vec<u8>, // low 8 bits of material id

    height_cache_x0: i32,
    height_cache_z0: i32,
    height_cache_valid: bool,

    cave_mask: Vec<u8>,   // 0/1 samples
    cave_shift: u32,     // e.g. 4
    cave_dim: usize,      // side / step

    svo: svo::SvoScratch,
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
            height_cache_x0: 0,
            height_cache_z0: 0,
            height_cache_valid: false,
            cave_mask: Vec::new(),
            cave_shift: 0,
            cave_dim: 0,
            svo: svo::SvoScratch::new(),
        }
    }

    #[inline]
    fn ensure_coltop(&mut self, side: usize) {
        let need = side * side;
        if self.col_top_y.len() != need {
            self.col_top_y.resize(need, 255);
            self.col_top_mat.resize(need, 0);
        } else {
            self.col_top_y.fill(255);
            self.col_top_mat.fill(0);
        }
    }

    #[inline]
    fn ensure_height_cache(&mut self, w: usize, h: usize) {
        let need = w * h;
        if self.height_cache.len() != need {
            self.height_cache.resize(need, 0);
        }
        self.height_cache_w = w;
        self.height_cache_h = h;
    }


    #[inline]
    fn ensure_2d_i32(v: &mut Vec<i32>, side: usize, fill: i32) {
        let need = side * side;
        if v.len() != need {
            v.resize(need, fill);
        } else {
            v.fill(fill);
        }
    }

    #[inline]
    fn ensure_3d_u8(v: &mut Vec<u8>, side: usize, fill: u8) {
        let need = side * side * side;
        if v.len() != need {
            v.resize(need, fill);
        } else {
            v.fill(fill);
        }
    }

    #[inline]
    fn ensure_cave_mask(&mut self, side: usize, step: usize) {
        debug_assert!(side % step == 0);
        let dim = side / step;
        let need = dim * dim * dim;
        if self.cave_mask.len() != need {
            self.cave_mask.resize(need, 0);
        } else {
            self.cave_mask.fill(0);
        }
        debug_assert!(step.is_power_of_two());
        self.cave_shift = step.trailing_zeros();

        self.cave_dim = dim;
    }

}

// -----------------------------------------------------------------------------
// Phase helpers
// -----------------------------------------------------------------------------

fn build_height_cache<'g>(
    gen: &'g WorldGen,
    ctx: ChunkCtx,
    scratch: &mut BuildScratch,
    cancel: &AtomicBool,
    tim: &mut BuildTimingsMs,
) -> HeightSampler<'g> {
    // Only cache the chunk footprint (+ optional small margin).
    // If you want a tiny safety halo, set margin_vox = ctx.vpm (1 meter).
    let margin_vox: i32 = 0;

    let cache_x0 = ctx.ox - margin_vox;
    let cache_z0 = ctx.oz - margin_vox;

    // inclusive max (so +1 sizing)
    let cache_x1 = ctx.ox + (ctx.size_i - 1) + margin_vox;
    let cache_z1 = ctx.oz + (ctx.size_i - 1) + margin_vox;

    let cache_w = (cache_x1 - cache_x0 + 1) as usize;
    let cache_h = (cache_z1 - cache_z0 + 1) as usize;

    let same =
        scratch.height_cache_valid &&
        scratch.height_cache_x0 == cache_x0 &&
        scratch.height_cache_z0 == cache_z0 &&
        scratch.height_cache_w == cache_w &&
        scratch.height_cache_h == cache_h;

    if !same {
        scratch.ensure_height_cache(cache_w, cache_h);
        tim.cache_w = cache_w as u32;
        tim.cache_h = cache_h as u32;

        let step_m = config::VOXEL_SIZE_M_F64;
        let x0m = (cache_x0 as f64) * step_m;
        let z0m = (cache_z0 as f64) * step_m;

        scratch.height_cache
            .par_chunks_mut(cache_w)
            .enumerate()
            .for_each(|(z, row)| {
                if (z & 15) == 0 && should_cancel(cancel) {
                    return;
                }

                let zm = z0m + (z as f64) * step_m;
                let mut xm = x0m;

                // tight inner loop: no indexing into xs_m
                for x in 0..cache_w {
                    row[x] = gen.ground_height_m(xm, zm);
                    xm += step_m;
                }
            });

        scratch.height_cache_x0 = cache_x0;
        scratch.height_cache_z0 = cache_z0;
        scratch.height_cache_valid = true;
    }


    HeightSampler {
        gen,
        cache_ptr: scratch.height_cache.as_ptr(),
        cache_w,
        x0: cache_x0,
        x1: cache_x1,
        z0: cache_z0,
        z1: cache_z1,
    }
}


fn build_ground_2d(
    ctx: ChunkCtx,
    scratch: &mut BuildScratch,
    cancel: &AtomicBool,
    height: &HeightSampler<'_>,
) {
    BuildScratch::ensure_2d_i32(&mut scratch.ground, ctx.side, 0);

    // Fast path: cache exactly covers the chunk footprint.
    // If it doesn't, fall back to sampling.
    let want_w = ctx.side;
    let want_h = ctx.side;

    let cache_ok =
        scratch.height_cache_w == want_w &&
        scratch.height_cache_h == want_h &&
        height.x0 == ctx.ox &&
        height.z0 == ctx.oz;

    if cache_ok {
        // memcpy per row (parallel)
        scratch
            .ground
            .par_chunks_mut(ctx.side)
            .zip(scratch.height_cache.par_chunks(ctx.side))
            .enumerate()
            .for_each(|(lz, (dst, src))| {
                if (lz & 15) == 0 && should_cancel(cancel) {
                    return;
                }
                dst.copy_from_slice(src);
            });
        return;
    }

    // Fallback (should be rare if you keep margin_vox = 0)
    scratch
        .ground
        .par_chunks_mut(ctx.side)
        .enumerate()
        .for_each(|(lz, row)| {
            if (lz & 15) == 0 && should_cancel(cancel) {
                return;
            }
            let wz = ctx.oz + lz as i32;
            for lx in 0..ctx.side {
                let wx = ctx.ox + lx as i32;
                row[lx] = height.at(wx, wz);
            }
        });
}


/// Fills `tree_top` with an estimated top-of-tree Y (in voxel coords relative to chunk),
/// using `tree_instance_at_meter` and cached ground heights.
fn build_tree_top(
    gen: &WorldGen,
    ctx: ChunkCtx,
    scratch: &mut BuildScratch,
    cancel: &AtomicBool,
    tim: &mut BuildTimingsMs,
    height: &HeightSampler<'_>,
) {
    BuildScratch::ensure_2d_i32(&mut scratch.tree_top, ctx.side, -1);

    // pad region in meters
    let pad_m = 4;
    let xm0 = (ctx.ox.div_euclid(ctx.vpm)) - pad_m;
    let xm1 = ((ctx.ox + ctx.size_i).div_euclid(ctx.vpm)) + pad_m;
    let zm0 = (ctx.oz.div_euclid(ctx.vpm)) - pad_m;
    let zm1 = ((ctx.oz + ctx.size_i).div_euclid(ctx.vpm)) + pad_m;

    tim.tree_cells_tested = ((xm1 - xm0 + 1) * (zm1 - zm0 + 1)) as u32;

    for zm in zm0..=zm1 {
        if ((zm - zm0) & 3) == 0 && should_cancel(cancel) {
            return;
        }

        for xm in xm0..=xm1 {
            let Some((trunk_h_vox, crown_r_vox)) = gen.tree_instance_at_meter(xm, zm) else {
                continue;
            };
            tim.tree_instances += 1;

            // world position in voxels (at meter grid point)
            let wx = xm * ctx.vpm;
            let wz = zm * ctx.vpm;

            // ground height at the tree base
            let g = height.at(wx, wz);

            // approximate top of tree
            let top_world_y = g + trunk_h_vox + crown_r_vox;
            let top_local_y = top_world_y - ctx.oy;

            // stamp a disk into tree_top
            let r = crown_r_vox.max(1);
            let r2 = r * r;

            // disk bounds in world voxels
            let x0 = wx - r;
            let x1 = wx + r;
            let z0 = wz - r;
            let z1 = wz + r;

            // clamp to chunk-local coords
            let lx0 = (x0 - ctx.ox).max(0).min(ctx.size_i - 1) as usize;
            let lx1 = (x1 - ctx.ox).max(0).min(ctx.size_i - 1) as usize;
            let lz0 = (z0 - ctx.oz).max(0).min(ctx.size_i - 1) as usize;
            let lz1 = (z1 - ctx.oz).max(0).min(ctx.size_i - 1) as usize;

            for lz in lz0..=lz1 {
                let wz2 = ctx.oz + lz as i32;
                let dz = wz2 - wz;
                for lx in lx0..=lx1 {
                    let wx2 = ctx.ox + lx as i32;
                    let dx = wx2 - wx;
                    if dx * dx + dz * dz <= r2 {
                        let idx = idx_xz(ctx.side, lx, lz);
                        scratch.tree_top[idx] = scratch.tree_top[idx].max(top_local_y);
                    }
                }
            }
        }
    }
}

/// Fill per-voxel material into scratch.material.
/// Tree overlay is supplied as a closure returning material id at local coords.
fn fill_material(
    gen: &WorldGen,
    ctx: ChunkCtx,
    scratch: &mut BuildScratch,
    cancel: &AtomicBool,
    tree_material_local: &(dyn Fn(usize, usize, usize) -> u8 + Sync),
    edits: &[EditEntry],
) {
    const AIR8: u8 = AIR as u8;
    const DIRT8: u8 = DIRT as u8;
    const GRASS8: u8 = GRASS as u8;
    const STONE8: u8 = STONE as u8;

    BuildScratch::ensure_3d_u8(&mut scratch.material, ctx.side, AIR8);

    let side = ctx.side;
    let side2 = side * side;

    let ground: &[i32] = &scratch.ground;
    let tree_top: &[i32] = &scratch.tree_top;

    let cave_mask: &[u8] = &scratch.cave_mask;
    let cave_shift: u32 = scratch.cave_shift;
    let cave_dim: usize = scratch.cave_dim;


    // cave depth gate (same as before)
    let max_depth_vox: i32 = (48.0_f32 * (ctx.vpm as f32)).round() as i32;

    scratch
        .material
        .par_chunks_mut(side2)
        .enumerate()
        .for_each(|(ly, slab)| {
            if (ly & 7) == 0 && should_cancel(cancel) {
                return;
            }

            let wy = ctx.oy + ly as i32;

            for lz in 0..side {
                let row_off = lz * side;

                for lx in 0..side {
                    let col = row_off + lx;
                    let g = ground[col];

                    // 1) terrain
                    let mut m: u8 = if wy < g {
                        if wy >= g - ctx.dirt_depth { DIRT8 } else { STONE8 }
                    } else if wy == g {
                        GRASS8
                    } else {
                        AIR8
                    };

                    // 2) caves (depth gate)
                    if m != AIR8 {
                        let depth_vox = g - wy;
                        if depth_vox >= 0 && depth_vox <= max_depth_vox {
                            if cave_mask_at_shifted(cave_mask, cave_dim, cave_shift, lx, ly, lz) {
                                m = AIR8;
                            }
                        }
                    }

                    // 3) trees overlay
                    if m == AIR8 {
                        let ttop = tree_top[col]; // -1 means no tree influence in this column
                        if ttop >= 0 && (ly as i32) <= ttop {
                            let tm = tree_material_local(lx, ly, lz);
                            if tm != AIR8 {
                                m = tm;
                            }
                        }
                    }


                    slab[row_off + lx] = m;
                }
            }
        });

    // 4) apply edits (edits win)
    // NOTE: your edits are chunk-local linear indices; this is O(#edits).
    for e in edits {
        let i = e.idx as usize;
        if i < scratch.material.len() {
            scratch.material[i] = (e.mat & 0xFF) as u8;
        }
    }
}


fn build_cave_mask_coarse(
    gen: &WorldGen,
    ctx: ChunkCtx,
    scratch: &mut BuildScratch,
    cancel: &AtomicBool,
) {
    // Tune: 2 = higher quality, 4 = much faster, 8 = very blocky but extremely fast.
    let step: usize = 4;

    scratch.ensure_cave_mask(ctx.side, step);

    let dim = scratch.cave_dim;
    let side = ctx.side;

    // constants mirrored from WorldGen::carve_cave
    let max_depth_vox: i32 = (48.0_f32 * (ctx.vpm as f32)).round() as i32;

    scratch
        .cave_mask
        .par_chunks_mut(dim * dim) // y-slabs in the coarse grid
        .enumerate()
        .for_each(|(sy, slab)| {
            if (sy & 3) == 0 && should_cancel(cancel) {
                return;
            }

            // sample y in voxel coords (center of the step cell)
            let ly = (sy * step + (step / 2)).min(side - 1);
            let wy = ctx.oy + ly as i32;

            // sample y in voxel coords (center of the step cell)
            let ly = (sy * step + (step / 2)).min(side - 1);
            let wy = ctx.oy + ly as i32;

            for sz in 0..dim {
                let row_off = sz * dim;

                // coarse cell z-range
                let z0 = sz * step;
                let z1 = (z0 + step - 1).min(side - 1);

                for sx in 0..dim {
                    // coarse cell x-range
                    let x0 = sx * step;
                    let x1 = (x0 + step - 1).min(side - 1);

                    // coarse cell y-range
                    let y0 = sy * step;
                    let y1 = (y0 + step - 1).min(side - 1);

                    let mut carve = false;

                    // Test the 8 corners of the step-cube.
                    // If ANY corner wants carving, mark this coarse cell carved.
                    for &ly in &[y0, y1] {
                        let wy = ctx.oy + ly as i32;
                        for &lz in &[z0, z1] {
                            let wz = ctx.oz + lz as i32;
                            for &lx in &[x0, x1] {
                                let wx = ctx.ox + lx as i32;

                                // ground for this (x,z) column
                                let g = scratch.ground[idx_xz(side, lx, lz)];

                                let depth_vox = g - wy;
                                if depth_vox > 0 && depth_vox <= max_depth_vox {
                                    if gen.carve_cave(wx, wy, wz, g) {
                                        carve = true;
                                        break;
                                    }
                                }
                            }
                            if carve { break; }
                        }
                        if carve { break; }
                    }

                    slab[row_off + sx] = carve as u8;
                }
            }


        });
}

#[inline(always)]
fn idx3(dim: usize, x: usize, y: usize, z: usize) -> usize {
    (y * dim * dim) + (z * dim) + x
}

#[inline(always)]
fn cave_mask_at_shifted(
    cave_mask: &[u8],
    cave_dim: usize,
    cave_shift: u32,
    lx: usize,
    ly: usize,
    lz: usize,
) -> bool {
    let sx = (lx >> cave_shift).min(cave_dim - 1);
    let sy = (ly >> cave_shift).min(cave_dim - 1);
    let sz = (lz >> cave_shift).min(cave_dim - 1);
    cave_mask[idx3(cave_dim, sx, sy, sz)] != 0
}



fn build_colinfo_words(ctx: ChunkCtx, scratch: &BuildScratch) -> Vec<u32> {
    debug_assert_eq!(ctx.side, 64, "colinfo packing assumes chunk_size=64");

    // packed u16: (mat8<<8) | y8, y8=255 means empty column
    // 2 entries per u32 => 2048 u32 words for 64*64 columns.
    let mut colinfo_words = vec![0u32; 2048];

    for lz in 0..64usize {
        for lx in 0..64usize {
            let col = idx_xz(64, lx, lz);

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
}

fn build_col_tops(ctx: ChunkCtx, scratch: &mut BuildScratch, cancel: &AtomicBool, tim: &mut BuildTimingsMs) {
    const AIR8: u8 = AIR as u8;

    scratch.ensure_coltop(ctx.side);

    let side = ctx.side;
    let side2 = side * side;

    // Parallel over columns (x,z)
    scratch
        .col_top_y
        .par_iter_mut()
        .zip(scratch.col_top_mat.par_iter_mut())
        .enumerate()
        .for_each(|(col, (out_y, out_m))| {
            if (col & 1023) == 0 && should_cancel(cancel) {
                return;
            }

            let lx = col % side;
            let lz = col / side;

            // scan from top to bottom
            for ly in (0..side).rev() {
                let idx = ly * side2 + lz * side + lx;
                let m = unsafe { *scratch.material.get_unchecked(idx) };
                if m != AIR8 {
                    *out_y = ly as u8;
                    *out_m = m;
                    return;
                }
            }

            *out_y = 255;
            *out_m = 0;
        });

    // If you want a “non-empty columns” counter:
    tim.solid_voxels = scratch.col_top_y.iter().filter(|&&y| y != 255).count() as u32;
}


fn build_macro_occ(ctx: ChunkCtx, prefix_buf: &[u32], cancel: &AtomicBool) -> Vec<u32> {
    let side = ctx.side;

    // Macro occupancy bitset (8x8x8 => 512 bits => 16 u32 words)
    let macro_dim: usize = 8;
    debug_assert_eq!(side % macro_dim, 0);
    let cell: usize = side / macro_dim;

    let mut macro_words = vec![0u32; MACRO_WORDS_PER_CHUNK_USIZE];

    for mz in 0..macro_dim {
        if should_cancel(cancel) {
            return Vec::new();
        }
        for my in 0..macro_dim {
            for mx in 0..macro_dim {
                let x0 = mx * cell;
                let y0 = my * cell;
                let z0 = mz * cell;

                let sum = prefix::prefix_sum_cube(prefix_buf, side, x0, y0, z0, cell);
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
}

// -----------------------------------------------------------------------------
// Public entrypoint (pipeline)
// -----------------------------------------------------------------------------
pub fn build_chunk_svo_sparse_cancelable_with_scratch(
    gen: &WorldGen,
    chunk_origin: [i32; 3],
    chunk_size: u32,
    cancel: &AtomicBool,
    scratch: &mut BuildScratch,
    edits: &[EditEntry],
) -> BuildOutput {
    let mut tim = BuildTimingsMs::default();
    let t_total = std::time::Instant::now();

    let ctx = ChunkCtx::new(chunk_origin, chunk_size);

    cancel_if!(cancel, tim);

    // Height cache / sampler
    let height = time_it!(tim, height_cache, {
        build_height_cache(gen, ctx, scratch, cancel, &mut tim)
    });
    cancel_if!(cancel, tim);

    // Tree mask (exact type is from WorldGen; keep it local and use via closure)
    let tree_mask = time_it!(tim, tree_mask, {
        let (_tree_cache_unused, tree_mask) = gen.build_tree_cache_with_mask(
            ctx.ox,
            ctx.oy,
            ctx.oz,
            ctx.size_i,
            &|wx, wz| height.at(wx, wz),
            cancel,
        );
        tree_mask
    });
    cancel_if!(cancel, tim);

    // Ground 2D + mip
    time_it!(tim, ground_2d, {
        build_ground_2d(ctx, scratch, cancel, &height);
    });
    cancel_if!(cancel, tim);

    // Tree top + mip (tree mip currently not used later, but kept for parity with original)
    time_it!(tim, tree_top, {
        build_tree_top(gen, ctx, scratch, cancel, &mut tim, &height);
    });
    cancel_if!(cancel, tim);

    let _tree_mip = time_it!(tim, tree_mip, {
        build_max_mip_inplace(&scratch.tree_top, ctx.size_u, &mut scratch.tree_levels)
    });

    // Material fill
    time_it!(tim, material_fill, {
        // Build coarse cave mask FIRST (included in material_fill time for now)
        build_cave_mask_coarse(gen, ctx, scratch, cancel);

        let tree_mat = |lx: usize, ly: usize, lz: usize| -> u8 {
            (tree_mask.material_local(lx, ly, lz) & 0xFF) as u8
        };
        fill_material(gen, ctx, scratch, cancel, &tree_mat, edits);
    });
    cancel_if!(cancel, tim);

    // Column tops
    time_it!(tim, colinfo, {
        build_col_tops(ctx, scratch, cancel, &mut tim);
    });
    cancel_if!(cancel, tim);

    // Pack colinfo words
    let colinfo_words = time_it!(tim, colinfo, {
        build_colinfo_words(ctx, scratch)
    });
    cancel_if!(cancel, tim);

    // Prefix sum (summed-volume table)
    prefix::ensure_prefix(&mut scratch.prefix, ctx.side);

    time_it!(tim, prefix_x, {
        prefix::prefix_pass_x(&mut scratch.prefix, &scratch.material, ctx.side, cancel);
    });
    cancel_if!(cancel, tim);

    time_it!(tim, prefix_y, {
        prefix::prefix_pass_y(&mut scratch.prefix, ctx.side, cancel);
    });
    cancel_if!(cancel, tim);

    time_it!(tim, prefix_z, {
        prefix::prefix_pass_z(&mut scratch.prefix, ctx.side, cancel);
    });
    cancel_if!(cancel, tim);

    // Macro occupancy words
    let macro_words = time_it!(tim, macro_occ, {
        build_macro_occ(ctx, &scratch.prefix, cancel)
    });
    cancel_if!(cancel, tim);

    // mip
    let ground_mip = time_it!(tim, ground_mip, {
        build_minmax_mip_inplace(
            &scratch.ground,
            ctx.size_u,
            &mut scratch.ground_min_levels,
            &mut scratch.ground_max_levels,
        )
    });


    // SVO build (bottom-up) + ropes
    let nodes = time_it!(tim, svo_build, {
        svo::build_svo_bottom_up(
            ctx.size_u,
            ctx.oy,
            &scratch.material,
            &scratch.prefix,
            &ground_mip,
            ctx.dirt_depth,
            cancel,
            &mut scratch.svo,
        )
    });
    cancel_if!(cancel, tim);

    let ropes_vec = time_it!(tim, ropes, { ropes::build_ropes(&nodes) });

    tim.nodes = nodes.len() as u32;
    tim.total = ms_since(t_total);

    BuildOutput {
        nodes,
        macro_words,
        ropes: ropes_vec,
        colinfo_words,
        timings: tim,
    }
}
