// src/svo/builder_prefix.rs
//
// Prefix sum / summed-volume table for solid occupancy.
// prefix[x,y,z] = sum of occupancy in [0..x)×[0..y)×[0..z)

use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;

use crate::world::materials::AIR;

#[inline]
fn should_cancel(cancel: &AtomicBool) -> bool {
    cancel.load(Ordering::Relaxed)
}

#[inline]
fn idx_prefix(dim: usize, x: usize, y: usize, z: usize) -> usize {
    (z * dim * dim) + (y * dim) + x
}

pub fn ensure_prefix(prefix: &mut Vec<u32>, side: usize) {
    let dim = side + 1;
    let need = dim * dim * dim;
    if prefix.len() != need {
        prefix.resize(need, 0);
    } else {
        prefix.fill(0);
    }
}

/// Query sum over cube [x0..x0+size) × [y0..y0+size) × [z0..z0+size)
pub fn prefix_sum_cube(prefix: &[u32], side: usize, x0: usize, y0: usize, z0: usize, size: usize) -> u32 {
    let dim = side + 1;
    let x1 = x0 + size;
    let y1 = y0 + size;
    let z1 = z0 + size;

    let a = prefix[idx_prefix(dim, x1, y1, z1)] as i64;
    let b = prefix[idx_prefix(dim, x0, y1, z1)] as i64;
    let c = prefix[idx_prefix(dim, x1, y0, z1)] as i64;
    let d = prefix[idx_prefix(dim, x1, y1, z0)] as i64;
    let e = prefix[idx_prefix(dim, x0, y0, z1)] as i64;
    let f = prefix[idx_prefix(dim, x0, y1, z0)] as i64;
    let g = prefix[idx_prefix(dim, x1, y0, z0)] as i64;
    let h = prefix[idx_prefix(dim, x0, y0, z0)] as i64;

    let s = a - b - c - d + e + f + g - h;
    debug_assert!(s >= 0);
    s as u32
}

/// Pass 1: write occupancy and prefix along X for each (y,z) row.
/// Layout: idx_prefix(dim,x,y,z) => x contiguous.
pub fn prefix_pass_x(prefix: &mut [u32], material: &[u8], side: usize, cancel: &AtomicBool) {
    let dim = side + 1;
    let plane = dim * dim;

    for z in 1..=side {
        if (z & 7) == 0 && should_cancel(cancel) {
            return;
        }

        let slab0 = z * plane;
        let slab = &mut prefix[slab0..slab0 + plane];

        for y in 1..=side {
            // prefix row slice for this (y,z): x contiguous
            let row = &mut slab[y * dim..(y + 1) * dim];

            let base_m = ((y - 1) * side * side) + ((z - 1) * side);

            let mut run: u32 = 0;
            row[0] = 0;

            // x = 1..=side writes row[x]
            for x in 1..=side {
                let m = unsafe { *material.get_unchecked(base_m + (x - 1)) };
                run += (m != (AIR as u8)) as u32;
                unsafe {
                    *row.get_unchecked_mut(x) = run;
                }
            }
        }
    }
}

/// Pass 2: prefix along Y within each Z-plane.
pub fn prefix_pass_y(prefix: &mut [u32], side: usize, cancel: &AtomicBool) {
    let dim = side + 1;
    let plane = dim * dim;

    for z in 1..=side {
        if (z & 7) == 0 && should_cancel(cancel) {
            return;
        }

        let slab0 = z * plane;
        let slab = &mut prefix[slab0..slab0 + plane];

        for x in 1..=side {
            let mut run: u32 = 0;
            for y in 1..=side {
                let idx = y * dim + x;
                unsafe {
                    run += *slab.get_unchecked(idx);
                    *slab.get_unchecked_mut(idx) = run;
                }
            }
        }
    }
}


/// Pass 3: prefix along Z: prefix[x,y,z] += prefix[x,y,z-1].
pub fn prefix_pass_z(prefix: &mut [u32], side: usize, cancel: &AtomicBool) {
    let dim = side + 1;
    let plane = dim * dim;

    for x in 1..=side {
        if (x & 7) == 0 && should_cancel(cancel) {
            return;
        }
        for y in 1..=side {
            let base = y * dim + x;
            let mut run = unsafe { *prefix.get_unchecked(base) };
            for z in 1..=side {
                let idx = z * plane + base;
                unsafe {
                    run += *prefix.get_unchecked(idx);
                    *prefix.get_unchecked_mut(idx) = run;
                }
            }
        }
    }
}



