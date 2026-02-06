

use rayon::prelude::*;

pub struct MinMaxMipView<'a> {
    pub root_side: u32,
    pub min_levels: &'a [Vec<i32>],
    pub max_levels: &'a [Vec<i32>],
}

pub fn build_minmax_mip_inplace<'a>(
    base: &[i32],
    side: u32,
    min_levels: &'a mut Vec<Vec<i32>>,
    max_levels: &'a mut Vec<Vec<i32>>,
) -> MinMaxMipView<'a> {
    debug_assert!(side.is_power_of_two());
    debug_assert_eq!(base.len(), (side * side) as usize);

    let levels = side.trailing_zeros() as usize + 1;

    if min_levels.len() != levels {
        min_levels.resize_with(levels, Vec::new);
    }
    if max_levels.len() != levels {
        max_levels.resize_with(levels, Vec::new);
    }

    
    min_levels[0].resize(base.len(), 0);
    min_levels[0].copy_from_slice(base);

    max_levels[0].resize(base.len(), 0);
    max_levels[0].copy_from_slice(base);

    let mut cur_side = side;

    
    
    for lvl in 1..levels {
        let next_side = cur_side / 2;
        let need = (next_side * next_side) as usize;

        let (min_prev, min_rest) = min_levels.split_at_mut(lvl);
        let (max_prev, max_rest) = max_levels.split_at_mut(lvl);

        let cur_min: &[i32] = &min_prev[lvl - 1];
        let cur_max: &[i32] = &max_prev[lvl - 1];

        let mn: &mut Vec<i32> = &mut min_rest[0];
        let mx: &mut Vec<i32> = &mut max_rest[0];
        mn.resize(need, 0);
        mx.resize(need, 0);

        let cur_side_us = cur_side as usize;
        let next_side_us = next_side as usize;

        mn.par_chunks_mut(next_side_us)
            .zip(mx.par_chunks_mut(next_side_us))
            .enumerate()
            .for_each(|(z, (mn_row, mx_row))| {
                let row0 = (2 * z) * cur_side_us;
                let row1 = row0 + cur_side_us;

                for x in 0..next_side_us {
                    let col0 = 2 * x;
                    let i00 = row0 + col0;
                    let i10 = i00 + 1;
                    let i01 = row1 + col0;
                    let i11 = i01 + 1;

                    let a0 = cur_min[i00];
                    let a1 = cur_min[i10];
                    let a2 = cur_min[i01];
                    let a3 = cur_min[i11];
                    mn_row[x] = a0.min(a1).min(a2).min(a3);

                    let b0 = cur_max[i00];
                    let b1 = cur_max[i10];
                    let b2 = cur_max[i01];
                    let b3 = cur_max[i11];
                    mx_row[x] = b0.max(b1).max(b2).max(b3);
                }
            });

        cur_side = next_side;
    }

    MinMaxMipView {
        root_side: side,
        min_levels: &min_levels[..],
        max_levels: &max_levels[..],
    }
}

impl<'a> MinMaxMipView<'a> {
    #[inline]
    pub fn query(&self, x0: i32, z0: i32, size: u32) -> (i32, i32) {
        debug_assert!(size.is_power_of_two());
        debug_assert!(size <= self.root_side);
        debug_assert!(x0 >= 0 && z0 >= 0);

        let level = size.trailing_zeros() as usize;
        debug_assert!(level < self.min_levels.len());

        let side = self.root_side >> level;
        let x = (x0 as u32) / size;
        let z = (z0 as u32) / size;
        let idx = (z * side + x) as usize;

        (self.min_levels[level][idx], self.max_levels[level][idx])
    }
}

pub struct MaxMipView<'a> {
    pub root_side: u32,
    pub levels: &'a [Vec<i32>],
}

pub fn build_max_mip_inplace<'a>(
    base: &[i32],
    side: u32,
    levels: &'a mut Vec<Vec<i32>>,
) -> MaxMipView<'a> {
    debug_assert!(side.is_power_of_two());
    debug_assert_eq!(base.len(), (side * side) as usize);

    let nlevels = side.trailing_zeros() as usize + 1;

    if levels.len() != nlevels {
        levels.resize_with(nlevels, Vec::new);
    }

    
    levels[0].resize(base.len(), 0);
    levels[0].copy_from_slice(base);

    let mut cur_side = side;

    
    for lvl in 1..nlevels {
        let next_side = cur_side / 2;
        let need = (next_side * next_side) as usize;

        let (prev, rest) = levels.split_at_mut(lvl);
        let cur: &[i32] = &prev[lvl - 1];
        let out: &mut Vec<i32> = &mut rest[0];
        out.resize(need, 0);

        let cur_side_us = cur_side as usize;
        let next_side_us = next_side as usize;

        out.par_chunks_mut(next_side_us)
            .enumerate()
            .for_each(|(z, out_row)| {
                let row0 = (2 * z) * cur_side_us;
                let row1 = row0 + cur_side_us;

                for x in 0..next_side_us {
                    let col0 = 2 * x;
                    let i00 = row0 + col0;
                    let i10 = i00 + 1;
                    let i01 = row1 + col0;
                    let i11 = i01 + 1;

                    let a = cur[i00];
                    let b = cur[i10];
                    let c = cur[i01];
                    let d = cur[i11];
                    out_row[x] = a.max(b).max(c).max(d);
                }
            });

        cur_side = next_side;
    }

    MaxMipView {
        root_side: side,
        levels: &levels[..],
    }
}
