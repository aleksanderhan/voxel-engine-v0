pub struct MinMaxMip {
    pub root_side: u32,
    pub min_levels: Vec<Vec<i32>>,
    pub max_levels: Vec<Vec<i32>>,
}

pub fn build_minmax_mip(base: &[i32], side: u32) -> MinMaxMip {
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

    MinMaxMip {
        root_side: side,
        min_levels,
        max_levels,
    }
}

impl MinMaxMip {
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

pub struct MaxMip {
    pub root_side: u32,
    pub levels: Vec<Vec<i32>>,
}

pub fn build_max_mip(base: &[i32], side: u32) -> MaxMip {
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
    #[inline]
    pub fn query_max(&self, x0: i32, z0: i32, size: u32) -> i32 {
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
