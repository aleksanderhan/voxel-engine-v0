




use glam::Vec3;

use crate::app::config;
use crate::{world::WorldGen};

#[derive(Clone, Copy, Debug)]
pub struct ClipLevelParams {
    pub origin_x_m: f32,
    pub origin_z_m: f32,
    pub cell_size_m: f32,
    
    pub off_x: u32,
    pub off_z: u32,
}


#[derive(Clone, Copy, Debug)]
pub struct ClipmapParamsCpu {
    pub levels: u32,
    pub res: u32,
    pub base_cell_m: f32,
    pub _pad0: f32,
    pub level: [ClipLevelParams; config::CLIPMAP_LEVELS_USIZE],
}





pub struct ClipmapUpload {
    pub level: u32,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub data_f32: Vec<f32>,
}

pub struct Clipmap {
    last_origin_cell: [(i32, i32); config::CLIPMAP_LEVELS_USIZE],
    
    
    tex_offset: [(u16, u16); config::CLIPMAP_LEVELS_USIZE],
    last_update_time_s_global: f32,
}

impl Clipmap {
    pub fn new() -> Self {
        Self {
            last_origin_cell: [(i32::MIN, i32::MIN); config::CLIPMAP_LEVELS_USIZE],
            tex_offset: [(0, 0); config::CLIPMAP_LEVELS_USIZE],
            last_update_time_s_global: f32::NEG_INFINITY,
        }
    }

    #[inline]
    fn level_cell_size(i: u32) -> f32 {
        config::CLIPMAP_BASE_CELL_M * (1u32 << i) as f32
    }

    
    
    
    
    #[inline]
    fn snapped_origin_cell(cam_x_m: f32, cam_z_m: f32, cell_m: f32) -> (i32, i32) {
        let half = (config::CLIPMAP_RES as f32) * 0.5;
        let ox_m = cam_x_m - half * cell_m;
        let oz_m = cam_z_m - half * cell_m;

        let ox_c = (ox_m / cell_m).floor() as i32;
        let oz_c = (oz_m / cell_m).floor() as i32;
        (ox_c, oz_c)
    }

    #[inline]
    fn cell_to_origin_m(cell_x: i32, cell_z: i32, cell_m: f32) -> (f32, f32) {
        (cell_x as f32 * cell_m, cell_z as f32 * cell_m)
    }

    #[inline]
    fn wrap_i32_mod_u16(v: i32, m: i32) -> u16 {
        let mut r = v % m;
        if r < 0 {
            r += m;
        }
        r as u16
    }

    #[inline]
    fn sample_height_f32(world: &WorldGen, wx_m: f32, wz_m: f32) -> f32 {
        let vs = config::VOXEL_SIZE_M_F32;

        let wx_vx = (wx_m / vs).floor() as i32;
        let wz_vx = (wz_m / vs).floor() as i32;

        let h_vx = world.ground_height(wx_vx, wz_vx);
        (h_vx as f32) * vs
    }


    fn build_full_level(world: &WorldGen, ox_m: f32, oz_m: f32, cell_m: f32) -> Vec<f32> {
        let res = config::CLIPMAP_RES as usize;
        let mut data = vec![0.0f32; res * res];

        for tz in 0..res {
            let wz_m = oz_m + (tz as f32 + 0.5) * cell_m;
            let row = tz * res;

            for tx in 0..res {
                let wx_m = ox_m + (tx as f32 + 0.5) * cell_m;
                data[row + tx] = Self::sample_height_f32(world, wx_m, wz_m);
            }
        }

        data
    }


    fn build_row_patch(
        world: &WorldGen,
        ox_m: f32,
        oz_m: f32,
        cell_m: f32,
        logical_z0: u32,
        rows: u32,
    ) -> Vec<f32> {
        let res = config::CLIPMAP_RES as usize;
        let w = res as u32;

        let h = rows as usize;
        let mut data = vec![0.0f32; (w as usize) * h];

        for rz in 0..rows {
            let lz = logical_z0 + rz;
            let wz_m = oz_m + (lz as f32 + 0.5) * cell_m;

            let row = (rz as usize) * res;
            for tx in 0..res {
                let wx_m = ox_m + (tx as f32 + 0.5) * cell_m;
                data[row + tx] = Self::sample_height_f32(world, wx_m, wz_m);
            }
        }

        data
    }

    fn build_col_patch(
        world: &WorldGen,
        ox_m: f32,
        oz_m: f32,
        cell_m: f32,
        logical_x0: u32,
        cols: u32,
    ) -> Vec<f32> {
        let res = config::CLIPMAP_RES as usize;
        let w = cols as usize;
        let h = res;

        let mut data = vec![0.0f32; w * h];

        for tz in 0..res {
            let wz_m = oz_m + (tz as f32 + 0.5) * cell_m;
            let row = tz * w;

            for cx in 0..cols {
                let lx = logical_x0 + cx;
                let wx_m = ox_m + (lx as f32 + 0.5) * cell_m;
                data[row + (cx as usize)] = Self::sample_height_f32(world, wx_m, wz_m);
            }
        }

        data
    }


    pub fn update(
        &mut self,
        world: &WorldGen,
        cam_pos_m: Vec3,
        time_s: f32,
    ) -> (ClipmapParamsCpu, Vec<ClipmapUpload>) {
        let mut params = ClipmapParamsCpu {
            levels: config::CLIPMAP_LEVELS,
            res: config::CLIPMAP_RES,
            base_cell_m: config::CLIPMAP_BASE_CELL_M,
            _pad0: 0.0,
            level: [ClipLevelParams {
                origin_x_m: 0.0,
                origin_z_m: 0.0,
                cell_size_m: 1.0,
                off_x: 0,
                off_z: 0,
            }; config::CLIPMAP_LEVELS_USIZE],
        };

        let mut uploads = Vec::new();

        let res_u = config::CLIPMAP_RES;
        let res_i = res_u as i32;

        const EDGE_GUARD_TEXELS: i32 = 16; 

        
        
        
        let mut cell_m_arr   = [0.0f32; config::CLIPMAP_LEVELS_USIZE];
        let mut new_ox_c_arr = [0i32;   config::CLIPMAP_LEVELS_USIZE];
        let mut new_oz_c_arr = [0i32;   config::CLIPMAP_LEVELS_USIZE];
        let mut new_ox_m_arr = [0.0f32; config::CLIPMAP_LEVELS_USIZE];
        let mut new_oz_m_arr = [0.0f32; config::CLIPMAP_LEVELS_USIZE];

        let mut dx_arr    = [0i32; config::CLIPMAP_LEVELS_USIZE];
        let mut dz_arr    = [0i32; config::CLIPMAP_LEVELS_USIZE];
        let mut first_arr = [false; config::CLIPMAP_LEVELS_USIZE];
        let mut moved_arr = [false; config::CLIPMAP_LEVELS_USIZE];
        let mut force_arr = [false; config::CLIPMAP_LEVELS_USIZE];

        let mut any_first = false;
        let mut any_moved = false;
        let mut any_force = false;

        for i in 0..config::CLIPMAP_LEVELS {
            let li = i as usize;

            let cell_m = Self::level_cell_size(i);
            cell_m_arr[li] = cell_m;

            let (new_ox_c, new_oz_c) = Self::snapped_origin_cell(cam_pos_m.x, cam_pos_m.z, cell_m);
            let (new_ox_m, new_oz_m) = Self::cell_to_origin_m(new_ox_c, new_oz_c, cell_m);

            new_ox_c_arr[li] = new_ox_c;
            new_oz_c_arr[li] = new_oz_c;
            new_ox_m_arr[li] = new_ox_m;
            new_oz_m_arr[li] = new_oz_m;

            let (old_ox_c, old_oz_c) = self.last_origin_cell[li];
            let first = old_ox_c == i32::MIN;
            first_arr[li] = first;
            any_first |= first;

            let dx = if first { 0 } else { new_ox_c - old_ox_c };
            let dz = if first { 0 } else { new_oz_c - old_oz_c };
            dx_arr[li] = dx;
            dz_arr[li] = dz;

            let moved = first || dx != 0 || dz != 0;
            moved_arr[li] = moved;
            any_moved |= moved;

            let force_by_bounds = if first {
                true
            } else {
                let (old_ox_m, old_oz_m) = Self::cell_to_origin_m(old_ox_c, old_oz_c, cell_m);

                let cam_lx = ((cam_pos_m.x - old_ox_m) / cell_m).floor() as i32;
                let cam_lz = ((cam_pos_m.z - old_oz_m) / cell_m).floor() as i32;

                cam_lx < EDGE_GUARD_TEXELS
                    || cam_lz < EDGE_GUARD_TEXELS
                    || cam_lx >= (res_i - EDGE_GUARD_TEXELS)
                    || cam_lz >= (res_i - EDGE_GUARD_TEXELS)
            };

            force_arr[li] = force_by_bounds;
            any_force |= force_by_bounds;
        }

        
        
        
        let cadence_ok = any_first
            || (time_s - self.last_update_time_s_global) >= config::CLIPMAP_MIN_UPDATE_INTERVAL_S;

        let do_frame_update = any_first || any_force || (cadence_ok && any_moved);

        if do_frame_update {
            self.last_update_time_s_global = time_s;
        }

        
        
        
        for i in 0..config::CLIPMAP_LEVELS {
            let li = i as usize;

            let cell_m = cell_m_arr[li];

            let new_ox_c = new_ox_c_arr[li];
            let new_oz_c = new_oz_c_arr[li];
            let new_ox_m = new_ox_m_arr[li];
            let new_oz_m = new_oz_m_arr[li];

            let first = first_arr[li];
            let dx = dx_arr[li];
            let dz = dz_arr[li];
            let moved = moved_arr[li];

            
            let (mut off_x, mut off_z) = self.tex_offset[li];

            let do_update = do_frame_update && moved;

            if do_update {
                if first {
                    
                    self.last_origin_cell[li] = (new_ox_c, new_oz_c);
                    off_x = 0;
                    off_z = 0;
                    self.tex_offset[li] = (off_x, off_z);

                    let full = Self::build_full_level(world, new_ox_m, new_oz_m, cell_m);
                    uploads.push(ClipmapUpload {
                        level: i,
                        x: 0,
                        y: 0,
                        w: res_u,
                        h: res_u,
                        data_f32: full,
                    });
                } else {
                    let adx = dx.unsigned_abs() as u32;
                    let adz = dz.unsigned_abs() as u32;
                    let big_jump = adx >= res_u || adz >= res_u;

                    if big_jump {
                        
                        self.last_origin_cell[li] = (new_ox_c, new_oz_c);
                        off_x = 0;
                        off_z = 0;
                        self.tex_offset[li] = (off_x, off_z);

                        let full = Self::build_full_level(world, new_ox_m, new_oz_m, cell_m);
                        uploads.push(ClipmapUpload {
                            level: i,
                            x: 0,
                            y: 0,
                            w: res_u,
                            h: res_u,
                            data_f32: full,
                        });
                    } else {
                        
                        off_x = Self::wrap_i32_mod_u16((off_x as i32) + dx, res_i);
                        off_z = Self::wrap_i32_mod_u16((off_z as i32) + dz, res_i);
                        self.tex_offset[li] = (off_x, off_z);
                        self.last_origin_cell[li] = (new_ox_c, new_oz_c);

                        
                        let map_x = |lx: u32| -> u32 { (lx + (off_x as u32)) % res_u };
                        let map_z = |lz: u32| -> u32 { (lz + (off_z as u32)) % res_u };

                        
                        if dx != 0 {
                            let cols = dx.unsigned_abs() as u32;

                            if dx > 0 {
                                
                                let logical_x0 = res_u - cols;
                                let sx0 = map_x(logical_x0);
                                let end = sx0 + cols;

                                if end <= res_u {
                                    let data = Self::build_col_patch(
                                        world, new_ox_m, new_oz_m, cell_m, logical_x0, cols,
                                    );
                                    uploads.push(ClipmapUpload {
                                        level: i,
                                        x: sx0,
                                        y: 0,
                                        w: cols,
                                        h: res_u,
                                        data_f32: data,
                                    });
                                } else {
                                    let a = res_u - sx0;
                                    let b = end - res_u;

                                    let data_a = Self::build_col_patch(
                                        world, new_ox_m, new_oz_m, cell_m, logical_x0, a,
                                    );
                                    uploads.push(ClipmapUpload {
                                        level: i,
                                        x: sx0,
                                        y: 0,
                                        w: a,
                                        h: res_u,
                                        data_f32: data_a,
                                    });

                                    let data_b = Self::build_col_patch(
                                        world, new_ox_m, new_oz_m, cell_m, logical_x0 + a, b,
                                    );
                                    uploads.push(ClipmapUpload {
                                        level: i,
                                        x: 0,
                                        y: 0,
                                        w: b,
                                        h: res_u,
                                        data_f32: data_b,
                                    });
                                }
                            } else {
                                
                                let logical_x0 = 0;
                                let sx0 = map_x(logical_x0);
                                let end = sx0 + cols;

                                if end <= res_u {
                                    let data = Self::build_col_patch(
                                        world, new_ox_m, new_oz_m, cell_m, logical_x0, cols,
                                    );
                                    uploads.push(ClipmapUpload {
                                        level: i,
                                        x: sx0,
                                        y: 0,
                                        w: cols,
                                        h: res_u,
                                        data_f32: data,
                                    });
                                } else {
                                    let a = res_u - sx0;
                                    let b = end - res_u;

                                    let data_a = Self::build_col_patch(
                                        world, new_ox_m, new_oz_m, cell_m, logical_x0, a,
                                    );
                                    uploads.push(ClipmapUpload {
                                        level: i,
                                        x: sx0,
                                        y: 0,
                                        w: a,
                                        h: res_u,
                                        data_f32: data_a,
                                    });

                                    let data_b = Self::build_col_patch(
                                        world, new_ox_m, new_oz_m, cell_m, logical_x0 + a, b,
                                    );
                                    uploads.push(ClipmapUpload {
                                        level: i,
                                        x: 0,
                                        y: 0,
                                        w: b,
                                        h: res_u,
                                        data_f32: data_b,
                                    });
                                }
                            }
                        }

                        
                        if dz != 0 {
                            let rows = dz.unsigned_abs() as u32;

                            if dz > 0 {
                                
                                let logical_z0 = res_u - rows;
                                let sy0 = map_z(logical_z0);
                                let end = sy0 + rows;

                                if end <= res_u {
                                    let data = Self::build_row_patch(
                                        world, new_ox_m, new_oz_m, cell_m, logical_z0, rows,
                                    );
                                    uploads.push(ClipmapUpload {
                                        level: i,
                                        x: 0,
                                        y: sy0,
                                        w: res_u,
                                        h: rows,
                                        data_f32: data,
                                    });
                                } else {
                                    let a = res_u - sy0;
                                    let b = end - res_u;

                                    let data_a = Self::build_row_patch(
                                        world, new_ox_m, new_oz_m, cell_m, logical_z0, a,
                                    );
                                    uploads.push(ClipmapUpload {
                                        level: i,
                                        x: 0,
                                        y: sy0,
                                        w: res_u,
                                        h: a,
                                        data_f32: data_a,
                                    });

                                    let data_b = Self::build_row_patch(
                                        world, new_ox_m, new_oz_m, cell_m, logical_z0 + a, b,
                                    );
                                    uploads.push(ClipmapUpload {
                                        level: i,
                                        x: 0,
                                        y: 0,
                                        w: res_u,
                                        h: b,
                                        data_f32: data_b,
                                    });
                                }
                            } else {
                                
                                let logical_z0 = 0;
                                let sy0 = map_z(logical_z0);
                                let end = sy0 + rows;

                                if end <= res_u {
                                    let data = Self::build_row_patch(
                                        world, new_ox_m, new_oz_m, cell_m, logical_z0, rows,
                                    );
                                    uploads.push(ClipmapUpload {
                                        level: i,
                                        x: 0,
                                        y: sy0,
                                        w: res_u,
                                        h: rows,
                                        data_f32: data,
                                    });
                                } else {
                                    let a = res_u - sy0;
                                    let b = end - res_u;

                                    let data_a = Self::build_row_patch(
                                        world, new_ox_m, new_oz_m, cell_m, logical_z0, a,
                                    );
                                    uploads.push(ClipmapUpload {
                                        level: i,
                                        x: 0,
                                        y: sy0,
                                        w: res_u,
                                        h: a,
                                        data_f32: data_a,
                                    });

                                    let data_b = Self::build_row_patch(
                                        world, new_ox_m, new_oz_m, cell_m, logical_z0 + a, b,
                                    );
                                    uploads.push(ClipmapUpload {
                                        level: i,
                                        x: 0,
                                        y: 0,
                                        w: res_u,
                                        h: b,
                                        data_f32: data_b,
                                    });
                                }
                            }
                        }
                    }
                }
            }

            
            let (commit_ox_m, commit_oz_m) = if self.last_origin_cell[li].0 == i32::MIN {
                (new_ox_m, new_oz_m) 
            } else {
                let (cx, cz) = self.last_origin_cell[li];
                Self::cell_to_origin_m(cx, cz, cell_m)
            };

            params.level[li] = ClipLevelParams {
                origin_x_m: commit_ox_m,
                origin_z_m: commit_oz_m,
                cell_size_m: cell_m,
                off_x: off_x as u32,
                off_z: off_z as u32,
            };
        }

        (params, uploads)
    }


    pub fn invalidate_all(&mut self) {
        self.last_origin_cell = [(i32::MIN, i32::MIN); config::CLIPMAP_LEVELS_USIZE];
        self.tex_offset = [(0, 0); config::CLIPMAP_LEVELS_USIZE];
        self.last_update_time_s_global = f32::NEG_INFINITY;
    }

}

