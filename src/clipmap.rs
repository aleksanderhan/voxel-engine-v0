// src/clipmap.rs
//
// CPU-updated 2D clipmap height texture data (nested levels around camera).
//
// Strategy here (simple + robust):
// - Each level is a full CLIPMAP_RES x CLIPMAP_RES height map (meters).
// - Level i has cell size = CLIPMAP_BASE_CELL_M * 2^i.
// - We "snap" the level origin to cell boundaries so updates are stable.
// - When the camera moves enough that a level origin changes, we refresh that entire level.
//   (This is not the classic incremental ring update, but it is already far cheaper than SVO,
//    and it keeps the implementation small and correct. You can upgrade to strip updates later.)

use glam::Vec3;

use crate::{config, world::WorldGen};

#[derive(Clone, Copy, Debug)]
pub struct ClipLevelParams {
    pub origin_x_m: f32,
    pub origin_z_m: f32,
    pub cell_size_m: f32,
    pub inv_cell_size_m: f32,
}

/// GPU payload for clipmap params (mirrors `ClipmapGpu` in `render/gpu_types.rs`).
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
    /// CLIPMAP_RES * CLIPMAP_RES heights in meters (row-major).
    pub data_m: Vec<f32>,
}

pub struct Clipmap {
    last_origin_cell: [(i32, i32); config::CLIPMAP_LEVELS_USIZE],
    last_update_time_s: [f32; config::CLIPMAP_LEVELS_USIZE],
}

impl Clipmap {
    pub fn new() -> Self {
        Self {
            last_origin_cell: [(i32::MIN, i32::MIN); config::CLIPMAP_LEVELS_USIZE],
            last_update_time_s: [f32::NEG_INFINITY; config::CLIPMAP_LEVELS_USIZE],
        }
    }

    #[inline]
    fn level_cell_size(i: u32) -> f32 {
        config::CLIPMAP_BASE_CELL_M * (1u32 << i) as f32
    }

    /// Compute snapped origin in *cell coordinates* (integer grid), for a given camera xz.
    ///
    /// We keep the camera roughly centered in the texture:
    /// origin = snap(cam - (res/2)*cell).
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

    /// Update clipmap data around camera.
    ///
    /// Inputs:
    /// - `cam_pos_m`: camera position in meters.
    /// - `time_s`: monotonically increasing time (seconds).
    ///
    /// Outputs:
    /// - params for GPU
    /// - uploads for any levels that changed origin (full level refresh)
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
                inv_cell_size_m: 1.0,
            }; config::CLIPMAP_LEVELS_USIZE],
        };

        let mut uploads = Vec::new();

        for i in 0..config::CLIPMAP_LEVELS {
            let li = i as usize;
            let cell_m = Self::level_cell_size(i);
            let inv = 1.0 / cell_m;

            let (ox_c, oz_c) = Self::snapped_origin_cell(cam_pos_m.x, cam_pos_m.z, cell_m);
            let (ox_m, oz_m) = Self::cell_to_origin_m(ox_c, oz_c, cell_m);

            params.level[li] = ClipLevelParams {
                origin_x_m: ox_m,
                origin_z_m: oz_m,
                cell_size_m: cell_m,
                inv_cell_size_m: inv,
            };

            let changed = self.last_origin_cell[li] != (ox_c, oz_c);
            let allow_by_time =
                (time_s - self.last_update_time_s[li]) >= config::CLIPMAP_MIN_UPDATE_INTERVAL_S;

            if changed && allow_by_time {
                self.last_origin_cell[li] = (ox_c, oz_c);
                self.last_update_time_s[li] = time_s;

                // Full refresh of this level.
                let res = config::CLIPMAP_RES as usize;
                let mut data = vec![0.0f32; res * res];

                // We sample the procedural ground height at world (x,z) mapped from cells.
                // WorldGen::ground_height returns voxels, so convert to meters.
                let vs = config::VOXEL_SIZE_M_F32;

                for tz in 0..res {
                    let wz_m = oz_m + (tz as f32 + 0.5) * cell_m;
                    let wz_vx = (wz_m / vs).floor() as i32;

                    let row = tz * res;
                    for tx in 0..res {
                        let wx_m = ox_m + (tx as f32 + 0.5) * cell_m;
                        let wx_vx = (wx_m / vs).floor() as i32;

                        let h_vx = world.ground_height(wx_vx, wz_vx);
                        data[row + tx] = (h_vx as f32) * vs;
                    }
                }

                uploads.push(ClipmapUpload { level: i, data_m: data });
            }
        }

        (params, uploads)
    }
}
