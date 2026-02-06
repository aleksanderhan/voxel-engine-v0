use glam::Vec3;
use crate::app::config;
use super::{ChunkManager, ground};
use crate::streaming::types;



const ANG_BINS: usize = 720;


const PADDING_CHUNKS: i32 = 1;

pub fn ensure_visible_columns(mgr: &mut ChunkManager, center: types::ChunkKey, cam_pos_m: Vec3) {
    
    let nx = (2 * config::KEEP_RADIUS + 1) as i32;
    let nz = nx;
    let len = (nx * nz) as usize;

    if mgr.vis.col_visible.len() != len {
        mgr.vis.col_visible.resize(len, 0);
    }
    mgr.vis.col_visible.fill(0);

    
    let vs = config::VOXEL_SIZE_M_F32;
    let cam_x_vox = (cam_pos_m.x / vs) as f32;
    let cam_y_vox = (cam_pos_m.y / vs) as f32;
    let cam_z_vox = (cam_pos_m.z / vs) as f32;

    
    let mut max_slope = vec![f32::NEG_INFINITY; ANG_BINS];

    
    
    let mut cols: Vec<(usize, i32, usize, f32)> = Vec::with_capacity(len);

    let [ox, _, oz] = mgr.grid.grid_origin_chunk;
    let cs = config::CHUNK_SIZE as f32;

    for dz in 0..nz {
        for dx in 0..nx {
            let cx = ox + dx;
            let cz = oz + dz;

            let ddx = (cx - center.x) as i32;
            let ddz = (cz - center.z) as i32;

            
            let r = config::KEEP_RADIUS;
            if ddx*ddx + ddz*ddz > r*r {
                continue;
            }

            
            if ddx*ddx + ddz*ddz <= config::ACTIVE_RADIUS*config::ACTIVE_RADIUS {
                if let Some(idx) = ground::column_index(mgr, cx, cz) {
                    mgr.vis.col_visible[idx] = 1;
                }
                continue;
            }

            let Some(idx) = ground::column_index(mgr, cx, cz) else { continue; };
            let Some(h_vox_i32) = ground::ground_y_vox_for_column(mgr, cx, cz) else { continue; };

            
            let wx_vox = (cx as f32) * cs + (cs * 0.5);
            let wz_vox = (cz as f32) * cs + (cs * 0.5);

            let vx = wx_vox - cam_x_vox;
            let vz = wz_vox - cam_z_vox;

            let dist2 = vx*vx + vz*vz;
            if dist2 < 1e-4 {
                
                mgr.vis.col_visible[idx] = 1;
                continue;
            }

            let h_vox = h_vox_i32 as f32;
            let slope = (h_vox - cam_y_vox) / dist2.sqrt(); 

            let ang = vz.atan2(vx); 
            let mut t = ang;
            if t < 0.0 { t += std::f32::consts::TAU; }
            let bin = ((t / std::f32::consts::TAU) * (ANG_BINS as f32)) as usize;
            let bin = bin.min(ANG_BINS - 1);

            
            let d2_chunks = ddx*ddx + ddz*ddz;

            cols.push((bin, d2_chunks, idx, slope));
        }
    }

    
    cols.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    
    let eps = 1e-4_f32;
    for (bin, _d2, idx, slope) in cols {
        if slope > max_slope[bin] + eps {
            mgr.vis.col_visible[idx] = 1;
            max_slope[bin] = slope;
        }
    }

    
    if PADDING_CHUNKS > 0 {
        dilate_visibility(mgr, nx, nz, PADDING_CHUNKS);
    }
}

fn dilate_visibility(mgr: &mut ChunkManager, nx: i32, nz: i32, pad: i32) {
    let old = mgr.vis.col_visible.clone();
    let ix = |x: i32, z: i32| -> usize { (z * nx + x) as usize };

    for z in 0..nz {
        for x in 0..nx {
            let i = ix(x, z);
            if old[i] != 0 {
                continue;
            }
            
            'nbr: for dz in -pad..=pad {
                for dx in -pad..=pad {
                    let xx = x + dx;
                    let zz = z + dz;
                    if xx < 0 || zz < 0 || xx >= nx || zz >= nz { continue; }
                    if old[ix(xx, zz)] != 0 {
                        mgr.vis.col_visible[i] = 1;
                        break 'nbr;
                    }
                }
            }
        }
    }
}

pub fn column_visible(mgr: &ChunkManager, cx: i32, cz: i32) -> bool {
    let Some(idx) = ground::column_index(mgr, cx, cz) else { return false; };
    mgr.vis.col_visible.get(idx).copied().unwrap_or(0) != 0
}
