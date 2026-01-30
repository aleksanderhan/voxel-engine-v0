// src/render/gpu_types.rs
// -----------------------
//
// Fix: ClipLevelParams now has `packed_offsets`, but we keep its existing
// `inv_cell_size_m` field on CPU side.
// GPU uniform uses vec4 per level with packed offsets in .w.

use bytemuck::{Pod, Zeroable};
use crate::config;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct NodeGpu {
    pub child_base: u32,
    pub child_mask: u32,
    pub material: u32,
    pub key: u32, // packed spatial key: level + coord at that level
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct NodeRopesGpu {
    pub px: u32,
    pub nx: u32,
    pub py: u32,
    pub ny: u32,
    pub pz: u32,
    pub nz: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}


#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ChunkMetaGpu {
    pub origin: [i32; 4],
    pub node_base: u32,
    pub node_count: u32,
    pub macro_base: u32,
    pub colinfo_base: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CameraGpu {
    pub view_inv: [[f32; 4]; 4],
    pub proj_inv: [[f32; 4]; 4],
    pub cam_pos: [f32; 4],

    pub chunk_size: u32,
    pub chunk_count: u32,
    pub max_steps: u32,
    pub frame_index: u32,

    pub voxel_params: [f32; 4],

    pub grid_origin_chunk: [i32; 4],
    pub grid_dims: [u32; 4],

    pub render_present_px: [u32; 4],
}

/// Clipmap uniform payload.
///
/// Matches `shaders/clipmap.wgsl`.
///
/// Per level vec4<f32>:
///   x = origin_x_m
///   y = origin_z_m
///   z = cell_size_m
///   w = unused (0)
///
/// Per level vec4<u32>:
///   x = off_x (toroidal offset in texels)
///   y = off_z
///   z/w unused (0)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ClipmapGpu {
    pub levels: u32,
    pub res: u32,
    pub base_cell_m: f32,
    pub _pad0: f32,

    pub level:  [[f32; 4]; config::CLIPMAP_LEVELS_USIZE],
    pub offset: [[u32; 4]; config::CLIPMAP_LEVELS_USIZE],
}

impl ClipmapGpu {
    pub fn from_cpu(cpu: &crate::clipmap::ClipmapParamsCpu) -> Self {
        let mut level  = [[0.0f32; 4]; config::CLIPMAP_LEVELS_USIZE];
        let mut offset = [[0u32; 4]; config::CLIPMAP_LEVELS_USIZE];

        for i in 0..config::CLIPMAP_LEVELS_USIZE {
            let p = cpu.level[i];

            level[i] = [p.origin_x_m, p.origin_z_m, p.cell_size_m, 0.0];

            // NOTE: these fields change on CPU side in the next patch (ClipLevelParams)
            offset[i] = [p.off_x, p.off_z, 0, 0];
        }

        Self {
            levels: cpu.levels,
            res: cpu.res,
            base_cell_m: cpu.base_cell_m,
            _pad0: 0.0,
            level,
            offset,
        }
    }
}


#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct OverlayGpu {
    // packed digits: d0 | d1<<8 | d2<<16 | d3<<24 (d0=ones, d3=thousands)
    pub digits_packed: u32,

    // HUD rectangle in framebuffer pixel coords (top-left origin)
    pub origin_x: u32,
    pub origin_y: u32,
    pub total_w:  u32,

    pub digit_h:  u32,
    pub scale:    u32,
    pub stride:   u32, // digit_w + gap
    pub _pad0:    u32, // explicit padding to 32 bytes
}

impl OverlayGpu {
    pub fn from_fps_and_dims(fps: u32, width: u32, _height: u32, scale: u32) -> Self {
        // digits
        let mut v = fps.min(9999);
        let d0 = (v % 10) as u32; v /= 10;
        let d1 = (v % 10) as u32; v /= 10;
        let d2 = (v % 10) as u32; v /= 10;
        let d3 = (v % 10) as u32;

        let digits_packed = d0 | (d1 << 8) | (d2 << 16) | (d3 << 24);

        // layout
        let margin: u32 = 12;
        let digit_w = 3 * scale;
        let digit_h = 5 * scale;
        let gap     = 1 * scale;
        let stride  = digit_w + gap;
        let total_w = 4 * digit_w + 3 * gap;

        let ox_i = width as i32 - margin as i32 - total_w as i32;
        let oy_i = margin as i32;

        let origin_x = ox_i.max(0) as u32;
        let origin_y = oy_i.max(0) as u32;

        Self {
            digits_packed,
            origin_x,
            origin_y,
            total_w,
            digit_h,
            scale,
            stride,
            _pad0: 0,
        }
    }
}
