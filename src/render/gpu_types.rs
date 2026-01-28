// src/render/gpu_types.rs
// -----------------------
//
// Fix: ClipLevelParams now has `packed_offsets`, but we keep its existing
// `inv_cell_size_m` field on CPU side.
// GPU uniform uses vec4 per level with packed offsets in .w.

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct NodeGpu {
    pub child_base: u32,
    pub child_mask: u32,
    pub material: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ChunkMetaGpu {
    pub origin: [i32; 4],
    pub node_base: u32,
    pub node_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
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
    pub _pad0: u32,

    pub voxel_params: [f32; 4],

    pub grid_origin_chunk: [i32; 4],
    pub grid_dims: [u32; 4],
}

/// Clipmap uniform payload.
///
/// Matches `shaders/clipmap.wgsl`.
///
/// Per level vec4:
///   x = origin_x_m
///   y = origin_z_m
///   z = cell_size_m
///   w = packed offsets bits (off_x u16 | (off_z u16 << 16)) stored via f32::from_bits
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ClipmapGpu {
    pub levels: u32,
    pub res: u32,
    pub base_cell_m: f32,
    pub _pad0: f32,

    pub level: [[f32; 4]; crate::config::CLIPMAP_LEVELS_USIZE],
}

impl ClipmapGpu {
    pub fn from_cpu(cpu: &crate::clipmap::ClipmapParamsCpu) -> Self {
        let mut level = [[0.0f32; 4]; crate::config::CLIPMAP_LEVELS_USIZE];

        for i in 0..crate::config::CLIPMAP_LEVELS_USIZE {
            let p = cpu.level[i];
            level[i] = [
                p.origin_x_m,
                p.origin_z_m,
                p.cell_size_m,
                f32::from_bits(p.packed_offsets),
            ];
        }

        Self {
            levels: cpu.levels,
            res: cpu.res,
            base_cell_m: cpu.base_cell_m,
            _pad0: 0.0,
            level,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct OverlayGpu {
    pub fps: u32,
    pub width: u32,
    pub height: u32,
    pub _pad0: u32,
}
