// src/render/gpu_types.rs
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

    pub grid_origin_chunk: [i32; 4], // (cx0, cy0, cz0, unused)
    pub grid_dims: [u32; 4],         // (nx, ny, nz, unused)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct OverlayGpu {
    pub fps: u32,
    pub width: u32,
    pub height: u32,
    pub _pad0: u32,
}
