// src/render/gpu_types.rs
// -----------------------

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CameraGpu {
    pub view_inv: [[f32; 4]; 4],
    pub proj_inv: [[f32; 4]; 4],
    pub cam_pos: [f32; 4],
    // x=time_seconds, y=fog_density, z=voxel_size_m, w=seed (float bits)
    pub params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct OverlayGpu {
    pub fps: u32,
    pub width: u32,
    pub height: u32,
    pub _pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct StreamGpu {
    pub origin_x: i32,
    pub origin_y: i32,
    pub origin_z: i32,
    pub _pad0: i32,

    pub ox: u32,
    pub oy: u32,
    pub oz: u32,
    pub dirty_count: u32,

    // NEW: batching controls
    pub dirty_offset: u32, // start index into dirty_slots[]
    pub build_count: u32,  // how many dirty slots to process this frame
    pub _pad1: u32,
    pub _pad2: u32,
}
