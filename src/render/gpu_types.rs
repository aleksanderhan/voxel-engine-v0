






use bytemuck::{Pod, Zeroable};
use crate::app::config;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct NodeGpu {
    pub child_base: u32,
    pub child_mask: u32,
    pub material: u32,
    pub key: u32, 
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
    pub macro_empty: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CameraGpu {
    pub view_inv: [[f32; 4]; 4],
    pub proj_inv: [[f32; 4]; 4],

    pub view_proj: [[f32; 4]; 4],
    pub prev_view_proj: [[f32; 4]; 4],
    
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
    
    pub digits_packed: u32,

    
    pub origin_x: u32,
    pub origin_y: u32,
    pub total_w:  u32,

    pub digit_h:  u32,
    pub scale:    u32,
    pub stride:   u32,

    
    pub text_len: u32, 
    pub text_p0:  u32, 
    pub text_p1:  u32, 
    pub text_p2:  u32, 

    
    pub _pad0:    u32,
}


fn pack4(a: u8, b: u8, c: u8, d: u8) -> u32 {
    (a as u32)
        | ((b as u32) << 8)
        | ((c as u32) << 16)
        | ((d as u32) << 24)
}

fn pack_text_12(s: &str) -> (u32, u32, u32, u32) {
    
    let mut buf = [b' '; 12];
    for (i, ch) in s.bytes().take(12).enumerate() {
        let u = if (b'a'..=b'z').contains(&ch) { ch - 32 } else { ch };
        buf[i] = u;
    }

    let len = s.len().min(12) as u32;
    let p0 = pack4(buf[0], buf[1], buf[2], buf[3]);
    let p1 = pack4(buf[4], buf[5], buf[6], buf[7]);
    let p2 = pack4(buf[8], buf[9], buf[10], buf[11]);
    (len, p0, p1, p2)
}

impl OverlayGpu {
    pub fn from_fps_and_edit(
        fps: u32,
        edit_mat: u32,
        width: u32,
        _height: u32,
        scale: u32,
    ) -> Self {
        
        let mut v = fps.min(9999);
        let d0 = (v % 10) as u32; v /= 10;
        let d1 = (v % 10) as u32; v /= 10;
        let d2 = (v % 10) as u32; v /= 10;
        let d3 = (v % 10) as u32;
        let digits_packed = d0 | (d1 << 8) | (d2 << 16) | (d3 << 24);

        
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

        
        
        let label: &str = match edit_mat {
            crate::world::materials::AIR   => "AIR",
            crate::world::materials::DIRT  => "DIRT",
            crate::world::materials::STONE => "STONE",
            crate::world::materials::WOOD  => "WOOD",
            crate::world::materials::LIGHT => "LIGHT",
            _ => "UNKNOWN", 
        };

        let (text_len, text_p0, text_p1, text_p2) = pack_text_12(label);

        Self {
            digits_packed,
            origin_x,
            origin_y,
            total_w,
            digit_h,
            scale,
            stride,
            text_len,
            text_p0,
            text_p1,
            text_p2,
            _pad0: 0,
        }
    }
}
