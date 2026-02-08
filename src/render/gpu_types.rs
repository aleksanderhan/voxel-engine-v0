// src/render/gpu_types.rs
// -----------------------
//
// Fix: ClipLevelParams now has `packed_offsets`, but we keep its existing
// `inv_cell_size_m` field on CPU side.
// GPU uniform uses vec4 per level with packed offsets in .w.

use bytemuck::{Pod, Zeroable};
use crate::app::config;

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
    pub ray00: [f32; 4],
    pub ray_dx: [f32; 4],
    pub ray_dy: [f32; 4],

    pub chunk_size: u32,
    pub chunk_count: u32,
    pub max_steps: u32,
    pub frame_index: u32,

    pub voxel_params: [f32; 4],

    pub grid_origin_chunk: [i32; 4],
    pub grid_dims: [u32; 4],

    pub render_present_px: [u32; 4],

    pub profile_flags: u32,

    // IMPORTANT: WGSL uniform layout aligns the next vec3<u32> to 16 bytes.
    // So we must add 8 bytes here to reach offset 416.
    pub profile_mode: u32,
    pub _pad_profile: [u32; 2],

    // Now keep the rest 16B-aligned blocks.
    pub _pad0: [u32; 4],
    pub _pad1: [u32; 4],
    pub _pad2: [u32; 4],
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
    pub stride:   u32,

    // --- NEW: mode label ("DIG", "DIRT", "STONE", ...)
    pub text_len: u32, // number of chars <= 20
    pub text_p0:  u32, // 4 ASCII bytes packed little-endian
    pub text_p1:  u32, // 4 ASCII bytes
    pub text_p2:  u32, // 4 ASCII bytes
    pub text_p3:  u32, // 4 ASCII bytes
    pub text_p4:  u32, // 4 ASCII bytes

    // Profiling HUD lines (up to 5 lines, 20 chars each)
    pub prof0_len: u32,
    pub prof0_p0:  u32,
    pub prof0_p1:  u32,
    pub prof0_p2:  u32,
    pub prof0_p3:  u32,
    pub prof0_p4:  u32,

    pub prof1_len: u32,
    pub prof1_p0:  u32,
    pub prof1_p1:  u32,
    pub prof1_p2:  u32,
    pub prof1_p3:  u32,
    pub prof1_p4:  u32,

    pub prof2_len: u32,
    pub prof2_p0:  u32,
    pub prof2_p1:  u32,
    pub prof2_p2:  u32,
    pub prof2_p3:  u32,
    pub prof2_p4:  u32,

    pub prof3_len: u32,
    pub prof3_p0:  u32,
    pub prof3_p1:  u32,
    pub prof3_p2:  u32,
    pub prof3_p3:  u32,
    pub prof3_p4:  u32,

    pub prof4_len: u32,
    pub prof4_p0:  u32,
    pub prof4_p1:  u32,
    pub prof4_p2:  u32,
    pub prof4_p3:  u32,
    pub prof4_p4:  u32,

    // pad to 16-byte boundary
    pub _pad0: u32,
}


fn pack4(a: u8, b: u8, c: u8, d: u8) -> u32 {
    (a as u32)
        | ((b as u32) << 8)
        | ((c as u32) << 16)
        | ((d as u32) << 24)
}

fn pack_text_20(s: &str) -> (u32, u32, u32, u32, u32, u32) {
    // Uppercase, ASCII only, pad with spaces, clamp to 20 chars.
    let mut buf = [b' '; 20];
    for (i, ch) in s.bytes().take(20).enumerate() {
        let u = if (b'a'..=b'z').contains(&ch) { ch - 32 } else { ch };
        buf[i] = u;
    }

    let len = s.len().min(20) as u32;
    let p0 = pack4(buf[0], buf[1], buf[2], buf[3]);
    let p1 = pack4(buf[4], buf[5], buf[6], buf[7]);
    let p2 = pack4(buf[8], buf[9], buf[10], buf[11]);
    let p3 = pack4(buf[12], buf[13], buf[14], buf[15]);
    let p4 = pack4(buf[16], buf[17], buf[18], buf[19]);
    (len, p0, p1, p2, p3, p4)
}

impl OverlayGpu {
    pub fn from_fps_edit_profile(
        fps: u32,
        edit_mat: u32,
        width: u32,
        _height: u32,
        scale: u32,
        profile_lines: &[&str],
    ) -> Self {
        // ---- FPS digits ----
        let mut v = fps.min(9999);
        let d0 = (v % 10) as u32; v /= 10;
        let d1 = (v % 10) as u32; v /= 10;
        let d2 = (v % 10) as u32; v /= 10;
        let d3 = (v % 10) as u32;
        let digits_packed = d0 | (d1 << 8) | (d2 << 16) | (d3 << 24);

        // ---- Layout (same as before) ----
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

        // ---- Label ----
        // IMPORTANT: use material IDs -> strings (NOT raw ID bytes).
        let label: &str = match edit_mat {
            crate::world::materials::AIR   => "AIR",
            crate::world::materials::DIRT  => "DIRT",
            crate::world::materials::STONE => "STONE",
            crate::world::materials::WOOD  => "WOOD",
            crate::world::materials::LIGHT => "LIGHT",
            _ => "UNKNOWN", // fallback;
        };

        let (text_len, text_p0, text_p1, text_p2, text_p3, text_p4) = pack_text_20(label);

        let mut prof_lines = [(0u32, 0u32, 0u32, 0u32, 0u32, 0u32); 5];
        for (i, line) in profile_lines.iter().take(5).enumerate() {
            prof_lines[i] = pack_text_20(line);
        }

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
            text_p3,
            text_p4,
            prof0_len: prof_lines[0].0,
            prof0_p0: prof_lines[0].1,
            prof0_p1: prof_lines[0].2,
            prof0_p2: prof_lines[0].3,
            prof0_p3: prof_lines[0].4,
            prof0_p4: prof_lines[0].5,

            prof1_len: prof_lines[1].0,
            prof1_p0: prof_lines[1].1,
            prof1_p1: prof_lines[1].2,
            prof1_p2: prof_lines[1].3,
            prof1_p3: prof_lines[1].4,
            prof1_p4: prof_lines[1].5,

            prof2_len: prof_lines[2].0,
            prof2_p0: prof_lines[2].1,
            prof2_p1: prof_lines[2].2,
            prof2_p2: prof_lines[2].3,
            prof2_p3: prof_lines[2].4,
            prof2_p4: prof_lines[2].5,

            prof3_len: prof_lines[3].0,
            prof3_p0: prof_lines[3].1,
            prof3_p1: prof_lines[3].2,
            prof3_p2: prof_lines[3].3,
            prof3_p3: prof_lines[3].4,
            prof3_p4: prof_lines[3].5,

            prof4_len: prof_lines[4].0,
            prof4_p0: prof_lines[4].1,
            prof4_p1: prof_lines[4].2,
            prof4_p2: prof_lines[4].3,
            prof4_p3: prof_lines[4].4,
            prof4_p4: prof_lines[4].5,

            _pad0: 0,
        }
    }
}
