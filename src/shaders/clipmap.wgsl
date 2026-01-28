// src/shaders/clipmap.wgsl
//
// Clipmap heightfield sampling + ray intersection fallback.
//
// This file is concatenated AFTER ray_core.wgsl, so it can reuse:
// - MAT_* constants
// - SUN_DIR / SUN_COLOR / SUN_INTENSITY
// - sky_color(), cloud_sun_transmittance()
// - fog helpers
// - etc.
//
// It is also concatenated BEFORE ray_main.wgsl, so ray_main can call functions here.
//
// Bindings are only present in the PRIMARY compute pass bind group (group(0)):
// - @binding(6) clipmap uniform
// - @binding(7) clipmap height texture array

const CLIP_LEVELS : u32 = 5u;
const CLIP_RES    : i32 = 256; // must match config::CLIPMAP_RES

struct ClipmapParams {
  levels     : u32,
  res        : u32,
  base_cell_m: f32,
  _pad0      : f32,
  // x=origin_x_m, y=origin_z_m, z=cell_size_m, w=inv_cell_size_m
  level      : array<vec4<f32>, 5>,
};

@group(0) @binding(6) var<uniform> clip : ClipmapParams;
@group(0) @binding(7) var clip_height : texture_2d_array<f32>;

fn clip_choose_level(xz: vec2<f32>) -> u32 {
  // Choose the finest level that still comfortably contains the query point.
  // We use a conservative "inner" coverage to avoid sampling right on the edges.
  let cam_xz = vec2<f32>(cam.cam_pos.x, cam.cam_pos.z);
  let d = max(abs(xz.x - cam_xz.x), abs(xz.y - cam_xz.y));

  // half coverage per level (in meters): (CLIP_RES * cell_size)/2
  // inner = 0.45*half.
  var best: u32 = CLIP_LEVELS - 1u;

  for (var i: u32 = 0u; i < CLIP_LEVELS; i = i + 1u) {
    let cell = clip.level[i].z;
    let half = 0.5 * f32(CLIP_RES) * cell;
    let inner = 0.45 * half;
    if (d <= inner) { best = i; break; }
  }

  return best;
}

fn clip_height_at_level(xz: vec2<f32>, level: u32) -> f32 {
  let p = clip.level[level];
  let ox = p.x;
  let oz = p.y;
  let inv = p.w;

  let u = (xz.x - ox) * inv;
  let v = (xz.y - oz) * inv;

  let ix = clamp(i32(floor(u)), 0, CLIP_RES - 1);
  let iz = clamp(i32(floor(v)), 0, CLIP_RES - 1);

  // NOTE: texture_2d_array load signature is:
  // textureLoad(tex, coords, array_index, mip_level)
  return textureLoad(clip_height, vec2<i32>(ix, iz), i32(level), 0).x;
}

fn clip_height_at(xz: vec2<f32>) -> f32 {
  let lvl = clip_choose_level(xz);
  return clip_height_at_level(xz, lvl);
}

fn clip_normal_at(xz: vec2<f32>) -> vec3<f32> {
  let lvl = clip_choose_level(xz);
  let p = clip.level[lvl];
  let cell = p.z;

  let hL = clip_height_at_level(xz + vec2<f32>(-cell, 0.0), lvl);
  let hR = clip_height_at_level(xz + vec2<f32>( cell, 0.0), lvl);
  let hD = clip_height_at_level(xz + vec2<f32>(0.0, -cell), lvl);
  let hU = clip_height_at_level(xz + vec2<f32>(0.0,  cell), lvl);

  let dx = (hR - hL) / max(2.0 * cell, 1e-4);
  let dz = (hU - hD) / max(2.0 * cell, 1e-4);

  return normalize(vec3<f32>(-dx, 1.0, -dz));
}

struct ClipHit {
  hit : bool,
  t   : f32,
  n   : vec3<f32>,
  mat : u32,
};

fn clip_trace_heightfield(ro: vec3<f32>, rd: vec3<f32>, t_min: f32, t_max: f32) -> ClipHit {
  // Only meaningful if we're not pointing upward too much.
  if (rd.y >= -1e-4) {
    return ClipHit(false, BIG_F32, vec3<f32>(0.0), MAT_AIR);
  }

  var t = max(t_min, 0.0);
  var p = ro + rd * t;
  var h = clip_height_at(p.xz);
  var s_prev = p.y - h;
  var t_prev = t;

  // Adaptive marching using vertical gap / rd.y as a step estimate.
  for (var i: u32 = 0u; i < 160u; i = i + 1u) {
    if (t > t_max) { break; }

    p = ro + rd * t;
    h = clip_height_at(p.xz);
    let s = p.y - h;

    if (s <= 0.0 && s_prev > 0.0) {
      // Bracketed hit -> refine with bisection.
      var a = t_prev;
      var b = t;
      for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let m = 0.5 * (a + b);
        let pm = ro + rd * m;
        let hm = clip_height_at(pm.xz);
        let sm = pm.y - hm;
        if (sm > 0.0) { a = m; } else { b = m; }
      }
      let th = 0.5 * (a + b);
      let ph = ro + rd * th;

      let n = clip_normal_at(ph.xz);

      // Simple material choice (you can extend with a material-id clipmap later).
      return ClipHit(true, th, n, MAT_GRASS);
    }

    s_prev = s;
    t_prev = t;

    // dt from vertical distance; clamp keeps it stable.
    let vy = max(-rd.y, 0.15);
    var dt = clamp(abs(s) / vy, 0.50, 16.0);
    t = t + dt;
  }

  return ClipHit(false, BIG_F32, vec3<f32>(0.0), MAT_AIR);
}

// Far-terrain shading (no voxel occluder shadows).
fn shade_clip_hit(ro: vec3<f32>, rd: vec3<f32>, ch: ClipHit) -> vec3<f32> {
  let hp = ro + ch.t * rd;

  let base = color_for_material(ch.mat);

  let cloud = cloud_sun_transmittance(hp, SUN_DIR);
  let diff = max(dot(ch.n, SUN_DIR), 0.0);

  let ambient = 0.22;
  let direct = SUN_COLOR * SUN_INTENSITY * diff * cloud;

  return base * (ambient + (1.0 - ambient) * direct);
}
