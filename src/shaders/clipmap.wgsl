// src/shaders/clipmap.wgsl
// ------------------------
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

// NOTE: levels/res are provided by the uniform, but we keep a compile-time max here
// to match the fixed-size uniform array.
const CLIP_LEVELS_MAX : u32 = 5u;

// March tuning (performance critical)
const HF_MAX_STEPS : u32 = 80u;
const HF_BISECT    : u32 = 6u;

// dt clamp (meters along ray)
const HF_DT_MIN : f32 = 1.00;
const HF_DT_MAX : f32 = 48.0;

struct ClipmapParams {
  levels      : u32,
  res         : u32,
  base_cell_m : f32,
  _pad0       : f32,
  // x=origin_x_m, y=origin_z_m, z=cell_size_m, w=inv_cell_size_m
  level       : array<vec4<f32>, 5>,
};

@group(0) @binding(6) var<uniform> clip : ClipmapParams;
@group(0) @binding(7) var clip_height : texture_2d_array<f32>;

// --------------------------
// Level selection
// --------------------------

fn clip_choose_level(xz: vec2<f32>) -> u32 {
  // Choose the finest level that still comfortably contains the query point.
  // Conservative inner coverage avoids sampling near edges.
  let cam_xz = vec2<f32>(cam.cam_pos.x, cam.cam_pos.z);
  let d = max(abs(xz.x - cam_xz.x), abs(xz.y - cam_xz.y));

  let res_i = max(i32(clip.res), 1);
  let res_f = f32(res_i);

  // Default to coarsest.
  var best: u32 = max(clip.levels, 1u) - 1u;

  let n = min(clip.levels, CLIP_LEVELS_MAX);
  for (var i: u32 = 0u; i < n; i = i + 1u) {
    let cell = clip.level[i].z;
    let half = 0.5 * res_f * cell;
    let inner = 0.45 * half;
    if (d <= inner) { best = i; break; }
  }

  return best;
}

// --------------------------
// Height sampling (fast path)
// --------------------------

fn clip_height_at_level(world_xz: vec2<f32>, level: u32) -> f32 {
  let res_i = max(i32(clip.res), 1);

  let p = clip.level[level];
  let origin = vec2<f32>(p.x, p.y);
  let inv_cell = p.w;

  // Map world -> texel
  // CPU wrote heights for texel centers at (tx+0.5)*cell, but sampling by floor() here
  // is consistent for heightfield marching and stable across movement due to snapping.
  let uv = (world_xz - origin) * inv_cell;

  var ix = i32(floor(uv.x));
  var iz = i32(floor(uv.y));

  ix = clamp(ix, 0, res_i - 1);
  iz = clamp(iz, 0, res_i - 1);

  return textureLoad(clip_height, vec2<i32>(ix, iz), i32(level), 0).x;
}

fn clip_height_at(xz: vec2<f32>) -> f32 {
  let lvl = clip_choose_level(xz);
  return clip_height_at_level(xz, lvl);
}

// --------------------------
// Normal (2-tap)
// --------------------------

fn clip_normal_at_level_2tap(world_xz: vec2<f32>, level: u32) -> vec3<f32> {
  let cell = clip.level[level].z;

  // 2-tap forward differences (2 texture loads + 1 reuse)
  let h  = clip_height_at_level(world_xz, level);
  let hx = clip_height_at_level(world_xz + vec2<f32>(cell, 0.0), level);
  let hz = clip_height_at_level(world_xz + vec2<f32>(0.0, cell), level);

  let dhx = (hx - h) / max(cell, 1e-4);
  let dhz = (hz - h) / max(cell, 1e-4);

  return normalize(vec3<f32>(-dhx, 1.0, -dhz));
}

fn clip_normal_at(xz: vec2<f32>) -> vec3<f32> {
  let lvl = clip_choose_level(xz);
  return clip_normal_at_level_2tap(xz, lvl);
}

// --------------------------
// Trace
// --------------------------

struct ClipHit {
  hit : bool,
  t   : f32,
  n   : vec3<f32>,
  mat : u32,
};

fn clip_trace_heightfield(ro: vec3<f32>, rd: vec3<f32>, t_min: f32, t_max: f32) -> ClipHit {
  // If we're not pointing downward, don't bother.
  if (rd.y >= -1e-4) {
    return ClipHit(false, BIG_F32, vec3<f32>(0.0), MAT_AIR);
  }

  // Start
  var t = max(t_min, 0.0);

  var p = ro + rd * t;

  // Cache the chosen clip level (avoid repeated clip_choose_level inside helpers)
  var lvl: u32 = clip_choose_level(p.xz);

  var h = clip_height_at_level(p.xz, lvl);
  var s_prev = p.y - h;
  var t_prev = t;

  for (var i: u32 = 0u; i < HF_MAX_STEPS; i = i + 1u) {
    if (t > t_max) { break; }

    p = ro + rd * t;

    // Update cached level once per step (not per sample)
    lvl = clip_choose_level(p.xz);

    h = clip_height_at_level(p.xz, lvl);
    let s = p.y - h;

    // Bracketed hit -> bisection refine
    if (s <= 0.0 && s_prev > 0.0) {
      var a = t_prev;
      var b = t;

      for (var k: u32 = 0u; k < HF_BISECT; k = k + 1u) {
        let m = 0.5 * (a + b);
        let pm = ro + rd * m;

        let mlvl = clip_choose_level(pm.xz);
        let hm = clip_height_at_level(pm.xz, mlvl);
        let sm = pm.y - hm;

        if (sm > 0.0) { a = m; } else { b = m; }
      }

      let th = 0.5 * (a + b);
      let ph = ro + rd * th;

      let hlvl = clip_choose_level(ph.xz);
      let n = clip_normal_at_level_2tap(ph.xz, hlvl);

      return ClipHit(true, th, n, MAT_GRASS);
    }

    s_prev = s;
    t_prev = t;

    // Adaptive dt from vertical gap (more aggressive than before).
    let vy = max(-rd.y, 0.12);
    var dt = abs(s) / vy;
    dt = clamp(dt, HF_DT_MIN, HF_DT_MAX);

    t = t + dt;
  }

  return ClipHit(false, BIG_F32, vec3<f32>(0.0), MAT_AIR);
}

// --------------------------
// Shading
// --------------------------

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
