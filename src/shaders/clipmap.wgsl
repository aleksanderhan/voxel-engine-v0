// src/shaders/clipmap.wgsl
// ------------------------
//
// Toroidal (ring) clipmap storage, but NO LOD selection in shader.
// We always sample a single fixed level (coarsest by default).

const CLIP_LEVELS_MAX : u32 = 16u;

// March tuning
const HF_MAX_STEPS : u32 = 96u;
const HF_BISECT    : u32 = 5u;

// dt clamp (meters along ray)
const HF_DT_MIN : f32 = 1.00;
const HF_DT_MAX : f32 = 48.0;

struct ClipmapParams {
  levels      : u32,
  res         : u32,
  base_cell_m : f32,
  _pad0       : f32,
  level       : array<vec4<f32>, CLIP_LEVELS_MAX>,
  offset      : array<vec4<u32>, CLIP_LEVELS_MAX>,
};

@group(0) @binding(6) var<uniform> clip : ClipmapParams;
@group(0) @binding(7) var clip_height : texture_2d_array<f32>;

fn imod(a: i32, m: i32) -> i32 {
  var r = a % m;
  if (r < 0) { r = r + m; }
  return r;
}

fn clip_offsets(level: u32) -> vec2<i32> {
  let o = clip.offset[level];
  return vec2<i32>(i32(o.x), i32(o.y));
}

// Fixed level choice: coarsest level (max range).
// If you want a specific level index, return that instead.
fn clip_fixed_level() -> u32 {
  let n = min(clip.levels, CLIP_LEVELS_MAX);
  return max(n, 1u) - 1u;
}

struct HSample { h: f32, ok: bool };

fn clip_height_at_level_ok(world_xz: vec2<f32>, level: u32) -> HSample {
  let res_i = max(i32(clip.res), 1);
  let p = clip.level[level];
  let origin = vec2<f32>(p.x, p.y);
  let cell   = max(p.z, 1e-6);

  let uv = (world_xz - origin) / cell;
  let ix = i32(floor(uv.x));
  let iz = i32(floor(uv.y));

  if (ix < 0 || iz < 0 || ix >= res_i || iz >= res_i) {
    return HSample(0.0, false);
  }

  let off = clip_offsets(level);
  let sx = imod(ix + off.x, res_i);
  let sz = imod(iz + off.y, res_i);

  let h = textureLoad(clip_height, vec2<i32>(sx, sz), i32(level), 0).x;
  return HSample(h, true);
}

fn clip_height_texel(level: u32, ix: i32, iz: i32) -> f32 {
  let res_i = max(i32(clip.res), 1);

  if (ix < 0 || iz < 0 || ix >= res_i || iz >= res_i) {
    return -BIG_F32;
  }

  let off = clip_offsets(level);
  let sx = imod(ix + off.x, res_i);
  let sz = imod(iz + off.y, res_i);

  return textureLoad(clip_height, vec2<i32>(sx, sz), i32(level), 0).x;
}

// Bilinear height sample. IMPORTANT: matches CPU building at (tx+0.5)*cell.
// We shift by 0.5 so integer texels correspond to sample centers.
fn clip_height_at_level(world_xz: vec2<f32>, level: u32) -> f32 {
  let res_i = max(i32(clip.res), 1);
  let p = clip.level[level];
  let origin = vec2<f32>(p.x, p.y);
  let cell   = max(p.z, 1e-6);

  // uv in texel units, where texel centers are at N+0.5
  let uv = (world_xz - origin) / cell;

  // Align so ix,iz index texel centers
  let st = uv - vec2<f32>(0.5, 0.5);

  let ix0 = i32(floor(st.x));
  let iz0 = i32(floor(st.y));
  let ix1 = ix0 + 1;
  let iz1 = iz0 + 1;

  // If the bilerp footprint goes out of range, mark invalid
  if (ix0 < 0 || iz0 < 0 || ix1 >= res_i || iz1 >= res_i) {
    return -BIG_F32;
  }

  let fx = fract(st.x);
  let fz = fract(st.y);

  let h00 = clip_height_texel(level, ix0, iz0);
  let h10 = clip_height_texel(level, ix1, iz0);
  let h01 = clip_height_texel(level, ix0, iz1);
  let h11 = clip_height_texel(level, ix1, iz1);

  if (h00 <= -0.5 * BIG_F32 || h10 <= -0.5 * BIG_F32 || h01 <= -0.5 * BIG_F32 || h11 <= -0.5 * BIG_F32) {
    return -BIG_F32;
  }

  let hx0 = mix(h00, h10, fx);
  let hx1 = mix(h01, h11, fx);
  return mix(hx0, hx1, fz);
}

fn clip_normal_at_level_2tap(world_xz: vec2<f32>, level: u32) -> vec3<f32> {
  let cell = clip.level[level].z;

  let h0 = clip_height_at_level_ok(world_xz, level);
  let hx = clip_height_at_level_ok(world_xz + vec2<f32>(cell, 0.0), level);
  let hz = clip_height_at_level_ok(world_xz + vec2<f32>(0.0, cell), level);

  if (!h0.ok || !hx.ok || !hz.ok) {
    return vec3<f32>(0.0, 1.0, 0.0);
  }

  let dhx = (hx.h - h0.h) / max(cell, 1e-4);
  let dhz = (hz.h - h0.h) / max(cell, 1e-4);

  return normalize(vec3<f32>(-dhx, 1.0, -dhz));
}

struct ClipHit {
  hit : bool,
  t   : f32,
  n   : vec3<f32>,
  mat : u32,
};

fn clip_trace_heightfield(ro: vec3<f32>, rd: vec3<f32>, t_min: f32, t_max: f32) -> ClipHit {
  if (rd.y >= -1e-4) {
    return ClipHit(false, BIG_F32, vec3<f32>(0.0), MAT_AIR);
  }

  var t = max(t_min, 0.0);
  var p = ro + rd * t;

  // Start with the finest level that contains the starting point.
  // From here on, we only move to COARSER levels if/when we fall out of coverage.
  var lvl: u32 = clip_best_level(p.xz, 2);

  var h = clip_height_at_level(p.xz, lvl);
  var s_prev = p.y - h;
  var t_prev = t;

  for (var i: u32 = 0u; i < HF_MAX_STEPS; i = i + 1u) {
    if (t > t_max) { break; }

    p = ro + rd * t;

    // Coverage-only "LOD": ensure p is inside the chosen level window.
    // IMPORTANT: this only ever increases lvl (coarser), never decreases.
    lvl = clip_ensure_contains(p.xz, lvl, 2);

    h = clip_height_at_level(p.xz, lvl);
    let s = p.y - h;

    if (s <= 0.0 && s_prev > 0.0) {
      var a = t_prev;
      var b = t;

      // Bisection against the SAME surface (same lvl), no level switching here.
      for (var k: u32 = 0u; k < HF_BISECT; k = k + 1u) {
        let m = 0.5 * (a + b);
        let pm = ro + rd * m;

        let hm = clip_height_at_level(pm.xz, lvl);
        let sm = pm.y - hm;

        if (sm > 0.0) { a = m; } else { b = m; }
      }

      let th = 0.5 * (a + b);
      let ph = ro + rd * th;

      let n = clip_normal_at_level_2tap(ph.xz, lvl);
      return ClipHit(true, th, n, MAT_GRASS);
    }

    s_prev = s;
    t_prev = t;

    // Step control: vertical clearance + xz-texel travel limit
    let vy = max(-rd.y, 0.12);
    let vh = max(length(rd.xz), 1e-4);

    let dt_y = abs(s) / vy;

    let cell = clip.level[lvl].z;
    let dt_xz = (2.0 * cell) / vh; // ~2 texels per step in xz

    var dt = min(dt_y, dt_xz);

    // Keep your clamp policy (you can swap 0.25 back to HF_DT_MIN if you prefer)
    dt = clamp(dt, 0.25, HF_DT_MAX);

    t = t + dt;
  }

  return ClipHit(false, BIG_F32, vec3<f32>(0.0), MAT_AIR);
}


fn material_variation_clip(world_p: vec3<f32>, cell_size_m: f32, strength: f32) -> f32 {
  let cell = floor(world_p / cell_size_m);
  return (hash31(cell) - 0.5) * strength;
}

fn apply_material_variation_clip(base: vec3<f32>, mat: u32, hp: vec3<f32>) -> vec3<f32> {
  var c = base;

  if (mat == MAT_GRASS) {
    let v0 = material_variation_clip(hp, 3.0, 1.0);
    let v1 = material_variation_clip(hp, 0.75, 0.35);
    let v  = v0 + v1;

    c += vec3<f32>(0.10 * v, 0.18 * v, 0.06 * v);
    c *= (1.0 + 0.06 * v);
    c = clamp(c, vec3<f32>(0.0), vec3<f32>(2.0));
  } else if (mat == MAT_DIRT) {
    let v = material_variation_clip(hp, 1.5, 0.8);
    c += vec3<f32>(0.05 * v, 0.03 * v, 0.01 * v);
    c *= (1.0 + 0.08 * v);
    c = clamp(c, vec3<f32>(0.0), vec3<f32>(2.0));
  } else if (mat == MAT_STONE) {
    let v = material_variation_clip(hp, 2.0, 0.9);
    c *= (1.0 + 0.10 * v);
    c = clamp(c, vec3<f32>(0.0), vec3<f32>(2.0));
  }

  return c;
}

fn shade_clip_hit(ro: vec3<f32>, rd: vec3<f32>, ch: ClipHit, sky_up: vec3<f32>) -> vec3<f32> {
  let hp = ro + ch.t * rd;

  var base = color_for_material(ch.mat);
  base = apply_material_variation_clip(base, ch.mat, hp);

  let voxel_size = cam.voxel_params.x;
  let hp_shadow  = hp + ch.n * (0.75 * voxel_size);

  let vis  = sun_transmittance(hp_shadow, SUN_DIR);
  let diff = max(dot(ch.n, SUN_DIR), 0.0);

  // AO-lite for terrain: cheap taps against heightfield itself (fixed lvl)
  let lvl = clip_best_level(hp.xz, 2);
  let cell = clip.level[lvl].z;


  let h0  = clip_height_at_level(hp.xz, lvl);
  let hx1 = clip_height_at_level(hp.xz + vec2<f32>( cell, 0.0), lvl);
  let hx0 = clip_height_at_level(hp.xz + vec2<f32>(-cell, 0.0), lvl);
  let hz1 = clip_height_at_level(hp.xz + vec2<f32>(0.0,  cell), lvl);
  let hz0 = clip_height_at_level(hp.xz + vec2<f32>(0.0, -cell), lvl);

  let occ =
    max(0.0, hx1 - h0) +
    max(0.0, hx0 - h0) +
    max(0.0, hz1 - h0) +
    max(0.0, hz0 - h0);

  let ao = clamp(1.0 - 0.65 * occ / max(cell, 1e-3), 0.45, 1.0);

  let amb_col = hemi_ambient(ch.n, sky_up);
  let amb_strength = 0.10;
  let ambient = amb_col * amb_strength * ao;

  let direct = SUN_COLOR * SUN_INTENSITY * (diff * diff) * vis;

  let vdir = normalize(-rd);
  let hdir = normalize(vdir + SUN_DIR);
  let ndv  = max(dot(ch.n, vdir), 0.0);
  let ndh  = max(dot(ch.n, hdir), 0.0);

  var rough = 0.85;
  if (ch.mat == MAT_STONE) { rough = 0.50; }
  if (ch.mat == MAT_DIRT)  { rough = 0.90; }
  if (ch.mat == MAT_GRASS) { rough = 0.88; }

  let shininess = mix(8.0, 96.0, 1.0 - rough);
  let spec = pow(ndh, shininess);

  var f0 = 0.03;
  if (ch.mat == MAT_STONE) { f0 = 0.04; }
  let fres = f0 + (1.0 - f0) * pow(1.0 - clamp(ndv, 0.0, 1.0), 5.0);

  let spec_col = SUN_COLOR * SUN_INTENSITY * spec * fres * vis;

  return base * (ambient + direct) + 0.18 * spec_col;
}

fn clip_level_contains(xz: vec2<f32>, level: u32, guard: i32) -> bool {
  let res_i = max(i32(clip.res), 1);
  let p = clip.level[level];
  let origin = vec2<f32>(p.x, p.y);
  let cell   = max(p.z, 1e-6);

  let uv = (xz - origin) / cell;
  let ix = i32(floor(uv.x));
  let iz = i32(floor(uv.y));

  return (ix >= guard) && (iz >= guard) && (ix < (res_i - guard)) && (iz < (res_i - guard));
}

// Finest available level that contains xz (coverage-based).
fn clip_best_level(xz: vec2<f32>, guard: i32) -> u32 {
  let n = min(clip.levels, CLIP_LEVELS_MAX);
  for (var i: u32 = 0u; i < n; i = i + 1u) {
    if (clip_level_contains(xz, i, guard)) { return i; }
  }
  return max(n, 1u) - 1u;
}

// Ensure containment by moving only to coarser levels.
fn clip_ensure_contains(xz: vec2<f32>, lvl_in: u32, guard: i32) -> u32 {
  let n = min(clip.levels, CLIP_LEVELS_MAX);
  var lvl = min(lvl_in, n - 1u);
  loop {
    if (clip_level_contains(xz, lvl, guard) || lvl >= (n - 1u)) { break; }
    lvl = lvl + 1u;
  }
  return lvl;
}
