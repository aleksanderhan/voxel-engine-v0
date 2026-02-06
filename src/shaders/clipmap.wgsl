// src/shaders/clipmap.wgsl
// ------------------------
//
// Toroidal (ring) clipmap storage, but NO LOD selection in shader.
// We always sample a single fixed level (coarsest by default).

// Tunables live in common.wgsl.

struct ClipmapParams {
  levels      : u32,
  res         : u32,
  base_cell_m : f32,
  _pad0       : f32,
  level       : array<vec4<f32>, CLIP_LEVELS_MAX>,
  offset      : array<vec4<u32>, CLIP_LEVELS_MAX>,
};

@group(0) @binding(7) var<uniform> clip : ClipmapParams;
@group(0) @binding(8) var clip_height : texture_2d_array<f32>;

fn imod(a: i32, m: i32) -> i32 {
  var r = a % m;
  if (r < 0) { r = r + m; }
  return r;
}

fn clip_offsets(level: u32) -> vec2<i32> {
  let o = clip.offset[level];
  return vec2<i32>(i32(o.x), i32(o.y));
}

struct HSample { h: f32, ok: bool };

fn clip_height_at_level_ok(world_xz: vec2<f32>, level: u32) -> HSample {
  let res_i = max(i32(clip.res), 1);

  let st = clip_st(world_xz, level);

  let ix = i32(floor(st.x));
  let iz = i32(floor(st.y));

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

fn clip_height_at_level_nearest(world_xz: vec2<f32>, level: u32) -> f32 {
  let res_i = max(i32(clip.res), 1);
  let p = clip.level[level];
  let origin = vec2<f32>(p.x, p.y);
  let cell   = max(p.z, 1e-6);

  // texel space where centers are at N+0.5
  let uv = (world_xz - origin) / cell;
  let st = uv - vec2<f32>(0.5, 0.5);

  // nearest texel center
  let ix = i32(floor(st.x + 0.5));
  let iz = i32(floor(st.y + 0.5));

  // toroidal wrap (no range checks)
  let off = clip_offsets(level);
  let sx = imod(ix + off.x, res_i);
  let sz = imod(iz + off.y, res_i);

  return textureLoad(clip_height, vec2<i32>(sx, sz), i32(level), 0).x;
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
  if (!ENABLE_CLIPMAP) {
    return ClipHit(false, BIG_F32, vec3<f32>(0.0), MAT_AIR);
  }
  if (clip.levels == 0u) {
    return ClipHit(false, BIG_F32, vec3<f32>(0.0), MAT_AIR);
  }

  // Same behavior: only trace when ray points downward.
  if (rd.y >= -1e-4) {
    return ClipHit(false, BIG_F32, vec3<f32>(0.0), MAT_AIR);
  }

  let guard: i32 = 2;
  let nlv: u32   = max(min(clip.levels, CLIP_LEVELS_MAX), 1u);

  var t: f32 = max(t_min, 0.0);

  // Start at finest level that covers start point; only ever move to coarser.
  var p: vec3<f32> = ro + rd * t;
  var lvl: u32     = clip_best_level(p.xz, guard);

  // Initial signed height
  var h0: f32   = clip_height_at_level_nearest(p.xz, lvl);
  if (h0 <= -0.5 * BIG_F32) {
    return ClipHit(false, BIG_F32, vec3<f32>(0.0), MAT_AIR);
  }
  var s_prev: f32 = p.y - h0;
  var t_prev: f32 = t;

  for (var i: u32 = 0u; i < HF_MAX_STEPS; i = i + 1u) {
    if (t > t_max) { break; }

    p = ro + rd * t;

    // Coverage-only "LOD": ensure p is inside chosen level window (coarsen only).
    lvl = clip_ensure_contains(p.xz, lvl, guard);

    let h: f32 = clip_height_at_level_nearest(p.xz, lvl);
    if (h <= -0.5 * BIG_F32) {
      return ClipHit(false, BIG_F32, vec3<f32>(0.0), MAT_AIR);
    }
    let s: f32 = p.y - h;

    // Crossing from above -> below: refine with bisection at SAME lvl.
    if (s <= 0.0 && s_prev > 0.0) {
      var a: f32 = t_prev;
      var b: f32 = t;

      for (var k: u32 = 0u; k < HF_BISECT; k = k + 1u) {
        let m  = 0.5 * (a + b);
        let pm = ro + rd * m;

        let hm = clip_height_at_level(pm.xz, lvl);
        if (hm <= -0.5 * BIG_F32) {
          return ClipHit(false, BIG_F32, vec3<f32>(0.0), MAT_AIR);
        }
        let sm = pm.y - hm;

        if (sm > 0.0) { a = m; } else { b = m; }
      }

      let th = 0.5 * (a + b);
      let ph = ro + rd * th;
      let n  = clip_normal_at_level_2tap(ph.xz, lvl);

      return ClipHit(true, th, n, MAT_GRASS);
    }

    s_prev = s;
    t_prev = t;

    // Step control: vertical clearance + xz-texel travel limit (same policy).
    let vy = max(-rd.y, 0.12);
    let vh = max(length(rd.xz), 1e-4);

    let dt_y  = abs(s) / vy;
    let cell  = clip.level[lvl].z;
    let dt_xz = (6.0 * cell) / vh; // ~6 texels per step in xz

    var dt = min(dt_y, dt_xz);
    dt = clamp(dt, 0.6, HF_DT_MAX);

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

fn clip_level_contains(xz: vec2<f32>, level: u32, guard: i32) -> bool {
  let res_i = max(i32(clip.res), 1);

  let st = clip_st(xz, level);

  let ix0 = i32(floor(st.x));
  let iz0 = i32(floor(st.y));
  let ix1 = ix0 + 1;
  let iz1 = iz0 + 1;

  // Need full bilerp footprint inside [0..res-1]
  return (ix0 >= guard) &&
         (iz0 >= guard) &&
         (ix1 < (res_i - guard)) &&
         (iz1 < (res_i - guard));
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
  if (n == 0u) {
    return 0u;
  }
  var lvl = min(lvl_in, n - 1u);
  loop {
    if (clip_level_contains(xz, lvl, guard) || lvl >= (n - 1u)) { break; }
    lvl = lvl + 1u;
  }
  return lvl;
}

fn clip_st(world_xz: vec2<f32>, level: u32) -> vec2<f32> {
  let p = clip.level[level];
  let origin = vec2<f32>(p.x, p.y);
  let cell   = max(p.z, 1e-6);

  // uv in texel units, texel centers are at N+0.5
  let uv = (world_xz - origin) / cell;

  // st is aligned so integer ix/iz index texel centers (matches clip_height_at_level)
  return uv - vec2<f32>(0.5, 0.5);
}
  
