// src/shaders/clipmap.wgsl
// ------------------------
//
// UPDATED for toroidal (ring) clipmap storage.
// level[i].w stores packed offsets (u16 off_x | (u16 off_z << 16)) via f32 bits.
// We compute inv_cell = 1.0 / cell in shader (since CPU still keeps inv too).

const CLIP_LEVELS_MAX : u32 = 16u;

// March tuning
const HF_MAX_STEPS : u32 = 160u;
const HF_BISECT    : u32 = 6u;

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

fn clip_level_half_extent(level: u32) -> f32 {
  let res_f = f32(max(i32(clip.res), 1));
  let cell  = clip.level[level].z;
  return 0.5 * res_f * cell;
}

// Two thresholds = hysteresis band
// - inner_in: how deep inside a *finer* level you must be before switching down (to finer)
// - outer_out: how far out of the *current* level you must be before switching up (to coarser)
fn clip_inner_in(level: u32) -> f32 {
  // stricter than your old 0.45 * half (switch to finer only if clearly inside)
  return 0.42 * clip_level_half_extent(level);
}

fn clip_outer_out(level: u32) -> f32 {
  // looser than your old 0.45 * half (switch to coarser only if clearly outside)
  return 0.48 * clip_level_half_extent(level);
}

// Raw “best effort” level (your old logic, slightly simplified)
fn clip_choose_level_raw(xz: vec2<f32>) -> u32 {
  let cam_xz = vec2<f32>(cam.cam_pos.x, cam.cam_pos.z);
  let d = max(abs(xz.x - cam_xz.x), abs(xz.y - cam_xz.y));

  let n = min(clip.levels, CLIP_LEVELS_MAX);
  let guard: i32 = 2;

  var best: u32 = max(clip.levels, 1u) - 1u;

  for (var i: u32 = 0u; i < n; i = i + 1u) {
    if (d <= clip_inner_in(i) && clip_level_contains(xz, i, guard)) {
      best = i;
      break;
    }
  }
  return best;
}

// Stateful hysteresis choice (uses previous level as state)
fn clip_choose_level_hyst(xz: vec2<f32>, prev: u32) -> u32 {
  let cam_xz = vec2<f32>(cam.cam_pos.x, cam.cam_pos.z);
  let d = max(abs(xz.x - cam_xz.x), abs(xz.y - cam_xz.y));

  let n = min(clip.levels, CLIP_LEVELS_MAX);
  let guard: i32 = 2;

  var lvl = min(prev, n - 1u);

  // If current level doesn't contain point (guarded), walk coarser until it does.
  loop {
    if (clip_level_contains(xz, lvl, guard) || lvl >= (n - 1u)) { break; }
    lvl = lvl + 1u;
  }

  // Switch to coarser only when clearly outside current threshold
  loop {
    if (lvl >= (n - 1u)) { break; }
    if (d > clip_outer_out(lvl)) {
      lvl = lvl + 1u;
      continue;
    }
    break;
  }

  // Switch to finer only when clearly inside finer threshold
  loop {
    if (lvl == 0u) { break; }
    let fine = lvl - 1u;
    if (d <= clip_inner_in(fine) && clip_level_contains(xz, fine, guard)) {
      lvl = fine;
      continue;
    }
    break;
  }

  return lvl;
}


fn clip_height_at_level(world_xz: vec2<f32>, level: u32) -> f32 {
  let res_i = max(i32(clip.res), 1);
  let p = clip.level[level];
  let origin = vec2<f32>(p.x, p.y);
  let cell   = max(p.z, 1e-6);

  let uv = (world_xz - origin) / cell;
  let ix = i32(floor(uv.x));
  let iz = i32(floor(uv.y));

  // IMPORTANT: don't clamp to edge (that creates the mangling).
  // If we're outside the committed window, return very low terrain.
  if (ix < 0 || iz < 0 || ix >= res_i || iz >= res_i) {
    return -BIG_F32;
  }

  let off = clip_offsets(level);
  let sx = imod(ix + off.x, res_i);
  let sz = imod(iz + off.y, res_i);

  return textureLoad(clip_height, vec2<i32>(sx, sz), i32(level), 0).x;
}

fn clip_height_at(xz: vec2<f32>) -> f32 {
  let lvl = clip_choose_level_raw(xz);
  return clip_height_at_level(xz, lvl);
}


fn clip_normal_at_level_2tap(world_xz: vec2<f32>, level: u32) -> vec3<f32> {
  let cell = clip.level[level].z;

  let h  = clip_height_at_level(world_xz, level);
  let hx = clip_height_at_level(world_xz + vec2<f32>(cell, 0.0), level);
  let hz = clip_height_at_level(world_xz + vec2<f32>(0.0, cell), level);

  let dhx = (hx - h) / max(cell, 1e-4);
  let dhz = (hz - h) / max(cell, 1e-4);

  return normalize(vec3<f32>(-dhx, 1.0, -dhz));
}

fn clip_normal_at(xz: vec2<f32>) -> vec3<f32> {
  let lvl = clip_choose_level_raw(xz);
  return clip_normal_at_level_2tap(xz, lvl);
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

  // Choose initial level and enforce monotonic (only same/coarser) as we march
  var lvl_prev: u32 = clip_choose_level_raw(p.xz);
  var lvl: u32 = lvl_prev;

  var h = clip_height_at_level(p.xz, lvl);
  var s_prev = p.y - h;
  var t_prev = t;

  for (var i: u32 = 0u; i < HF_MAX_STEPS; i = i + 1u) {
    if (t > t_max) { break; }

    p = ro + rd * t;

    // Monotonic LOD: prevent switching to a finer level mid-ray
    let cand = clip_choose_level_hyst(p.xz, lvl_prev);
    lvl = max(lvl_prev, cand);
    lvl_prev = lvl;

    h = clip_height_at_level(p.xz, lvl);
    let s = p.y - h;

    if (s <= 0.0 && s_prev > 0.0) {
      var a = t_prev;
      var b = t;

      // Bisection against the SAME surface (same lvl), not a moving LOD surface
      for (var k: u32 = 0u; k < HF_BISECT; k = k + 1u) {
        let m = 0.5 * (a + b);
        let pm = ro + rd * m;

        let hm = clip_height_at_level(pm.xz, lvl);
        let sm = pm.y - hm;

        if (sm > 0.0) { a = m; } else { b = m; }
      }

      let th = 0.5 * (a + b);
      let ph = ro + rd * th;

      // Normal: you can keep using the same lvl for consistency
      let n = clip_normal_at_level_2tap(ph.xz, lvl);

      return ClipHit(true, th, n, MAT_GRASS);
    }

    s_prev = s;
    t_prev = t;

    // Grazing-ray stabilization: also limit xz travel per step
    let vy = max(-rd.y, 0.12);
    let vh = max(length(rd.xz), 1e-4);

    // Step suggested by vertical clearance (your original idea)
    let dt_y = abs(s) / vy;

    // Limit step so we advance only a small number of texels in xz
    let cell = clip.level[lvl].z;
    let dt_xz = (2.0 * cell) / vh; // ~2 texels per step in xz

    // Take the smaller of the two (prevents skipping when looking sideways)
    var dt = min(dt_y, dt_xz);

    // Slightly smaller minimum helps near-ground detail (optional but usually better)
    dt = clamp(dt, 0.25, HF_DT_MAX);

    t = t + dt;

  }

  return ClipHit(false, BIG_F32, vec3<f32>(0.0), MAT_AIR);
}


fn material_variation_clip(world_p: vec3<f32>, cell_size_m: f32, strength: f32) -> f32 {
  let cell = floor(world_p / cell_size_m);
  return (hash31(cell) - 0.5) * strength;
}

// a slightly more “can't miss it” grass tint
fn apply_material_variation_clip(base: vec3<f32>, mat: u32, hp: vec3<f32>) -> vec3<f32> {
  var c = base;

  // Stable patch noise (no shimmer)
  if (mat == MAT_GRASS) {
    let v0 = material_variation_clip(hp, 3.0, 1.0);    // big patches
    let v1 = material_variation_clip(hp, 0.75, 0.35);  // small breakup
    let v  = v0 + v1;

    // additive tint (visible)
    c += vec3<f32>(0.10 * v, 0.18 * v, 0.06 * v);

    // subtle brightness
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

  // base + variation
  var base = color_for_material(ch.mat);
  base = apply_material_variation_clip(base, ch.mat, hp);

  // lighting model consistent with voxels
  let voxel_size = cam.voxel_params.x;
  let hp_shadow  = hp + ch.n * (0.75 * voxel_size);

  let vis  = sun_transmittance(hp_shadow, SUN_DIR);
  let diff = max(dot(ch.n, SUN_DIR), 0.0);

  // AO-lite for terrain: reuse voxel AO idea with cheap taps against heightfield itself
  // We approximate occlusion using height samples around the hitpoint.
  let lvl = clip_choose_level_raw(hp.xz);
  let cell = clip.level[lvl].z;

  let h0  = clip_height_at_level(hp.xz, lvl);
  let hx1 = clip_height_at_level(hp.xz + vec2<f32>( cell, 0.0), lvl);
  let hx0 = clip_height_at_level(hp.xz + vec2<f32>(-cell, 0.0), lvl);
  let hz1 = clip_height_at_level(hp.xz + vec2<f32>(0.0,  cell), lvl);
  let hz0 = clip_height_at_level(hp.xz + vec2<f32>(0.0, -cell), lvl);

  // If neighbors are higher than current point, it feels “more occluded”.
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

  // Spec + fresnel like voxel path (simple)
  let vdir = normalize(-rd);
  let hdir = normalize(vdir + SUN_DIR);
  let ndv  = max(dot(ch.n, vdir), 0.0);
  let ndh  = max(dot(ch.n, hdir), 0.0);

  // terrain roughness
  var rough = 0.85;
  if (ch.mat == MAT_STONE) { rough = 0.50; }
  if (ch.mat == MAT_DIRT)  { rough = 0.90; }
  if (ch.mat == MAT_GRASS) { rough = 0.88; }

  let shininess = mix(8.0, 96.0, 1.0 - rough);
  let spec = pow(ndh, shininess);

  // fresnel (scalar)
  var f0 = 0.03;
  if (ch.mat == MAT_STONE) { f0 = 0.04; }
  let fres = f0 + (1.0 - f0) * pow(1.0 - clamp(ndv, 0.0, 1.0), 5.0);

  let spec_col = SUN_COLOR * SUN_INTENSITY * spec * fres * vis;

  return base * (ambient + direct) + 0.18 * spec_col;
}
