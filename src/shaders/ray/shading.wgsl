// src/shaders/ray/shading.wgsl
// ----------------------------
// Shading (split outputs for temporal accumulation of local voxel lights)
//
// What changed:
// - Added ShadeOut + shade_hit_split(): returns (base_hdr, local_hdr)
// - Added shade_hit(): wrapper that preserves old behavior (base+local) so
//   existing callsites don’t break while you wire up the new buffers.
// - Everything else kept compatible with your current symbols.
//
// Notes:
// - This file does NOT implement the temporal accumulation pass. It only
//   splits the shading so you can accumulate local_hdr elsewhere.
// - Assumes these exist in other includes:
//   - HitGeom, ClipHit, cam, clip, chunk_size, etc.
//   - hash31(), hash12(), safe_normalize(), is_bad_vec3(), etc.
//   - grass_* helpers, sun_transmittance_* and cloud_* helpers
//   - macro_* helpers (macro_cell_size, macro_bit_index, macro_test, etc.)

// --- Ambient floor so caves are never pitch black (HDR-linear space) ---
const AMBIENT_FLOOR_STRENGTH : f32 = 0.020;                 // try 0.01..0.05
const AMBIENT_FLOOR_COLOR    : vec3<f32> = vec3<f32>(0.9, 0.95, 1.0); // slight cool tint

fn color_for_material(m: u32) -> vec3<f32> {
  if (m == MAT_AIR)   { return vec3<f32>(0.0); }
  if (m == MAT_GRASS) { return vec3<f32>(0.18, 0.75, 0.18); }
  if (m == MAT_DIRT)  { return vec3<f32>(0.45, 0.30, 0.15); }
  if (m == MAT_STONE) { return vec3<f32>(0.50, 0.50, 0.55); }
  if (m == MAT_WOOD)  { return vec3<f32>(0.38, 0.26, 0.14); }
  if (m == MAT_LEAF)  { return vec3<f32>(0.10, 0.55, 0.12); }
  if (m == MAT_LIGHT) { return vec3<f32>(1.0, 0.95, 0.75); }
  return vec3<f32>(1.0, 0.0, 1.0);
}

fn hemi_ambient(n: vec3<f32>, sky_up: vec3<f32>) -> vec3<f32> {
  let upw = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
  let grd = FOG_COLOR_GROUND;
  return mix(grd, sky_up, upw);
}

fn material_variation(world_p: vec3<f32>, cell_size_m: f32, dist_m: f32) -> f32 {
  let scale = max(cell_size_m * 2.5, 1e-3);
  let uv = world_p.xz / scale;
  let raw = (value_noise(uv) - 0.5) * 2.0;
  let fade = exp(-dist_m / 75.0);
  return raw * fade;
}

fn apply_material_variation(
  base: vec3<f32>,
  mat: u32,
  hp: vec3<f32>,
  dist_m: f32,
  strength: f32
) -> vec3<f32> {
  var c = base;
  let cell_size = max(0.08, cam.voxel_params.x * 0.75);
  let v = ALBEDO_VAR_GAIN * material_variation(hp, cell_size, dist_m) * strength;

  if (mat == MAT_GRASS) {
    c += vec3<f32>(0.02 * v, 0.05 * v, 0.01 * v);
    c *= (1.0 + 0.06 * v);
  } else if (mat == MAT_DIRT) {
    c += vec3<f32>(0.04 * v, 0.02 * v, 0.01 * v);
    c *= (1.0 + 0.08 * v);
  } else if (mat == MAT_STONE) {
    c *= (1.0 + 0.10 * v);
  } else if (mat == MAT_WOOD) {
    c += vec3<f32>(0.05 * v, 0.02 * v, 0.00 * v);
    c *= (1.0 + 0.07 * v);
  } else if (mat == MAT_LEAF) {
    c += vec3<f32>(0.00 * v, 0.03 * v, 0.00 * v);
    c *= (1.0 + 0.04 * v);
  }

  return clamp(c, vec3<f32>(0.0), vec3<f32>(1.5));
}

// --- AO that ignores MAT_LIGHT (and treats leaves/grass as partial occluders) ---

fn occ_at_material_aware(
  p: vec3<f32>,
  root_bmin: vec3<f32>,
  root_size: f32,
  node_base: u32,
  macro_base: u32
) -> f32 {
  // Query actual leaf material (use world lookup if we stepped outside this chunk).
  let eps = 1e-4 * root_size;
  let bmin = root_bmin - vec3<f32>(eps);
  let bmax = root_bmin + vec3<f32>(root_size + eps);
  let inside = all(p >= bmin) && all(p < bmax);
  var q: LeafQuery;
  if (inside) {
    q = query_leaf_at(p, root_bmin, root_size, node_base, INVALID_U32);
  } else {
    q = query_leaf_world_no_macro(p);
  }

  // Ignore air and lights completely (so placing a lamp never darkens AO)
  if (q.mat == MAT_AIR || q.mat == MAT_LIGHT) {
    return 0.0;
  }

  // Treat foliage as partial occlusion (optional but looks nicer)
  if (q.mat == MAT_LEAF)  { return 0.35; }
  if (q.mat == MAT_GRASS) { return 0.25; }

  // Everything else is solid occluder
  return 1.0;
}

fn voxel_ao_material4(
  hp: vec3<f32>,
  n: vec3<f32>,
  root_bmin: vec3<f32>,
  root_size: f32,
  node_base: u32,
  macro_base: u32
) -> f32 {
  let vs = cam.voxel_params.x;

  // AO radius: keep your old intent (low-freq), but stable
  let cell = macro_cell_size(root_size);
  let r = max(0.75 * cell, 1.25 * vs);

  // Stable TBN
  let up_ref = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
  let t = normalize(cross(up_ref, n));
  let b = normalize(cross(n, t));

  // Nudge off surface
  let p0 = hp + n * (0.75 * vs);

  var occ: f32 = 0.0;
  occ += occ_at_material_aware(p0 + t * r, root_bmin, root_size, node_base, macro_base);
  occ += occ_at_material_aware(p0 - t * r, root_bmin, root_size, node_base, macro_base);
  occ += occ_at_material_aware(p0 + b * r, root_bmin, root_size, node_base, macro_base);
  occ += occ_at_material_aware(p0 - b * r, root_bmin, root_size, node_base, macro_base);

  let occ_n = occ * 0.25;

  // Same shaping as before
  return clamp(1.0 - 0.70 * occ_n, 0.35, 1.0);
}


fn fresnel_schlick(ndv: f32, f0: f32) -> f32 {
  return f0 + (1.0 - f0) * pow(1.0 - clamp(ndv, 0.0, 1.0), 5.0);
}

fn material_roughness(mat: u32) -> f32 {
  if (mat == MAT_STONE) { return 0.45; }
  if (mat == MAT_WOOD)  { return 0.70; }
  if (mat == MAT_LEAF)  { return 0.80; }
  if (mat == MAT_GRASS) { return 0.85; }
  if (mat == MAT_DIRT)  { return 0.90; }
  if (mat == MAT_LIGHT) { return 0.25; }
  return 0.90;
}

fn material_f0(mat: u32) -> f32 {
  if (mat == MAT_STONE) { return 0.04; }
  if (mat == MAT_WOOD)  { return 0.03; }
  if (mat == MAT_LEAF)  { return 0.05; }
  if (mat == MAT_GRASS) { return 0.04; }
  if (mat == MAT_DIRT)  { return 0.02; }
  if (mat == MAT_LIGHT) { return 0.08; }
  return 0.02;
}

fn material_emission(mat: u32) -> vec3<f32> {
  if (mat == MAT_LIGHT) {
    // HDR emission. Tune this to taste.
    return 18.0 * vec3<f32>(1.0, 0.95, 0.75);
  }
  return vec3<f32>(0.0);
}

// -----------------------------------------------------------------------------
// Split shading output
// -----------------------------------------------------------------------------
struct ShadeOut {
  base_hdr   : vec3<f32>,
  local_hdr  : vec3<f32>,
  local_w    : f32,       // 1 when local_hdr is a valid sample this frame, else 0
};


// NEW: split shading (base + local voxel lights)
fn shade_hit_split(
  ro: vec3<f32>,
  rd: vec3<f32>,
  hg: HitGeom,
  sky_up: vec3<f32>,
  seed: u32
) -> ShadeOut {
  let hp = ro + hg.t * rd;

  // Emissive lamp voxel itself: keep fully in base_hdr (do NOT TAA as "local")
  if (hg.mat == MAT_LIGHT) {
    let v = normalize(-rd);
    let ndv = max(dot(hg.n, v), 0.0);

    let core = 22.0 * vec3<f32>(1.0, 0.95, 0.75);
    let rim  = 10.0 * pow(1.0 - ndv, 3.0) * vec3<f32>(1.0, 0.85, 0.55);

    return ShadeOut(core + rim, vec3<f32>(0.0), 0.0);
  }

  let vs        = cam.voxel_params.x;
  let hp_shadow = hp + hg.n * (0.75 * vs);

  let Tc       = cloud_sun_transmittance(hp_shadow, SUN_DIR);
  let vis_geom = sun_transmittance_geom_only(hp_shadow, SUN_DIR);

  let sv_raw = sky_visibility(hp_shadow);
  let var_strength = mix(0.35, 1.0, smoothstep(0.15, 0.45, sv_raw));

  var base = color_for_material(hg.mat);
  base = apply_material_variation(base, hg.mat, hp, hg.t, var_strength);

  // Gate extra grass work harder in primary
  if (hg.mat == MAT_GRASS) {
    if (grass_allowed_primary(hg.t, hg.n, seed)) {
      let tip = clamp(fract(hp.y / max(vs, 1e-6)), 0.0, 1.0);

      base = mix(base, base + vec3<f32>(0.10, 0.10, 0.02), 0.35 * tip);

      let back = pow(clamp(dot(-SUN_DIR, hg.n), 0.0, 1.0), 2.0);
      base += 0.22 * back * vec3<f32>(0.18, 0.35, 0.10);
    }
  }

  let diff = max(dot(hg.n, SUN_DIR), 0.0);

  // AO (macro) — smooth distance fade to avoid “sphere” contour
  var ao = 1.0;
  if (hg.hit != 0u) {
    let ao_fade = exp(-hg.t / 55.0);
    if (ao_fade > 0.01) {
      let ao_raw = voxel_ao_material4(hp, hg.n, hg.root_bmin, hg.root_size, hg.node_base, hg.macro_base);
      ao = mix(1.0, ao_raw, ao_fade);
    }
  }


  // Ambient
  let amb_col      = hemi_ambient(hg.n, sky_up);
  let amb_strength = select(0.10, 0.14, hg.mat == MAT_LEAF);

  let sv     = max(sv_raw, 0.08);

  var ambient = amb_col * amb_strength * ao * sv;

  // Constant floor (fades outdoors)
  let outdoor_fade = smoothstep(0.35, 0.90, sv_raw);
  let floor_k = AMBIENT_FLOOR_STRENGTH * (1.0 - 0.65 * outdoor_fade);
  ambient += floor_k * ao * AMBIENT_FLOOR_COLOR;

  if (hg.mat == MAT_STONE) {
    ambient *= vec3<f32>(0.92, 0.95, 1.05);
  }

  // Leaf dapple
  var dapple = 1.0;
  if (hg.mat == MAT_LEAF) {
    let time_s = cam.voxel_params.y;
    let d0 = sin(dot(hp.xz, vec2<f32>(3.0, 2.2)) + time_s * 3.5);
    let d1 = sin(dot(hp.xz, vec2<f32>(6.5, 4.1)) - time_s * 6.0);
    dapple = 0.90 + 0.10 * (0.6 * d0 + 0.4 * d1);
  }

  // Specular
  let v = safe_normalize(-rd);
  let h = safe_normalize(v + SUN_DIR);

  let ndv = max(dot(hg.n, v), 0.0);
  let ndh = max(dot(hg.n, h), 0.0);

  let rough     = material_roughness(hg.mat);
  let shininess = mix(8.0, 96.0, 1.0 - rough);
  let spec      = pow(ndh, shininess);

  let f0   = material_f0(hg.mat);
  let fres = fresnel_schlick(ndv, f0);

  let direct   = SUN_COLOR * SUN_INTENSITY * (diff * diff) * vis_geom * Tc * dapple;
  let spec_col = SUN_COLOR * SUN_INTENSITY * spec * fres * vis_geom * Tc;

  // Local voxel lights — smooth distance fade to avoid “sphere” contour
  var local_light = vec3<f32>(0.0);
  var local_w: f32 = 0.0;

  let ll_fade = exp(-hg.t / 45.0);
  if (ll_fade > 0.01) {
    let cave = (sv_raw < 0.20);

    // Keep perf: sample less often as distance increases
    // (near: full rate; mid: half; far: quarter; outside fade: none)
    let rate_mask = select(3u, 0u, cave); // original cave/outdoor behavior
    let extra_mask = select(0u, 1u, hg.t > 20.0); // optional: half-rate after 20m
    let m = rate_mask | extra_mask;

    if ((seed & m) == 0u) {
      local_light = gather_voxel_lights(
        hp, hg.n, hg.root_bmin, hg.root_size, hg.node_base, hg.macro_base, seed
      ) * ll_fade;

      // Tell the TAA pass this sample is “weaker” (optional but helps blending)
      local_w = ll_fade;
    }
  }


  let base_hdr  = base * (ambient + direct) + 0.20 * spec_col;
  let local_hdr = base * local_light;

  return ShadeOut(base_hdr, local_hdr, local_w);
}

// Your clip shading stays single-output (you can split later if desired)
fn shade_clip_hit(ro: vec3<f32>, rd: vec3<f32>, ch: ClipHit, sky_up: vec3<f32>, seed: u32) -> vec3<f32> {
  let hp = ro + ch.t * rd;

  if (ch.mat == MAT_LIGHT) {
    let facing = 0.65 + 0.35 * max(dot(ch.n, normalize(-rd)), 0.0);
    return material_emission(ch.mat) * facing;
  }

  var base = color_for_material(ch.mat);
  base = apply_material_variation_clip(base, ch.mat, hp);

  let voxel_size = cam.voxel_params.x;
  let hp_shadow  = hp + ch.n * (0.75 * voxel_size);

  let vis  = sun_transmittance(hp_shadow, SUN_DIR);
  let diff = max(dot(ch.n, SUN_DIR), 0.0);

  // AO-lite for terrain: gate hard for grass in primary
  var ao = 1.0;
  if (ch.mat == MAT_GRASS && grass_allowed_primary(ch.t, ch.n, seed)) {
    let lvl  = clip_best_level(hp.xz, 2);
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

    ao = clamp(1.0 - 0.65 * occ / max(cell, 1e-3), 0.45, 1.0);
  }

  let amb_col      = hemi_ambient(ch.n, sky_up);
  let amb_strength = 0.10;
  var ambient      = amb_col * amb_strength * ao;

  // tiny constant floor for terrain too (optional)
  ambient += (0.015 * ao) * vec3<f32>(0.9, 0.95, 1.0);

  let direct = SUN_COLOR * SUN_INTENSITY * (diff * diff) * vis;

  let vdir = safe_normalize(-rd);
  let hdir = safe_normalize(vdir + SUN_DIR);
  let ndv  = max(dot(ch.n, vdir), 0.0);
  let ndh  = max(dot(ch.n, hdir), 0.0);

  var rough = 0.85;
  if (ch.mat == MAT_STONE) { rough = 0.50; }
  if (ch.mat == MAT_DIRT)  { rough = 0.90; }
  if (ch.mat == MAT_GRASS) { rough = 0.88; }

  let shininess = mix(8.0, 96.0, 1.0 - rough);
  let spec      = pow(ndh, shininess);

  var f0 = 0.03;
  if (ch.mat == MAT_STONE) { f0 = 0.04; }
  let fres = f0 + (1.0 - f0) * pow(1.0 - clamp(ndv, 0.0, 1.0), 5.0);

  let spec_col = SUN_COLOR * SUN_INTENSITY * spec * fres * vis;

  return base * (ambient + direct) + 0.18 * spec_col;
}

fn sky_visibility(p: vec3<f32>) -> f32 {
  // Up ray: if blocked, returns ~0. If open, returns ~1.
  let vs = cam.voxel_params.x;
  let pu = p + vec3<f32>(0.0, 1.0, 0.0) * (0.75 * vs);
  return sun_transmittance_geom_only(pu, vec3<f32>(0.0, 1.0, 0.0));
}
