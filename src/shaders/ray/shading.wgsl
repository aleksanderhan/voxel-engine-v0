//// --------------------------------------------------------------------------
//// Shading
//// --------------------------------------------------------------------------

// Performance gates for primary pass (world-space distance) live in common.wgsl.

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

fn material_variation(world_p: vec3<f32>, cell_size_m: f32) -> f32 {
  let cell = floor(world_p / cell_size_m);
  return (hash31(cell) - 0.5) * 2.0;
}

fn apply_material_variation(base: vec3<f32>, mat: u32, hp: vec3<f32>) -> vec3<f32> {
  var c = base;
  let v = ALBEDO_VAR_GAIN * material_variation(hp, 0.05);

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

fn grass_wind_dir_xz(p_ws: vec3<f32>, time_s: f32) -> vec2<f32> {
  let w = wind_field(p_ws, time_s).xz;
  var dir = w + vec2<f32>(0.6, 0.2);
  if (dot(dir, dir) < 1e-4) {
    dir = vec2<f32>(0.7, 0.25);
  }
  return normalize(dir);
}

fn grass_blade_slivers(p_ws: vec3<f32>, time_s: f32, freq: f32, detail: f32) -> f32 {
  let dir = grass_wind_dir_xz(p_ws, time_s);
  let perp = vec2<f32>(-dir.y, dir.x);
  let p = vec2<f32>(dot(p_ws.xz, dir), dot(p_ws.xz, perp)) * freq;

  let cell = floor(p);
  let f = fract(p);

  let h0 = hash31(vec3<f32>(cell.x, cell.y, 11.3));
  let h1 = hash31(vec3<f32>(cell.x + 3.7, cell.y + 7.1, 5.9));

  let angle = (h0 - 0.5) * 0.35;
  let ca = cos(angle);
  let sa = sin(angle);
  let pr = vec2<f32>(f.x * ca - f.y * sa, f.x * sa + f.y * ca);

  let center = 0.5 + (h1 - 0.5) * 0.6;
  let u = clamp(pr.x, 0.0, 1.0);

  let taper = mix(0.065, 0.012, pow(u, 1.8));
  let width = taper * mix(0.75, 1.25, h0);
  let aa = mix(1.6, 0.6, detail) / freq;

  let dist = abs(pr.y - center);
  let sliver = smoothstep(width + aa, width - aa, dist);
  let length_mask = smoothstep(0.0, 0.15, u) * smoothstep(1.0, 0.75, u);
  return sliver * length_mask;
}

fn grass_blade_field(p_ws: vec3<f32>, time_s: f32, t: f32) -> f32 {
  let detail = 1.0 - smoothstep(18.0, 55.0, t);

  let low = grass_blade_slivers(p_ws, time_s, 9.0, detail * 0.6);
  let high = grass_blade_slivers(p_ws + vec3<f32>(1.7, 0.0, 2.3), time_s, 18.0, detail);

  let blend = mix(0.15, 1.0, detail);
  return mix(low, 0.5 * (low + high), blend);
}

fn grass_top_albedo(base: vec3<f32>, hp: vec3<f32>, t: f32) -> vec3<f32> {
  let vs = cam.voxel_params.x;
  let tip = clamp(fract(hp.y / max(vs, 1e-6)), 0.0, 1.0);

  let time_s = cam.voxel_params.y;
  let blade = grass_blade_field(hp, time_s, t);

  let patches = fbm(hp.xz * 0.08);
  let clumps = fbm(hp.xz * 0.35 + vec2<f32>(12.3, 7.1));
  let density = smoothstep(0.25, 0.75, patches) * smoothstep(0.20, 0.80, clumps);

  let coverage = blade * density;

  let cool_base = base * vec3<f32>(0.78, 0.90, 0.82);
  let warm_tip = base * vec3<f32>(1.08, 1.16, 0.92) + vec3<f32>(0.02, 0.04, 0.01);
  let grad = smoothstep(0.0, 1.0, tip);
  let blade_col = mix(cool_base, warm_tip, grad);

  var out = mix(base * 0.85, blade_col, coverage);
  let base_shadow = mix(0.68, 1.0, grad + 0.25 * coverage);
  out *= base_shadow;
  return max(out, vec3<f32>(0.02));
}

fn base_albedo(mat: u32, hp: vec3<f32>, t: f32) -> vec3<f32> {
  var base = color_for_material(mat);
  if (t <= FAR_SHADING_DIST) {
    base = apply_material_variation(base, mat, hp);
  }
  return base;
}

fn macro_occ_at_ws(
  p_ws: vec3<f32>,
  root_bmin: vec3<f32>,
  root_size: f32,
  macro_base: u32
) -> f32 {
  if (macro_base == INVALID_U32) { return 0.0; }

  let lp = p_ws - root_bmin;
  if (lp.x < 0.0 || lp.y < 0.0 || lp.z < 0.0 ||
      lp.x >= root_size || lp.y >= root_size || lp.z >= root_size) {
    return 0.0;
  }

  let cell = macro_cell_size(root_size);
  let mx = clamp(u32(floor(lp.x / cell)), 0u, MACRO_DIM - 1u);
  let my = clamp(u32(floor(lp.y / cell)), 0u, MACRO_DIM - 1u);
  let mz = clamp(u32(floor(lp.z / cell)), 0u, MACRO_DIM - 1u);

  let bit = macro_bit_index(mx, my, mz);
  // 1.0 means "occupied", 0.0 means "empty"
  return select(0.0, 1.0, macro_test(macro_base, bit));
}


fn voxel_ao_local(
  hp: vec3<f32>,
  n: vec3<f32>,
  root_bmin: vec3<f32>,
  root_size: f32,
  node_base: u32,
  macro_base: u32
) -> f32 {
  // Macro-only AO: extremely fast, stable, and usually good enough.
  // You can tune r independently of voxel size.
  let r = 0.75 * cam.voxel_params.x;

  let up_ref = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
  let t = normalize(cross(up_ref, n));
  let b = normalize(cross(n, t));

  // 4 taps (same pattern as before, but macro occupancy)
  var occ = 0.0;
  occ += macro_occ_at_ws(hp + t * r, root_bmin, root_size, macro_base);
  occ += macro_occ_at_ws(hp - t * r, root_bmin, root_size, macro_base);
  occ += macro_occ_at_ws(hp + b * r, root_bmin, root_size, macro_base);
  occ += macro_occ_at_ws(hp - b * r, root_bmin, root_size, macro_base);

  let occ_n = occ * 0.25;

  // Map to AO term (match your old curve-ish behavior)
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
    return 18.0 * vec3<f32>(1.0, 0.95, 0.75);
  }
  return vec3<f32>(0.0);
}

struct ShadeOut {
  base_hdr   : vec3<f32>,
  local_hdr  : vec3<f32>,
  local_w    : f32,
};

fn shade_hit_split(
  ro: vec3<f32>,
  rd: vec3<f32>,
  hg: HitGeom,
  sky_up: vec3<f32>,
  seed: u32,
  sun_shadow: f32
) -> ShadeOut {
  let base_hdr = shade_hit(ro, rd, hg, sky_up, seed, sun_shadow);

  // Local voxel lighting from MAT_LIGHT blocks.
  // Keep it separate so it can be temporally accumulated without fog.
  var local_hdr = vec3<f32>(0.0);
  var local_w   = 0.0;

  if (hg.hit != 0u && hg.mat != MAT_LIGHT && hg.t <= LOCAL_LIGHT_MAX_DIST) {
    let hp = ro + hg.t * rd;
    let albedo = base_albedo(hg.mat, hp, hg.t);
    local_hdr = gather_voxel_lights(
      hp,
      hg.n,
      hg.root_bmin,
      hg.root_size,
      hg.node_base,
      hg.macro_base
    );
    local_hdr *= albedo;
    local_w = 1.0;
  }

  return ShadeOut(base_hdr, local_hdr, local_w);
}

fn shade_hit(
  ro: vec3<f32>,
  rd: vec3<f32>,
  hg: HitGeom,
  sky_up: vec3<f32>,
  seed: u32,
  sun_shadow: f32
) -> vec3<f32> {
  let hp = ro + hg.t * rd;

  var base = base_albedo(hg.mat, hp, hg.t);

  // Gate extra grass work harder in primary
  if (hg.mat == MAT_GRASS) {
    if (ENABLE_GRASS && grass_allowed_primary(hg.t, hg.n, rd, seed)) {
      base = grass_top_albedo(base, hp, hg.t);

      // Wrap diffuse so blades don’t look “flat-lit”
      let ndl = dot(hg.n, SUN_DIR);
      let wrap = 0.35; // 0..1 (higher = softer, leafier)
      let diff_wrap = clamp((ndl + wrap) / (1.0 + wrap), 0.0, 1.0);

      // Strong backscatter / “subsurface-ish” pop when sun is behind grass
      let back = pow(clamp(dot(-SUN_DIR, hg.n), 0.0, 1.0), 1.6);
      let view_graze = pow(1.0 - max(dot(normalize(-rd), hg.n), 0.0), 2.0);

      // Add warm-green transmission
      base += (0.22 * back + 0.10 * view_graze * back) * vec3<f32>(0.18, 0.34, 0.10);

      // Bake wrapped diffuse into albedo a bit (keeps it “full” under sun)
      base *= (0.90 + 0.10 * diff_wrap);
    }
  }

  let vs        = cam.voxel_params.x;
  let hp_shadow = hp + hg.n * (0.75 * vs);

  var Tc: f32 = 1.0;
  if (PRIMARY_CLOUD_SHADOWS) {
    Tc = cloud_sun_transmittance_fast(hp_shadow, SUN_DIR);
  }
  let sun_vis = Tc * clamp(sun_shadow, 0.0, 1.0);
  let diff = max(dot(hg.n, SUN_DIR), 0.0);

  // AO for voxels: only when the hit is a real voxel hit (not sky / miss)
  var ao = 1.0;
  if (hg.hit != 0u && hg.t <= VOXEL_AO_MAX_DIST) {
    ao = voxel_ao_local(hp, hg.n, hg.root_bmin, hg.root_size, hg.node_base, hg.macro_base);
  }

    let amb_col      = hemi_ambient(hg.n, sky_up);

  // Much stronger baseline; leaves slightly higher.
  let amb_strength = select(0.10, 0.13, hg.mat == MAT_LEAF);

  // Do NOT square AO (it crushes the shadow side).
  // If you want a curve, use a gentle power instead.
  let ao_term = pow(clamp(ao, 0.0, 1.0), 1.0);

  var ambient = amb_col * amb_strength * ao_term;


  if (hg.mat == MAT_STONE) {
    ambient *= vec3<f32>(0.92, 0.95, 1.05);
  }

  // Leaf dapple (cheap) - keep as-is
  var dapple = 1.0;
  if (hg.mat == MAT_LEAF && hg.t <= FAR_SHADING_DIST) {
    let time_s = cam.voxel_params.y;
    let d0 = sin(dot(hp.xz, vec2<f32>(3.0, 2.2)) + time_s * 3.5);
    let d1 = sin(dot(hp.xz, vec2<f32>(6.5, 4.1)) - time_s * 6.0);
    dapple = 0.90 + 0.10 * (0.6 * d0 + 0.4 * d1);
  }

  var spec_col = vec3<f32>(0.0);
  if (hg.t <= FAR_SHADING_DIST) {
    let v = normalize(-rd);
    let h = normalize(v + SUN_DIR);

    let ndv = max(dot(hg.n, v), 0.0);
    let ndh = max(dot(hg.n, h), 0.0);

    let rough     = material_roughness(hg.mat);
    let shininess = mix(8.0, 96.0, 1.0 - rough);
    let spec      = pow(ndh, shininess);

    let f0   = material_f0(hg.mat);
    let fres = fresnel_schlick(ndv, f0);
    spec_col = SUN_COLOR * SUN_INTENSITY * spec * fres * sun_vis;
  }

  var grass_ts: f32 = 1.0;
  if (hg.mat == MAT_GRASS && ENABLE_GRASS && grass_allowed_primary(hg.t, hg.n, rd, seed)) {
    let time_s = cam.voxel_params.y;
    let strength = cam.voxel_params.z;
    let cell = grass_cell_from_world(hp, rd, hg.root_bmin, vs, i32(cam.chunk_size));
    let lod = grass_lod_from_t(hg.t);
    grass_ts = grass_self_shadow(hp, cell.bmin_m, cell.id_vox, time_s, strength, lod);
  }

  let direct = SUN_COLOR * SUN_INTENSITY * (0.35 * diff + 0.65 * diff * diff) * sun_vis * dapple * grass_ts;

  let emissive = material_emission(hg.mat);
  return base * (ambient + direct) + 0.20 * spec_col + emissive;
}

fn shade_clip_hit(
  ro: vec3<f32>,
  rd: vec3<f32>,
  ch: ClipHit,
  sky_up: vec3<f32>,
  seed: u32,
  allow_grass: bool
) -> vec3<f32> {
  let hp = ro + ch.t * rd;

  var base = color_for_material(ch.mat);
  if (ch.t <= FAR_SHADING_DIST) {
    base = apply_material_variation_clip(base, ch.mat, hp);
  }
  if (allow_grass && ch.mat == MAT_GRASS && ENABLE_GRASS && grass_allowed_primary(ch.t, ch.n, rd, seed)) {
    base = grass_top_albedo(base, hp, ch.t);
  }

  let voxel_size = cam.voxel_params.x;
  let hp_shadow  = hp + ch.n * (0.75 * voxel_size);

  var vis = 1.0;
  if (PRIMARY_CLOUD_SHADOWS) {
    vis = cloud_sun_transmittance_fast(hp_shadow, SUN_DIR);
  }
  let diff = max(dot(ch.n, SUN_DIR), 0.0);

  // AO-lite for terrain: gate hard for grass in primary
  var ao = 1.0;
  if (allow_grass && ch.mat == MAT_GRASS && ENABLE_GRASS && grass_allowed_primary(ch.t, ch.n, rd, seed) && ch.t <= FAR_SHADING_DIST) {
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
  let ambient      = amb_col * amb_strength * ao;

  let direct = SUN_COLOR * SUN_INTENSITY * (diff * diff) * vis;

  var spec_col = vec3<f32>(0.0);
  if (ch.t <= FAR_SHADING_DIST) {
    let vdir = normalize(-rd);
    let hdir = normalize(vdir + SUN_DIR);
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

    spec_col = SUN_COLOR * SUN_INTENSITY * spec * fres * vis;
  }

  return base * (ambient + direct) + 0.18 * spec_col;
}
