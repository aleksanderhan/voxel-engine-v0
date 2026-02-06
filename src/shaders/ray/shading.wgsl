//// --------------------------------------------------------------------------
//// Shading
//// --------------------------------------------------------------------------

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

fn voxel_ao_local(
  hp: vec3<f32>,
  n: vec3<f32>,
  root_bmin: vec3<f32>,
  root_size: f32,
  node_base: u32,
  macro_base: u32
) -> f32 {
  let r = 0.75 * cam.voxel_params.x;

  let up_ref = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
  let t = normalize(cross(up_ref, n));
  let b = normalize(cross(n, t));

  var occ = 0.0;

  let q0 = query_leaf_at(hp + t * r, root_bmin, root_size, node_base, macro_base);
  occ += select(0.0, 1.0, q0.mat != MAT_AIR);

  let q1 = query_leaf_at(hp - t * r, root_bmin, root_size, node_base, macro_base);
  occ += select(0.0, 1.0, q1.mat != MAT_AIR);

  let q2 = query_leaf_at(hp + b * r, root_bmin, root_size, node_base, macro_base);
  occ += select(0.0, 1.0, q2.mat != MAT_AIR);

  let q3 = query_leaf_at(hp - b * r, root_bmin, root_size, node_base, macro_base);
  occ += select(0.0, 1.0, q3.mat != MAT_AIR);

  let h0 = normalize(n + 0.65 * t + 0.35 * b);
  let q4 = query_leaf_at(hp + h0 * r, root_bmin, root_size, node_base, macro_base);
  occ += select(0.0, 1.0, q4.mat != MAT_AIR);

  let h1 = normalize(n - 0.65 * t + 0.35 * b);
  let q5 = query_leaf_at(hp + h1 * r, root_bmin, root_size, node_base, macro_base);
  occ += select(0.0, 1.0, q5.mat != MAT_AIR);

  let occ_n = occ * (1.0 / 6.0);
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
  seed: u32
) -> ShadeOut {
  let base_hdr = shade_hit(ro, rd, hg, sky_up, seed);

  // Local voxel lighting from MAT_LIGHT blocks.
  // Keep it separate so it can be temporally accumulated without fog.
  var local_hdr = vec3<f32>(0.0);
  var local_w   = 0.0;

  if (hg.hit != 0u && hg.mat != MAT_LIGHT) {
    let hp = ro + hg.t * rd;
    local_hdr = gather_voxel_lights(
      hp,
      hg.n,
      hg.root_bmin,
      hg.root_size,
      hg.node_base,
      hg.macro_base,
      seed
    );
    local_w = 1.0;
  }

  return ShadeOut(base_hdr, local_hdr, local_w);
}

fn shade_hit(ro: vec3<f32>, rd: vec3<f32>, hg: HitGeom, sky_up: vec3<f32>, seed: u32) -> vec3<f32> {
  let hp = ro + hg.t * rd;

  var base = color_for_material(hg.mat);
  base = apply_material_variation(base, hg.mat, hp);

  // Gate extra grass work harder in primary
  if (hg.mat == MAT_GRASS) {
    if (grass_allowed_primary(hg.t, hg.n, seed)) {
      let vs  = cam.voxel_params.x;
      let tip = clamp(fract(hp.y / max(vs, 1e-6)), 0.0, 1.0);

      base = mix(base, base + vec3<f32>(0.10, 0.10, 0.02), 0.35 * tip);

      let back = pow(clamp(dot(-SUN_DIR, hg.n), 0.0, 1.0), 2.0);
      base += 0.22 * back * vec3<f32>(0.18, 0.35, 0.10);
    }
  }

  let vs        = cam.voxel_params.x;
  let hp_shadow = hp + hg.n * (0.75 * vs);

  let Tc       = cloud_sun_transmittance(hp_shadow, SUN_DIR);
  let vis_geom = sun_transmittance_geom_only(hp_shadow, SUN_DIR);

  let diff = max(dot(hg.n, SUN_DIR), 0.0);

  // AO for voxels: only when the hit is a real voxel hit (not sky / miss)
  let ao = select(1.0, voxel_ao_local(hp, hg.n, hg.root_bmin, hg.root_size, hg.node_base, hg.macro_base), hg.hit != 0u);

  let amb_col      = hemi_ambient(hg.n, sky_up);
  let amb_strength = select(0.10, 0.14, hg.mat == MAT_LEAF);
  var ambient      = amb_col * amb_strength * ao;

  if (hg.mat == MAT_STONE) {
    ambient *= vec3<f32>(0.92, 0.95, 1.05);
  }

  // Leaf dapple (cheap) - keep as-is
  var dapple = 1.0;
  if (hg.mat == MAT_LEAF) {
    let time_s = cam.voxel_params.y;
    let d0 = sin(dot(hp.xz, vec2<f32>(3.0, 2.2)) + time_s * 3.5);
    let d1 = sin(dot(hp.xz, vec2<f32>(6.5, 4.1)) - time_s * 6.0);
    dapple = 0.90 + 0.10 * (0.6 * d0 + 0.4 * d1);
  }

  let v = normalize(-rd);
  let h = normalize(v + SUN_DIR);

  let ndv = max(dot(hg.n, v), 0.0);
  let ndh = max(dot(hg.n, h), 0.0);

  let rough     = material_roughness(hg.mat);
  let shininess = mix(8.0, 96.0, 1.0 - rough);
  let spec      = pow(ndh, shininess);

  let f0   = material_f0(hg.mat);
  let fres = fresnel_schlick(ndv, f0);

  let direct   = SUN_COLOR * SUN_INTENSITY * (diff * diff) * vis_geom * Tc * dapple;
  let spec_col = SUN_COLOR * SUN_INTENSITY * spec * fres * vis_geom * Tc;

  let emissive = material_emission(hg.mat);
  return base * (ambient + direct) + 0.20 * spec_col + emissive;
}

fn shade_clip_hit(ro: vec3<f32>, rd: vec3<f32>, ch: ClipHit, sky_up: vec3<f32>, seed: u32) -> vec3<f32> {
  let hp = ro + ch.t * rd;

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
  let ambient      = amb_col * amb_strength * ao;

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
  let spec      = pow(ndh, shininess);

  var f0 = 0.03;
  if (ch.mat == MAT_STONE) { f0 = 0.04; }
  let fres = f0 + (1.0 - f0) * pow(1.0 - clamp(ndv, 0.0, 1.0), 5.0);

  let spec_col = SUN_COLOR * SUN_INTENSITY * spec * fres * vis;

  return base * (ambient + direct) + 0.18 * spec_col;
}
