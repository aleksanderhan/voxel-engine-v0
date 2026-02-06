




const VOXEL_AO_MAX_DIST       : f32 = 40.0;
const LOCAL_LIGHT_MAX_DIST    : f32 = 50.0;
const FAR_SHADING_DIST        : f32 = 80.0;
const PRIMARY_CLOUD_SHADOWS   : bool = false;

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

fn base_albedo(mat: u32, hp: vec3<f32>, t: f32) -> vec3<f32> {
  var base = color_for_material(mat);
  if (t <= FAR_SHADING_DIST) {
    base = apply_material_variation(base, mat, hp);
  }
  return base;
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

  let occ_n = occ * (1.0 / 4.0);
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

fn shade_hit(ro: vec3<f32>, rd: vec3<f32>, hg: HitGeom, sky_up: vec3<f32>, seed: u32) -> vec3<f32> {
  let hp = ro + hg.t * rd;

  var base = base_albedo(hg.mat, hp, hg.t);

  
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

  var Tc: f32 = 1.0;
  if (PRIMARY_CLOUD_SHADOWS) {
    Tc = cloud_sun_transmittance_fast(hp_shadow, SUN_DIR);
  }
  let vis_geom: f32 = 1.0;

  let diff = max(dot(hg.n, SUN_DIR), 0.0);

  
  var ao = 1.0;
  if (hg.hit != 0u && hg.t <= VOXEL_AO_MAX_DIST) {
    ao = voxel_ao_local(hp, hg.n, hg.root_bmin, hg.root_size, hg.node_base, hg.macro_base);
  }

  let amb_col      = hemi_ambient(hg.n, sky_up);
  let amb_strength = select(0.03, 0.05, hg.mat == MAT_LEAF);
  let ao_term      = ao * ao * ao;
  var ambient      = amb_col * amb_strength * ao_term;

  if (hg.mat == MAT_STONE) {
    ambient *= vec3<f32>(0.92, 0.95, 1.05);
  }

  
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
    spec_col = SUN_COLOR * SUN_INTENSITY * spec * fres * vis_geom * Tc;
  }

  let direct   = SUN_COLOR * SUN_INTENSITY * (diff * diff) * vis_geom * Tc * dapple;
  let emissive = material_emission(hg.mat);
  return base * (ambient + direct) + 0.20 * spec_col + emissive;
}

fn shade_clip_hit(ro: vec3<f32>, rd: vec3<f32>, ch: ClipHit, sky_up: vec3<f32>, seed: u32) -> vec3<f32> {
  let hp = ro + ch.t * rd;

  var base = color_for_material(ch.mat);
  if (ch.t <= FAR_SHADING_DIST) {
    base = apply_material_variation_clip(base, ch.mat, hp);
  }

  let voxel_size = cam.voxel_params.x;
  let hp_shadow  = hp + ch.n * (0.75 * voxel_size);

  var vis = 1.0;
  if (PRIMARY_CLOUD_SHADOWS) {
    vis = cloud_sun_transmittance_fast(hp_shadow, SUN_DIR);
  }
  let diff = max(dot(ch.n, SUN_DIR), 0.0);

  
  var ao = 1.0;
  if (ch.mat == MAT_GRASS && grass_allowed_primary(ch.t, ch.n, seed) && ch.t <= FAR_SHADING_DIST) {
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
