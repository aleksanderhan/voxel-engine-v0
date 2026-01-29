// src/shaders/ray/shading.wgsl
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
  return vec3<f32>(1.0, 0.0, 1.0);
}

fn hemi_ambient(n: vec3<f32>, sky_up: vec3<f32>) -> vec3<f32> {
  let upw = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
  let grd = FOG_COLOR_GROUND;
  return mix(grd, sky_up, upw);
}

fn hash31(p: vec3<f32>) -> f32 {
  let h = dot(p, vec3<f32>(127.1, 311.7, 74.7));
  return fract(sin(h) * 43758.5453);
}

fn material_variation(world_p: vec3<f32>, cell_size_m: f32) -> f32 {
  let cell = floor(world_p / cell_size_m);
  return (hash31(cell) - 0.5) * 2.0;
}

fn apply_material_variation(base: vec3<f32>, mat: u32, hp: vec3<f32>) -> vec3<f32> {
  var c = base;
  let v = material_variation(hp, 0.05);

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
  node_base: u32
) -> f32 {
  let r = 0.75 * cam.voxel_params.x;

  let up_ref = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
  let t = normalize(cross(up_ref, n));
  let b = normalize(cross(n, t));

  var occ = 0.0;

  let q0 = query_leaf_at(hp + t * r, root_bmin, root_size, node_base);
  occ += select(0.0, 1.0, q0.mat != MAT_AIR);

  let q1 = query_leaf_at(hp - t * r, root_bmin, root_size, node_base);
  occ += select(0.0, 1.0, q1.mat != MAT_AIR);

  let q2 = query_leaf_at(hp + b * r, root_bmin, root_size, node_base);
  occ += select(0.0, 1.0, q2.mat != MAT_AIR);

  let q3 = query_leaf_at(hp - b * r, root_bmin, root_size, node_base);
  occ += select(0.0, 1.0, q3.mat != MAT_AIR);

  let h0 = normalize(n + 0.65 * t + 0.35 * b);
  let q4 = query_leaf_at(hp + h0 * r, root_bmin, root_size, node_base);
  occ += select(0.0, 1.0, q4.mat != MAT_AIR);

  let h1 = normalize(n - 0.65 * t + 0.35 * b);
  let q5 = query_leaf_at(hp + h1 * r, root_bmin, root_size, node_base);
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
  return 0.90;
}

fn material_f0(mat: u32) -> f32 {
  if (mat == MAT_STONE) { return 0.04; }
  if (mat == MAT_WOOD)  { return 0.03; }
  if (mat == MAT_LEAF)  { return 0.05; }
  if (mat == MAT_GRASS) { return 0.04; }
  if (mat == MAT_DIRT)  { return 0.02; }
  return 0.02;
}

fn shade_hit(ro: vec3<f32>, rd: vec3<f32>, hg: HitGeom, sky_up: vec3<f32>) -> vec3<f32> {
  let hp = ro + hg.t * rd;

  var base = color_for_material(hg.mat);
  base = apply_material_variation(base, hg.mat, hp);

  if (hg.mat == MAT_GRASS) {
    let vs = cam.voxel_params.x;
    let tip = clamp(fract(hp.y / max(vs, 1e-6)), 0.0, 1.0);

    base = mix(base, base + vec3<f32>(0.10, 0.10, 0.02), 0.35 * tip);

    let back = pow(clamp(dot(-SUN_DIR, hg.n), 0.0, 1.0), 2.0);
    base += 0.18 * back * vec3<f32>(0.20, 0.35, 0.08);
  }

  let vs = cam.voxel_params.x;
  let hp_shadow  = hp + hg.n * (0.75 * vs);

  let vis  = sun_transmittance(hp_shadow, SUN_DIR);
  let diff = max(dot(hg.n, SUN_DIR), 0.0);

  let ao = select(1.0, voxel_ao_local(hp, hg.n, hg.root_bmin, hg.root_size, hg.node_base), hg.hit != 0u);

  let amb_col      = hemi_ambient(hg.n, sky_up);
  let amb_strength = select(0.10, 0.14, hg.mat == MAT_LEAF);
  let ambient      = amb_col * amb_strength * ao;

  var dapple = 1.0;
  if (hg.mat == MAT_LEAF) {
    let time_s = cam.voxel_params.y;
    let d0 = sin(dot(hp.xz, vec2<f32>(3.0, 2.2)) + time_s * 3.5);
    let d1 = sin(dot(hp.xz, vec2<f32>(6.5, 4.1)) - time_s * 6.0);
    dapple = 0.90 + 0.10 * (0.6 * d0 + 0.4 * d1);
  }

  let direct = SUN_COLOR * SUN_INTENSITY * (diff * diff) * vis * dapple;

  let v = normalize(-rd);
  let h = normalize(v + SUN_DIR);

  let ndv = max(dot(hg.n, v), 0.0);
  let ndh = max(dot(hg.n, h), 0.0);

  let rough     = material_roughness(hg.mat);
  let shininess = mix(8.0, 96.0, 1.0 - rough);
  let spec      = pow(ndh, shininess);

  let f0   = material_f0(hg.mat);
  let fres = fresnel_schlick(ndv, f0);

  let spec_col = SUN_COLOR * SUN_INTENSITY * spec * fres * vis;

  return base * (ambient + direct) + 0.20 * spec_col;
}
