// ray_leaf_wind.wgsl
//
// Leaf wind field + displaced leaf cube intersection.

fn hash1(p: vec3<f32>) -> f32 {
  let h = dot(p, vec3<f32>(127.1, 311.7, 74.7));
  return fract(sin(h) * 43758.5453);
}

fn wind_field(pos_m: vec3<f32>, t: f32) -> vec3<f32> {
  let cell = floor(pos_m * 2.5);
  let ph0 = hash1(cell);
  let ph1 = hash1(cell + vec3<f32>(19.0, 7.0, 11.0));

  let dir = normalize(vec2<f32>(0.9, 0.4));

  let h = clamp((pos_m.y - 2.0) / 12.0, 0.0, 1.0);

  let gust    = sin(t * 0.9 + dot(pos_m.xz, vec2<f32>(0.35, 0.22)) + ph0 * 6.28318);
  let flutter = sin(t * 4.2 + dot(pos_m.xz, vec2<f32>(1.7,  1.1 )) + ph1 * 6.28318);

  let xz = dir * (0.75 * gust + 0.25 * flutter) * h;
  let y  = 0.25 * flutter * h;

  return vec3<f32>(xz.x, y, xz.y);
}

fn clamp_len(v: vec3<f32>, max_len: f32) -> vec3<f32> {
  let l2 = dot(v, v);
  if (l2 <= max_len * max_len) { return v; }
  return v * (max_len / sqrt(l2));
}

fn leaf_cube_offset(bmin: vec3<f32>, size: f32, time_s: f32, strength: f32) -> vec3<f32> {
  let center = bmin + vec3<f32>(0.5 * size);

  var w = wind_field(center, time_s) * strength;
  w = vec3<f32>(w.x, 0.15 * w.y, w.z);

  let amp = 0.35 * size;
  return clamp_len(w * amp, 0.45 * size);
}

struct LeafCubeHit {
  hit  : bool,
  t    : f32,
  n    : vec3<f32>,
};

fn leaf_displaced_cube_hit(
  ro: vec3<f32>,
  rd: vec3<f32>,
  bmin: vec3<f32>,
  size: f32,
  time_s: f32,
  strength: f32,
  t_min: f32,
  t_max: f32
) -> LeafCubeHit {
  let off   = leaf_cube_offset(bmin, size, time_s, strength);
  let bmin2 = bmin + off;

  // Use slab-style AABB (axis-aligned bounding box) hit normal to avoid edge/face flipping artifacts.
  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));
  let bh  = aabb_hit_normal_inv(ro, rd, inv, bmin2, size, t_min, t_max);

  return LeafCubeHit(bh.hit, bh.t, bh.n);
}
