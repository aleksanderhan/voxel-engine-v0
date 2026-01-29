// src/shaders/ray/leaves.wgsl
//// --------------------------------------------------------------------------
//// Leaves (displaced cube intersection)
//// --------------------------------------------------------------------------

fn leaf_cube_offset(bmin: vec3<f32>, size: f32, time_s: f32, strength: f32) -> vec3<f32> {
  let center = bmin + vec3<f32>(0.5 * size);

  // fade displacement with distance to camera
  let d = length(center - cam.cam_pos.xyz);
  let fade = smoothstep(LEAF_LOD_DISP_START, LEAF_LOD_DISP_END, d);
  let strength_eff = strength * (1.0 - fade);

  var w = wind_field(center, time_s) * strength_eff;
  w = vec3<f32>(w.x, LEAF_VERTICAL_REDUCE * w.y, w.z);

  let amp = LEAF_OFFSET_AMP * size;
  return clamp_len(w * amp, LEAF_OFFSET_MAX_FRAC * size);
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

  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));
  let bh  = aabb_hit_normal_inv(ro, rd, inv, bmin2, size, t_min, t_max);

  return LeafCubeHit(bh.hit, bh.t, bh.n);
}
