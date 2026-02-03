// src/shaders/ray/aabb.wgsl
//// --------------------------------------------------------------------------
//// AABB helpers (vectorized slab + reuse)
//// --------------------------------------------------------------------------

fn max3(v: vec3<f32>) -> f32 { return max(v.x, max(v.y, v.z)); }
fn min3(v: vec3<f32>) -> f32 { return min(v.x, min(v.y, v.z)); }

struct CubeSlab {
  tminv   : vec3<f32>,
  tmaxv   : vec3<f32>,
  t_enter : f32,
  t_exit  : f32,
};

// Compute slab interval for axis-aligned cube using precomputed inv(rd).
fn cube_slab_inv(
  ro: vec3<f32>,
  inv: vec3<f32>,
  bmin: vec3<f32>,
  size: f32
) -> CubeSlab {
  let bmax = bmin + vec3<f32>(size);

  let t0 = (bmin - ro) * inv;
  let t1 = (bmax - ro) * inv;

  let tminv = min(t0, t1);
  let tmaxv = max(t0, t1);

  let t_enter = max3(tminv);
  let t_exit  = min3(tmaxv);

  return CubeSlab(tminv, tmaxv, t_enter, t_exit);
}

struct BoxHit {
  hit    : bool,
  t      : f32,
  t_exit : f32,        // <-- add this so callsites can step without recomputing
  n      : vec3<f32>,
};

// Normal + hit interval computed from a precomputed slab.
// (Uses argmax(tminv) for face selection; no eps comparisons.)
fn cube_hit_normal_from_slab(
  rd: vec3<f32>,
  slab: CubeSlab,
  t_min: f32,
  t_max: f32
) -> BoxHit {
  let t0 = max(slab.t_enter, t_min);

  if (slab.t_exit < t0 || t0 > t_max) {
    return BoxHit(false, BIG_F32, slab.t_exit, vec3<f32>(0.0));
  }

  // Which axis produced t_enter? (argmax of tminv)
  var axis: u32 = 0u;
  var best: f32 = slab.tminv.x;
  if (slab.tminv.y > best) { best = slab.tminv.y; axis = 1u; }
  if (slab.tminv.z > best) { best = slab.tminv.z; axis = 2u; }

  var n = vec3<f32>(0.0);
  if (axis == 0u) { n = vec3<f32>(select( 1.0, -1.0, rd.x > 0.0), 0.0, 0.0); }
  if (axis == 1u) { n = vec3<f32>(0.0, select( 1.0, -1.0, rd.y > 0.0), 0.0); }
  if (axis == 2u) { n = vec3<f32>(0.0, 0.0, select( 1.0, -1.0, rd.z > 0.0)); }

  return BoxHit(true, t0, slab.t_exit, n);
}

// Keep your original signature, but now vectorized internally.
fn aabb_hit_normal_inv(
  ro: vec3<f32>,
  rd: vec3<f32>,
  inv: vec3<f32>,
  bmin: vec3<f32>,
  size: f32,
  t_min: f32,
  t_max: f32
) -> BoxHit {
  let slab = cube_slab_inv(ro, inv, bmin, size);
  return cube_hit_normal_from_slab(rd, slab, t_min, t_max);
}
