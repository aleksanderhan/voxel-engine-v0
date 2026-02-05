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
  rd: vec3<f32>,
  inv: vec3<f32>,
  bmin: vec3<f32>,
  size: f32
) -> CubeSlab {
  let bmax = bmin + vec3<f32>(size);

  // Fast path: fully non-parallel ray (no per-axis branching needed).
  if (all(abs(rd) >= vec3<f32>(EPS_INV))) {
    let t0 = (bmin - ro) * inv;
    let t1 = (bmax - ro) * inv;

    let tminv = min(t0, t1);
    let tmaxv = max(t0, t1);

    let t_enter = max3(tminv);
    let t_exit  = min3(tmaxv);

    return CubeSlab(tminv, tmaxv, t_enter, t_exit);
  }

  // Start with a valid “wide open” interval; we’ll fill per-axis.
  var t0 = vec3<f32>(0.0);
  var t1 = vec3<f32>(0.0);

  // X axis
  if (abs(rd.x) >= EPS_INV) {
    t0.x = (bmin.x - ro.x) * inv.x;
    t1.x = (bmax.x - ro.x) * inv.x;
  } else {
    // Ray parallel to X slabs: must already be within [bmin.x, bmax.x]
    if (ro.x < bmin.x || ro.x > bmax.x) {
      return CubeSlab(vec3<f32>(BIG_F32), vec3<f32>(-BIG_F32), BIG_F32, -BIG_F32);
    }
    t0.x = -BIG_F32;
    t1.x =  BIG_F32;
  }

  // Y axis
  if (abs(rd.y) >= EPS_INV) {
    t0.y = (bmin.y - ro.y) * inv.y;
    t1.y = (bmax.y - ro.y) * inv.y;
  } else {
    if (ro.y < bmin.y || ro.y > bmax.y) {
      return CubeSlab(vec3<f32>(BIG_F32), vec3<f32>(-BIG_F32), BIG_F32, -BIG_F32);
    }
    t0.y = -BIG_F32;
    t1.y =  BIG_F32;
  }

  // Z axis
  if (abs(rd.z) >= EPS_INV) {
    t0.z = (bmin.z - ro.z) * inv.z;
    t1.z = (bmax.z - ro.z) * inv.z;
  } else {
    if (ro.z < bmin.z || ro.z > bmax.z) {
      return CubeSlab(vec3<f32>(BIG_F32), vec3<f32>(-BIG_F32), BIG_F32, -BIG_F32);
    }
    t0.z = -BIG_F32;
    t1.z =  BIG_F32;
  }

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

  // If we are using the true entry (t_enter >= t_min), normal comes from entry face.
  // If t_enter < t_min, we are effectively "inside/clamped", so normal must come from exit face.
  let use_entry = slab.t_enter >= t_min;

  var axis: u32 = 0u;

  if (use_entry) {
    // argmax(tminv) => entry face
    var best: f32 = slab.tminv.x;
    if (slab.tminv.y > best) { best = slab.tminv.y; axis = 1u; }
    if (slab.tminv.z > best) { best = slab.tminv.z; axis = 2u; }

    var n = vec3<f32>(0.0);
    // entry normal points *against* ray direction
    if (axis == 0u) { n = vec3<f32>(select( 1.0, -1.0, rd.x > 0.0), 0.0, 0.0); }
    if (axis == 1u) { n = vec3<f32>(0.0, select( 1.0, -1.0, rd.y > 0.0), 0.0); }
    if (axis == 2u) { n = vec3<f32>(0.0, 0.0, select( 1.0, -1.0, rd.z > 0.0)); }

    return BoxHit(true, t0, slab.t_exit, n);
  } else {
    // argmin(tmaxv) => exit face
    var best: f32 = slab.tmaxv.x;
    if (slab.tmaxv.y < best) { best = slab.tmaxv.y; axis = 1u; }
    if (slab.tmaxv.z < best) { best = slab.tmaxv.z; axis = 2u; }

    var n = vec3<f32>(0.0);
    // exit normal points *with* ray direction
    if (axis == 0u) { n = vec3<f32>(select(-1.0,  1.0, rd.x > 0.0), 0.0, 0.0); }
    if (axis == 1u) { n = vec3<f32>(0.0, select(-1.0,  1.0, rd.y > 0.0), 0.0); }
    if (axis == 2u) { n = vec3<f32>(0.0, 0.0, select(-1.0,  1.0, rd.z > 0.0)); }

    return BoxHit(true, t0, slab.t_exit, n);
  }
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
  let slab = cube_slab_inv(ro, rd, inv, bmin, size);
  return cube_hit_normal_from_slab(rd, slab, t_min, t_max);
}
