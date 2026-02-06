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

  // Start with the usual slab math
  var t0 = (bmin - ro) * inv;
  var t1 = (bmax - ro) * inv;

  // If safe_inv() uses a huge sentinel for rd≈0, detect that here.
  // Pick something far above any real 1/rd you expect.
  let INV_PARALLEL: f32 = 1e20;

  // X axis parallel handling
  if (abs(inv.x) > INV_PARALLEL) {
    // Half-open cube: [bmin, bmax). Being at/above bmax counts as outside.
    if (ro.x < bmin.x || ro.x >= bmax.x) {
      // Force a miss: enter > exit
      return CubeSlab(
        vec3<f32>( BIG_F32),
        vec3<f32>(-BIG_F32),
        BIG_F32,
        -BIG_F32
      );
    }
    // Inside the slab: make axis interval infinite so it doesn't constrain.
    t0.x = -BIG_F32;
    t1.x =  BIG_F32;
  }

  // Y axis
  if (abs(inv.y) > INV_PARALLEL) {
    if (ro.y < bmin.y || ro.y >= bmax.y) {
      return CubeSlab(
        vec3<f32>( BIG_F32),
        vec3<f32>(-BIG_F32),
        BIG_F32,
        -BIG_F32
      );
    }
    t0.y = -BIG_F32;
    t1.y =  BIG_F32;
  }

  // Z axis
  if (abs(inv.z) > INV_PARALLEL) {
    if (ro.z < bmin.z || ro.z >= bmax.z) {
      return CubeSlab(
        vec3<f32>( BIG_F32),
        vec3<f32>(-BIG_F32),
        BIG_F32,
        -BIG_F32
      );
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

// Normal + hit interval computed from a precomputed slab.
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

  // --- robust axis selection (handles ties on edges/corners) ---
  let te = slab.t_enter;

  // eps scaled to the magnitude of te (keeps it stable across distances)
  let eps = 1e-5 * max(1.0, abs(te));

  let mx = abs(slab.tminv.x - te) <= eps;
  let my = abs(slab.tminv.y - te) <= eps;
  let mz = abs(slab.tminv.z - te) <= eps;

  // If nothing matches due to weird numeric cases, fall back to argmax
  var axis: u32 = 0u;

  if (mx || my || mz) {
    // tie-break using largest |rd| among matching axes
    var best: f32 = -1.0;

    if (mx) { best = abs(rd.x); axis = 0u; }
    if (my && abs(rd.y) > best) { best = abs(rd.y); axis = 1u; }
    if (mz && abs(rd.z) > best) { best = abs(rd.z); axis = 2u; }
  } else {
    // fallback: argmax(tminv)
    var bestv: f32 = slab.tminv.x;
    axis = 0u;
    if (slab.tminv.y > bestv) { bestv = slab.tminv.y; axis = 1u; }
    if (slab.tminv.z > bestv) { bestv = slab.tminv.z; axis = 2u; }
  }

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
