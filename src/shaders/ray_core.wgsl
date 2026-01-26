// ray_core.wgsl
//
// Core SVO queries + primary traversal.

fn safe_inv(x: f32) -> f32 {
  return select(1.0 / x, BIG_F32, abs(x) < EPS_INV);
}

struct LeafQuery {
  bmin : vec3<f32>, // cube min (world meters)
  size : f32,       // cube size (world meters)
  mat  : u32,       // 0 = empty / air
};

fn query_leaf_at(
  p_in: vec3<f32>,
  root_bmin: vec3<f32>,
  root_size: f32,
  node_base: u32
) -> LeafQuery {
  var idx: u32 = node_base;
  var bmin: vec3<f32> = root_bmin;
  var size: f32 = root_size;

  let min_leaf: f32 = cam.voxel_params.x;

  var p = p_in;

  for (var d: u32 = 0u; d < 32u; d = d + 1u) {
    let n = nodes[idx];

    if (n.child_base == LEAF_U32) {
      return LeafQuery(bmin, size, n.material);
    }

    if (size <= min_leaf) {
      return LeafQuery(bmin, size, 0u);
    }

    let half = size * 0.5;
    let mid  = bmin + vec3<f32>(half);

    // epsilon proportional to current cell size
    let e = 1e-6 * size;

    // HALF-OPEN: points exactly on the plane go to the "lower" child
    let hx = select(0u, 1u, p.x > mid.x + e);
    let hy = select(0u, 1u, p.y > mid.y + e);
    let hz = select(0u, 1u, p.z > mid.z + e);
    let ci = hx | (hy << 1u) | (hz << 2u);

    let child_bmin = bmin + vec3<f32>(
      select(0.0, half, hx != 0u),
      select(0.0, half, hy != 0u),
      select(0.0, half, hz != 0u)
    );

    let bit = 1u << ci;
    if ((n.child_mask & bit) == 0u) {
      return LeafQuery(child_bmin, half, 0u);
    }

    let rank = child_rank(n.child_mask, ci);
    idx = node_base + (n.child_base + rank);

    bmin = child_bmin;
    size = half;
  }

  return LeafQuery(bmin, size, 0u);
}

// Return t where the ray exits this axis-aligned cube, using a precomputed inv direction.
fn exit_time_from_cube_inv(
  ro: vec3<f32>,
  rd: vec3<f32>,
  inv: vec3<f32>,
  bmin: vec3<f32>,
  size: f32
) -> f32 {
  let bmax = bmin + vec3<f32>(size);

  let tx = (select(bmin.x, bmax.x, rd.x > 0.0) - ro.x) * inv.x;
  let ty = (select(bmin.y, bmax.y, rd.y > 0.0) - ro.y) * inv.y;
  let tz = (select(bmin.z, bmax.z, rd.z > 0.0) - ro.z) * inv.z;

  return min(tx, min(ty, tz));
}

struct HitGeom {
  hit : bool,
  t   : f32,
  mat : u32,
  n   : vec3<f32>,
};

fn trace_chunk_hybrid_interval(
  ro: vec3<f32>,
  rd: vec3<f32>,
  ch: ChunkMeta,
  t_enter: f32,
  t_exit: f32
) -> HitGeom {
  let voxel_size = cam.voxel_params.x;

  let root_bmin_vox = vec3<f32>(f32(ch.origin.x), f32(ch.origin.y), f32(ch.origin.z));
  let root_bmin = root_bmin_vox * voxel_size;
  let root_size = f32(cam.chunk_size) * voxel_size;

  // IMPORTANT: epsilon must be tiny. 0.25*voxel pushes you inside neighbors.
  let eps_step = 1e-4 * voxel_size;

  // Start at interval enter.
  var tcur = max(t_enter, 0.0) + eps_step;

  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  for (var step_i: u32 = 0u; step_i < cam.max_steps; step_i = step_i + 1u) {
    if (tcur > t_exit) { break; }

    // Query point slightly forward to avoid split-plane ties.
    let p  = ro + tcur * rd;
    let pq = p + rd * (1e-4 * voxel_size);

    let q = query_leaf_at(pq, root_bmin, root_size, ch.node_base);

    if (q.mat != 0u) {
      // ----- LEAVES (displaced cube) -----
      if (q.mat == 5u) {
        let time_s   = cam.voxel_params.y;
        let strength = cam.voxel_params.z;

        // Use the full trusted interval, not (tcur - nudge).
        let h2 = leaf_displaced_cube_hit(
          ro, rd,
          q.bmin, q.size,
          time_s, strength,
          t_enter,
          t_exit
        );

        if (h2.hit) {
          return HitGeom(true, h2.t, 5u, h2.n);
        }

        // Displaced leaf missed: treat as empty and skip out of the undisplaced cell.
        let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
        tcur = max(t_leave, tcur) + eps_step;
        continue;
      }

      // ----- SOLIDS -----
      // Compute the true entry time + stable normal over the trusted interval.
      let bh = aabb_hit_normal_inv(
        ro, rd, inv,
        q.bmin, q.size,
        t_enter,
        t_exit
      );

      if (bh.hit) {
        return HitGeom(true, bh.t, q.mat, bh.n);
      }

      // If somehow not hit in interval, skip.
      let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
      tcur = max(t_leave, tcur) + eps_step;
      continue;
    }

    // Empty: skip to cube exit
    let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
    tcur = max(t_leave, tcur) + eps_step;
  }

  return HitGeom(false, BIG_F32, 0u, vec3<f32>(0.0));
}


struct BoxHit {
  hit : bool,
  t   : f32,
  n   : vec3<f32>,
};

fn aabb_hit_normal_inv(
  ro: vec3<f32>,
  rd: vec3<f32>,
  inv: vec3<f32>,
  bmin: vec3<f32>,
  size: f32,
  t_min: f32,
  t_max: f32
) -> BoxHit {
  let bmax = bmin + vec3<f32>(size);

  let tx0 = (bmin.x - ro.x) * inv.x;
  let tx1 = (bmax.x - ro.x) * inv.x;
  let ty0 = (bmin.y - ro.y) * inv.y;
  let ty1 = (bmax.y - ro.y) * inv.y;
  let tz0 = (bmin.z - ro.z) * inv.z;
  let tz1 = (bmax.z - ro.z) * inv.z;

  let tminx = min(tx0, tx1);
  let tmaxx = max(tx0, tx1);
  let tminy = min(ty0, ty1);
  let tmaxy = max(ty0, ty1);
  let tminz = min(tz0, tz1);
  let tmaxz = max(tz0, tz1);

  let t_enter = max(tminx, max(tminy, tminz));
  let t_exit  = min(tmaxx, min(tmaxy, tmaxz));

  let t0 = max(t_enter, t_min);
  if (t_exit < t0 || t0 > t_max) {
    return BoxHit(false, BIG_F32, vec3<f32>(0.0));
  }

  // Pick which slab produced t_enter with epsilon and a stable tie-break.
  // (Edge/corner hits are ambiguous; we pick the axis most aligned with the ray to avoid flickery grids.)
  let eps = 1e-6 * size;

  var n = vec3<f32>(0.0);

  var best_abs = -1.0;
  var pick: u32 = 0u;

  if (abs(t_enter - tminx) <= eps) {
    let a = abs(rd.x);
    if (a > best_abs) { best_abs = a; pick = 0u; }
  }
  if (abs(t_enter - tminy) <= eps) {
    let a = abs(rd.y);
    if (a > best_abs) { best_abs = a; pick = 1u; }
  }
  if (abs(t_enter - tminz) <= eps) {
    let a = abs(rd.z);
    if (a > best_abs) { best_abs = a; pick = 2u; }
  }

  if (pick == 0u) { n = vec3<f32>(select( 1.0, -1.0, rd.x > 0.0), 0.0, 0.0); }
  if (pick == 1u) { n = vec3<f32>(0.0, select( 1.0, -1.0, rd.y > 0.0), 0.0); }
  if (pick == 2u) { n = vec3<f32>(0.0, 0.0, select( 1.0, -1.0, rd.z > 0.0)); }

  return BoxHit(true, t0, n);
}
