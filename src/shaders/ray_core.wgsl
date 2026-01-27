// ray_core.wgsl
//
// Core SVO queries + primary traversal helpers.
//
// This file provides the "inner loop" building blocks for ray traversal through a chunk:
// - safe inverse for ray directions (avoid INF/NaN on near-zero components)
// - point query into an SVO (query_leaf_at): returns the leaf cell containing a point
// - fast "skip" step: compute when a ray exits the current cell (exit_time_from_cube_inv)
// - chunk tracing loop over a trusted [t_enter, t_exit] interval
// - AABB hit with stable normal selection (aabb_hit_normal_inv)
//
// Assumptions / dependencies:
// - Uses globals from common.wgsl: BIG_F32, EPS_INV, LEAF_U32, cam, nodes, child_rank, ChunkMeta.
// - Uses leaf_displaced_cube_hit() for animated/displaced materials (material id 5).
// - All distances are in world meters; voxel_size comes from cam.voxel_params.x.

// -----------------------------------------------------------------------------
// Numeric helpers
// -----------------------------------------------------------------------------

/// Safe reciprocal for ray directions.
///
/// If |x| is extremely small, return a large finite value instead of 1/x.
/// This avoids INF/NaN in subsequent multiply-based intersection math.
fn safe_inv(x: f32) -> f32 {
  return select(1.0 / x, BIG_F32, abs(x) < EPS_INV);
}

// -----------------------------------------------------------------------------
// Leaf query: point -> leaf cell in SVO
// -----------------------------------------------------------------------------

/// Result of querying the SVO at a point.
///
/// bmin/size describe the axis-aligned cube that contains the point.
/// mat is the leaf material (0 means empty).
struct LeafQuery {
  bmin : vec3<f32>, // cube minimum corner in world meters
  size : f32,       // cube edge length in world meters
  mat  : u32,       // material id; 0 = empty/air
};

/// Walk the SVO to find the leaf cell that contains point `p_in`.
///
/// Inputs:
/// - p_in       : query point in world meters
/// - root_bmin  : chunk root cube minimum in world meters
/// - root_size  : chunk root cube size in world meters
/// - node_base  : base index into the global `nodes` arena for this chunk
///
/// Behavior:
/// - Descends up to 32 levels.
/// - Uses a half-open split rule so points on split planes go to the "lower" child
///   (helps prevent flicker due to tie ambiguity).
/// - If a desired child doesn't exist in child_mask, returns that child cell as empty.
/// - If the cell becomes smaller than `min_leaf` (voxel_size), returns empty as a guard.
fn query_leaf_at(
  p_in: vec3<f32>,
  root_bmin: vec3<f32>,
  root_size: f32,
  node_base: u32
) -> LeafQuery {
  // Current node index in the nodes arena.
  var idx: u32 = node_base;

  // Current cube bounds as we descend.
  var bmin: vec3<f32> = root_bmin;
  var size: f32 = root_size;

  // Minimum leaf size (one voxel in world meters).
  let min_leaf: f32 = cam.voxel_params.x;

  // Local copy (we sometimes tweak the point slightly to avoid ties).
  var p = p_in;

  // Hard cap depth to avoid infinite loops on corrupt data.
  for (var d: u32 = 0u; d < 32u; d = d + 1u) {
    let n = nodes[idx];

    // Leaf node: material stored directly in node.
    if (n.child_base == LEAF_U32) {
      return LeafQuery(bmin, size, n.material);
    }

    // Safety: if we got smaller than a voxel, stop descending and treat as empty.
    if (size <= min_leaf) {
      return LeafQuery(bmin, size, 0u);
    }

    // Split cube into 8 octants.
    let half = size * 0.5;
    let mid  = bmin + vec3<f32>(half);

    // Epsilon proportional to cell size to stabilize comparisons near split planes.
    let e = 1e-6 * size;

    // HALF-OPEN child selection:
    // Only strictly greater than mid + e goes to the "upper" half; ties stay in lower.
    let hx = select(0u, 1u, p.x > mid.x + e);
    let hy = select(0u, 1u, p.y > mid.y + e);
    let hz = select(0u, 1u, p.z > mid.z + e);
    let ci = hx | (hy << 1u) | (hz << 2u); // child index 0..7

    // Compute selected child cube minimum.
    let child_bmin = bmin + vec3<f32>(
      select(0.0, half, hx != 0u),
      select(0.0, half, hy != 0u),
      select(0.0, half, hz != 0u)
    );

    // If this child doesn't exist, it's empty.
    let bit = 1u << ci;
    if ((n.child_mask & bit) == 0u) {
      return LeafQuery(child_bmin, half, 0u);
    }

    // Compact child list addressing:
    // rank = number of set bits below ci, giving offset into compact array.
    let rank = child_rank(n.child_mask, ci);

    // Move to the child node:
    // idx = node_base + (child_base + rank)
    // (child_base is stored relative to node_base for this chunk.)
    idx = node_base + (n.child_base + rank);

    // Update cube bounds for next iteration.
    bmin = child_bmin;
    size = half;
  }

  // If we hit the depth cap, treat as empty.
  return LeafQuery(bmin, size, 0u);
}

// -----------------------------------------------------------------------------
// Fast stepping: "skip to exit" time for the current cube
// -----------------------------------------------------------------------------

/// Compute the parametric t where the ray exits an axis-aligned cube.
///
/// Uses precomputed inv direction for speed and numerical consistency.
/// This is used as a "skip" step when the current cell is empty (or misses).
///
/// Note:
/// - This returns the *earliest* crossing among x/y/z exit planes along the ray direction.
fn exit_time_from_cube_inv(
  ro: vec3<f32>,
  rd: vec3<f32>,
  inv: vec3<f32>,
  bmin: vec3<f32>,
  size: f32
) -> f32 {
  let bmax = bmin + vec3<f32>(size);

  // For each axis, choose the far plane in the direction we travel.
  // If rd.x > 0 => exiting plane is bmax.x else bmin.x, etc.
  let tx = (select(bmin.x, bmax.x, rd.x > 0.0) - ro.x) * inv.x;
  let ty = (select(bmin.y, bmax.y, rd.y > 0.0) - ro.y) * inv.y;
  let tz = (select(bmin.z, bmax.z, rd.z > 0.0) - ro.z) * inv.z;

  // Earliest exit among axes.
  return min(tx, min(ty, tz));
}

// -----------------------------------------------------------------------------
// Chunk tracing: hybrid point-query + interval stepping
// -----------------------------------------------------------------------------

/// Hit record returned by tracing within a chunk.
struct HitGeom {
  hit : bool,      // true if a surface was hit
  t   : f32,       // ray parameter at hit
  mat : u32,       // material id
  n   : vec3<f32>, // geometric normal (axis-aligned for cubes; custom for displaced)
};

/// Trace a ray through a single chunk over a trusted interval [t_enter, t_exit].
///
/// Strategy:
/// - March along the ray, but at each step query the SVO leaf cell containing the point.
/// - If the cell is empty, jump directly to the cell's exit time (fast skip).
/// - If the cell is solid, compute a stable AABB entry time + normal within the trusted interval.
/// - Special case: material 5 is "displaced cube" (animated surface) and uses a specialized hit test.
///
/// Why "trusted interval":
/// Typically you first intersect the ray with the chunk's root AABB, yielding [t_enter, t_exit].
/// All subsequent hits are constrained to that interval to prevent neighbors/precision issues.
fn trace_chunk_hybrid_interval(
  ro: vec3<f32>,
  rd: vec3<f32>,
  ch: ChunkMeta,
  t_enter: f32,
  t_exit: f32
) -> HitGeom {
  // World-space size of one voxel.
  let voxel_size = cam.voxel_params.x;

  // Chunk root cube in world meters.
  let root_bmin_vox = vec3<f32>(f32(ch.origin.x), f32(ch.origin.y), f32(ch.origin.z));
  let root_bmin = root_bmin_vox * voxel_size;
  let root_size = f32(cam.chunk_size) * voxel_size;

  // Small positive nudge along the ray to avoid getting stuck on boundaries.
  // Must be tiny; too large pushes you into neighbor cells.
  let eps_step = 1e-4 * voxel_size;

  // Start at the interval entry, clamped to t>=0, with a small forward nudge.
  var tcur = max(t_enter, 0.0) + eps_step;

  // Precompute inv direction for intersection/exit math.
  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  // Main stepping loop: bounded by cam.max_steps.
  for (var step_i: u32 = 0u; step_i < cam.max_steps; step_i = step_i + 1u) {
    if (tcur > t_exit) { break; }

    // Current point on ray.
    // We query a slightly-forward point (pq) to avoid split-plane ties.
    let p  = ro + tcur * rd;
    let pq = p + rd * (1e-4 * voxel_size);

    // SVO point query: which leaf cell are we in, and what is its material?
    let q = query_leaf_at(pq, root_bmin, root_size, ch.node_base);

    // Non-empty cell: try to produce an actual surface hit.
    if (q.mat != 0u) {
      // ----- Special material: displaced cube (animated surface) -----
      //
      // Instead of using the undisplaced AABB, we delegate to a function that
      // models displacement over time and returns hit time and normal.
      if (q.mat == 5u) {
        let time_s   = cam.voxel_params.y;
        let strength = cam.voxel_params.z;

        // Use the full trusted interval, not just local tcur, for robust intersection.
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

        // Displaced surface missed inside this cell. Treat it as empty and skip out.
        let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
        tcur = max(t_leave, tcur) + eps_step;
        continue;
      }

      // ----- Solid (undisplaced) cube hit -----
      //
      // Compute entry time and a stable face normal within [t_enter, t_exit].
      let bh = aabb_hit_normal_inv(
        ro, rd, inv,
        q.bmin, q.size,
        t_enter,
        t_exit
      );

      if (bh.hit) {
        return HitGeom(true, bh.t, q.mat, bh.n);
      }

      // If we didn't get a valid hit in the trusted interval (should be rare),
      // treat it like empty and skip out of the cell to keep forward progress.
      let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
      tcur = max(t_leave, tcur) + eps_step;
      continue;
    }

    // Empty cell: skip directly to the cell exit time (fast traversal).
    let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
    tcur = max(t_leave, tcur) + eps_step;
  }

  // No hit within step budget / interval.
  return HitGeom(false, BIG_F32, 0u, vec3<f32>(0.0));
}

// -----------------------------------------------------------------------------
// AABB hit + stable normal selection
// -----------------------------------------------------------------------------

/// Box hit record used for solid cube intersection.
struct BoxHit {
  hit : bool,
  t   : f32,       // entry time (clamped to t_min)
  n   : vec3<f32>, // chosen face normal
};

/// AABB intersection using precomputed inv direction, plus a stable normal.
///
/// Inputs:
/// - ro/rd/inv: ray origin, direction, and safe inverse direction
/// - bmin/size: cube definition
/// - t_min/t_max: trusted interval clamp
///
/// Returns:
/// - hit=false if the AABB interval does not overlap [t_min, t_max]
/// - otherwise hit=true, t = clamped entry time, n = stable face normal
///
/// Normal stability:
/// - In edge/corner hits, multiple slabs can produce the same t_enter.
/// - We choose the axis most aligned with the ray (largest |rd|) among tied slabs,
///   which tends to reduce flickery axis switching when grazing edges.
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

  // Slab times along each axis (two planes per axis).
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

  // Combined interval across axes.
  let t_enter = max(tminx, max(tminy, tminz));
  let t_exit  = min(tmaxx, min(tmaxy, tmaxz));

  // Clamp entry time to trusted interval start.
  let t0 = max(t_enter, t_min);

  // Reject if the AABB interval doesn't overlap the trusted [t_min, t_max].
  if (t_exit < t0 || t0 > t_max) {
    return BoxHit(false, BIG_F32, vec3<f32>(0.0));
  }

  // Pick which slab produced t_enter with epsilon and a stable tie-break.
  let eps = 1e-6 * size;

  var n = vec3<f32>(0.0);

  // best_abs tracks the largest |rd.axis| among tied candidate slabs.
  var best_abs = -1.0;
  var pick: u32 = 0u; // 0=x, 1=y, 2=z

  // Candidate: X slab
  if (abs(t_enter - tminx) <= eps) {
    let a = abs(rd.x);
    if (a > best_abs) { best_abs = a; pick = 0u; }
  }
  // Candidate: Y slab
  if (abs(t_enter - tminy) <= eps) {
    let a = abs(rd.y);
    if (a > best_abs) { best_abs = a; pick = 1u; }
  }
  // Candidate: Z slab
  if (abs(t_enter - tminz) <= eps) {
    let a = abs(rd.z);
    if (a > best_abs) { best_abs = a; pick = 2u; }
  }

  // Normal points against the ray direction (entering face).
  if (pick == 0u) { n = vec3<f32>(select( 1.0, -1.0, rd.x > 0.0), 0.0, 0.0); }
  if (pick == 1u) { n = vec3<f32>(0.0, select( 1.0, -1.0, rd.y > 0.0), 0.0); }
  if (pick == 2u) { n = vec3<f32>(0.0, 0.0, select( 1.0, -1.0, rd.z > 0.0)); }

  return BoxHit(true, t0, n);
}
