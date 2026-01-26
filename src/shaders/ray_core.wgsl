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
  p: vec3<f32>,
  root_bmin: vec3<f32>,
  root_size: f32,
  node_base: u32
) -> LeafQuery {
  var idx: u32 = node_base;
  var bmin: vec3<f32> = root_bmin;
  var size: f32 = root_size;

  let min_leaf: f32 = cam.voxel_params.x;
  let eps: f32 = 1e-7;

  for (var d: u32 = 0u; d < 32u; d = d + 1u) {
    let n = nodes[idx];

    if (n.child_base == LEAF_U32) {
      return LeafQuery(bmin, size, n.material);
    }

    if (size <= min_leaf + eps) {
      return LeafQuery(bmin, size, 0u);
    }

    let half = size * 0.5;
    let mid  = bmin + vec3<f32>(half);

    let hx = select(0u, 1u, p.x >= mid.x);
    let hy = select(0u, 1u, p.y >= mid.y);
    let hz = select(0u, 1u, p.z >= mid.z);
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
  // chunk-local constants
  let voxel_size = cam.voxel_params.x;

  let root_bmin_vox = vec3<f32>(f32(ch.origin.x), f32(ch.origin.y), f32(ch.origin.z));
  let root_bmin = root_bmin_vox * voxel_size;
  let root_size = f32(cam.chunk_size) * voxel_size;

  // We *trust* interval from grid DDA. Just clamp + nudge.
  var tcur = max(t_enter, 0.0) + 1e-4;

  // Precompute inv once (3D)
  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  for (var step_i: u32 = 0u; step_i < cam.max_steps; step_i = step_i + 1u) {
    if (tcur > t_exit) { break; }

    let p = ro + tcur * rd;
    let q = query_leaf_at(p, root_bmin, root_size, ch.node_base);

    if (q.mat != 0u) {
      // Leaves: intersect displaced cube for visible cube motion.
      if (q.mat == 5u) {
        let time_s   = cam.voxel_params.y;
        let strength = cam.voxel_params.z;

        let h2 = leaf_displaced_cube_hit(
          ro, rd,
          q.bmin, q.size,
          time_s, strength,
          tcur - 1e-4,
          t_exit
        );

        if (h2.hit) {
          return HitGeom(true, h2.t, 5u, h2.n);
        }

        // Displaced cube missed: treat as empty and keep marching.
        let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
        tcur = max(t_leave, tcur) + 1e-4;
        continue;
      }

      // Other materials: shade static cube face normal at current sample.
      let hp = ro + tcur * rd;
      let nn = cube_normal(hp, q.bmin, q.size);
      return HitGeom(true, tcur, q.mat, nn);
    }

    // Skip empty region.
    let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
    tcur = max(t_leave, tcur) + 1e-4;
  }

  return HitGeom(false, BIG_F32, 0u, vec3<f32>(0.0));
}

