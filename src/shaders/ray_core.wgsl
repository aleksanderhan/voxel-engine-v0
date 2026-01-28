// ray_core.wgsl
//
// Consolidated core:
// - SVO queries + hybrid traversal
// - Leaf wind + displaced hit
// - Shadows + sun transmittance
// - Material palette + shading

// Depends on: common.wgsl (constants, structs, bindings, helpers)

fn safe_inv(x: f32) -> f32 {
  return select(1.0 / x, BIG_F32, abs(x) < EPS_INV);
}

// ------------------------------------------------------------
// Leaf query: point -> leaf cell in SVO
// ------------------------------------------------------------

struct LeafQuery {
  bmin : vec3<f32>,
  size : f32,
  mat  : u32,
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
      return LeafQuery(bmin, size, MAT_AIR);
    }

    let half = size * 0.5;
    let mid  = bmin + vec3<f32>(half);

    let e = 1e-6 * size;

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
      return LeafQuery(child_bmin, half, MAT_AIR);
    }

    let rank = child_rank(n.child_mask, ci);
    idx = node_base + (n.child_base + rank);

    bmin = child_bmin;
    size = half;
  }

  return LeafQuery(bmin, size, MAT_AIR);
}

// ------------------------------------------------------------
// Fast stepping: cube exit time with inv dir
// ------------------------------------------------------------

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

// ------------------------------------------------------------
// AABB hit + stable normal selection
// ------------------------------------------------------------

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

  let eps = 1e-6 * size;

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

  var n = vec3<f32>(0.0);
  if (pick == 0u) { n = vec3<f32>(select( 1.0, -1.0, rd.x > 0.0), 0.0, 0.0); }
  if (pick == 1u) { n = vec3<f32>(0.0, select( 1.0, -1.0, rd.y > 0.0), 0.0); }
  if (pick == 2u) { n = vec3<f32>(0.0, 0.0, select( 1.0, -1.0, rd.z > 0.0)); }

  return BoxHit(true, t0, n);
}

// ------------------------------------------------------------
// Leaf wind field + displaced cube hit
// ------------------------------------------------------------

fn hash1(p: vec3<f32>) -> f32 {
  let h = dot(p, vec3<f32>(127.1, 311.7, 74.7));
  return fract(sin(h) * 43758.5453);
}

fn wind_field(pos_m: vec3<f32>, t: f32) -> vec3<f32> {
  let cell = floor(pos_m * WIND_CELL_FREQ);

  let ph0 = hash1(cell);
  let ph1 = hash1(cell + WIND_PHASE_OFF_1);

  let dir = normalize(WIND_DIR_XZ);

  let h = clamp((pos_m.y - WIND_RAMP_Y0) / max(WIND_RAMP_Y1 - WIND_RAMP_Y0, 1e-3), 0.0, 1.0);

  let gust = sin(
    t * WIND_GUST_TIME_FREQ +
    dot(pos_m.xz, WIND_GUST_XZ_FREQ) +
    ph0 * TAU
  );

  let flutter = sin(
    t * WIND_FLUTTER_TIME_FREQ +
    dot(pos_m.xz, WIND_FLUTTER_XZ_FREQ) +
    ph1 * TAU
  );

  let xz = dir * (WIND_GUST_WEIGHT * gust + WIND_FLUTTER_WEIGHT * flutter) * h;
  let y  = WIND_VERTICAL_SCALE * flutter * h;

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

// ------------------------------------------------------------
// Chunk tracing: hybrid point-query + interval stepping
// ------------------------------------------------------------

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

  let eps_step = 1e-4 * voxel_size;

  var tcur = max(t_enter, 0.0) + eps_step;

  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  for (var step_i: u32 = 0u; step_i < cam.max_steps; step_i = step_i + 1u) {
    if (tcur > t_exit) { break; }

    let p  = ro + tcur * rd;
    let pq = p + rd * (1e-4 * voxel_size);

    let q = query_leaf_at(pq, root_bmin, root_size, ch.node_base);

    if (q.mat != MAT_AIR) {
      if (q.mat == MAT_LEAF) {
        let time_s   = cam.voxel_params.y;
        let strength = cam.voxel_params.z;

        let h2 = leaf_displaced_cube_hit(
          ro, rd,
          q.bmin, q.size,
          time_s, strength,
          t_enter,
          t_exit
        );

        if (h2.hit) {
          return HitGeom(true, h2.t, MAT_LEAF, h2.n);
        }

        let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
        tcur = max(t_leave, tcur) + eps_step;
        continue;
      }

      let bh = aabb_hit_normal_inv(
        ro, rd, inv,
        q.bmin, q.size,
        t_enter,
        t_exit
      );

      if (bh.hit) {
        return HitGeom(true, bh.t, q.mat, bh.n);
      }

      let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
      tcur = max(t_leave, tcur) + eps_step;
      continue;
    }

    let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
    tcur = max(t_leave, tcur) + eps_step;
  }

  return HitGeom(false, BIG_F32, MAT_AIR, vec3<f32>(0.0));
}

// ------------------------------------------------------------
// Shadow traversal
// ------------------------------------------------------------

fn trace_chunk_shadow_interval(
  ro: vec3<f32>,
  rd: vec3<f32>,
  ch: ChunkMeta,
  t_enter: f32,
  t_exit: f32
) -> bool {
  let voxel_size = cam.voxel_params.x;
  let nudge_s = 0.18 * voxel_size;

  let root_bmin_vox = vec3<f32>(f32(ch.origin.x), f32(ch.origin.y), f32(ch.origin.z));
  let root_bmin = root_bmin_vox * voxel_size;
  let root_size = f32(cam.chunk_size) * voxel_size;

  var tcur = max(t_enter, 0.0) + nudge_s;
  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  for (var step_i: u32 = 0u; step_i < SHADOW_STEPS; step_i = step_i + 1u) {
    if (tcur > t_exit) { break; }

    let p = ro + tcur * rd;
    let q = query_leaf_at(p, root_bmin, root_size, ch.node_base);

    if (q.mat != MAT_AIR) {
      if (q.mat == MAT_LEAF) {
        if (!SHADOW_DISPLACED_LEAVES) {
          return true;
        }

        let time_s   = cam.voxel_params.y;
        let strength = cam.voxel_params.z;

        let h2 = leaf_displaced_cube_hit(
          ro, rd,
          q.bmin, q.size,
          time_s, strength,
          tcur - nudge_s,
          t_exit
        );

        if (h2.hit) { return true; }

        let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
        tcur = max(t_leave, tcur) + nudge_s;
        continue;
      }

      return true;
    }

    let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
    tcur = max(t_leave, tcur) + nudge_s;
  }

  return false;
}

fn in_shadow(p: vec3<f32>, sun_dir: vec3<f32>) -> bool {
  let voxel_size   = cam.voxel_params.x;
  let nudge_s      = 0.18 * voxel_size;
  let chunk_size_m = f32(cam.chunk_size) * voxel_size;

  let go = cam.grid_origin_chunk;
  let gd = cam.grid_dims;

  let grid_bmin = vec3<f32>(
    f32(go.x) * chunk_size_m,
    f32(go.y) * chunk_size_m,
    f32(go.z) * chunk_size_m
  );

  let grid_bmax = grid_bmin + vec3<f32>(
    f32(gd.x) * chunk_size_m,
    f32(gd.y) * chunk_size_m,
    f32(gd.z) * chunk_size_m
  );

  let bias = max(SHADOW_BIAS, 0.50 * voxel_size);
  let ro   = p + sun_dir * bias;
  let rd   = sun_dir;

  let rtg = intersect_aabb(ro, rd, grid_bmin, grid_bmax);
  let t_enter = max(rtg.x, 0.0);
  let t_exit  = rtg.y;
  if (t_exit < t_enter) { return false; }

  let start_t = t_enter + nudge_s;
  let p0 = ro + start_t * rd;

  var t_local: f32 = 0.0;
  let t_exit_local = max(t_exit - start_t, 0.0);

  var c = chunk_coord_from_pos(p0, chunk_size_m);
  var cx: i32 = c.x;
  var cy: i32 = c.y;
  var cz: i32 = c.z;

  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  let step_x: i32 = select(-1, 1, rd.x > 0.0);
  let step_y: i32 = select(-1, 1, rd.y > 0.0);
  let step_z: i32 = select(-1, 1, rd.z > 0.0);

  let bx = select(f32(cx) * chunk_size_m, f32(cx + 1) * chunk_size_m, rd.x > 0.0);
  let by = select(f32(cy) * chunk_size_m, f32(cy + 1) * chunk_size_m, rd.y > 0.0);
  let bz = select(f32(cz) * chunk_size_m, f32(cz + 1) * chunk_size_m, rd.z > 0.0);

  var tMaxX: f32 = (bx - p0.x) * inv.x;
  var tMaxY: f32 = (by - p0.y) * inv.y;
  var tMaxZ: f32 = (bz - p0.z) * inv.z;

  let tDeltaX: f32 = abs(chunk_size_m * inv.x);
  let tDeltaY: f32 = abs(chunk_size_m * inv.y);
  let tDeltaZ: f32 = abs(chunk_size_m * inv.z);

  if (abs(rd.x) < EPS_INV) { tMaxX = BIG_F32; }
  if (abs(rd.y) < EPS_INV) { tMaxY = BIG_F32; }
  if (abs(rd.z) < EPS_INV) { tMaxZ = BIG_F32; }

  let max_chunk_steps = min((gd.x + gd.y + gd.z) * 6u + 8u, 1024u);

  for (var s: u32 = 0u; s < max_chunk_steps; s = s + 1u) {
    if (t_local > t_exit_local) { break; }

    let tNextLocal = min(tMaxX, min(tMaxY, tMaxZ));

    let slot = grid_lookup_slot(cx, cy, cz);
    if (slot != INVALID_U32 && slot < cam.chunk_count) {
      let ch = chunks[slot];

      let cell_enter = start_t + t_local;
      let cell_exit  = start_t + min(tNextLocal, t_exit_local);

      if (trace_chunk_shadow_interval(ro, rd, ch, cell_enter, cell_exit)) {
        return true;
      }
    }

    if (tMaxX < tMaxY) {
      if (tMaxX < tMaxZ) { cx += step_x; t_local = tMaxX; tMaxX += tDeltaX; }
      else               { cz += step_z; t_local = tMaxZ; tMaxZ += tDeltaZ; }
    } else {
      if (tMaxY < tMaxZ) { cy += step_y; t_local = tMaxY; tMaxY += tDeltaY; }
      else               { cz += step_z; t_local = tMaxZ; tMaxZ += tDeltaZ; }
    }

    let ox = cam.grid_origin_chunk.x;
    let oy = cam.grid_origin_chunk.y;
    let oz = cam.grid_origin_chunk.z;

    let nx = i32(cam.grid_dims.x);
    let ny = i32(cam.grid_dims.y);
    let nz = i32(cam.grid_dims.z);

    if (cx < ox || cy < oy || cz < oz || cx >= ox + nx || cy >= oy + ny || cz >= oz + nz) {
      break;
    }
  }

  return false;
}

fn trace_chunk_shadow_trans_interval(
  ro: vec3<f32>,
  rd: vec3<f32>,
  ch: ChunkMeta,
  t_enter: f32,
  t_exit: f32
) -> f32 {
  let voxel_size = cam.voxel_params.x;
  let nudge_s = 0.18 * voxel_size;

  let root_bmin_vox = vec3<f32>(f32(ch.origin.x), f32(ch.origin.y), f32(ch.origin.z));
  let root_bmin = root_bmin_vox * voxel_size;
  let root_size = f32(cam.chunk_size) * voxel_size;

  var tcur = max(t_enter, 0.0) + nudge_s;
  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  var trans = 1.0;

  for (var step_i: u32 = 0u; step_i < VSM_STEPS; step_i = step_i + 1u) {
    if (tcur > t_exit) { break; }
    if (trans < MIN_TRANS) { break; }

    let p = ro + tcur * rd;
    let qeps = 1e-4 * cam.voxel_params.x;
    let pq   = p + rd * qeps;

    let q = query_leaf_at(pq, root_bmin, root_size, ch.node_base);

    if (q.mat != MAT_AIR) {
      if (q.mat == MAT_LEAF) {
        trans *= LEAF_LIGHT_TRANSMIT;
        let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
        tcur = max(t_leave, tcur) + nudge_s;
        continue;
      }
      return 0.0;
    }

    let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
    tcur = max(t_leave, tcur) + nudge_s;
  }

  return trans;
}

fn sun_transmittance(p: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
  let Tc = cloud_sun_transmittance(p, sun_dir);

  let voxel_size   = cam.voxel_params.x;
  let nudge_s      = 0.18 * voxel_size;
  let chunk_size_m = f32(cam.chunk_size) * voxel_size;

  let go = cam.grid_origin_chunk;
  let gd = cam.grid_dims;

  let grid_bmin = vec3<f32>(
    f32(go.x) * chunk_size_m,
    f32(go.y) * chunk_size_m,
    f32(go.z) * chunk_size_m
  );

  let grid_bmax = grid_bmin + vec3<f32>(
    f32(gd.x) * chunk_size_m,
    f32(gd.y) * chunk_size_m,
    f32(gd.z) * chunk_size_m
  );

  let bias = max(SHADOW_BIAS, 0.50 * voxel_size);
  let ro   = p + sun_dir * bias;
  let rd   = sun_dir;

  let rtg = intersect_aabb(ro, rd, grid_bmin, grid_bmax);
  let t_enter = max(rtg.x, 0.0);
  let t_exit  = rtg.y;
  if (t_exit < t_enter) { return Tc; }

  let start_t = t_enter + nudge_s;
  let p0 = ro + start_t * rd;

  var t_local: f32 = 0.0;
  let t_exit_local = max(t_exit - start_t, 0.0);

  var c = chunk_coord_from_pos(p0, chunk_size_m);
  var cx: i32 = c.x;
  var cy: i32 = c.y;
  var cz: i32 = c.z;

  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  let step_x: i32 = select(-1, 1, rd.x > 0.0);
  let step_y: i32 = select(-1, 1, rd.y > 0.0);
  let step_z: i32 = select(-1, 1, rd.z > 0.0);

  let bx = select(f32(cx) * chunk_size_m, f32(cx + 1) * chunk_size_m, rd.x > 0.0);
  let by = select(f32(cy) * chunk_size_m, f32(cy + 1) * chunk_size_m, rd.y > 0.0);
  let bz = select(f32(cz) * chunk_size_m, f32(cz + 1) * chunk_size_m, rd.z > 0.0);

  var tMaxX: f32 = (bx - p0.x) * inv.x;
  var tMaxY: f32 = (by - p0.y) * inv.y;
  var tMaxZ: f32 = (bz - p0.z) * inv.z;

  let tDeltaX: f32 = abs(chunk_size_m * inv.x);
  let tDeltaY: f32 = abs(chunk_size_m * inv.y);
  let tDeltaZ: f32 = abs(chunk_size_m * inv.z);

  if (abs(rd.x) < EPS_INV) { tMaxX = BIG_F32; }
  if (abs(rd.y) < EPS_INV) { tMaxY = BIG_F32; }
  if (abs(rd.z) < EPS_INV) { tMaxZ = BIG_F32; }

  var trans = 1.0;

  let max_chunk_steps = min((gd.x + gd.y + gd.z) * 6u + 8u, 512u);

  for (var s: u32 = 0u; s < max_chunk_steps; s = s + 1u) {
    if (t_local > t_exit_local) { break; }
    if (trans < MIN_TRANS) { break; }

    let tNextLocal = min(tMaxX, min(tMaxY, tMaxZ));
    let slot = grid_lookup_slot(cx, cy, cz);

    if (slot != INVALID_U32 && slot < cam.chunk_count) {
      let ch = chunks[slot];

      let cell_enter = start_t + t_local;
      let cell_exit  = start_t + min(tNextLocal, t_exit_local);

      trans *= trace_chunk_shadow_trans_interval(ro, rd, ch, cell_enter, cell_exit);
      if (trans < MIN_TRANS) { break; }
    }

    if (tMaxX < tMaxY) {
      if (tMaxX < tMaxZ) { cx += step_x; t_local = tMaxX; tMaxX += tDeltaX; }
      else               { cz += step_z; t_local = tMaxZ; tMaxZ += tDeltaZ; }
    } else {
      if (tMaxY < tMaxZ) { cy += step_y; t_local = tMaxY; tMaxY += tDeltaY; }
      else               { cz += step_z; t_local = tMaxZ; tMaxZ += tDeltaZ; }
    }

    let ox = cam.grid_origin_chunk.x;
    let oy = cam.grid_origin_chunk.y;
    let oz = cam.grid_origin_chunk.z;

    let nx = i32(cam.grid_dims.x);
    let ny = i32(cam.grid_dims.y);
    let nz = i32(cam.grid_dims.z);

    if (cx < ox || cy < oy || cz < oz || cx >= ox + nx || cy >= oy + ny || cz >= oz + nz) {
      break;
    }
  }

  return trans * Tc;
}

// ------------------------------------------------------------
// Shading
// ------------------------------------------------------------

fn color_for_material(m: u32) -> vec3<f32> {
  if (m == MAT_AIR)   { return vec3<f32>(0.0); }

  if (m == MAT_GRASS) { return vec3<f32>(0.18, 0.75, 0.18); }
  if (m == MAT_DIRT)  { return vec3<f32>(0.45, 0.30, 0.15); }
  if (m == MAT_STONE) { return vec3<f32>(0.50, 0.50, 0.55); }
  if (m == MAT_WOOD)  { return vec3<f32>(0.38, 0.26, 0.14); }
  if (m == MAT_LEAF)  { return vec3<f32>(0.10, 0.55, 0.12); }

  return vec3<f32>(1.0, 0.0, 1.0);
}

fn hemi_ambient(n: vec3<f32>) -> vec3<f32> {
  let upw = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
  let sky = sky_color(vec3<f32>(0.0, 1.0, 0.0));
  let grd = FOG_COLOR_GROUND; // or a warmer ground tint
  return mix(grd, sky, upw);
}

fn hash31(p: vec3<f32>) -> f32 {
  let h = dot(p, vec3<f32>(127.1, 311.7, 74.7));
  return fract(sin(h) * 43758.5453);
}

fn material_variation(world_p: vec3<f32>, cell_size_m: f32) -> f32 {
  let cell = floor(world_p / cell_size_m);
  return (hash31(cell) - 0.5) * 2.0; // now v in [-1, +1]
}

fn apply_material_variation(base: vec3<f32>, mat: u32, hp: vec3<f32>) -> vec3<f32> {
  var c = base;

  // One shared noise value per patch (stable, no shimmering)
  let v = material_variation(hp, 0.05); // 0.05m patches for most things

  if (mat == MAT_GRASS) {
    // small additive tint + slight brightness
    c += vec3<f32>(0.02 * v, 0.05 * v, 0.01 * v);
    c *= (1.0 + 0.06 * v);
  } else if (mat == MAT_DIRT) {
    c += vec3<f32>(0.04 * v, 0.02 * v, 0.01 * v);
    c *= (1.0 + 0.08 * v);
  } else if (mat == MAT_STONE) {
    // stone mostly brightness variation, little hue
    c *= (1.0 + 0.10 * v);
  } else if (mat == MAT_WOOD) {
    // wood: warm variation
    c += vec3<f32>(0.05 * v, 0.02 * v, 0.00 * v);
    c *= (1.0 + 0.07 * v);
  } else if (mat == MAT_LEAF) {
    // keep leaves subtle; they already have dapple + wind
    c += vec3<f32>(0.00 * v, 0.03 * v, 0.00 * v);
    c *= (1.0 + 0.04 * v);
  }

  return clamp(c, vec3<f32>(0.0), vec3<f32>(1.5));
}

fn shade_hit(ro: vec3<f32>, rd: vec3<f32>, hg: HitGeom) -> vec3<f32> {
  let hp = ro + hg.t * rd;
  var base = color_for_material(hg.mat);
  base = apply_material_variation(base, hg.mat, hp);

  let voxel_size = cam.voxel_params.x;
  let hp_shadow  = hp + hg.n * (0.75 * voxel_size);

  let vis = sun_transmittance(hp_shadow, SUN_DIR);   // includes clouds + canopy

  let diff = max(dot(hg.n, SUN_DIR), 0.0);

  let amb_col = hemi_ambient(hg.n);
  let amb_strength = select(0.10, 0.14, hg.mat == MAT_LEAF); // tune
  let ambient = amb_col * amb_strength;

  var dapple = 1.0;
  if (hg.mat == MAT_LEAF) {
    let time_s = cam.voxel_params.y;
    let d0 = sin(dot(hp.xz, vec2<f32>(3.0, 2.2)) + time_s * 3.5);
    let d1 = sin(dot(hp.xz, vec2<f32>(6.5, 4.1)) - time_s * 6.0);
    dapple = 0.90 + 0.10 * (0.6 * d0 + 0.4 * d1);
  }

  let direct = SUN_COLOR * SUN_INTENSITY * (diff * diff) * vis * dapple;
  return base * (ambient + direct);
}
