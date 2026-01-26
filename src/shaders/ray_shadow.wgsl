// ray_shadow.wgsl
//
// Shadow ray traversal.

const SHADOW_STEPS : u32 = 32u;

// If false: leaves cast shadows using their undisplaced voxel cube (faster).
// If true: shadows match displaced leaf cubes (slower).
const SHADOW_DISPLACED_LEAVES : bool = true;

fn trace_chunk_shadow_interval(
  ro: vec3<f32>,
  rd: vec3<f32>,
  ch: ChunkMeta,
  t_enter: f32,
  t_exit: f32
) -> bool {
  let voxel_size = cam.voxel_params.x;

  let root_bmin_vox = vec3<f32>(f32(ch.origin.x), f32(ch.origin.y), f32(ch.origin.z));
  let root_bmin = root_bmin_vox * voxel_size;
  let root_size = f32(cam.chunk_size) * voxel_size;

  var tcur = max(t_enter, 0.0) + 1e-5;

  // Precompute inv once (3D)
  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  for (var step_i: u32 = 0u; step_i < SHADOW_STEPS; step_i = step_i + 1u) {
    if (tcur > t_exit) { break; }

    let p = ro + tcur * rd;
    let q = query_leaf_at(p, root_bmin, root_size, ch.node_base);

    if (q.mat != 0u) {
      // Leaves: optional fast shadows (undisplaced).
      if (q.mat == 5u) {
        if (!SHADOW_DISPLACED_LEAVES) {
          return true;
        }

        let time_s   = cam.voxel_params.y;
        let strength = cam.voxel_params.z;

        let h2 = leaf_displaced_cube_hit(
          ro, rd,
          q.bmin, q.size,
          time_s, strength,
          tcur - 1e-5,
          t_exit
        );

        if (h2.hit) { return true; }

        // Displaced leaf missed: skip.
        let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
        tcur = max(t_leave, tcur) + 1e-4;
        continue;
      }

      return true;
    }

    let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
    tcur = max(t_leave, tcur) + 1e-4;
  }

  return false;
}


fn in_shadow(p: vec3<f32>, sun_dir: vec3<f32>) -> bool {
  let voxel_size = cam.voxel_params.x;
  let chunk_size_m = f32(cam.chunk_size) * voxel_size;

  // Grid bounds in meters
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

  // Bias to reduce acne
  let ro = p + sun_dir * SHADOW_BIAS;
  let rd = sun_dir;

  // Intersect ray with grid AABB (so we only DDA inside)
  let rtg = intersect_aabb(ro, rd, grid_bmin, grid_bmax);
  var t_enter = max(rtg.x, 0.0);
  let t_exit  = rtg.y;
  if (t_exit < t_enter) { return false; }

  // Start inside interval
  let start_t = t_enter + 1e-4;
  let p0 = ro + start_t * rd;

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

  var tcur = start_t;

  // Similar bound as primary
  let max_chunk_steps = min((gd.x + gd.y + gd.z) * 6u + 8u, 1024u);

  for (var s: u32 = 0u; s < max_chunk_steps; s = s + 1u) {
    if (tcur > t_exit) { break; }

    let tNext = min(tMaxX, min(tMaxY, tMaxZ));
    let slot = grid_lookup_slot(cx, cy, cz);

    if (slot != INVALID_U32 && slot < cam.chunk_count) {
      let ch = chunks[slot];

      // Interval for THIS chunk cell
      let cell_enter = tcur;
      let cell_exit  = min(tNext, t_exit);

      if (trace_chunk_shadow_interval(ro, rd, ch, cell_enter, cell_exit)) {
        return true;
      }
    }

    // Step
    if (tMaxX < tMaxY) {
      if (tMaxX < tMaxZ) {
        cx = cx + step_x;
        tcur = tMaxX;
        tMaxX = tMaxX + tDeltaX;
      } else {
        cz = cz + step_z;
        tcur = tMaxZ;
        tMaxZ = tMaxZ + tDeltaZ;
      }
    } else {
      if (tMaxY < tMaxZ) {
        cy = cy + step_y;
        tcur = tMaxY;
        tMaxY = tMaxY + tDeltaY;
      } else {
        cz = cz + step_z;
        tcur = tMaxZ;
        tMaxZ = tMaxZ + tDeltaZ;
      }
    }

    // bounds check
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

// --- Fast shadow transmittance for volumetrics ---
// Leaves are semi-transparent, solids are opaque.
const VSM_STEPS : u32 = 16u;           // smaller than SHADOW_STEPS
const LEAF_LIGHT_TRANSMIT : f32 = 0.55; // 0..1, lower = darker canopy
const MIN_TRANS : f32 = 0.03;

fn trace_chunk_shadow_trans_interval(
  ro: vec3<f32>,
  rd: vec3<f32>,
  ch: ChunkMeta,
  t_enter: f32,
  t_exit: f32
) -> f32 {
  let voxel_size = cam.voxel_params.x;

  let root_bmin_vox = vec3<f32>(f32(ch.origin.x), f32(ch.origin.y), f32(ch.origin.z));
  let root_bmin = root_bmin_vox * voxel_size;
  let root_size = f32(cam.chunk_size) * voxel_size;

  var tcur = max(t_enter, 0.0) + 1e-5;
  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  var trans = 1.0;

  for (var step_i: u32 = 0u; step_i < VSM_STEPS; step_i = step_i + 1u) {
    if (tcur > t_exit) { break; }
    if (trans < MIN_TRANS) { break; }

    let p = ro + tcur * rd;
    let q = query_leaf_at(p, root_bmin, root_size, ch.node_base);

    if (q.mat != 0u) {
      if (q.mat == 5u) {
        // Treat leaf voxels as a thin participating occluder
        trans *= LEAF_LIGHT_TRANSMIT;

        // Skip forward through that leaf cube (cheap)
        let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
        tcur = max(t_leave, tcur) + 1e-4;
        continue;
      }

      // Solid material blocks completely
      return 0.0;
    }

    // Empty: skip
    let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
    tcur = max(t_leave, tcur) + 1e-4;
  }

  return trans;
}

fn sun_transmittance(p: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
  let voxel_size = cam.voxel_params.x;
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

  let ro = p + sun_dir * SHADOW_BIAS;
  let rd = sun_dir;

  let rtg = intersect_aabb(ro, rd, grid_bmin, grid_bmax);
  var t_enter = max(rtg.x, 0.0);
  let t_exit  = rtg.y;
  if (t_exit < t_enter) { return 1.0; }

  let start_t = t_enter + 1e-4;
  let p0 = ro + start_t * rd;

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

  var tcur = start_t;
  var trans = 1.0;

  let max_chunk_steps = min((gd.x + gd.y + gd.z) * 6u + 8u, 512u);

  for (var s: u32 = 0u; s < max_chunk_steps; s = s + 1u) {
    if (tcur > t_exit) { break; }
    if (trans < MIN_TRANS) { break; }

    let tNext = min(tMaxX, min(tMaxY, tMaxZ));
    let slot = grid_lookup_slot(cx, cy, cz);

    if (slot != INVALID_U32 && slot < cam.chunk_count) {
      let ch = chunks[slot];
      trans *= trace_chunk_shadow_trans_interval(ro, rd, ch, tcur, min(tNext, t_exit));
      if (trans < MIN_TRANS) { break; }
    }

    // Step DDA
    if (tMaxX < tMaxY) {
      if (tMaxX < tMaxZ) { cx += step_x; tcur = tMaxX; tMaxX += tDeltaX; }
      else               { cz += step_z; tcur = tMaxZ; tMaxZ += tDeltaZ; }
    } else {
      if (tMaxY < tMaxZ) { cy += step_y; tcur = tMaxY; tMaxY += tDeltaY; }
      else               { cz += step_z; tcur = tMaxZ; tMaxZ += tDeltaZ; }
    }

    // bounds check (same as yours)
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

  return trans;
}
