// ray_shadow.wgsl
//
// Shadow ray traversal for two purposes:
//
// 1) Hard shadows for surface shading (boolean occlusion):
//    - in_shadow(): casts a ray toward the sun and returns true if *any* opaque
//      voxel blocks the ray before it exits the streamed grid.
//
// 2) Sun transmittance for volumetrics (continuous attenuation):
//    - sun_transmittance(): casts the same kind of ray, but returns a scalar in [0..1]
//      representing how much sunlight reaches a point through semi-occluding leaves.
//      Solids are fully opaque (0), leaves reduce transmittance multiplicatively.
//
// Traversal approach (both cases):
// - Intersect the ray with the streamed grid AABB (world-space bounds of chunk grid).
// - Traverse chunk-cells with a 3D DDA in *chunk space* (like main_primary).
// - For each visited chunk cell, run a short inner loop that steps through SVO leaf cells
//   by point-querying the octree and skipping to the leaf cube exit time.
//   This keeps the shadow test fast without fully marching voxels.
//
// Dependencies (from other WGSL files):
// - common.wgsl: cam, chunks, chunk_grid, constants, intersect_aabb(), etc.
// - ray_core.wgsl: safe_inv(), query_leaf_at(), exit_time_from_cube_inv(), aabb_hit_normal_inv().
// - ray_leaf_wind.wgsl: leaf_displaced_cube_hit() (optional leaf displacement in shadows).
// - grid_lookup_slot(), chunk_coord_from_pos(), INVALID_U32 are defined in ray_main.wgsl in your setup.
//
// Coordinate / units:
// - All t values are in world meters along the ray (same as primary).
// - voxel_size = cam.voxel_params.x.

const SHADOW_STEPS : u32 = 32u; // Max leaf-cube "skips" per chunk interval (hard shadows).

// If false: leaves cast shadows using their *undisplaced* voxel cube (faster, less accurate).
// If true : shadows match displaced leaf cubes (slower, but consistent with animated leaf hits).
const SHADOW_DISPLACED_LEAVES : bool = true;

// -----------------------------------------------------------------------------
// Per-chunk shadow query (boolean occlusion)
// -----------------------------------------------------------------------------
//
// Returns true if any non-empty leaf cell blocks the ray within [t_enter, t_exit] for this chunk.
//
// Stepping method:
// - Maintain a current t (tcur).
// - Sample a point p on the ray, query which leaf cube contains it (query_leaf_at).
// - If empty: skip to the exit time of that cube.
// - If non-empty: return true (occluded), with special handling for leaves if enabled.

fn trace_chunk_shadow_interval(
  ro: vec3<f32>,
  rd: vec3<f32>,
  ch: ChunkMeta,
  t_enter: f32,
  t_exit: f32
) -> bool {
  let voxel_size = cam.voxel_params.x;

  // Small step used to move forward after we skip out of a cell.
  // Larger than the primary tracer's epsilon because shadows tolerate a bit more bias.
  let nudge_s = 0.18 * voxel_size;

  // Chunk root bounds in world meters.
  let root_bmin_vox = vec3<f32>(f32(ch.origin.x), f32(ch.origin.y), f32(ch.origin.z));
  let root_bmin = root_bmin_vox * voxel_size;
  let root_size = f32(cam.chunk_size) * voxel_size;

  // Start just inside the trusted interval.
  var tcur = max(t_enter, 0.0) + nudge_s;

  // Precompute inv direction once (used by exit_time_from_cube_inv).
  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  for (var step_i: u32 = 0u; step_i < SHADOW_STEPS; step_i = step_i + 1u) {
    if (tcur > t_exit) { break; }

    // Query which SVO leaf cell we're in.
    let p = ro + tcur * rd;
    let q = query_leaf_at(p, root_bmin, root_size, ch.node_base);

    if (q.mat != 0u) {
      // Leaf material: optionally test displaced cube for exact matching shadows.
      if (q.mat == 5u) {
        // Fast mode: treat the undisplaced leaf voxel as a blocker.
        if (!SHADOW_DISPLACED_LEAVES) {
          return true;
        }

        // Accurate mode: intersect the displaced leaf cube.
        let time_s   = cam.voxel_params.y;
        let strength = cam.voxel_params.z;

        // Use a slightly wider interval start (tcur - nudge) so we don't miss due to nudging.
        let h2 = leaf_displaced_cube_hit(
          ro, rd,
          q.bmin, q.size,
          time_s, strength,
          tcur - nudge_s,
          t_exit
        );

        // If displaced cube hits, the sun is blocked.
        if (h2.hit) { return true; }

        // Displaced cube missed inside this (undisplaced) leaf cell:
        // treat it as empty and skip out of the underlying cell cube.
        let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
        tcur = max(t_leave, tcur) + nudge_s;
        continue;
      }

      // Any other non-empty material is treated as fully opaque to sunlight.
      return true;
    }

    // Empty cell: skip to its exit and continue.
    let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
    tcur = max(t_leave, tcur) + nudge_s;
  }

  // No blocker found in this chunk interval.
  return false;
}

// -----------------------------------------------------------------------------
// High-level hard shadow query (boolean) from point -> sun
// -----------------------------------------------------------------------------
//
// Casts a ray from point p toward sun_dir, within the streamed grid volume.
// Returns true if anything blocks the sun (i.e. the point is in shadow).

fn in_shadow(p: vec3<f32>, sun_dir: vec3<f32>) -> bool {
  let voxel_size   = cam.voxel_params.x;
  let nudge_s      = 0.18 * voxel_size;
  let chunk_size_m = f32(cam.chunk_size) * voxel_size;

  // Compute world-space bounds of the streamed chunk grid (AABB).
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

  // Bias the shadow ray start along sun_dir to reduce self-shadowing artifacts.
  // Combines a fixed bias constant with a voxel-scale bias.
  let bias = max(SHADOW_BIAS, 0.50 * voxel_size); // tune: ~0.25..1.0 voxels
  let ro   = p + sun_dir * bias;
  let rd   = sun_dir;

  // Ray vs grid AABB: if we never enter the grid, nothing can occlude => not in shadow.
  let rtg = intersect_aabb(ro, rd, grid_bmin, grid_bmax);
  let t_enter = max(rtg.x, 0.0);
  let t_exit  = rtg.y;
  if (t_exit < t_enter) { return false; }

  // Start slightly inside to avoid boundary tie issues.
  let start_t = t_enter + nudge_s;
  let p0 = ro + start_t * rd;

  // DDA traversal in *chunk cells*, using LOCAL t (t=0 at p0).
  var t_local: f32 = 0.0;
  let t_exit_local = max(t_exit - start_t, 0.0);

  // Initial chunk coordinate from p0.
  var c = chunk_coord_from_pos(p0, chunk_size_m);
  var cx: i32 = c.x;
  var cy: i32 = c.y;
  var cz: i32 = c.z;

  // Precompute inverse direction and per-axis stepping.
  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  let step_x: i32 = select(-1, 1, rd.x > 0.0);
  let step_y: i32 = select(-1, 1, rd.y > 0.0);
  let step_z: i32 = select(-1, 1, rd.z > 0.0);

  // Next chunk boundary planes in world space (meters).
  let bx = select(f32(cx) * chunk_size_m, f32(cx + 1) * chunk_size_m, rd.x > 0.0);
  let by = select(f32(cy) * chunk_size_m, f32(cy + 1) * chunk_size_m, rd.y > 0.0);
  let bz = select(f32(cz) * chunk_size_m, f32(cz + 1) * chunk_size_m, rd.z > 0.0);

  // Parametric t to those planes (LOCAL from p0).
  var tMaxX: f32 = (bx - p0.x) * inv.x;
  var tMaxY: f32 = (by - p0.y) * inv.y;
  var tMaxZ: f32 = (bz - p0.z) * inv.z;

  // Delta t per chunk cell crossing.
  let tDeltaX: f32 = abs(chunk_size_m * inv.x);
  let tDeltaY: f32 = abs(chunk_size_m * inv.y);
  let tDeltaZ: f32 = abs(chunk_size_m * inv.z);

  // If rd axis is near zero, never step that axis.
  if (abs(rd.x) < EPS_INV) { tMaxX = BIG_F32; }
  if (abs(rd.y) < EPS_INV) { tMaxY = BIG_F32; }
  if (abs(rd.z) < EPS_INV) { tMaxZ = BIG_F32; }

  // Cap chunk stepping to avoid pathological loops.
  let max_chunk_steps = min((gd.x + gd.y + gd.z) * 6u + 8u, 1024u);

  for (var s: u32 = 0u; s < max_chunk_steps; s = s + 1u) {
    if (t_local > t_exit_local) { break; }

    // LOCAL end of current chunk interval.
    let tNextLocal = min(tMaxX, min(tMaxY, tMaxZ));

    // If this chunk cell maps to a resident chunk slot, test occlusion inside it.
    let slot = grid_lookup_slot(cx, cy, cz);
    if (slot != INVALID_U32 && slot < cam.chunk_count) {
      let ch = chunks[slot];

      // Convert LOCAL -> ABSOLUTE t for the inner chunk traversal.
      let cell_enter = start_t + t_local;
      let cell_exit  = start_t + min(tNextLocal, t_exit_local);

      if (trace_chunk_shadow_interval(ro, rd, ch, cell_enter, cell_exit)) {
        return true; // blocked: in shadow
      }
    }

    // Advance DDA to next chunk cell.
    if (tMaxX < tMaxY) {
      if (tMaxX < tMaxZ) { cx += step_x; t_local = tMaxX; tMaxX += tDeltaX; }
      else               { cz += step_z; t_local = tMaxZ; tMaxZ += tDeltaZ; }
    } else {
      if (tMaxY < tMaxZ) { cy += step_y; t_local = tMaxY; tMaxY += tDeltaY; }
      else               { cz += step_z; t_local = tMaxZ; tMaxZ += tDeltaZ; }
    }

    // Stop if we leave the streamed grid bounds.
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

  // Traversed the grid without hitting an occluder.
  return false;
}

// -----------------------------------------------------------------------------
// Sun transmittance (scalar) for volumetrics
// -----------------------------------------------------------------------------
//
// Similar to in_shadow(), but instead of boolean occlusion it returns a continuous
// attenuation factor in [0..1] used for volumetric scattering.
//
// - Opaque solids => 0.0
// - Leaf voxels   => multiply by LEAF_LIGHT_TRANSMIT per encountered leaf cell
// - Early exit when trans < MIN_TRANS for speed

fn sun_transmittance(p: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
  let voxel_size   = cam.voxel_params.x;
  let nudge_s      = 0.18 * voxel_size;
  let chunk_size_m = f32(cam.chunk_size) * voxel_size;

  // Streamed grid bounds.
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

  // Bias start to reduce self-occlusion.
  let bias = max(SHADOW_BIAS, 0.50 * voxel_size);
  let ro   = p + sun_dir * bias;
  let rd   = sun_dir;

  // If we don't intersect the streamed grid, sunlight is unoccluded.
  let rtg = intersect_aabb(ro, rd, grid_bmin, grid_bmax);
  let t_enter = max(rtg.x, 0.0);
  let t_exit  = rtg.y;
  if (t_exit < t_enter) { return 1.0; }

  // Start just inside.
  let start_t = t_enter + nudge_s;
  let p0 = ro + start_t * rd;

  // DDA LOCAL from p0.
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

  // Running multiplicative transmittance.
  var trans = 1.0;

  // Lower cap than in_shadow() because this is called a lot for volumetrics.
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

      // Multiply by per-chunk transmittance over this interval.
      trans *= trace_chunk_shadow_trans_interval(ro, rd, ch, cell_enter, cell_exit);
      if (trans < MIN_TRANS) { break; }
    }

    // Step DDA.
    if (tMaxX < tMaxY) {
      if (tMaxX < tMaxZ) { cx += step_x; t_local = tMaxX; tMaxX += tDeltaX; }
      else               { cz += step_z; t_local = tMaxZ; tMaxZ += tDeltaZ; }
    } else {
      if (tMaxY < tMaxZ) { cy += step_y; t_local = tMaxY; tMaxY += tDeltaY; }
      else               { cz += step_z; t_local = tMaxZ; tMaxZ += tDeltaZ; }
    }

    // Stop if outside the streamed grid.
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

// -----------------------------------------------------------------------------
// Fast shadow transmittance for volumetrics (leafy canopy)
// -----------------------------------------------------------------------------
//
// This is a cheaper "participating occluder" approximation used by sun_transmittance().
//
// Rules:
// - Solids block completely: return 0.
// - Leaf voxels reduce transmittance multiplicatively: trans *= LEAF_LIGHT_TRANSMIT.
// - Empty space skips quickly using exit_time_from_cube_inv.
// - Uses fewer steps than hard shadows (VSM_STEPS < SHADOW_STEPS).

const VSM_STEPS : u32 = 24u;
const LEAF_LIGHT_TRANSMIT : f32 = 0.50; // per-leaf attenuation (0..1). Lower => darker canopy.
const MIN_TRANS : f32 = 0.03;           // early-out threshold

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

    // Query point slightly forward to reduce split-plane ambiguity.
    let p = ro + tcur * rd;
    let qeps = 1e-4 * cam.voxel_params.x;
    let pq   = p + rd * qeps;

    let q = query_leaf_at(pq, root_bmin, root_size, ch.node_base);

    if (q.mat != 0u) {
      if (q.mat == 5u) {
        // Leaf voxel: attenuate but do not fully block.
        trans *= LEAF_LIGHT_TRANSMIT;

        // Skip out of the leaf cube quickly.
        let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
        tcur = max(t_leave, tcur) + nudge_s;
        continue;
      }

      // Any other solid: fully opaque.
      return 0.0;
    }

    // Empty: skip.
    let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
    tcur = max(t_leave, tcur) + nudge_s;
  }

  return trans;
}
