//// --------------------------------------------------------------------------
//// Sun transmittance (geometry-only + full with clouds)
//// --------------------------------------------------------------------------
fn trace_chunk_shadow_trans_interval(
  ro: vec3<f32>,
  rd: vec3<f32>,
  ch: ChunkMeta,
  t_enter: f32,
  t_exit: f32
) -> f32 {
  let voxel_size = cam.voxel_params.x;
  let nudge_s = 0.20 * voxel_size;

  let root_bmin_vox = vec3<f32>(f32(ch.origin.x), f32(ch.origin.y), f32(ch.origin.z));
  let root_bmin = root_bmin_vox * voxel_size;
  let root_size = f32(cam.chunk_size) * voxel_size;

  var tcur = max(t_enter, 0.0) + nudge_s;
  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  var trans = 1.0;

  for (var step_i: u32 = 0u; step_i < VSM_STEPS; step_i = step_i + 1u) {
    if (tcur > t_exit) { break; }
    if (trans < MIN_TRANS) { break; }

    let p  = ro + tcur * rd;
    let pq = p + rd * (1e-4 * cam.voxel_params.x);

    let q = query_leaf_at(pq, root_bmin, root_size, ch.node_base, ch.macro_base);

    // slab once (for stepping)
    let slab    = cube_slab_inv(ro, inv, q.bmin, q.size);
    let t_leave = slab.t_exit;

    if (q.mat != MAT_AIR) {
      if (q.mat == MAT_LEAF) {
        if (VOLUME_DISPLACED_LEAVES) {
          // --- NEW: distance gate for displaced-leaf shadow test ---
          // If far, skip expensive displaced intersection and just apply transmit.
          let center = q.bmin + vec3<f32>(0.5 * q.size);
          let d = length(center - cam.cam_pos.xyz);

          if (d < LEAF_LOD_DISP_END) {
            let time_s   = cam.voxel_params.y;
            let strength = cam.voxel_params.z;

            let h2 = leaf_displaced_cube_hit(
              ro, rd,
              q.bmin, q.size,
              time_s, strength,
              tcur - nudge_s, t_exit
            );
            if (h2.hit) { trans *= LEAF_LIGHT_TRANSMIT; }
          } else {
            trans *= LEAF_LIGHT_TRANSMIT;
          }
          // --------------------------------------------------------

          tcur = max(t_leave, tcur) + nudge_s;
          continue;
        }

        trans *= LEAF_LIGHT_TRANSMIT;
        tcur = max(t_leave, tcur) + nudge_s;
        continue;
      }

      if (q.mat == MAT_GRASS) {
        trans *= GRASS_LIGHT_TRANSMIT;
        tcur = max(t_leave, tcur) + nudge_s;
        continue;
      }

      return 0.0;
    }

    tcur = max(t_leave, tcur) + nudge_s;
  }

  return trans;
}


fn sun_transmittance_geom_only(p: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
  let voxel_size   = cam.voxel_params.x;
  let nudge_s      = 0.18 * voxel_size;
  let chunk_size_m = f32(cam.chunk_size) * voxel_size;

  let go = cam.grid_origin_chunk;
  let gd = cam.grid_dims;

  let grid_bmin = vec3<f32>(f32(go.x), f32(go.y), f32(go.z)) * chunk_size_m;
  let grid_bmax = grid_bmin + vec3<f32>(f32(gd.x), f32(gd.y), f32(gd.z)) * chunk_size_m;

  let bias = max(SHADOW_BIAS, 0.50 * voxel_size);
  let ro   = p + sun_dir * bias;
  let rd   = sun_dir;

  let rtg = intersect_aabb(ro, rd, grid_bmin, grid_bmax);
  let t_enter = max(rtg.x, 0.0);
  let t_exit  = rtg.y;
  if (t_exit < t_enter) { return 1.0; }

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

  // ---- HOISTED: grid bounds for the DDA loop (was recomputed each step)
  let ox: i32 = go.x;
  let oy: i32 = go.y;
  let oz: i32 = go.z;
  let nx: i32 = i32(gd.x);
  let ny: i32 = i32(gd.y);
  let nz: i32 = i32(gd.z);
  let gx0: i32 = ox;
  let gy0: i32 = oy;
  let gz0: i32 = oz;
  let gx1: i32 = ox + nx;
  let gy1: i32 = oy + ny;
  let gz1: i32 = oz + nz;
  // ---------------------------------------------------------------

  for (var s: u32 = 0u; s < max_chunk_steps; s = s + 1u) {
    if (t_local > t_exit_local) { break; }
    if (trans < MIN_TRANS) { break; }

    let tNextLocal = min(tMaxX, min(tMaxY, tMaxZ));
    let slot = grid_lookup_slot(cx, cy, cz);

    if (slot != INVALID_U32 && slot < cam.chunk_count) {
      let ch2 = chunks[slot];

      let cell_enter = start_t + t_local;
      let cell_exit2 = start_t + min(tNextLocal, t_exit_local);

      trans *= trace_chunk_shadow_trans_interval(ro, rd, ch2, cell_enter, cell_exit2);
      if (trans < MIN_TRANS) { break; }
    }

    if (tMaxX < tMaxY) {
      if (tMaxX < tMaxZ) { cx += step_x; t_local = tMaxX; tMaxX += tDeltaX; }
      else               { cz += step_z; t_local = tMaxZ; tMaxZ += tDeltaZ; }
    } else {
      if (tMaxY < tMaxZ) { cy += step_y; t_local = tMaxY; tMaxY += tDeltaY; }
      else               { cz += step_z; t_local = tMaxZ; tMaxZ += tDeltaZ; }
    }

    // bounds check (now uses hoisted constants)
    if (cx < gx0 || cy < gy0 || cz < gz0 || cx >= gx1 || cy >= gy1 || cz >= gz1) { break; }
  }

  return trans;
}

fn sun_transmittance(p: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
  let Tc = cloud_sun_transmittance(p, sun_dir);
  return Tc * sun_transmittance_geom_only(p, sun_dir);
}
