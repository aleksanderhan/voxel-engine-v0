// src/shaders/ray/chunk_trace.wgsl
//// --------------------------------------------------------------------------
//// HitGeom + hybrid chunk tracing
//// --------------------------------------------------------------------------

struct HitGeom {
  hit      : u32,
  mat      : u32,
  _pad0    : u32,
  _pad1    : u32,

  t        : f32,
  _pad2    : vec3<f32>,

  n        : vec3<f32>,
  _pad3    : f32,

  root_bmin : vec3<f32>,
  root_size : f32,
  node_base : u32,
  _pad4     : vec3<u32>,
};

fn miss_hitgeom() -> HitGeom {
  var h : HitGeom;
  h.hit = 0u;
  h.mat = MAT_AIR;
  h.t   = BIG_F32;
  h.n   = vec3<f32>(0.0);
  h.root_bmin = vec3<f32>(0.0);
  h.root_size = 0.0;
  h.node_base = 0u;
  return h;
}

fn trace_chunk_hybrid_interval(
  ro: vec3<f32>,
  rd: vec3<f32>,
  ch: ChunkMeta,
  t_enter: f32,
  t_exit: f32
) -> HitGeom {
  let vs = cam.voxel_params.x;

  let root_bmin_vox = vec3<f32>(f32(ch.origin.x), f32(ch.origin.y), f32(ch.origin.z));
  let root_bmin = root_bmin_vox * vs;
  let root_size = f32(cam.chunk_size) * vs;

  let eps_step = 1e-4 * vs;
  var tcur = max(t_enter, 0.0) + eps_step;

  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));
  let chunk_size_i = i32(cam.chunk_size);

  for (var step_i: u32 = 0u; step_i < cam.max_steps; step_i = step_i + 1u) {
    if (tcur > t_exit) { break; }

    let p  = ro + tcur * rd;
    let pq = p + rd * (1e-4 * vs);

    let q = query_leaf_at(pq, root_bmin, root_size, ch.node_base);

    // ---- NEW: compute slab ONCE for this leaf, reuse for stepping/probing
    let slab    = cube_slab_inv(ro, inv, q.bmin, q.size);
    let t_leave = slab.t_exit;
    // ---------------------------------------------------------------

    // AIR: probe grass slab just below when in blade layer
    if (q.mat == MAT_AIR) {
      let layer_h = GRASS_LAYER_HEIGHT_VOX * vs;
      let bias = 0.05 * vs;

      var local = (p - root_bmin) - rd * bias;
      local = clamp(local, vec3<f32>(0.0), vec3<f32>(root_size - 1e-6));

      var ix = i32(floor(local.x / vs));
      var iz = i32(floor(local.z / vs));
      ix = clamp(ix, 0, chunk_size_i - 1);
      iz = clamp(iz, 0, chunk_size_i - 1);

      var iy = i32(floor((local.y - 1e-4 * vs) / vs)) - 1;
      iy = clamp(iy, 0, chunk_size_i - 1);

      let cell_bmin = root_bmin + vec3<f32>(f32(ix), f32(iy), f32(iz)) * vs;

      let y_top = cell_bmin.y + vs;
      if (p.y >= y_top - 1e-4 * vs && p.y <= y_top + layer_h + 1e-4 * vs) {
        let c = cell_bmin + vec3<f32>(0.5 * vs, 0.5 * vs, 0.5 * vs);
        if (is_grass(c, root_bmin, root_size, ch.node_base)) {
          let cell_id_vox = vec3<f32>(
            f32(ch.origin.x + ix),
            f32(ch.origin.y + iy),
            f32(ch.origin.z + iz)
          );

          // was: exit_time_from_cube_inv(...)
          let t_leave_air = t_leave;
          let tmax_probe  = min(min(t_leave_air, t_exit), tcur + q.size + 2.0 * vs);

          let time_s   = cam.voxel_params.y;
          let strength = cam.voxel_params.z;

          let gh = try_grass_slab_hit(ro, rd, tcur, tmax_probe, cell_bmin, cell_id_vox, vs, time_s, strength);
          if (gh.hit) {
            var out = miss_hitgeom();
            out.hit = 1u;
            out.t   = gh.t;
            out.mat = MAT_GRASS;
            out.n   = gh.n;
            out.root_bmin = root_bmin;
            out.root_size = root_size;
            out.node_base = ch.node_base;
            return out;
          }
        }
      }

      // was: exit_time_from_cube_inv(...)
      tcur = max(t_leave, tcur) + eps_step;
      continue;
    }

    // Leaves: displaced cube hit
    if (q.mat == MAT_LEAF) {
      let time_s   = cam.voxel_params.y;
      let strength = cam.voxel_params.z;

      let h2 = leaf_displaced_cube_hit(ro, rd, q.bmin, q.size, time_s, strength, t_enter, t_exit);

      if (h2.hit) {
        var out = miss_hitgeom();
        out.hit = 1u;
        out.t   = h2.t;
        out.mat = MAT_LEAF;
        out.n   = h2.n;
        out.root_bmin = root_bmin;
        out.root_size = root_size;
        out.node_base = ch.node_base;
        return out;
      }

      // was: exit_time_from_cube_inv(...)
      tcur = max(t_leave, tcur) + eps_step;
      continue;
    }

    // Solid: AABB face hit (use slab we already computed)
    let bh = cube_hit_normal_from_slab(rd, slab, t_enter, t_exit);

    if (bh.hit) {
      // If grass solid, try blades above the voxel under this hit (from above)
      if (q.mat == MAT_GRASS) {
        let time_s   = cam.voxel_params.y;
        let strength = cam.voxel_params.z;

        let hp = ro + bh.t * rd;

        let cell = pick_grass_cell_in_chunk(
          hp, rd,
          root_bmin,
          vec3<i32>(ch.origin.x, ch.origin.y, ch.origin.z),
          vs,
          chunk_size_i
        );

        let tmax_probe = min(bh.t, t_exit);
        let gh = try_grass_slab_hit(ro, rd, t_enter, tmax_probe, cell.bmin_m, cell.id_vox, vs, time_s, strength);
        if (gh.hit) {
          var out = miss_hitgeom();
          out.hit = 1u;
          out.t   = gh.t;
          out.mat = MAT_GRASS;
          out.n   = gh.n;
          out.root_bmin = root_bmin;
          out.root_size = root_size;
          out.node_base = ch.node_base;
          return out;
        }
      }

      var out = miss_hitgeom();
      out.hit = 1u;
      out.t   = bh.t;
      out.mat = q.mat;
      out.n   = bh.n;
      out.root_bmin = root_bmin;
      out.root_size = root_size;
      out.node_base = ch.node_base;
      return out;
    }

    // Miss: step using the already-computed slab exit
    tcur = max(t_leave, tcur) + eps_step;
  }

  return miss_hitgeom();
}

//// --------------------------------------------------------------------------
//// Scene voxel tracing over streamed chunk grid (for main_primary)
//// --------------------------------------------------------------------------

struct VoxTraceResult {
  in_grid : bool,
  best    : HitGeom,
  t_exit  : f32,
};

fn trace_scene_voxels(ro: vec3<f32>, rd: vec3<f32>) -> VoxTraceResult {
  // Fast out if no chunks
  if (cam.chunk_count == 0u) {
    return VoxTraceResult(false, miss_hitgeom(), 0.0);
  }

  let voxel_size   = cam.voxel_params.x;
  let chunk_size_m = f32(cam.chunk_size) * voxel_size;

  let go = cam.grid_origin_chunk;
  let gd = cam.grid_dims;

  let grid_bmin = vec3<f32>(f32(go.x), f32(go.y), f32(go.z)) * chunk_size_m;
  let grid_bmax = grid_bmin + vec3<f32>(f32(gd.x), f32(gd.y), f32(gd.z)) * chunk_size_m;

  let rtg = intersect_aabb(ro, rd, grid_bmin, grid_bmax);
  var t_enter = max(rtg.x, 0.0);
  let t_exit  = rtg.y;

  if (t_exit < t_enter) {
    return VoxTraceResult(false, miss_hitgeom(), 0.0);
  }

  let nudge_p = PRIMARY_NUDGE_VOXEL_FRAC * voxel_size;

  let start_t = t_enter + nudge_p;
  let p0 = ro + start_t * rd;

  var c = chunk_coord_from_pos(p0, chunk_size_m);
  var cx: i32 = c.x;
  var cy: i32 = c.y;
  var cz: i32 = c.z;

  var t_local: f32 = 0.0;

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

  var best = miss_hitgeom();
  let t_exit_local = max(t_exit - start_t, 0.0);

  let max_chunk_steps = min((gd.x + gd.y + gd.z) * 6u + 8u, 1024u);

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

    let tNextLocal = min(tMaxX, min(tMaxY, tMaxZ));
    if (best.hit != 0u && (start_t + tNextLocal) >= best.t) { break; }

    let slot = grid_lookup_slot(cx, cy, cz);
    if (slot != INVALID_U32 && slot < cam.chunk_count) {
      let ch = chunks[slot];

      let cell_enter = start_t + t_local;
      let cell_exit  = start_t + min(tNextLocal, t_exit_local);

      let h = trace_chunk_hybrid_interval(ro, rd, ch, cell_enter, cell_exit);
      if (h.hit != 0u && h.t < best.t) { best = h; }
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

  return VoxTraceResult(true, best, t_exit);
}
