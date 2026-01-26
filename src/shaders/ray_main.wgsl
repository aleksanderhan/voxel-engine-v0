// ray_main.wgsl
//
// Entry point: per-pixel ray, traverse chunk grid with 3D DDA,
// only tracing chunks the ray actually passes through.
//
// This version assumes you added these helpers/constants in common.wgsl:
//   - sky_color(rd)
//   - fog_transmittance(ro, rd, t)
//   - phase_mie(costh)
//   - sun_transmittance(p, SUN_DIR)   (from ray_shadow.wgsl)
//   - fog_density_base()
//   - FOG_MAX_DIST, GODRAY_MAX_DIST, GODRAY_STEPS, GODRAY_STRENGTH
//
// And that the following are already available (as in your current codebase):
//   - safe_inv, EPS_INV, BIG_F32
//   - intersect_aabb, ray_dir_from_pixel
//   - trace_chunk_hybrid_interval, HitGeom
//   - shade_hit
//   - SUN_DIR, SUN_COLOR, SUN_INTENSITY
//   - hash12 (or replace jitter with 0.0)

const INVALID_U32 : u32 = 0xFFFFFFFFu;

// Convert a chunk coordinate (cx,cy,cz) to a slot index into `chunks[]`,
// or INVALID_U32 if outside the grid or not loaded.
fn grid_lookup_slot(cx: i32, cy: i32, cz: i32) -> u32 {
  let ox = cam.grid_origin_chunk.x;
  let oy = cam.grid_origin_chunk.y;
  let oz = cam.grid_origin_chunk.z;

  let ix_i = cx - ox;
  let iy_i = cy - oy;
  let iz_i = cz - oz;

  if (ix_i < 0 || iy_i < 0 || iz_i < 0) { return INVALID_U32; }

  let nx = cam.grid_dims.x;
  let ny = cam.grid_dims.y;
  let nz = cam.grid_dims.z;

  let ix = u32(ix_i);
  let iy = u32(iy_i);
  let iz = u32(iz_i);

  if (ix >= nx || iy >= ny || iz >= nz) { return INVALID_U32; }

  let idx = (iz * ny * nx) + (iy * nx) + ix;
  return chunk_grid[idx];
}

// Floor division by chunk size in meters, producing chunk coordinates.
fn chunk_coord_from_pos(p: vec3<f32>, chunk_size_m: f32) -> vec3<i32> {
  return vec3<i32>(
    i32(floor(p.x / chunk_size_m)),
    i32(floor(p.y / chunk_size_m)),
    i32(floor(p.z / chunk_size_m))
  );
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(out_img);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let res = vec2<f32>(f32(dims.x), f32(dims.y));
  let px  = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

  let ro = cam.cam_pos.xyz;
  let rd = ray_dir_from_pixel(px, res);

  // Sky (with sun disc)
  let sky = sky_color(rd);

  // If no chunks, just fog the sky a bit (optional) and return.
  if (cam.chunk_count == 0u) {
    textureStore(out_img, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(sky, 1.0));
    return;
  }

  let voxel_size   = cam.voxel_params.x;
  let chunk_size_m = f32(cam.chunk_size) * voxel_size;

  // Grid AABB in meters
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

  // Intersect ray with the grid bounds so DDA stays in-range.
  let rtg    = intersect_aabb(ro, rd, grid_bmin, grid_bmax);
  var t_enter = max(rtg.x, 0.0);
  let t_exit  = rtg.y;

  // If ray misses the grid region -> sky
  if (t_exit < t_enter) {
    textureStore(out_img, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(sky, 1.0));
    return;
  }

  // Start point inside the grid region (nudge a bit to avoid boundary ambiguity)
  let start_t = t_enter + 1e-4;
  let p0 = ro + start_t * rd;

  var c = chunk_coord_from_pos(p0, chunk_size_m);
  var cx: i32 = c.x;
  var cy: i32 = c.y;
  var cz: i32 = c.z;

  // DDA stepping setup
  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  let step_x: i32 = select(-1, 1, rd.x > 0.0);
  let step_y: i32 = select(-1, 1, rd.y > 0.0);
  let step_z: i32 = select(-1, 1, rd.z > 0.0);

  // Next boundary in meters for each axis
  let bx = select(f32(cx) * chunk_size_m, f32(cx + 1) * chunk_size_m, rd.x > 0.0);
  let by = select(f32(cy) * chunk_size_m, f32(cy + 1) * chunk_size_m, rd.y > 0.0);
  let bz = select(f32(cz) * chunk_size_m, f32(cz + 1) * chunk_size_m, rd.z > 0.0);

  // Parametric t to those boundaries
  var tMaxX: f32 = (bx - p0.x) * inv.x;
  var tMaxY: f32 = (by - p0.y) * inv.y;
  var tMaxZ: f32 = (bz - p0.z) * inv.z;

  // How far in t to cross one full chunk along each axis
  let tDeltaX: f32 = abs(chunk_size_m * inv.x);
  let tDeltaY: f32 = abs(chunk_size_m * inv.y);
  let tDeltaZ: f32 = abs(chunk_size_m * inv.z);

  // If direction component is ~0, disable stepping on that axis
  if (abs(rd.x) < EPS_INV) { tMaxX = BIG_F32; }
  if (abs(rd.y) < EPS_INV) { tMaxY = BIG_F32; }
  if (abs(rd.z) < EPS_INV) { tMaxZ = BIG_F32; }

  // Best hit across visited chunks
  var best = HitGeom(false, BIG_F32, 0u, vec3<f32>(0.0));

  // Conservative upper bound on how many chunk cells we might traverse in the grid.
  let max_chunk_steps = min((gd.x + gd.y + gd.z) * 6u + 8u, 1024u);

  // DDA loop through chunk cells
  var tcur = start_t;

  for (var s: u32 = 0u; s < max_chunk_steps; s = s + 1u) {
    if (tcur > t_exit) { break; }

    let tNext = min(tMaxX, min(tMaxY, tMaxZ));
    if (best.hit && tNext >= best.t) { break; }

    let slot = grid_lookup_slot(cx, cy, cz);
    if (slot != INVALID_U32 && slot < cam.chunk_count) {
      let ch = chunks[slot];

      let cell_enter = tcur;
      let cell_exit  = min(tNext, t_exit);

      let h = trace_chunk_hybrid_interval(ro, rd, ch, cell_enter, cell_exit);
      if (h.hit && h.t < best.t) { best = h; }
    }

    // Step to next cell
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

    // Quick bounds check
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

  // Surface / background
  let surface = select(sky, shade_hit(ro, rd, best), best.hit);

  // Apply fog up to hit distance (or up to grid exit if no hit)
  let t_end = select(min(t_exit, FOG_MAX_DIST), min(best.t, FOG_MAX_DIST), best.hit);

  let T = fog_transmittance(ro, rd, t_end);

  // Use sky as the fog color for consistency (cheap “aerial perspective”)
  let fog_col = sky;

  // Godrays (single scattering)
  let ins = godray_inscatter(ro, rd, t_end, px);

  let col = surface * T + fog_col * (1.0 - T) + ins;

  textureStore(out_img, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(col, 1.0));
}
