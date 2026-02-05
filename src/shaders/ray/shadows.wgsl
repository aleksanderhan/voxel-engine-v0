// src/shaders/ray/shadows.wgsl
// ----------------------------
// Sun transmittance (geometry-only + full with clouds)
//
// FIX: MAT_LIGHT voxels must NOT act like “black occluders” for sun/sky visibility.
// If they do, placing a lamp can *reduce* sky_visibility() (used to gate ambient),
// making nearby cave surfaces darker even though the lamp adds local light.
//
// So: treat MAT_LIGHT as transparent (like MAT_AIR) in shadow rays.

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
    let pq = p + rd * (0.1 * voxel_size); // try 0.01..0.05


    let q = query_leaf_at(pq, root_bmin, root_size, ch.node_base, ch.macro_base);

    // slab once (for stepping)
    let slab    = cube_slab_inv(ro, rd, inv, q.bmin, q.size);
    let t_leave = slab.t_exit;

    if (q.mat != MAT_AIR) {
      // --- IMPORTANT FIX: lights do not occlude the sun/sky for visibility ---
      if (q.mat == MAT_LIGHT) {
        tcur = max(t_leave, tcur) + nudge_s;
        continue;
      }
      // ----------------------------------------------------------------------

      if (q.mat == MAT_LEAF) {
        if (VOLUME_DISPLACED_LEAVES) {
          // distance gate for displaced-leaf shadow test
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

      // Any other solid fully blocks
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

  var c = chunk_coord_from_pos_dir(p0, rd, chunk_size_m);
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

  // hoisted grid bounds for loop bounds check
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

  for (var s: u32 = 0u; s < max_chunk_steps; s = s + 1u) {
    if (t_local > t_exit_local) { break; }
    if (trans < MIN_TRANS) { break; }

    let tNextLocal = min(tMaxX, min(tMaxY, tMaxZ));
    let slot = grid_lookup_slot(cx, cy, cz);

    if (slot != INVALID_U32 && slot < cam.chunk_count) {
      let ch2 = chunks[slot];

      let cell_enter = start_t + t_local;
      let cell_exit2 = start_t + min(tNextLocal, t_exit_local);

      // This function now treats MAT_LIGHT as transparent too.
      trans *= trace_chunk_shadow_trans_interval(ro, rd, ch2, cell_enter, cell_exit2);
      if (trans < MIN_TRANS) { break; }
    }

    // advance DDA
    if (tMaxX < tMaxY) {
      if (tMaxX < tMaxZ) { cx += step_x; t_local = tMaxX; tMaxX += tDeltaX; }
      else               { cz += step_z; t_local = tMaxZ; tMaxZ += tDeltaZ; }
    } else {
      if (tMaxY < tMaxZ) { cy += step_y; t_local = tMaxY; tMaxY += tDeltaY; }
      else               { cz += step_z; t_local = tMaxZ; tMaxZ += tDeltaZ; }
    }

    // bounds check
    if (cx < gx0 || cy < gy0 || cz < gz0 || cx >= gx1 || cy >= gy1 || cz >= gz1) { break; }
  }

  return trans;
}

fn sun_transmittance(p: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
  let Tc = cloud_sun_transmittance(p, sun_dir);
  return Tc * sun_transmittance_geom_only(p, sun_dir);
}

// -----------------------------------------------------------------------------
// Soft sky visibility (fixes hard cave ambient cutoffs)
// -----------------------------------------------------------------------------

// Keep low to avoid perf spikes. 3–5 is usually enough.
const SKYVIS_SAMPLES : u32 = 4u;

// Cone angle around +Y in radians (0.12..0.30 typical)
const SKYVIS_CONE_ANGLE : f32 = 0.22;

// Optional shaping to reduce “binary” feel
const SKYVIS_CURVE_POW : f32 = 0.70;

// Build a stable TBN around a given "up" direction.
fn tbn_from_dir(n: vec3<f32>) -> mat3x3<f32> {
  let up_ref = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
  let t = normalize(cross(up_ref, n));
  let b = normalize(cross(n, t));
  return mat3x3<f32>(t, b, n);
}

// Uniform cone sampling around +Z in local space, then rotate into world via TBN.
// u in [0,1), v in [0,1)
fn sample_cone_dir_local(u: f32, v: f32, cone_angle: f32) -> vec3<f32> {
  // cos(theta) in [cos(cone), 1]
  let cos_min = cos(cone_angle);
  let cos_t   = mix(cos_min, 1.0, u);
  let sin_t   = sqrt(max(0.0, 1.0 - cos_t * cos_t));
  let phi     = 6.28318530718 * v;

  // cone points along +Z in local
  return vec3<f32>(cos(phi) * sin_t, sin(phi) * sin_t, cos_t);
}

// Stable per-point random (don’t use frame_index; avoid shimmer).
fn skyvis_seed01(p: vec3<f32>) -> vec2<f32> {
  // Quantize in voxel space for stability
  let vs = cam.voxel_params.x;
  let q  = floor(p / max(vs, 1e-6));
  let a  = hash31(q + vec3<f32>(11.3, 7.1, 3.7));
  let b  = hash31(q + vec3<f32>(5.9,  2.2, 9.4));
  return vec2<f32>(a, b);
}

// Soft sky visibility: average a few geometry-only transmittance rays in a cone around +Y.
fn sky_visibility_soft_geom(p: vec3<f32>) -> f32 {
  // Bias up a hair (same intent as your old sky_visibility())
  let vs = cam.voxel_params.x;
  let pu = p + vec3<f32>(0.0, 1.0, 0.0) * (0.75 * vs);

  let up = vec3<f32>(0.0, 1.0, 0.0);
  let tbn = tbn_from_dir(up);

  let s = skyvis_seed01(pu);

  var sum: f32 = 0.0;

  // Simple stratified pattern (stable)
  for (var i: u32 = 0u; i < SKYVIS_SAMPLES; i = i + 1u) {
    let fi = f32(i);

    // Two stable pseudo-randoms per sample
    let u = fract(s.x + (fi + 1.0) * 0.381966); // golden-ish
    let v = fract(s.y + (fi + 1.0) * 0.618034);

    // Sample cone around local +Z, map local +Z -> world +Y using TBN columns:
    // Our tbn_from_dir makes n be the 3rd column, so local +Z maps to world "up".
    let dl = sample_cone_dir_local(u, v, SKYVIS_CONE_ANGLE);
    let dir = normalize(tbn * dl);

    sum += sun_transmittance_geom_only(pu, dir);
  }

  let avg = sum / max(1.0, f32(SKYVIS_SAMPLES));

  // Gentle curve: makes mid-values more common (reduces “either 0 or 1” feel).
  return pow(clamp(avg, 0.0, 1.0), SKYVIS_CURVE_POW);
}
