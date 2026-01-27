// ray_main.wgsl
//
// 3-pass compute shader module.
// Entry points (all @workgroup_size(8,8,1)):
//
//   1) main_primary   (full-res)
//      - traces the world (SVO traversal + shading)
//      - applies exponential height fog (transmittance)
//      - writes:
//          color_img : RGBA16F storage (HDR-ish color)
//          depth_img : R32F   storage (scene distance t in meters)
//
//   2) main_godray    (quarter-res)
//      - approximates volumetric light shafts by integrating fog scattering
//        along a subset of rays.
//      - Uses 4 taps per 4x4 full-res block (rotated pattern) to reduce cost
//        and hide grid artifacts.
//      - writes:
//          godray_out : RGBA16F storage (quarter-res)
//
//   3) main_composite (full-res)
//      - reads primary color + quarter-res godrays
//      - upsamples godrays (manual bilerp) and lightly sharpens
//      - writes:
//          out_img : RGBA16F storage (final post output)
//
// Bindings overview (must match Rust bind group layouts):
//
//   group(0) shared scene bindings are in common.wgsl:
//     @binding(0) cam
//     @binding(1) chunks
//     @binding(2) nodes
//     @binding(3) chunk_grid
//
//   group(0) pass-specific (PRIMARY outputs):
//     @binding(4) color_img (storage write, rgba16f)
//     @binding(5) depth_img (storage write, r32f)
//
//   group(1) GODRAY pass inputs/outputs:
//     @binding(0) depth_tex        (sampled, full-res, r32f)
//     @binding(1) godray_hist_tex  (sampled, quarter-res, rgba16f)  // currently unused here
//     @binding(2) godray_out       (storage write, quarter-res, rgba16f)
//
//   group(2) COMPOSITE pass inputs/outputs:
//     @binding(0) color_tex   (sampled, full-res, rgba16f)
//     @binding(1) godray_tex  (sampled, quarter-res, rgba16f)
//     @binding(2) out_img     (storage write, full-res, rgba16f)
//
// Dependencies (defined elsewhere):
// - common.wgsl: cam, chunks, nodes, chunk_grid, SUN_DIR, fog_* helpers, hash12, etc.
// - ray_core.wgsl: safe_inv, trace_chunk_hybrid_interval, HitGeom, etc.
// - shading helpers: shade_hit(), sun_transmittance() (not shown in this snippet).

// -----------------------------------------------------------------------------
// Tuning knobs (find these first)
// -----------------------------------------------------------------------------

// Sentinel for invalid chunk grid entries (must match common.wgsl INVALID/LEAF style).
const INVALID_U32 : u32 = 0xFFFFFFFFu;

// Primary pass
const PRIMARY_NUDGE_VOXEL_FRAC : f32 = 1e-4; // pushed into grid to avoid boundary ambiguity

// Godray pass sampling pattern
const GODRAY_FRAME_FPS : f32 = 60.0;     // used to quantize time into a stable frame index
const GODRAY_BLOCK_SIZE : i32 = 4;       // quarter-res pixel covers a 4x4 full-res block
const GODRAY_PATTERN_HASH_SCALE : f32 = 0.73;

// Tap jitter hashes (arbitrary constants; keep fixed for stable noise character)
const J0_SCALE : f32 = 1.31;
const J1_SCALE : f32 = 2.11;
const J2_SCALE : f32 = 3.01;
const J3_SCALE : f32 = 4.19;

const J0_F : vec2<f32> = vec2<f32>(0.11, 0.17);
const J1_F : vec2<f32> = vec2<f32>(0.23, 0.29);
const J2_F : vec2<f32> = vec2<f32>(0.37, 0.41);
const J3_F : vec2<f32> = vec2<f32>(0.53, 0.59);

// Godray integration
const GODRAY_TV_CUTOFF : f32 = 0.02;     // stop integrating when view transmittance is very low
const GODRAY_STEPS_FAST : u32 = 5u;      // cheap fixed step count

// Composite pass
const COMPOSITE_SHARPEN : f32 = 0.35;    // unsharp-mask amount (0.2..0.6)
const COMPOSITE_GOD_SCALE : f32 = 1.5;   // overall beam contribution into final color
const COMPOSITE_BEAM_COMPRESS : bool = true; // apply 1-exp(-x) compression to beams

// -----------------------------------------------------------------------------
// Pass-specific bindings
// -----------------------------------------------------------------------------

// PRIMARY pass outputs (storage writes).
@group(0) @binding(4) var color_img : texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var depth_img : texture_storage_2d<r32float, write>;

// GODRAY pass bindings.
@group(1) @binding(0) var depth_tex       : texture_2d<f32>;   // full-res depth (r32float)
@group(1) @binding(1) var godray_hist_tex : texture_2d<f32>;   // quarter-res history (rgba16f); optional/unused
@group(1) @binding(2) var godray_out      : texture_storage_2d<rgba16float, write>; // quarter-res output

// COMPOSITE pass bindings.
@group(2) @binding(0) var color_tex  : texture_2d<f32>;        // full-res color (rgba16f)
@group(2) @binding(1) var godray_tex : texture_2d<f32>;        // quarter-res godrays (rgba16f)
@group(2) @binding(2) var out_img    : texture_storage_2d<rgba16float, write>; // final output (rgba16f)

// -----------------------------------------------------------------------------
// Chunk grid helpers
// -----------------------------------------------------------------------------

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

fn chunk_coord_from_pos(p: vec3<f32>, chunk_size_m: f32) -> vec3<i32> {
  return vec3<i32>(
    i32(floor(p.x / chunk_size_m)),
    i32(floor(p.y / chunk_size_m)),
    i32(floor(p.z / chunk_size_m))
  );
}

// -----------------------------------------------------------------------------
// Quarter-res upsample (manual bilerp)
// -----------------------------------------------------------------------------

fn godray_sample_bilerp(px_full: vec2<f32>) -> vec3<f32> {
  let q = px_full * 0.25;
  let q0 = vec2<i32>(i32(floor(q.x)), i32(floor(q.y)));
  let f  = fract(q);

  let qdims = textureDimensions(godray_tex);
  let x0 = clamp(q0.x, 0, i32(qdims.x) - 1);
  let y0 = clamp(q0.y, 0, i32(qdims.y) - 1);
  let x1 = min(x0 + 1, i32(qdims.x) - 1);
  let y1 = min(y0 + 1, i32(qdims.y) - 1);

  let c00 = textureLoad(godray_tex, vec2<i32>(x0, y0), 0).xyz;
  let c10 = textureLoad(godray_tex, vec2<i32>(x1, y0), 0).xyz;
  let c01 = textureLoad(godray_tex, vec2<i32>(x0, y1), 0).xyz;
  let c11 = textureLoad(godray_tex, vec2<i32>(x1, y1), 0).xyz;

  let cx0 = mix(c00, c10, f.x);
  let cx1 = mix(c01, c11, f.x);
  return mix(cx0, cx1, f.y);
}

// -----------------------------------------------------------------------------
// Godray integration (cheap volumetric scattering)
// -----------------------------------------------------------------------------

fn godray_integrate(ro: vec3<f32>, rd: vec3<f32>, t_end: f32, j: f32) -> vec3<f32> {
  let base = fog_density_base();
  if (base <= 0.0 || t_end <= 0.0) { return vec3<f32>(0.0); }

  // NOTE: use the blended phase so beams stay visible off-axis (side views).
  let costh = dot(rd, SUN_DIR);
  let phase = phase_blended(costh);

  let dt = t_end / f32(GODRAY_STEPS_FAST);

  var sum = vec3<f32>(0.0);

  for (var i: u32 = 0u; i < GODRAY_STEPS_FAST; i = i + 1u) {
    let ti = (f32(i) + 0.5 + j) * dt;
    if (ti <= 0.0) { continue; }

    let p  = ro + rd * ti;

    let Tv = fog_transmittance(ro, rd, ti);
    if (Tv < GODRAY_TV_CUTOFF) { break; }

    // Includes your canopy / voxel occluders.
    let Ts = sun_transmittance(p, SUN_DIR);

    // If your cloud code in common.wgsl is active, it should also be applied inside
    // sun_transmittance() (or multiply Ts here by a cheap "cloud sun occlusion" term).
    // This file assumes sun_transmittance already includes *all* occluders.

    let dens = base * exp(-FOG_HEIGHT_FALLOFF * p.y);

    sum += (SUN_COLOR * SUN_INTENSITY) * (dens * dt) * Tv * Ts * phase;
  }

  return sum * GODRAY_STRENGTH;
}

// -----------------------------------------------------------------------------
// PASS 1: PRIMARY (full-res trace + fog)
// -----------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn main_primary(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(color_img);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let res = vec2<f32>(f32(dims.x), f32(dims.y));
  let px  = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

  let ro = cam.cam_pos.xyz;
  let rd = ray_dir_from_pixel(px, res);

  let sky = sky_color(rd);

  let voxel_size = cam.voxel_params.x;
  let nudge_p = PRIMARY_NUDGE_VOXEL_FRAC * voxel_size;

  if (cam.chunk_count == 0u) {
    let ip = vec2<i32>(i32(gid.x), i32(gid.y));
    textureStore(color_img, ip, vec4<f32>(sky, 1.0));
    textureStore(depth_img, ip, vec4<f32>(FOG_MAX_DIST, 0.0, 0.0, 0.0));
    return;
  }

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

  let rtg = intersect_aabb(ro, rd, grid_bmin, grid_bmax);
  var t_enter = max(rtg.x, 0.0);
  let t_exit  = rtg.y;

  if (t_exit < t_enter) {
    let ip = vec2<i32>(i32(gid.x), i32(gid.y));
    textureStore(color_img, ip, vec4<f32>(sky, 1.0));
    textureStore(depth_img, ip, vec4<f32>(FOG_MAX_DIST, 0.0, 0.0, 0.0));
    return;
  }

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

  var best = HitGeom(false, BIG_F32, 0u, vec3<f32>(0.0));
  let t_exit_local = max(t_exit - start_t, 0.0);

  let max_chunk_steps = min((gd.x + gd.y + gd.z) * 6u + 8u, 1024u);

  for (var s: u32 = 0u; s < max_chunk_steps; s = s + 1u) {
    if (t_local > t_exit_local) { break; }

    let tNextLocal = min(tMaxX, min(tMaxY, tMaxZ));
    if (best.hit && (start_t + tNextLocal) >= best.t) { break; }

    let slot = grid_lookup_slot(cx, cy, cz);
    if (slot != INVALID_U32 && slot < cam.chunk_count) {
      let ch = chunks[slot];

      let cell_enter = start_t + t_local;
      let cell_exit  = start_t + min(tNextLocal, t_exit_local);

      let h = trace_chunk_hybrid_interval(ro, rd, ch, cell_enter, cell_exit);
      if (h.hit && h.t < best.t) { best = h; }
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
    if (cx < ox || cy < oy || cz < oz || cx >= ox + nx || cy >= oy + ny || cz >= oz + nz) { break; }
  }

  let surface = select(sky, shade_hit(ro, rd, best), best.hit);
  let t_scene = select(min(t_exit, FOG_MAX_DIST), min(best.t, FOG_MAX_DIST), best.hit);

  let T = fog_transmittance(ro, rd, t_scene);
  let col = surface * T + sky * (1.0 - T);

  let ip = vec2<i32>(i32(gid.x), i32(gid.y));
  textureStore(color_img, ip, vec4<f32>(col, 1.0));
  textureStore(depth_img, ip, vec4<f32>(t_scene, 0.0, 0.0, 0.0));
}

// -----------------------------------------------------------------------------
// PASS 2: GODRAYS (quarter-res)
// -----------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn main_godray(@builtin(global_invocation_id) gid: vec3<u32>) {
  let qdims = textureDimensions(godray_out);
  if (gid.x >= qdims.x || gid.y >= qdims.y) { return; }

  let fdims = textureDimensions(depth_tex);
  let ro = cam.cam_pos.xyz;

  let frame = floor(cam.voxel_params.y * GODRAY_FRAME_FPS);
  let qpx = vec2<f32>(f32(gid.x), f32(gid.y));

  let flip = select(0.0, 1.0, hash12(qpx * GODRAY_PATTERN_HASH_SCALE + vec2<f32>(frame, frame * 0.21)) > 0.5);

  let base_x = i32(gid.x) * GODRAY_BLOCK_SIZE;
  let base_y = i32(gid.y) * GODRAY_BLOCK_SIZE;

  // Pattern A: (1,1) (3,1) (1,3) (3,3)
  // Pattern B: (2,1) (1,2) (3,2) (2,3)
  let ax0 = select(1, 2, flip > 0.5);
  let ay0 = 1;
  let ax1 = select(3, 1, flip > 0.5);
  let ay1 = select(1, 2, flip > 0.5);
  let ax2 = select(1, 3, flip > 0.5);
  let ay2 = select(3, 2, flip > 0.5);
  let ax3 = select(3, 2, flip > 0.5);
  let ay3 = 3;

  let fp0 = vec2<i32>(clamp(base_x + ax0, 0, i32(fdims.x) - 1),
                      clamp(base_y + ay0, 0, i32(fdims.y) - 1));
  let fp1 = vec2<i32>(clamp(base_x + ax1, 0, i32(fdims.x) - 1),
                      clamp(base_y + ay1, 0, i32(fdims.y) - 1));
  let fp2 = vec2<i32>(clamp(base_x + ax2, 0, i32(fdims.x) - 1),
                      clamp(base_y + ay2, 0, i32(fdims.y) - 1));
  let fp3 = vec2<i32>(clamp(base_x + ax3, 0, i32(fdims.x) - 1),
                      clamp(base_y + ay3, 0, i32(fdims.y) - 1));

  let res_full = vec2<f32>(f32(fdims.x), f32(fdims.y));

  var acc = vec3<f32>(0.0);
  var wsum = 0.0;

  let j0 = (hash12(qpx * J0_SCALE + vec2<f32>(frame * J0_F.x, frame * J0_F.y)) - 0.5);
  let j1 = (hash12(qpx * J1_SCALE + vec2<f32>(frame * J1_F.x, frame * J1_F.y)) - 0.5);
  let j2 = (hash12(qpx * J2_SCALE + vec2<f32>(frame * J2_F.x, frame * J2_F.y)) - 0.5);
  let j3 = (hash12(qpx * J3_SCALE + vec2<f32>(frame * J3_F.x, frame * J3_F.y)) - 0.5);

  let t_scene0 = textureLoad(depth_tex, fp0, 0).x;
  let t_end0 = min(t_scene0, GODRAY_MAX_DIST);
  if (t_end0 > 0.0 && fog_density_base() > 0.0) {
    let px0 = vec2<f32>(f32(fp0.x) + 0.5, f32(fp0.y) + 0.5);
    let rd0 = ray_dir_from_pixel(px0, res_full);
    acc += godray_integrate(ro, rd0, t_end0, j0);
    wsum += 1.0;
  }

  let t_scene1 = textureLoad(depth_tex, fp1, 0).x;
  let t_end1 = min(t_scene1, GODRAY_MAX_DIST);
  if (t_end1 > 0.0 && fog_density_base() > 0.0) {
    let px1 = vec2<f32>(f32(fp1.x) + 0.5, f32(fp1.y) + 0.5);
    let rd1 = ray_dir_from_pixel(px1, res_full);
    acc += godray_integrate(ro, rd1, t_end1, j1);
    wsum += 1.0;
  }

  let t_scene2 = textureLoad(depth_tex, fp2, 0).x;
  let t_end2 = min(t_scene2, GODRAY_MAX_DIST);
  if (t_end2 > 0.0 && fog_density_base() > 0.0) {
    let px2 = vec2<f32>(f32(fp2.x) + 0.5, f32(fp2.y) + 0.5);
    let rd2 = ray_dir_from_pixel(px2, res_full);
    acc += godray_integrate(ro, rd2, t_end2, j2);
    wsum += 1.0;
  }

  let t_scene3 = textureLoad(depth_tex, fp3, 0).x;
  let t_end3 = min(t_scene3, GODRAY_MAX_DIST);
  if (t_end3 > 0.0 && fog_density_base() > 0.0) {
    let px3 = vec2<f32>(f32(fp3.x) + 0.5, f32(fp3.y) + 0.5);
    let rd3 = ray_dir_from_pixel(px3, res_full);
    acc += godray_integrate(ro, rd3, t_end3, j3);
    wsum += 1.0;
  }

  let raw = select(vec3<f32>(0.0), acc / wsum, wsum > 0.0);
  let outv = max(raw, vec3<f32>(0.0));

  textureStore(godray_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(outv, 1.0));
}

// -----------------------------------------------------------------------------
// PASS 3: COMPOSITE (full-res)
// -----------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn main_composite(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(out_img);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let ip = vec2<i32>(i32(gid.x), i32(gid.y));
  let base = textureLoad(color_tex, ip, 0).xyz;

  let px = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

  let g = godray_sample_bilerp(px);

  let gx = godray_sample_bilerp(px + vec2<f32>( 1.0, 0.0)) + godray_sample_bilerp(px + vec2<f32>(-1.0, 0.0));
  let gy = godray_sample_bilerp(px + vec2<f32>(0.0,  1.0)) + godray_sample_bilerp(px + vec2<f32>(0.0, -1.0));
  let blur = 0.25 * (gx + gy);

  let god_raw = max(g + COMPOSITE_SHARPEN * (g - blur), vec3<f32>(0.0));

  // Optional beam compression: keeps beams visible without nuking highlights.
  let god = select(god_raw, (vec3<f32>(1.0) - exp(-god_raw)), COMPOSITE_BEAM_COMPRESS);

  textureStore(out_img, ip, vec4<f32>(base + COMPOSITE_GOD_SCALE * god, 1.0));
}
