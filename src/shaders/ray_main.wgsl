// ray_main.wgsl
//
// 3-pass compute:
//  - main_primary   : full-res trace + fog -> writes color_img + depth_img
//  - main_godray    : quarter-res godrays (4 taps per 4x4) -> writes godray_out
//  - main_composite : full-res composite color + godrays -> writes out_img

const INVALID_U32 : u32 = 0xFFFFFFFFu;

// ---------- Pass-specific bindings ----------
@group(0) @binding(4) var color_img : texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var depth_img : texture_storage_2d<r32float, write>;

// Godray pass
@group(1) @binding(0) var depth_tex       : texture_2d<f32>;   // full-res, r32float
@group(1) @binding(1) var godray_hist_tex : texture_2d<f32>;   // quarter-res, rgba16float (can be unused)
@group(1) @binding(2) var godray_out      : texture_storage_2d<rgba16float, write>;

// Composite pass
@group(2) @binding(0) var color_tex  : texture_2d<f32>;        // full-res, rgba16float
@group(2) @binding(1) var godray_tex : texture_2d<f32>;        // quarter-res, rgba16float
@group(2) @binding(2) var out_img    : texture_storage_2d<rgba16float, write>;

// ---------- Grid helpers ----------
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

// ---------- Quarter-res upsample (manual bilerp) ----------
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

// ---------- Godray integrate (3 steps) ----------
const GODRAY_STEPS_FAST : u32 = 4u;

fn godray_integrate(ro: vec3<f32>, rd: vec3<f32>, t_end: f32, j: f32) -> vec3<f32> {
  let base = fog_density_base();
  if (base <= 0.0 || t_end <= 0.0) { return vec3<f32>(0.0); }

  let costh = dot(rd, SUN_DIR);
  let phase = phase_mie(costh);

  let dt = t_end / f32(GODRAY_STEPS_FAST);

  var sum = vec3<f32>(0.0);

  for (var i: u32 = 0u; i < GODRAY_STEPS_FAST; i = i + 1u) {
    let ti = (f32(i) + 0.5 + j) * dt;
    if (ti <= 0.0) { continue; }

    let p  = ro + rd * ti;

    let Tv = fog_transmittance(ro, rd, ti);
    if (Tv < 0.02) { break; }

    let Ts = sun_transmittance(p, SUN_DIR);

    let dens = base * exp(-FOG_HEIGHT_FALLOFF * p.y);

    sum += (SUN_COLOR * SUN_INTENSITY) * (dens * dt) * Tv * Ts * phase;
  }

  return sum * GODRAY_STRENGTH;
}

// ---------- PASS 1: PRIMARY ----------
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
  let nudge_p = 1e-4 * voxel_size;

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

  // DDA uses LOCAL t from p0 (t=0 at p0)
  var t_local: f32 = 0.0;

  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));
  let step_x: i32 = select(-1, 1, rd.x > 0.0);
  let step_y: i32 = select(-1, 1, rd.y > 0.0);
  let step_z: i32 = select(-1, 1, rd.z > 0.0);

  let bx = select(f32(cx) * chunk_size_m, f32(cx + 1) * chunk_size_m, rd.x > 0.0);
  let by = select(f32(cy) * chunk_size_m, f32(cy + 1) * chunk_size_m, rd.y > 0.0);
  let bz = select(f32(cz) * chunk_size_m, f32(cz + 1) * chunk_size_m, rd.z > 0.0);

  // These are LOCAL deltas from p0 (good)
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

  // Convert grid exit to LOCAL
  let t_exit_local = max(t_exit - start_t, 0.0);

  let max_chunk_steps = min((gd.x + gd.y + gd.z) * 6u + 8u, 1024u);

  for (var s: u32 = 0u; s < max_chunk_steps; s = s + 1u) {
    if (t_local > t_exit_local) { break; }

    let tNextLocal = min(tMaxX, min(tMaxY, tMaxZ));

    // Compare against best hit in ABSOLUTE space
    if (best.hit && (start_t + tNextLocal) >= best.t) { break; }

    let slot = grid_lookup_slot(cx, cy, cz);
    if (slot != INVALID_U32 && slot < cam.chunk_count) {
      let ch = chunks[slot];

      let cell_enter = start_t + t_local;
      let cell_exit  = start_t + min(tNextLocal, t_exit_local);

      let h = trace_chunk_hybrid_interval(ro, rd, ch, cell_enter, cell_exit);
      if (h.hit && h.t < best.t) { best = h; }
    }

    // Step (LOCAL)
    if (tMaxX < tMaxY) {
      if (tMaxX < tMaxZ) { cx += step_x; t_local = tMaxX; tMaxX += tDeltaX; }
      else               { cz += step_z; t_local = tMaxZ; tMaxZ += tDeltaZ; }
    } else {
      if (tMaxY < tMaxZ) { cy += step_y; t_local = tMaxY; tMaxY += tDeltaY; }
      else               { cz += step_z; t_local = tMaxZ; tMaxZ += tDeltaZ; }
    }

    // bounds
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

// ---------- PASS 2: GODRAYS (quarter-res, 4 taps per 4x4, rotated pattern) ----------
@compute @workgroup_size(8, 8, 1)
fn main_godray(@builtin(global_invocation_id) gid: vec3<u32>) {
  let qdims = textureDimensions(godray_out);
  if (gid.x >= qdims.x || gid.y >= qdims.y) { return; }

  let fdims = textureDimensions(depth_tex);
  let ro = cam.cam_pos.xyz;

  // frame-locked jitter seed (prevents shimmering)
  let frame = floor(cam.voxel_params.y * 60.0);
  let qpx = vec2<f32>(f32(gid.x), f32(gid.y));

  // Rotate the 4-tap pattern per-quarter pixel to break the grid.
  // flip = 0 or 1
  let flip = select(0.0, 1.0, hash12(qpx * 0.73 + vec2<f32>(frame, frame * 0.21)) > 0.5);

  let base_x = i32(gid.x) * 4;
  let base_y = i32(gid.y) * 4;

  // 4 taps inside the 4x4 block. Two patterns:
  // A: (1,1) (3,1) (1,3) (3,3)
  // B: (2,1) (1,2) (3,2) (2,3)  (offset/rotated)
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

  // Per-tap deterministic jitter (still frame-locked)
  let j0 = (hash12(qpx * 1.31 + vec2<f32>(frame * 0.11, frame * 0.17)) - 0.5);
  let j1 = (hash12(qpx * 2.11 + vec2<f32>(frame * 0.23, frame * 0.29)) - 0.5);
  let j2 = (hash12(qpx * 3.01 + vec2<f32>(frame * 0.37, frame * 0.41)) - 0.5);
  let j3 = (hash12(qpx * 4.19 + vec2<f32>(frame * 0.53, frame * 0.59)) - 0.5);

  // Tap helper inline
  // 0
  let t_scene0 = textureLoad(depth_tex, fp0, 0).x;
  let t_end0 = min(t_scene0, GODRAY_MAX_DIST);
  if (t_end0 > 0.0 && fog_density_base() > 0.0) {
    let px0 = vec2<f32>(f32(fp0.x) + 0.5, f32(fp0.y) + 0.5);
    let rd0 = ray_dir_from_pixel(px0, res_full);
    acc += godray_integrate(ro, rd0, t_end0, j0);
    wsum += 1.0;
  }
  // 1
  let t_scene1 = textureLoad(depth_tex, fp1, 0).x;
  let t_end1 = min(t_scene1, GODRAY_MAX_DIST);
  if (t_end1 > 0.0 && fog_density_base() > 0.0) {
    let px1 = vec2<f32>(f32(fp1.x) + 0.5, f32(fp1.y) + 0.5);
    let rd1 = ray_dir_from_pixel(px1, res_full);
    acc += godray_integrate(ro, rd1, t_end1, j1);
    wsum += 1.0;
  }
  // 2
  let t_scene2 = textureLoad(depth_tex, fp2, 0).x;
  let t_end2 = min(t_scene2, GODRAY_MAX_DIST);
  if (t_end2 > 0.0 && fog_density_base() > 0.0) {
    let px2 = vec2<f32>(f32(fp2.x) + 0.5, f32(fp2.y) + 0.5);
    let rd2 = ray_dir_from_pixel(px2, res_full);
    acc += godray_integrate(ro, rd2, t_end2, j2);
    wsum += 1.0;
  }
  // 3
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

// ---------- PASS 3: COMPOSITE (full-res, sharper upsample) ----------
@compute @workgroup_size(8, 8, 1)
fn main_composite(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(out_img);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let ip = vec2<i32>(i32(gid.x), i32(gid.y));
  let base = textureLoad(color_tex, ip, 0).xyz;

  let px = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

  // Just bilerp quarter-res -> full-res (sharpest)
  let g = godray_sample_bilerp(px);

  // Optional: mild "unsharp mask" using 4-neighborhood of bilerp
  // (keeps shafts defined without block edges)
  let gx = godray_sample_bilerp(px + vec2<f32>( 1.0, 0.0)) + godray_sample_bilerp(px + vec2<f32>(-1.0, 0.0));
  let gy = godray_sample_bilerp(px + vec2<f32>(0.0,  1.0)) + godray_sample_bilerp(px + vec2<f32>(0.0, -1.0));
  let blur = 0.25 * (gx + gy);             // very light blur estimate
  let god  = max(g + 0.35 * (g - blur), vec3<f32>(0.0)); // sharpen amount: 0.2..0.6

  textureStore(out_img, ip, vec4<f32>(base + god, 1.0));
}
