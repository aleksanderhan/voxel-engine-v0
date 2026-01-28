// src/shaders/ray_main.wgsl
// -------------------------
// ray_main.wgsl
//
// Compute entrypoints only.
// Depends on: common.wgsl + ray_core.wgsl + clipmap.wgsl

@group(0) @binding(4) var color_img : texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var depth_img : texture_storage_2d<r32float, write>;

@group(1) @binding(0) var depth_tex       : texture_2d<f32>;
@group(1) @binding(1) var godray_hist_tex : texture_2d<f32>;
@group(1) @binding(2) var godray_out      : texture_storage_2d<rgba16float, write>;

@group(2) @binding(0) var color_tex  : texture_2d<f32>;
@group(2) @binding(1) var godray_tex : texture_2d<f32>;
@group(2) @binding(2) var out_img    : texture_storage_2d<rgba16float, write>;

fn tonemap_exp(hdr: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(1.0) - exp(-hdr * POST_EXPOSURE);
}

// Quarter-res upsample (manual bilerp)
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

fn godray_integrate(ro: vec3<f32>, rd: vec3<f32>, t_end: f32, j: f32) -> vec3<f32> {
  let base = fog_density_godray();
  if (base <= 0.0 || t_end <= 0.0) { return vec3<f32>(0.0); }

  let costh = dot(rd, SUN_DIR);
  let phase = phase_blended(costh);

  let dt = t_end / f32(GODRAY_STEPS_FAST);

  var sum = vec3<f32>(0.0);

  var ts_lp: f32    = 1.0;
  var shaft_lp: f32 = 0.0;

  let a_ts    = 1.0 - exp(-dt * 3.0);
  let a_shaft = 1.0 - exp(-dt * 5.0);

  for (var i: u32 = 0u; i < GODRAY_STEPS_FAST; i = i + 1u) {
    let ti = (f32(i) + 0.5 + j) * dt;
    if (ti <= 0.0) { continue; }

    let p = ro + rd * ti;

    let Tv = fog_transmittance_godray(ro, rd, ti);
    if (Tv < GODRAY_TV_CUTOFF) { break; }

    let Ts0 = sun_transmittance(p, SUN_DIR);
    let Ts_soft = pow(clamp(Ts0, 0.0, 1.0), 0.75);

    let ts_prev = ts_lp;
    ts_lp = mix(ts_lp, Ts_soft, a_ts);

    let dTs = abs(Ts_soft - ts_prev);

    var shaft = smoothstep(GODRAY_EDGE0, GODRAY_EDGE1, dTs);
    shaft = sqrt(shaft);
    shaft_lp = mix(shaft_lp, shaft, a_shaft);
    shaft = shaft_lp;

    let haze_ramp = 1.0 - exp(-ti / GODRAY_HAZE_NEAR_FADE);
    let haze = GODRAY_BASE_HAZE * haze_ramp;

    let shaft_sun_gate = smoothstep(0.10, 0.55, ts_lp);

    let w = haze + (1.0 - haze) * (shaft * shaft_sun_gate);

    let hfall = GODRAY_SCATTER_HEIGHT_FALLOFF;
    let hmin  = GODRAY_SCATTER_MIN_FRAC;
    let height_term = max(exp(-hfall * p.y), hmin);

    let dens = base * height_term;

    let strength_scale = 0.70;

    sum += (SUN_COLOR * SUN_INTENSITY) * (dens * dt) * Tv * ts_lp * phase * w * strength_scale;
  }

  return sum * GODRAY_STRENGTH;
}

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

  // If no SVO chunks, still render clipmap terrain.
  if (cam.chunk_count == 0u) {
    let hf = clip_trace_heightfield(ro, rd, 0.0, FOG_MAX_DIST);

    let surface = select(sky, shade_clip_hit(ro, rd, hf), hf.hit);
    let t_scene = select(FOG_MAX_DIST, min(hf.t, FOG_MAX_DIST), hf.hit);

    let T = fog_transmittance_primary(ro, rd, t_scene);
    let fogc = fog_color(rd);
    let fog_amt = (1.0 - T) * FOG_PRIMARY_VIS;
    let col = mix(surface, fogc, fog_amt);

    let ip = vec2<i32>(i32(gid.x), i32(gid.y));
    textureStore(color_img, ip, vec4<f32>(col, 1.0));
    textureStore(depth_img, ip, vec4<f32>(t_scene, 0.0, 0.0, 0.0));
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

  // Outside streamed SVO grid => clipmap fallback.
  if (t_exit < t_enter) {
    let hf = clip_trace_heightfield(ro, rd, 0.0, FOG_MAX_DIST);

    let surface = select(sky, shade_clip_hit(ro, rd, hf), hf.hit);
    let t_scene = select(FOG_MAX_DIST, min(hf.t, FOG_MAX_DIST), hf.hit);

    let T = fog_transmittance_primary(ro, rd, t_scene);
    let fogc = fog_color(rd);
    let fog_amt = (1.0 - T) * FOG_PRIMARY_VIS;
    let col = mix(surface, fogc, fog_amt);

    let ip = vec2<i32>(i32(gid.x), i32(gid.y));
    textureStore(color_img, ip, vec4<f32>(col, 1.0));
    textureStore(depth_img, ip, vec4<f32>(t_scene, 0.0, 0.0, 0.0));
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

  var best = HitGeom(false, BIG_F32, MAT_AIR, vec3<f32>(0.0));
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

  // If no voxel hit, try heightfield clipmap fallback.
  let hf = clip_trace_heightfield(ro, rd, 0.0, FOG_MAX_DIST);

  let use_vox = best.hit;
  let use_hf  = (!best.hit) && hf.hit;

  let surface = select(
    sky,
    select(shade_clip_hit(ro, rd, hf), shade_hit(ro, rd, best), use_vox),
    (use_vox || use_hf)
  );

  let t_scene = select(
    min(t_exit, FOG_MAX_DIST),
    select(min(hf.t, FOG_MAX_DIST), min(best.t, FOG_MAX_DIST), use_vox),
    (use_vox || use_hf)
  );

  let T = fog_transmittance_primary(ro, rd, t_scene);
  let fogc = fog_color(rd);

  let fog_amt = (1.0 - T) * FOG_PRIMARY_VIS;
  let col = mix(surface, fogc, fog_amt);

  let ip = vec2<i32>(i32(gid.x), i32(gid.y));
  textureStore(color_img, ip, vec4<f32>(col, 1.0));
  textureStore(depth_img, ip, vec4<f32>(t_scene, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn main_godray(@builtin(global_invocation_id) gid: vec3<u32>) {
  let qdims = textureDimensions(godray_out);
  if (gid.x >= qdims.x || gid.y >= qdims.y) { return; }

  let fdims = textureDimensions(depth_tex);
  let ro = cam.cam_pos.xyz;

  let hip  = vec2<i32>(i32(gid.x), i32(gid.y));
  let qpx  = vec2<f32>(f32(gid.x), f32(gid.y));

  let frame = floor(cam.voxel_params.y * GODRAY_FRAME_FPS);
  let flip = select(
    0.0, 1.0,
    hash12(qpx * GODRAY_PATTERN_HASH_SCALE + vec2<f32>(frame, frame * 0.21)) > 0.5
  );

  let base_x = i32(gid.x) * GODRAY_BLOCK_SIZE;
  let base_y = i32(gid.y) * GODRAY_BLOCK_SIZE;

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

  let j0 = (hash12(qpx * J0_SCALE + vec2<f32>(frame * J0_F.x, frame * J0_F.y)) - 0.5);
  let j1 = (hash12(qpx * J1_SCALE + vec2<f32>(frame * J1_F.x, frame * J1_F.y)) - 0.5);
  let j2 = (hash12(qpx * J2_SCALE + vec2<f32>(frame * J2_F.x, frame * J2_F.y)) - 0.5);
  let j3 = (hash12(qpx * J3_SCALE + vec2<f32>(frame * J3_F.x, frame * J3_F.y)) - 0.5);

  let t_scene0 = textureLoad(depth_tex, fp0, 0).x;
  let t_scene1 = textureLoad(depth_tex, fp1, 0).x;
  let t_scene2 = textureLoad(depth_tex, fp2, 0).x;
  let t_scene3 = textureLoad(depth_tex, fp3, 0).x;

  var acc = vec3<f32>(0.0);
  var wsum = 0.0;

  let t_end0 = min(t_scene0, GODRAY_MAX_DIST);
  if (t_end0 > 0.0 && fog_density_godray() > 0.0) {
    let px0 = vec2<f32>(f32(fp0.x) + 0.5, f32(fp0.y) + 0.5);
    acc += godray_integrate(ro, ray_dir_from_pixel(px0, res_full), t_end0, j0);
    wsum += 1.0;
  }

  let t_end1 = min(t_scene1, GODRAY_MAX_DIST);
  if (t_end1 > 0.0 && fog_density_godray() > 0.0) {
    let px1 = vec2<f32>(f32(fp1.x) + 0.5, f32(fp1.y) + 0.5);
    acc += godray_integrate(ro, ray_dir_from_pixel(px1, res_full), t_end1, j1);
    wsum += 1.0;
  }

  let t_end2 = min(t_scene2, GODRAY_MAX_DIST);
  if (t_end2 > 0.0 && fog_density_godray() > 0.0) {
    let px2 = vec2<f32>(f32(fp2.x) + 0.5, f32(fp2.y) + 0.5);
    acc += godray_integrate(ro, ray_dir_from_pixel(px2, res_full), t_end2, j2);
    wsum += 1.0;
  }

  let t_end3 = min(t_scene3, GODRAY_MAX_DIST);
  if (t_end3 > 0.0 && fog_density_godray() > 0.0) {
    let px3 = vec2<f32>(f32(fp3.x) + 0.5, f32(fp3.y) + 0.5);
    acc += godray_integrate(ro, ray_dir_from_pixel(px3, res_full), t_end3, j3);
    wsum += 1.0;
  }

  let cur = max(select(vec3<f32>(0.0), acc / wsum, wsum > 0.0), vec3<f32>(0.0));

  let hist = textureLoad(godray_hist_tex, hip, 0).xyz;

  let dmin = min(min(t_scene0, t_scene1), min(t_scene2, t_scene3));
  let dmax = max(max(t_scene0, t_scene1), max(t_scene2, t_scene3));
  let span = (dmax - dmin) / max(dmin, 1e-3);
  let edge = smoothstep(0.03, 0.15, span);

  let delta = length(cur - hist);
  let react = smoothstep(0.03, 0.18, delta);

  let stable = 1.0 - max(edge, react);

  let clamp_w = max(cur * 0.75, vec3<f32>(0.02));
  let hist_clamped = clamp(hist, cur - clamp_w, cur + clamp_w);

  let hist_w = clamp(GODRAY_TS_LP_ALPHA * stable, 0.0, 0.90);

  let blended = mix(cur, hist_clamped, hist_w);

  textureStore(godray_out, hip, vec4<f32>(blended, 1.0));
}

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

  var god_raw = max(g + COMPOSITE_SHARPEN * (g - blur), vec3<f32>(0.0));

  god_raw = max(god_raw - vec3<f32>(GODRAY_BLACK_LEVEL), vec3<f32>(0.0));

  let god = (vec3<f32>(1.0) - exp(-god_raw));

  let hdr = max(base + COMPOSITE_GOD_SCALE * god, vec3<f32>(0.0));
  let ldr = tonemap_exp(hdr);

  textureStore(out_img, ip, vec4<f32>(ldr, 1.0));
}
