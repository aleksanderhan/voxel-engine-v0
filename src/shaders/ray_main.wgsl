// src/shaders/ray_main.wgsl
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

// Depth-aware godray upsample needs full-res depth in composite.
@group(2) @binding(3) var depth_full : texture_2d<f32>;

fn tonemap_aces(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x*(a*x + b)) / (x*(c*x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn gamma_encode(x: vec3<f32>) -> vec3<f32> {
  return pow(x, vec3<f32>(1.0 / 2.2));
}

// Bloom helper
fn bright_extract_hue(x: vec3<f32>, thresh: f32) -> vec3<f32> {
  // Luminance threshold -> keeps chroma, avoids "white bloom"
  let lum_w = vec3<f32>(0.2126, 0.7152, 0.0722);
  let l = dot(x, lum_w);

  let over = max(l - thresh, 0.0);
  let w = over / max(l, 1e-4);
  return x * w;
}


// Depth-aware quarter-res upsample (manual bilerp + depth similarity gate)
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

  // depth similarity gate (edge-aware)
  let ip = vec2<i32>(i32(floor(px_full.x)), i32(floor(px_full.y)));
  let d0 = textureLoad(depth_full, ip, 0).x;

  // approximate corresponding full-res positions for the quarter-res corners
  let p00 = vec2<i32>(clamp(x0 * 4, 0, i32(textureDimensions(depth_full).x) - 1),
                      clamp(y0 * 4, 0, i32(textureDimensions(depth_full).y) - 1));
  let p10 = vec2<i32>(clamp(x1 * 4, 0, i32(textureDimensions(depth_full).x) - 1),
                      clamp(y0 * 4, 0, i32(textureDimensions(depth_full).y) - 1));
  let p01 = vec2<i32>(clamp(x0 * 4, 0, i32(textureDimensions(depth_full).x) - 1),
                      clamp(y1 * 4, 0, i32(textureDimensions(depth_full).y) - 1));
  let p11 = vec2<i32>(clamp(x1 * 4, 0, i32(textureDimensions(depth_full).x) - 1),
                      clamp(y1 * 4, 0, i32(textureDimensions(depth_full).y) - 1));

  let d00 = textureLoad(depth_full, p00, 0).x;
  let d10 = textureLoad(depth_full, p10, 0).x;
  let d01 = textureLoad(depth_full, p01, 0).x;
  let d11 = textureLoad(depth_full, p11, 0).x;

  // tolerance increases with distance
  let tol = 0.02 + 0.06 * smoothstep(10.0, 80.0, d0);

  let w00 = exp(-abs(d00 - d0) / tol);
  let w10 = exp(-abs(d10 - d0) / tol);
  let w01 = exp(-abs(d01 - d0) / tol);
  let w11 = exp(-abs(d11 - d0) / tol);

  // Weighted bilerp: apply depth weights per corner before mixing.
  // This avoids "mix across edges then gate" shimmer.
  let wf00 = w00 * (1.0 - f.x) * (1.0 - f.y);
  let wf10 = w10 * (f.x)       * (1.0 - f.y);
  let wf01 = w01 * (1.0 - f.x) * (f.y);
  let wf11 = w11 * (f.x)       * (f.y);

  let wsum = max(wf00 + wf10 + wf01 + wf11, 1e-4);
  let c = (c00 * wf00 + c10 * wf10 + c01 * wf01 + c11 * wf11) / wsum;

  let k = 0.25;

  // decompress back to linear energy (matches main_godray compression)
  let one = vec3<f32>(1.0);
  let denom = max(one - c, vec3<f32>(1e-4));
  let c_lin = (k * c) / denom;

  return c_lin;
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

    let Ts_geom = sun_transmittance_geom_only(p, SUN_DIR);
    let Tc      = cloud_sun_transmittance(p, SUN_DIR);

    let Tc_vol  = mix(1.0, Tc, CLOUD_GODRAY_W);
    let Ts_soft = pow(clamp(Ts_geom, 0.0, 1.0), 0.75);

    let ts_prev = ts_lp;
    ts_lp = mix(ts_lp, Ts_soft, a_ts);

    let dTs = max(0.0, ts_prev - Ts_soft);

    var shaft = smoothstep(GODRAY_EDGE0, GODRAY_EDGE1, dTs);
    shaft *= (1.0 - Ts_soft);
    shaft = pow(clamp(shaft, 0.0, 1.0), 0.55);

    shaft_lp = mix(shaft_lp, shaft, a_shaft);
    shaft = shaft_lp;

    let haze_ramp = 1.0 - exp(-ti / GODRAY_HAZE_NEAR_FADE);
    let haze = GODRAY_BASE_HAZE * haze_ramp * pow(ts_lp, 2.0);

    let shaft_sun_gate = smoothstep(0.35, 0.80, ts_lp);
    let w_raw = haze + (1.0 - haze) * (shaft * shaft_sun_gate);
    let w = clamp(w_raw, 0.0, 1.0);

    let hfall = GODRAY_SCATTER_HEIGHT_FALLOFF;
    let hmin  = GODRAY_SCATTER_MIN_FRAC;
    let height_term = max(exp(-hfall * p.y), hmin);

    let dens = base * height_term;

    let strength_scale = 0.70;

    sum += (SUN_COLOR * SUN_INTENSITY)
     * (dens * dt) * Tv * ts_lp * phase * w
     * Tc_vol
     * strength_scale;
  }

  // --- new knobs (from common.wgsl) ---
  var gr = sum * GODRAY_ENERGY_BOOST;
  gr = gr / (gr + vec3<f32>(GODRAY_KNEE_INTEGRATE));
  return gr;
}

// ------------------------------------------------------------
// Helper: physically-correct aerial perspective
// col = surface*T + ins*(1-T)
// ------------------------------------------------------------
fn apply_fog(surface: vec3<f32>, ro: vec3<f32>, rd: vec3<f32>, t_scene: f32) -> vec3<f32> {
  let T    = fog_transmittance_primary(ro, rd, t_scene);
  let fogc = fog_color(rd);
  let ins  = fog_inscatter(rd, fogc);
  return surface * T + ins * (1.0 - T);
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

    let col = apply_fog(surface, ro, rd, t_scene);

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

    let col = apply_fog(surface, ro, rd, t_scene);

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

  var best = miss_hitgeom();
  let t_exit_local = max(t_exit - start_t, 0.0);

  let max_chunk_steps = min((gd.x + gd.y + gd.z) * 6u + 8u, 1024u);

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

  let use_vox = (best.hit != 0u);
  let use_hf  = (!use_vox) && hf.hit;

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

  let col = apply_fog(surface, ro, rd, t_scene);

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

  // --------------------------------------------------------------------------
  // Stable sampling pattern (NO per-frame flip). This kills the “A/B” shimmer.
  // --------------------------------------------------------------------------
  let base_x = i32(gid.x) * GODRAY_BLOCK_SIZE;
  let base_y = i32(gid.y) * GODRAY_BLOCK_SIZE;

  // 8 taps inside the 4x4 block (denser sampling -> less coarse)
  let ax0 = 1; let ay0 = 1;
  let ax1 = 3; let ay1 = 1;
  let ax2 = 1; let ay2 = 3;
  let ax3 = 3; let ay3 = 3;

  let ax4 = 2; let ay4 = 1;
  let ax5 = 1; let ay5 = 2;
  let ax6 = 3; let ay6 = 2;
  let ax7 = 2; let ay7 = 3;

  let fp0 = vec2<i32>(clamp(base_x + ax0, 0, i32(fdims.x) - 1),
                      clamp(base_y + ay0, 0, i32(fdims.y) - 1));
  let fp1 = vec2<i32>(clamp(base_x + ax1, 0, i32(fdims.x) - 1),
                      clamp(base_y + ay1, 0, i32(fdims.y) - 1));
  let fp2 = vec2<i32>(clamp(base_x + ax2, 0, i32(fdims.x) - 1),
                      clamp(base_y + ay2, 0, i32(fdims.y) - 1));
  let fp3 = vec2<i32>(clamp(base_x + ax3, 0, i32(fdims.x) - 1),
                      clamp(base_y + ay3, 0, i32(fdims.y) - 1));
  let fp4 = vec2<i32>(clamp(base_x + ax4, 0, i32(fdims.x) - 1),
                      clamp(base_y + ay4, 0, i32(fdims.y) - 1));

  let res_full = vec2<f32>(f32(fdims.x), f32(fdims.y));

  // --------------------------------------------------------------------------
  // Stable per-pixel jitter (NO time term).
  // Keeps noise “static” instead of flickering.
  // --------------------------------------------------------------------------
  let j0 = 0.20 * (hash12(qpx * J0_SCALE) - 0.5);
  let j1 = 0.20 * (hash12(qpx * J1_SCALE + vec2<f32>(11.0, 3.0)) - 0.5);
  let j2 = 0.20 * (hash12(qpx * J2_SCALE + vec2<f32>(5.0, 17.0)) - 0.5);
  let j3 = 0.20 * (hash12(qpx * J3_SCALE + vec2<f32>(23.0, 29.0)) - 0.5);

  let t_scene0 = textureLoad(depth_tex, fp0, 0).x;
  let t_scene1 = textureLoad(depth_tex, fp1, 0).x;
  let t_scene2 = textureLoad(depth_tex, fp2, 0).x;
  let t_scene3 = textureLoad(depth_tex, fp3, 0).x;
  let t_scene4 = textureLoad(depth_tex, fp4, 0).x;

  var acc = vec3<f32>(0.0);
  var wsum = 0.0;

  // Optional: tiny quantization reduces “end-distance thrash” on leafy depth.
  // If you don’t want this, replace each t_endN with min(t_sceneN, GODRAY_MAX_DIST).
  let qstep = 0.1; // meters

  let t_end0 = min(floor(t_scene0 / qstep) * qstep, GODRAY_MAX_DIST);
  if (t_end0 > 0.0 && fog_density_godray() > 0.0) {
    let px0 = vec2<f32>(f32(fp0.x) + 0.5, f32(fp0.y) + 0.5);
    acc += godray_integrate(ro, ray_dir_from_pixel(px0, res_full), t_end0, j0);
    wsum += 1.0;
  }

  let t_end1 = min(floor(t_scene1 / qstep) * qstep, GODRAY_MAX_DIST);
  if (t_end1 > 0.0 && fog_density_godray() > 0.0) {
    let px1 = vec2<f32>(f32(fp1.x) + 0.5, f32(fp1.y) + 0.5);
    acc += godray_integrate(ro, ray_dir_from_pixel(px1, res_full), t_end1, j1);
    wsum += 1.0;
  }

  let t_end2 = min(floor(t_scene2 / qstep) * qstep, GODRAY_MAX_DIST);
  if (t_end2 > 0.0 && fog_density_godray() > 0.0) {
    let px2 = vec2<f32>(f32(fp2.x) + 0.5, f32(fp2.y) + 0.5);
    acc += godray_integrate(ro, ray_dir_from_pixel(px2, res_full), t_end2, j2);
    wsum += 1.0;
  }

  let t_end3 = min(floor(t_scene3 / qstep) * qstep, GODRAY_MAX_DIST);
  if (t_end3 > 0.0 && fog_density_godray() > 0.0) {
    let px3 = vec2<f32>(f32(fp3.x) + 0.5, f32(fp3.y) + 0.5);
    acc += godray_integrate(ro, ray_dir_from_pixel(px3, res_full), t_end3, j3);
    wsum += 1.0;
  }
  let t_end4 = min(floor(t_scene4 / qstep) * qstep, GODRAY_MAX_DIST);
  if (t_end4 > 0.0 && fog_density_godray() > 0.0) {
    let px4 = vec2<f32>(f32(fp4.x) + 0.5, f32(fp4.y) + 0.5);
    acc += godray_integrate(ro, ray_dir_from_pixel(px4, res_full), t_end4, /* j */ 0.0);
    wsum += 1.0;
  }

  let cur_lin = max(select(vec3<f32>(0.0), acc / wsum, wsum > 0.0), vec3<f32>(0.0));

  // Compress before temporal blending (matches your decompression in bilerp)
  let cur = cur_lin / (cur_lin + vec3<f32>(0.25));

  let hist = textureLoad(godray_hist_tex, hip, 0).xyz;

  let dmin = min(min(t_scene0, t_scene1), min(t_scene2, t_scene3));
  let dmax = max(max(t_scene0, t_scene1), max(t_scene2, t_scene3));
  let span = (dmax - dmin) / max(dmin, 1e-3);
  let edge = smoothstep(0.06, 0.30, span);

  // React widens clamp only; it does NOT reduce history weight.
  let delta = length(cur - hist);
  let react = smoothstep(0.03, 0.18, delta);

  // History weight only reduced by depth discontinuity.
  let stable = 1.0 - edge;

  // Wider clamp when changing so history can follow (prevents A/B flip-flop).
  let clamp_scale = mix(1.25, 2.5, react);
  let clamp_w = max(cur * clamp_scale, vec3<f32>(0.04));
  let hist_clamped = clamp(hist, cur - clamp_w, cur + clamp_w);

  // Always integrate at least a bit (kills “raw noise popping through”).
  let hist_w = clamp(0.28 + GODRAY_TS_LP_ALPHA * stable, 0.18, 0.94);

  let blended = mix(cur, hist_clamped, hist_w);

  textureStore(godray_out, hip, vec4<f32>(blended, 1.0));
}

fn tonemap_aces_luma(hdr: vec3<f32>) -> vec3<f32> {
  let w = vec3<f32>(0.2126, 0.7152, 0.0722);
  let l_in = max(dot(hdr, w), 1e-6);

  // ACES fit on scalar luminance
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  let l_out = clamp((l_in*(a*l_in + b)) / (l_in*(c*l_in + d) + e), 0.0, 1.0);

  // rescale RGB to preserve chroma
  return hdr * (l_out / l_in);
}

@compute @workgroup_size(8, 8, 1)
fn main_composite(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(out_img);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let ip = vec2<i32>(i32(gid.x), i32(gid.y));
  let base = textureLoad(color_tex, ip, 0).xyz;

  let px = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

  // Godrays upsample (depth-aware)
  let g  = godray_sample_bilerp(px);
  let gx = godray_sample_bilerp(px + vec2<f32>( 1.0, 0.0)) + godray_sample_bilerp(px + vec2<f32>(-1.0, 0.0));
  let gy = godray_sample_bilerp(px + vec2<f32>(0.0,  1.0)) + godray_sample_bilerp(px + vec2<f32>(0.0, -1.0));
  let gd = godray_sample_bilerp(px + vec2<f32>( 1.0,  1.0))
        + godray_sample_bilerp(px + vec2<f32>(-1.0,  1.0))
        + godray_sample_bilerp(px + vec2<f32>( 1.0, -1.0))
        + godray_sample_bilerp(px + vec2<f32>(-1.0, -1.0));

  let blur = (gx + gy + 0.7 * gd) / (4.0 + 0.7 * 4.0);

  var god_lin = max(g + COMPOSITE_SHARPEN * (g - blur), vec3<f32>(0.0));
  god_lin = max(god_lin - vec3<f32>(GODRAY_BLACK_LEVEL), vec3<f32>(0.0));

  // --- new knobs (from common.wgsl) ---
  god_lin = god_lin / (god_lin + vec3<f32>(GODRAY_KNEE_COMPOSITE));

  let d = textureLoad(depth_full, ip, 0).x;
  let god_far = smoothstep(GODRAY_FADE_NEAR, GODRAY_FADE_FAR, d);

  let god_scale = GODRAY_COMPOSITE_SCALE * mix(1.0, 0.25, god_far);

  var hdr = max(base + god_scale * god_lin, vec3<f32>(0.0));

  // --- Bloom (hue-preserving + distance-faded) ---
  let bloom_thresh = 1.4;
  let bloom_k      = 0.12;

  let bloom_k_eff = bloom_k * mix(1.0, 0.0, god_far);

  let b0 = bright_extract_hue(hdr, bloom_thresh);

  let ipx1 = vec2<i32>(clamp(ip.x + 2, 0, i32(dims.x) - 1), ip.y);
  let ipx0 = vec2<i32>(clamp(ip.x - 2, 0, i32(dims.x) - 1), ip.y);
  let ipy1 = vec2<i32>(ip.x, clamp(ip.y + 2, 0, i32(dims.y) - 1));
  let ipy0 = vec2<i32>(ip.x, clamp(ip.y - 2, 0, i32(dims.y) - 1));

  // Bloom taps: use current hdr neighborhood approx (avoid reintroducing old exp() godray path)
  let hx1 = max(textureLoad(color_tex, ipx1, 0).xyz, vec3<f32>(0.0));
  let hx0 = max(textureLoad(color_tex, ipx0, 0).xyz, vec3<f32>(0.0));
  let hy1 = max(textureLoad(color_tex, ipy1, 0).xyz, vec3<f32>(0.0));
  let hy0 = max(textureLoad(color_tex, ipy0, 0).xyz, vec3<f32>(0.0));

  let bloom = (b0
    + bright_extract_hue(hx1, bloom_thresh)
    + bright_extract_hue(hx0, bloom_thresh)
    + bright_extract_hue(hy1, bloom_thresh)
    + bright_extract_hue(hy0, bloom_thresh)) / 5.0;

  let bloom_max = 0.35 * max(hdr, vec3<f32>(0.0));
  hdr += bloom_k_eff * min(bloom, bloom_max);

  // --------------------------------------------------------------------------
  // Distance-safe saturation compensation (in HDR, before tonemap)
  // --------------------------------------------------------------------------
  let lum_w = vec3<f32>(0.2126, 0.7152, 0.0722);
  let l_hdr = max(dot(hdr, lum_w), 1e-6);
  let gray_hdr = vec3<f32>(l_hdr);

  let t_sat = smoothstep(30.0, 100.0, d);

  var sat_boost = 1.00 + 0.55 * t_sat;

  let hi = smoothstep(1.6, 6.0, l_hdr);
  sat_boost = mix(sat_boost, 1.0, 0.55 * hi);

  hdr = mix(gray_hdr, hdr, sat_boost);

  // --------------------------------------------------------------------------
  // Tonemap (luma-preserving)
  // --------------------------------------------------------------------------
  var c = tonemap_aces_luma(hdr * POST_EXPOSURE);
  c = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));

  c = pow(c, vec3<f32>(0.98));

  // Dither/grain before gamma to reduce banding
  let fi = f32(cam.frame_index & 1023u);
  let n = hash12(px * 1.7 + vec2<f32>(fi, 0.0)) - 0.5;
  c += vec3<f32>(n / 1024.0);

  let ldr = gamma_encode(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)));
  textureStore(out_img, ip, vec4<f32>(ldr, 1.0));
}
