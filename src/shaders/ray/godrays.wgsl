// src/shaders/ray/godrays.wgsl
//// --------------------------------------------------------------------------
//// Godrays: integration + quarter-res temporal pixel
//// --------------------------------------------------------------------------

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

    sum += (SUN_COLOR * SUN_INTENSITY)
     * (dens * dt) * Tv * ts_lp * phase * w
     * Tc_vol
     * 0.70;
  }

  var gr = sum * GODRAY_ENERGY_BOOST;
  gr = gr / (gr + vec3<f32>(GODRAY_KNEE_INTEGRATE));
  return gr;
}

fn compute_godray_quarter_pixel(
  gid: vec2<u32>,
  depth_tex: texture_2d<f32>,
  godray_hist_tex: texture_2d<f32>
) -> vec3<f32> {
  let fdims = textureDimensions(depth_tex);
  let ro = cam.cam_pos.xyz;

  let hip = vec2<i32>(i32(gid.x), i32(gid.y));
  let qpx = vec2<f32>(f32(gid.x), f32(gid.y));

  let base_x = i32(gid.x) * GODRAY_BLOCK_SIZE;
  let base_y = i32(gid.y) * GODRAY_BLOCK_SIZE;

  // 5 taps
  let fp0 = vec2<i32>(clamp(base_x + 1, 0, i32(fdims.x) - 1),
                      clamp(base_y + 1, 0, i32(fdims.y) - 1));
  let fp1 = vec2<i32>(clamp(base_x + 3, 0, i32(fdims.x) - 1),
                      clamp(base_y + 1, 0, i32(fdims.y) - 1));
  let fp2 = vec2<i32>(clamp(base_x + 1, 0, i32(fdims.x) - 1),
                      clamp(base_y + 3, 0, i32(fdims.y) - 1));
  let fp3 = vec2<i32>(clamp(base_x + 3, 0, i32(fdims.x) - 1),
                      clamp(base_y + 3, 0, i32(fdims.y) - 1));
  let fp4 = vec2<i32>(clamp(base_x + 2, 0, i32(fdims.x) - 1),
                      clamp(base_y + 1, 0, i32(fdims.y) - 1));

  let res_full = vec2<f32>(f32(fdims.x), f32(fdims.y));

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

  let qstep = 0.1;

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
    acc += godray_integrate(ro, ray_dir_from_pixel(px4, res_full), t_end4, 0.0);
    wsum += 1.0;
  }

  let cur_lin = max(select(vec3<f32>(0.0), acc / wsum, wsum > 0.0), vec3<f32>(0.0));
  let cur     = cur_lin / (cur_lin + vec3<f32>(0.25)); // compress

  let hist = textureLoad(godray_hist_tex, hip, 0).xyz;

  let dmin = min(min(t_scene0, t_scene1), min(t_scene2, t_scene3));
  let dmax = max(max(t_scene0, t_scene1), max(t_scene2, t_scene3));
  let span = (dmax - dmin) / max(dmin, 1e-3);
  let edge = smoothstep(0.06, 0.30, span);

  let delta = length(cur - hist);
  let react = smoothstep(0.03, 0.18, delta);

  let stable = 1.0 - edge;

  let clamp_scale = mix(1.25, 2.5, react);
  let clamp_w = max(cur * clamp_scale, vec3<f32>(0.04));
  let hist_clamped = clamp(hist, cur - clamp_w, cur + clamp_w);

  let hist_w = clamp(0.28 + GODRAY_TS_LP_ALPHA * stable, 0.18, 0.94);
  return mix(cur, hist_clamped, hist_w);
}

fn godray_decompress(cur: vec3<f32>) -> vec3<f32> {
  // must match quarter pass compression
  let k = 0.25;
  let one = vec3<f32>(1.0);
  let denom = max(one - cur, vec3<f32>(1e-4));
  return (k * cur) / denom;
}

// Sample godray quarter-res texture in normalized UV, hardware bilinear
fn godray_sample_linear(
  uv: vec2<f32>,
  godray_tex: texture_2d<f32>,
  godray_samp: sampler
) -> vec3<f32> {
  let c = textureSampleLevel(godray_tex, godray_samp, uv, 0.0).xyz;
  return godray_decompress(c);
}
