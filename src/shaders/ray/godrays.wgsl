// src/shaders/ray/godrays.wgsl
//// --------------------------------------------------------------------------
//// Godrays: integration + quarter-res temporal pixel
//// --------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// 5-tap collapsed godray integration
// - One march out to t_end_max
// - Per-tap lightweight LP state + accumulation
// -----------------------------------------------------------------------------

struct Godray5 {
  g0: vec3<f32>,
  g1: vec3<f32>,
  g2: vec3<f32>,
  g3: vec3<f32>,
  g4: vec3<f32>,
};

fn godray_integrate_5(
  ro: vec3<f32>,
  rd: vec3<f32>,
  t_end0: f32,
  t_end1: f32,
  t_end2: f32,
  t_end3: f32,
  t_end4: f32,
  j0: f32,
  j1: f32,
  j2: f32,
  j3: f32,
  j4: f32,
  j_phase: f32
) -> Godray5 {
  let base = fog_density_godray();
  if (base <= 0.0) {
    return Godray5(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  }

  // Enabled taps
  let e0 = t_end0 > 0.0;
  let e1 = t_end1 > 0.0;
  let e2 = t_end2 > 0.0;
  let e3 = t_end3 > 0.0;
  let e4 = t_end4 > 0.0;

  let t_end_max = max(max(max(t_end0, t_end1), max(t_end2, t_end3)), t_end4);
  if (t_end_max <= 0.0) {
    return Godray5(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  }

  let costh = dot(rd, SUN_DIR);
  let phase = phase_blended(costh);

  // March spacing based on max distance (single loop)
  let N  = f32(GODRAY_STEPS_FAST);
  let dt = t_end_max / N;

  // LP coefficients
  let a_ts    = 1.0 - exp(-dt * 3.0);
  let a_shaft = 1.0 - exp(-dt * 5.0);

  // Per-tap running state
  var ts0: f32 = 1.0; var sh0: f32 = 0.0; var sum0 = vec3<f32>(0.0);
  var ts1: f32 = 1.0; var sh1: f32 = 0.0; var sum1 = vec3<f32>(0.0);
  var ts2: f32 = 1.0; var sh2: f32 = 0.0; var sum2 = vec3<f32>(0.0);
  var ts3: f32 = 1.0; var sh3: f32 = 0.0; var sum3 = vec3<f32>(0.0);
  var ts4: f32 = 1.0; var sh4: f32 = 0.0; var sum4 = vec3<f32>(0.0);

  for (var i: u32 = 0u; i < GODRAY_STEPS_FAST; i = i + 1u) {
    let ti = max((f32(i) + 0.5 + j_phase) * dt, 0.0);
    if (ti <= 0.0) { continue; }

    // Shared expensive work at this ti
    let p  = ro + rd * ti;

    let Tv = fog_transmittance_godray(ro, rd, ti);
    if (Tv < GODRAY_TV_CUTOFF) { break; }

    let Ts_geom = sun_transmittance_geom_only(p, SUN_DIR);
    let Tc      = cloud_sun_transmittance_fast(p, SUN_DIR);

    let Tc_vol  = mix(1.0, Tc, CLOUD_GODRAY_W);
    let Ts_soft = pow(clamp(Ts_geom, 0.0, 1.0), 0.75);

    // Height falloff + density at this ti
    let haze_ramp = 1.0 - exp(-ti / GODRAY_HAZE_NEAR_FADE);
    let hfall = GODRAY_SCATTER_HEIGHT_FALLOFF;
    let hmin  = GODRAY_SCATTER_MIN_FRAC;
    let height_term = max(exp(-hfall * p.y), hmin);
    let dens = base * height_term;

    let common_factor = (SUN_COLOR * SUN_INTENSITY)
      * (dens * dt) * Tv * phase
      * Tc_vol
      * 0.70;

    let mu = dot(rd, SUN_DIR);
    let view_gate = 0.10 + 0.90 * smoothstep(0.0, 0.20, mu);

    // ---- Tap 0
    if (e0) {
      let tcut = ti + j0 * dt;
      let a = (tcut <= t_end0);
      let wa = select(0.0, 1.0, a);

      if (wa > 0.0) {
        let ts_prev = ts0;
        ts0 = mix(ts0, Ts_soft, a_ts);

        let dTs = max(0.0, ts_prev - Ts_soft);

        var shaft = smoothstep(GODRAY_EDGE0, GODRAY_EDGE1, dTs);
        shaft *= (1.0 - Ts_soft);
        shaft = pow(clamp(shaft, 0.0, 1.0), 0.55);

        sh0 = mix(sh0, shaft, a_shaft);
        shaft = sh0;

        // ---- NEW: edge-only energy boost (uses low-passed shaft)
        let edge_boost = 1.0 + GODRAY_EDGE_ENERGY_BOOST * shaft;

        let shaft_sun_gate = smoothstep(0.35, 0.80, ts0);

        // (keeping your current tap0 w-shaping)
        let haze = GODRAY_BASE_HAZE * haze_ramp * pow(ts0, 3.5);
        let haze_cap = 0.08;
        let haze_floor = min(haze, haze_cap);

        let shaft_gain = GODRAY_SHAFT_GAIN;
        let shaft_term = clamp(shaft * shaft_sun_gate * shaft_gain, 0.0, 1.0);

        let w = clamp(haze_floor + (1.0 - haze_floor) * shaft_term, 0.0, 1.0);

        sum0 += common_factor * view_gate * edge_boost * (ts0 * w);
      }
    }

    // ---- Tap 1
    if (e1) {
      let tcut = ti + j1 * dt;
      let a = (tcut <= t_end1);
      let wa = select(0.0, 1.0, a);

      if (wa > 0.0) {
        let ts_prev = ts1;
        ts1 = mix(ts1, Ts_soft, a_ts);

        let dTs = max(0.0, ts_prev - Ts_soft);

        var shaft = smoothstep(GODRAY_EDGE0, GODRAY_EDGE1, dTs);
        shaft *= (1.0 - Ts_soft);
        shaft = pow(clamp(shaft, 0.0, 1.0), 0.55);

        sh1 = mix(sh1, shaft, a_shaft);
        shaft = sh1;

        // ---- NEW
        let edge_boost = 1.0 + GODRAY_EDGE_ENERGY_BOOST * shaft;

        let shaft_sun_gate = smoothstep(0.35, 0.80, ts1);
        let haze = GODRAY_BASE_HAZE * haze_ramp * pow(ts1, 2.0);

        let w_raw = haze + (1.0 - haze) * (shaft * shaft_sun_gate);
        let w = clamp(w_raw, 0.0, 1.0);

        sum1 += common_factor * edge_boost * (ts1 * w);
      }
    }

    // ---- Tap 2
    if (e2) {
      let tcut = ti + j2 * dt;
      let a = (tcut <= t_end2);
      let wa = select(0.0, 1.0, a);

      if (wa > 0.0) {
        let ts_prev = ts2;
        ts2 = mix(ts2, Ts_soft, a_ts);

        let dTs = max(0.0, ts_prev - Ts_soft);

        var shaft = smoothstep(GODRAY_EDGE0, GODRAY_EDGE1, dTs);
        shaft *= (1.0 - Ts_soft);
        shaft = pow(clamp(shaft, 0.0, 1.0), 0.55);

        sh2 = mix(sh2, shaft, a_shaft);
        shaft = sh2;

        // ---- NEW
        let edge_boost = 1.0 + GODRAY_EDGE_ENERGY_BOOST * shaft;

        let shaft_sun_gate = smoothstep(0.35, 0.80, ts2);
        let haze = GODRAY_BASE_HAZE * haze_ramp * pow(ts2, 2.0);

        let w_raw = haze + (1.0 - haze) * (shaft * shaft_sun_gate);
        let w = clamp(w_raw, 0.0, 1.0);

        sum2 += common_factor * edge_boost * (ts2 * w);
      }
    }

    // ---- Tap 3
    if (e3) {
      let tcut = ti + j3 * dt;
      let a = (tcut <= t_end3);
      let wa = select(0.0, 1.0, a);

      if (wa > 0.0) {
        let ts_prev = ts3;
        ts3 = mix(ts3, Ts_soft, a_ts);

        let dTs = max(0.0, ts_prev - Ts_soft);

        var shaft = smoothstep(GODRAY_EDGE0, GODRAY_EDGE1, dTs);
        shaft *= (1.0 - Ts_soft);
        shaft = pow(clamp(shaft, 0.0, 1.0), 0.55);

        sh3 = mix(sh3, shaft, a_shaft);
        shaft = sh3;

        // ---- NEW
        let edge_boost = 1.0 + GODRAY_EDGE_ENERGY_BOOST * shaft;

        let shaft_sun_gate = smoothstep(0.35, 0.80, ts3);
        let haze = GODRAY_BASE_HAZE * haze_ramp * pow(ts3, 2.0);

        let w_raw = haze + (1.0 - haze) * (shaft * shaft_sun_gate);
        let w = clamp(w_raw, 0.0, 1.0);

        sum3 += common_factor * edge_boost * (ts3 * w);
      }
    }

    // ---- Tap 4 (center)
    if (e4) {
      let tcut = ti + j4 * dt;
      let a = (tcut <= t_end4);
      let wa = select(0.0, 1.0, a);

      if (wa > 0.0) {
        let ts_prev = ts4;
        ts4 = mix(ts4, Ts_soft, a_ts);

        let dTs = max(0.0, ts_prev - Ts_soft);

        var shaft = smoothstep(GODRAY_EDGE0, GODRAY_EDGE1, dTs);
        shaft *= (1.0 - Ts_soft);
        shaft = pow(clamp(shaft, 0.0, 1.0), 0.55);

        sh4 = mix(sh4, shaft, a_shaft);
        shaft = sh4;

        // ---- NEW
        let edge_boost = 1.0 + GODRAY_EDGE_ENERGY_BOOST * shaft;

        let shaft_sun_gate = smoothstep(0.35, 0.80, ts4);
        let haze = GODRAY_BASE_HAZE * haze_ramp * pow(ts4, 2.0);

        let w_raw = haze + (1.0 - haze) * (shaft * shaft_sun_gate);
        let w = clamp(w_raw, 0.0, 1.0);

        sum4 += common_factor * edge_boost * (ts4 * w);
      }
    }
  }

  var g0 = sum0 * GODRAY_ENERGY_BOOST;
  var g1 = sum1 * GODRAY_ENERGY_BOOST;
  var g2 = sum2 * GODRAY_ENERGY_BOOST;
  var g3 = sum3 * GODRAY_ENERGY_BOOST;
  var g4 = sum4 * GODRAY_ENERGY_BOOST;

  g0 = g0 / (g0 + vec3<f32>(GODRAY_KNEE_INTEGRATE));
  g1 = g1 / (g1 + vec3<f32>(GODRAY_KNEE_INTEGRATE));
  g2 = g2 / (g2 + vec3<f32>(GODRAY_KNEE_INTEGRATE));
  g3 = g3 / (g3 + vec3<f32>(GODRAY_KNEE_INTEGRATE));
  g4 = g4 / (g4 + vec3<f32>(GODRAY_KNEE_INTEGRATE));

  return Godray5(g0, g1, g2, g3, g4);
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

  // 5 depth taps (same as before)
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

  // Jitter as before
  let j0 = 0.20 * (hash12(qpx * J0_SCALE) - 0.5);
  let j1 = 0.20 * (hash12(qpx * J1_SCALE + vec2<f32>(11.0, 3.0)) - 0.5);
  let j2 = 0.20 * (hash12(qpx * J2_SCALE + vec2<f32>(5.0, 17.0)) - 0.5);
  let j3 = 0.20 * (hash12(qpx * J3_SCALE + vec2<f32>(23.0, 29.0)) - 0.5);

  // Phase jitter for raymarch sample positions (NOT cutoff jitter).
  // In [-0.5 .. +0.5]. Keep it small and stable-ish.
  let jf = f32(cam.frame_index & 255u);

  // Option A: fully stable per pixel (best for stability, less dither):
  // let j_phase = hash12(qpx * 0.91 + vec2<f32>(31.7, 12.3)) - 0.5;

  // Option B: slowly varying with time/frame (kills banding more, relies on TAA):
  let j_phase = hash12(qpx * 0.91 + vec2<f32>(31.7, 12.3) + vec2<f32>(jf * 0.07, jf * 0.03)) - 0.5;

  // Load depths (same)
  let t_scene0 = textureLoad(depth_tex, fp0, 0).x;
  let t_scene1 = textureLoad(depth_tex, fp1, 0).x;
  let t_scene2 = textureLoad(depth_tex, fp2, 0).x;
  let t_scene3 = textureLoad(depth_tex, fp3, 0).x;
  let t_scene4 = textureLoad(depth_tex, fp4, 0).x;

  // Early-out: if no godray fog, just blend history toward 0 quickly
  let fog_ok = fog_density_godray() > 0.0;

  // Quantize end distances (same)
  let qstep = 0.1;
  let t_end0 = min(floor(t_scene0 / qstep) * qstep, GODRAY_MAX_DIST);
  let t_end1 = min(floor(t_scene1 / qstep) * qstep, GODRAY_MAX_DIST);
  let t_end2 = min(floor(t_scene2 / qstep) * qstep, GODRAY_MAX_DIST);
  let t_end3 = min(floor(t_scene3 / qstep) * qstep, GODRAY_MAX_DIST);
  let t_end4 = min(floor(t_scene4 / qstep) * qstep, GODRAY_MAX_DIST);

  // -------------------------------------------------------------------------
  // NEW: compute ray direction ONCE for the block center and reuse it
  // Center of the 4x4 block (base + 2,2) in full-res pixel coords.
  // -------------------------------------------------------------------------
  let res_full = vec2<f32>(f32(fdims.x), f32(fdims.y));
  
  // ~sub-texel in full-res pixels
  let j_rd = vec2<f32>(
    hash12(qpx + vec2<f32>(1.7, 9.2)) - 0.5,
    hash12(qpx + vec2<f32>(8.3, 2.1)) - 0.5
  );
  let center_px = vec2<f32>(f32(base_x + 2) + 0.5, f32(base_y + 2) + 0.5) + 0.75 * j_rd;
  let rd_center = ray_dir_from_pixel(center_px, res_full);

  // -------------------------------------------------------------------------

  var acc = vec3<f32>(0.0);
  var wsum = 0.0;

  if (fog_ok) {
    let g5 = godray_integrate_5(
      ro, rd_center,
      t_end0, t_end1, t_end2, t_end3, t_end4,
      j0, j1, j2, j3, 0.0,
      j_phase
    );

    if (t_end0 > 0.0) { acc += g5.g0; wsum += 1.0; }
    if (t_end1 > 0.0) { acc += g5.g1; wsum += 1.0; }
    if (t_end2 > 0.0) { acc += g5.g2; wsum += 1.0; }
    if (t_end3 > 0.0) { acc += g5.g3; wsum += 1.0; }
    if (t_end4 > 0.0) { acc += g5.g4; wsum += 1.0; }
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
