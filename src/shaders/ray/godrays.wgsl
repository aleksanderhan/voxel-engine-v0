






fn approx_pow_0p75(x: f32) -> f32 {
  let y = clamp(x, 0.0, 1.0);
  
  return mix(y, sqrt(y), 0.5);
}

fn approx_pow_0p55(x: f32) -> f32 {
  let y = clamp(x, 0.0, 1.0);
  let s = sqrt(y);          
  
  return mix(s, y, 0.20);
}



fn exp_neg_fast(x_in: f32) -> f32 {
  let x = max(x_in, 0.0);
  let x2 = x * x;
  
  return 1.0 / (1.0 + x + 0.48 * x2);
}

fn godray_steps_for_tend(t_end: f32) -> u32 {
  
  let raw_f = ceil(t_end * GODRAY_STEPS_PER_METER);

  
  var s: u32 = u32(clamp(raw_f, f32(GODRAY_MIN_STEPS), f32(GODRAY_STEPS_FAST)));

  
  
  s = ((s + (GODRAY_STEP_Q - 1u)) / GODRAY_STEP_Q) * GODRAY_STEP_Q;

  
  return min(s, GODRAY_STEPS_FAST);
}

fn godray_integrate_1(
  ro: vec3<f32>,
  rd: vec3<f32>,
  t_end: f32,
  j: f32,
  j_phase: f32
) -> vec3<f32> {
  let base = fog_density_godray();
  if (base <= 0.0 || t_end <= 0.0) { return vec3<f32>(0.0); }

  let phase = phase_blended(dot(rd, SUN_DIR));
  let mu = dot(rd, SUN_DIR);
  let view_gate = 0.10 + 0.90 * smoothstep(0.0, 0.20, mu);

  let steps: u32 = godray_steps_for_tend(t_end);
  let dt = t_end / f32(steps);

  let a_ts    = 1.0 - exp(-dt * 3.0);
  let a_shaft = 1.0 - exp(-dt * 5.0);

  var ts: f32 = 1.0;
  var sh: f32 = 0.0;
  var sum = vec3<f32>(0.0);

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  let k  = FOG_HEIGHT_FALLOFF;
  let y0 = ro.y;
  let dy = rd.y;

  
  
  let f = clamp(0.5 + j_phase, 0.0, 1.0);
  let one_m_f = 1.0 - f;

  var Tv_acc: f32 = 1.0;

  let horiz = abs(dy) < 1e-4;

  
  var dens_const: f32 = 0.0;
  var trans_to_sample_h: f32 = 1.0;
  var trans_to_end_h: f32 = 1.0;

  
  var inv_kdy: f32 = 0.0;
  var B: f32 = 0.0;
  var r_f: f32 = 1.0;
  var r_rem: f32 = 1.0;

  if (horiz) {
    
    let a = exp(-k * y0);             
    dens_const = base * a;

    
    trans_to_sample_h = exp_neg_fast(dens_const * dt * f);
    trans_to_end_h    = exp_neg_fast(dens_const * dt * one_m_f);
  } else {
    inv_kdy = 1.0 / (k * dy);

    
    B = exp(-k * y0);                 

    
    r_f   = exp(-k * dy * dt * f);        
    r_rem = exp(-k * dy * dt * one_m_f);  
  }
  

  var Ts_geom: f32 = 1.0;
  for (var i: u32 = 0u; i < steps; i = i + 1u) {
    let ti = (f32(i) + 0.5 + j_phase) * dt;

    
    if (ti + j * dt > t_end) { break; }

    let p  = ro + rd * ti;

    
    if (horiz) {
      
      Tv_acc *= trans_to_sample_h;
    } else {
      
      let B_s = B * r_f;
      let d_od = base * (B - B_s) * inv_kdy;          
      Tv_acc *= exp_neg_fast(max(d_od, 0.0));
    }

    let Tv = Tv_acc;
    if (Tv < GODRAY_TV_CUTOFF) { break; }
    

    if ((i & 3u) == 0u) {           
      Ts_geom = sun_transmittance_geom_only(p, SUN_DIR);
    }
    let Tc      = cloud_sun_transmittance_godray(p, SUN_DIR);

    let Tc_vol  = mix(1.0, Tc, CLOUD_GODRAY_W);

    
    let Ts_clamped = clamp(Ts_geom, 0.0, 1.0);
    let Ts_soft = approx_pow_0p75(Ts_clamped);

    
    let haze_ramp = 1.0 - exp(-ti / GODRAY_HAZE_NEAR_FADE);
    

    let height_term = max(exp(-GODRAY_SCATTER_HEIGHT_FALLOFF * p.y), GODRAY_SCATTER_MIN_FRAC);
    let dens = base * height_term;

    let common_factor = (SUN_COLOR * SUN_INTENSITY)
      * (dens * dt) * Tv * phase
      * Tc_vol
      * 0.70;

    
    let ts_prev = ts;
    ts = mix(ts, Ts_soft, a_ts);

    let dTs = max(0.0, ts_prev - Ts_soft);

    var shaft = smoothstep(GODRAY_EDGE0, GODRAY_EDGE1, dTs);
    shaft *= (1.0 - Ts_soft);
    shaft = approx_pow_0p55(clamp(shaft, 0.0, 1.0));

    sh = mix(sh, shaft, a_shaft);
    shaft = sh;

    let edge_boost = 1.0 + GODRAY_EDGE_ENERGY_BOOST * shaft;

    let shaft_sun_gate = smoothstep(0.35, 0.80, ts);
    let haze = GODRAY_BASE_HAZE * haze_ramp * (ts * ts); 

    let w = clamp(haze + (1.0 - haze) * (shaft * shaft_sun_gate), 0.0, 1.0);
    sum += common_factor * view_gate * edge_boost * (ts * w);

    
    if (horiz) {
      
      Tv_acc *= trans_to_end_h;
    } else {
      
      let B_s  = B * r_f;
      let B_e  = B_s * r_rem;
      let d_od2 = base * (B_s - B_e) * inv_kdy;        
      Tv_acc *= exp_neg_fast(max(d_od2, 0.0));
      B = B_e;
    }
    
  }

  var g = sum * GODRAY_ENERGY_BOOST;
  let knee = vec3<f32>(GODRAY_KNEE_INTEGRATE);
  g = g / (g + knee);
  return g;
}






fn godray_decompress(c: vec3<f32>) -> vec3<f32> {
  
  let k = 0.25;
  let denom = max(vec3<f32>(1.0) - c, vec3<f32>(1e-4));
  return (k * c) / denom;
}

fn godray_compress(lin: vec3<f32>) -> vec3<f32> {
  
  let k = 0.25;
  return lin / (lin + vec3<f32>(k));
}





fn prev_uv_from_world(p_ws: vec3<f32>) -> vec2<f32> {
  let clip = cam.prev_view_proj * vec4<f32>(p_ws, 1.0);
  let invw = 1.0 / max(clip.w, 1e-6);
  let ndc  = clip.xy * invw;          
  return ndc * 0.5 + vec2<f32>(0.5);  
}

fn in_unit_square(uv: vec2<f32>) -> bool {
  return all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0));
}





fn compute_godray_pixel(
  gid: vec2<u32>,
  depth_tex: texture_2d<f32>,
  godray_hist_tex: texture_2d<f32>,
  godray_hist_samp: sampler
) -> vec4<f32> {
  let fdims_u = textureDimensions(depth_tex);
  let fdims_i = vec2<i32>(i32(fdims_u.x), i32(fdims_u.y));
  let res_full = vec2<f32>(f32(fdims_u.x), f32(fdims_u.y));

  let ro = cam.cam_pos.xyz;

  let ip = vec2<i32>(i32(gid.x), i32(gid.y));
  let qpx = vec2<f32>(f32(gid.x), f32(gid.y));

  
  let fi = f32(cam.frame_index & 1023u);
  let fj = vec2<f32>(fi, fi * 1.371);

  
  let ip_l = vec2<i32>(clamp(ip.x - 1, 0, fdims_i.x - 1), ip.y);
  let ip_r = vec2<i32>(clamp(ip.x + 1, 0, fdims_i.x - 1), ip.y);
  let ip_u = vec2<i32>(ip.x, clamp(ip.y - 1, 0, fdims_i.y - 1));
  let ip_d = vec2<i32>(ip.x, clamp(ip.y + 1, 0, fdims_i.y - 1));

  let t_c = textureLoad(depth_tex, ip,   0).x;
  let t_l = textureLoad(depth_tex, ip_l, 0).x;
  let t_r = textureLoad(depth_tex, ip_r, 0).x;
  let t_u = textureLoad(depth_tex, ip_u, 0).x;
  let t_d = textureLoad(depth_tex, ip_d, 0).x;

  let dmin = min(t_c, min(min(t_l, t_r), min(t_u, t_d)));
  let dmax = max(t_c, max(max(t_l, t_r), max(t_u, t_d)));

  let span   = (dmax - dmin) / max(dmin, 1e-3);
  let stable = 1.0 - smoothstep(0.06, 0.30, span);

  let t_hist = dmin;

  
  let j4      = 0.20 * (hash12(qpx * J0_SCALE + fj) - 0.5);
  let j_phase =        (hash12(qpx * 0.91 + fj * 0.73 + vec2<f32>(31.7, 12.3)) - 0.5);

  
  let qstep = 0.03;
  let dq = (hash12(qpx * 1.37 + fj * 1.19 + vec2<f32>(9.2, 1.1)) - 0.5) * qstep;

  
  let j_rd = vec2<f32>(
    hash12(qpx + fj + vec2<f32>(1.7, 9.2)) - 0.5,
    hash12(qpx + fj * 1.13 + vec2<f32>(8.3, 2.1)) - 0.5
  );

  
  let t_end = min(floor((t_c + dq) / qstep) * qstep, GODRAY_MAX_DIST);

  let fog_ok = fog_density_godray() > 0.0;

  
  let px_center = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5) + 0.35 * j_rd;
  let rd = ray_dir_from_pixel(px_center, res_full);

  
  var cur_lin = vec3<f32>(0.0);
  if (fog_ok && t_end > 0.0) {
    cur_lin = godray_integrate_1(ro, rd, t_end, j4, j_phase);
  }

  
  let p_ws    = ro + rd * t_hist;
  let uv_prev = prev_uv_from_world(p_ws);

  var hist_lin   = cur_lin;
  var hist_valid = 0.0;

  let hd_u = textureDimensions(godray_hist_tex);
  let hdf  = vec2<f32>(f32(hd_u.x), f32(hd_u.y));
  let uv_cur = (vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5)) / hdf;

  if (in_unit_square(uv_prev)) {
    let h = textureSampleLevel(godray_hist_tex, godray_hist_samp, uv_prev, 0.0);

    let hist_depth = h.w;
    hist_lin = godray_decompress(h.xyz);

    let rel = abs(hist_depth - t_hist) / max(t_hist, 1e-3);
    let depth_ok   = 1.0 - smoothstep(0.08, 0.20, rel);
    let depth_sane = select(0.0, 1.0, hist_depth > 1e-3);

    let vel_px = length((uv_prev - uv_cur) * hdf);
    let motion_ok = 1.0 - smoothstep(0.75, 2.50, vel_px);

    hist_valid = depth_ok * depth_sane * motion_ok;
  }

  
  let delta_lin = length(cur_lin - hist_lin);
  let energy    = max(length(cur_lin), 1e-3);
  let delta_rel = delta_lin / (0.05 + energy);
  let react     = smoothstep(0.10, 0.45, delta_rel);

  
  let clamp_scale = mix(0.55, 0.18, stable);
  let clamp_w     = max(cur_lin * clamp_scale, vec3<f32>(0.015));
  let hist_clamped = clamp(hist_lin, cur_lin - clamp_w, cur_lin + clamp_w);

  
  let hist_w_base = mix(0.25, 0.92, stable);
  let hist_w      = mix(hist_w_base, 0.03, react) * hist_valid;

  let out_lin = mix(cur_lin, hist_clamped, hist_w);

  let out_c = godray_compress(max(out_lin, vec3<f32>(0.0)));
  return vec4<f32>(out_c, t_hist);
}

fn godray_sample_linear(
  uv: vec2<f32>,
  godray_tex: texture_2d<f32>,
  godray_samp: sampler
) -> vec3<f32> {
  
  let c = textureSampleLevel(godray_tex, godray_samp, uv, 0.0).xyz;
  return godray_decompress(c);
}
