//// --------------------------------------------------------------------------
//// Composite helpers (godray upsample + tonemap)
//// --------------------------------------------------------------------------

fn bright_extract_hue(x: vec3<f32>, thresh: f32) -> vec3<f32> {
  let lum_w = vec3<f32>(0.2126, 0.7152, 0.0722);
  let l = dot(x, lum_w);

  let over = max(l - thresh, 0.0);
  let w = over / max(l, 1e-4);
  return x * w;
}

fn gamma_encode(x: vec3<f32>) -> vec3<f32> {
  return pow(x, vec3<f32>(1.0 / 2.2));
}

fn composite_pixel_mapped(
  ip_render: vec2<i32>,
  px_render: vec2<f32>,
  color_tex: texture_2d<f32>,
  godray_tex: texture_2d<f32>,
  godray_samp: sampler,
  depth_full: texture_2d<f32>,
  depth_render: texture_2d<f32>,
  normal_render: texture_2d<f32>
) -> vec4<f32> {
  // --- render dims (color buffer) ---
  let rd_u = textureDimensions(color_tex);
  let rd_i = vec2<i32>(i32(rd_u.x), i32(rd_u.y));
  let rd_f = vec2<f32>(f32(rd_u.x), f32(rd_u.y));

  // Clamp render ip
  let ip_r = vec2<i32>(
    clamp(ip_render.x, 0, rd_i.x - 1),
    clamp(ip_render.y, 0, rd_i.y - 1)
  );

  // --- full dims (depth buffer) ---
  let fd_u = textureDimensions(depth_full);
  let fd_i = vec2<i32>(i32(fd_u.x), i32(fd_u.y));
  let fd_f = vec2<f32>(f32(fd_u.x), f32(fd_u.y));

  // Map render pixel -> full pixel (important!)
  // px_render is in render pixel coords; convert to full pixel coords.
  let px_full = px_render * (fd_f / rd_f);

  let ip_f = vec2<i32>(
    clamp(i32(floor(px_full.x)), 0, fd_i.x - 1),
    clamp(i32(floor(px_full.y)), 0, fd_i.y - 1)
  );

  // UV for godray sampling MUST match the full-res screen mapping
  let uv_full = (px_full + vec2<f32>(0.5)) / fd_f;

  // Depth edge weights MUST be full-res taps too
  let d0 = textureLoad(depth_full, ip_f, 0).x;

  // Depth/normal-aware upsample of base color
  let px_render_clamped = vec2<f32>(
    clamp(px_render.x, 0.0, rd_f.x - 1.0),
    clamp(px_render.y, 0.0, rd_f.y - 1.0)
  );
  let base_ip0 = vec2<i32>(
    clamp(i32(floor(px_render_clamped.x)), 0, rd_i.x - 1),
    clamp(i32(floor(px_render_clamped.y)), 0, rd_i.y - 1)
  );
  let base_ip1 = vec2<i32>(
    clamp(base_ip0.x + 1, 0, rd_i.x - 1),
    clamp(base_ip0.y + 1, 0, rd_i.y - 1)
  );

  let frac = fract(px_render_clamped);
  let ref_normal_raw = textureLoad(normal_render, ip_r, 0);
  let ref_valid = ref_normal_raw.w > 0.5;
  let ref_n = normalize(ref_normal_raw.xyz * 2.0 - vec3<f32>(1.0));

  var base = vec3<f32>(0.0);
  var wsum = 0.0;
  let tol = 0.02 + 0.06 * smoothstep(10.0, 80.0, d0);

  let ip_s00 = vec2<i32>(base_ip0.x, base_ip0.y);
  let ip_s10 = vec2<i32>(base_ip1.x, base_ip0.y);
  let ip_s01 = vec2<i32>(base_ip0.x, base_ip1.y);
  let ip_s11 = vec2<i32>(base_ip1.x, base_ip1.y);

  let w00 = (1.0 - frac.x) * (1.0 - frac.y);
  let w10 = frac.x * (1.0 - frac.y);
  let w01 = (1.0 - frac.x) * frac.y;
  let w11 = frac.x * frac.y;

  // sample 00
  var d_s = textureLoad(depth_render, ip_s00, 0).x;
  var d_w = 1.0 - smoothstep(0.0, tol, abs(d_s - d0));
  var n_raw = textureLoad(normal_render, ip_s00, 0);
  var n_valid = n_raw.w > 0.5;
  var n_w = 1.0;
  if (ref_valid && n_valid) {
    let n_s = normalize(n_raw.xyz * 2.0 - vec3<f32>(1.0));
    n_w = pow(max(dot(ref_n, n_s), 0.0), 4.0);
  }
  var w = w00 * d_w * n_w * select(0.0, 1.0, n_valid);
  var c_sample = textureLoad(color_tex, ip_s00, 0).xyz;
  base += c_sample * w;
  wsum += w;

  // sample 10
  d_s = textureLoad(depth_render, ip_s10, 0).x;
  d_w = 1.0 - smoothstep(0.0, tol, abs(d_s - d0));
  n_raw = textureLoad(normal_render, ip_s10, 0);
  n_valid = n_raw.w > 0.5;
  n_w = 1.0;
  if (ref_valid && n_valid) {
    let n_s = normalize(n_raw.xyz * 2.0 - vec3<f32>(1.0));
    n_w = pow(max(dot(ref_n, n_s), 0.0), 4.0);
  }
  w = w10 * d_w * n_w * select(0.0, 1.0, n_valid);
  c_sample = textureLoad(color_tex, ip_s10, 0).xyz;
  base += c_sample * w;
  wsum += w;

  // sample 01
  d_s = textureLoad(depth_render, ip_s01, 0).x;
  d_w = 1.0 - smoothstep(0.0, tol, abs(d_s - d0));
  n_raw = textureLoad(normal_render, ip_s01, 0);
  n_valid = n_raw.w > 0.5;
  n_w = 1.0;
  if (ref_valid && n_valid) {
    let n_s = normalize(n_raw.xyz * 2.0 - vec3<f32>(1.0));
    n_w = pow(max(dot(ref_n, n_s), 0.0), 4.0);
  }
  w = w01 * d_w * n_w * select(0.0, 1.0, n_valid);
  c_sample = textureLoad(color_tex, ip_s01, 0).xyz;
  base += c_sample * w;
  wsum += w;

  // sample 11
  d_s = textureLoad(depth_render, ip_s11, 0).x;
  d_w = 1.0 - smoothstep(0.0, tol, abs(d_s - d0));
  n_raw = textureLoad(normal_render, ip_s11, 0);
  n_valid = n_raw.w > 0.5;
  n_w = 1.0;
  if (ref_valid && n_valid) {
    let n_s = normalize(n_raw.xyz * 2.0 - vec3<f32>(1.0));
    n_w = pow(max(dot(ref_n, n_s), 0.0), 4.0);
  }
  w = w11 * d_w * n_w * select(0.0, 1.0, n_valid);
  c_sample = textureLoad(color_tex, ip_s11, 0).xyz;
  base += c_sample * w;
  wsum += w;

  if (wsum > 0.0) {
    base /= wsum;
  } else {
    base = textureLoad(color_tex, ip_r, 0).xyz;
  }

  var god_lin = vec3<f32>(0.0);
  var god_far: f32 = 0.0;
  var god_scale: f32 = 0.0;

  if (ENABLE_GODRAYS) {
    // Godray taps in godray texel units
    let gd_u = textureDimensions(godray_tex);
    let gd_f = vec2<f32>(f32(gd_u.x), f32(gd_u.y));
    let du = vec2<f32>(1.0 / gd_f.x, 0.0);
    let dv = vec2<f32>(0.0, 1.0 / gd_f.y);

    // Sample godrays in the correct UV space
    let gC = godray_sample_linear(uv_full,       godray_tex, godray_samp);
    let gE = godray_sample_linear(uv_full + du,  godray_tex, godray_samp);
    let gW = godray_sample_linear(uv_full - du,  godray_tex, godray_samp);
    let gN = godray_sample_linear(uv_full + dv,  godray_tex, godray_samp);
    let gS = godray_sample_linear(uv_full - dv,  godray_tex, godray_samp);

    let ipE = vec2<i32>(min(ip_f.x + 1, fd_i.x - 1), ip_f.y);
    let ipW = vec2<i32>(max(ip_f.x - 1, 0),          ip_f.y);
    let ipN = vec2<i32>(ip_f.x, min(ip_f.y + 1, fd_i.y - 1));
    let ipS = vec2<i32>(ip_f.x, max(ip_f.y - 1, 0));

    let dE = textureLoad(depth_full, ipE, 0).x;
    let dW = textureLoad(depth_full, ipW, 0).x;
    let dN = textureLoad(depth_full, ipN, 0).x;
    let dS = textureLoad(depth_full, ipS, 0).x;

    let tol = 0.02 + 0.06 * smoothstep(10.0, 80.0, d0);

    let wE = 1.0 - smoothstep(0.0, tol, abs(dE - d0));
    let wW = 1.0 - smoothstep(0.0, tol, abs(dW - d0));
    let wN = 1.0 - smoothstep(0.0, tol, abs(dN - d0));
    let wS = 1.0 - smoothstep(0.0, tol, abs(dS - d0));

    let wsum = max(wE + wW + wN + wS, 1e-4);
    let blur = (gE * wE + gW * wW + gN * wN + gS * wS) / wsum;

    god_lin = max(gC + COMPOSITE_SHARPEN * (gC - blur), vec3<f32>(0.0));
    god_lin = max(god_lin - vec3<f32>(GODRAY_BLACK_LEVEL), vec3<f32>(0.0));
    god_lin = god_lin / (god_lin + vec3<f32>(GODRAY_KNEE_COMPOSITE));

    god_far   = smoothstep(GODRAY_FADE_NEAR, GODRAY_FADE_FAR, d0);
    god_scale = GODRAY_COMPOSITE_SCALE * mix(1.0, 0.25, god_far);
  }

  // --- Keep far godrays more "sun-yellow" instead of washing to white ---
  let lum_w = vec3<f32>(0.2126, 0.7152, 0.0722);

  // Sun hue normalized (avoid divide-by-zero)
  let sun_hue = normalize(max(SUN_COLOR, vec3<f32>(1e-4)));

  // Match brightness but enforce sun hue
  let g_lum = max(dot(god_lin, lum_w), 0.0);
  let god_sun = sun_hue * g_lum;

  // Apply more hue-lock as distance increases
  let hue_lock = 0.55 * god_far;          // try 0.35..0.85
  god_lin = mix(god_lin, god_sun, hue_lock);

  // Optional extra warmth (subtle) for very far shafts
  let warm = mix(vec3<f32>(1.0), vec3<f32>(1.08, 1.03, 0.92), god_far); // tweak to taste
  god_lin *= warm;

  var hdr = max(base + god_scale * god_lin, vec3<f32>(0.0));

  // Bloom (hue-preserving + distance-faded)
  if (ENABLE_BLOOM) {
    let bloom_thresh = 1.4;
    let bloom_k      = 0.12;
    let bloom_k_eff  = bloom_k * mix(1.0, 0.0, god_far);

    let b0 = bright_extract_hue(hdr, bloom_thresh);

    let ipx1 = vec2<i32>(clamp(ip_r.x + 2, 0, rd_i.x - 1), ip_r.y);
    let ipx0 = vec2<i32>(clamp(ip_r.x - 2, 0, rd_i.x - 1), ip_r.y);
    let ipy1 = vec2<i32>(ip_r.x, clamp(ip_r.y + 2, 0, rd_i.y - 1));
    let ipy0 = vec2<i32>(ip_r.x, clamp(ip_r.y - 2, 0, rd_i.y - 1));


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
  }

  // Distance-safe saturation compensation (HDR)
  let l_hdr  = max(dot(hdr, lum_w), 1e-6);
  let gray_h = vec3<f32>(l_hdr);

  let t_sat = smoothstep(30.0, 100.0, d0);

  var sat_boost = 1.00 + 0.55 * t_sat;

  let hi = smoothstep(1.6, 6.0, l_hdr);
  sat_boost = mix(sat_boost, 1.0, 0.55 * hi);

  hdr = mix(gray_h, hdr, sat_boost);

  // --- Grade knobs (constants or uniforms)
  let exposure = 1.10;
  let contrast = 1.05;
  let temp     = 0.03; // +warm, -cool

  hdr *= exposure;

  // temperature: push R up, B down a bit
  hdr *= vec3<f32>(1.0 + temp, 1.0, 1.0 - temp);

  // contrast around mid-gray
  let mid = vec3<f32>(0.18);
  hdr = (hdr - mid) * contrast + mid;
  hdr = max(hdr, vec3<f32>(0.0));


  // --- Filmic tonemap (single global tonemap) ---
  let white_point: f32 = 6.0; // try 4..10 (bigger = brighter highlights)
  var c = tonemap_filmic_white_scale(hdr * POST_EXPOSURE, white_point);

  // Clamp to display range (still linear at this point)
  c = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));

  // Optional tiny "print" bias (keep subtle)
  c = pow(c, vec3<f32>(0.98));

  // Dither/grain before gamma
  let fi = f32(cam.frame_index & 255u);
  let n0 = hash12(px_render + vec2<f32>(fi, 0.0)) - 0.5;
  let n1 = hash12(px_render * 0.73 + vec2<f32>(0.0, fi)) - 0.5;
  let n  = 0.6 * n0 + 0.4 * n1;
  c += vec3<f32>(n / 1536.0);

  // Gamma encode to LDR output
  let ldr = gamma_encode(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)));
  return vec4<f32>(ldr, 1.0);
}

// LUT-less filmic tonemap (Hable/Uncharted2-style), per-channel.
// Input: HDR linear. Output: LDR-ish (0..~1), still linear until gamma.
fn tonemap_filmic_hable(x: vec3<f32>) -> vec3<f32> {
  // These are the classic Hable curve constants.
  let A: f32 = 0.22;
  let B: f32 = 0.30;
  let C: f32 = 0.10;
  let D: f32 = 0.20;
  let E: f32 = 0.01;
  let F: f32 = 0.30;

  let num = x * (A * x + C * B) + D * E;
  let den = x * (A * x + B)     + D * F;

  return (num / max(den, vec3<f32>(1e-6))) - vec3<f32>(E / F);
}

// Optional: normalize so "white" maps nicely.
// (Keeps highlights from feeling dim if you raise exposure.)
fn tonemap_filmic_white_scale(x: vec3<f32>, white_point: f32) -> vec3<f32> {
  let w = tonemap_filmic_hable(vec3<f32>(white_point));
  let invw = vec3<f32>(1.0) / max(w, vec3<f32>(1e-6));
  return tonemap_filmic_hable(x) * invw;
}
