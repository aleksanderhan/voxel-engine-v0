// src/shaders/ray/composite.wgsl
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

fn tonemap_aces_luma(hdr: vec3<f32>) -> vec3<f32> {
  let w = vec3<f32>(0.2126, 0.7152, 0.0722);
  let l_in = max(dot(hdr, w), 1e-6);

  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  let l_out = clamp((l_in*(a*l_in + b)) / (l_in*(c*l_in + d) + e), 0.0, 1.0);

  return hdr * (l_out / l_in);
}

fn godray_sample_bilerp(
  px_full: vec2<f32>,
  godray_tex: texture_2d<f32>,
  depth_full: texture_2d<f32>
) -> vec3<f32> {
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

  let ip = vec2<i32>(i32(floor(px_full.x)), i32(floor(px_full.y)));
  let d0 = textureLoad(depth_full, ip, 0).x;

  let df = textureDimensions(depth_full);
  let p00 = vec2<i32>(clamp(x0 * 4, 0, i32(df.x) - 1), clamp(y0 * 4, 0, i32(df.y) - 1));
  let p10 = vec2<i32>(clamp(x1 * 4, 0, i32(df.x) - 1), clamp(y0 * 4, 0, i32(df.y) - 1));
  let p01 = vec2<i32>(clamp(x0 * 4, 0, i32(df.x) - 1), clamp(y1 * 4, 0, i32(df.y) - 1));
  let p11 = vec2<i32>(clamp(x1 * 4, 0, i32(df.x) - 1), clamp(y1 * 4, 0, i32(df.y) - 1));

  let d00 = textureLoad(depth_full, p00, 0).x;
  let d10 = textureLoad(depth_full, p10, 0).x;
  let d01 = textureLoad(depth_full, p01, 0).x;
  let d11 = textureLoad(depth_full, p11, 0).x;

  let tol = 0.02 + 0.06 * smoothstep(10.0, 80.0, d0);

  let w00 = exp(-abs(d00 - d0) / tol);
  let w10 = exp(-abs(d10 - d0) / tol);
  let w01 = exp(-abs(d01 - d0) / tol);
  let w11 = exp(-abs(d11 - d0) / tol);

  let wf00 = w00 * (1.0 - f.x) * (1.0 - f.y);
  let wf10 = w10 * (f.x)       * (1.0 - f.y);
  let wf01 = w01 * (1.0 - f.x) * (f.y);
  let wf11 = w11 * (f.x)       * (f.y);

  let wsum = max(wf00 + wf10 + wf01 + wf11, 1e-4);
  let c = (c00 * wf00 + c10 * wf10 + c01 * wf01 + c11 * wf11) / wsum;

  // decompress back to linear energy (k must match quarter pass compression)
  let k = 0.25;
  let one = vec3<f32>(1.0);
  let denom = max(one - c, vec3<f32>(1e-4));
  return (k * c) / denom;
}

fn composite_pixel(
  ip: vec2<i32>,
  px: vec2<f32>,
  color_tex: texture_2d<f32>,
  godray_tex: texture_2d<f32>,
  depth_full: texture_2d<f32>
) -> vec4<f32> {
  let dims = textureDimensions(color_tex);
  let base = textureLoad(color_tex, ip, 0).xyz;

  let g  = godray_sample_bilerp(px, godray_tex, depth_full);
  let gx = godray_sample_bilerp(px + vec2<f32>( 1.0, 0.0), godray_tex, depth_full)
         + godray_sample_bilerp(px + vec2<f32>(-1.0, 0.0), godray_tex, depth_full);
  let gy = godray_sample_bilerp(px + vec2<f32>(0.0,  1.0), godray_tex, depth_full)
         + godray_sample_bilerp(px + vec2<f32>(0.0, -1.0), godray_tex, depth_full);
  let gd = godray_sample_bilerp(px + vec2<f32>( 1.0,  1.0), godray_tex, depth_full)
         + godray_sample_bilerp(px + vec2<f32>(-1.0,  1.0), godray_tex, depth_full)
         + godray_sample_bilerp(px + vec2<f32>( 1.0, -1.0), godray_tex, depth_full)
         + godray_sample_bilerp(px + vec2<f32>(-1.0, -1.0), godray_tex, depth_full);

  let blur = (gx + gy + 0.7 * gd) / (4.0 + 0.7 * 4.0);

  var god_lin = max(g + COMPOSITE_SHARPEN * (g - blur), vec3<f32>(0.0));
  god_lin = max(god_lin - vec3<f32>(GODRAY_BLACK_LEVEL), vec3<f32>(0.0));
  god_lin = god_lin / (god_lin + vec3<f32>(GODRAY_KNEE_COMPOSITE));

  let d = textureLoad(depth_full, ip, 0).x;
  let god_far = smoothstep(GODRAY_FADE_NEAR, GODRAY_FADE_FAR, d);

  let god_scale = GODRAY_COMPOSITE_SCALE * mix(1.0, 0.25, god_far);
  var hdr = max(base + god_scale * god_lin, vec3<f32>(0.0));

  // Bloom (hue-preserving + distance-faded)
  let bloom_thresh = 1.4;
  let bloom_k      = 0.12;
  let bloom_k_eff  = bloom_k * mix(1.0, 0.0, god_far);

  let b0 = bright_extract_hue(hdr, bloom_thresh);

  let ipx1 = vec2<i32>(clamp(ip.x + 2, 0, i32(dims.x) - 1), ip.y);
  let ipx0 = vec2<i32>(clamp(ip.x - 2, 0, i32(dims.x) - 1), ip.y);
  let ipy1 = vec2<i32>(ip.x, clamp(ip.y + 2, 0, i32(dims.y) - 1));
  let ipy0 = vec2<i32>(ip.x, clamp(ip.y - 2, 0, i32(dims.y) - 1));

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

  // Distance-safe saturation compensation (HDR)
  let lum_w = vec3<f32>(0.2126, 0.7152, 0.0722);
  let l_hdr = max(dot(hdr, lum_w), 1e-6);
  let gray_hdr = vec3<f32>(l_hdr);

  let t_sat = smoothstep(30.0, 100.0, d);

  var sat_boost = 1.00 + 0.55 * t_sat;

  let hi = smoothstep(1.6, 6.0, l_hdr);
  sat_boost = mix(sat_boost, 1.0, 0.55 * hi);

  hdr = mix(gray_hdr, hdr, sat_boost);

  // Tonemap (luma-preserving)
  var c = tonemap_aces_luma(hdr * POST_EXPOSURE);
  c = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
  c = pow(c, vec3<f32>(0.98));

  // Dither/grain before gamma
  let fi = f32(cam.frame_index & 1023u);
  let n = hash12(px * 1.7 + vec2<f32>(fi, 0.0)) - 0.5;
  c += vec3<f32>(n / 1024.0);

  let ldr = gamma_encode(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)));
  return vec4<f32>(ldr, 1.0);
}
