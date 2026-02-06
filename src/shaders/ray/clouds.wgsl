//// --------------------------------------------------------------------------
//// Clouds (cheap volumetric slab)
//// --------------------------------------------------------------------------

// 0..1 height profile inside slab (soft bottom/top)
fn cloud_height01(y: f32) -> f32 {
  if (!ENABLE_CLOUDS) { return 0.0; }
  let h0 = CLOUD_BASE_H;
  let h1 = CLOUD_TOP_H;
  let t  = clamp((y - h0) / max(h1 - h0, 1e-3), 0.0, 1.0);

  // softer edges, fuller middle
  let edge_in  = smoothstep(0.00, 0.12, t);
  let edge_out = 1.0 - smoothstep(0.78, 1.00, t);
  return edge_in * edge_out;
}

// 2D field used for coverage (xz only)
fn cloud_coverage_field(xz: vec2<f32>, time_s: f32) -> f32 {
  if (!ENABLE_CLOUDS) { return 0.0; }
  var uv = xz * CLOUD_UV_SCALE + CLOUD_WIND * time_s;
  let n  = fbm(uv);
  let n2 = fbm(uv * 2.3 + vec2<f32>(13.2, 7.1));
  return 0.65 * n + 0.35 * n2;
}

// Coverage in [0..1]
fn cloud_coverage_at_xz(xz: vec2<f32>, time_s: f32) -> f32 {
  if (!ENABLE_CLOUDS) { return 0.0; }
  let field = cloud_coverage_field(xz, time_s);
  return smoothstep(CLOUD_COVERAGE, CLOUD_COVERAGE + CLOUD_SOFTNESS, field);
}

// Pseudo-3D density: xz noise + y-warped detail (still cheap)
fn cloud_density(p: vec3<f32>, time_s: f32) -> f32 {
  if (!ENABLE_CLOUDS) { return 0.0; }
  // outside slab => no density
  if (p.y <= CLOUD_BASE_H || p.y >= CLOUD_TOP_H) { return 0.0; }

  let h = cloud_height01(p.y);
  if (h <= 0.0) { return 0.0; }

  // Base coverage gate (controls "how many clouds")
  let cov = cloud_coverage_at_xz(p.xz, time_s);
  if (cov <= 0.001) { return 0.0; }

  // Base “billow” signal (xz) + cheap y decorrelation
  var uv = p.xz * CLOUD_UV_SCALE + CLOUD_WIND * time_s;
  uv += 0.003 * p.y; // makes layers vary with height

  let base = fbm(uv * 1.2);

  // Detail: higher frequency, lightly y-shifted
  let det  = fbm(uv * 3.7 + vec2<f32>(0.11 * p.y, -0.07 * p.y));

  // Combine: make cores dense, edges airy
  var d = mix(base, det, CLOUD_DETAIL_W);
  d = smoothstep(0.45, 0.95, d); // “cloudy part” of noise
  d = pow(d, CLOUD_PUFF_POW);

  // Coverage pushes density up/down, and height shapes it
  d *= cov;
  d *= h;

  return clamp(d, 0.0, 1.0);
}

// March along sun dir to get self-shadowing (0..1 transmittance)
fn cloud_light_transmittance(p: vec3<f32>, time_s: f32) -> f32 {
  if (!ENABLE_CLOUDS) { return 1.0; }
  // If sun is below horizon, don’t self-shadow
  if (SUN_DIR.y <= 0.01) { return 1.0; }

  // Step from p towards sun, within slab
  let tmax = (CLOUD_TOP_H - p.y) / max(SUN_DIR.y, 1e-3);
  if (tmax <= 0.0) { return 1.0; }

  let steps = CLOUD_STEPS_LIGHT;
  let dt    = tmax / f32(max(steps, 1u));

  var od: f32 = 0.0; // optical depth accumulator
  for (var i: u32 = 0u; i < steps; i = i + 1u) {
    let ti = (f32(i) + 0.5) * dt;
    let ps = p + SUN_DIR * ti;
    let dens = cloud_density(ps, time_s);
    od += dens * dt;
  }

  // Convert to transmittance. CLOUD_DENSITY controls “thickness”
  return exp(-CLOUD_DENSITY * od * CLOUD_SHADOW_ABSORB);
}

fn cloud_sun_transmittance(p: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
  if (!ENABLE_CLOUDS) { return 1.0; }
  if (sun_dir.y <= 0.01) { return 1.0; }

  // Intersect the sun ray with slab [BASE..TOP]
  let y0 = CLOUD_BASE_H;
  let y1 = CLOUD_TOP_H;

  let t0 = (y0 - p.y) / sun_dir.y;
  let t1 = (y1 - p.y) / sun_dir.y;

  let ta = min(t0, t1);
  let tb = max(t0, t1);

  // Entire slab is behind p along +sun_dir
  if (tb <= 0.0) { return 1.0; }

  // Forward segment inside slab
  let t_start = max(ta, 0.0);
  let t_end   = tb;

  if (t_end <= t_start) { return 1.0; }

  let time_s = cam.voxel_params.y;

  // Very cheap integration (4 steps)
  let steps: u32 = 4u;
  let dt = (t_end - t_start) / f32(steps);

  var od: f32 = 0.0;
  for (var i: u32 = 0u; i < steps; i = i + 1u) {
    let ti = t_start + (f32(i) + 0.5) * dt;
    let ps = p + sun_dir * ti;
    od += cloud_density(ps, time_s) * dt;
  }

  let Tc = exp(-CLOUD_DENSITY * od * CLOUD_SHADOW_ABSORB);
  return mix(1.0, Tc, CLOUD_SHADOW_STRENGTH);
}


// A faster version used by godrays (2 steps)
fn cloud_sun_transmittance_fast(p: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
  if (!ENABLE_CLOUDS) { return 1.0; }
  if (sun_dir.y <= 0.01) { return 1.0; }

  let y0 = CLOUD_BASE_H;
  let y1 = CLOUD_TOP_H;

  let t0 = (y0 - p.y) / sun_dir.y;
  let t1 = (y1 - p.y) / sun_dir.y;

  let ta = min(t0, t1);
  let tb = max(t0, t1);

  let t_start = max(ta, 0.0);
  let t_end   = max(tb, 0.0);

  if (t_end <= t_start) { return 1.0; }

  let time_s = cam.voxel_params.y;

  let steps: u32 = 2u;
  let dt = (t_end - t_start) / f32(steps);

  var od: f32 = 0.0;
  for (var i: u32 = 0u; i < steps; i = i + 1u) {
    let ti = t_start + (f32(i) + 0.5) * dt;
    let ps = p + sun_dir * ti;
    od += cloud_density(ps, time_s) * dt;
  }

  return exp(-CLOUD_DENSITY * od * CLOUD_SHADOW_ABSORB);
}

fn cloud_density_godray(p: vec3<f32>, time_s: f32) -> f32 {
  if (!ENABLE_CLOUDS) { return 0.0; }
  if (p.y <= CLOUD_BASE_H || p.y >= CLOUD_TOP_H) { return 0.0; }

  let h = cloud_height01(p.y);
  if (h <= 0.0) { return 0.0; }

  // 2 octave value noise only (much cheaper than fbm(5))
  var uv = p.xz * CLOUD_UV_SCALE + CLOUD_WIND * time_s;
  let n0 = value_noise(uv);
  let n1 = value_noise(uv * 2.0 + vec2<f32>(17.0, 9.0));
  let field = 0.70 * n0 + 0.30 * n1;

  let cov = smoothstep(CLOUD_COVERAGE, CLOUD_COVERAGE + CLOUD_SOFTNESS, field);
  return cov * h;
}

fn cloud_sun_transmittance_godray(p: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
  if (!ENABLE_CLOUDS) { return 1.0; }
  if (sun_dir.y <= 0.01) { return 1.0; }

  let y0 = CLOUD_BASE_H;
  let y1 = CLOUD_TOP_H;

  let t0 = (y0 - p.y) / sun_dir.y;
  let t1 = (y1 - p.y) / sun_dir.y;

  let ta = min(t0, t1);
  let tb = max(t0, t1);

  if (tb <= 0.0) { return 1.0; }

  let t_start = max(ta, 0.0);
  let t_end   = tb;
  if (t_end <= t_start) { return 1.0; }

  let time_s = cam.voxel_params.y;

  // One sample at segment midpoint
  let tm  = t_start + 0.5 * (t_end - t_start);
  let pm  = p + sun_dir * tm;

  let dens = cloud_density_godray(pm, time_s);
  let len  = (t_end - t_start);

  // Same “optical depth -> transmittance” shape as your other funcs
  return exp(-CLOUD_DENSITY * dens * len * CLOUD_SHADOW_ABSORB);
}
