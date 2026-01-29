// src/shaders/ray/wind.wgsl
//// --------------------------------------------------------------------------
//// Wind + small helpers (shared by leaves/grass)
//// --------------------------------------------------------------------------

fn hash1(p: vec3<f32>) -> f32 {
  let h = dot(p, vec3<f32>(127.1, 311.7, 74.7));
  return fract(sin(h) * 43758.5453);
}

fn wind_field(pos_m: vec3<f32>, t: f32) -> vec3<f32> {
  let cell = floor(pos_m * WIND_CELL_FREQ);

  let ph0 = hash1(cell);
  let ph1 = hash1(cell + WIND_PHASE_OFF_1);

  let dir = normalize(WIND_DIR_XZ);

  let h = clamp((pos_m.y - WIND_RAMP_Y0) / max(WIND_RAMP_Y1 - WIND_RAMP_Y0, 1e-3), 0.0, 1.0);

  let gust = sin(
    t * WIND_GUST_TIME_FREQ +
    dot(pos_m.xz, WIND_GUST_XZ_FREQ) +
    ph0 * TAU
  );

  let flutter = sin(
    t * WIND_FLUTTER_TIME_FREQ +
    dot(pos_m.xz, WIND_FLUTTER_XZ_FREQ) +
    ph1 * TAU
  );

  let xz = dir * (WIND_GUST_WEIGHT * gust + WIND_FLUTTER_WEIGHT * flutter) * h;
  let y  = WIND_VERTICAL_SCALE * flutter * h;

  return vec3<f32>(xz.x, y, xz.y);
}

fn clamp_len(v: vec3<f32>, max_len: f32) -> vec3<f32> {
  let l2 = dot(v, v);
  if (l2 <= max_len * max_len) { return v; }
  return v * (max_len / sqrt(l2));
}
