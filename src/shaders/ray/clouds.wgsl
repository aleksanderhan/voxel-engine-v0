// src/shaders/ray/clouds.wgsl
//// --------------------------------------------------------------------------
//// Clouds
//// --------------------------------------------------------------------------

fn cloud_coverage_at_xz(xz: vec2<f32>, time_s: f32) -> f32 {
  var uv = xz * CLOUD_UV_SCALE + CLOUD_WIND * time_s;

  let n  = fbm(uv);
  let n2 = fbm(uv * 2.3 + vec2<f32>(13.2, 7.1));
  let field = 0.65 * n + 0.35 * n2;

  return smoothstep(CLOUD_COVERAGE, CLOUD_COVERAGE + CLOUD_SOFTNESS, field);
}

fn cloud_sun_transmittance(p: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
  if (sun_dir.y <= 0.01) { return 1.0; }

  let t = (CLOUD_H - p.y) / sun_dir.y;
  if (t <= 0.0) { return 1.0; }

  let time_s = cam.voxel_params.y;
  let hit = p + sun_dir * t;
  let cov = cloud_coverage_at_xz(hit.xz, time_s);
  
  let Tc = exp(-CLOUD_SHADOW_ABSORB * cov);
  return mix(1.0, Tc, CLOUD_SHADOW_STRENGTH);
}

fn cloud_coverage_fast(xz: vec2<f32>, time_s: f32) -> f32 {
  // 2 octaves only
  var uv = xz * CLOUD_UV_SCALE + CLOUD_WIND * time_s;
  let n0 = value_noise(uv);
  let n1 = value_noise(uv * 2.1 + vec2<f32>(13.2, 7.1));
  let field = 0.70 * n0 + 0.30 * n1;
  return smoothstep(CLOUD_COVERAGE, CLOUD_COVERAGE + CLOUD_SOFTNESS, field);
}

fn cloud_sun_transmittance_fast(p: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
  if (sun_dir.y <= 0.01) { return 1.0; }
  let t = (CLOUD_H - p.y) / sun_dir.y;
  if (t <= 0.0) { return 1.0; }
  let time_s = cam.voxel_params.y;
  let hit = p + sun_dir * t;
  let cov = cloud_coverage_fast(hit.xz, time_s);
  return exp(-CLOUD_SHADOW_ABSORB * cov);
}
