//// --------------------------------------------------------------------------
//// Fog
//// --------------------------------------------------------------------------

fn fog_color_from_sky(rd: vec3<f32>, sky: vec3<f32>) -> vec3<f32> {
  let up = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);

  // Keep your clamp behavior identical
  let sky_clamped = min(sky, vec3<f32>(0.45));

  return mix(FOG_COLOR_GROUND, sky_clamped, FOG_COLOR_SKY_BLEND * up);
}


fn fog_inscatter(rd: vec3<f32>, fogc: vec3<f32>) -> vec3<f32> {
  let mu = clamp(dot(rd, SUN_DIR), 0.0, 1.0);
  let mu2 = mu * mu;
  let mu4 = mu2 * mu2;
  let mu8 = mu4 * mu4;
  let sun_scatter = mu8;
  return fogc + 0.35 * sun_scatter * SUN_COLOR;
}

fn fog_density_primary() -> f32 {
  if (!ENABLE_FOG) { return 0.0; }
  return max(cam.voxel_params.w * FOG_PRIMARY_SCALE, 0.0);
}

fn fog_density_godray() -> f32 {
  if (!ENABLE_FOG) { return 0.0; }
  return max(cam.voxel_params.w * FOG_GODRAY_SCALE, 0.0);
}

fn fog_optical_depth_with_base(base: f32, ro: vec3<f32>, rd: vec3<f32>, t: f32) -> f32 {
  if (base <= 0.0) { return 0.0; }

  let k = FOG_HEIGHT_FALLOFF;
  let log2e = 1.4426950408889634;
  let y0 = ro.y;
  let dy = rd.y;

  if (abs(dy) < 1e-4) {
    return base * exp2(-k * y0 * log2e) * t;
  }

  let a = exp2(-k * y0 * log2e);
  let b = exp2(-k * (y0 + dy * t) * log2e);
  return base * (a - b) / (k * dy);
}

fn fog_transmittance_primary(base: f32, ro: vec3<f32>, rd: vec3<f32>, t: f32) -> f32 {
  let od = max(fog_optical_depth_with_base(base, ro, rd, t), 0.0);
  return exp2(-od * 1.4426950408889634);
}

fn fog_transmittance_godray(base: f32, ro: vec3<f32>, rd: vec3<f32>, t: f32) -> f32 {
  let od = max(fog_optical_depth_with_base(base, ro, rd, t), 0.0);
  return exp2(-od * 1.4426950408889634);
}

fn apply_fog(
  surface: vec3<f32>,
  ro: vec3<f32>,
  rd: vec3<f32>,
  t_scene: f32,
  sky: vec3<f32>
) -> vec3<f32> {
  if (!ENABLE_FOG) {
    return surface;
  }
  let base = fog_density_primary();
  if (base <= 0.0) {
    return surface;
  }
  let T    = fog_transmittance_primary(base, ro, rd, t_scene);
  let fogc = fog_color_from_sky(rd, sky);
  let ins  = fog_inscatter(rd, fogc);
  return surface * T + ins * (1.0 - T);
}
