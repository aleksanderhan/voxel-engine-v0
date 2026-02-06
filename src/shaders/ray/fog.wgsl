



fn fog_color_from_sky(rd: vec3<f32>, sky: vec3<f32>) -> vec3<f32> {
  let up = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);

  
  let sky_clamped = min(sky, vec3<f32>(0.45));

  return mix(FOG_COLOR_GROUND, sky_clamped, FOG_COLOR_SKY_BLEND * up);
}


fn fog_inscatter(rd: vec3<f32>, fogc: vec3<f32>) -> vec3<f32> {
  let mu = clamp(dot(rd, SUN_DIR), 0.0, 1.0);
  let sun_scatter = pow(mu, 8.0);
  return fogc + 0.35 * sun_scatter * SUN_COLOR;
}

fn fog_density_primary() -> f32 {
  return max(cam.voxel_params.w * FOG_PRIMARY_SCALE, 0.0);
}

fn fog_density_godray() -> f32 {
  return max(cam.voxel_params.w * FOG_GODRAY_SCALE, 0.0);
}

fn fog_optical_depth_with_base(base: f32, ro: vec3<f32>, rd: vec3<f32>, t: f32) -> f32 {
  if (base <= 0.0) { return 0.0; }

  let k = FOG_HEIGHT_FALLOFF;
  let y0 = ro.y;
  let dy = rd.y;

  if (abs(dy) < 1e-4) {
    return base * exp(-k * y0) * t;
  }

  let a = exp(-k * y0);
  let b = exp(-k * (y0 + dy * t));
  return base * (a - b) / (k * dy);
}

fn fog_transmittance_primary(ro: vec3<f32>, rd: vec3<f32>, t: f32) -> f32 {
  let od = max(fog_optical_depth_with_base(fog_density_primary(), ro, rd, t), 0.0);
  return exp(-od);
}

fn fog_transmittance_godray(ro: vec3<f32>, rd: vec3<f32>, t: f32) -> f32 {
  let od = max(fog_optical_depth_with_base(fog_density_godray(), ro, rd, t), 0.0);
  return exp(-od);
}

fn apply_fog(
  surface: vec3<f32>,
  ro: vec3<f32>,
  rd: vec3<f32>,
  t_scene: f32,
  sky: vec3<f32>
) -> vec3<f32> {
  let T    = fog_transmittance_primary(ro, rd, t_scene);
  let fogc = fog_color_from_sky(rd, sky);
  let ins  = fog_inscatter(rd, fogc);
  return surface * T + ins * (1.0 - T);
}
