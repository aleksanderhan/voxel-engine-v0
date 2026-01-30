// src/shaders/ray/sky.wgsl
//// --------------------------------------------------------------------------
//// Sky
//// --------------------------------------------------------------------------

fn sky_color_base(rd: vec3<f32>) -> vec3<f32> {
  // Base vertical gradient
  let tsky = clamp(0.5 * (rd.y + 1.0), 0.0, 1.0);
  var col = mix(
    vec3<f32>(0.05, 0.08, 0.12),
    vec3<f32>(0.55, 0.75, 0.95),
    tsky
  );

  // Horizon/zenith shaping
  let y = clamp(rd.y, -0.2, 1.0);
  let horizon = exp(-abs(y) * 8.0);
  let zenith  = smoothstep(0.0, 1.0, y);
  col += 0.12 * horizon * vec3<f32>(0.80, 0.75, 0.70);
  col += 0.05 * zenith  * vec3<f32>(0.20, 0.35, 0.60);

  col *= SKY_EXPOSURE;

  // Sun term (NO acos)
  let mu = clamp(dot(rd, SUN_DIR), -1.0, 1.0);

  let SUN_DISC_ANGULAR_RADIUS : f32 = 0.009;
  let SUN_DISC_SOFTNESS       : f32 = 0.004;

  let mu_inner = cos(SUN_DISC_ANGULAR_RADIUS);
  let mu_outer = cos(SUN_DISC_ANGULAR_RADIUS + SUN_DISC_SOFTNESS);

  let disc = smoothstep(mu_outer, mu_inner, mu);

  let k_halo = 9000.0;
  let halo = 0.15 * exp(-k_halo * max(0.0, 1.0 - mu));

  col += SUN_COLOR * SUN_INTENSITY * (disc + halo);
  return col;
}

fn sky_color(rd: vec3<f32>) -> vec3<f32> {
  var col = sky_color_base(rd);

  // Clouds only affect the rendered sky (NOT the ambient driver anymore)
  var cloud = 0.0;

  if (rd.y > 0.01) {
    let ro = cam.cam_pos.xyz;
    let t = (CLOUD_H - ro.y) / rd.y;

    if (t > 0.0) {
      let hit = ro + rd * t;
      let time_s = cam.voxel_params.y;

      cloud = cloud_coverage_at_xz(hit.xz, time_s);

      let horizon2 = clamp((rd.y - CLOUD_HORIZON_Y0) / CLOUD_HORIZON_Y1, 0.0, 1.0);
      cloud *= horizon2;

      col *= mix(1.0, CLOUD_SKY_DARKEN, cloud);

      let toward_sun = clamp(dot(rd, SUN_DIR), 0.0, 1.0);
      let silver = pow(toward_sun, CLOUD_SILVER_POW) * CLOUD_SILVER_STR;
      let cloud_col = mix(CLOUD_BASE_COL, vec3<f32>(1.0), silver);

      col = mix(col, cloud_col, cloud * CLOUD_BLEND);
    }
  }

  // Optional: dim sun disc through clouds (keep your existing behavior)
  if (CLOUD_DIM_SUN_DISC) {
    let Tc_view = exp(-CLOUD_SHADOW_ABSORB * cloud * CLOUD_SUN_DISC_ABSORB_SCALE);
    // This scales only the sun contribution already in col; easiest is to re-add sun separately,
    // but keep it minimal: treat Tc_view as extra darkening for very bright sun spot:
    col = mix(col, col * Tc_view, cloud);
  }

  return col;
}
