// src/shaders/ray/sky.wgsl
//// --------------------------------------------------------------------------
//// Sky
//// --------------------------------------------------------------------------

fn sky_color_base(rd: vec3<f32>) -> vec3<f32> {
  // 4 (a,b) ramps (smoothstep edge pairs)
  let a0: f32 = -0.10; let b0: f32 =  0.85; // vertical gradient
  let a1: f32 = -0.20; let b1: f32 =  0.18; // horizon band
  let a2: f32 =  0.15; let b2: f32 =  1.00; // zenith lift
  let a3: f32 =  0.70; let b3: f32 =  0.98; // near-sun glow (mu space)

  // --- Base vertical gradient
  let t0 = smoothstep(a0, b0, rd.y);
  var col = mix(
    vec3<f32>(0.05, 0.08, 0.12), // deep
    vec3<f32>(0.55, 0.75, 0.95), // sky
    t0
  );

  // --- Horizon shaping (warm haze near y≈0)
  let th = 1.0 - smoothstep(a1, b1, abs(rd.y));
  col += (0.12 * th) * vec3<f32>(0.80, 0.75, 0.70);

  // --- Zenith shaping (cooler up high)
  let tz = smoothstep(a2, b2, rd.y);
  col += (0.05 * tz) * vec3<f32>(0.20, 0.35, 0.60);

  col *= SKY_EXPOSURE;

  // --- Sun disc + glow (no acos)
  let mu = clamp(dot(rd, SUN_DIR), -1.0, 1.0);

  let SUN_DISC_ANGULAR_RADIUS : f32 = 0.009;
  let SUN_DISC_SOFTNESS       : f32 = 0.004;

  let mu_inner = cos(SUN_DISC_ANGULAR_RADIUS);
  let mu_outer = cos(SUN_DISC_ANGULAR_RADIUS + SUN_DISC_SOFTNESS);

  let disc = smoothstep(mu_outer, mu_inner, mu);

  // broad glow using (a3,b3) in mu-space near 1.0
  let glow = smoothstep(a3, b3, mu);

  // tight halo (still cheap)
  let halo = 0.15 * exp(-9000.0 * max(0.0, 1.0 - mu));

  col += SUN_COLOR * SUN_INTENSITY * (disc + halo + 0.04 * glow);

  return col;
}


fn sky_color(rd: vec3<f32>) -> vec3<f32> {
  var col = sky_color_base(rd);

  // cloud coverage along view ray (at fixed cloud plane height)
  var cloud: f32 = 0.0;

  if (rd.y > 0.01) {
    let ro = cam.cam_pos.xyz;
    let t  = (CLOUD_H - ro.y) / rd.y;

    if (t > 0.0) {
      let hit    = ro + rd * t;
      let time_s = cam.voxel_params.y;

      cloud = cloud_coverage_at_xz(hit.xz, time_s);

      // fade clouds out near horizon
      let horizon2 = clamp((rd.y - CLOUD_HORIZON_Y0) / CLOUD_HORIZON_Y1, 0.0, 1.0);
      cloud *= horizon2;

      // silver lining toward sun
      let toward_sun = clamp(dot(rd, SUN_DIR), 0.0, 1.0);
      let silver = pow(toward_sun, CLOUD_SILVER_POW) * CLOUD_SILVER_STR;
      let cloud_col = mix(CLOUD_BASE_COL, vec3<f32>(1.0), silver);

      // single blend (NO black-multiply)
      let cloud_a = CLOUD_BLEND * sqrt(clamp(cloud, 0.0, 1.0));
      col = mix(col, cloud_col, cloud_a);
    }
  }

  // Optional: dim ONLY very near the sun (prevents “black sky patches”)
  if (CLOUD_DIM_SUN_DISC) {
    let mu = clamp(dot(rd, SUN_DIR), 0.0, 1.0);
    let near_sun = smoothstep(0.92, 0.995, mu); // tight gate

    // floor so it can't go charcoal
    let Tc_view_raw = exp(-CLOUD_SUN_DISC_ABSORB * cloud);
    let Tc_view     = max(Tc_view_raw, 0.40);

    col = mix(col, col * Tc_view, cloud * near_sun);
  }

  return col;
}

