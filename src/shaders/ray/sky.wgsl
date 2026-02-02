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
  let halo = 0.03 * exp(-2500.0 * max(0.0, 1.0 - mu));


  col += SUN_COLOR * SUN_INTENSITY * (disc + halo + 0.04 * glow);

  return col;
}


fn sky_color(rd: vec3<f32>) -> vec3<f32> {
  // Base sky + sun
  var col = sky_color_base(rd);

  // We'll track a stable "how cloudy is this view ray" scalar for optional sun-disc dimming.
  // 0 = clear, 1 = fully cloud-occluded (approx).
  var cloud_view: f32 = 0.0;

  // ------------------------------------------------------------------------
  // Volumetric slab clouds (cheap front-to-back)
  // ------------------------------------------------------------------------
  if (rd.y > 0.01) {
    let ro = cam.cam_pos.xyz;
    let time_s = cam.voxel_params.y;

    // Intersect view ray with slab [CLOUD_BASE_H .. CLOUD_TOP_H]
    let y0 = CLOUD_BASE_H;
    let y1 = CLOUD_TOP_H;

    let t0 = (y0 - ro.y) / rd.y;
    let t1 = (y1 - ro.y) / rd.y;

    var t_enter = min(t0, t1);
    var t_exit  = max(t0, t1);

    t_enter = max(t_enter, 0.0);

    if (t_exit > t_enter) {
      // Fade clouds out near horizon
      let horizon2 = clamp((rd.y - CLOUD_HORIZON_Y0) / CLOUD_HORIZON_Y1, 0.0, 1.0);

      let steps = CLOUD_STEPS_VIEW;
      let dt    = (t_exit - t_enter) / f32(max(steps, 1u));

      var T: f32 = 1.0;         // view transmittance through clouds
      var acc = vec3<f32>(0.0); // accumulated cloud radiance

      // Phase for forward scattering (cheap + stable)
      let phase = phase_blended(dot(rd, SUN_DIR));

      for (var i: u32 = 0u; i < steps; i = i + 1u) {
        let ti = t_enter + (f32(i) + 0.5) * dt;
        let p  = ro + rd * ti;

        var dens = cloud_density(p, time_s);
        dens *= horizon2;

        if (dens > 1e-4) {
          // Self-shadow towards sun (adds “volume”)
          let Tl = cloud_light_transmittance(p, time_s);

          // “Silver lining” bias near sun direction
          let toward_sun = clamp(dot(rd, SUN_DIR), 0.0, 1.0);
          let silver = pow(toward_sun, CLOUD_SILVER_POW) * CLOUD_SILVER_STR;

          let cloud_col = mix(CLOUD_BASE_COL, vec3<f32>(1.0), silver);

          // Extinction step (Beer-Lambert)
          let a = 1.0 - exp(-CLOUD_DENSITY * dens * dt);

          // Single-scatter-ish add
          let scatter = cloud_col * (SUN_COLOR * SUN_INTENSITY) * (phase * Tl);

          acc += T * a * scatter;
          T *= (1.0 - a);

          // Track view cloudiness as max occlusion (stable, cheap)
          cloud_view = max(cloud_view, 1.0 - T);

          // Early out once opaque
          if (T < 0.02) { break; }
        }
      }

      // Composite: sky behind attenuated + cloud radiance
      col = col * T + acc;
    }
  }

  // ------------------------------------------------------------------------
  // Optional: dim ONLY very near the sun (prevents “black sky patches”)
  // ------------------------------------------------------------------------
  if (CLOUD_DIM_SUN_DISC) {
    let mu = clamp(dot(rd, SUN_DIR), 0.0, 1.0);
    let near_sun = smoothstep(0.92, 0.995, mu); // tight gate

    // floor so it can't go charcoal
    let Tc_view_raw = exp(-CLOUD_SUN_DISC_ABSORB * cloud_view);
    let Tc_view     = max(Tc_view_raw, 0.40);

    col = mix(col, col * Tc_view, cloud_view * near_sun);
  }

  return col;
}


