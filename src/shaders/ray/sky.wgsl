// src/shaders/ray/sky.wgsl
// ------------------------
// src/shaders/ray/sky.wgsl
//// --------------------------------------------------------------------------
//// Sky
//// --------------------------------------------------------------------------

fn sky_bg(rd: vec3<f32>) -> vec3<f32> {
  let a0: f32 = -0.10; let b0: f32 =  0.85;
  let a1: f32 = -0.20; let b1: f32 =  0.18;
  let a2: f32 =  0.15; let b2: f32 =  1.00;

  let t0 = smoothstep(a0, b0, rd.y);
  var col = mix(
    vec3<f32>(0.05, 0.08, 0.12),
    vec3<f32>(0.55, 0.75, 0.95),
    t0
  );

  let th = 1.0 - smoothstep(a1, b1, abs(rd.y));
  col += (0.12 * th) * vec3<f32>(0.80, 0.75, 0.70);

  let tz = smoothstep(a2, b2, rd.y);
  col += (0.05 * tz) * vec3<f32>(0.20, 0.35, 0.60);

  col *= SKY_EXPOSURE;
  return col;
}

fn sky_sun(rd: vec3<f32>) -> vec3<f32> {
  let mu = clamp(dot(rd, SUN_DIR), -1.0, 1.0);

  let SUN_DISC_ANGULAR_RADIUS : f32 = 0.009;
  let SUN_DISC_SOFTNESS       : f32 = 0.004;

  let mu_inner = cos(SUN_DISC_ANGULAR_RADIUS);
  let mu_outer = cos(SUN_DISC_ANGULAR_RADIUS + SUN_DISC_SOFTNESS);

  let disc = smoothstep(mu_outer, mu_inner, mu);
  let glow = smoothstep(0.70, 0.98, mu);
  let halo = 0.03 * exp(-2500.0 * max(0.0, 1.0 - mu));

  return SUN_COLOR * SUN_INTENSITY * (disc + halo + 0.04 * glow);
}

fn sky_color(rd: vec3<f32>) -> vec3<f32> {
  // Background (no sun) + separate sun term
  let bg  = sky_bg(rd);
  let sun = sky_sun(rd);

  var T_view: f32 = 1.0;         // view transmittance through clouds
  var acc  = vec3<f32>(0.0);     // cloud radiance

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

          // Silver lining bias near sun direction
          let toward_sun = clamp(dot(rd, SUN_DIR), 0.0, 1.0);
          let silver = pow(toward_sun, CLOUD_SILVER_POW) * CLOUD_SILVER_STR;

          let cloud_col = mix(CLOUD_BASE_COL, vec3<f32>(1.0), silver);

          // Beer-Lambert step extinction
          let tau = CLOUD_DENSITY * dens * dt;
          let a   = 1.0 - exp(-tau);

          // Single-scatter-ish add
          let scatter = cloud_col * (SUN_COLOR * SUN_INTENSITY) * (phase * Tl);

          acc   += T_view * a * scatter;
          T_view *= (1.0 - a);

          if (T_view < 0.02) { break; }
        }
      }
    }
  }

  // ------------------------------------------------------------------------
  // Composite: background attenuated by clouds + cloud radiance
  // ------------------------------------------------------------------------
  var col = bg * T_view + acc;

  // ------------------------------------------------------------------------
  // Separately attenuated sun (disc/halo/glow) through clouds
  //
  // FIX:
  // Drive sun-disc attenuation from the same *visual* cloud transmittance T_view,
  // and raise it to a power so moderately-opaque clouds kill the disc.
  // ------------------------------------------------------------------------
  var T_sun: f32 = 1.0;

  if (CLOUD_DIM_SUN_DISC && rd.y > 0.01) {
    // If a cloud makes the sky behind it dim (T_view < 1),
    // the disc should dim *much more*.
    T_sun = pow(clamp(T_view, 0.0, 1.0), CLOUD_SUN_DISC_DIM_POW);
    T_sun = max(T_sun, CLOUD_SUN_DISC_DIM_FLOOR);
  }

  col += sun * T_sun;

  return col;
}
