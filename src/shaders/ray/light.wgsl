// src/shaders/ray/light.wgsl
// ---------------------------------
// Cheap voxel light gather for MAT_LIGHT voxels (no DDA).
//
// What changed (for TAA-friendly behavior):
// - Removed "empty air bail" (it caused random hit/miss speckle in open air)
// - Normalization uses a CONSTANT denom (LIGHT_RAYS), so early-exit doesn’t boost brightness
//
// NOTE: Temporal accumulation is done in a separate pass; this file just produces
// a noisy but unbiased estimate (local radiance) suitable for accumulation.

// Tunables live in common.wgsl.

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

fn hash3_i32(p: vec3<i32>) -> u32 {
  let ux = u32(p.x) * 73856093u;
  let uy = u32(p.y) * 19349663u;
  let uz = u32(p.z) * 83492791u;
  return hash_u32(ux ^ uy ^ uz);
}

fn make_tbn(n: vec3<f32>) -> mat3x3<f32> {
  let up_ref = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
  let t = normalize(cross(up_ref, n));
  let b = normalize(cross(n, t));
  return mat3x3<f32>(t, b, n);
}

fn rot_about_axis(v: vec3<f32>, axis: vec3<f32>, ang: f32) -> vec3<f32> {
  let a = normalize(axis);
  let s = sin(ang);
  let c = cos(ang);
  return v * c + cross(a, v) * s + a * dot(a, v) * (1.0 - c);
}

// Use your real HDR emission for lighting too.
fn light_emission_radiance() -> vec3<f32> {
  return material_emission(MAT_LIGHT);
}

// -----------------------------------------------------------------------------
// Direction set
// -----------------------------------------------------------------------------

fn sphere_dir_local(i: u32, count: u32) -> vec3<f32> {
  let fi = f32(i);
  let count_f = max(1.0, f32(count));
  let golden = 2.399963229728653; // golden angle in radians
  let y = 1.0 - 2.0 * ((fi + 0.5) / count_f);
  let r = sqrt(max(0.0, 1.0 - y * y));
  let phi = golden * fi;
  return vec3<f32>(cos(phi) * r, sin(phi) * r, y);
}

// -----------------------------------------------------------------------------
// Fixed-step raymarch: walk in voxel-sized steps (no DDA)
// -----------------------------------------------------------------------------

// Returns vec4(hitPosWS.xyz, hitFlag).
// hitFlag = 1: hit a light voxel
// hitFlag = 0: no light (either blocked by solid or ran out of steps)
fn march_hit_light(p0: vec3<f32>, dir: vec3<f32>, max_steps: u32, vs: f32) -> vec4<f32> {
  let step_len = max(vs, 1e-6);
  for (var s: u32 = 0u; s < max_steps; s = s + 1u) {
    let t = (f32(s) + 0.5) * step_len;
    let pos = p0 + dir * t;
    let leaf = query_leaf_world(pos);

    if (leaf.mat == MAT_LIGHT) {
      return vec4<f32>(pos, 1.0);
    }

    if (leaf.mat != MAT_AIR) {
      // SOLID blocks the ray
      return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
  }

  return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

// -----------------------------------------------------------------------------
// Main local light gather
// -----------------------------------------------------------------------------

fn gather_voxel_lights(
  hp: vec3<f32>,
  n: vec3<f32>,
  root_bmin: vec3<f32>,
  root_size: f32,
  node_base: u32,
  macro_base: u32
) -> vec3<f32> {
  var hits: u32 = 0u;

  let vs = cam.voxel_params.x;

  // nudge off surface to reduce self hits
  let p0 = hp + n * (0.75 * vs);

  // basis for orienting directions
  let tbn = make_tbn(n);

  // de-pattern rotation: stable per surface cell (no per-frame jitter)
  // to avoid speckled/jittered light around emitters.
  let surf_v = vec3<i32>(floor((hp - root_bmin) / max(vs, 1e-6)));
  let h0 = hash3_i32(surf_v);
  let h1 = hash_u32(h0);
  let rot = 6.28318530718 * (f32(h1 & 1023u) / 1024.0);

  // attenuation helpers
  let soft_r  = LIGHT_SOFT_RADIUS_VOX * vs;
  let soft_r2 = soft_r * soft_r;

  let min_r = LIGHT_NEAR_CLAMP_VOX * vs;
  let min_r2 = min_r * min_r;

  let range_m = LIGHT_RANGE_VOX * vs;
  let range2  = range_m * range_m;

  let Le = light_emission_radiance();

  var sum = vec3<f32>(0.0);

  for (var i: u32 = 0u; i < LIGHT_RAYS; i = i + 1u) {
    var ldir = normalize(tbn * normalize(sphere_dir_local(i, LIGHT_RAYS)));
    ldir = normalize(rot_about_axis(ldir, n, rot));

    let hit = march_hit_light(p0, ldir, LIGHT_MAX_DIST_VOX, vs);
    if (hit.w > 0.5) {
      hits += 1u;

      let pL = hit.xyz;

      let L  = pL - hp;
      let r2 = max(dot(L, L), min_r2);
      let r  = sqrt(r2);
      let ldir_ws = L / r;

      // direct wrap diffuse (subtle)
      let ndl_raw = dot(n, ldir_ws);
      let ndl = clamp((ndl_raw + LIGHT_WRAP) / (1.0 + LIGHT_WRAP), 0.0, 1.0);

      // finite range rolloff
      var att_range = clamp(1.0 - (r2 / max(range2, 1e-6)), 0.0, 1.0);
      att_range = att_range * att_range;

      // softened inverse-square
      let falloff = 1.0 / (r2 + soft_r2);

      // DIRECT
      sum += Le * (LIGHT_DIRECT_GAIN * ndl * falloff * att_range);

      // INDIRECT FILL (cheap “bounce”)
      if (r2 < 0.35 * range2) {
        let bounce = LIGHT_INDIRECT_GAIN * falloff * att_range;
        sum += Le * (0.25 * bounce);
      }

      if (hits >= LIGHT_EARLY_HITS) {
        break;
      }
    }
  }

  // IMPORTANT: constant denom so early exits don't boost brightness
  let denom = max(1.0, f32(LIGHT_RAYS));
  return sum * (1.0 / denom);
}
