// src/shaders/ray/light.wgsl
// --------------------------
// Deterministic voxel light gather for MAT_LIGHT voxels.
//
// Goals:
// - Uniform, stable lighting before TAA (no per-frame or per-surface jitter).
// - Low-discrepancy hemisphere sampling for smooth distribution.

// Tunables live in common.wgsl.

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

fn make_tbn(n: vec3<f32>) -> mat3x3<f32> {
  let up_ref = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
  let t = normalize(cross(up_ref, n));
  let b = normalize(cross(n, t));
  return mat3x3<f32>(t, b, n);
}

// Van der Corput radical inverse (base 2).
fn radical_inverse_vdc(bits: u32) -> f32 {
  var b = bits;
  b = (b << 16u) | (b >> 16u);
  b = ((b & 0x55555555u) << 1u) | ((b & 0xAAAAAAAAu) >> 1u);
  b = ((b & 0x33333333u) << 2u) | ((b & 0xCCCCCCCCu) >> 2u);
  b = ((b & 0x0F0F0F0Fu) << 4u) | ((b & 0xF0F0F0F0u) >> 4u);
  b = ((b & 0x00FF00FFu) << 8u) | ((b & 0xFF00FF00u) >> 8u);
  return f32(b) * 2.3283064365386963e-10; // 1/2^32
}

fn hammersley(i: u32, n: u32) -> vec2<f32> {
  let fi = f32(i);
  let fn = max(1.0, f32(n));
  return vec2<f32>(fi / fn, radical_inverse_vdc(i));
}

// Cosine-weighted hemisphere sample oriented around +Z.
fn hemisphere_cosine(u: vec2<f32>) -> vec3<f32> {
  let phi = 6.28318530718 * u.x;
  let cos_theta = sqrt(1.0 - u.y);
  let sin_theta = sqrt(u.y);
  return vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

// Use your real HDR emission for lighting too.
fn light_emission_radiance() -> vec3<f32> {
  return material_emission(MAT_LIGHT);
}

// -----------------------------------------------------------------------------
// 3D DDA: walk voxels along ray without skipping 1-voxel lights
// -----------------------------------------------------------------------------

// Returns vec4(hitPosWS.xyz, hitFlag).
// hitFlag = 1: hit a light voxel
// hitFlag = 0: no light (either blocked by solid or ran out of steps)
fn dda_hit_light(p0: vec3<f32>, dir: vec3<f32>, max_steps: u32, vs: f32) -> vec4<f32> {
  var gpos = p0 / max(vs, 1e-6);
  var cell = vec3<i32>(floor(gpos));

  let d = dir;

  let step = vec3<i32>(
    select(-1, 1, d.x >= 0.0),
    select(-1, 1, d.y >= 0.0),
    select(-1, 1, d.z >= 0.0)
  );

  let next_boundary = vec3<f32>(
    f32(cell.x + select(0, 1, d.x >= 0.0)),
    f32(cell.y + select(0, 1, d.y >= 0.0)),
    f32(cell.z + select(0, 1, d.z >= 0.0))
  );

  let inv = vec3<f32>(
    select(1e30, 1.0 / abs(d.x), abs(d.x) > 1e-6),
    select(1e30, 1.0 / abs(d.y), abs(d.y) > 1e-6),
    select(1e30, 1.0 / abs(d.z), abs(d.z) > 1e-6)
  );

  var tMax = vec3<f32>(
    (next_boundary.x - gpos.x) * inv.x,
    (next_boundary.y - gpos.y) * inv.y,
    (next_boundary.z - gpos.z) * inv.z
  );

  let tDelta = inv;

  for (var s: u32 = 0u; s < max_steps; s = s + 1u) {
    let center_ws = (vec3<f32>(f32(cell.x) + 0.5, f32(cell.y) + 0.5, f32(cell.z) + 0.5)) * vs;
    let leaf = query_leaf_world(center_ws);

    if (leaf.mat == MAT_LIGHT) {
      return vec4<f32>(center_ws, 1.0);
    }

    if (leaf.mat != MAT_AIR) {
      // SOLID blocks the ray
      return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // step to next voxel boundary
    if (tMax.x < tMax.y) {
      if (tMax.x < tMax.z) { cell.x += step.x; tMax.x += tDelta.x; }
      else                 { cell.z += step.z; tMax.z += tDelta.z; }
    } else {
      if (tMax.y < tMax.z) { cell.y += step.y; tMax.y += tDelta.y; }
      else                 { cell.z += step.z; tMax.z += tDelta.z; }
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
    let u = hammersley(i, LIGHT_RAYS);
    let ldir = normalize(tbn * hemisphere_cosine(u));

    let hit = dda_hit_light(p0, ldir, LIGHT_MAX_DIST_VOX, vs);
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
