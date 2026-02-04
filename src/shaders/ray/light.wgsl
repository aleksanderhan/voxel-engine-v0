// light.wgsl
//
// Deterministic voxel light gathering (Fix 3):
// - No RNG rays
// - Uses a fixed low-discrepancy direction set (Hammersley + cosine hemisphere)
// - Stable (no shimmer), good-looking, predictable
//
// Expected external symbols (defined elsewhere in your shader project):
// - `cam.voxel_params.x` = voxel size in world meters
// - `query_leaf_at(p, root_bmin, root_size, node_base, macro_base) -> leaf`
// - `leaf.mat` material id
// - `MAT_AIR`, `MAT_LIGHT` material ids
//
// This file only contains the lighting code; it does not declare bindings.

//// --------------------------------------------------------------------------
//// Voxel light sources (local lighting from MAT_LIGHT voxels)
//// --------------------------------------------------------------------------

const LIGHT_MAX_DIST_VOX : f32 = 64.0; // reach in voxels
const LIGHT_RAYS         : u32 = 32u;  // # of probe rays (tune 12..64)
const LIGHT_STEPS        : u32 = u32(LIGHT_MAX_DIST_VOX);  // steps along each ray
const LIGHT_INTENSITY    : f32 = 128.0; // overall brightness (tune)

fn light_emission_color() -> vec3<f32> {
  // warm emitter
  return vec3<f32>(1.0, 0.95, 0.75);
}

// Voxel-grid DDA visibility between p and q.
// Returns 1.0 = visible, 0.0 = blocked.
fn light_visibility_segment(
  p: vec3<f32>,
  q: vec3<f32>,
  root_bmin: vec3<f32>,
  root_size: f32,
  node_base: u32,
  macro_base: u32
) -> f32 {
  let vs = cam.voxel_params.x;

  let d = q - p;
  let dist = length(d);
  if (dist < 1e-5) { return 1.0; }

  let dir = d / dist;

  // --- Nudge off the surface to avoid immediate self-hit
  let p0 = p + dir * (0.60 * vs);

  // Convert to voxel-space coordinates relative to root_bmin:
  // voxel index i = floor((x - root_bmin)/vs)
  let lp0 = (p0 - root_bmin) / max(vs, 1e-6);
  let lq  = (q  - root_bmin) / max(vs, 1e-6);

  // Early out if start is way outside the chunk bounds (optional but helps).
  // We still allow rays that start slightly outside; clamp indices later.
  // (If you prefer strict bounds, you can return 0.0 here instead.)
  // if (any(lp0 < vec3<f32>(-1.0)) || any(lp0 > vec3<f32>(f32(cam.chunk_size) + 1.0))) { ... }

  var ix: i32 = i32(floor(lp0.x));
  var iy: i32 = i32(floor(lp0.y));
  var iz: i32 = i32(floor(lp0.z));

  let tx: f32 = lp0.x;
  let ty: f32 = lp0.y;
  let tz: f32 = lp0.z;

  let stepX: i32 = select(-1, 1, dir.x > 0.0);
  let stepY: i32 = select(-1, 1, dir.y > 0.0);
  let stepZ: i32 = select(-1, 1, dir.z > 0.0);

  // Distance (in param t) to cross one voxel cell along each axis
  // We operate in voxel-space t, where moving 1 voxel along x is t += 1/|dir.x|
  let invX = safe_inv(dir.x);
  let invY = safe_inv(dir.y);
  let invZ = safe_inv(dir.z);

  var tMaxX: f32 = BIG_F32;
  var tMaxY: f32 = BIG_F32;
  var tMaxZ: f32 = BIG_F32;

  var tDeltaX: f32 = BIG_F32;
  var tDeltaY: f32 = BIG_F32;
  var tDeltaZ: f32 = BIG_F32;

  if (abs(dir.x) >= EPS_INV) {
    let nextX = f32(ix + select(0, 1, stepX > 0));
    tMaxX   = (nextX - tx) * invX;
    tDeltaX = abs(invX);
  }
  if (abs(dir.y) >= EPS_INV) {
    let nextY = f32(iy + select(0, 1, stepY > 0));
    tMaxY   = (nextY - ty) * invY;
    tDeltaY = abs(invY);
  }
  if (abs(dir.z) >= EPS_INV) {
    let nextZ = f32(iz + select(0, 1, stepZ > 0));
    tMaxZ   = (nextZ - tz) * invZ;
    tDeltaZ = abs(invZ);
  }

  // How many voxels can the segment cross? upper bound:
  // dist/vs ~ voxel length; clamp for safety.
  let maxSteps: u32 = u32(clamp(dist / max(vs, 1e-6) + 4.0, 4.0, 256.0));

  // Endpoint in voxel-space (to stop once we’ve reached the light’s cell)
  let end_ix: i32 = i32(floor(lq.x));
  let end_iy: i32 = i32(floor(lq.y));
  let end_iz: i32 = i32(floor(lq.z));

  // Chunk voxel bounds (assuming chunk is cam.chunk_size^3, root_bmin is chunk origin)
  let cs: i32 = i32(cam.chunk_size);

  for (var s: u32 = 0u; s < maxSteps; s = s + 1u) {
    // If we've reached the voxel cell containing q, we consider it visible.
    if (ix == end_ix && iy == end_iy && iz == end_iz) {
      return 1.0;
    }

    // Bounds check: leaving the chunk => treat as visible (no occluder inside this chunk).
    // If you want “outside is blocked”, flip this.
    if (ix < 0 || iy < 0 || iz < 0 || ix >= cs || iy >= cs || iz >= cs) {
      return 1.0;
    }

    // Sample at voxel center in world space
    let center_ws = root_bmin + (vec3<f32>(f32(ix), f32(iy), f32(iz)) + vec3<f32>(0.5)) * vs;
    let leaf = query_leaf_world(center_ws);

    // Blocked by any solid that isn't the light material itself
    if (leaf.mat != MAT_AIR && leaf.mat != MAT_LIGHT) { return 0.0; }

    // Advance to next voxel boundary
    if (tMaxX < tMaxY) {
      if (tMaxX < tMaxZ) {
        ix += stepX;
        tMaxX += tDeltaX;
      } else {
        iz += stepZ;
        tMaxZ += tDeltaZ;
      }
    } else {
      if (tMaxY < tMaxZ) {
        iy += stepY;
        tMaxY += tDeltaY;
      } else {
        iz += stepZ;
        tMaxZ += tDeltaZ;
      }
    }
  }

  // If we ran out of steps, assume blocked (conservative).
  return 0.0;
}

fn gather_voxel_lights(
  hp: vec3<f32>,
  n: vec3<f32>,
  root_bmin: vec3<f32>,
  root_size: f32,
  node_base: u32,
  macro_base: u32
) -> vec3<f32> {
  let vs = cam.voxel_params.x;

  // Build tangent space around the surface normal
  let tbn = make_tbn(n);

  // Start point nudged off the surface (prevents self-occlusion)
  let p0 = hp + n * (0.85 * vs);

  // Tunables (keep your constants, but this version is less sensitive)
  let max_dist_vox: f32 = LIGHT_MAX_DIST_VOX;
  let falloff_radius = 24.0 * vs;                // a bit larger helps caves feel “filled”
  let inv_r2 = 1.0 / max(falloff_radius * falloff_radius, 1e-6);

  var sum = vec3<f32>(0.0);
  var hits: f32 = 0.0;

  // For each deterministic hemisphere direction
  for (var i: u32 = 0u; i < LIGHT_RAYS; i = i + 1u) {
    let xi  = hammersley_2d(i, LIGHT_RAYS);
    let ldir_ws = normalize(tbn * sample_hemi_cosine(xi.x, xi.y));

    // March outward in voxel steps, sampling at voxel centers
    // NOTE: stepping in whole voxels prevents skipping small emitters.
    for (var s: u32 = 1u; s <= u32(max_dist_vox); s = s + 1u) {
      let t_vox = f32(s);

      let p = p0 + ldir_ws * (t_vox * vs);

      // Query leaf at this sample point
      let leaf = query_leaf_world(p);

      // If we hit solid that isn't light: ray is blocked
      if (leaf.mat != MAT_AIR && leaf.mat != MAT_LIGHT) {
        break;
      }

      // Found a light voxel
      if (leaf.mat == MAT_LIGHT) {
        let L  = p - hp;
        let r2 = max(dot(L, L), (0.75 * vs) * (0.75 * vs));
        let r  = sqrt(r2);
        let ldir = L / r;

        let ndl = max(dot(n, ldir), 0.0);
        if (ndl > 0.0) {
          // Occlusion check (voxel DDA along the segment)
          let vis = light_visibility_segment(p0, p, root_bmin, root_size, node_base, macro_base);

          // Smooth falloff with explicit radius:
          // 1 / (1 + r^2 / R^2)
          let falloff = 1.0 / (1.0 + r2 * inv_r2);

          sum += light_emission_color() * (LIGHT_INTENSITY * ndl * falloff * vis);
          hits += 1.0;
        }

        // Stop after first light hit per ray (stable & cheap)
        break;
      }
    }
  }

  // Normalize: don’t wash out small lights
  if (hits > 0.0) {
    sum *= 1.0 / hits;

    // Gentle coverage scaling: big emitters still win, tiny ones still show up
    let coverage = clamp(hits / f32(LIGHT_RAYS), 0.0, 1.0);
    sum *= mix(0.45, 1.0, coverage);
  }

  return sum;
}


//// --------------------------------------------------------------------------
//// Deterministic direction set (Hammersley + radical inverse)
//// --------------------------------------------------------------------------

// Van der Corput radical inverse in base 2 (bit-reversal), deterministic.
fn radical_inverse_vdc(bits_in: u32) -> f32 {
  var bits = bits_in;
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);

  // 2^-32 = 2.3283064365386963e-10
  return f32(bits) * 2.3283064365386963e-10;
}

// Hammersley point set: ( (i+0.5)/N, radicalInverse(i) ).
fn hammersley_2d(i: u32, n: u32) -> vec2<f32> {
  let u = (f32(i) + 0.5) / f32(n);
  let v = radical_inverse_vdc(i);
  return vec2<f32>(u, v);
}

//// --------------------------------------------------------------------------
//// Hemisphere sampling + basis
//// --------------------------------------------------------------------------

// cosine-weighted hemisphere sample around +Z, then rotate to normal via TBN
fn sample_hemi_cosine(u1: f32, u2: f32) -> vec3<f32> {
  let r = sqrt(u1);
  let phi = 6.28318530718 * u2;
  let x = r * cos(phi);
  let y = r * sin(phi);
  let z = sqrt(max(0.0, 1.0 - u1));
  return vec3<f32>(x, y, z);
}

fn make_tbn(n: vec3<f32>) -> mat3x3<f32> {
  let up_ref = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
  let t = normalize(cross(up_ref, n));
  let b = normalize(cross(n, t));
  return mat3x3<f32>(t, b, n);
}
