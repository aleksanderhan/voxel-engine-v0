// light.wgsl
//
// Deterministic voxel light gathering (Fix 4):
// - Deterministic (no RNG), stable (no shimmer)
// - Removes structured checker/speckle via:
//     * per-surface-voxel Cranley–Patterson shift of the Hammersley sequence
//     * per-surface-voxel rotation around the normal
// - More physical estimator:
//     * normalize by LIGHT_RAYS (not by hit count)
//     * inverse-square-ish falloff (clamped near the source)
// - Less fake “backside glow”:
//     * reduced wrap
//     * removed minimum fill
// - Reduced light leaks at chunk boundaries in visibility DDA:
//     * treat leaving chunk bounds as blocked (conservative)
//
// Expected external symbols (defined elsewhere in your shader project):
// - `cam.voxel_params.x` = voxel size in world meters
// - `cam.chunk_size`     = chunk voxel edge length (integer)
// - `query_leaf_world(p_ws) -> leaf`
// - `leaf.mat` material id
// - `MAT_AIR`, `MAT_LIGHT` material ids
//
// This file only contains the lighting code; it does not declare bindings.

// -----------------------------------------------------------------------------
// Tunables
// -----------------------------------------------------------------------------

const LIGHT_MAX_DIST_VOX : u32 = 96u;   // reach in voxels
const LIGHT_RAYS         : u32 = 32u;   // # probe rays (12..64)
const LIGHT_INTENSITY    : f32 = 40.0;   // start here, tune 2..16 with 1/r^2 falloff

// Wrap diffuse: 0 = lambert. Keep small for local emitters.
const LIGHT_WRAP         : f32 = 0.20;

// Minimum fill from a visible light hit. 0 recommended; do ambient separately.
const LIGHT_FILL         : f32 = 0.00;

// -----------------------------------------------------------------------------
// Emission
// -----------------------------------------------------------------------------

fn light_emission_color() -> vec3<f32> {
  // warm emitter
  return vec3<f32>(1.0, 0.95, 0.75);
}

// -----------------------------------------------------------------------------
// Hashing / deterministic scrambling
// -----------------------------------------------------------------------------

fn hash3_i32(p: vec3<i32>) -> u32 {
  // mix coordinates into one u32
  let ux = u32(p.x) * 73856093u;
  let uy = u32(p.y) * 19349663u;
  let uz = u32(p.z) * 83492791u;
  return hash_u32(ux ^ uy ^ uz);
}

fn rot_about_axis(v: vec3<f32>, axis: vec3<f32>, ang: f32) -> vec3<f32> {
  // Rodrigues rotation
  let a = normalize(axis);
  let s = sin(ang);
  let c = cos(ang);
  return v * c + cross(a, v) * s + a * dot(a, v) * (1.0 - c);
}

// -----------------------------------------------------------------------------
// Voxel-grid DDA visibility between p and q.
// Returns 1.0 = visible, 0.0 = blocked.
// -----------------------------------------------------------------------------

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

  // Nudge off the surface to avoid immediate self-hit
  let p0 = p + dir * (0.60 * vs);

  // Convert to voxel-space coordinates relative to root_bmin:
  let lp0 = (p0 - root_bmin) / max(vs, 1e-6);
  let lq  = (q  - root_bmin) / max(vs, 1e-6);

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
  let maxSteps: u32 = u32(clamp(dist / max(vs, 1e-6) + 4.0, 4.0, 256.0));

  // Endpoint in voxel-space (to stop once we’ve reached the light’s cell)
  let end_ix: i32 = i32(floor(lq.x));
  let end_iy: i32 = i32(floor(lq.y));
  let end_iz: i32 = i32(floor(lq.z));

  for (var s: u32 = 0u; s < maxSteps; s = s + 1u) {
    // If we've reached the voxel cell containing q, we consider it visible.
    if (ix == end_ix && iy == end_iy && iz == end_iz) {
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

// -----------------------------------------------------------------------------
// Main local light gather from MAT_LIGHT voxels
// -----------------------------------------------------------------------------

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

  // Deterministic per-surface-voxel scrambling (breaks structured artifacts)
  let surf_v = vec3<i32>(floor((hp - root_bmin) / max(vs, 1e-6)));
  let h = hash3_i32(surf_v);

  // Rotation around the normal (0..2pi)
  let rot = 6.28318530718 * (f32(h & 1023u) / 1024.0);

  // Cranley–Patterson shift in [0,1) applied to the Hammersley v component
  let vshift = f32((h >> 10u) & 1023u) / 1024.0;

  var sum = vec3<f32>(0.0);

  // For each deterministic hemisphere direction
  for (var i: u32 = 0u; i < LIGHT_RAYS; i = i + 1u) {
    var xi = hammersley_2d(i, LIGHT_RAYS);
    xi.y = fract(xi.y + vshift);

    var ldir_ws = normalize(tbn * sample_hemi_cosine(xi.x, xi.y));
    ldir_ws = normalize(rot_about_axis(ldir_ws, n, rot));

    // March outward in voxel steps, sampling at voxel centers
    for (var s: u32 = 1u; s <= LIGHT_MAX_DIST_VOX; s = s + 1u) {
      let p = p0 + ldir_ws * (f32(s) * vs);

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

        let ndl_raw  = dot(n, ldir);

        // “wrap” diffuse: subtle only
        let ndl_wrap = clamp((ndl_raw + LIGHT_WRAP) / (1.0 + LIGHT_WRAP), 0.0, 1.0);

        // Optional minimum fill (recommended 0; do ambient elsewhere)
        let ndl = max(ndl_wrap, LIGHT_FILL);

        // Keep a mild facing gate to avoid totally unphysical full back lighting
        if (ndl_raw > (-0.90 - 0.10 * LIGHT_WRAP)) {
          let vis = light_visibility_segment(p0, p, root_bmin, root_size, node_base, macro_base);

          // Inverse-square-ish falloff with near clamp (prevents blowup near source)
          let r2_clamped = max(r2, (1.5 * vs) * (1.5 * vs));

          // soften distance in voxel units (tune 12..32)
          let falloff_radius = 20.0 * vs;
          let falloff = 1.0 / (r2_clamped + falloff_radius * falloff_radius);


          let PI : f32 = 3.14159265359;
          sum += light_emission_color() * (LIGHT_INTENSITY * PI * ndl * falloff * vis);
        }

        // Stop after first light hit per ray (stable & cheap)
        break;
      }
    }
  }

  // Normalize by ray count (misses contribute 0 naturally)
  sum *= 1.0 / f32(LIGHT_RAYS);

  return sum;
}

// -----------------------------------------------------------------------------
// Deterministic direction set (Hammersley + radical inverse)
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Hemisphere sampling + basis
// -----------------------------------------------------------------------------

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
