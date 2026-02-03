//// --------------------------------------------------------------------------
//// Voxel light sources (local lighting from MAT_LIGHT voxels)
//// --------------------------------------------------------------------------

const LIGHT_MAX_DIST_VOX : f32 = 14.0; // reach in voxels
const LIGHT_RAYS         : u32 = 18u;  // # of probe rays (tune 12..32)
const LIGHT_STEPS        : u32 = 10u;  // steps along each ray
const LIGHT_INTENSITY    : f32 = 32.0; // overall brightness (tune)

fn light_emission_color() -> vec3<f32> {
  // warm emitter
  return vec3<f32>(1.0, 0.95, 0.75);
}

// Cheap “is there something blocking between p and q” test.
// Uses a few samples along the segment in voxel space.
// Returns 1.0 = visible, 0.0 = blocked.
fn light_visibility_segment(
  p: vec3<f32>,
  q: vec3<f32>,
  root_bmin: vec3<f32>,
  root_size: f32,
  node_base: u32,
  macro_base: u32
) -> f32 {
  let d = q - p;
  let dist = length(d);
  if (dist < 1e-4) { return 1.0; }

  let dir = d / dist;

  // step in *world meters*; tie to voxel size
  let vs = cam.voxel_params.x;
  let step_m = max(vs * 0.9, 1e-4);

  // sample count clamped
  let steps_f = clamp(dist / step_m, 1.0, 24.0);
  let steps = u32(steps_f);

  // start a bit off the surface to avoid self-intersection
  var t = step_m;

  for (var i: u32 = 0u; i < steps; i = i + 1u) {
    let s = p + dir * t;

    let leaf = query_leaf_at(s, root_bmin, root_size, node_base, macro_base);

    // any solid voxel blocks, but allow the light voxel itself
    if (leaf.mat != MAT_AIR && leaf.mat != MAT_LIGHT) {
      return 0.0;
    }

    t += step_m;
    if (t >= dist) { break; }
  }

  return 1.0;
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

  // Build a stable seed tied to *world voxel cell* so it doesn't shimmer with camera.
  // Using the voxel cell coord around hp gives stability per surface region.
  let cell = vec3<i32>(
    i32(floor(hp.x / max(vs, 1e-6))),
    i32(floor(hp.y / max(vs, 1e-6))),
    i32(floor(hp.z / max(vs, 1e-6)))
  );

  // Mix cell coords + normal octant + frame (small frame influence to dither temporally)
  let nxq = u32(select(0, 1, n.x > 0.0)) | (u32(select(0, 1, n.y > 0.0)) << 1u) | (u32(select(0, 1, n.z > 0.0)) << 2u);
  var seed: u32 = hash3_u32(bitcast<u32>(cell.x), bitcast<u32>(cell.y), bitcast<u32>(cell.z));
  seed = hash_u32_2(seed, nxq);
  seed = hash_u32_2(seed, cam.frame_index & 255u); // tiny temporal dither

  let tbn = make_tbn(n);

  var sum = vec3<f32>(0.0);
  var hits: f32 = 0.0;

  // Offset start to avoid self-intersection
  let p0 = hp + n * (0.75 * vs);

  // Probe rays in a cosine hemisphere around the normal
  for (var i: u32 = 0u; i < LIGHT_RAYS; i = i + 1u) {
    let u1 = rng_next(&seed);
    let u2 = rng_next(&seed);

    let local_dir = sample_hemi_cosine(u1, u2);   // around +Z
    let dir = normalize(tbn * local_dir);         // rotate to normal

    // march outward
    for (var s: u32 = 0u; s < LIGHT_STEPS; s = s + 1u) {
      // nonlinear spacing: denser near the surface
      let tf = (f32(s) + 1.0) / f32(LIGHT_STEPS);
      let dist_vox = LIGHT_MAX_DIST_VOX * tf * tf; // quadratic
      let p = p0 + dir * (dist_vox * vs);

      let leaf = query_leaf_at(p, root_bmin, root_size, node_base, macro_base);
      if (leaf.mat == MAT_LIGHT) {
        // treat the found voxel as a point light at p
        let L = p - hp;
        let r2 = max(dot(L, L), (0.75 * vs) * (0.75 * vs));
        let r  = sqrt(r2);
        let ldir = L / r;

        let ndl = max(dot(n, ldir), 0.0);
        if (ndl > 0.0) {
          // Visibility: reuse your segment test (optional but recommended)
          let vis = light_visibility_segment(p0, p, root_bmin, root_size, node_base, macro_base);

          // falloff; smoother than hard inverse-square
          let falloff = 1.0 / (1.0 + 0.20 * r2);

          sum += light_emission_color() * (LIGHT_INTENSITY * ndl * falloff * vis);
          hits += 1.0;
        }

        // stop this ray after first light hit (prevents multiple discrete hits per ray)
        break;
      }

      // Optional: early-out if we hit solid geometry (blocks the probe)
      if (leaf.mat != MAT_AIR && leaf.mat != MAT_LIGHT) {
        break;
      }
    }
  }

  // Normalize by ray count so tuning is stable
  sum *= (1.0 / f32(LIGHT_RAYS));

  // Optional gentle boost if we got any hits, helps small lamps
  sum *= (1.0 + 0.25 * clamp(hits / 4.0, 0.0, 1.0));

  return sum;
}


fn hash_u32_2(a: u32, b: u32) -> u32 {
  return hash_u32(a ^ (b * 0x9e3779b9u));
}

fn rng_next(state: ptr<function, u32>) -> f32 {
  (*state) = hash_u32(*state);
  return f32(*state) * U32_TO_F01; // [0,1)
}

// cosine-ish hemisphere sample around +Z, then rotate to normal
fn sample_hemi_cosine(u1: f32, u2: f32) -> vec3<f32> {
  // cosine-weighted hemisphere
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
