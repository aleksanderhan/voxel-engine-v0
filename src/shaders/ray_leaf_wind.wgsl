// ray_leaf_wind.wgsl
//
// Leaf wind field + displaced leaf cube intersection.
//
// Purpose:
// Some materials (e.g. "leaves") are rendered as cubes whose *position is displaced*
// by a procedural wind field. Instead of moving geometry on the CPU, the shader
// applies a small per-leaf offset at intersection time.
//
// How it's used in the tracer (see ray_core.wgsl):
// - When a queried leaf material matches the "leaf" id (in your code: mat == 5),
//   the tracer calls leaf_displaced_cube_hit() instead of intersecting the
//   undisplaced cell cube.
// - This returns a hit time and a stable face normal using the shared AABB routine
//   aabb_hit_normal_inv() (from ray_core.wgsl), which reduces face/edge flicker.
//
// Dependencies:
// - safe_inv() from ray_core.wgsl
// - aabb_hit_normal_inv() from ray_core.wgsl

// -----------------------------------------------------------------------------
// Tuning knobs (find these first)
// -----------------------------------------------------------------------------
//
// These constants control the "look" of leaf motion and how far cubes are allowed
// to move. Keep them here so you can tweak wind without hunting through code.

// World-space frequency -> cell size (larger = smoother coherent motion).
// NOTE: the code uses `floor(pos_m * WIND_CELL_FREQ)` so cell_size ≈ 1 / WIND_CELL_FREQ.
const WIND_CELL_FREQ : f32 = 2.5;

// Dominant horizontal wind direction (XZ). Normalized internally.
const WIND_DIR_XZ : vec2<f32> = vec2<f32>(0.9, 0.4);

// Height ramp: wind starts at y=WIND_RAMP_Y0 and reaches full strength by y=WIND_RAMP_Y1.
const WIND_RAMP_Y0 : f32 = 2.0;
const WIND_RAMP_Y1 : f32 = 14.0;

// Gust (slow) and flutter (fast) temporal frequencies (rad-ish, but we're in sin() anyway).
const WIND_GUST_TIME_FREQ    : f32 = 0.9;
const WIND_FLUTTER_TIME_FREQ : f32 = 4.2;

// Spatial frequencies for gust/flutter variation in XZ.
const WIND_GUST_XZ_FREQ    : vec2<f32> = vec2<f32>(0.35, 0.22);
const WIND_FLUTTER_XZ_FREQ : vec2<f32> = vec2<f32>(1.7,  1.1);

// Mixing between gust and flutter for horizontal motion.
const WIND_GUST_WEIGHT    : f32 = 0.75;
const WIND_FLUTTER_WEIGHT : f32 = 0.25;

// Vertical wobble amount (before the later per-leaf vertical reduction).
const WIND_VERTICAL_SCALE : f32 = 0.25;

// Reduce vertical motion after scaling by external strength (leaves sway sideways).
const LEAF_VERTICAL_REDUCE : f32 = 0.15;

// Convert wind vector magnitude into a cube offset (scaled by size).
const LEAF_OFFSET_AMP : f32 = 0.45;  // fraction of cube size

// Clamp offset length to avoid jumping across neighbors (fraction of cube size).
const LEAF_OFFSET_MAX_FRAC : f32 = 0.45;

// Hash phase offsets (keep these fixed; changing them changes the "randomness seed").
const WIND_PHASE_OFF_1 : vec3<f32> = vec3<f32>(19.0, 7.0, 11.0);
const TAU : f32 = 6.28318530718;

// -----------------------------------------------------------------------------
// Hash / noise helpers
// -----------------------------------------------------------------------------

/// 3D -> 1D hash (cheap, deterministic).
/// Used to randomize phase per wind "cell" so the field doesn't look perfectly periodic.
fn hash1(p: vec3<f32>) -> f32 {
  let h = dot(p, vec3<f32>(127.1, 311.7, 74.7));
  return fract(sin(h) * 43758.5453);
}

// -----------------------------------------------------------------------------
// Wind field
// -----------------------------------------------------------------------------

/// Procedural wind field at world position `pos_m` (meters) and time `t` (seconds).
///
/// Output is a 3D displacement direction (not yet scaled to cube size).
///
/// Design:
/// - Quantize space into large cells so nearby leaves move coherently.
/// - Use two hashed phases (ph0/ph1) to avoid obvious repetition.
/// - Mix a slow "gust" with a fast "flutter".
/// - Height factor ramps wind from 0 near the ground to 1 above some height.
/// - Dominant direction lives in XZ (horizontal wind), with a smaller vertical component.
fn wind_field(pos_m: vec3<f32>, t: f32) -> vec3<f32> {
  // Large-scale wind cell coordinates.
  // cell_size ≈ 1 / WIND_CELL_FREQ meters.
  let cell = floor(pos_m * WIND_CELL_FREQ);

  // Two independent phase seeds for gust vs flutter.
  let ph0 = hash1(cell);
  let ph1 = hash1(cell + WIND_PHASE_OFF_1);

  // Base horizontal wind direction (normalized 2D vector in XZ).
  let dir = normalize(WIND_DIR_XZ);

  // Height ramp: wind starts near WIND_RAMP_Y0 and reaches full by WIND_RAMP_Y1.
  let h = clamp((pos_m.y - WIND_RAMP_Y0) / max(WIND_RAMP_Y1 - WIND_RAMP_Y0, 1e-3), 0.0, 1.0);

  // Slow gust: low frequency in time and space.
  let gust = sin(
    t * WIND_GUST_TIME_FREQ +
    dot(pos_m.xz, WIND_GUST_XZ_FREQ) +
    ph0 * TAU
  );

  // Fast flutter: higher frequency variation.
  let flutter = sin(
    t * WIND_FLUTTER_TIME_FREQ +
    dot(pos_m.xz, WIND_FLUTTER_XZ_FREQ) +
    ph1 * TAU
  );

  // Horizontal displacement dominates; flutter adds small variation.
  let xz = dir * (WIND_GUST_WEIGHT * gust + WIND_FLUTTER_WEIGHT * flutter) * h;

  // Small vertical wobble (reduced later again in leaf_cube_offset()).
  let y  = WIND_VERTICAL_SCALE * flutter * h;

  return vec3<f32>(xz.x, y, xz.y);
}

/// Clamp a vector to a maximum Euclidean length.
fn clamp_len(v: vec3<f32>, max_len: f32) -> vec3<f32> {
  let l2 = dot(v, v);
  if (l2 <= max_len * max_len) { return v; }
  return v * (max_len / sqrt(l2));
}

// -----------------------------------------------------------------------------
// Leaf cube displacement policy
// -----------------------------------------------------------------------------

/// Compute the displacement offset applied to a leaf cube.
///
/// Inputs:
/// - bmin/size  : original cube bounds (world meters)
/// - time_s     : time in seconds
/// - strength   : overall wind strength scalar (external knob)
///
/// Policy:
/// - Evaluate wind at cube center to move the whole cube coherently.
/// - Reduce vertical movement (leaves sway mostly sideways).
/// - Scale displacement amplitude with cube size (so small cubes don't teleport).
/// - Clamp to a fraction of cube size to prevent crossing too far into neighbors.
fn leaf_cube_offset(bmin: vec3<f32>, size: f32, time_s: f32, strength: f32) -> vec3<f32> {
  // Sample wind at cube center for coherent motion.
  let center = bmin + vec3<f32>(0.5 * size);

  // Base wind vector, scaled by external strength.
  var w = wind_field(center, time_s) * strength;

  // Reduce vertical component further (visual preference + fewer intersection surprises).
  w = vec3<f32>(w.x, LEAF_VERTICAL_REDUCE * w.y, w.z);

  // Scale by cube size so displacement is relative, not absolute.
  let amp = LEAF_OFFSET_AMP * size;

  // Clamp final offset to avoid huge overlaps.
  return clamp_len(w * amp, LEAF_OFFSET_MAX_FRAC * size);
}

// -----------------------------------------------------------------------------
// Displaced cube intersection
// -----------------------------------------------------------------------------

/// Result of intersecting a displaced leaf cube.
struct LeafCubeHit {
  hit  : bool,
  t    : f32,       // ray parameter at hit (entry time)
  n    : vec3<f32>, // stable face normal
};

/// Intersect a ray with a leaf cube displaced by the wind field.
///
/// Steps:
/// 1) Compute displacement offset from wind.
/// 2) Shift the cube (bmin2 = bmin + off).
/// 3) Intersect using slab-based AABB routine with stable normal selection.
///    (Using the same intersection code as solids avoids edge/face normal flipping artifacts.)
///
/// The trusted interval [t_min, t_max] is forwarded so the intersection is constrained
/// to the current chunk/root interval used by the tracer.
fn leaf_displaced_cube_hit(
  ro: vec3<f32>,
  rd: vec3<f32>,
  bmin: vec3<f32>,
  size: f32,
  time_s: f32,
  strength: f32,
  t_min: f32,
  t_max: f32
) -> LeafCubeHit {
  // Compute wind-driven displacement and shift cube.
  let off   = leaf_cube_offset(bmin, size, time_s, strength);
  let bmin2 = bmin + off;

  // Use slab-style AABB hit + stable normal (precomputed inv avoids division issues).
  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));
  let bh  = aabb_hit_normal_inv(ro, rd, inv, bmin2, size, t_min, t_max);

  return LeafCubeHit(bh.hit, bh.t, bh.n);
}
