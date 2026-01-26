// common.wgsl
//
// Shared structs, bindings, constants, and utility functions.

// ------------------------------------------------------------
// GPU structs (must match Rust side layouts)
// ------------------------------------------------------------

struct Node {
  child_base : u32,  // If internal: index of first child in compact list. If leaf: 0xFFFFFFFF.
  child_mask : u32,  // Bitmask of existing children (bits 0..7).
  material   : u32,  // For leaf nodes: material id (0 = empty / air).
  _pad       : u32,
};

struct Camera {
  view_inv    : mat4x4<f32>,
  proj_inv    : mat4x4<f32>,
  cam_pos     : vec4<f32>,

  chunk_size  : u32,
  chunk_count : u32,
  max_steps   : u32,
  _pad0       : u32,

  // voxel_params:
  // x = voxel_size_m
  // y = time_seconds
  // z = wind_strength
  // w = unused
  voxel_params : vec4<f32>,

  grid_origin_chunk : vec4<i32>,
  grid_dims         : vec4<u32>,
};

struct ChunkMeta {
  origin     : vec4<i32>, // world voxel coords
  node_base  : u32,       // base index in packed node buffer
  node_count : u32,
  _pad0      : u32,
  _pad1      : u32,
};

@group(0) @binding(0) var<uniform> cam : Camera;
@group(0) @binding(1) var<storage, read> chunks : array<ChunkMeta>;
@group(0) @binding(2) var<storage, read> nodes  : array<Node>;
@group(0) @binding(3) var<storage, read> chunk_grid : array<u32>;

// ------------------------------------------------------------
// Constants
// ------------------------------------------------------------

const LEAF_U32 : u32 = 0xFFFFFFFFu;
const BIG_F32  : f32 = 1e30;
const EPS_INV  : f32 = 1e-8;

// Directional sun at 45° elevation.
const SUN_DIR : vec3<f32> = vec3<f32>(0.5, 0.70710678, 0.5);

// Shadow tuning
const SHADOW_BIAS  : f32 = 2e-4;

// ------------------------------------------------------------
// Ray reconstruction (pixel -> world ray direction)
// ------------------------------------------------------------

fn ray_dir_from_pixel(px: vec2<f32>, res: vec2<f32>) -> vec3<f32> {
  let ndc = vec4<f32>(
    2.0 * px.x / res.x - 1.0,
    1.0 - 2.0 * px.y / res.y,
    1.0,
    1.0
  );

  let view = cam.proj_inv * ndc;
  let vdir = vec4<f32>(view.xyz / view.w, 0.0);

  let wdir = (cam.view_inv * vdir).xyz;
  return normalize(wdir);
}

// ------------------------------------------------------------
// Ray / AABB intersection (slab method)
// ------------------------------------------------------------

fn intersect_aabb(ro: vec3<f32>, rd: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
  let eps = 1e-8;

  var t_enter = -1e30;
  var t_exit  =  1e30;

  if (abs(rd.x) < eps) {
    if (ro.x < bmin.x || ro.x > bmax.x) { return vec2<f32>(1.0, 0.0); }
  } else {
    let inv = 1.0 / rd.x;
    let t0 = (bmin.x - ro.x) * inv;
    let t1 = (bmax.x - ro.x) * inv;
    t_enter = max(t_enter, min(t0, t1));
    t_exit  = min(t_exit,  max(t0, t1));
  }

  if (abs(rd.y) < eps) {
    if (ro.y < bmin.y || ro.y > bmax.y) { return vec2<f32>(1.0, 0.0); }
  } else {
    let inv = 1.0 / rd.y;
    let t0 = (bmin.y - ro.y) * inv;
    let t1 = (bmax.y - ro.y) * inv;
    t_enter = max(t_enter, min(t0, t1));
    t_exit  = min(t_exit,  max(t0, t1));
  }

  if (abs(rd.z) < eps) {
    if (ro.z < bmin.z || ro.z > bmax.z) { return vec2<f32>(1.0, 0.0); }
  } else {
    let inv = 1.0 / rd.z;
    let t0 = (bmin.z - ro.z) * inv;
    let t1 = (bmax.z - ro.z) * inv;
    t_enter = max(t_enter, min(t0, t1));
    t_exit  = min(t_exit,  max(t0, t1));
  }

  return vec2<f32>(t_enter, t_exit);
}

// ------------------------------------------------------------
// Sparse children addressing (compact child list)
// ------------------------------------------------------------

fn child_rank(mask: u32, ci: u32) -> u32 {
  let bit = 1u << ci;
  let lower = mask & (bit - 1u);
  return countOneBits(lower);
}

// ------------------------------------------------------------
// Cube-face normal helper (used for shading)
// ------------------------------------------------------------

fn cube_normal(hp: vec3<f32>, bmin: vec3<f32>, size: f32) -> vec3<f32> {
  let bmax = bmin + vec3<f32>(size);

  let dx0 = abs(hp.x - bmin.x);
  let dx1 = abs(bmax.x - hp.x);
  let dy0 = abs(hp.y - bmin.y);
  let dy1 = abs(bmax.y - hp.y);
  let dz0 = abs(hp.z - bmin.z);
  let dz1 = abs(bmax.z - hp.z);

  var best = dy0;
  var n = vec3<f32>(0.0, -1.0, 0.0);

  if (dy1 < best) { best = dy1; n = vec3<f32>(0.0,  1.0, 0.0); }

  if (dx0 < best) { best = dx0; n = vec3<f32>(-1.0, 0.0, 0.0); }
  if (dx1 < best) { best = dx1; n = vec3<f32>( 1.0, 0.0, 0.0); }

  if (dz0 < best) { best = dz0; n = vec3<f32>(0.0, 0.0, -1.0); }
  if (dz1 < best) {            n = vec3<f32>(0.0, 0.0,  1.0); }

  return n;
}

// --- Sun / sky ---
const SUN_COLOR     : vec3<f32> = vec3<f32>(1.0, 0.98, 0.90);
const SUN_INTENSITY : f32 = 4.0;

// Sun disc tuning
const SUN_DISC_ANGULAR_RADIUS : f32 = 0.009;   // ~0.5° in radians (0.0087) + a touch
const SUN_DISC_SOFTNESS       : f32 = 0.004;

// --- Fog / volumetrics ---
// Use cam.voxel_params.w as fog density (you said unused).
// Suggested values: 0.0 .. 0.03 depending on your meter scale.
fn fog_density_base() -> f32 { return max(cam.voxel_params.w, 0.0); }

const FOG_HEIGHT_FALLOFF : f32 = 0.08;   // larger = fog dies faster with height
const FOG_MAX_DIST       : f32 = 120.0;  // meters, cap so we don't overdo far marching

// Godrays (volumetric light shafts)
const GODRAY_MAX_DIST    : f32 = 60.0;
const GODRAY_STRENGTH    : f32 = 2.0;

// Simple hash for jitter (pixel+time)
fn hash12(p: vec2<f32>) -> f32 {
  let h = dot(p, vec2<f32>(127.1, 311.7));
  return fract(sin(h) * 43758.5453);
}

// Henyey-Greenstein-ish phase approximation (cheap)
fn phase_mie(costh: f32) -> f32 {
  // g in [0..1): forward scattering; 0.6 looks “godray-ish”
  let g = 0.6;
  let gg = g * g;
  // (1 - g^2) / (1 + g^2 - 2 g cosθ)^(3/2)
  let denom = pow(1.0 + gg - 2.0 * g * costh, 1.5);
  return (1.0 - gg) / max(denom, 1e-3);
}

// Sky with sun disc (no atmosphere sim, just clean + cheap)
fn sky_color(rd: vec3<f32>) -> vec3<f32> {
  let tsky = clamp(0.5 * (rd.y + 1.0), 0.0, 1.0);
  let base = mix(
    vec3<f32>(0.05, 0.08, 0.12),
    vec3<f32>(0.6, 0.8, 1.0),
    tsky
  );

  // Sun disc
  let mu = dot(rd, SUN_DIR); // cos(angle to sun)
  let ang = acos(clamp(mu, -1.0, 1.0));
  let disc = 1.0 - smoothstep(SUN_DISC_ANGULAR_RADIUS, SUN_DISC_ANGULAR_RADIUS + SUN_DISC_SOFTNESS, ang);

  // Small halo
  let halo = exp(-ang * 30.0) * 0.15;

  return base + SUN_COLOR * SUN_INTENSITY * (disc + halo);
}

// Optical depth for exponential height fog along segment [0..t]
// density(y) = base * exp(-falloff * y)  (assuming y is in meters and >= 0 is “up”)
fn fog_optical_depth(ro: vec3<f32>, rd: vec3<f32>, t: f32) -> f32 {
  let base = fog_density_base();
  if (base <= 0.0) { return 0.0; }

  let k = FOG_HEIGHT_FALLOFF;
  let y0 = ro.y;
  let dy = rd.y;

  // Integral: ∫ base * exp(-k (y0 + s dy)) ds from 0..t
  // If dy ~ 0: base * exp(-k y0) * t
  if (abs(dy) < 1e-4) {
    return base * exp(-k * y0) * t;
  }

  let a = exp(-k * y0);
  let b = exp(-k * (y0 + dy * t));
  return base * (a - b) / (k * dy);
}

fn fog_transmittance(ro: vec3<f32>, rd: vec3<f32>, t: f32) -> f32 {
  let od = max(fog_optical_depth(ro, rd, t), 0.0);
  return exp(-od);
}
