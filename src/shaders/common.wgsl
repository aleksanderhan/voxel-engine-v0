// common.wgsl
//
// Shared WGSL code used across compute passes.
//
// Contents:
// - GPU-side struct definitions (must match Rust `gpu_types.rs` layouts).
// - Bindings for the "scene" group (camera + chunk/node arenas + grid).
// - Common constants (sun direction, numeric epsilons, rendering knobs).
// - Utility functions:
//     * ray reconstruction (pixel -> world ray)
//     * ray/AABB intersection (slab method)
//     * sparse child addressing for compact child lists
//     * cube-face normal extraction (voxel shading)
//     * sky + sun + fractal clouds (FBM)
//     * cloud sunlight transmittance (used by lighting + godrays)
//     * fog / volumetric helpers (optical depth + transmittance)
//
// Keep this file stable: any change to struct layout or bindings must be mirrored
// on the Rust side and in bind group layouts.

// ------------------------------------------------------------
// Tuning knobs (find these first)
// ------------------------------------------------------------

// --- Numeric ---
const LEAF_U32 : u32 = 0xFFFFFFFFu; // Sentinel for "leaf node" in child_base.
const BIG_F32  : f32 = 1e30;        // "Infinity" for ray math.
const EPS_INV  : f32 = 1e-8;        // Epsilon for safe inverses / parallel tests.

// --- Sun / sky ---
const SUN_DIR : vec3<f32> = vec3<f32>(0.61237244, 0.5, 0.61237244); // fixed directional light
const SUN_COLOR     : vec3<f32> = vec3<f32>(1.0, 0.98, 0.90);
const SUN_INTENSITY : f32 = 2.5;

// Sun disc tuning (angular radius ~0.5° in radians, with softness)
const SUN_DISC_ANGULAR_RADIUS : f32 = 0.009;
const SUN_DISC_SOFTNESS       : f32 = 0.004;

// Sky brightness scalar (overall dimmer sky)
const SKY_EXPOSURE : f32 = 0.40;

// --- Shadows ---
const SHADOW_BIAS : f32 = 2e-4; // also see ray_shadow.wgsl for per-ray biasing

// --- Fog / volumetrics ---
// Fog density comes from cam.voxel_params.w; these shape its behavior.
const FOG_HEIGHT_FALLOFF : f32 = 0.10;  // larger = fog decays faster with height
const FOG_MAX_DIST       : f32 = 100.0; // meters: cap fog integration distance

// Godray tuning (used by ray_main.wgsl, but kept here for shared helpers)
const GODRAY_MAX_DIST    : f32 = 90.0;
const GODRAY_STRENGTH    : f32 = 6.0;

// Phase tuning: makes godrays visible from the side by blending isotropic + Mie-ish.
const INV_4PI      : f32 = 0.0795774715; // 1/(4*pi)
const PHASE_G      : f32 = 0.15;         // anisotropy (0=iso, 0.6=very forward)
const PHASE_MIE_W  : f32 = 0.65;         // 0..1, weight of Mie vs isotropic

// --- Fractal clouds (sky + lighting) ---
// Cloud plane in world meters
const CLOUD_H : f32 = 220.0;

// World XZ -> UV scale (smaller => larger cloud features)
const CLOUD_UV_SCALE : f32 = 0.002;

// Wind in UV units per second (direction + speed)
const CLOUD_WIND : vec2<f32> = vec2<f32>(0.020, 0.012);

// Cloud coverage threshold (lower => more clouds)
const CLOUD_COVERAGE : f32 = 0.50;

// Cloud edge softness
const CLOUD_SOFTNESS : f32 = 0.10;

// Fade clouds near horizon to avoid harsh banding
const CLOUD_HORIZON_Y0 : f32 = 0.02;
const CLOUD_HORIZON_Y1 : f32 = 0.25;

// How much clouds darken the underlying sky gradient (0.0..1.0)
const CLOUD_SKY_DARKEN : f32 = 0.95;

// How strongly clouds block direct sun light (bigger => darker ground under clouds)
const CLOUD_ABSORB : f32 = 6.5;

// Cloud tint and “silver lining” strength near sun
const CLOUD_BASE_COL   : vec3<f32> = vec3<f32>(0.72, 0.74, 0.76);
const CLOUD_SILVER_POW : f32 = 8.0;
const CLOUD_SILVER_STR : f32 = 0.6;
const CLOUD_BLEND      : f32 = 0.85;

// Optional: dim sun disc when it is behind cloud from the view ray
const CLOUD_DIM_SUN_DISC : bool = true;
const CLOUD_SUN_DISC_ABSORB_SCALE : f32 = 0.8; // 0..1, smaller dims disc less

// ------------------------------------------------------------
// GPU structs (must match Rust side layouts)
// ------------------------------------------------------------
//
// Alignment note:
// WGSL uses strict alignment rules (uniforms are std140-ish).
// Using vec4 lanes + explicit pads keeps layouts predictable.

struct Node {
  // SVO node with compact children list.
  //
  // child_base:
  //   - internal: base index of compact child list
  //   - leaf: LEAF_U32
  //
  // child_mask:
  //   - occupancy mask for children bits 0..7
  //
  // material:
  //   - leaf payload (0 commonly means empty/air)
  //
  // _pad:
  //   - keep stride 16 bytes
  child_base : u32,
  child_mask : u32,
  material   : u32,
  _pad       : u32,
};

struct Camera {
  // Inverse matrices for ray reconstruction.
  view_inv    : mat4x4<f32>,
  proj_inv    : mat4x4<f32>,

  // Camera world position (vec4 for alignment).
  cam_pos     : vec4<f32>,

  // Chunk/grid parameters.
  chunk_size  : u32,
  chunk_count : u32,
  max_steps   : u32,
  _pad0       : u32,

  // Packed per-frame params:
  // x = voxel_size_m
  // y = time_seconds
  // z = wind_strength
  // w = fog_density
  voxel_params : vec4<f32>,

  // Chunk grid mapping
  grid_origin_chunk : vec4<i32>,
  grid_dims         : vec4<u32>,
};

struct ChunkMeta {
  // Per-chunk metadata
  origin     : vec4<i32>, // world voxel coords (xyz)
  node_base  : u32,       // base index into nodes arena
  node_count : u32,
  _pad0      : u32,
  _pad1      : u32,
};

// ------------------------------------------------------------
// Scene bindings (group(0))
// ------------------------------------------------------------

@group(0) @binding(0) var<uniform> cam : Camera;
@group(0) @binding(1) var<storage, read> chunks : array<ChunkMeta>;
@group(0) @binding(2) var<storage, read> nodes  : array<Node>;
@group(0) @binding(3) var<storage, read> chunk_grid : array<u32>;

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
//
// Returns (t_enter, t_exit). Miss => t_enter > t_exit.

fn intersect_aabb(ro: vec3<f32>, rd: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
  let eps = 1e-8;

  var t_enter = -1e30;
  var t_exit  =  1e30;

  // X slabs
  if (abs(rd.x) < eps) {
    if (ro.x < bmin.x || ro.x > bmax.x) { return vec2<f32>(1.0, 0.0); }
  } else {
    let inv = 1.0 / rd.x;
    let t0 = (bmin.x - ro.x) * inv;
    let t1 = (bmax.x - ro.x) * inv;
    t_enter = max(t_enter, min(t0, t1));
    t_exit  = min(t_exit,  max(t0, t1));
  }

  // Y slabs
  if (abs(rd.y) < eps) {
    if (ro.y < bmin.y || ro.y > bmax.y) { return vec2<f32>(1.0, 0.0); }
  } else {
    let inv = 1.0 / rd.y;
    let t0 = (bmin.y - ro.y) * inv;
    let t1 = (bmax.y - ro.y) * inv;
    t_enter = max(t_enter, min(t0, t1));
    t_exit  = min(t_exit,  max(t0, t1));
  }

  // Z slabs
  if (abs(rd.z) < eps) {
    if (ro.z < bmin.z || ro.z > bmax.z) { return vec2<f32>(1.0, 0.0); }
  } else {
    let inv = 1.0 / rd.z;
    let t0 = (bmin.z - ro.z) * inv;
    let t1 = (bmax.z - ro.z) * inv; // NOTE: fixed (was ro.x in your pasted snippet)
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

// ------------------------------------------------------------
// Hash / noise / FBM (fractal clouds)
// ------------------------------------------------------------

// Hash in [0,1) from a 2D point
fn hash12(p: vec2<f32>) -> f32 {
  let h = dot(p, vec2<f32>(127.1, 311.7));
  return fract(sin(h) * 43758.5453);
}

fn hash21(p: vec2<f32>) -> f32 {
  let h = dot(p, vec2<f32>(127.1, 311.7));
  return fract(sin(h) * 43758.5453);
}

// 2D value noise with smooth interpolation
fn value_noise(p: vec2<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);

  let a = hash21(i);
  let b = hash21(i + vec2<f32>(1.0, 0.0));
  let c = hash21(i + vec2<f32>(0.0, 1.0));
  let d = hash21(i + vec2<f32>(1.0, 1.0));

  let u = f * f * (3.0 - 2.0 * f);

  let x1 = mix(a, b, u.x);
  let x2 = mix(c, d, u.x);
  return mix(x1, x2, u.y);
}

// FBM = fractal Brownian motion (sum of noise octaves).
fn fbm(p: vec2<f32>) -> f32 {
  var x = p;
  var sum = 0.0;
  var amp = 0.5;

  let rot = mat2x2<f32>(0.8, -0.6, 0.6, 0.8);

  // 5 octaves: drop to 4 if you need more perf.
  for (var i: u32 = 0u; i < 5u; i = i + 1u) {
    sum += amp * value_noise(x);
    x = rot * x * 2.0 + vec2<f32>(17.0, 9.0);
    amp *= 0.5;
  }
  return sum; // ~[0..1]
}

// Cloud coverage [0..1] evaluated at a world-space point on the cloud plane.
fn cloud_coverage_at_xz(xz: vec2<f32>, time_s: f32) -> f32 {
  var uv = xz * CLOUD_UV_SCALE + CLOUD_WIND * time_s;

  let n  = fbm(uv);
  let n2 = fbm(uv * 2.3 + vec2<f32>(13.2, 7.1));
  let field = 0.65 * n + 0.35 * n2;

  return smoothstep(CLOUD_COVERAGE, CLOUD_COVERAGE + CLOUD_SOFTNESS, field);
}

// Sunlight transmittance due to clouds along ray starting at p toward sun_dir.
// 1.0 = clear, 0.0 = fully blocked (thick clouds).
fn cloud_sun_transmittance(p: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
  // If sun ray doesn't go upward, it won't intersect the cloud layer.
  if (sun_dir.y <= 0.01) { return 1.0; }

  let t = (CLOUD_H - p.y) / sun_dir.y;
  if (t <= 0.0) { return 1.0; }

  let time_s = cam.voxel_params.y;
  let hit = p + sun_dir * t;

  let cov = cloud_coverage_at_xz(hit.xz, time_s);

  // Exponential absorb: keeps gaps bright, thick areas dim.
  return exp(-CLOUD_ABSORB * cov);
}

// ------------------------------------------------------------
// Phase function for volumetrics / godrays
// ------------------------------------------------------------

// Henyey–Greenstein-ish phase approximation.
fn phase_mie(costh: f32) -> f32 {
  let g = PHASE_G;
  let gg = g * g;
  let denom = pow(1.0 + gg - 2.0 * g * costh, 1.5);
  return (1.0 - gg) / max(denom, 1e-3);
}

// Blend forward Mie-ish with isotropic so shafts show from side angles too.
fn phase_blended(costh: f32) -> f32 {
  let mie = phase_mie(costh);
  return mix(INV_4PI, mie, PHASE_MIE_W);
}

// ------------------------------------------------------------
// Sky with sun disc + fractal clouds
// ------------------------------------------------------------

fn sky_color(rd: vec3<f32>) -> vec3<f32> {
  // Base sky gradient
  let tsky = clamp(0.5 * (rd.y + 1.0), 0.0, 1.0);
  var col = mix(
    vec3<f32>(0.05, 0.08, 0.12),
    vec3<f32>(0.55, 0.75, 0.95),
    tsky
  );

  // Apply global sky exposure (dimmer overall sky)
  col *= SKY_EXPOSURE;

  // Sun disc + halo
  let mu  = dot(rd, SUN_DIR);
  let ang = acos(clamp(mu, -1.0, 1.0));
  let disc = 1.0 - smoothstep(
    SUN_DISC_ANGULAR_RADIUS,
    SUN_DISC_ANGULAR_RADIUS + SUN_DISC_SOFTNESS,
    ang
  );
  let halo = exp(-ang * 30.0) * 0.15;

  // Clouds (viewed against sky) only when looking upward-ish.
  var cloud = 0.0;

  if (rd.y > 0.01) {
    // Intersect view ray with cloud plane y=CLOUD_H
    let ro = cam.cam_pos.xyz;
    let t = (CLOUD_H - ro.y) / rd.y;

    if (t > 0.0) {
      let hit = ro + rd * t;
      let time_s = cam.voxel_params.y;

      cloud = cloud_coverage_at_xz(hit.xz, time_s);

      // Horizon fade
      let horizon = clamp((rd.y - CLOUD_HORIZON_Y0) / CLOUD_HORIZON_Y1, 0.0, 1.0);
      cloud *= horizon;

      // Darken underlying sky
      col *= mix(1.0, CLOUD_SKY_DARKEN, cloud);

      // Cloud color + silver lining toward sun
      let toward_sun = clamp(mu, 0.0, 1.0);
      let silver = pow(toward_sun, CLOUD_SILVER_POW) * CLOUD_SILVER_STR;
      let cloud_col = mix(CLOUD_BASE_COL, vec3<f32>(1.0), silver);

      col = mix(col, cloud_col, cloud * CLOUD_BLEND);
    }
  }

  // Optionally dim sun disc/halo when obscured by clouds in the view direction
  var sun_term = (disc + halo);
  if (CLOUD_DIM_SUN_DISC) {
    let Tc_view = exp(-CLOUD_ABSORB * cloud * CLOUD_SUN_DISC_ABSORB_SCALE);
    sun_term *= Tc_view;
  }

  // Add sun on top (so it still reads), but allow cloud dimming above
  col += SUN_COLOR * SUN_INTENSITY * sun_term;

  return col;
}

// ------------------------------------------------------------
// Fog / volumetrics helpers
// ------------------------------------------------------------

// Fog density comes from cam.voxel_params.w
fn fog_density_base() -> f32 {
  return max(cam.voxel_params.w, 0.0);
}

// Optical depth for exponential height fog integrated along [0..t].
// density(y) = base * exp(-k * y)
fn fog_optical_depth(ro: vec3<f32>, rd: vec3<f32>, t: f32) -> f32 {
  let base = fog_density_base();
  if (base <= 0.0) { return 0.0; }

  let k = FOG_HEIGHT_FALLOFF;
  let y0 = ro.y;
  let dy = rd.y;

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
