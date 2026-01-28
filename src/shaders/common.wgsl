// common.wgsl
//
// Shared WGSL across compute passes.
// - Constants / tuning knobs (ALL constants live here)
// - GPU-side struct defs + scene bindings
// - Shared math + grid helpers + sky/cloud/fog utilities

// ------------------------------------------------------------
// IDs / numeric
// ------------------------------------------------------------

const LEAF_U32 : u32 = 0xFFFFFFFFu; // Sentinel for "leaf node" in child_base.
const INVALID_U32 : u32 = 0xFFFFFFFFu; // Sentinel for invalid grid slot (chunk_grid entry).

const BIG_F32  : f32 = 1e30;
const EPS_INV  : f32 = 1e-8;

// Materials (keeps magic numbers out of core code)
const MAT_AIR   : u32 = 0u;
const MAT_GRASS : u32 = 1u;
const MAT_DIRT  : u32 = 2u;
const MAT_STONE : u32 = 3u;
const MAT_WOOD  : u32 = 4u;
const MAT_LEAF  : u32 = 5u;

// ------------------------------------------------------------
// Sun / sky
// ------------------------------------------------------------

const SUN_DIR : vec3<f32> = vec3<f32>(0.61237244, 0.5, 0.61237244);
const SUN_COLOR     : vec3<f32> = vec3<f32>(1.0, 0.98, 0.90);
const SUN_INTENSITY : f32 = 3.5;

const SUN_DISC_ANGULAR_RADIUS : f32 = 0.009;
const SUN_DISC_SOFTNESS       : f32 = 0.004;

const SKY_EXPOSURE : f32 = 0.40;

// ------------------------------------------------------------
// Shadows
// ------------------------------------------------------------

const SHADOW_BIAS : f32 = 2e-4;

// Shadow traversal tuning
const SHADOW_STEPS : u32 = 32u;

// If false: leaves cast shadows using their undisplaced cube (faster).
// If true : shadows match displaced leaf cubes (slower but consistent).
const SHADOW_DISPLACED_LEAVES : bool = false;

// Volumetric “sun transmittance” tuning (leafy canopy)
const VSM_STEPS : u32 = 24u;
const LEAF_LIGHT_TRANSMIT : f32 = 0.50;
const MIN_TRANS : f32 = 0.03;

// ------------------------------------------------------------
// Fog / volumetrics
// ------------------------------------------------------------

const FOG_HEIGHT_FALLOFF : f32 = 0.18;
const FOG_MAX_DIST       : f32 = 100.0;

const FOG_PRIMARY_SCALE : f32 = 0.02;
const FOG_GODRAY_SCALE  : f32 = 2.0;

const FOG_PRIMARY_VIS   : f32 = 0.08;

const FOG_COLOR_GROUND     : vec3<f32> = vec3<f32>(0.62, 0.64, 0.66);
const FOG_COLOR_SKY_BLEND  : f32 = 0.20;

const GODRAY_MAX_DIST    : f32 = 80.0;
const GODRAY_STRENGTH    : f32 = 14.0;

const GODRAY_OFFAXIS_POW : f32 = 3.0;
const GODRAY_OFFAXIS_W   : f32 = 0.18;

// Godray scattering height behavior (ONLY affects added beam light, not fog density)
const GODRAY_SCATTER_HEIGHT_FALLOFF : f32 = 0.04; // << smaller than FOG_HEIGHT_FALLOFF (0.18)
const GODRAY_SCATTER_MIN_FRAC       : f32 = 0.35; // floor as fraction of sea-level scatter

const GODRAY_SIDE_BOOST : f32 = 0.75; // 0..1
const GODRAY_BLACK_LEVEL : f32 = 0.010; // try 0.010..0.030

const GODRAY_TS_LP_ALPHA   : f32 = 0.30; // 0.2..0.5 (higher = smoother, less noisy)
const GODRAY_EDGE0         : f32 = 0.004;
const GODRAY_EDGE1         : f32 = 0.035;

const GODRAY_BASE_HAZE     : f32 = 0.02; // 0.02..0.10 (tiny DC term)
const GODRAY_HAZE_NEAR_FADE: f32 = 18.0; // meters: haze ramps in with distance

const CLOUD_GODRAY_W : f32 = 0.25; // 0..1 (how much clouds affect godrays brightness)

const INV_4PI      : f32 = 0.0795774715;
const PHASE_G      : f32 = 0.25;
const PHASE_MIE_W  : f32 = 0.30;

// ------------------------------------------------------------
// Fractal clouds
// ------------------------------------------------------------

const CLOUD_H : f32 = 200.0;
const CLOUD_UV_SCALE : f32 = 0.002;
const CLOUD_WIND : vec2<f32> = vec2<f32>(0.020, 0.012);

const CLOUD_COVERAGE : f32 = 0.45;
const CLOUD_SOFTNESS : f32 = 0.10;

const CLOUD_HORIZON_Y0 : f32 = 0.02;
const CLOUD_HORIZON_Y1 : f32 = 0.25;

const CLOUD_SKY_DARKEN : f32 = 0.95;
const CLOUD_ABSORB : f32 = 10.0;

const CLOUD_BASE_COL   : vec3<f32> = vec3<f32>(0.72, 0.74, 0.76);
const CLOUD_SILVER_POW : f32 = 8.0;
const CLOUD_SILVER_STR : f32 = 0.6;
const CLOUD_BLEND      : f32 = 0.85;

const CLOUD_DIM_SUN_DISC : bool = true;
const CLOUD_SUN_DISC_ABSORB_SCALE : f32 = 0.8;

// ------------------------------------------------------------
// Leaf wind (displaced cubes)
// ------------------------------------------------------------

const WIND_CELL_FREQ : f32 = 2.5;
const WIND_DIR_XZ : vec2<f32> = vec2<f32>(0.9, 0.4);

const WIND_RAMP_Y0 : f32 = 2.0;
const WIND_RAMP_Y1 : f32 = 14.0;

const WIND_GUST_TIME_FREQ    : f32 = 0.9;
const WIND_FLUTTER_TIME_FREQ : f32 = 4.2;

const WIND_GUST_XZ_FREQ    : vec2<f32> = vec2<f32>(0.35, 0.22);
const WIND_FLUTTER_XZ_FREQ : vec2<f32> = vec2<f32>(1.7,  1.1);

const WIND_GUST_WEIGHT    : f32 = 0.75;
const WIND_FLUTTER_WEIGHT : f32 = 0.25;

const WIND_VERTICAL_SCALE : f32 = 0.25;
const LEAF_VERTICAL_REDUCE : f32 = 0.15;

const LEAF_OFFSET_AMP : f32 = 0.75;
const LEAF_OFFSET_MAX_FRAC : f32 = 0.75;

const WIND_PHASE_OFF_1 : vec3<f32> = vec3<f32>(19.0, 7.0, 11.0);
const TAU : f32 = 6.28318530718;

// ------------------------------------------------------------
// Ray/main pass knobs (moved from ray_main)
// ------------------------------------------------------------

const PRIMARY_NUDGE_VOXEL_FRAC : f32 = 1e-4;

// Godray sampling pattern
const GODRAY_FRAME_FPS : f32 = 60.0;
const GODRAY_BLOCK_SIZE : i32 = 4;
const GODRAY_PATTERN_HASH_SCALE : f32 = 0.73;

const J0_SCALE : f32 = 1.31;
const J1_SCALE : f32 = 2.11;
const J2_SCALE : f32 = 3.01;
const J3_SCALE : f32 = 4.19;

const J0_F : vec2<f32> = vec2<f32>(0.11, 0.17);
const J1_F : vec2<f32> = vec2<f32>(0.23, 0.29);
const J2_F : vec2<f32> = vec2<f32>(0.37, 0.41);
const J3_F : vec2<f32> = vec2<f32>(0.53, 0.59);

const GODRAY_TV_CUTOFF : f32 = 0.02;
const GODRAY_STEPS_FAST : u32 = 8u;

// Composite
const COMPOSITE_SHARPEN : f32 = 0.15;
const COMPOSITE_GOD_SCALE : f32 = 8.00;
const COMPOSITE_BEAM_COMPRESS : bool = true;

// Post
const POST_EXPOSURE : f32 = 0.15;

// ------------------------------------------------------------
// GPU structs (must match Rust layouts)
// ------------------------------------------------------------

struct Node {
  child_base : u32,
  child_mask : u32,
  material   : u32,
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

  // x = voxel_size_m, y = time_seconds, z = wind_strength, w = fog_density
  voxel_params : vec4<f32>,

  grid_origin_chunk : vec4<i32>,
  grid_dims         : vec4<u32>,
};

struct ChunkMeta {
  origin     : vec4<i32>,
  node_base  : u32,
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
// Ray reconstruction
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
// AABB intersection (slab)
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
// Chunk-grid helpers (moved from ray_main; used by shadows too)
// ------------------------------------------------------------

fn grid_lookup_slot(cx: i32, cy: i32, cz: i32) -> u32 {
  let ox = cam.grid_origin_chunk.x;
  let oy = cam.grid_origin_chunk.y;
  let oz = cam.grid_origin_chunk.z;

  let ix_i = cx - ox;
  let iy_i = cy - oy;
  let iz_i = cz - oz;

  if (ix_i < 0 || iy_i < 0 || iz_i < 0) { return INVALID_U32; }

  let nx = cam.grid_dims.x;
  let ny = cam.grid_dims.y;
  let nz = cam.grid_dims.z;

  let ix = u32(ix_i);
  let iy = u32(iy_i);
  let iz = u32(iz_i);

  if (ix >= nx || iy >= ny || iz >= nz) { return INVALID_U32; }

  let idx = (iz * ny * nx) + (iy * nx) + ix;
  return chunk_grid[idx];
}

fn chunk_coord_from_pos(p: vec3<f32>, chunk_size_m: f32) -> vec3<i32> {
  return vec3<i32>(
    i32(floor(p.x / chunk_size_m)),
    i32(floor(p.y / chunk_size_m)),
    i32(floor(p.z / chunk_size_m))
  );
}

// ------------------------------------------------------------
// Hash / noise / FBM (clouds)
// ------------------------------------------------------------

fn hash12(p: vec2<f32>) -> f32 {
  let h = dot(p, vec2<f32>(127.1, 311.7));
  return fract(sin(h) * 43758.5453);
}

fn hash21(p: vec2<f32>) -> f32 {
  let h = dot(p, vec2<f32>(127.1, 311.7));
  return fract(sin(h) * 43758.5453);
}

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

fn fbm(p: vec2<f32>) -> f32 {
  var x = p;
  var sum = 0.0;
  var amp = 0.5;

  let rot = mat2x2<f32>(0.8, -0.6, 0.6, 0.8);

  for (var i: u32 = 0u; i < 5u; i = i + 1u) {
    sum += amp * value_noise(x);
    x = rot * x * 2.0 + vec2<f32>(17.0, 9.0);
    amp *= 0.5;
  }
  return sum;
}

fn cloud_coverage_at_xz(xz: vec2<f32>, time_s: f32) -> f32 {
  var uv = xz * CLOUD_UV_SCALE + CLOUD_WIND * time_s;

  let n  = fbm(uv);
  let n2 = fbm(uv * 2.3 + vec2<f32>(13.2, 7.1));
  let field = 0.65 * n + 0.35 * n2;

  return smoothstep(CLOUD_COVERAGE, CLOUD_COVERAGE + CLOUD_SOFTNESS, field);
}

fn cloud_sun_transmittance(p: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
  if (sun_dir.y <= 0.01) { return 1.0; }

  let t = (CLOUD_H - p.y) / sun_dir.y;
  if (t <= 0.0) { return 1.0; }

  let time_s = cam.voxel_params.y;
  let hit = p + sun_dir * t;
  let cov = cloud_coverage_at_xz(hit.xz, time_s);
  return exp(-CLOUD_ABSORB * cov);
}

// ------------------------------------------------------------
// Phase functions
// ------------------------------------------------------------

fn phase_mie(costh: f32) -> f32 {
  let g = PHASE_G;
  let gg = g * g;
  let denom = pow(1.0 + gg - 2.0 * g * costh, 1.5);
  // HG normalized by 1/(4π)
  return INV_4PI * (1.0 - gg) / max(denom, 1e-3);
}

fn phase_blended(costh: f32) -> f32 {
  let mie = phase_mie(costh);              // now already normalized
  return mix(INV_4PI, mie, PHASE_MIE_W);   // isotropic ↔ mie
}


// ------------------------------------------------------------
// Sky
// ------------------------------------------------------------

fn sky_color(rd: vec3<f32>) -> vec3<f32> {
  let tsky = clamp(0.5 * (rd.y + 1.0), 0.0, 1.0);
  var col = mix(
    vec3<f32>(0.05, 0.08, 0.12),
    vec3<f32>(0.55, 0.75, 0.95),
    tsky
  );

  col *= SKY_EXPOSURE;

  let mu  = dot(rd, SUN_DIR);
  let ang = acos(clamp(mu, -1.0, 1.0));
  let disc = 1.0 - smoothstep(
    SUN_DISC_ANGULAR_RADIUS,
    SUN_DISC_ANGULAR_RADIUS + SUN_DISC_SOFTNESS,
    ang
  );
  let halo = exp(-ang * 30.0) * 0.15;

  var cloud = 0.0;

  if (rd.y > 0.01) {
    let ro = cam.cam_pos.xyz;
    let t = (CLOUD_H - ro.y) / rd.y;

    if (t > 0.0) {
      let hit = ro + rd * t;
      let time_s = cam.voxel_params.y;

      cloud = cloud_coverage_at_xz(hit.xz, time_s);

      let horizon = clamp((rd.y - CLOUD_HORIZON_Y0) / CLOUD_HORIZON_Y1, 0.0, 1.0);
      cloud *= horizon;

      col *= mix(1.0, CLOUD_SKY_DARKEN, cloud);

      let toward_sun = clamp(mu, 0.0, 1.0);
      let silver = pow(toward_sun, CLOUD_SILVER_POW) * CLOUD_SILVER_STR;
      let cloud_col = mix(CLOUD_BASE_COL, vec3<f32>(1.0), silver);

      col = mix(col, cloud_col, cloud * CLOUD_BLEND);
    }
  }

  var sun_term = (disc + halo);
  if (CLOUD_DIM_SUN_DISC) {
    let Tc_view = exp(-CLOUD_ABSORB * cloud * CLOUD_SUN_DISC_ABSORB_SCALE);
    sun_term *= Tc_view;
  }

  col += SUN_COLOR * SUN_INTENSITY * sun_term;
  return col;
}

// Fog color used by primary composition
fn fog_color(rd: vec3<f32>) -> vec3<f32> {
  let up = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
  let sky = sky_color(rd);
  return mix(FOG_COLOR_GROUND, sky, FOG_COLOR_SKY_BLEND * up);
}

// ------------------------------------------------------------
// Fog helpers
// ------------------------------------------------------------

fn fog_density_primary() -> f32 {
  return max(cam.voxel_params.w * FOG_PRIMARY_SCALE, 0.0);
}

fn fog_density_godray() -> f32 {
  return max(cam.voxel_params.w * FOG_GODRAY_SCALE, 0.0);
}

fn fog_optical_depth_with_base(base: f32, ro: vec3<f32>, rd: vec3<f32>, t: f32) -> f32 {
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

fn fog_transmittance_primary(ro: vec3<f32>, rd: vec3<f32>, t: f32) -> f32 {
  let od = max(fog_optical_depth_with_base(fog_density_primary(), ro, rd, t), 0.0);
  return exp(-od);
}

fn fog_transmittance_godray(ro: vec3<f32>, rd: vec3<f32>, t: f32) -> f32 {
  let od = max(fog_optical_depth_with_base(fog_density_godray(), ro, rd, t), 0.0);
  return exp(-od);
}
