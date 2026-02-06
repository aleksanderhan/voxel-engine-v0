// src/shaders/common.wgsl
// -----------------------
// src/shaders/common.wgsl
//
// Shared WGSL:
// - Constants / tuning knobs
// - GPU-side struct defs + scene bindings (group 0)
// - Shared math + grid helpers + hashes/noise
//
// NOTE: Pass-specific textures/images + compute entrypoints live in ray_main.wgsl.

//// --------------------------------------------------------------------------
//// IDs / numeric
//// --------------------------------------------------------------------------

const LEAF_U32    : u32 = 0xFFFFFFFFu; // Node.child_base sentinel
const INVALID_U32 : u32 = 0xFFFFFFFFu;

const BIG_F32 : f32 = 1e30;
const EPS_INV : f32 = 1e-8;

//// --------------------------------------------------------------------------
//// Feature toggles
//// --------------------------------------------------------------------------

const ENABLE_GRASS   : bool = true;
const ENABLE_CLIPMAP : bool = true;
const ENABLE_GODRAYS : bool = true;
const ENABLE_CLOUDS  : bool = true;
const ENABLE_FOG     : bool = true;
const ENABLE_BLOOM   : bool = true;

//// --------------------------------------------------------------------------
//// Materials
//// --------------------------------------------------------------------------

const MAT_AIR   : u32 = 0u;
const MAT_GRASS : u32 = 1u;
const MAT_DIRT  : u32 = 2u;
const MAT_STONE : u32 = 3u;
const MAT_WOOD  : u32 = 4u;
const MAT_LEAF  : u32 = 5u;
const MAT_LIGHT : u32 = 6u;

//// --------------------------------------------------------------------------
//// Sun / sky (shared, but shading logic lives in ray_core.wgsl)
//// --------------------------------------------------------------------------

const SUN_DIR       : vec3<f32> = vec3<f32>(0.61237244, 0.5, 0.61237244);
const SUN_COLOR     : vec3<f32> = vec3<f32>(1.0, 0.94, 0.72);
const SUN_INTENSITY : f32       = 5.0;

//// --------------------------------------------------------------------------
//// Shadows / volumetric shadowing
//// --------------------------------------------------------------------------

const SHADOW_BIAS  : f32 = 2e-4;

const VOLUME_DISPLACED_LEAVES : bool = true;

const VSM_STEPS            : u32 = 16u;
const LEAF_LIGHT_TRANSMIT  : f32 = 0.50;
const GRASS_LIGHT_TRANSMIT : f32 = 0.70;
const MIN_TRANS            : f32 = 0.03;

//// --------------------------------------------------------------------------
//// Godrays (tuning knobs)
//// --------------------------------------------------------------------------

const GODRAY_ENERGY_BOOST    : f32 = 8.0;
const GODRAY_KNEE_INTEGRATE  : f32 = 0.35;
const GODRAY_COMPOSITE_SCALE : f32 = 6.5;

const GODRAY_FADE_NEAR : f32 = 40.0;
const GODRAY_FADE_FAR  : f32 = 80.0;

const GODRAY_KNEE_COMPOSITE : f32 = 0.25;

//// --------------------------------------------------------------------------
//// Fog / volumetrics
//// --------------------------------------------------------------------------

const FOG_HEIGHT_FALLOFF : f32 = 0.18;
const FOG_MAX_DIST       : f32 = 120.0;

const FOG_PRIMARY_SCALE : f32 = 0.02;
const FOG_GODRAY_SCALE  : f32 = 1.0;

const FOG_COLOR_GROUND    : vec3<f32> = vec3<f32>(0.40, 0.42, 0.45);
const FOG_COLOR_SKY_BLEND : f32       = 0.10;

const GODRAY_MAX_DIST : f32 = 100.0;

const GODRAY_SCATTER_HEIGHT_FALLOFF : f32 = 0.04;
const GODRAY_SCATTER_MIN_FRAC       : f32 = 0.35;

const GODRAY_BLACK_LEVEL : f32 = 0.004;

const GODRAY_TS_LP_ALPHA : f32 = 0.40;
const GODRAY_EDGE0       : f32 = 0.004;
const GODRAY_EDGE1       : f32 = 0.035;

const GODRAY_BASE_HAZE       : f32 = 0.02;
const GODRAY_HAZE_NEAR_FADE  : f32 = 18.0;

const GODRAY_TV_CUTOFF   : f32 = 0.02;

// If near-camera shafts look under-sampled: MIN_STEPS 8 → 10/12
// If mid/far looks under-sampled: STEPS_PER_METER 1.0 → 1.25
// If still too slow: STEPS_PER_METER 1.0 → 0.75 (keep MIN_STEPS)
const GODRAY_STEPS_FAST  : u32 = 24u;
const GODRAY_STEPS_PER_METER = 1.25;
const GODRAY_MIN_STEPS: u32 = 8u;        // keep some detail near silhouettes
const GODRAY_STEP_Q:   u32 = 4u;         // quantize step count to reduce temporal shimmer

const GODRAY_SHAFT_GAIN: f32 = 3.0;

const GODRAY_EDGE_ENERGY_BOOST: f32 = 2.5; // try 1.0 .. 4.0

//// --------------------------------------------------------------------------
//// Phase
//// --------------------------------------------------------------------------

const INV_4PI     : f32 = 0.0795774715;
const PHASE_G     : f32 = 0.15;
const PHASE_MIE_W : f32 = 0.20;

//// --------------------------------------------------------------------------
//// Clouds (cheap volumetric slab)
//// --------------------------------------------------------------------------

// Vertical slab in meters
const CLOUD_BASE_H : f32 = 170.0;
const CLOUD_TOP_H  : f32 = 250.0;

// Horizontal scale and wind
const CLOUD_UV_SCALE : f32       = 0.0016;
const CLOUD_WIND     : vec2<f32> = vec2<f32>(0.020, 0.012);

// Coverage thresholding (raise coverage => fewer clouds)
const CLOUD_COVERAGE : f32 = 0.40;
const CLOUD_SOFTNESS : f32 = 0.08;

// Density + shaping
const CLOUD_DENSITY     : f32 = 0.058; // extinction scale (bigger = thicker/darker)
const CLOUD_PUFF_POW    : f32 = 1.8;   // >1 makes denser cores, airier edges
const CLOUD_DETAIL_W    : f32 = 0.35;  // detail noise weight

// March quality
const CLOUD_STEPS_VIEW  : u32 = 8u;  // view ray steps in sky
const CLOUD_STEPS_LIGHT : u32 = 4u;  // sun-light shadow steps

// Horizon fade (keep)
const CLOUD_HORIZON_Y0 : f32 = 0.02;
const CLOUD_HORIZON_Y1 : f32 = 0.25;

// Appearance knobs
const CLOUD_BASE_COL   : vec3<f32> = vec3<f32>(0.72, 0.74, 0.76);
const CLOUD_SILVER_POW : f32       = 8.0;
const CLOUD_SILVER_STR : f32       = 0.6;

// How much clouds attenuate SUNLIGHT hitting the world
const CLOUD_SHADOW_ABSORB   : f32 = 6.0;
const CLOUD_SHADOW_STRENGTH : f32 = 0.8;

// Sun-disc dim behavior (keep)
const CLOUD_DIM_SUN_DISC : bool = true;

// NEW: make sun-disc dim track *visual opacity* (T_view) strongly.
// If a cloud looks opaque-ish, the disc should vanish quickly.
const CLOUD_SUN_DISC_DIM_POW   : f32 = 8.0;  // try 4..12 (bigger = dimmer disc)
const CLOUD_SUN_DISC_DIM_FLOOR : f32 = 0.0;  // set ~0.01 if you want “always visible” sun

// Godray coupling (keep)
const CLOUD_GODRAY_W : f32 = 0.50;

const SKY_EXPOSURE : f32 = 0.40;

//// --------------------------------------------------------------------------
//// Leaf wind (displaced cubes)
//// --------------------------------------------------------------------------

const WIND_CELL_FREQ : f32       = 2.5;
const WIND_DIR_XZ    : vec2<f32> = vec2<f32>(0.9, 0.4);

const WIND_RAMP_Y0 : f32 = 2.0;
const WIND_RAMP_Y1 : f32 = 14.0;

const WIND_GUST_TIME_FREQ    : f32 = 0.9;
const WIND_FLUTTER_TIME_FREQ : f32 = 4.2;

const WIND_GUST_XZ_FREQ    : vec2<f32> = vec2<f32>(0.35, 0.22);
const WIND_FLUTTER_XZ_FREQ : vec2<f32> = vec2<f32>(1.7,  1.1);

const WIND_GUST_WEIGHT    : f32 = 0.75;
const WIND_FLUTTER_WEIGHT : f32 = 0.25;

const WIND_VERTICAL_SCALE  : f32 = 0.25;
const LEAF_VERTICAL_REDUCE : f32 = 0.15;

const LEAF_OFFSET_AMP       : f32 = 0.75;
const LEAF_OFFSET_MAX_FRAC  : f32 = 0.75;

const LEAF_LOD_DISP_START : f32 = 25.0;
const LEAF_LOD_DISP_END   : f32 = 70.0;

const WIND_PHASE_OFF_1 : vec3<f32> = vec3<f32>(19.0, 7.0, 11.0);
const TAU             : f32        = 6.28318530718;

//// --------------------------------------------------------------------------
//// Ray / post
//// --------------------------------------------------------------------------

const PRIMARY_NUDGE_VOXEL_FRAC : f32 = 1e-4;

// Primary hit cache (temporal reprojection) tuning.
const PRIMARY_HIT_MARGIN        : f32 = 0.08;
const PRIMARY_HIT_WINDOW        : f32 = 0.60;
const PRIMARY_HIT_DEPTH_REL0    : f32 = 0.04;
const PRIMARY_HIT_DEPTH_REL1    : f32 = 0.12;

// Sun shadow temporal reuse (primary shading).
const SHADOW_TAA_ALPHA      : f32 = 0.20;
const SHADOW_SUBSAMPLE_MASK : u32 = 7u; // 0b111 => 1/8 pixels per frame (blue-noise via seed)
const PRIMARY_HIT_MOTION_PX0    : f32 = 0.50;
const PRIMARY_HIT_MOTION_PX1    : f32 = 1.75;

const J0_SCALE : f32 = 1.31;
const J1_SCALE : f32 = 2.11;
const J2_SCALE : f32 = 3.01;
const J3_SCALE : f32 = 4.19;

const COMPOSITE_SHARPEN : f32 = 0.1;

const POST_EXPOSURE : f32 = 0.15;

//// --------------------------------------------------------------------------
//// Clipmap heightfield tuning
//// --------------------------------------------------------------------------

const CLIP_LEVELS_MAX : u32 = 16u;

// March tuning
const HF_MAX_STEPS : u32 = 96u;
const HF_BISECT    : u32 = 5u;

// dt clamp (meters along ray)
const HF_DT_MAX : f32 = 48.0;

//// --------------------------------------------------------------------------
//// Local voxel light gather (MAT_LIGHT)
//// --------------------------------------------------------------------------

// How far rays march in voxels (controls “search radius” for lights)
const LIGHT_MAX_DIST_VOX : u32 = 32u;   // try 16..64

// Number of rays
const LIGHT_RAYS : u32 = 24u;          // try 16..32

// Softens inverse-square near the light (in voxels)
const LIGHT_SOFT_RADIUS_VOX : f32 = 3.0;

// Clamp minimum distance to avoid blow-ups (in voxels)
const LIGHT_NEAR_CLAMP_VOX : f32 = 1.25;

// Finite range rolloff (in voxels). Past this, contributions fade to ~0.
const LIGHT_RANGE_VOX : f32 = 80.0;     // try 32..96

// Direct diffuse “wrap” (0 = pure Lambert, 0.1..0.25 = nicer in caves)
const LIGHT_WRAP : f32 = 0.15;

// Gains
const LIGHT_DIRECT_GAIN   : f32 = 1.00;
const LIGHT_INDIRECT_GAIN : f32 = 0.65; // cheap “bounce fill”

// Stop after N light hits (perf only; output is normalized with LIGHT_RAYS)
const LIGHT_EARLY_HITS : u32 = 24u;

//// --------------------------------------------------------------------------
//// Shading gates (world-space distance)
//// --------------------------------------------------------------------------

const VOXEL_AO_MAX_DIST     : f32 = 40.0;
const LOCAL_LIGHT_MAX_DIST  : f32 = 50.0;
const FAR_SHADING_DIST      : f32 = 64.0;
const PRIMARY_CLOUD_SHADOWS : bool = false;

//// --------------------------------------------------------------------------
//// Local TAA
//// --------------------------------------------------------------------------

// Tune this: lower = steadier but slower response
const LOCAL_TAA_ALPHA : f32 = 0.12;

//// --------------------------------------------------------------------------
//// Grass “hair” (procedural blades)
//// --------------------------------------------------------------------------

const GRASS_LAYER_HEIGHT_VOX      : f32 = 1.20;
const GRASS_BLADE_COUNT           : u32 = 2u;
const GRASS_TRACE_STEPS           : u32 = 7u;
const GRASS_HIT_EPS_VOX           : f32 = 0.02;
const GRASS_STEP_MIN_VOX          : f32 = 0.03;

const GRASS_VOXEL_SEGS            : f32 = 3.0;
const GRASS_VOXEL_THICKNESS_VOX   : f32 = 0.08;
const GRASS_VOXEL_TAPER           : f32 = 0.70;
const GRASS_OVERHANG_VOX          : f32 = 0.20;

const GRASS_LOD_MID_START : f32 = 15.0;
const GRASS_LOD_FAR_START : f32 = 40.0;

const GRASS_BLADE_COUNT_MID : u32 = 2u;
const GRASS_BLADE_COUNT_FAR : u32 = 1u;

const GRASS_SEGS_MID : u32 = 2u;
const GRASS_SEGS_FAR : u32 = 1u;

const GRASS_TRACE_STEPS_MID : u32 = 6u;
const GRASS_TRACE_STEPS_FAR : u32 = 4u;

// Primary-pass grass gating (tune these)
const GRASS_PRIMARY_MAX_DIST : f32 = 14.0; // meters-ish
const GRASS_PRIMARY_MIN_NY   : f32 = 0.60; // only fairly upward normals
const GRASS_PRIMARY_RATE_MASK: u32 = 7u;   // 0 => all pixels, 1 => 1/2, 3 => 1/4, 7 => 1/8 ...


// Misc
const ALBEDO_VAR_GAIN = 3.5;

//// --------------------------------------------------------------------------
//// GPU structs (must match Rust layouts)
//// --------------------------------------------------------------------------

struct Node {
  child_base : u32,
  child_mask : u32,
  material   : u32,
  key       : u32,
};

struct NodeRopes {
  px: u32,
  nx: u32,
  py: u32,
  ny: u32,
  pz: u32,
  nz: u32,
  _pad0: u32,
  _pad1: u32,
};

struct Camera {
  view_inv    : mat4x4<f32>,
  proj_inv    : mat4x4<f32>,

  view_proj: mat4x4<f32>,
  prev_view_proj: mat4x4<f32>,

  cam_pos     : vec4<f32>,
  ray00       : vec4<f32>,
  ray_dx      : vec4<f32>,
  ray_dy      : vec4<f32>,

  chunk_size  : u32,
  chunk_count : u32,
  max_steps   : u32,
  frame_index : u32,

  voxel_params : vec4<f32>,

  grid_origin_chunk : vec4<i32>,
  grid_dims         : vec4<u32>,

  // xy = render size in pixels, zw = present size in pixels
  render_present_px : vec4<u32>,
};

struct ChunkMeta {
  origin       : vec4<i32>,
  node_base    : u32,
  node_count   : u32,
  macro_base   : u32,
  colinfo_base : u32,
  macro_empty  : u32,
  _pad0        : u32,
  _pad1        : u32,
  _pad2        : u32,
};

//// --------------------------------------------------------------------------
//// Scene bindings (group 0) - shared across passes
//// --------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> cam : Camera;
@group(0) @binding(1) var<storage, read> chunks     : array<ChunkMeta>;
@group(0) @binding(2) var<storage, read> nodes      : array<Node>;
@group(0) @binding(3) var<storage, read> chunk_grid : array<u32>;
@group(0) @binding(9)  var<storage, read> macro_occ : array<u32>;
@group(0) @binding(10) var<storage, read> node_ropes: array<NodeRopes>;
@group(0) @binding(11) var<storage, read> chunk_colinfo : array<u32>;


//// --------------------------------------------------------------------------
//// Shared helpers
//// --------------------------------------------------------------------------

const MACRO_DIM : u32 = 8u;              // 8x8x8 macro cells per chunk
const MACRO_WORDS_PER_CHUNK : u32 = 16u; // 512 bits / 32
const TILE_SIZE: u32 = 8u;
const MAX_TILE_CHUNKS: u32 = 256u;
const PRIMARY_MAX_TILE_CHUNKS: u32 = 96u;

var<workgroup> WG_TILE_COUNT : atomic<u32>;
var<workgroup> WG_TILE_SLOTS : array<u32, MAX_TILE_CHUNKS>;

fn macro_cell_size(root_size: f32) -> f32 {
  return root_size / f32(MACRO_DIM);
}

// bit index = mx + 8*(my + 8*mz) in [0..511]
fn macro_bit_index(mx: u32, my: u32, mz: u32) -> u32 {
  return mx + MACRO_DIM * (my + MACRO_DIM * mz);
}

fn macro_test(macro_base: u32, bit: u32) -> bool {
  let w = bit >> 5u;
  let b = bit & 31u;
  let word = macro_occ[macro_base + w];
  return (word & (1u << b)) != 0u;
}

fn safe_inv(x: f32) -> f32 {
  return select(1.0 / x, BIG_F32, abs(x) < EPS_INV);
}

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
  let l2 = dot(v, v);
  // 1e-12 is conservative for f32; tweak if you want.
  if (l2 <= 1e-12) {
    return vec3<f32>(0.0, 1.0, 0.0); // arbitrary stable fallback
  }
  return v * inverseSqrt(l2);
}


fn ray_dir_from_pixel(px: vec2<f32>) -> vec3<f32> {
  let d = cam.ray00.xyz + px.x * cam.ray_dx.xyz + px.y * cam.ray_dy.xyz;
  return normalize(d);
}

fn prev_uv_from_world(p_ws: vec3<f32>) -> vec2<f32> {
  let clip = cam.prev_view_proj * vec4<f32>(p_ws, 1.0);
  let invw = 1.0 / max(clip.w, 1e-6);
  let ndc  = clip.xy * invw;          // -1..+1
  return ndc * 0.5 + vec2<f32>(0.5);  // 0..1
}

fn in_unit_square(uv: vec2<f32>) -> bool {
  return all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0));
}

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

fn child_rank(mask: u32, ci: u32) -> u32 {
  let bit = 1u << ci;
  let lower = mask & (bit - 1u);
  return countOneBits(lower);
}

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
  // Bias inward to avoid precision-induced chunk flips at exact boundaries.
  let eps = 1e-6 * chunk_size_m;
  let px = p.x - sign(p.x) * eps;
  let py = p.y - sign(p.y) * eps;
  let pz = p.z - sign(p.z) * eps;
  return vec3<i32>(
    i32(floor(px / chunk_size_m)),
    i32(floor(py / chunk_size_m)),
    i32(floor(pz / chunk_size_m))
  );
}

fn chunk_coord_from_pos_dir(p: vec3<f32>, rd: vec3<f32>, chunk_size_m: f32) -> vec3<i32> {
  // Bias along ray direction to stay on the same side of boundaries during DDA.
  let eps = 1e-6 * chunk_size_m;
  let px = p.x + sign(rd.x) * eps;
  let py = p.y + sign(rd.y) * eps;
  let pz = p.z + sign(rd.z) * eps;
  return vec3<i32>(
    i32(floor(px / chunk_size_m)),
    i32(floor(py / chunk_size_m)),
    i32(floor(pz / chunk_size_m))
  );
}

fn chunk_max_depth() -> u32 {
  // chunk_size is power-of-two; log2 = 31 - clz
  return 31u - countLeadingZeros(cam.chunk_size);
}

// ---- Column info (64x64) ----

fn col_idx_64(ix: u32, iz: u32) -> u32 {
  return iz * 64u + ix; // 0..4095
}

// returns (y8, mat8). y8==255 => empty column.
fn colinfo_load(ch: ChunkMeta, ix: u32, iz: u32) -> vec2<u32> {
  let idx  = col_idx_64(ix, iz);
  let word = chunk_colinfo[ch.colinfo_base + (idx >> 1u)];
  let half = select(word & 0xFFFFu, (word >> 16u) & 0xFFFFu, (idx & 1u) != 0u);

  let y8   = half & 0xFFu;
  let mat8 = (half >> 8u) & 0xFFu;
  return vec2<u32>(y8, mat8);
}

//// --------------------------------------------------------------------------
//// Hash / noise helpers (shared)
//// --------------------------------------------------------------------------

const U32_TO_F01 : f32 = 1.0 / 4294967296.0; // 2^-32

fn hash_u32(x: u32) -> u32 {
  // Murmur3 finalizer-like mix (good diffusion, cheap)
  var v = x;
  v ^= v >> 16u;
  v *= 0x7feb352du;
  v ^= v >> 15u;
  v *= 0x846ca68bu;
  v ^= v >> 16u;
  return v;
}

fn hash2_u32(x: u32, y: u32) -> u32 {
  // simple combine then mix
  return hash_u32(x * 0x8da6b343u ^ y * 0xd8163841u);
}

// Replacement: 2D -> [0,1)
fn hash12(p: vec2<f32>) -> f32 {
  let ix: u32 = bitcast<u32>(i32(floor(p.x)));
  let iy: u32 = bitcast<u32>(i32(floor(p.y)));
  let h: u32 = hash2_u32(ix, iy);
  return f32(h) * U32_TO_F01;
}

fn hash3_u32(x: u32, y: u32, z: u32) -> u32 {
  return hash_u32(x * 0x8da6b343u ^ y * 0xd8163841u ^ z * 0xcb1ab31fu);
}

fn hash31(p: vec3<f32>) -> f32 {
  let ix: u32 = bitcast<u32>(i32(floor(p.x)));
  let iy: u32 = bitcast<u32>(i32(floor(p.y)));
  let iz: u32 = bitcast<u32>(i32(floor(p.z)));
  let h: u32 = hash3_u32(ix, iy, iz);
  return f32(h) * U32_TO_F01;
}

fn value_noise(p: vec2<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);

  let a = hash12(i);
  let b = hash12(i + vec2<f32>(1.0, 0.0));
  let c = hash12(i + vec2<f32>(0.0, 1.0));
  let d = hash12(i + vec2<f32>(1.0, 1.0));

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

fn render_dims_f() -> vec2<f32> {
  return vec2<f32>(f32(cam.render_present_px.x), f32(cam.render_present_px.y));
}
fn present_dims_f() -> vec2<f32> {
  return vec2<f32>(f32(cam.render_present_px.z), f32(cam.render_present_px.w));
}

// Map a present pixel center (in present pixel coords) -> normalized UV over render
fn uv_render_from_present_px(px_present: vec2<f32>) -> vec2<f32> {
  let pd = present_dims_f();
  return (px_present) / pd;
}

// Map present pixel center -> render pixel center (float)
fn px_render_from_present_px(px_present: vec2<f32>) -> vec2<f32> {
  let rd = render_dims_f();
  let uv = uv_render_from_present_px(px_present);
  return uv * rd;
}

fn ip_render_from_present_px(px_present: vec2<f32>) -> vec2<i32> {
  let rd = render_dims_f();
  let pr = px_render_from_present_px(px_present);
  let ix = clamp(i32(floor(pr.x)), 0, i32(rd.x) - 1);
  let iy = clamp(i32(floor(pr.y)), 0, i32(rd.y) - 1);
  return vec2<i32>(ix, iy);
}

fn is_nan_f32(x: f32) -> bool {
  // NaN is the only float where x != x
  return x != x;
}

fn is_inf_f32(x: f32) -> bool {
  // Treat very large magnitude as inf/overflow.
  // f32 max is ~3.4e38; choose a slightly smaller guard.
  return abs(x) > 1.0e30;
}

fn is_bad_vec3(v: vec3<f32>) -> bool {
  return is_nan_f32(v.x) || is_nan_f32(v.y) || is_nan_f32(v.z) ||
         is_inf_f32(v.x) || is_inf_f32(v.y) || is_inf_f32(v.z);
}

fn ray_eps_vec(rd: vec3<f32>, eps: f32) -> vec3<f32> {
  // Push inside the next cell on each axis deterministically.
  // Using >= keeps it stable when rd component is exactly 0.
  return vec3<f32>(
    select(-eps, eps, rd.x >= 0.0),
    select(-eps, eps, rd.y >= 0.0),
    select(-eps, eps, rd.z >= 0.0)
  );
}
