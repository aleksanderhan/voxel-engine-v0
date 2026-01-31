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
//// Materials
//// --------------------------------------------------------------------------

const MAT_AIR   : u32 = 0u;
const MAT_GRASS : u32 = 1u;
const MAT_DIRT  : u32 = 2u;
const MAT_STONE : u32 = 3u;
const MAT_WOOD  : u32 = 4u;
const MAT_LEAF  : u32 = 5u;

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
const SHADOW_STEPS : u32 = 32u;

const SHADOW_DISPLACED_LEAVES : bool = false;
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

const GODRAY_FADE_NEAR : f32 = 60.0;
const GODRAY_FADE_FAR  : f32 = 160.0;

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

const GODRAY_MAX_DIST : f32 = 50.0;

const GODRAY_SCATTER_HEIGHT_FALLOFF : f32 = 0.04;
const GODRAY_SCATTER_MIN_FRAC       : f32 = 0.35;

const GODRAY_BLACK_LEVEL : f32 = 0.004;

const GODRAY_TS_LP_ALPHA : f32 = 0.40;
const GODRAY_EDGE0       : f32 = 0.004;
const GODRAY_EDGE1       : f32 = 0.035;

const GODRAY_BASE_HAZE       : f32 = 0.02;
const GODRAY_HAZE_NEAR_FADE  : f32 = 18.0;
const CLOUD_GODRAY_W         : f32 = 0.50;

const GODRAY_TV_CUTOFF   : f32 = 0.02;
const GODRAY_STEPS_FAST  : u32 = 16u;

const GODRAY_SHAFT_GAIN: f32 = 3.0;

const GODRAY_EDGE_ENERGY_BOOST: f32 = 2.5; // try 1.0 .. 4.0


//// --------------------------------------------------------------------------
//// Phase
//// --------------------------------------------------------------------------

const INV_4PI     : f32 = 0.0795774715;
const PHASE_G     : f32 = 0.15;
const PHASE_MIE_W : f32 = 0.20;

//// --------------------------------------------------------------------------
//// Fractal clouds
//// --------------------------------------------------------------------------

const CLOUD_H        : f32      = 200.0;
const CLOUD_UV_SCALE : f32      = 0.002;
const CLOUD_WIND     : vec2<f32> = vec2<f32>(0.020, 0.012);

const CLOUD_COVERAGE : f32 = 0.45;
const CLOUD_SOFTNESS : f32 = 0.10;

const CLOUD_HORIZON_Y0 : f32 = 0.02;
const CLOUD_HORIZON_Y1 : f32 = 0.25;

// How clouds darken the SKY appearance (keep low)
const CLOUD_SKY_DARKEN : f32 = 0.2;

// How much clouds attenuate SUNLIGHT hitting the world (can be much higher)
const CLOUD_SHADOW_ABSORB   : f32 = 6.0;   // try 4..12
const CLOUD_SHADOW_STRENGTH : f32 = 0.8;   // 0..1 (mix control)


const CLOUD_BASE_COL   : vec3<f32> = vec3<f32>(0.72, 0.74, 0.76);
const CLOUD_SILVER_POW : f32       = 8.0;
const CLOUD_SILVER_STR : f32       = 0.6;
const CLOUD_BLEND      : f32       = 0.85;

const CLOUD_DIM_SUN_DISC            : bool = true;
const CLOUD_SUN_DISC_ABSORB_SCALE   : f32  = 0.8;

const SKY_EXPOSURE : f32 = 0.40;

//// --------------------------------------------------------------------------
//// Leaf wind (displaced cubes)
//// --------------------------------------------------------------------------

const WIND_CELL_FREQ : f32      = 2.5;
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

const GODRAY_BLOCK_SIZE : i32 = 2;

const J0_SCALE : f32 = 1.31;
const J1_SCALE : f32 = 2.11;
const J2_SCALE : f32 = 3.01;
const J3_SCALE : f32 = 4.19;

const COMPOSITE_SHARPEN : f32 = 0.15;

const POST_EXPOSURE : f32 = 0.15;

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

// Grass level-of-detail (LOD) distances in meters-ish (assuming rd normalized).
const GRASS_LOD_MID_START : f32 = 15.0;
const GRASS_LOD_FAR_START : f32 = 40.0;

// Mid/far quality knobs (tune freely)
const GRASS_BLADE_COUNT_MID : u32 = 2u;
const GRASS_BLADE_COUNT_FAR : u32 = 1u;

const GRASS_SEGS_MID : u32 = 2u;
const GRASS_SEGS_FAR : u32 = 1u;

const GRASS_TRACE_STEPS_MID : u32 = 6u;
const GRASS_TRACE_STEPS_FAR : u32 = 4u;

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
};

//// --------------------------------------------------------------------------
//// Scene bindings (group 0) - shared across passes
//// --------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> cam : Camera;
@group(0) @binding(1) var<storage, read> chunks     : array<ChunkMeta>;
@group(0) @binding(2) var<storage, read> nodes      : array<Node>;
@group(0) @binding(3) var<storage, read> chunk_grid : array<u32>;
@group(0) @binding(8) var<storage, read> macro_occ : array<u32>;
@group(0) @binding(9) var<storage, read> node_ropes: array<NodeRopes>;
@group(0) @binding(10) var<storage, read> chunk_colinfo : array<u32>;


//// --------------------------------------------------------------------------
//// Shared helpers
//// --------------------------------------------------------------------------

const MACRO_DIM : u32 = 8u;          // 8x8x8 macro cells per chunk
const MACRO_WORDS_PER_CHUNK : u32 = 16u; // 512 bits / 32

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
  return vec3<i32>(
    i32(floor(p.x / chunk_size_m)),
    i32(floor(p.y / chunk_size_m)),
    i32(floor(p.z / chunk_size_m))
  );
}

fn chunk_max_depth() -> u32 {
  // chunk_size is power-of-two; log2 = 31 - clz
  return 31u - countLeadingZeros(cam.chunk_size);
}

// ---- Column info (64x64) ----
// 4096 columns, packed 2x u16 per u32 => 2048 u32 words per chunk.
const CHUNK_COL_WORDS_PER_CHUNK : u32 = 2048u;

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
  // Most of your callsites already pass integer-ish coordinates (floor() outputs),
  // but this also works fine for pixel coords by flooring.
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
