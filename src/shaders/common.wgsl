const BIG_F32 : f32 = 1e30;
const EPS     : f32 = 1e-6;

const TAU : f32 = 6.28318530718;

// Sun / sky
const SUN_DIR : vec3<f32> = vec3<f32>(0.599760, 0.549780, 0.579770);
const SUN_COLOR : vec3<f32> = vec3<f32>(1.0, 0.98, 0.90);
const SUN_INTENSITY : f32 = 3.5;

// Replace acos-based disc with cosine thresholds.
// cos(0.010) ≈ 0.9999500
// cos(0.016) ≈ 0.9998720   (radius + softness)
const SUN_DISC_COS_HARD : f32 = 0.9999500;
const SUN_DISC_COS_SOFT : f32 = 0.9998720;

// Fog
const FOG_HEIGHT_FALLOFF : f32 = 0.12;
const FOG_MAX_DIST       : f32 = 240.0;
const FOG_PRIMARY_VIS    : f32 = 0.0;

// Godrays
const GODRAY_MAX_DIST    : f32 = 70.0;
const GODRAY_STEPS       : u32 = 12u;
const GODRAY_STRENGTH    : f32 = 0.0;
const GODRAY_FRAME_FPS   : f32 = 60.0;

// Post / composite
const POST_EXPOSURE : f32 = 1.15;
const COMPOSITE_GOD_SCALE : f32 = 2.2;

struct Camera {
  view_inv : mat4x4<f32>,
  proj_inv : mat4x4<f32>,
  cam_pos  : vec4<f32>,
  params   : vec4<f32>,
};

struct Stream {
  origin_x : i32,
  origin_y : i32,
  origin_z : i32,
  _pad0    : i32,

  ox : u32,
  oy : u32,
  oz : u32,
  dirty_count : u32,

  dirty_offset : u32,
  build_count  : u32,
  _pad1        : u32,
  _pad2        : u32,
};


@group(0) @binding(3) var<uniform> stream : Stream;
@group(0) @binding(0) var<uniform> cam : Camera;

fn inv_rd(rd: vec3<f32>) -> vec3<f32> {
  let s = select(vec3<f32>(-1.0), vec3<f32>(1.0), rd >= vec3<f32>(0.0));
  return s / max(abs(rd), vec3<f32>(1e-8));
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

// Cheap-ish halo: use a high power of mu instead of exp(-ang*k).
// Looks similar and avoids acos/exp for sky halo.
fn sky_color(rd: vec3<f32>) -> vec3<f32> {
  let t = clamp(0.5 * (rd.y + 1.0), 0.0, 1.0);

  var col = mix(
    vec3<f32>(0.05, 0.08, 0.12),
    vec3<f32>(0.55, 0.75, 0.95),
    t
  );

  let mu = dot(rd, SUN_DIR);

  // Disc in mu-domain (no acos). Soft edge via smoothstep.
  let disc = smoothstep(SUN_DISC_COS_SOFT, SUN_DISC_COS_HARD, mu);

  // Halo: mu^N (only when facing sun). N tuned for a tight halo.
  let m = max(mu, 0.0);
  let m2 = m * m;
  let m4 = m2 * m2;
  let m8 = m4 * m4;
  let m16 = m8 * m8;
  let m32 = m16 * m16;
  let halo = m32 * 0.12;

  col += SUN_COLOR * SUN_INTENSITY * (disc + halo);
  return col;
}

fn fog_color(rd: vec3<f32>) -> vec3<f32> {
  let up = clamp(rd.y * 0.5 + 0.5, 0.0, 1.0);
  return mix(vec3<f32>(0.62, 0.64, 0.66), sky_color(rd), 0.22 * up);
}

fn fog_density() -> f32 {
  return max(cam.params.y, 0.0);
}

fn fog_optical_depth(ro: vec3<f32>, rd: vec3<f32>, t: f32) -> f32 {
  let base = fog_density();
  if (base <= 0.0 || t <= 0.0) { return 0.0; }

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

// -----------------------------------------------------------------------------
// Fast RNG: PCG-ish hash => float in [0,1)
// -----------------------------------------------------------------------------

fn pcg_hash(v: u32) -> u32 {
  var s = v * 747796405u + 2891336453u;
  let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (word >> 22u) ^ word;
}

fn hash_u01(seed: u32) -> f32 {
  // 1/2^32
  return f32(pcg_hash(seed)) * 2.3283064365386963e-10;
}
