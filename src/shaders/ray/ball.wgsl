// src/shaders/ray/ball.wgsl
// Blocky voxel-sphere (Minecraft-like): sharp voxel faces, voxel silhouette.

struct Ball {
  center_radius: vec4<f32>, // xyz center (meters), w radius (meters)
  material: u32,

  // q8 fixed-point scale relative to world voxel size:
  // 256 = 1.0x, 512 = 2.0x, 1024 = 4.0x, etc.
  voxel_scale_q8: u32,

  // padding to keep struct 16-byte aligned
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(11)
var<storage, read> balls: array<Ball>;

struct BallHit {
  hit : bool,
  t   : f32,
  n   : vec3<f32>,
  mat : u32,

  // for AO + stable shading
  v_vox : vec3<i32>,
  c_m   : vec3<f32>, // sphere center in meters
  r_m   : f32,       // sphere radius in meters
  vs    : f32,       // voxel size used for this ball (meters)
};

fn miss_ballhit() -> BallHit {
  return BallHit(
    false,
    BIG_F32,
    vec3<f32>(0.0),
    MAT_AIR,
    vec3<i32>(0),
    vec3<f32>(0.0),
    0.0,
    0.0
  );
}

fn ray_sphere_interval(ro: vec3<f32>, rd: vec3<f32>, c: vec3<f32>, r: f32) -> vec2<f32> {
  let oc = ro - c;
  let b  = dot(oc, rd);
  let c0 = dot(oc, oc) - r * r;
  let h  = b * b - c0;
  if (h < 0.0) { return vec2<f32>(1.0, 0.0); }
  let s = sqrt(h);
  return vec2<f32>(-b - s, -b + s);
}

// sphere vs voxel AABB occupancy (sphere intersects the voxel AABB)
fn ball_voxel_occupied(v_vox: vec3<i32>, c_m: vec3<f32>, r_m: f32, vs: f32) -> bool {
  let bmin = vec3<f32>(f32(v_vox.x), f32(v_vox.y), f32(v_vox.z)) * vs;
  let bmax = bmin + vec3<f32>(vs);

  // distance from sphere center to AABB (0 if inside)
  let d0 = max(bmin - c_m, vec3<f32>(0.0));
  let d1 = max(c_m - bmax, vec3<f32>(0.0));
  let d  = max(d0, d1);

  return dot(d, d) <= (r_m * r_m);
}

fn ball_voxel_ao_6(v_vox: vec3<i32>, c_m: vec3<f32>, r_m: f32, vs: f32) -> f32 {
  var occ: f32 = 0.0;
  occ += select(0.0, 1.0, ball_voxel_occupied(v_vox + vec3<i32>( 1, 0, 0), c_m, r_m, vs));
  occ += select(0.0, 1.0, ball_voxel_occupied(v_vox + vec3<i32>(-1, 0, 0), c_m, r_m, vs));
  occ += select(0.0, 1.0, ball_voxel_occupied(v_vox + vec3<i32>( 0, 1, 0), c_m, r_m, vs));
  occ += select(0.0, 1.0, ball_voxel_occupied(v_vox + vec3<i32>( 0,-1, 0), c_m, r_m, vs));
  occ += select(0.0, 1.0, ball_voxel_occupied(v_vox + vec3<i32>( 0, 0, 1), c_m, r_m, vs));
  occ += select(0.0, 1.0, ball_voxel_occupied(v_vox + vec3<i32>( 0, 0,-1), c_m, r_m, vs));

  // tuned for blocky look: modest darkening with a floor
  return clamp(1.0 - 0.40 * (occ * (1.0 / 6.0)), 0.60, 1.0);
}

fn trace_ball_voxels_blocky(
  ro: vec3<f32>,
  rd: vec3<f32>,
  t_min: f32,
  t_max: f32,
  c_m: vec3<f32>,
  r_m: f32,
  mat: u32,
  voxel_scale_q8: u32
) -> BallHit {
  // Option B: per-ball voxel size
  let vs_world = cam.voxel_params.x;
  let vs = vs_world * (f32(voxel_scale_q8) / 256.0);

  // Early reject with analytic sphere interval (does NOT make it look smooth)
  let itv = ray_sphere_interval(ro, rd, c_m, r_m);
  var t0 = max(itv.x, t_min);
  let t1 = min(itv.y, t_max);
  if (t1 <= t0) { return miss_ballhit(); }

  // Start point
  let eps = 1e-4 * vs;
  t0 = max(t0, 0.0) + eps;

  var p = ro + rd * t0;
  var v = vec3<i32>(floor(p / vs));

  // DDA setup
  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));
  let stepX: i32 = select(-1, 1, rd.x > 0.0);
  let stepY: i32 = select(-1, 1, rd.y > 0.0);
  let stepZ: i32 = select(-1, 1, rd.z > 0.0);

  let nextBx = f32(select(v.x, v.x + 1, stepX > 0)) * vs;
  let nextBy = f32(select(v.y, v.y + 1, stepY > 0)) * vs;
  let nextBz = f32(select(v.z, v.z + 1, stepZ > 0)) * vs;

  var tMaxX: f32 = select(BIG_F32, t0 + (nextBx - p.x) * inv.x, abs(rd.x) >= EPS_INV);
  var tMaxY: f32 = select(BIG_F32, t0 + (nextBy - p.y) * inv.y, abs(rd.y) >= EPS_INV);
  var tMaxZ: f32 = select(BIG_F32, t0 + (nextBz - p.z) * inv.z, abs(rd.z) >= EPS_INV);

  let tDeltaX: f32 = select(BIG_F32, vs * abs(inv.x), abs(rd.x) >= EPS_INV);
  let tDeltaY: f32 = select(BIG_F32, vs * abs(inv.y), abs(rd.y) >= EPS_INV);
  let tDeltaZ: f32 = select(BIG_F32, vs * abs(inv.z), abs(rd.z) >= EPS_INV);

  // If we start inside an occupied voxel: return that voxel's AABB entry at ~t0
  if (ball_voxel_occupied(v, c_m, r_m, vs)) {
    let bmin = vec3<f32>(f32(v.x), f32(v.y), f32(v.z)) * vs;
    let rt   = intersect_aabb(ro, rd, bmin, bmin + vec3<f32>(vs));
    let n0   = normalize(-rd); // ambiguous; good enough for inside-start
    return BallHit(true, max(rt.x, t0), n0, mat, v, c_m, r_m, vs);
  }

  let MAX_STEPS: u32 = 512u;
  var tcur: f32 = t0;
  var nstep: vec3<f32> = vec3<f32>(0.0);

  for (var i: u32 = 0u; i < MAX_STEPS; i = i + 1u) {
    if (tcur > t1) { break; }

    // Step to next boundary and remember which face we crossed (normal)
    if (tMaxX < tMaxY) {
      if (tMaxX < tMaxZ) {
        tcur = tMaxX; tMaxX += tDeltaX;
        v.x += stepX;
        nstep = vec3<f32>(-f32(stepX), 0.0, 0.0);
      } else {
        tcur = tMaxZ; tMaxZ += tDeltaZ;
        v.z += stepZ;
        nstep = vec3<f32>(0.0, 0.0, -f32(stepZ));
      }
    } else {
      if (tMaxY < tMaxZ) {
        tcur = tMaxY; tMaxY += tDeltaY;
        v.y += stepY;
        nstep = vec3<f32>(0.0, -f32(stepY), 0.0);
      } else {
        tcur = tMaxZ; tMaxZ += tDeltaZ;
        v.z += stepZ;
        nstep = vec3<f32>(0.0, 0.0, -f32(stepZ));
      }
    }

    if (tcur > t1) { break; }

    if (ball_voxel_occupied(v, c_m, r_m, vs)) {
      // Exact hit with the voxel AABB (cube faces)
      let bmin = vec3<f32>(f32(v.x), f32(v.y), f32(v.z)) * vs;
      let rt   = intersect_aabb(ro, rd, bmin, bmin + vec3<f32>(vs));
      let thit = max(rt.x, t_min);
      return BallHit(true, thit, nstep, mat, v, c_m, r_m, vs);
    }
  }

  return miss_ballhit();
}

fn trace_balls(ro: vec3<f32>, rd: vec3<f32>, t_min: f32, t_max: f32) -> BallHit {
  let nballs: u32 = cam.dyn_counts.x;
  var best = miss_ballhit();

  for (var i: u32 = 0u; i < nballs; i = i + 1u) {
    let c  = balls[i].center_radius.xyz;
    let r  = balls[i].center_radius.w;
    let m  = balls[i].material;
    let vsq = balls[i].voxel_scale_q8;

    let h = trace_ball_voxels_blocky(ro, rd, t_min, t_max, c, r, m, vsq);
    if (h.hit && h.t < best.t) { best = h; }
  }
  return best;
}

fn shade_ball_hit(ro: vec3<f32>, rd: vec3<f32>, bh: BallHit, sky_up: vec3<f32>) -> vec3<f32> {
  let hp = ro + bh.t * rd;
  let n  = normalize(bh.n);

  var base = color_for_material(bh.mat);

  // Per-voxel variation (stable, blocky)
  let h = hash3i(bh.v_vox);
  let v = 0.90 + 0.20 * h; // 0.90..1.10
  base *= v;

  // Face shading
  base *= face_shade(n);

  // Voxel AO (must use the same voxel size that produced the hit)
  let ao = ball_voxel_ao_6(bh.v_vox, bh.c_m, bh.r_m, bh.vs);

  // Shadow ray offset scale: use bh.vs so large voxels don't self-shadow weirdly
  let hp_shadow = hp + n * (0.75 * bh.vs);
  let vis  = sun_transmittance(hp_shadow, SUN_DIR);
  let diff = max(dot(n, SUN_DIR), 0.0);

  let amb    = hemi_ambient(n, sky_up) * 0.12 * ao;
  let direct = SUN_COLOR * SUN_INTENSITY * (diff * diff) * vis;

  return base * (amb + direct);
}

fn hash3i(v: vec3<i32>) -> f32 {
  let x = u32(v.x) * 1664525u + 1013904223u;
  let y = u32(v.y) * 22695477u + 1u;
  let z = u32(v.z) * 1103515245u + 12345u;
  let h = x ^ y ^ z;
  return f32(h & 0x00FFFFFFu) / f32(0x01000000u);
}

fn face_shade(n: vec3<f32>) -> f32 {
  if (n.y >  0.5) { return 1.00; } // top
  if (n.y < -0.5) { return 0.55; } // bottom
  return 0.80;                     // sides
}
