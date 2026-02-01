// src/shaders/ray/voxel.wgsl
// Blocky voxel-sphere (Minecraft-like): sharp voxel faces, voxel silhouette.

struct Voxel {
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
var<storage, read> voxels: array<Voxel>;

struct VoxelHit {
  hit : bool,
  t   : f32,
  n   : vec3<f32>,
  mat : u32,

  // stable id for variation
  v_vox : vec3<i32>,

  c_m   : vec3<f32>, // center
  r_m   : f32,       // collider radius (still useful as metadata)

  half  : f32,       // <-- NEW: cube half-extent in meters
};


fn miss_voxelhit() -> VoxelHit {
  return VoxelHit(
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

// collider vs voxel AABB occupancy (collider intersects the voxel AABB)
fn voxel_cell_occupied(v_vox: vec3<i32>, c_m: vec3<f32>, r_m: f32, vs: f32) -> bool {
  // voxel center in world meters
  let vc = (vec3<f32>(f32(v_vox.x), f32(v_vox.y), f32(v_vox.z)) + vec3<f32>(0.5)) * vs;
  let d  = vc - c_m;
  return dot(d, d) <= (r_m * r_m);
}

fn voxel_cell_ao_6(v_vox: vec3<i32>, c_m: vec3<f32>, r_m: f32, vs: f32) -> f32 {
  var occ: f32 = 0.0;
  occ += select(0.0, 1.0, voxel_cell_occupied(v_vox + vec3<i32>( 1, 0, 0), c_m, r_m, vs));
  occ += select(0.0, 1.0, voxel_cell_occupied(v_vox + vec3<i32>(-1, 0, 0), c_m, r_m, vs));
  occ += select(0.0, 1.0, voxel_cell_occupied(v_vox + vec3<i32>( 0, 1, 0), c_m, r_m, vs));
  occ += select(0.0, 1.0, voxel_cell_occupied(v_vox + vec3<i32>( 0,-1, 0), c_m, r_m, vs));
  occ += select(0.0, 1.0, voxel_cell_occupied(v_vox + vec3<i32>( 0, 0, 1), c_m, r_m, vs));
  occ += select(0.0, 1.0, voxel_cell_occupied(v_vox + vec3<i32>( 0, 0,-1), c_m, r_m, vs));

  // tuned for blocky look: modest darkening with a floor
  return clamp(1.0 - 0.40 * (occ * (1.0 / 6.0)), 0.60, 1.0);
}

fn trace_voxels(ro: vec3<f32>, rd: vec3<f32>, t_min: f32, t_max: f32) -> VoxelHit {
  let nvoxels: u32 = cam.dyn_counts.x;
  var best = miss_voxelhit();

  for (var i: u32 = 0u; i < nvoxels; i = i + 1u) {
    let c   = voxels[i].center_radius.xyz;
    let r   = voxels[i].center_radius.w;
    let m   = voxels[i].material;
    let vsq = voxels[i].voxel_scale_q8;

    let h = trace_voxel_cube(ro, rd, t_min, t_max, c, r, m, vsq);
    if (h.hit && h.t < best.t) { best = h; }
  }
  return best;
}

fn shade_voxel_hit(ro: vec3<f32>, rd: vec3<f32>, vh: VoxelHit, sky_up: vec3<f32>) -> vec3<f32> {
  let hp = ro + vh.t * rd;

  // Face normal is already axis-aligned from ray_aabb_hit
  let n = normalize(vh.n);

  var base = color_for_material(vh.mat);

  // Stable per-object variation (optional)
  let h = hash3i(vh.v_vox);
  base *= (0.90 + 0.20 * h);

  base *= face_shade(n);

  // No voxel-cell AO anymore (it implied a sphere made of many cubes)
  let ao = 1.0;

  // Shadow offset based on cube size
  let hp_shadow = hp + n * (0.10 * max(vh.half, 1e-4));
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

fn aabb_entry_t_and_n(ro: vec3<f32>, rd: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec4<f32> {
  // returns (nx, ny, nz, t_enter)
  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  let t0 = (bmin - ro) * inv;
  let t1 = (bmax - ro) * inv;

  let tmin3 = min(t0, t1);
  let tmax3 = max(t0, t1);

  let t_enter = max(max(tmin3.x, tmin3.y), tmin3.z);
  // let t_exit  = min(min(tmax3.x, tmax3.y), tmax3.z); // not needed here

  let eps = 1e-5;

  var n = vec3<f32>(0.0);

  if (abs(t_enter - tmin3.x) < eps) { n = vec3<f32>(-sign(rd.x), 0.0, 0.0); }
  else if (abs(t_enter - tmin3.y) < eps) { n = vec3<f32>(0.0, -sign(rd.y), 0.0); }
  else { n = vec3<f32>(0.0, 0.0, -sign(rd.z)); }

  return vec4<f32>(n, t_enter);
}


fn ray_aabb_hit(ro: vec3<f32>, rd: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>, t_min: f32, t_max: f32) -> vec4<f32> {
  // returns (nx, ny, nz, t_enter). If miss => w = BIG_F32
  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  let t0 = (bmin - ro) * inv;
  let t1 = (bmax - ro) * inv;

  let tmin3 = min(t0, t1);
  let tmax3 = max(t0, t1);

  let t_enter = max(max(tmin3.x, tmin3.y), tmin3.z);
  let t_exit  = min(min(tmax3.x, tmax3.y), tmax3.z);

  let t_hit = max(t_enter, t_min);
  if (t_exit < t_hit || t_hit > t_max) {
    return vec4<f32>(0.0, 0.0, 0.0, BIG_F32);
  }

  // axis normal from which slab produced t_enter
  let eps = 1e-5;
  var n = vec3<f32>(0.0);

  if (abs(t_enter - tmin3.x) < eps) { n = vec3<f32>(-sign(rd.x), 0.0, 0.0); }
  else if (abs(t_enter - tmin3.y) < eps) { n = vec3<f32>(0.0, -sign(rd.y), 0.0); }
  else { n = vec3<f32>(0.0, 0.0, -sign(rd.z)); }

  return vec4<f32>(n, t_hit);
}

fn trace_voxel_cube(
  ro: vec3<f32>,
  rd: vec3<f32>,
  t_min: f32,
  t_max: f32,
  c_m: vec3<f32>,
  r_m: f32,
  mat: u32,
  voxel_scale_q8: u32
) -> VoxelHit {
  // Visual size: cube half-extent.
  // Keep collision spherical in CPU; this is just the render proxy.
  let s = f32(voxel_scale_q8) / 256.0;
  let half = r_m * s;

  let bmin = c_m - vec3<f32>(half);
  let bmax = c_m + vec3<f32>(half);

  let tn = ray_aabb_hit(ro, rd, bmin, bmax, t_min, t_max);
  if (tn.w >= 0.5 * BIG_F32) { return miss_voxelhit(); }

  // stable-ish integer id for variation (optional)
  let vsw = cam.voxel_params.x;
  let vid = vec3<i32>(floor(c_m / max(vsw, 1e-6)));

  return VoxelHit(true, tn.w, tn.xyz, mat, vid, c_m, r_m, half);
}
