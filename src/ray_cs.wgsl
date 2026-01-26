// ray_cs.wgsl
//
// Multi-chunk hybrid Sparse Voxel Octree (SVO) ray traversal compute shader.
//
// What this shader does
// - For each pixel, reconstruct a world-space ray from the camera matrices.
// - Test that ray against every streamed chunk (each chunk is a root SVO).
// - Traverse each chunk with a hybrid “DDA-like” loop:
//     * sample the SVO at the current ray position to get a leaf region (cube AABB + material)
//     * if material is empty -> jump to the exit of that cube (big skips through air)
//     * if material is solid -> report a hit (no shading yet)
//
// Leaf wind (voxel cubes that move)
// - Leaves remain cubes, but their cube AABB is displaced by a stable wind field.
// - When we hit a LEAF voxel, we intersect against the displaced cube.
//   If the displaced cube is missed, we treat that voxel as “empty for this ray” and keep marching.
//
// Sun shadows (hard shadows)
// - After we find the nearest surface hit (across ALL chunks), we cast ONE shadow ray toward the sun.
// - If any voxel blocks that ray (within loaded chunks), the point is in shadow.
//
// Performance choices
// - Nearest-hit selection is done before any shading or shadowing.
// - Chunk AABB is tested in main() to skip chunks that can't beat the current best hit.
// - Shadow rays are capped to fewer steps.
// - Optional: let leaves cast shadows using their undisplaced voxel cube (much faster).
//
// Notes on abbreviations (first use)
// - SVO  = Sparse Voxel Octree
// - AABB = Axis-Aligned Bounding Box
// - NDC  = Normalized Device Coordinates

// ------------------------------------------------------------
// GPU structs (must match Rust side layouts)
// ------------------------------------------------------------

struct Node {
  // If internal: index of first child in a compact child list.
  // If leaf: 0xFFFF_FFFF.
  child_base : u32,

  // Bitmask of existing children (bits 0..7 = octants).
  // Children are stored compactly in increasing octant index order.
  child_mask : u32,

  // For leaf nodes: material id (0 = empty / air).
  // For internal nodes: unused (0).
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

  // voxel_params:
  // x = voxel_size_m      (meters per voxel)
  // y = time_seconds      (animation time)
  // z = wind_strength     (artist knob)
  // w = unused
  voxel_params : vec4<f32>,
};

struct ChunkMeta {
  // Chunk origin in world voxel coordinates (integer grid).
  origin     : vec4<i32>,

  // Base index of this chunk's root node within the global packed node buffer.
  node_base  : u32,

  // Optional: number of nodes for debug/validation (unused here).
  node_count : u32,

  _pad0      : u32,
  _pad1      : u32,
};

@group(0) @binding(0) var<uniform> cam : Camera;
@group(0) @binding(1) var<storage, read> chunks : array<ChunkMeta>;
@group(0) @binding(2) var<storage, read> nodes  : array<Node>;
@group(0) @binding(3) var out_img : texture_storage_2d<rgba16float, write>;

// ------------------------------------------------------------
// Constants
// ------------------------------------------------------------

const LEAF_U32 : u32 = 0xFFFFFFFFu;
const BIG_F32  : f32 = 1e30;
const EPS_INV  : f32 = 1e-8;

// Directional sun at 45° elevation, pointing roughly toward +X,+Z.
// (1, sqrt(2), 1) normalized => length 2 => (0.5, 0.70710678, 0.5)
const SUN_DIR : vec3<f32> = vec3<f32>(0.5, 0.70710678, 0.5);

// Shadow tuning
const SHADOW_BIAS  : f32 = 2e-4; // meters; reduces self-shadow acne
const SHADOW_STEPS : u32 = 32u;  // cheaper than primary ray

// If false: leaves cast shadows using their undisplaced voxel cube (faster).
// If true: shadows match displaced leaf cubes (slower, more consistent).
const SHADOW_DISPLACED_LEAVES : bool = true;

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
// Returns (t_enter, t_exit). If t_exit < t_enter => no hit.
//
fn intersect_aabb(ro: vec3<f32>, rd: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
  let eps = 1e-8;

  var t_enter = -1e30;
  var t_exit  =  1e30;

  // X slab
  if (abs(rd.x) < eps) {
    if (ro.x < bmin.x || ro.x > bmax.x) { return vec2<f32>(1.0, 0.0); }
  } else {
    let inv = 1.0 / rd.x;
    let t0 = (bmin.x - ro.x) * inv;
    let t1 = (bmax.x - ro.x) * inv;
    t_enter = max(t_enter, min(t0, t1));
    t_exit  = min(t_exit,  max(t0, t1));
  }

  // Y slab
  if (abs(rd.y) < eps) {
    if (ro.y < bmin.y || ro.y > bmax.y) { return vec2<f32>(1.0, 0.0); }
  } else {
    let inv = 1.0 / rd.y;
    let t0 = (bmin.y - ro.y) * inv;
    let t1 = (bmax.y - ro.y) * inv;
    t_enter = max(t_enter, min(t0, t1));
    t_exit  = min(t_exit,  max(t0, t1));
  }

  // Z slab
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
//
// Children are stored compactly. To index child ci:
// - rank = number of set bits in mask below ci
// - child_index = child_base + rank
//
fn child_rank(mask: u32, ci: u32) -> u32 {
  let bit = 1u << ci;
  let lower = mask & (bit - 1u);
  return countOneBits(lower);
}

// ------------------------------------------------------------
// Materials + shading helpers
// ------------------------------------------------------------

fn color_for_material(m: u32) -> vec3<f32> {
  if (m == 0u) { return vec3<f32>(0.0); } // AIR

  // Terrain
  if (m == 1u) { return vec3<f32>(0.18, 0.75, 0.18); } // GRASS
  if (m == 2u) { return vec3<f32>(0.45, 0.30, 0.15); } // DIRT
  if (m == 3u) { return vec3<f32>(0.50, 0.50, 0.55); } // STONE

  // Trees
  if (m == 4u) { return vec3<f32>(0.38, 0.26, 0.14); } // WOOD
  if (m == 5u) { return vec3<f32>(0.10, 0.55, 0.12); } // LEAF

  return vec3<f32>(1.0, 0.0, 1.0); // unknown => magenta
}

// Normal of the closest cube face at hit point hp.
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
// Leaf wind field (stable grouping to avoid shimmer)
// ------------------------------------------------------------

fn hash1(p: vec3<f32>) -> f32 {
  let h = dot(p, vec3<f32>(127.1, 311.7, 74.7));
  return fract(sin(h) * 43758.5453);
}

fn wind_field(pos_m: vec3<f32>, t: f32) -> vec3<f32> {
  // Group nearby leaves into shared phase cells (~0.4m).
  let cell = floor(pos_m * 2.5);
  let ph0 = hash1(cell);
  let ph1 = hash1(cell + vec3<f32>(19.0, 7.0, 11.0));

  // Predominantly horizontal wind direction in XZ.
  let dir = normalize(vec2<f32>(0.9, 0.4));

  // More motion higher up.
  let h = clamp((pos_m.y - 2.0) / 12.0, 0.0, 1.0);

  // Two layered waves: gust (slow) + flutter (fast).
  let gust    = sin(t * 0.9 + dot(pos_m.xz, vec2<f32>(0.35, 0.22)) + ph0 * 6.28318);
  let flutter = sin(t * 4.2 + dot(pos_m.xz, vec2<f32>(1.7,  1.1 )) + ph1 * 6.28318);

  // Horizontal sway plus a small vertical component.
  let xz = dir * (0.75 * gust + 0.25 * flutter) * h;
  let y  = 0.25 * flutter * h;

  return vec3<f32>(xz.x, y, xz.y);
}

// ------------------------------------------------------------
// Leaf cubes that move: displaced AABB hit test
// ------------------------------------------------------------

fn clamp_len(v: vec3<f32>, max_len: f32) -> vec3<f32> {
  let l2 = dot(v, v);
  if (l2 <= max_len * max_len) { return v; }
  return v * (max_len / sqrt(l2));
}

// Compute a per-cube wind offset (meters), clamped so cubes still read as voxels.
fn leaf_cube_offset(bmin: vec3<f32>, size: f32, time_s: f32, strength: f32) -> vec3<f32> {
  let center = bmin + vec3<f32>(0.5 * size);

  // Wind is already grouped in space; keep it mostly horizontal here.
  var w = wind_field(center, time_s) * strength;
  w = vec3<f32>(w.x, 0.15 * w.y, w.z);

  // Amplitude relative to cube size.
  let amp = 0.35 * size;

  // Clamp prevents cubes drifting too far and breaking the voxel look.
  return clamp_len(w * amp, 0.45 * size);
}

struct LeafCubeHit {
  hit  : bool,
  t    : f32,
  n    : vec3<f32>,
};

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
  let off   = leaf_cube_offset(bmin, size, time_s, strength);
  let bmin2 = bmin + off;
  let bmax2 = bmin2 + vec3<f32>(size);

  let rt = intersect_aabb(ro, rd, bmin2, bmax2);
  var t0 = rt.x;
  let t1 = rt.y;

  if (t1 < t0) { return LeafCubeHit(false, BIG_F32, vec3<f32>(0.0)); }

  t0 = max(t0, t_min);
  if (t0 > min(t1, t_max)) { return LeafCubeHit(false, BIG_F32, vec3<f32>(0.0)); }

  let hp = ro + t0 * rd;
  let nn = cube_normal(hp, bmin2, size);

  return LeafCubeHit(true, t0, nn);
}

// ------------------------------------------------------------
// Hybrid traversal utilities
// ------------------------------------------------------------

fn safe_inv(x: f32) -> f32 {
  return select(1.0 / x, BIG_F32, abs(x) < EPS_INV);
}

struct LeafQuery {
  bmin : vec3<f32>, // cube min (world meters)
  size : f32,       // cube size (world meters)
  mat  : u32,       // 0 = empty / air
};

// Query the SVO to find the leaf region containing point p.
// If a child octant is missing, that sub-cube is empty.
fn query_leaf_at(
  p: vec3<f32>,
  root_bmin: vec3<f32>,
  root_size: f32,
  node_base: u32
) -> LeafQuery {
  var idx: u32 = node_base;
  var bmin: vec3<f32> = root_bmin;
  var size: f32 = root_size;

  // Stop descending once we're at voxel resolution (in meters).
  let min_leaf: f32 = cam.voxel_params.x;
  let eps: f32 = 1e-7;

  for (var d: u32 = 0u; d < 32u; d = d + 1u) {
    let n = nodes[idx];

    // Leaf node: material is authoritative for this region.
    if (n.child_base == LEAF_U32) {
      return LeafQuery(bmin, size, n.material);
    }

    // Safety: if the tree is deeper than expected, treat as empty.
    if (size <= min_leaf + eps) {
      return LeafQuery(bmin, size, 0u);
    }

    let half = size * 0.5;
    let mid  = bmin + vec3<f32>(half);

    // Choose octant based on p relative to mid.
    let hx = select(0u, 1u, p.x >= mid.x);
    let hy = select(0u, 1u, p.y >= mid.y);
    let hz = select(0u, 1u, p.z >= mid.z);
    let ci = hx | (hy << 1u) | (hz << 2u);

    let child_bmin = bmin + vec3<f32>(
      select(0.0, half, hx != 0u),
      select(0.0, half, hy != 0u),
      select(0.0, half, hz != 0u)
    );

    let bit = 1u << ci;

    // Missing child => empty leaf region at this scale.
    if ((n.child_mask & bit) == 0u) {
      return LeafQuery(child_bmin, half, 0u);
    }

    // Present child => find its compacted index.
    let rank = child_rank(n.child_mask, ci);
    idx = node_base + (n.child_base + rank);

    bmin = child_bmin;
    size = half;
  }

  // If we exceed max depth, treat as empty.
  return LeafQuery(bmin, size, 0u);
}

// Return the parametric t where the ray exits this axis-aligned cube.
// Used to jump across empty space quickly.
fn exit_time_from_cube(ro: vec3<f32>, rd: vec3<f32>, bmin: vec3<f32>, size: f32) -> f32 {
  let bmax = bmin + vec3<f32>(size);
  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  let tx = (select(bmin.x, bmax.x, rd.x > 0.0) - ro.x) * inv.x;
  let ty = (select(bmin.y, bmax.y, rd.y > 0.0) - ro.y) * inv.y;
  let tz = (select(bmin.z, bmax.z, rd.z > 0.0) - ro.z) * inv.z;

  return min(tx, min(ty, tz));
}

// ------------------------------------------------------------
// Shadow ray traversal (cheap occlusion query)
// ------------------------------------------------------------

// Per-chunk occlusion test: returns true if anything blocks the ray inside this chunk.
fn trace_chunk_shadow(ro: vec3<f32>, rd: vec3<f32>, ch: ChunkMeta, t_min: f32) -> bool {
  let voxel_size = cam.voxel_params.x;

  let root_bmin_vox = vec3<f32>(f32(ch.origin.x), f32(ch.origin.y), f32(ch.origin.z));
  let root_bmin = root_bmin_vox * voxel_size;
  let root_size = f32(cam.chunk_size) * voxel_size;
  let root_bmax = root_bmin + vec3<f32>(root_size);

  let rt = intersect_aabb(ro, rd, root_bmin, root_bmax);
  var t_enter = max(rt.x, t_min);
  let t_exit  = rt.y;

  if (t_exit < t_enter) { return false; }

  var tcur = t_enter;

  for (var step_i: u32 = 0u; step_i < SHADOW_STEPS; step_i = step_i + 1u) {
    if (tcur > t_exit) { break; }

    let p = ro + tcur * rd;
    let q = query_leaf_at(p, root_bmin, root_size, ch.node_base);

    if (q.mat != 0u) {
      // Leaves: optional fast shadows (undisplaced).
      if (q.mat == 5u) {
        if (!SHADOW_DISPLACED_LEAVES) {
          return true;
        }

        // Accurate (slower) displaced-leaf shadowing.
        let time_s   = cam.voxel_params.y;
        let strength = cam.voxel_params.z;

        let h2 = leaf_displaced_cube_hit(
          ro, rd,
          q.bmin, q.size,
          time_s, strength,
          tcur - 1e-5,
          t_exit
        );

        if (h2.hit) { return true; }

        // Displaced leaf missed: skip using the undisplaced leaf region.
        let t_leave = exit_time_from_cube(ro, rd, q.bmin, q.size);
        tcur = max(t_leave, tcur) + 1e-4;
        continue;
      }

      // Any other solid voxel blocks the sun.
      return true;
    }

    // Skip empty region.
    let t_leave = exit_time_from_cube(ro, rd, q.bmin, q.size);
    tcur = max(t_leave, tcur) + 1e-4;
  }

  return false;
}

// Scene-level hard shadow: if any chunk blocks toward the sun, the point is shadowed.
fn in_shadow(p: vec3<f32>, sun_dir: vec3<f32>) -> bool {
  let ro = p + sun_dir * SHADOW_BIAS;
  let rd = sun_dir;

  for (var i: u32 = 0u; i < cam.chunk_count; i = i + 1u) {
    if (trace_chunk_shadow(ro, rd, chunks[i], 0.0)) {
      return true;
    }
  }
  return false;
}

// ------------------------------------------------------------
// Primary ray traversal (geometry-only hit)
// ------------------------------------------------------------

struct HitGeom {
  hit : bool,
  t   : f32,
  mat : u32,
  n   : vec3<f32>,
};

fn trace_chunk_hybrid(ro: vec3<f32>, rd: vec3<f32>, ch: ChunkMeta) -> HitGeom {
  let voxel_size = cam.voxel_params.x;

  let root_bmin_vox = vec3<f32>(f32(ch.origin.x), f32(ch.origin.y), f32(ch.origin.z));
  let root_bmin = root_bmin_vox * voxel_size;
  let root_size = f32(cam.chunk_size) * voxel_size;
  let root_bmax = root_bmin + vec3<f32>(root_size);

  let rt = intersect_aabb(ro, rd, root_bmin, root_bmax);
  let t_enter = max(rt.x, 0.0);
  let t_exit  = rt.y;

  if (t_exit < t_enter) {
    return HitGeom(false, BIG_F32, 0u, vec3<f32>(0.0));
  }

  var tcur = t_enter + 1e-4;

  for (var step_i: u32 = 0u; step_i < cam.max_steps; step_i = step_i + 1u) {
    if (tcur > t_exit) { break; }

    let p = ro + tcur * rd;
    let q = query_leaf_at(p, root_bmin, root_size, ch.node_base);

    if (q.mat != 0u) {
      // Leaves: intersect displaced cube for visible cube motion.
      if (q.mat == 5u) {
        let time_s   = cam.voxel_params.y;
        let strength = cam.voxel_params.z;

        let h2 = leaf_displaced_cube_hit(
          ro, rd,
          q.bmin, q.size,
          time_s, strength,
          tcur - 1e-4,
          t_exit
        );

        if (h2.hit) {
          return HitGeom(true, h2.t, 5u, h2.n);
        }

        // Displaced cube missed: treat as empty and keep marching.
        let t_leave = exit_time_from_cube(ro, rd, q.bmin, q.size);
        tcur = max(t_leave, tcur) + 1e-4;
        continue;
      }

      // Other materials: shade the static cube face normal at current sample.
      let hp = ro + tcur * rd;
      let nn = cube_normal(hp, q.bmin, q.size);
      return HitGeom(true, tcur, q.mat, nn);
    }

    let t_leave = exit_time_from_cube(ro, rd, q.bmin, q.size);
    tcur = max(t_leave, tcur) + 1e-4;
  }

  return HitGeom(false, BIG_F32, 0u, vec3<f32>(0.0));
}

// ------------------------------------------------------------
// Shading (done once per pixel, after nearest hit is selected)
// ------------------------------------------------------------

fn shade_hit(ro: vec3<f32>, rd: vec3<f32>, hg: HitGeom) -> vec3<f32> {
  let hp = ro + hg.t * rd;
  let base = color_for_material(hg.mat);

  // One shadow ray total (not per candidate chunk).
  let shadow = select(1.0, 0.0, in_shadow(hp, SUN_DIR));

  let diff = max(dot(hg.n, SUN_DIR), 0.0);
  let ambient = select(0.22, 0.28, hg.mat == 5u);

  // Optional foliage dapple (only runs if the final hit is a leaf).
  var dapple = 1.0;
  if (hg.mat == 5u) {
    let time_s = cam.voxel_params.y;
    let d0 = sin(dot(hp.xz, vec2<f32>(3.0, 2.2)) + time_s * 3.5);
    let d1 = sin(dot(hp.xz, vec2<f32>(6.5, 4.1)) - time_s * 6.0);
    dapple = 0.90 + 0.10 * (0.6 * d0 + 0.4 * d1);
  }

  return base * (ambient + (1.0 - ambient) * diff * shadow) * dapple;
}

// ------------------------------------------------------------
// Entry point: per-pixel ray, test all chunks, keep nearest hit
// ------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(out_img);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let res = vec2<f32>(f32(dims.x), f32(dims.y));
  let px  = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

  let ro = cam.cam_pos.xyz;
  let rd = ray_dir_from_pixel(px, res);

  // Background (simple sky gradient).
  let tsky = clamp(0.5 * (rd.y + 1.0), 0.0, 1.0);
  let sky = mix(
    vec3<f32>(0.05, 0.08, 0.12),
    vec3<f32>(0.6, 0.8, 1.0),
    tsky
  );

  // Nearest geometry hit across all chunks.
  var best = HitGeom(false, BIG_F32, 0u, vec3<f32>(0.0));

  // Chunk AABB early-out in main(): skip chunks that can't beat current best.t.
  let voxel_size = cam.voxel_params.x;
  let chunk_size_m = f32(cam.chunk_size) * voxel_size;

  for (var i: u32 = 0u; i < cam.chunk_count; i = i + 1u) {
    let ch = chunks[i];

    let root_bmin = vec3<f32>(f32(ch.origin.x), f32(ch.origin.y), f32(ch.origin.z)) * voxel_size;
    let root_bmax = root_bmin + vec3<f32>(chunk_size_m);

    let rt = intersect_aabb(ro, rd, root_bmin, root_bmax);
    let t_enter = max(rt.x, 0.0);
    let t_exit  = rt.y;

    if (t_exit < t_enter) { continue; }
    if (t_enter >= best.t) { continue; }

    let h = trace_chunk_hybrid(ro, rd, ch);
    if (h.hit && h.t < best.t) {
      best = h;
    }
  }

  let col = select(sky, shade_hit(ro, rd, best), best.hit);
  textureStore(out_img, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(col, 1.0));
}
