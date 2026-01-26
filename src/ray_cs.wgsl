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
//     * if material is solid -> shade and return the hit
//
// Leaf wind (voxel cubes that move)
// - Leaves are still voxel cubes, but their cube AABB is displaced by a stable wind field.
// - When we query a leaf voxel cube, we intersect the ray against the *displaced* cube.
//   If it hits, we shade using the displaced cube normal.
//   If it misses (because the cube moved away), we treat it as empty and keep marching.
//
// Notes on abbreviations (first use)
// - SVO  = Sparse Voxel Octree
// - AABB = Axis-Aligned Bounding Box
// - NDC  = Normalized Device Coordinates

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

const LEAF_U32 : u32 = 0xFFFFFFFFu;
const BIG_F32  : f32 = 1e30;
const EPS_INV  : f32 = 1e-8;

// ------------------------------------------------------------
// Ray reconstruction (pixel -> world ray direction)
// ------------------------------------------------------------
//
// We reconstruct a ray by:
// 1) mapping pixel coords to NDC
// 2) unprojecting with inverse projection
// 3) transforming with inverse view to world space
//
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
// Returns (t_enter, t_exit). If t_exit < t_enter -> no hit.
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
// Sparse children addressing
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
//
// Design goals:
// - Nearby leaves should move together (avoid "sparkle").
// - Motion increases with height (less motion near ground).
// - Combine a slow gust + faster flutter.
//
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

  // More motion higher up (tuned for your tree scale).
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
//
// Leaves remain cubes. We displace the cube AABB by a wind offset,
// intersect the ray against the displaced cube, and shade if it hits.
//
// Important: displacement is clamped to < ~0.5 * cube size so it still
// reads as "voxel cubes" and doesn't detach too far from the canopy.
//
fn clamp_len(v: vec3<f32>, max_len: f32) -> vec3<f32> {
  let l2 = dot(v, v);
  if (l2 <= max_len * max_len) { return v; }
  return v * (max_len / sqrt(l2));
}

fn leaf_cube_offset(bmin: vec3<f32>, size: f32, time_s: f32, strength: f32) -> vec3<f32> {
  let center = bmin + vec3<f32>(0.5 * size);

  // Wind is already grouped in space; keep it mostly horizontal here.
  var w = wind_field(center, time_s) * strength;
  w = vec3<f32>(w.x, 0.15 * w.y, w.z);

  // Amplitude relative to cube size. Tune this first.
  let amp = 0.35 * size;

  // Clamp prevents cubes drifting too far and breaking the voxel look.
  return clamp_len(w * amp, 0.45 * size);
}

struct LeafCubeHit {
  hit  : bool,
  t    : f32,
  n    : vec3<f32>,
  bmin : vec3<f32>, // displaced bmin used for normal
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
  let off  = leaf_cube_offset(bmin, size, time_s, strength);
  let bmin2 = bmin + off;
  let bmax2 = bmin2 + vec3<f32>(size);

  // Intersect the displaced cube.
  let rt = intersect_aabb(ro, rd, bmin2, bmax2);
  var t0 = rt.x;
  let t1 = rt.y;

  if (t1 < t0) {
    return LeafCubeHit(false, BIG_F32, vec3<f32>(0.0), bmin2);
  }

  // Clamp against the caller's valid interval for this chunk.
  t0 = max(t0, t_min);
  if (t0 > min(t1, t_max)) {
    return LeafCubeHit(false, BIG_F32, vec3<f32>(0.0), bmin2);
  }

  let hp = ro + t0 * rd;
  let nn = cube_normal(hp, bmin2, size);

  return LeafCubeHit(true, t0, nn, bmin2);
}

// ------------------------------------------------------------
// Hybrid “skip-empty leaf” traversal (per chunk)
// ------------------------------------------------------------
//
// We iterate along the ray and repeatedly query the SVO at the current position.
// The query returns the *leaf region cube* that contains the point.
// If that leaf region is empty, we jump to its exit face.
//
// This gives large steps through air, but still hits solid voxels precisely.
//
fn safe_inv(x: f32) -> f32 {
  return select(1.0 / x, BIG_F32, abs(x) < EPS_INV);
}

struct LeafQuery {
  bmin : vec3<f32>, // cube min (world meters)
  size : f32,       // cube size (world meters)
  mat  : u32,       // 0 = empty / air
};

// Query the SVO to find the leaf region containing point p.
// If the child octant is missing, we return "empty leaf cube" for that space.
fn query_leaf_at(
  p: vec3<f32>,
  root_bmin: vec3<f32>,
  root_size: f32,
  node_base: u32
) -> LeafQuery {
  var idx: u32 = node_base;
  var bmin: vec3<f32> = root_bmin;
  var size: f32 = root_size;

  // Stop descending once we're at voxel resolution.
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

  // If we somehow exceed max depth, treat as empty.
  return LeafQuery(bmin, size, 0u);
}

// Compute the parametric t where the ray exits a cube.
// Used to "skip" empty leaf regions efficiently.
fn exit_time_from_cube(ro: vec3<f32>, rd: vec3<f32>, bmin: vec3<f32>, size: f32) -> f32 {
  let bmax = bmin + vec3<f32>(size);
  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  let tx = (select(bmin.x, bmax.x, rd.x > 0.0) - ro.x) * inv.x;
  let ty = (select(bmin.y, bmax.y, rd.y > 0.0) - ro.y) * inv.y;
  let tz = (select(bmin.z, bmax.z, rd.z > 0.0) - ro.z) * inv.z;

  return min(tx, min(ty, tz));
}

struct Hit {
  hit : bool,
  t   : f32,
  col : vec3<f32>,
};

fn trace_chunk_hybrid(ro: vec3<f32>, rd: vec3<f32>, ch: ChunkMeta) -> Hit {
  let voxel_size = cam.voxel_params.x;

  // Chunk root cube in world meters.
  let root_bmin_vox = vec3<f32>(f32(ch.origin.x), f32(ch.origin.y), f32(ch.origin.z));
  let root_bmin = root_bmin_vox * voxel_size;
  let root_size = f32(cam.chunk_size) * voxel_size;
  let root_bmax = root_bmin + vec3<f32>(root_size);

  // Early reject: ray misses the chunk AABB.
  let rt = intersect_aabb(ro, rd, root_bmin, root_bmax);
  let t_enter = max(rt.x, 0.0);
  let t_exit  = rt.y;

  if (t_exit < t_enter) {
    return Hit(false, BIG_F32, vec3<f32>(0.0));
  }

  // Start slightly inside to avoid self-intersection at boundaries.
  var tcur = t_enter + 1e-4;

  for (var step_i: u32 = 0u; step_i < cam.max_steps; step_i = step_i + 1u) {
    if (tcur > t_exit) { break; }

    // Query SVO leaf region containing current point.
    let p = ro + tcur * rd;
    let q = query_leaf_at(p, root_bmin, root_size, ch.node_base);

    // Non-empty material: shade (with special path for leaves).
    if (q.mat != 0u) {
      let sun = normalize(vec3<f32>(0.6, 1.0, 0.2));

      // Leaves: intersect displaced cube to get visible cube motion.
      if (q.mat == 5u) {
        let time_s   = cam.voxel_params.y;
        let strength = cam.voxel_params.z;

        // Use a small backstep so "starting on the face" still counts as a hit.
        let h2 = leaf_displaced_cube_hit(
          ro, rd,
          q.bmin, q.size,
          time_s, strength,
          tcur - 1e-4,
          t_exit
        );

        if (h2.hit) {
          let hp = ro + h2.t * rd;
          let base = color_for_material(5u);

          // Lighting from displaced cube normal.
          let diff = max(dot(h2.n, sun), 0.0);

          // Small moving dapple helps show motion even under flat lighting.
          let d0 = sin(dot(hp.xz, vec2<f32>(3.0, 2.2)) + time_s * 3.5);
          let d1 = sin(dot(hp.xz, vec2<f32>(6.5, 4.1)) - time_s * 6.0);
          let dapple = 0.90 + 0.10 * (0.6 * d0 + 0.4 * d1);

          let col = base * (0.22 + 0.78 * diff) * dapple;
          return Hit(true, h2.t, col);
        }

        // Leaf cube moved away from the ray at this sample point: treat as empty and continue.
        let t_leave = exit_time_from_cube(ro, rd, q.bmin, q.size);
        tcur = max(t_leave, tcur) + 1e-4;
        continue;
      }

      // Everything else: shade the static cube face we’re currently inside.
      let hp = ro + tcur * rd;
      let base = color_for_material(q.mat);
      let nn = cube_normal(hp, q.bmin, q.size);

      let diff = max(dot(nn, sun), 0.0);
      let col = base * (0.25 + 0.75 * diff);
      return Hit(true, tcur, col);
    }

    // Empty leaf region: skip to the exit face and continue.
    let t_leave = exit_time_from_cube(ro, rd, q.bmin, q.size);
    tcur = max(t_leave, tcur) + 1e-4;
  }

  return Hit(false, BIG_F32, vec3<f32>(0.0));
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
  var best_col: vec3<f32> = mix(
    vec3<f32>(0.05, 0.08, 0.12),
    vec3<f32>(0.6, 0.8, 1.0),
    tsky
  );
  var best_t: f32 = BIG_F32;

  // Test all resident chunks and keep the closest hit.
  for (var i: u32 = 0u; i < cam.chunk_count; i = i + 1u) {
    let h = trace_chunk_hybrid(ro, rd, chunks[i]);
    if (h.hit && h.t < best_t) {
      best_t = h.t;
      best_col = h.col;
    }
  }

  textureStore(out_img, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(best_col, 1.0));
}
