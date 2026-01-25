// ray_cs.wgsl
//
// Multi-chunk Hybrid SVO traversal (skip-empty) compute shader.
//
// Goal:
// - Keep the “feel” of DDA stepping (advance along ray), but instead of stepping voxel-by-voxel,
//   we jump across EMPTY LEAF REGIONS of the SVO.
// - Now supports MULTIPLE chunks at once:
//   - CPU uploads a list of ChunkMeta (origin + node_base into packed node buffer).
//   - For each pixel ray we test all chunks, trace each hit candidate, and keep the nearest hit.
//
// Notes on abbreviations (first use):
// - SVO  = Sparse Voxel Octree
// - AABB = Axis-Aligned Bounding Box
// - NDC  = Normalized Device Coordinates

struct Node {
  // If internal: index of the first child in a *compact list*.
  // If leaf: 0xFFFF_FFFF.
  child_base : u32,

  // Bit i indicates whether octant i exists (i in 0..7).
  // Children are stored compactly in increasing octant index order (only set bits).
  child_mask : u32,

  // For leaf nodes: material id (0 = empty / air).
  // For internal nodes: unused (0) in this shader.
  material   : u32,

  _pad       : u32,
};

struct Camera {
  // Inverse matrices so the shader can reconstruct per-pixel world rays:
  // - proj_inv : clip/NDC -> view space
  // - view_inv : view -> world space
  view_inv     : mat4x4<f32>,
  proj_inv     : mat4x4<f32>,

  // Camera position in world space (xyz used).
  cam_pos      : vec4<f32>,

  // Chunk edge length in voxels (power of two). All chunks share this size.
  chunk_size   : u32,

  // Number of valid entries in `chunks`.
  chunk_count  : u32,

  // Safety cap: max number of skip-iterations per chunk (NOT voxel steps).
  max_steps    : u32,

  _pad0        : u32,
};

struct ChunkMeta {
  origin    : vec4<i32>, // chunk min corner in world voxel coords
  node_base : u32,       // base index into packed nodes array
  node_count: u32,       // optional (unused by shader, but handy for debugging)
  _pad0     : u32,
  _pad1     : u32,
};

@group(0) @binding(0) var<uniform> cam : Camera;
@group(0) @binding(1) var<storage, read> chunks : array<ChunkMeta>;
@group(0) @binding(2) var<storage, read> nodes  : array<Node>;
@group(0) @binding(3) var out_img : texture_storage_2d<rgba16float, write>;

const LEAF_U32 : u32 = 0xFFFFFFFFu;
const BIG_F32  : f32 = 1e30;
const EPS_INV  : f32 = 1e-8;


// ------------------------------------------------------------
// Ray reconstruction
// ------------------------------------------------------------

// Convert pixel -> world-space ray direction using inverse matrices.
fn ray_dir_from_pixel(px: vec2<f32>, res: vec2<f32>) -> vec3<f32> {
  // NDC coordinates in [-1, 1]. We use z=1 to point forward.
  let ndc = vec4<f32>(
    2.0 * px.x / res.x - 1.0,
    1.0 - 2.0 * px.y / res.y,
    1.0,
    1.0
  );

  // NDC -> view. Make it a direction by setting w=0.
  let view = cam.proj_inv * ndc;
  let vdir = vec4<f32>(view.xyz / view.w, 0.0);

  // View -> world direction.
  let wdir = (cam.view_inv * vdir).xyz;
  return normalize(wdir);
}


// ------------------------------------------------------------
// Ray / AABB intersection
// ------------------------------------------------------------

// Ray vs AABB intersection (slab method).
// Returns [t_enter, t_exit]. If miss, returns [1, 0].
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

// Rank addressing for compact children.
// rank = number of set bits in `mask` with index < ci.
fn child_rank(mask: u32, ci: u32) -> u32 {
  let bit = 1u << ci;
  let lower = mask & (bit - 1u);
  return countOneBits(lower);
}


// ------------------------------------------------------------
// Materials + simple shading
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

// Approximate face normal at hit point `hp` on a cube [bmin, bmin+size].
// Chooses the closest face by distance-to-plane.
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
// Hybrid “skip-empty leaf” traversal (per chunk)
// ------------------------------------------------------------

fn safe_inv(x: f32) -> f32 {
  return select(1.0 / x, BIG_F32, abs(x) < EPS_INV);
}

struct LeafQuery {
  bmin : vec3<f32>, // world-space cube min
  size : f32,       // cube edge length
  mat  : u32,       // 0 for empty
};

// Point-descent into the SVO to find the leaf (or missing child => empty leaf) that contains p.
// p must be inside the chunk root cube.
//
// Multi-chunk change:
// - `node_base` offsets into the packed node buffer.
// - Root of this chunk is always at `node_base + 0`.
fn query_leaf_at(
  p: vec3<f32>,
  root_bmin: vec3<f32>,
  root_size: f32,
  node_base: u32
) -> LeafQuery {
  var idx: u32 = node_base;
  var bmin: vec3<f32> = root_bmin;
  var size: f32 = root_size;

  for (var d: u32 = 0u; d < 32u; d = d + 1u) {
    let n = nodes[idx];

    if (n.child_base == LEAF_U32) {
      return LeafQuery(bmin, size, n.material);
    }

    if (size <= 1.0) {
      return LeafQuery(bmin, size, 0u);
    }

    let half = size * 0.5;
    let mid = bmin + vec3<f32>(half);

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
    if ((n.child_mask & bit) == 0u) {
      return LeafQuery(child_bmin, half, 0u);
    }

    // IMPORTANT: `n.child_base` is also relative to the packed `nodes` array,
    // because CPU stored absolute indices inside the packed buffer.
    //
    // Your CPU builder currently builds per-chunk nodes starting at 0.
    // When packing chunks, you must add `node_base` to any internal `child_base`.
    //
    // If you DID NOT patch child_base during packing, then you MUST treat child_base as
    // relative-to-chunk and add node_base here. We do that (safe) version below.

    let rank = child_rank(n.child_mask, ci);

    // Treat child_base as chunk-relative:
    idx = node_base + (n.child_base + rank);

    bmin = child_bmin;
    size = half;
  }

  return LeafQuery(bmin, size, 0u);
}

// Compute the ray parameter t at which the ray exits a cube AABB,
// assuming the current point is inside the cube.
// Exit is the nearest forward face along the ray.
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

// Trace one chunk using hybrid skip-empty traversal.
// Returns a Hit with the ray parameter t of the accepted hit (for nearest selection).
fn trace_chunk_hybrid(ro: vec3<f32>, rd: vec3<f32>, ch: ChunkMeta) -> Hit {
  let root_bmin = vec3<f32>(
    f32(ch.origin.x),
    f32(ch.origin.y),
    f32(ch.origin.z)
  );
  let root_size = f32(cam.chunk_size);
  let root_bmax = root_bmin + vec3<f32>(root_size);

  let rt = intersect_aabb(ro, rd, root_bmin, root_bmax);
  let t_enter = max(rt.x, 0.0);
  let t_exit  = rt.y;

  if (t_exit < t_enter) {
    return Hit(false, BIG_F32, vec3<f32>(0.0));
  }

  var tcur = t_enter + 1e-4;

  for (var step_i: u32 = 0u; step_i < cam.max_steps; step_i = step_i + 1u) {
    if (tcur > t_exit) { break; }

    let p = ro + tcur * rd;

    // Point query down the SVO (node_base selects which chunk's tree).
    let q = query_leaf_at(p, root_bmin, root_size, ch.node_base);

    if (q.mat != 0u) {
      let hp = ro + tcur * rd;
      let base = color_for_material(q.mat);

      let nn = cube_normal(hp, q.bmin, q.size);
      let sun = normalize(vec3<f32>(0.6, 1.0, 0.2));
      let diff = max(dot(nn, sun), 0.0);
      let col = base * (0.25 + 0.75 * diff);

      return Hit(true, tcur, col);
    }

    let t_leave = exit_time_from_cube(ro, rd, q.bmin, q.size);
    tcur = max(t_leave, tcur) + 1e-4;
  }

  return Hit(false, BIG_F32, vec3<f32>(0.0));
}


// ------------------------------------------------------------
// Entry point: loop over chunks and keep nearest hit
// ------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(out_img);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let res = vec2<f32>(f32(dims.x), f32(dims.y));
  let px  = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

  let ro = cam.cam_pos.xyz;
  let rd = ray_dir_from_pixel(px, res);

  var best_t: f32 = BIG_F32;
  var best_col: vec3<f32> = vec3<f32>(0.0);

  // NOTE: chunks is a runtime-sized array; looping by cam.chunk_count is ok.
  for (var i: u32 = 0u; i < cam.chunk_count; i = i + 1u) {
    let h = trace_chunk_hybrid(ro, rd, chunks[i]);
    if (h.hit && h.t < best_t) {
      best_t = h.t;
      best_col = h.col;
    }
  }

  let out_col = vec4<f32>(best_col, 1.0);
  textureStore(out_img, vec2<i32>(i32(gid.x), i32(gid.y)), out_col);
}
