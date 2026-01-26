struct NodeGpu {
  child_base : u32,
  child_mask : u32,
  material   : u32,
  _pad       : u32,
};

struct ChunkBuildParams {
  origin_vox: vec4<i32>,
  node_base: u32,
  node_count: u32,
  chunk_size: u32,
  brick_size: u32,
  seed: u32,
  voxels_per_meter: u32,
  _pad0: u32,
};

@group(0) @binding(0) var<uniform> params: ChunkBuildParams;
@group(0) @binding(1) var<storage, read_write> nodes : array<NodeGpu>;

const LEAF : u32 = 0xFFFFFFFFu;

// materials (match your Rust constants)
const AIR   : u32 = 0u;
const GRASS : u32 = 1u;
const DIRT  : u32 = 2u;
const STONE : u32 = 3u;

// --- hashing (ported from your world/hash.rs style) ---
fn hash_u32(v_in: u32) -> u32 {
  var v = v_in;
  v = v ^ (v >> 16u);
  v = v * 0x7feb352du;
  v = v ^ (v >> 15u);
  v = v * 0x846ca68bu;
  v = v ^ (v >> 16u);
  return v;
}

fn hash2(seed: u32, x: i32, z: i32) -> u32 {
  let a = u32(x) * 0x9e3779b1u;
  let b = u32(z) * 0x85ebca6bu;
  return hash_u32(seed ^ a ^ b);
}

fn u01(v: u32) -> f32 {
  return f32(v) * (1.0 / 4294967296.0);
}

// very cheap height function (replace later with your exact noise)
fn ground_height_vox(x_vox: i32, z_vox: i32) -> i32 {
  let vpm_u = max(1u, params.voxels_per_meter);
  let vpm = i32(vpm_u);

  // convert to meters grid (coarse)
  let xm = x_vox / vpm;
  let zm = z_vox / vpm;

  // 2 octave-ish hash noise
  let r0 = hash2(params.seed, xm, zm);
  let r1 = hash2(params.seed ^ 0xA5A5A5A5u, xm * 2, zm * 2);

  let h0 = (u01(r0) - 0.5) * 2.0;
  let h1 = (u01(r1) - 0.5) * 2.0;

  let base_m = 10.0;
  let amp_m  = 18.0;

  let hills_m = h0 * amp_m + h1 * 3.0;
  let height_m = base_m + hills_m;

  return i32(round(height_m * f32(vpm_u)));
}

fn material_at_world(x_vox: i32, y_vox: i32, z_vox: i32) -> u32 {
  let g = ground_height_vox(x_vox, z_vox);
  let vpm = i32(max(1u, params.voxels_per_meter));

  if (y_vox < g) {
    if (y_vox >= g - 3 * vpm) { return DIRT; }
    return STONE;
  }
  if (y_vox == g) { return GRASS; }
  return AIR;
}

// Fixed layout indices (relative to node_base)
fn idx_root() -> u32 { return 0u; }
fn idx_l2(i: u32) -> u32 { return 1u + i; }                 // i in [0..7]
fn idx_l1(j: u32) -> u32 { return 1u + 8u + j; }            // j in [0..63]
fn idx_leaf(k: u32) -> u32 { return 1u + 8u + 64u + k; }    // k in [0..511]

// Parent->child base mapping (dense, always 8 children)
fn l2_child_base_rel(i: u32) -> u32 {
  return idx_l1(i * 8u);
}
fn l1_child_base_rel(j: u32) -> u32 {
  return idx_leaf(j * 8u);
}

@compute @workgroup_size(64)
fn main_chunk_build(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tid = gid.x;

  // total nodes = 1 + 8 + 64 + 512 = 585
  if (tid >= 585u) { return; }

  // Safety: don't write past the allocated range
  if (tid >= params.node_count) { return; }

  let base = params.node_base;

  // root
  if (tid == idx_root()) {
    var n: NodeGpu;
    n.child_base = base + idx_l2(0u);
    n.child_mask = 0xFFu;
    n.material = 0u;
    n._pad = 0u;
    nodes[base + tid] = n;
    return;
  }

  // level2 [1..9)
  if (tid >= 1u && tid < 1u + 8u) {
    let i = tid - 1u;

    var n: NodeGpu;
    n.child_base = base + l2_child_base_rel(i);
    n.child_mask = 0xFFu;
    n.material = 0u;
    n._pad = 0u;
    nodes[base + tid] = n;
    return;
  }

  // level1 [9..73)
  if (tid >= 1u + 8u && tid < 1u + 8u + 64u) {
    let j = tid - (1u + 8u);

    var n: NodeGpu;
    n.child_base = base + l1_child_base_rel(j);
    n.child_mask = 0xFFu;
    n.material = 0u;
    n._pad = 0u;
    nodes[base + tid] = n;
    return;
  }

  // leaves: one per brick cell (8x8x8 bricks => 512)
  let leaf_k = tid - (1u + 8u + 64u); // 0..511

  let bpa = max(1u, params.chunk_size / max(1u, params.brick_size)); // bricks per axis
  // decode brick coords
  let bx = i32(leaf_k % bpa);
  let by = i32((leaf_k / bpa) % bpa);
  let bz = i32(leaf_k / (bpa * bpa));

  // sample at brick center (good enough for stage 1)
  let half = i32(max(1u, params.brick_size) / 2u);
  let bs_i = i32(max(1u, params.brick_size));

  let wx = params.origin_vox.x + bx * bs_i + half;
  let wy = params.origin_vox.y + by * bs_i + half;
  let wz = params.origin_vox.z + bz * bs_i + half;

  let m = material_at_world(wx, wy, wz);

  var leaf: NodeGpu;
  leaf.child_base = LEAF;
  leaf.child_mask = 0u;
  leaf.material = m;
  leaf._pad = 0u;

  nodes[base + tid] = leaf;
}
