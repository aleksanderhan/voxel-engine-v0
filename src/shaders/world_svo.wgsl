// src/shaders/world_svo.wgsl
// --------------------------
//
// SVO per chunk (32^3 voxels, depth=5).
// IMPORTANT: GRID_X/Y/Z MUST MATCH src/config.rs

const CHUNK_VOXELS : u32 = 32u;
const CHUNK_DEPTH  : u32 = 5u;

// MUST MATCH config.rs (r=8,2,8 => 17,5,17)
const GRID_X       : u32 = 17u;
const GRID_Y       : u32 = 5u;
const GRID_Z       : u32 = 17u;

const MAX_CHUNKS   : u32 = GRID_X * GRID_Y * GRID_Z;

const FLAG_ACTIVE : u32 = 1u;
const FLAG_DIRTY  : u32 = 2u;
const FLAG_READY  : u32 = 4u;

const NODES_PER_CHUNK : u32 = 37449u;

const MAX_CHUNK_STEPS : u32 = 256u;
const MAX_STACK       : u32 = 48u;
const MAX_NODE_VISITS : u32 = 256u;

const TRACE_MAX_DIST_DEFAULT : f32 = 140.0;

struct ChunkMeta {
  x : i32,
  y : i32,
  z : i32,
  flags : u32,
};

@group(0) @binding(1) var<storage, read_write> nodes : array<u32>;
@group(0) @binding(2) var<storage, read_write> chunk_meta : array<ChunkMeta>;
@group(0) @binding(6) var<storage, read> dirty_slots : array<u32>;

fn imod(a: i32, m: i32) -> i32 { return (a % m + m) % m; }

fn slot_from_phys(px: i32, py: i32, pz: i32) -> u32 {
  return u32((pz * i32(GRID_X) + px) * i32(GRID_Y) + py);
}

fn find_slot_for_chunk(cc: vec3<i32>) -> u32 {
  let ix = cc.x - stream.origin_x;
  let iy = cc.y - stream.origin_y;
  let iz = cc.z - stream.origin_z;

  if (ix < 0 || iy < 0 || iz < 0) { return 0xffffffffu; }
  if (ix >= i32(GRID_X) || iy >= i32(GRID_Y) || iz >= i32(GRID_Z)) { return 0xffffffffu; }

  let px = imod(ix + i32(stream.ox), i32(GRID_X));
  let py = imod(iy + i32(stream.oy), i32(GRID_Y));
  let pz = imod(iz + i32(stream.oz), i32(GRID_Z));

  let slot = slot_from_phys(px, py, pz);
  let cm = chunk_meta[slot];

  if ((cm.flags & FLAG_ACTIVE) == 0u) { return 0xffffffffu; }

  // KEY CHANGE:
  // Render is allowed while DIRTY (stale data) as long as READY is set.
  // New / repurposed slots start NOT READY, so they won't render wrong data.
  if ((cm.flags & FLAG_READY) == 0u) { return 0xffffffffu; }

  if (cm.x != cc.x || cm.y != cc.y || cm.z != cc.z) { return 0xffffffffu; }

  return slot;
}

fn chunk_size_m() -> f32 {
  return cam.params.z * f32(CHUNK_VOXELS);
}

fn chunk_origin_m(cc: vec3<i32>) -> vec3<f32> {
  let cs = chunk_size_m();
  return vec3<f32>(f32(cc.x) * cs, f32(cc.y) * cs, f32(cc.z) * cs);
}

fn node_base(slot: u32) -> u32 { return slot * NODES_PER_CHUNK; }

fn child_index(parent: u32, child_id: u32) -> u32 {
  return parent * 8u + 1u + (child_id & 7u);
}

fn child_id_from_bits(xb: u32, yb: u32, zb: u32) -> u32 {
  return ((xb & 1u) << 2u) | ((yb & 1u) << 1u) | (zb & 1u);
}

fn node_index_from_coord(coord: vec3<u32>, levels: u32) -> u32 {
  var idx: u32 = 0u;
  for (var l: u32 = 0u; l < levels; l = l + 1u) {
    let shift = (levels - 1u) - l;
    let xb = (coord.x >> shift) & 1u;
    let yb = (coord.y >> shift) & 1u;
    let zb = (coord.z >> shift) & 1u;
    let cid = child_id_from_bits(xb, yb, zb);
    idx = child_index(idx, cid);
  }
  return idx;
}

// -----------------------------------------------------------------------------
// Procedural terrain
// -----------------------------------------------------------------------------

fn terrain_height_m(x: f32, z: f32) -> f32 {
  let h0 = 2.0 * sin(x * 0.05) + 2.0 * cos(z * 0.05);
  let h1 = 1.2 * sin((x + z) * 0.03);
  return h0 + h1;
}

fn voxel_material(world_p_m: vec3<f32>) -> u32 {
  if (world_p_m.y < -10.0) { return 2u; }

  let h = terrain_height_m(world_p_m.x, world_p_m.z);
  if (world_p_m.y < h) {
    if (world_p_m.y > h - 1.2) { return 1u; }
    return 2u;
  }

  let s = sin(world_p_m.x * 0.12) * cos(world_p_m.z * 0.12);
  if (s > 0.985 && world_p_m.y < 6.0) { return 3u; }
  return 0u;
}

// -----------------------------------------------------------------------------
// Batching helpers
// -----------------------------------------------------------------------------

fn batch_index(local_i: u32) -> u32 {
  return stream.dirty_offset + local_i;
}

// -----------------------------------------------------------------------------
// Build passes (batched)
// -----------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn build_leaves(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= 32u || gid.y >= 32u) { return; }

  let local_chunk_i = gid.z / 32u;
  let vz            = gid.z - local_chunk_i * 32u;

  if (local_chunk_i >= stream.build_count) { return; }

  let di = batch_index(local_chunk_i);
  if (di >= stream.dirty_count) { return; }

  let slot = dirty_slots[di];
  if (slot >= MAX_CHUNKS) { return; }

  let cm = chunk_meta[slot];
  if ((cm.flags & FLAG_ACTIVE) == 0u) { return; }
  if ((cm.flags & FLAG_DIRTY) == 0u) { return; }

  let base = node_base(slot);

  let cs = chunk_size_m();
  let origin = vec3<f32>(f32(cm.x) * cs, f32(cm.y) * cs, f32(cm.z) * cs);
  let vs = cam.params.z;

  let local = vec3<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5, f32(vz) + 0.5);
  let wp = origin + local * vs;

  let mat = voxel_material(wp);
  let leaf_idx = node_index_from_coord(vec3<u32>(gid.x, gid.y, vz), 5u);
  nodes[base + leaf_idx] = mat;
}

@compute @workgroup_size(8, 8, 1)
fn build_L4(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= 16u || gid.y >= 16u) { return; }

  let local_chunk_i = gid.z / 16u;
  let iz4           = gid.z - local_chunk_i * 16u;

  if (local_chunk_i >= stream.build_count) { return; }

  let di = batch_index(local_chunk_i);
  if (di >= stream.dirty_count) { return; }

  let slot = dirty_slots[di];
  if (slot >= MAX_CHUNKS) { return; }

  let cm = chunk_meta[slot];
  if ((cm.flags & FLAG_ACTIVE) == 0u) { return; }
  if ((cm.flags & FLAG_DIRTY) == 0u) { return; }

  let base = node_base(slot);

  let coord4 = vec3<u32>(gid.x, gid.y, iz4);
  let n4 = node_index_from_coord(coord4, 4u);

  var mask: u32 = 0u;
  for (var c: u32 = 0u; c < 8u; c = c + 1u) {
    let li = child_index(n4, c);
    let mat = nodes[base + li];
    if (mat != 0u) { mask = mask | (1u << c); }
  }

  nodes[base + n4] = mask & 0xffu;
}

@compute @workgroup_size(1, 1, 1)
fn build_L3(@builtin(global_invocation_id) gid: vec3<u32>) {
  let local_chunk_i = gid.z / 8u;
  let iz3           = gid.z - local_chunk_i * 8u;

  if (local_chunk_i >= stream.build_count) { return; }

  let di = batch_index(local_chunk_i);
  if (di >= stream.dirty_count) { return; }

  let slot = dirty_slots[di];
  if (slot >= MAX_CHUNKS) { return; }

  let cm = chunk_meta[slot];
  if ((cm.flags & FLAG_ACTIVE) == 0u) { return; }
  if ((cm.flags & FLAG_DIRTY) == 0u) { return; }

  let base = node_base(slot);

  for (var ix: u32 = 0u; ix < 8u; ix = ix + 1u) {
    for (var iy: u32 = 0u; iy < 8u; iy = iy + 1u) {
      let coord3 = vec3<u32>(ix, iy, iz3);
      let n3 = node_index_from_coord(coord3, 3u);

      var mask: u32 = 0u;
      for (var c: u32 = 0u; c < 8u; c = c + 1u) {
        let n4 = child_index(n3, c);
        let m4 = nodes[base + n4] & 0xffu;
        if (m4 != 0u) { mask = mask | (1u << c); }
      }
      nodes[base + n3] = mask & 0xffu;
    }
  }
}

@compute @workgroup_size(1, 1, 1)
fn build_L2(@builtin(global_invocation_id) gid: vec3<u32>) {
  let local_chunk_i = gid.z / 4u;
  let iz2           = gid.z - local_chunk_i * 4u;

  if (local_chunk_i >= stream.build_count) { return; }

  let di = batch_index(local_chunk_i);
  if (di >= stream.dirty_count) { return; }

  let slot = dirty_slots[di];
  if (slot >= MAX_CHUNKS) { return; }

  let cm = chunk_meta[slot];
  if ((cm.flags & FLAG_ACTIVE) == 0u) { return; }
  if ((cm.flags & FLAG_DIRTY) == 0u) { return; }

  let base = node_base(slot);

  for (var ix: u32 = 0u; ix < 4u; ix = ix + 1u) {
    for (var iy: u32 = 0u; iy < 4u; iy = iy + 1u) {
      let coord2 = vec3<u32>(ix, iy, iz2);
      let n2 = node_index_from_coord(coord2, 2u);

      var mask: u32 = 0u;
      for (var c: u32 = 0u; c < 8u; c = c + 1u) {
        let n3 = child_index(n2, c);
        let m3 = nodes[base + n3] & 0xffu;
        if (m3 != 0u) { mask = mask | (1u << c); }
      }
      nodes[base + n2] = mask & 0xffu;
    }
  }
}

@compute @workgroup_size(1, 1, 1)
fn build_L1(@builtin(global_invocation_id) gid: vec3<u32>) {
  let local_chunk_i = gid.z / 2u;
  let iz1           = gid.z - local_chunk_i * 2u;

  if (local_chunk_i >= stream.build_count) { return; }

  let di = batch_index(local_chunk_i);
  if (di >= stream.dirty_count) { return; }

  let slot = dirty_slots[di];
  if (slot >= MAX_CHUNKS) { return; }

  let cm = chunk_meta[slot];
  if ((cm.flags & FLAG_ACTIVE) == 0u) { return; }
  if ((cm.flags & FLAG_DIRTY) == 0u) { return; }

  let base = node_base(slot);

  for (var ix: u32 = 0u; ix < 2u; ix = ix + 1u) {
    for (var iy: u32 = 0u; iy < 2u; iy = iy + 1u) {
      let coord1 = vec3<u32>(ix, iy, iz1);
      let n1 = node_index_from_coord(coord1, 1u);

      var mask: u32 = 0u;
      for (var c: u32 = 0u; c < 8u; c = c + 1u) {
        let n2 = child_index(n1, c);
        let m2 = nodes[base + n2] & 0xffu;
        if (m2 != 0u) { mask = mask | (1u << c); }
      }
      nodes[base + n1] = mask & 0xffu;
    }
  }
}

@compute @workgroup_size(1, 1, 1)
fn build_L0(@builtin(global_invocation_id) gid: vec3<u32>) {
  let local_chunk_i = gid.x;
  if (local_chunk_i >= stream.build_count) { return; }

  let di = batch_index(local_chunk_i);
  if (di >= stream.dirty_count) { return; }

  let slot = dirty_slots[di];
  if (slot >= MAX_CHUNKS) { return; }

  let cm = chunk_meta[slot];
  if ((cm.flags & FLAG_ACTIVE) == 0u) { return; }
  if ((cm.flags & FLAG_DIRTY) == 0u) { return; }

  let base = node_base(slot);
  let root: u32 = 0u;

  var mask: u32 = 0u;
  for (var c: u32 = 0u; c < 8u; c = c + 1u) {
    let n1 = child_index(root, c);
    let m1 = nodes[base + n1] & 0xffu;
    if (m1 != 0u) { mask = mask | (1u << c); }
  }
  nodes[base + root] = mask & 0xffu;
}

@compute @workgroup_size(64, 1, 1)
fn clear_dirty(@builtin(global_invocation_id) gid: vec3<u32>) {
  let local_i = gid.x;
  if (local_i >= stream.build_count) { return; }

  let di = batch_index(local_i);
  if (di >= stream.dirty_count) { return; }

  let slot = dirty_slots[di];
  if (slot >= MAX_CHUNKS) { return; }

  var cm = chunk_meta[slot];

  // Build completed for this slot:
  // - clear DIRTY
  // - set READY so it can render (even if it later becomes DIRTY again)
  cm.flags = (cm.flags & (~FLAG_DIRTY)) | FLAG_READY;

  chunk_meta[slot] = cm;
}

// -----------------------------------------------------------------------------
// Ray tracing
// -----------------------------------------------------------------------------

struct Hit {
  hit : bool,
  t   : f32,
  n   : vec3<f32>,
  mat : u32,
};

fn intersect_aabb(ro: vec3<f32>, rd: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec4<f32> {
  let inv = inv_rd(rd);

  let t0 = (bmin - ro) * inv;
  let t1 = (bmax - ro) * inv;

  let tmin3 = min(t0, t1);
  let tmax3 = max(t0, t1);

  let tmin = max(max(tmin3.x, tmin3.y), tmin3.z);
  let tmax = min(min(tmax3.x, tmax3.y), tmax3.z);

  var axis: f32 = 0.0;
  var best = tmin3.x;
  if (tmin3.y > best) { best = tmin3.y; axis = 1.0; }
  if (tmin3.z > best) { best = tmin3.z; axis = 2.0; }

  var s: f32 = 1.0;
  if (axis < 0.5) { s = select( 1.0, -1.0, rd.x > 0.0); }
  else if (axis < 1.5) { s = select( 1.0, -1.0, rd.y > 0.0); }
  else { s = select( 1.0, -1.0, rd.z > 0.0); }

  return vec4<f32>(tmin, tmax, axis, s);
}

fn normal_from_axis(axis: f32, sign: f32) -> vec3<f32> {
  if (axis < 0.5) { return vec3<f32>(sign, 0.0, 0.0); }
  if (axis < 1.5) { return vec3<f32>(0.0, sign, 0.0); }
  return vec3<f32>(0.0, 0.0, sign);
}

fn trace_chunk_svo(ro: vec3<f32>, rd: vec3<f32>, slot: u32, cc: vec3<i32>, seg_t0: f32, seg_t1: f32) -> Hit {
  let base = node_base(slot);
  let cs = chunk_size_m();
  let org = chunk_origin_m(cc);

  let bmin = org;
  let bmax = org + vec3<f32>(cs);

  let it = intersect_aabb(ro, rd, bmin, bmax);
  var t0 = it.x;
  var t1 = it.y;

  t0 = max(t0, seg_t0);
  t1 = min(t1, seg_t1);

  if (t1 < t0 || t1 < 0.0) {
    return Hit(false, 0.0, vec3<f32>(0.0), 0u);
  }

  var stack_node : array<u32, MAX_STACK>;
  var stack_bmin : array<vec3<f32>, MAX_STACK>;
  var stack_bmax : array<vec3<f32>, MAX_STACK>;
  var stack_t0   : array<f32, MAX_STACK>;
  var stack_t1   : array<f32, MAX_STACK>;
  var stack_d    : array<u32, MAX_STACK>;
  var stack_ax   : array<f32, MAX_STACK>;
  var stack_sg   : array<f32, MAX_STACK>;

  var sp: i32 = 0;

  stack_node[0] = 0u;
  stack_bmin[0] = bmin;
  stack_bmax[0] = bmax;
  stack_t0[0]   = t0;
  stack_t1[0]   = t1;
  stack_d[0]    = 0u;
  stack_ax[0]   = it.z;
  stack_sg[0]   = it.w;

  var visits: u32 = 0u;

  loop {
    if (sp < 0) { break; }
    if (visits >= MAX_NODE_VISITS) { break; }
    visits = visits + 1u;

    let n = stack_node[u32(sp)];
    let nbmin = stack_bmin[u32(sp)];
    let nbmax = stack_bmax[u32(sp)];
    let nt0 = stack_t0[u32(sp)];
    let nt1 = stack_t1[u32(sp)];
    let depth = stack_d[u32(sp)];
    let ax = stack_ax[u32(sp)];
    let sg = stack_sg[u32(sp)];
    sp = sp - 1;

    if (nt1 < nt0) { continue; }

    if (depth == CHUNK_DEPTH) {
      let mat = nodes[base + n];
      if (mat != 0u) {
        let t_hit = max(nt0, 0.0);
        let nn = normal_from_axis(ax, sg);
        return Hit(true, t_hit, nn, mat);
      }
      continue;
    }

    let mask = nodes[base + n] & 0xffu;
    if (mask == 0u) { continue; }

    let half = 0.5 * (nbmax - nbmin);

    var ct0 : array<f32, 8>;
    var ct1 : array<f32, 8>;
    var cax : array<f32, 8>;
    var csg : array<f32, 8>;
    var cid : array<u32, 8>;
    var cb0 : array<vec3<f32>, 8>;
    var cb1 : array<vec3<f32>, 8>;
    var count: u32 = 0u;

    for (var c: u32 = 0u; c < 8u; c = c + 1u) {
      if ((mask & (1u << c)) == 0u) { continue; }

      let xb = (c >> 2u) & 1u;
      let yb = (c >> 1u) & 1u;
      let zb =  c        & 1u;

      let cmin = nbmin + vec3<f32>(f32(xb), f32(yb), f32(zb)) * half;
      let cmax = cmin + half;

      let itc = intersect_aabb(ro, rd, cmin, cmax);
      var tca = itc.x;
      var tcb = itc.y;

      tca = max(tca, nt0);
      tcb = min(tcb, nt1);

      if (tcb < tca) { continue; }

      cid[count] = c;
      ct0[count] = tca;
      ct1[count] = tcb;
      cax[count] = itc.z;
      csg[count] = itc.w;
      cb0[count] = cmin;
      cb1[count] = cmax;
      count = count + 1u;
      if (count == 8u) { break; }
    }

    for (var i: u32 = 0u; i + 1u < count; i = i + 1u) {
      var best_i = i;
      var best_t = ct0[i];
      for (var j: u32 = i + 1u; j < count; j = j + 1u) {
        if (ct0[j] < best_t) { best_t = ct0[j]; best_i = j; }
      }
      if (best_i != i) {
        let t0s = ct0[i]; ct0[i] = ct0[best_i]; ct0[best_i] = t0s;
        let t1s = ct1[i]; ct1[i] = ct1[best_i]; ct1[best_i] = t1s;
        let axs = cax[i]; cax[i] = cax[best_i]; cax[best_i] = axs;
        let sgs = csg[i]; csg[i] = csg[best_i]; csg[best_i] = sgs;
        let ids = cid[i]; cid[i] = cid[best_i]; cid[best_i] = ids;
        let b0s = cb0[i]; cb0[i] = cb0[best_i]; cb0[best_i] = b0s;
        let b1s = cb1[i]; cb1[i] = cb1[best_i]; cb1[best_i] = b1s;
      }
    }

    for (var k: i32 = i32(count) - 1; k >= 0; k = k - 1) {
      if (sp + 1 >= i32(MAX_STACK)) { break; }
      sp = sp + 1;

      let c = cid[u32(k)];
      let cn = child_index(n, c);

      stack_node[u32(sp)] = cn;
      stack_bmin[u32(sp)] = cb0[u32(k)];
      stack_bmax[u32(sp)] = cb1[u32(k)];
      stack_t0[u32(sp)]   = ct0[u32(k)];
      stack_t1[u32(sp)]   = ct1[u32(k)];
      stack_d[u32(sp)]    = depth + 1u;
      stack_ax[u32(sp)]   = cax[u32(k)];
      stack_sg[u32(sp)]   = csg[u32(k)];
    }
  }

  return Hit(false, 0.0, vec3<f32>(0.0), 0u);
}

fn dda_init(ro: vec3<f32>, rd: vec3<f32>, cs: f32) -> vec4<f32> {
  let c = vec3<i32>(i32(floor(ro.x / cs)), i32(floor(ro.y / cs)), i32(floor(ro.z / cs)));

  let bx = f32(c.x) * cs + select(0.0, cs, rd.x > 0.0);
  let by = f32(c.y) * cs + select(0.0, cs, rd.y > 0.0);
  let bz = f32(c.z) * cs + select(0.0, cs, rd.z > 0.0);

  let tmx = select(BIG_F32, (bx - ro.x) / rd.x, abs(rd.x) > 1e-8);
  let tmy = select(BIG_F32, (by - ro.y) / rd.y, abs(rd.y) > 1e-8);
  let tmz = select(BIG_F32, (bz - ro.z) / rd.z, abs(rd.z) > 1e-8);

  return vec4<f32>(tmx, tmy, tmz, 0.0);
}

fn dda_delta(rd: vec3<f32>, cs: f32) -> vec3<f32> {
  return vec3<f32>(
    select(BIG_F32, cs / abs(rd.x), abs(rd.x) > 1e-8),
    select(BIG_F32, cs / abs(rd.y), abs(rd.y) > 1e-8),
    select(BIG_F32, cs / abs(rd.z), abs(rd.z) > 1e-8)
  );
}

fn dda_step_axis(tmax: vec3<f32>) -> u32 {
  var ax: u32 = 0u;
  var best = tmax.x;
  if (tmax.y < best) { best = tmax.y; ax = 1u; }
  if (tmax.z < best) { ax = 2u; }
  return ax;
}

fn step_chunk_coord(cc: ptr<function, vec3<i32>>, rd: vec3<f32>, axis: u32) {
  if (axis == 0u) { (*cc).x = (*cc).x + select(-1, 1, rd.x > 0.0); }
  else if (axis == 1u) { (*cc).y = (*cc).y + select(-1, 1, rd.y > 0.0); }
  else { (*cc).z = (*cc).z + select(-1, 1, rd.z > 0.0); }
}

fn trace_world_svo(ro: vec3<f32>, rd: vec3<f32>, tmax_world: f32) -> Hit {
  let cs = chunk_size_m();
  let tmax_world2 = min(tmax_world, TRACE_MAX_DIST_DEFAULT);

  var cc = vec3<i32>(i32(floor(ro.x / cs)), i32(floor(ro.y / cs)), i32(floor(ro.z / cs)));

  var t = 0.0;
  let tinit = dda_init(ro, rd, cs);
  var tmaxv = tinit.xyz;
  let tdel  = dda_delta(rd, cs);

  for (var step: u32 = 0u; step < MAX_CHUNK_STEPS; step = step + 1u) {
    if (t > tmax_world2) { break; }

    let axis = dda_step_axis(tmaxv);
    let t_next = min(min(tmaxv.x, tmaxv.y), tmaxv.z);

    let slot = find_slot_for_chunk(cc);
    if (slot != 0xffffffffu) {
      let h = trace_chunk_svo(ro, rd, slot, cc, t, min(t_next, tmax_world2));
      if (h.hit && h.t <= tmax_world2) { return h; }
    }

    t = t_next + 1e-6;

    if (axis == 0u) { tmaxv.x = tmaxv.x + tdel.x; }
    else if (axis == 1u) { tmaxv.y = tmaxv.y + tdel.y; }
    else { tmaxv.z = tmaxv.z + tdel.z; }

    step_chunk_coord(&cc, rd, axis);
  }

  return Hit(false, 0.0, vec3<f32>(0.0), 0u);
}
