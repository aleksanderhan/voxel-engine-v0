// src/shaders/ray/chunk_trace.wgsl
// --------------------------------
//
// Implements:
// - Hybrid chunk tracing
// - Scene chunk-grid DDA (Digital Differential Analyzer) stepping
// - NEW: SVO (Sparse Voxel Octree) ray traversal inside a chunk using
//        midplane stepping (≤4 children per internal node), no 8-child slab tests,
//        no sorting. (This is 3A+3B.)

//// --------------------------------------------------------------------------
//// HitGeom + hybrid chunk tracing
//// --------------------------------------------------------------------------

struct HitGeom {
  hit      : u32,
  mat      : u32,
  _pad0    : u32,
  _pad1    : u32,

  t        : f32,
  _pad2    : vec3<f32>,

  n        : vec3<f32>,
  _pad3    : f32,

  root_bmin  : vec3<f32>,
  root_size  : f32,
  node_base  : u32,
  macro_base : u32,
  _pad4      : vec2<u32>,
};

fn miss_hitgeom() -> HitGeom {
  var h : HitGeom;
  h.hit = 0u;
  h.mat = MAT_AIR;
  h.t   = BIG_F32;
  h.n   = vec3<f32>(0.0);
  h.root_bmin = vec3<f32>(0.0);
  h.root_size = 0.0;
  h.node_base = 0u;
  h.macro_base = INVALID_U32;
  return h;
}

//// --------------------------------------------------------------------------
//// NEW: Ray → SVO leaf iterator (fast child expansion: 3A+3B)
//// --------------------------------------------------------------------------

// Requirements assumed from your shared includes:
// - struct Node { child_base: u32, child_mask: u32, material: u32, ... }
// - var<storage, read> nodes : array<Node>;
// - const LEAF_U32 : u32;
// - const MAT_AIR : u32;
// - const BIG_F32 : f32;
// - const EPS_INV : f32;
// - cube_slab_inv(ro, inv, bmin, size) -> CubeSlab { t_enter, t_exit, ... }

struct StackEntry {
  kind   : u32,        // 0 = node, 1 = air leaf (virtual missing child)
  idx    : u32,        // node index when kind=0
  mat    : u32,        // valid when kind=1
  _pad0  : u32,

  bmin   : vec3<f32>,
  size   : f32,

  t0     : f32,
  t1     : f32,
  _pad1  : vec2<f32>,
};

fn push_entry(stack: ptr<function, array<StackEntry, 64>>, sp: ptr<function, u32>, e: StackEntry) {
  if (*sp < 64u) {
    (*stack)[*sp] = e;
    *sp = *sp + 1u;
  }
}

fn pop_entry(stack: ptr<function, array<StackEntry, 64>>, sp: ptr<function, u32>) -> StackEntry {
  *sp = *sp - 1u;
  return (*stack)[*sp];
}

// rank = popcount(mask & ((1<<ci)-1))
fn rank_child(mask: u32, ci: u32) -> u32 {
  let before = (1u << ci) - 1u;
  return countOneBits(mask & before);
}

// Which half (bit 0/1) do we enter first on this axis?
fn axis_first_bit(rd_comp: f32, ro_comp: f32, t0: f32, mid_comp: f32) -> u32 {
  // Parallel-ish: decide from entry position
  if (abs(rd_comp) < EPS_INV) {
    let x = ro_comp + rd_comp * t0;
    return select(0u, 1u, x > mid_comp);
  }
  // If moving negative, we hit the "upper half" first (bit=1); else lower (bit=0)
  return select(0u, 1u, rd_comp < 0.0);
}

// Return the t-interval inside chosen half-space on one axis, clipped to [t0,t1].
// bit=0 => lower half, bit=1 => upper half.
fn axis_interval_for_bit(
  bit: u32,
  rd_comp: f32,
  ro_comp: f32,
  t0: f32,
  t1: f32,
  tm_raw: f32,      // (mid - ro) * inv, NOT clipped
  mid_comp: f32
) -> vec2<f32> {
  // Parallel-ish: constant side
  if (abs(rd_comp) < EPS_INV) {
    let x = ro_comp; // rd ~ 0
    let in_upper = x > mid_comp;
    let want_upper = (bit != 0u);
    if (in_upper == want_upper) { return vec2<f32>(t0, t1); }
    return vec2<f32>(BIG_F32, -BIG_F32);
  }

  // If the midplane crossing is outside this node interval, the ray stays in one half
  if (tm_raw <= t0 || tm_raw >= t1) {
    let x0 = ro_comp + rd_comp * t0;
    let in_upper = x0 > mid_comp;
    let want_upper = (bit != 0u);
    if (in_upper == want_upper) { return vec2<f32>(t0, t1); }
    return vec2<f32>(BIG_F32, -BIG_F32);
  }

  // Otherwise it crosses inside [t0,t1]; decide near vs far half
  // If rd < 0, upper half is near interval [t0,tm]; if rd > 0, lower half is near [t0,tm]
  let near_is_upper = (rd_comp < 0.0);
  let want_upper    = (bit != 0u);
  let want_near     = (want_upper == near_is_upper);

  // near => [t0,tm], far => [tm,t1]
  return select(vec2<f32>(tm_raw, t1), vec2<f32>(t0, tm_raw), want_near);
}


// Expand one internal node into ≤4 children, in correct near→far order, without sorting.
// Pushes far→near to the stack so pop() yields near→far.
fn push_children_along_ray(
  ro: vec3<f32>,
  rd: vec3<f32>,
  inv: vec3<f32>,
  node_base: u32,
  e: StackEntry,
  n: Node,
  stack: ptr<function, array<StackEntry, 64>>,
  sp: ptr<function, u32>
) {
  let half = e.size * 0.5;
  let mid  = e.bmin + vec3<f32>(half);

  // Midplane crossing times (RAW, unclipped) used for interval building
  let tmx_raw: f32 = (mid.x - ro.x) * inv.x;
  let tmy_raw: f32 = (mid.y - ro.y) * inv.y;
  let tmz_raw: f32 = (mid.z - ro.z) * inv.z;

  // Midplane crossing times used ONLY for ordering (must be inside (e.t0, e.t1), else BIG)
  var tmx: f32 = tmx_raw;
  var tmy: f32 = tmy_raw;
  var tmz: f32 = tmz_raw;

  if (abs(rd.x) < EPS_INV || tmx <= e.t0 || tmx >= e.t1) { tmx = BIG_F32; }
  if (abs(rd.y) < EPS_INV || tmy <= e.t0 || tmy >= e.t1) { tmy = BIG_F32; }
  if (abs(rd.z) < EPS_INV || tmz <= e.t0 || tmz >= e.t1) { tmz = BIG_F32; }

  // Initial child bits at node entry
  let bx0 = axis_first_bit(rd.x, ro.x, e.t0, mid.x);
  let by0 = axis_first_bit(rd.y, ro.y, e.t0, mid.y);
  let bz0 = axis_first_bit(rd.z, ro.z, e.t0, mid.z);

  var ci: u32 = bx0 | (by0 << 1u) | (bz0 << 2u);

  var gen: array<StackEntry, 4>;
  var gcount: u32 = 0u;

  // Working copies of crossing times (disable after flipping)
  var cxm: f32 = tmx;
  var cym: f32 = tmy;
  var czm: f32 = tmz;

  for (var k: u32 = 0u; k < 4u; k = k + 1u) {
    let hx = (ci & 1u);
    let hy = (ci >> 1u) & 1u;
    let hz = (ci >> 2u) & 1u;

    let cbmin = e.bmin + vec3<f32>(
      select(0.0, half, hx != 0u),
      select(0.0, half, hy != 0u),
      select(0.0, half, hz != 0u)
    );

    // Child interval = intersection of per-axis half-intervals
    // IMPORTANT: use *tm_raw*, not recomputed values
    let ix = axis_interval_for_bit(hx, rd.x, ro.x, e.t0, e.t1, tmx_raw, mid.x);
    let iy = axis_interval_for_bit(hy, rd.y, ro.y, e.t0, e.t1, tmy_raw, mid.y);
    let iz = axis_interval_for_bit(hz, rd.z, ro.z, e.t0, e.t1, tmz_raw, mid.z);

    let t0 = max(e.t0, max(ix.x, max(iy.x, iz.x)));
    let t1 = min(e.t1, min(ix.y, min(iy.y, iz.y)));

    if (t1 > t0) {
      let bit = 1u << ci;

      if ((n.child_mask & bit) == 0u) {
        gen[gcount] = StackEntry(1u, 0u, MAT_AIR, 0u, cbmin, half, t0, t1, vec2<f32>(0.0));
      } else {
        let r = rank_child(n.child_mask, ci);
        let child_idx = node_base + (n.child_base + r);
        gen[gcount] = StackEntry(0u, child_idx, 0u, 0u, cbmin, half, t0, t1, vec2<f32>(0.0));
      }

      gcount = gcount + 1u;
      if (gcount == 4u) { break; }
    }

    // Decide which midplane we cross next (smallest remaining)
    if (cxm < cym) {
      if (cxm < czm) { ci = ci ^ 1u;        cxm = BIG_F32; }
      else           { ci = ci ^ (1u<<2u);  czm = BIG_F32; }
    } else {
      if (cym < czm) { ci = ci ^ (1u<<1u);  cym = BIG_F32; }
      else           { ci = ci ^ (1u<<2u);  czm = BIG_F32; }
    }

    if (cxm == BIG_F32 && cym == BIG_F32 && czm == BIG_F32) { break; }
  }

  // Push far→near so stack pops near→far
  for (var i: i32 = i32(gcount) - 1; i >= 0; i = i - 1) {
    push_entry(stack, sp, gen[u32(i)]);
  }
}


struct LeafHit {
  hit  : bool,
  bmin : vec3<f32>,
  size : f32,
  mat  : u32,
  t0   : f32,
  t1   : f32,
};

fn next_leaf_in_chunk(
  ro: vec3<f32>,
  rd: vec3<f32>,
  inv: vec3<f32>,
  node_base: u32,
  macro_base: u32,   // kept for later; not used here
  min_leaf: f32,
  stack: ptr<function, array<StackEntry, 64>>,
  sp: ptr<function, u32>
) -> LeafHit {
  loop {
    if (*sp == 0u) { break; }

    let e = pop_entry(stack, sp);

    // virtual air leaf
    if (e.kind == 1u) {
      return LeafHit(true, e.bmin, e.size, e.mat, e.t0, e.t1);
    }

    let n = nodes[e.idx];

    // real leaf
    if (n.child_base == LEAF_U32) {
      return LeafHit(true, e.bmin, e.size, n.material, e.t0, e.t1);
    }

    // reached minimum leaf size: treat as air
    if (e.size <= min_leaf) {
      return LeafHit(true, e.bmin, e.size, MAT_AIR, e.t0, e.t1);
    }

    // 3A+3B: expand children along ray (≤4), ordered, no sort, no 8 slabs
    push_children_along_ray(ro, rd, inv, node_base, e, n, stack, sp);
  }

  return LeafHit(false, vec3<f32>(0.0), 0.0, MAT_AIR, BIG_F32, BIG_F32);
}

//// --------------------------------------------------------------------------
//// Chunk tracing using the iterator
//// --------------------------------------------------------------------------

fn trace_chunk_hybrid_interval(
  ro: vec3<f32>,
  rd: vec3<f32>,
  ch: ChunkMeta,
  t_enter: f32,
  t_exit: f32
) -> HitGeom {
  let vs = cam.voxel_params.x;

  let root_bmin_vox = vec3<f32>(f32(ch.origin.x), f32(ch.origin.y), f32(ch.origin.z));
  let root_bmin = root_bmin_vox * vs;
  let root_size = f32(cam.chunk_size) * vs;

  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));
  let min_leaf = vs;

  // Clip to chunk root AABB once
  let root_slab = cube_slab_inv(ro, inv, root_bmin, root_size);
  var rt0 = max(root_slab.t_enter, t_enter);
  var rt1 = min(root_slab.t_exit,  t_exit);
  if (rt1 <= rt0) { return miss_hitgeom(); }

  // Stack init (root node index is ch.node_base)
  var stack: array<StackEntry, 64>;
  var sp: u32 = 0u;
  push_entry(&stack, &sp,
    StackEntry(0u, ch.node_base, 0u, 0u, root_bmin, root_size, rt0, rt1, vec2<f32>(0.0))
  );

  let eps_step = 1e-4 * vs;
  let chunk_size_i = i32(cam.chunk_size);

  loop {
    let leaf = next_leaf_in_chunk(
      ro, rd, inv,
      ch.node_base,
      ch.macro_base,
      min_leaf,
      &stack, &sp
    );
    if (!leaf.hit) { break; }

    // Local “within-leaf” working t (for probes). Leaf stepping itself is via iterator.
    let tcur = max(leaf.t0, 0.0) + eps_step;
    if (tcur > leaf.t1) { continue; }

    let p  = ro + tcur * rd;
    let pq = p + rd * (1e-4 * vs); // keep your original biasing

    // AIR: probe grass slab just below when in blade layer
    if (leaf.mat == MAT_AIR) {
      // NOTE: your original code used q.bmin/q.size from query_leaf_at().
      // Here, leaf.bmin/leaf.size are the leaf cube itself.
      let layer_h = GRASS_LAYER_HEIGHT_VOX * vs;
      let bias = 0.05 * vs;

      var local = (p - root_bmin) - rd * bias;
      local = clamp(local, vec3<f32>(0.0), vec3<f32>(root_size - 1e-6));

      var ix = i32(floor(local.x / vs));
      var iz = i32(floor(local.z / vs));
      ix = clamp(ix, 0, chunk_size_i - 1);
      iz = clamp(iz, 0, chunk_size_i - 1);

      var iy = i32(floor((local.y - 1e-4 * vs) / vs)) - 1;
      iy = clamp(iy, 0, chunk_size_i - 1);

      let cell_bmin = root_bmin + vec3<f32>(f32(ix), f32(iy), f32(iz)) * vs;

      let y_top = cell_bmin.y + vs;
      if (p.y >= y_top - 1e-4 * vs && p.y <= y_top + layer_h + 1e-4 * vs) {
        let c = cell_bmin + vec3<f32>(0.5 * vs, 0.5 * vs, 0.5 * vs);
        if (is_grass(c, root_bmin, root_size, ch.node_base, ch.macro_base)) {
          let cell_id_vox = vec3<f32>(
            f32(ch.origin.x + ix),
            f32(ch.origin.y + iy),
            f32(ch.origin.z + iz)
          );

          // Use leaf exit as probe max boundary (instead of recomputing exit_time)
          let t_leave_air = leaf.t1;
          let tmax_probe  = min(min(t_leave_air, t_exit), tcur + leaf.size + 2.0 * vs);

          let time_s   = cam.voxel_params.y;
          let strength = cam.voxel_params.z;

          let gh = try_grass_slab_hit(ro, rd, tcur, tmax_probe, cell_bmin, cell_id_vox, vs, time_s, strength);
          if (gh.hit) {
            var out = miss_hitgeom();
            out.hit = 1u;
            out.t   = gh.t;
            out.mat = MAT_GRASS;
            out.n   = gh.n;
            out.root_bmin = root_bmin;
            out.root_size = root_size;
            out.node_base = ch.node_base;
            out.macro_base = ch.macro_base;
            return out;
          }
        }
      }

      // No manual stepping; iterator continues to next leaf
      continue;
    }

    // Leaves: displaced cube hit
    if (leaf.mat == MAT_LEAF) {
      let time_s   = cam.voxel_params.y;
      let strength = cam.voxel_params.z;

      let h2 = leaf_displaced_cube_hit(ro, rd, leaf.bmin, leaf.size, time_s, strength, t_enter, t_exit);

      if (h2.hit) {
        var out = miss_hitgeom();
        out.hit = 1u;
        out.t   = h2.t;
        out.mat = MAT_LEAF;
        out.n   = h2.n;
        out.root_bmin = root_bmin;
        out.root_size = root_size;
        out.node_base = ch.node_base;
        out.macro_base = ch.macro_base;
        return out;
      }

      continue;
    }

    // Solid: AABB (Axis-Aligned Bounding Box) face hit on this leaf cube
    let slab = cube_slab_inv(ro, inv, leaf.bmin, leaf.size);
    let bh = cube_hit_normal_from_slab(rd, slab, t_enter, t_exit);

    if (bh.hit) {
      // If grass solid, try blades above the voxel under this hit (from above)
      if (leaf.mat == MAT_GRASS) {
        let time_s   = cam.voxel_params.y;
        let strength = cam.voxel_params.z;

        let hp = ro + bh.t * rd;

        let cell = pick_grass_cell_in_chunk(
          hp, rd,
          root_bmin,
          vec3<i32>(ch.origin.x, ch.origin.y, ch.origin.z),
          vs,
          chunk_size_i
        );

        let tmax_probe = min(bh.t, t_exit);
        let gh = try_grass_slab_hit(ro, rd, t_enter, tmax_probe, cell.bmin_m, cell.id_vox, vs, time_s, strength);
        if (gh.hit) {
          var out = miss_hitgeom();
          out.hit = 1u;
          out.t   = gh.t;
          out.mat = MAT_GRASS;
          out.n   = gh.n;
          out.root_bmin = root_bmin;
          out.root_size = root_size;
          out.node_base = ch.node_base;
          out.macro_base = ch.macro_base;
          return out;
        }
      }

      var out = miss_hitgeom();
      out.hit = 1u;
      out.t   = bh.t;
      out.mat = leaf.mat;
      out.n   = bh.n;
      out.root_bmin = root_bmin;
      out.root_size = root_size;
      out.node_base = ch.node_base;
      out.macro_base = ch.macro_base;
      return out;
    }

    // Miss: iterator continues
  }

  return miss_hitgeom();
}

//// --------------------------------------------------------------------------
//// Scene voxel tracing over streamed chunk grid (for main_primary)
//// --------------------------------------------------------------------------

struct VoxTraceResult {
  in_grid : bool,
  best    : HitGeom,
  t_exit  : f32,
};

// DDA (Digital Differential Analyzer) = grid-stepping along the ray
fn trace_scene_voxels(ro: vec3<f32>, rd: vec3<f32>) -> VoxTraceResult {
  // Fast out if no chunks
  if (cam.chunk_count == 0u) {
    return VoxTraceResult(false, miss_hitgeom(), 0.0);
  }

  let voxel_size   = cam.voxel_params.x;
  let chunk_size_m = f32(cam.chunk_size) * voxel_size;

  let go = cam.grid_origin_chunk;
  let gd = cam.grid_dims;

  // Grid bounds in meters
  let grid_bmin = vec3<f32>(f32(go.x), f32(go.y), f32(go.z)) * chunk_size_m;
  let grid_bmax = grid_bmin + vec3<f32>(f32(gd.x), f32(gd.y), f32(gd.z)) * chunk_size_m;

  // Ray vs grid AABB
  let rtg = intersect_aabb(ro, rd, grid_bmin, grid_bmax);
  var t_enter = max(rtg.x, 0.0);
  let t_exit  = rtg.y;

  if (t_exit < t_enter) {
    return VoxTraceResult(false, miss_hitgeom(), 0.0);
  }

  // Nudge inside
  let nudge_p = PRIMARY_NUDGE_VOXEL_FRAC * voxel_size;
  let start_t = t_enter + nudge_p;
  let p0      = ro + start_t * rd;

  // World chunk coords at start
  let c = chunk_coord_from_pos(p0, chunk_size_m);

  // Local grid coords (0..dims-1) and running linear index
  let nx: i32 = i32(gd.x);
  let ny: i32 = i32(gd.y);
  let nz: i32 = i32(gd.z);

  var lcx: i32 = c.x - go.x;
  var lcy: i32 = c.y - go.y;
  var lcz: i32 = c.z - go.z;

  // If start is outside (should be rare if AABB hit), treat as out-of-grid.
  if (lcx < 0 || lcy < 0 || lcz < 0 || lcx >= nx || lcy >= ny || lcz >= nz) {
    return VoxTraceResult(false, miss_hitgeom(), 0.0);
  }

  let stride_x: i32 = 1;
  let stride_y: i32 = nx;
  let stride_z: i32 = nx * ny;

  var idx_i: i32 = (lcz * ny + lcy) * nx + lcx;

  // DDA setup (in ray-param space, relative to p0)
  var t_local: f32 = 0.0;
  let t_exit_local = max(t_exit - start_t, 0.0);

  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  let step_x: i32 = select(-1, 1, rd.x > 0.0);
  let step_y: i32 = select(-1, 1, rd.y > 0.0);
  let step_z: i32 = select(-1, 1, rd.z > 0.0);

  // Next chunk boundary in meters (world chunk coords from c)
  let bx = select(f32(c.x) * chunk_size_m, f32(c.x + 1) * chunk_size_m, rd.x > 0.0);
  let by = select(f32(c.y) * chunk_size_m, f32(c.y + 1) * chunk_size_m, rd.y > 0.0);
  let bz = select(f32(c.z) * chunk_size_m, f32(c.z + 1) * chunk_size_m, rd.z > 0.0);

  var tMaxX: f32 = (bx - p0.x) * inv.x;
  var tMaxY: f32 = (by - p0.y) * inv.y;
  var tMaxZ: f32 = (bz - p0.z) * inv.z;

  let tDeltaX: f32 = abs(chunk_size_m * inv.x);
  let tDeltaY: f32 = abs(chunk_size_m * inv.y);
  let tDeltaZ: f32 = abs(chunk_size_m * inv.z);

  if (abs(rd.x) < EPS_INV) { tMaxX = BIG_F32; }
  if (abs(rd.y) < EPS_INV) { tMaxY = BIG_F32; }
  if (abs(rd.z) < EPS_INV) { tMaxZ = BIG_F32; }

  // Linear-index step deltas
  let didx_x: i32 = select(-stride_x, stride_x, rd.x > 0.0);
  let didx_y: i32 = select(-stride_y, stride_y, rd.y > 0.0);
  let didx_z: i32 = select(-stride_z, stride_z, rd.z > 0.0);

  var best = miss_hitgeom();

  // Conservative cap
  let max_chunk_steps = min((gd.x + gd.y + gd.z) * 6u + 8u, 1024u);

  for (var s: u32 = 0u; s < max_chunk_steps; s = s + 1u) {
    if (t_local > t_exit_local) { break; }

    let tNextLocal = min(tMaxX, min(tMaxY, tMaxZ));
    if (best.hit != 0u && (start_t + tNextLocal) >= best.t) { break; }

    // Slot lookup via running linear index
    let slot = chunk_grid[u32(idx_i)];
    if (slot != INVALID_U32 && slot < cam.chunk_count) {
      let ch = chunks[slot];

      let cell_enter = start_t + t_local;
      let cell_exit  = start_t + min(tNextLocal, t_exit_local);

      let h = trace_chunk_hybrid_interval(ro, rd, ch, cell_enter, cell_exit);
      if (h.hit != 0u && h.t < best.t) { best = h; }
    }

    // Advance DDA + update (lc*, idx_i)
    if (tMaxX < tMaxY) {
      if (tMaxX < tMaxZ) {
        lcx += step_x; idx_i += didx_x;
        t_local = tMaxX; tMaxX += tDeltaX;
      } else {
        lcz += step_z; idx_i += didx_z;
        t_local = tMaxZ; tMaxZ += tDeltaZ;
      }
    } else {
      if (tMaxY < tMaxZ) {
        lcy += step_y; idx_i += didx_y;
        t_local = tMaxY; tMaxY += tDeltaY;
      } else {
        lcz += step_z; idx_i += didx_z;
        t_local = tMaxZ; tMaxZ += tDeltaZ;
      }
    }

    // Local bounds (no origin adds)
    if (lcx < 0 || lcy < 0 || lcz < 0 || lcx >= nx || lcy >= ny || lcz >= nz) { break; }
  }

  return VoxTraceResult(true, best, t_exit);
}
