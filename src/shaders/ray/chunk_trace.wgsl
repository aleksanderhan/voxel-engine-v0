// src/shaders/ray/chunk_trace.wgsl
// --------------------------------------------------------------------------
// HitGeom + chunk tracing (rope traversal + macro occupancy)
//
// Cleaned:
// - Removed cached-path SVO query (unused now).
// - Removed dead helpers (node_* key decode, global rope_next, etc).
// - Kept only what trace_scene_voxels -> trace_chunk_rope_interval needs.
// - Kept macro early-out + slab stepping + grass-on-solid-face probing.
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// HitGeom
// --------------------------------------------------------------------------

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

// Strict half-open: [bmin, bmax)
fn point_in_cube(p: vec3<f32>, bmin: vec3<f32>, size: f32) -> bool {
  let bmax = bmin + vec3<f32>(size);
  return (p.x >= bmin.x && p.x < bmax.x) &&
         (p.y >= bmin.y && p.y < bmax.y) &&
         (p.z >= bmin.z && p.z < bmax.z);
}

// --------------------------------------------------------------------------
// Macro occupancy early-out (returns a MAT_AIR cube if empty, else size==0)
// --------------------------------------------------------------------------

struct MacroQuery {
  bmin : vec3<f32>,
  size : f32,
  mat  : u32,
};

fn macro_leaf_or_invalid(
  p_in: vec3<f32>,
  root_bmin: vec3<f32>,
  root_size: f32,
  macro_base: u32
) -> MacroQuery {
  if (macro_base == INVALID_U32) {
    return MacroQuery(vec3<f32>(0.0), 0.0, 0u);
  }

  let cell = macro_cell_size(root_size);
  let lp = p_in - root_bmin;

  let mx = clamp(u32(floor(lp.x / cell)), 0u, MACRO_DIM - 1u);
  let my = clamp(u32(floor(lp.y / cell)), 0u, MACRO_DIM - 1u);
  let mz = clamp(u32(floor(lp.z / cell)), 0u, MACRO_DIM - 1u);

  let bit = macro_bit_index(mx, my, mz);

  if (!macro_test(macro_base, bit)) {
    let macro_bmin = root_bmin + vec3<f32>(
      f32(mx) * cell,
      f32(my) * cell,
      f32(mz) * cell
    );
    return MacroQuery(macro_bmin, cell, MAT_AIR);
  }

  return MacroQuery(vec3<f32>(0.0), 0.0, 0u);
}

// --------------------------------------------------------------------------
// Key decoding -> cube
// (matches the key encoding you already use)
// --------------------------------------------------------------------------

fn node_level(key: u32) -> u32 { return (key >> 18u) & 7u; }
fn node_x(key: u32) -> u32 { return (key      ) & 63u; }
fn node_y(key: u32) -> u32 { return (key >>  6u) & 63u; }
fn node_z(key: u32) -> u32 { return (key >> 12u) & 63u; }

fn node_cube_from_key(root_bmin: vec3<f32>, root_size: f32, key: u32) -> vec4<f32> {
  let lvl = node_level(key);
  let s = root_size / f32(1u << lvl);
  let bmin = root_bmin + vec3<f32>(f32(node_x(key)), f32(node_y(key)), f32(node_z(key))) * s;
  return vec4<f32>(bmin, s);
}

// --------------------------------------------------------------------------
// Rope helpers (LOCAL indices)
// --------------------------------------------------------------------------

fn node_at(node_base: u32, local_idx: u32) -> Node {
  return nodes[node_base + local_idx];
}

fn ropes_at(node_base: u32, local_idx: u32) -> NodeRopes {
  return node_ropes[node_base + local_idx];
}

fn rope_next_local(node_base: u32, local_idx: u32, face: u32) -> u32 {
  let r = ropes_at(node_base, local_idx);
  if (face == 0u) { return r.px; }
  if (face == 1u) { return r.nx; }
  if (face == 2u) { return r.py; }
  if (face == 3u) { return r.ny; }
  if (face == 4u) { return r.pz; }
  return r.nz;
}

fn exit_face_from_slab(rd: vec3<f32>, slab: CubeSlab) -> u32 {
  // 0=+X,1=-X,2=+Y,3=-Y,4=+Z,5=-Z
  let tx = slab.tmaxv.x;
  let ty = slab.tmaxv.y;
  let tz = slab.tmaxv.z;

  var axis: u32 = 0u;
  var best: f32 = tx;
  if (ty < best) { best = ty; axis = 1u; }
  if (tz < best) { best = tz; axis = 2u; }

  if (axis == 0u) { return select(1u, 0u, rd.x > 0.0); }
  if (axis == 1u) { return select(3u, 2u, rd.y > 0.0); }
  return select(5u, 4u, rd.z > 0.0);
}

// --------------------------------------------------------------------------
// Sparse descend from an arbitrary node cube (start_idx is LOCAL)
// Returns a leaf cube; if the leaf is missing-child AIR, has_node=false.
// --------------------------------------------------------------------------

struct LeafState {
  has_node  : bool,
  idx_local : u32,       // valid only if has_node
  bmin      : vec3<f32>,
  size      : f32,
  mat       : u32,
};

fn descend_leaf_sparse(
  p_in      : vec3<f32>,
  node_base : u32,
  start_idx : u32,
  start_bmin: vec3<f32>,
  start_size: f32
) -> LeafState {
  var idx  : u32 = start_idx;
  var bmin : vec3<f32> = start_bmin;
  var size : f32 = start_size;

  let max_d = chunk_max_depth();

  for (var d: u32 = 0u; d < 32u; d = d + 1u) {
    let n = node_at(node_base, idx);

    if (n.child_base == LEAF_U32) {
      return LeafState(true, idx, bmin, size, n.material);
    }

    if (d >= max_d) {
      return LeafState(true, idx, bmin, size, MAT_AIR);
    }

    let half = size * 0.5;
    let mid  = bmin + vec3<f32>(half);
    let e    = 1e-6 * size;

    let hx = select(0u, 1u, p_in.x > mid.x + e);
    let hy = select(0u, 1u, p_in.y > mid.y + e);
    let hz = select(0u, 1u, p_in.z > mid.z + e);
    let ci = hx | (hy << 1u) | (hz << 2u);

    let child_bmin = bmin + vec3<f32>(
      select(0.0, half, hx != 0u),
      select(0.0, half, hy != 0u),
      select(0.0, half, hz != 0u)
    );

    let bit = 1u << ci;
    if ((n.child_mask & bit) == 0u) {
      return LeafState(false, INVALID_U32, child_bmin, half, MAT_AIR);
    }

    let r = child_rank(n.child_mask, ci);
    idx  = n.child_base + r;      // LOCAL packed index
    bmin = child_bmin;
    size = half;
  }

  return LeafState(false, INVALID_U32, start_bmin, start_size, MAT_AIR);
}

// --------------------------------------------------------------------------
// REAL rope traversal: AIR leaf => rope jump => use key to get neighbor cube
// => descend starting from that neighbor node (NOT from root).
// --------------------------------------------------------------------------
fn trace_chunk_rope_interval(
  ro: vec3<f32>,
  rd: vec3<f32>,
  ch: ChunkMeta,
  t_enter: f32,
  t_exit: f32
) -> HitGeom {
  let vs = cam.voxel_params.x;

  let root_bmin_vox = vec3<f32>(f32(ch.origin.x), f32(ch.origin.y), f32(ch.origin.z));
  let root_bmin     = root_bmin_vox * vs;
  let root_size     = f32(cam.chunk_size) * vs;

  let eps_step = 1e-4 * vs;
  var tcur     = max(t_enter, 0.0) + eps_step;

  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  // Only probe grass when we are in small-enough air leaves
  let grass_probe_max_leaf = 2.0 * vs;

  let origin_vox_i = vec3<i32>(ch.origin.x, ch.origin.y, ch.origin.z);
  let time_s       = cam.voxel_params.y;
  let strength     = cam.voxel_params.z;

  var have_leaf: bool = false;
  var leaf: LeafState;

  for (var step_i: u32 = 0u; step_i < cam.max_steps; step_i = step_i + 1u) {
    if (tcur > t_exit) { break; }

    let p  = ro + tcur * rd;
    let pq = p + rd * (1e-4 * vs);

    // ----------------------------------------------------------------------
    // Macro empty => big AIR slab step
    // ----------------------------------------------------------------------
    let m = macro_leaf_or_invalid(pq, root_bmin, root_size, ch.macro_base);
    if (m.size > 0.0) {
      let slabm = cube_slab_inv(ro, inv, m.bmin, m.size);
      tcur = max(slabm.t_exit, tcur) + eps_step;
      have_leaf = false;
      continue;
    }

    // ----------------------------------------------------------------------
    // Ensure leaf contains pq
    // ----------------------------------------------------------------------
    if (!have_leaf || !point_in_cube(pq, leaf.bmin, leaf.size)) {
      leaf = descend_leaf_sparse(pq, ch.node_base, 0u, root_bmin, root_size);
      have_leaf = true;
    }

    let slab    = cube_slab_inv(ro, inv, leaf.bmin, leaf.size);
    let t_leave = slab.t_exit;

    // ----------------------------------------------------------------------
    // AIR: optional air-side grass + rope traversal
    // ----------------------------------------------------------------------
    if (leaf.mat == MAT_AIR) {
      // ------------------------------------------------------------
      // AIR-side grass (column map) — step across the AIR segment in xz
      // ------------------------------------------------------------
      if (leaf.size <= grass_probe_max_leaf) {
        let t0_probe = max(t_enter, tcur - eps_step);
        let t1_probe = min(t_leave, t_exit);

        if (t1_probe >= t0_probe) {
          // Decide how many samples based on projected travel in XZ
          let p0 = ro + rd * t0_probe;
          let p1 = ro + rd * t1_probe;

          let dxz = vec2<f32>(p1.x - p0.x, p1.z - p0.z);
          let dist_xz = length(dxz);

          // about 1 sample per voxel in xz (clamped)
          let n = clamp(u32(ceil(dist_xz / max(vs, 1e-6))), 1u, 16u);

          for (var si: u32 = 0u; si < n; si = si + 1u) {
            let a  = (f32(si) + 0.5) / f32(n);
            let ts = mix(t0_probe, t1_probe, a);
            let ps = ro + rd * ts;

            let lp = ps - root_bmin;
            let lx_u = clamp(u32(floor(lp.x / vs)), 0u, 63u);
            let lz_u = clamp(u32(floor(lp.z / vs)), 0u, 63u);

            let e16 = colinfo_entry_u16(ch.colinfo_base, lx_u, lz_u);
            let ci  = colinfo_decode(e16);

            if (ci.valid && ci.mat == MAT_GRASS) {
              let wx: i32 = origin_vox_i.x + i32(lx_u);
              let wy: i32 = origin_vox_i.y + i32(ci.y_vox);
              let wz: i32 = origin_vox_i.z + i32(lz_u);

              let bmin_m = vec3<f32>(f32(wx), f32(wy), f32(wz)) * vs;
              let id_vox = vec3<f32>(f32(wx), f32(wy), f32(wz));

              let gh = try_grass_slab_hit(
                ro, rd,
                t0_probe, t1_probe,
                bmin_m, id_vox,
                vs, time_s, strength
              );

              if (gh.hit) {
                var outg = miss_hitgeom();
                outg.hit = 1u;
                outg.t   = gh.t;
                outg.mat = MAT_GRASS;
                outg.n   = gh.n;
                outg.root_bmin  = root_bmin;
                outg.root_size  = root_size;
                outg.node_base  = ch.node_base;
                outg.macro_base = ch.macro_base;
                return outg;
              }
            }
          }
        }
      }


      // ---- Rope traversal for AIR
      let face = exit_face_from_slab(rd, slab);
      tcur = max(t_leave, tcur) + eps_step;

      if (!leaf.has_node) {
        have_leaf = false;
        continue;
      }

      let nidx = rope_next_local(ch.node_base, leaf.idx_local, face);
      if (nidx == INVALID_U32) {
        have_leaf = false;
        continue;
      }

      let nk = node_at(ch.node_base, nidx).key;
      let c  = node_cube_from_key(root_bmin, root_size, nk);

      leaf = descend_leaf_sparse(pq, ch.node_base, nidx, c.xyz, c.w);
      have_leaf = true;
      continue;
    }

    // ----------------------------------------------------------------------
    // Leaves: displaced cube hit
    // ----------------------------------------------------------------------
    if (leaf.mat == MAT_LEAF) {
      let h2 = leaf_displaced_cube_hit(
        ro, rd,
        leaf.bmin, leaf.size,
        time_s, strength,
        t_enter, t_exit
      );

      if (h2.hit) {
        var out = miss_hitgeom();
        out.hit = 1u;
        out.t   = h2.t;
        out.mat = MAT_LEAF;
        out.n   = h2.n;
        out.root_bmin  = root_bmin;
        out.root_size  = root_size;
        out.node_base  = ch.node_base;
        out.macro_base = ch.macro_base;
        return out;
      }

      tcur = max(t_leave, tcur) + eps_step;
      have_leaf = false;
      continue;
    }

    // ----------------------------------------------------------------------
    // Solid: face hit (+ optional grass-on-solid-face probe)
    // ----------------------------------------------------------------------
    let bh = cube_hit_normal_from_slab(rd, slab, t_enter, t_exit);
    if (bh.hit) {
      // Optional grass-on-solid-face probe when the solid voxel is grass
      if (leaf.mat == MAT_GRASS) {
        let hp = ro + bh.t * rd;

        let cell = pick_grass_cell_in_chunk(
          hp, rd,
          root_bmin,
          origin_vox_i,
          vs,
          i32(cam.chunk_size)
        );

        let tmax_probe = min(bh.t, t_exit);

        let gh = try_grass_slab_hit(
          ro, rd,
          t_enter, tmax_probe,
          cell.bmin_m, cell.id_vox,
          vs, time_s, strength
        );

        if (gh.hit) {
          var outg = miss_hitgeom();
          outg.hit = 1u;
          outg.t   = gh.t;
          outg.mat = MAT_GRASS;
          outg.n   = gh.n;
          outg.root_bmin  = root_bmin;
          outg.root_size  = root_size;
          outg.node_base  = ch.node_base;
          outg.macro_base = ch.macro_base;
          return outg;
        }
      }

      var out = miss_hitgeom();
      out.hit = 1u;
      out.t   = bh.t;
      out.mat = leaf.mat;
      out.n   = bh.n;
      out.root_bmin  = root_bmin;
      out.root_size  = root_size;
      out.node_base  = ch.node_base;
      out.macro_base = ch.macro_base;
      return out;
    }

    // ----------------------------------------------------------------------
    // Miss solid: step out
    // ----------------------------------------------------------------------
    tcur = max(t_leave, tcur) + eps_step;
    have_leaf = false;
  }

  return miss_hitgeom();
}


// --------------------------------------------------------------------------
// Scene voxel tracing over streamed chunk grid (for main_primary)
// --------------------------------------------------------------------------

struct VoxTraceResult {
  in_grid : bool,
  best    : HitGeom,
  t_exit  : f32,
};

fn trace_scene_voxels(ro: vec3<f32>, rd: vec3<f32>) -> VoxTraceResult {
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

  if (lcx < 0 || lcy < 0 || lcz < 0 || lcx >= nx || lcy >= ny || lcz >= nz) {
    return VoxTraceResult(false, miss_hitgeom(), 0.0);
  }

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

  let didx_x: i32 = select(-1, 1, rd.x > 0.0);
  let didx_y: i32 = select(-stride_y, stride_y, rd.y > 0.0);
  let didx_z: i32 = select(-stride_z, stride_z, rd.z > 0.0);

  var best = miss_hitgeom();

  let max_chunk_steps = min((gd.x + gd.y + gd.z) * 6u + 8u, 1024u);

  for (var s: u32 = 0u; s < max_chunk_steps; s = s + 1u) {
    if (t_local > t_exit_local) { break; }

    let tNextLocal = min(tMaxX, min(tMaxY, tMaxZ));
    if (best.hit != 0u && (start_t + tNextLocal) >= best.t) { break; }

    let slot = chunk_grid[u32(idx_i)];
    if (slot != INVALID_U32 && slot < cam.chunk_count) {
      let ch = chunks[slot];

      let cell_enter = start_t + t_local;
      let cell_exit  = start_t + min(tNextLocal, t_exit_local);

      let h = trace_chunk_rope_interval(ro, rd, ch, cell_enter, cell_exit);
      if (h.hit != 0u && h.t < best.t) { best = h; }
    }

    // Advance DDA
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

    if (lcx < 0 || lcy < 0 || lcz < 0 || lcx >= nx || lcy >= ny || lcz >= nz) { break; }
  }

  return VoxTraceResult(true, best, t_exit);
}

// --------------------------------------------------------------------------
// Column-top map (64x64)
// - Each (x,z) has a packed u16 entry: (mat8<<8) | y8
// - y8=255 means empty column (all air)
// - Two entries per u32 word => 2048 u32 words per chunk
// --------------------------------------------------------------------------

fn colinfo_entry_u16(colinfo_base: u32, lx: u32, lz: u32) -> u32 {
  // idx in [0..4095]
  let idx = lz * 64u + lx;
  let w = idx >> 1u;          // 0..2047
  let word = chunk_colinfo[colinfo_base + w];
  if ((idx & 1u) == 0u) {
    return word & 0xFFFFu;
  }
  return (word >> 16u) & 0xFFFFu;
}

struct ColInfo {
  valid : bool,
  y_vox : u32,   // 0..63
  mat   : u32,   // 0..255 (we only care if == MAT_GRASS)
};

fn colinfo_decode(e16: u32) -> ColInfo {
  let y8   = e16 & 0xFFu;
  let mat8 = (e16 >> 8u) & 0xFFu;
  if (y8 == 255u) {
    return ColInfo(false, 0u, 0u);
  }
  return ColInfo(true, y8, mat8);
}
