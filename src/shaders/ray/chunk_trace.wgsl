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

// --------------------------------------------------------------------------
// Macro cell cacheable query
// --------------------------------------------------------------------------

struct MacroCell {
  bmin    : vec3<f32>,
  size    : f32,
  is_empty: bool,
};

fn macro_cell_query(
  p_in: vec3<f32>,
  root_bmin: vec3<f32>,
  root_size: f32,
  macro_base: u32
) -> MacroCell {
  // No macro data => treat as "not empty" so traversal continues normally.
  if (macro_base == INVALID_U32) {
    return MacroCell(vec3<f32>(0.0), 0.0, false);
  }

  let cell = macro_cell_size(root_size);
  let lp   = p_in - root_bmin;

  let mx = clamp(u32(floor(lp.x / cell)), 0u, MACRO_DIM - 1u);
  let my = clamp(u32(floor(lp.y / cell)), 0u, MACRO_DIM - 1u);
  let mz = clamp(u32(floor(lp.z / cell)), 0u, MACRO_DIM - 1u);

  let macro_bmin = root_bmin + vec3<f32>(
    f32(mx) * cell,
    f32(my) * cell,
    f32(mz) * cell
  );

  let bit = macro_bit_index(mx, my, mz);
  let empty = !macro_test(macro_base, bit);

  return MacroCell(macro_bmin, cell, empty);
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

  // nearest existing ancestor node/cube we can start from (anchor)
  has_anchor   : bool,
  anchor_idx   : u32,
  anchor_bmin  : vec3<f32>,
  anchor_size  : f32,
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
      return LeafState(true, idx, bmin, size, n.material,
                       false, 0u, vec3<f32>(0.0), 0.0);
    }

    if (d >= max_d) {
      return LeafState(true, idx, bmin, size, MAT_AIR,
                       false, 0u, vec3<f32>(0.0), 0.0);
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
      // Missing child => implicit AIR cube, but parent node exists.
      return LeafState(
        false, INVALID_U32, child_bmin, half, MAT_AIR,
        true,  idx,         bmin,      size
      );
    }


    let r = child_rank(n.child_mask, ci);
    idx  = n.child_base + r;      // LOCAL packed index
    bmin = child_bmin;
    size = half;
  }

  return LeafState(
    false, INVALID_U32, start_bmin, start_size, MAT_AIR,
    false, INVALID_U32, vec3<f32>(0.0), 0.0
  );

}

// --------------------------------------------------------------------------
// REAL rope traversal: AIR leaf => rope jump => use key to get neighbor cube
// => descend starting from that neighbor node (NOT from root).
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// Macro 3D DDA (digital differential analyzer) for macro occupancy stepping
// --------------------------------------------------------------------------

struct MacroDDA {
  valid  : bool,
  cell   : f32,

  mx     : i32,
  my     : i32,
  mz     : i32,

  stepX  : i32,
  stepY  : i32,
  stepZ  : i32,

  tMaxX  : f32,
  tMaxY  : f32,
  tMaxZ  : f32,

  tDeltaX: f32,
  tDeltaY: f32,
  tDeltaZ: f32,
};

fn macro_dda_init(
  ro: vec3<f32>,
  rd: vec3<f32>,
  inv: vec3<f32>,
  tcur: f32,
  root_bmin: vec3<f32>,
  root_size: f32,
  macro_base: u32
) -> MacroDDA {
  var m: MacroDDA;
  m.valid = (macro_base != INVALID_U32);
  if (!m.valid) {
    m.cell = 0.0;
    return m;
  }

  m.cell = macro_cell_size(root_size);

  let p  = ro + rd * tcur;
  let lp = p - root_bmin;

  // current macro cell indices
  let mx0 = i32(floor(lp.x / m.cell));
  let my0 = i32(floor(lp.y / m.cell));
  let mz0 = i32(floor(lp.z / m.cell));

  // clamp into [0 .. MACRO_DIM-1]
  let md = i32(MACRO_DIM) - 1;
  m.mx = clamp(mx0, 0, md);
  m.my = clamp(my0, 0, md);
  m.mz = clamp(mz0, 0, md);

  m.stepX = select(-1, 1, rd.x > 0.0);
  m.stepY = select(-1, 1, rd.y > 0.0);
  m.stepZ = select(-1, 1, rd.z > 0.0);

  // next boundary plane index in macro grid space
  let nx = m.mx + select(0, 1, m.stepX > 0);
  let ny = m.my + select(0, 1, m.stepY > 0);
  let nz = m.mz + select(0, 1, m.stepZ > 0);

  // boundary positions in world space
  let bx = root_bmin.x + f32(nx) * m.cell;
  let by = root_bmin.y + f32(ny) * m.cell;
  let bz = root_bmin.z + f32(nz) * m.cell;

  // tMax values in ray-t space
  m.tMaxX = select(BIG_F32, tcur + (bx - p.x) * inv.x, abs(rd.x) >= EPS_INV);
  m.tMaxY = select(BIG_F32, tcur + (by - p.y) * inv.y, abs(rd.y) >= EPS_INV);
  m.tMaxZ = select(BIG_F32, tcur + (bz - p.z) * inv.z, abs(rd.z) >= EPS_INV);

  // tDelta: distance (in t) to cross one macro cell along each axis
  m.tDeltaX = select(BIG_F32, m.cell * abs(inv.x), abs(rd.x) >= EPS_INV);
  m.tDeltaY = select(BIG_F32, m.cell * abs(inv.y), abs(rd.y) >= EPS_INV);
  m.tDeltaZ = select(BIG_F32, m.cell * abs(inv.z), abs(rd.z) >= EPS_INV);

  return m;
}

fn macro_dda_exit_t(m: MacroDDA, t_exit: f32) -> f32 {
  return min(t_exit, min(m.tMaxX, min(m.tMaxY, m.tMaxZ)));
}

fn macro_dda_step_and_refresh(
  m: ptr<function, MacroDDA>,
  macro_base: u32,
  macro_empty: ptr<function, bool>
) {
  // --- step to next macro cell along smallest tMax axis
  if ((*m).tMaxX < (*m).tMaxY) {
    if ((*m).tMaxX < (*m).tMaxZ) {
      (*m).mx += (*m).stepX;
      (*m).tMaxX += (*m).tDeltaX;
    } else {
      (*m).mz += (*m).stepZ;
      (*m).tMaxZ += (*m).tDeltaZ;
    }
  } else {
    if ((*m).tMaxY < (*m).tMaxZ) {
      (*m).my += (*m).stepY;
      (*m).tMaxY += (*m).tDeltaY;
    } else {
      (*m).mz += (*m).stepZ;
      (*m).tMaxZ += (*m).tDeltaZ;
    }
  }

  // --- bounds check
  let md = i32(MACRO_DIM) - 1;
  if ((*m).mx < 0 || (*m).my < 0 || (*m).mz < 0 ||
      (*m).mx > md || (*m).my > md || (*m).mz > md) {
    (*m).valid = false;
    (*macro_empty) = false;
    return;
  }

  // --- refresh cached emptiness for the new cell
  let mxu = u32((*m).mx);
  let myu = u32((*m).my);
  let mzu = u32((*m).mz);
  let bit = macro_bit_index(mxu, myu, mzu);
  (*macro_empty) = !macro_test(macro_base, bit);
}

fn macro_chunk_is_empty(macro_base: u32) -> bool {
  // If there's no macro data, we can't prove emptiness.
  if (macro_base == INVALID_U32) { return false; }

  // 16 u32 words = 512 bits (8*8*8)
  var any: u32 = 0u;
  for (var i: u32 = 0u; i < MACRO_WORDS_PER_CHUNK; i = i + 1u) {
    any |= macro_occ[macro_base + i];
  }
  return any == 0u;
}



// --------------------------------------------------------------------------
// Rewritten traversal (macro DDA + leaf/rope traversal)
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
  let grass_probe_max_leaf = vs;

  let origin_vox_i = vec3<i32>(ch.origin.x, ch.origin.y, ch.origin.z);
  let time_s       = cam.voxel_params.y;
  let strength     = cam.voxel_params.z;

  var have_leaf: bool = false;
  var leaf: LeafState;

  // Macro DDA setup (valid==false if no macro data)
  var m = macro_dda_init(ro, rd, inv, tcur, root_bmin, root_size, ch.macro_base);
  var macro_empty: bool = false;
  if (m.valid) {
    let bit = macro_bit_index(u32(m.mx), u32(m.my), u32(m.mz));
    macro_empty = !macro_test(ch.macro_base, bit);
  }

  let MAX_ITERS: u32 = 192u + cam.chunk_size * 4u;

  for (var it: u32 = 0u; it < MAX_ITERS; it = it + 1u) {
    if (tcur > t_exit) { break; }

    // ------------------------------------------------------------
    // COARSE: macro empty jump using macro 3D DDA
    // ------------------------------------------------------------
    var t_macro_exit: f32 = t_exit;

    if (m.valid) {
      t_macro_exit = macro_dda_exit_t(m, t_exit);

      if (macro_empty) {
        // Jump across empty macro cell
        tcur = max(t_macro_exit, tcur) + eps_step;
        have_leaf = false;

        if (tcur > t_exit) { break; }

        // Enter next macro cell
        macro_dda_step_and_refresh(&m, ch.macro_base, &macro_empty);
        continue;
      }
    }

    // ------------------------------------------------------------
    // FINE: leaf traversal (rope traversal only on true leaf exit)
    // ------------------------------------------------------------
    let p  = ro + tcur * rd;
    let pq = p + rd * (1e-4 * vs);

    if (!have_leaf || !point_in_cube(pq, leaf.bmin, leaf.size)) {
      leaf = descend_leaf_sparse(pq, ch.node_base, 0u, root_bmin, root_size);
      have_leaf = true;
    }

    let slab    = cube_slab_inv(ro, inv, leaf.bmin, leaf.size);
    let t_leave = slab.t_exit;

    // Macro boundary event happens before leaf exit => advance to macro boundary
    // IMPORTANT: do not treat as leaf exit; do not do ropes.
    if (m.valid && (t_macro_exit < t_leave)) {
      tcur = max(t_macro_exit, tcur) + eps_step;

      if (tcur > t_exit) { break; }

      macro_dda_step_and_refresh(&m, ch.macro_base, &macro_empty);
      // leaf can straddle macro boundaries; keep cache
      continue;
    }

    // ------------------------------------------------------------
    // AIR leaf path
    // ------------------------------------------------------------
    if (leaf.mat == MAT_AIR) {
      // Grass probing scales with distance (LOD = level of detail):
      // - near: probe normally
      // - mid: probe only for smaller leaves
      // - far: skip probing entirely
      let lod_probe = grass_lod_from_t(tcur);

      var grass_leaf_limit = grass_probe_max_leaf; // default near
      if (lod_probe == 1u) {
        grass_leaf_limit = cam.voxel_params.x;     // mid: only probe 1-voxel leaves
      }

      if (lod_probe != 2u && leaf.size <= grass_leaf_limit) {
        let t0_probe = max(t_enter, tcur - eps_step);
        let t1_probe = min(t_leave, t_exit);

        if (t1_probe >= t0_probe) {
          let gh = probe_grass_columns_xz_dda(
            ro, rd, inv,
            t0_probe, t1_probe,
            root_bmin,
            origin_vox_i,
            vs,
            ch.colinfo_base,
            time_s,
            strength
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


      // True leaf exit => rope traversal
      let face = exit_face_from_slab(rd, slab);

      tcur = max(t_leave, tcur) + eps_step;
      if (tcur > t_exit) { break; }

      let p_next  = ro + tcur * rd;
      let pq_next = p_next + rd * (1e-4 * vs);

      // CASE A: real AIR node exists -> use its ropes
      if (leaf.has_node) {
        let nidx = rope_next_local(ch.node_base, leaf.idx_local, face);
        if (nidx == INVALID_U32) { have_leaf = false; continue; }

        let nk = node_at(ch.node_base, nidx).key;
        let c  = node_cube_from_key(root_bmin, root_size, nk);
        leaf = descend_leaf_sparse(pq_next, ch.node_base, nidx, c.xyz, c.w);
        have_leaf = true;
        continue;
      }

      // CASE B: missing-child AIR (virtual cube) -> use anchor
      if (leaf.has_anchor) {
        if (point_in_cube(pq_next, leaf.anchor_bmin, leaf.anchor_size)) {
          leaf = descend_leaf_sparse(
            pq_next,
            ch.node_base,
            leaf.anchor_idx,
            leaf.anchor_bmin,
            leaf.anchor_size
          );
          have_leaf = true;
          continue;
        }

        let nidx2 = rope_next_local(ch.node_base, leaf.anchor_idx, face);
        if (nidx2 == INVALID_U32) { have_leaf = false; continue; }

        let nk2 = node_at(ch.node_base, nidx2).key;
        let c2  = node_cube_from_key(root_bmin, root_size, nk2);

        leaf = descend_leaf_sparse(pq_next, ch.node_base, nidx2, c2.xyz, c2.w);
        have_leaf = true;
        continue;
      }

      // Fallback: re-descend from root next iter
      have_leaf = false;
      continue;
    }

    // ------------------------------------------------------------
    // Non-air leaf path (macro boundary already handled above)
    // ------------------------------------------------------------

    // Displaced leaf material
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

      // Miss: step out of this leaf
      tcur = max(t_leave, tcur) + eps_step;
      have_leaf = false;
      continue;
    }

    // Solid hit
    let bh = cube_hit_normal_from_slab(rd, slab, t_enter, t_exit);
    if (bh.hit) {
      // Optional grass-on-solid-face probe when solid voxel is grass
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

    // Miss solid: step out of leaf
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

  // Only trace as far as fog can contribute (huge perf win on big loaded grids)
  var t_enter = max(rtg.x, 0.0);
  let t_exit  = min(rtg.y, FOG_MAX_DIST);

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

  // Tight cap: how many chunk boundaries can this ray possibly cross?
  // (visited cells <= boundary_crossings + 1)
  var rem_x: i32 = 0;
  var rem_y: i32 = 0;
  var rem_z: i32 = 0;

  if (abs(rd.x) >= EPS_INV) {
    rem_x = select(lcx, (nx - 1 - lcx), step_x > 0);
  }
  if (abs(rd.y) >= EPS_INV) {
    rem_y = select(lcy, (ny - 1 - lcy), step_y > 0);
  }
  if (abs(rd.z) >= EPS_INV) {
    rem_z = select(lcz, (nz - 1 - lcz), step_z > 0);
  }

  let max_chunk_steps = min(u32(rem_x + rem_y + rem_z) + 1u, 512u);


  for (var s: u32 = 0u; s < max_chunk_steps; s = s + 1u) {
    if (t_local > t_exit_local) { break; }

    let tNextLocal = min(tMaxX, min(tMaxY, tMaxZ));
    if (best.hit != 0u && (start_t + tNextLocal) >= best.t) { break; }

    let slot = chunk_grid[u32(idx_i)];
    if (slot != INVALID_U32 && slot < cam.chunk_count) {
      let ch = chunks[slot];

      // Skip trivially empty chunks (macro occupancy says "no bits set")
      if (!macro_chunk_is_empty(ch.macro_base)) {
        let cell_enter = start_t + t_local;
        let cell_exit  = start_t + min(tNextLocal, t_exit_local);

        let h = trace_chunk_rope_interval(ro, rd, ch, cell_enter, cell_exit);
        if (h.hit != 0u && h.t < best.t) { best = h; }
      }
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


// Visit each (lx,lz) column crossed by [t0,t1] once (XZ DDA).
// Calls try_grass_slab_hit at most once per visited column.
// Returns hit or miss.
fn probe_grass_columns_xz_dda(
  ro: vec3<f32>,
  rd: vec3<f32>,
  inv: vec3<f32>,
  t0_in: f32,
  t1_in: f32,
  root_bmin: vec3<f32>,
  origin_vox_i: vec3<i32>,
  vs: f32,
  colinfo_base: u32,
  time_s: f32,
  strength: f32
) -> GrassHit {
  // Clamp & early out
  var t0 = t0_in;
  var t1 = t1_in;
  if (t1 <= t0) {
    return GrassHit(false, BIG_F32, vec3<f32>(0.0));
  }

  // Start point (slightly nudged to avoid exact-boundary issues)
  let eps = 1e-4 * vs;
  var t = t0 + eps;
  if (t > t1) { t = t0; }

  var p = ro + rd * t;

  // Local voxel coords in chunk (floating)
  var lx_f = (p.x - root_bmin.x) / vs;
  var lz_f = (p.z - root_bmin.z) / vs;

  // Integer column indices
  var lx: i32 = i32(floor(lx_f));
  var lz: i32 = i32(floor(lz_f));

  // Clamp to 0..63 (chunk is 64^3 in your code)
  lx = clamp(lx, 0, 63);
  lz = clamp(lz, 0, 63);

  // Steps in X and Z
  let stepX: i32 = select(-1, 1, rd.x > 0.0);
  let stepZ: i32 = select(-1, 1, rd.z > 0.0);

  // Setup X boundary crossing times (in ray-t space)
  var tMaxX: f32 = BIG_F32;
  var tDeltaX: f32 = BIG_F32;
  if (abs(rd.x) >= EPS_INV) {
    // Next voxel boundary in meters
    let nextX = root_bmin.x + f32(select(lx, lx + 1, stepX > 0)) * vs;
    tMaxX   = t + (nextX - p.x) * inv.x;
    tDeltaX = vs * abs(inv.x);
  }

  // Setup Z boundary crossing times
  var tMaxZ: f32 = BIG_F32;
  var tDeltaZ: f32 = BIG_F32;
  if (abs(rd.z) >= EPS_INV) {
    let nextZ = root_bmin.z + f32(select(lz, lz + 1, stepZ > 0)) * vs;
    tMaxZ   = t + (nextZ - p.z) * inv.z;
    tDeltaZ = vs * abs(inv.z);
  }

  // Safety cap: max columns crossed within a tiny leaf is small.
  // Still cap to be safe in edge cases / long segments.
  let MAX_COLS: u32 = 64u;

  for (var iter: u32 = 0u; iter < MAX_COLS; iter = iter + 1u) {
    if (t > t1) { break; }

    // End of current column interval is the next boundary (or t1)
    let tNext = min(min(tMaxX, tMaxZ), t1);

    // --- Check this column’s top map once
    let lx_u: u32 = u32(lx);
    let lz_u: u32 = u32(lz);

    let e16 = colinfo_entry_u16(colinfo_base, lx_u, lz_u);
    let ci  = colinfo_decode(e16);

    if (ci.valid && ci.mat == MAT_GRASS) {
      // World voxel coords for the grass cell (at column-top y)
      let wx: i32 = origin_vox_i.x + lx;
      let wy: i32 = origin_vox_i.y + i32(ci.y_vox);
      let wz: i32 = origin_vox_i.z + lz;

      let cell_bmin_m = vec3<f32>(f32(wx), f32(wy), f32(wz)) * vs;
      let id_vox      = vec3<f32>(f32(wx), f32(wy), f32(wz));

      // Tight per-column interval [t, tNext]
      let gh = try_grass_slab_hit(
        ro, rd,
        t, tNext,
        cell_bmin_m, id_vox,
        vs, time_s, strength
      );

      if (gh.hit) { return gh; }
    }

    // Move to next column
    if (tNext >= t1) { break; }

    // Step across whichever boundary is hit first
    if (tMaxX < tMaxZ) {
      lx += stepX;
      t = tMaxX + eps;
      tMaxX += tDeltaX;
    } else {
      lz += stepZ;
      t = tMaxZ + eps;
      tMaxZ += tDeltaZ;
    }

    // Stop if we walked out of chunk xz bounds
    if (lx < 0 || lx > 63 || lz < 0 || lz > 63) { break; }

    p = ro + rd * t;
  }

  return GrassHit(false, BIG_F32, vec3<f32>(0.0));
}
