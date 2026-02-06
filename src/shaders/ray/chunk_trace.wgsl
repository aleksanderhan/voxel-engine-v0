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

struct ChunkTraceResult {
  hit : HitGeom,
  anchor_valid: bool,
  anchor_key  : u32,
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

  if (lp.x < 0.0 || lp.y < 0.0 || lp.z < 0.0 ||
      lp.x >= root_size || lp.y >= root_size || lp.z >= root_size) {
    // outside chunk => don't claim empty
    return MacroCell(vec3<f32>(0.0), 0.0, false);
  }


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

struct AnchorNode {
  valid : bool,
  idx   : u32,
  bmin  : vec3<f32>,
  size  : f32,
  key   : u32,
};

fn anchor_from_key(
  node_base: u32,
  root_bmin: vec3<f32>,
  root_size: f32,
  key: u32
) -> AnchorNode {
  if (key == INVALID_U32) {
    return AnchorNode(false, 0u, vec3<f32>(0.0), 0.0, INVALID_U32);
  }

  let lvl = min(node_level(key), chunk_max_depth());
  let tx  = node_x(key);
  let ty  = node_y(key);
  let tz  = node_z(key);

  var idx  : u32 = 0u;
  var bmin : vec3<f32> = root_bmin;
  var size : f32 = root_size;

  for (var d: u32 = 0u; d < 32u; d = d + 1u) {
    if (d >= lvl) {
      let nk = node_at(node_base, idx).key;
      return AnchorNode(true, idx, bmin, size, nk);
    }

    let n = node_at(node_base, idx);
    if (n.child_base == LEAF_U32) {
      return AnchorNode(true, idx, bmin, size, n.key);
    }

    let shift = (lvl - 1u) - d;
    let hx = (tx >> shift) & 1u;
    let hy = (ty >> shift) & 1u;
    let hz = (tz >> shift) & 1u;
    let ci = hx | (hy << 1u) | (hz << 2u);

    let bit = 1u << ci;
    if ((n.child_mask & bit) == 0u) {
      return AnchorNode(true, idx, bmin, size, n.key);
    }

    let half = size * 0.5;
    bmin = bmin + vec3<f32>(
      select(0.0, half, hx != 0u),
      select(0.0, half, hy != 0u),
      select(0.0, half, hz != 0u)
    );
    size = half;
    let r = child_rank(n.child_mask, ci);
    idx = n.child_base + r;
  }

  return AnchorNode(true, idx, bmin, size, node_at(node_base, idx).key);
}

fn anchor_from_leaf(leaf: LeafState, node_base: u32) -> AnchorNode {
  if (leaf.has_node) {
    let k = node_at(node_base, leaf.idx_local).key;
    return AnchorNode(true, leaf.idx_local, leaf.bmin, leaf.size, k);
  }
  if (leaf.has_anchor) {
    let k = node_at(node_base, leaf.anchor_idx).key;
    return AnchorNode(true, leaf.anchor_idx, leaf.anchor_bmin, leaf.anchor_size, k);
  }
  return AnchorNode(false, 0u, vec3<f32>(0.0), 0.0, INVALID_U32);
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

// Add this near exit_face_from_slab
struct ExitFaceRes {
  face : u32,   // 0..5
  amb  : u32,   // 1 => ambiguous (edge/corner exit)
};

fn exit_face_from_slab_safe(rd: vec3<f32>, slab: CubeSlab) -> ExitFaceRes {
  // 0=+X,1=-X,2=+Y,3=-Y,4=+Z,5=-Z
  let te = slab.t_exit;
  let eps = 1e-5 * max(1.0, abs(te));

  // IMPORTANT: ignore near-zero ray components (prevents nonsense matches)
  let mx = (abs(rd.x) >= EPS_INV) && (abs(slab.tmaxv.x - te) <= eps);
  let my = (abs(rd.y) >= EPS_INV) && (abs(slab.tmaxv.y - te) <= eps);
  let mz = (abs(rd.z) >= EPS_INV) && (abs(slab.tmaxv.z - te) <= eps);

  let cx = select(0u, 1u, mx);
  let cy = select(0u, 1u, my);
  let cz = select(0u, 1u, mz);
  let c  = cx + cy + cz;

  // Edge/corner exit => ambiguous; don't rope-jump.
  if (c >= 2u) {
    return ExitFaceRes(0u, 1u);
  }

  // Single-axis (or fallback)
  var axis: u32 = 0u;

  if (c == 1u) {
    // choose the matching axis
    if (mx) { axis = 0u; }
    if (my) { axis = 1u; }
    if (mz) { axis = 2u; }
  } else {
    // fallback: argmin(tmaxv) (same intent as your old fallback)
    var bestv: f32 = slab.tmaxv.x;
    axis = 0u;
    if (slab.tmaxv.y < bestv) { bestv = slab.tmaxv.y; axis = 1u; }
    if (slab.tmaxv.z < bestv) { bestv = slab.tmaxv.z; axis = 2u; }
  }

  if (axis == 0u) { return ExitFaceRes(select(1u, 0u, rd.x > 0.0), 0u); }
  if (axis == 1u) { return ExitFaceRes(select(3u, 2u, rd.y > 0.0), 0u); }
  return ExitFaceRes(select(5u, 4u, rd.z > 0.0), 0u);
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
// Macro occupancy query (persistent DDA state)
// --------------------------------------------------------------------------

struct MacroStep {
  valid : bool,
  empty : bool,
  t_exit: f32,
};

struct MacroDDAState {
  enabled: bool,
  mx     : i32,
  my     : i32,
  mz     : i32,
  step_x : i32,
  step_y : i32,
  step_z : i32,
  tMaxX  : f32,
  tMaxY  : f32,
  tMaxZ  : f32,
  tDeltaX: f32,
  tDeltaY: f32,
  tDeltaZ: f32,
  cell   : f32,
};

struct MacroInterval {
  valid: bool,
  t0   : f32,
  t1   : f32,
};

fn macro_dda_init(
  ro: vec3<f32>,
  rd: vec3<f32>,
  inv: vec3<f32>,
  tcur: f32,
  root_bmin: vec3<f32>,
  root_size: f32,
  macro_base: u32
) -> MacroDDAState {
  var state: MacroDDAState;
  state.enabled = false;
  state.mx = 0;
  state.my = 0;
  state.mz = 0;
  state.step_x = 0;
  state.step_y = 0;
  state.step_z = 0;
  state.tMaxX = BIG_F32;
  state.tMaxY = BIG_F32;
  state.tMaxZ = BIG_F32;
  state.tDeltaX = BIG_F32;
  state.tDeltaY = BIG_F32;
  state.tDeltaZ = BIG_F32;
  state.cell = 0.0;

  if (macro_base == INVALID_U32) {
    return state;
  }

  let cell = macro_cell_size(root_size);
  let p = ro + rd * tcur;
  let lp = p - root_bmin;

  if (lp.x < 0.0 || lp.y < 0.0 || lp.z < 0.0 ||
      lp.x >= root_size || lp.y >= root_size || lp.z >= root_size) {
    return state;
  }

  let mx_i = i32(floor(lp.x / cell));
  let my_i = i32(floor(lp.y / cell));
  let mz_i = i32(floor(lp.z / cell));

  if (mx_i < 0 || my_i < 0 || mz_i < 0 ||
      mx_i >= i32(MACRO_DIM) || my_i >= i32(MACRO_DIM) || mz_i >= i32(MACRO_DIM)) {
    return state;
  }

  state.enabled = true;
  state.mx = mx_i;
  state.my = my_i;
  state.mz = mz_i;
  state.step_x = select(-1, 1, rd.x > 0.0);
  state.step_y = select(-1, 1, rd.y > 0.0);
  state.step_z = select(-1, 1, rd.z > 0.0);
  state.tDeltaX = abs(cell * inv.x);
  state.tDeltaY = abs(cell * inv.y);
  state.tDeltaZ = abs(cell * inv.z);
  state.cell = cell;

  if (abs(rd.x) >= EPS_INV) {
    let next_ix = mx_i + select(0, 1, rd.x > 0.0);
    let bx = root_bmin.x + f32(next_ix) * cell;
    state.tMaxX = tcur + (bx - p.x) * inv.x;
  }

  if (abs(rd.y) >= EPS_INV) {
    let next_iy = my_i + select(0, 1, rd.y > 0.0);
    let by = root_bmin.y + f32(next_iy) * cell;
    state.tMaxY = tcur + (by - p.y) * inv.y;
  }

  if (abs(rd.z) >= EPS_INV) {
    let next_iz = mz_i + select(0, 1, rd.z > 0.0);
    let bz = root_bmin.z + f32(next_iz) * cell;
    state.tMaxZ = tcur + (bz - p.z) * inv.z;
  }

  return state;
}

fn macro_coarse_interval(
  ro: vec3<f32>,
  rd: vec3<f32>,
  inv: vec3<f32>,
  root_bmin: vec3<f32>,
  root_size: f32,
  macro_base: u32,
  t_enter: f32,
  t_exit: f32,
  eps_step: f32
) -> MacroInterval {
  var out: MacroInterval;
  out.valid = false;
  out.t0 = 0.0;
  out.t1 = 0.0;

  var tcur = max(t_enter, 0.0) + eps_step;
  if (tcur > t_exit) { return out; }

  if (macro_base == INVALID_U32) {
    out.valid = true;
    out.t0 = tcur;
    out.t1 = t_exit;
    return out;
  }

  var state = macro_dda_init(ro, rd, inv, tcur, root_bmin, root_size, macro_base);
  if (!state.enabled) { return out; }

  let MAX_MACRO_ITERS: u32 = 128u;
  for (var i: u32 = 0u; i < MAX_MACRO_ITERS; i = i + 1u) {
    if (!state.enabled) { break; }
    macro_dda_sync(&state, tcur);
    let step = macro_dda_current(state, macro_base);
    if (!step.valid) { break; }

    let t_cell_exit = min(step.t_exit, t_exit);
    if (!step.empty) {
      out.valid = true;
      out.t0 = tcur;
      out.t1 = t_cell_exit;
      return out;
    }

    tcur = max(t_cell_exit, tcur) + eps_step;
    if (tcur > t_exit) { break; }

    if (state.enabled && tcur >= step.t_exit) {
      macro_dda_step(&state);
    }
  }

  return out;
}

fn macro_dda_t_exit(state: MacroDDAState) -> f32 {
  return min(state.tMaxX, min(state.tMaxY, state.tMaxZ));
}

fn macro_dda_step(state: ptr<function, MacroDDAState>) {
  let tNext = macro_dda_t_exit(*state);
  let epsTie = 1e-6 * max(1.0, abs(tNext));

  if (abs((*state).tMaxX - tNext) <= epsTie) {
    (*state).mx += (*state).step_x;
    (*state).tMaxX += (*state).tDeltaX;
  }
  if (abs((*state).tMaxY - tNext) <= epsTie) {
    (*state).my += (*state).step_y;
    (*state).tMaxY += (*state).tDeltaY;
  }
  if (abs((*state).tMaxZ - tNext) <= epsTie) {
    (*state).mz += (*state).step_z;
    (*state).tMaxZ += (*state).tDeltaZ;
  }

  if ((*state).mx < 0 || (*state).my < 0 || (*state).mz < 0 ||
      (*state).mx >= i32(MACRO_DIM) ||
      (*state).my >= i32(MACRO_DIM) ||
      (*state).mz >= i32(MACRO_DIM)) {
    (*state).enabled = false;
  }
}

fn macro_dda_sync(state: ptr<function, MacroDDAState>, tcur: f32) {
  for (var i: u32 = 0u; i < 32u; i = i + 1u) {
    if (!(*state).enabled) { break; }
    let t_exit = macro_dda_t_exit(*state);
    if (tcur < t_exit) { break; }
    macro_dda_step(state);
  }
}

fn macro_dda_current(state: MacroDDAState, macro_base: u32) -> MacroStep {
  var out: MacroStep;
  out.valid = state.enabled;
  out.empty = false;
  out.t_exit = macro_dda_t_exit(state);

  if (!state.enabled || macro_base == INVALID_U32) {
    out.valid = false;
    return out;
  }

  let mx = u32(state.mx);
  let my = u32(state.my);
  let mz = u32(state.mz);
  let bit = macro_bit_index(mx, my, mz);
  out.empty = !macro_test(macro_base, bit);
  return out;
}

// --------------------------------------------------------------------------
// Rewritten traversal (macro occupancy + leaf/rope traversal)
// --------------------------------------------------------------------------
fn trace_chunk_rope_interval(
  ro: vec3<f32>,
  rd: vec3<f32>,
  ch: ChunkMeta,
  t_enter: f32,
  t_exit: f32,
  use_anchor: bool,
  anchor_key: u32
) -> ChunkTraceResult {
  let vs = cam.voxel_params.x;

  let root_bmin_vox = vec3<f32>(f32(ch.origin.x), f32(ch.origin.y), f32(ch.origin.z));
  let root_bmin     = root_bmin_vox * vs;
  let root_size     = f32(cam.chunk_size) * vs;

  let eps_step = 1e-4 * vs;
  var tcur     = max(t_enter, 0.0) + eps_step;

  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));
  var macro_state = macro_dda_init(
    ro, rd, inv, tcur,
    root_bmin, root_size,
    ch.macro_base
  );

  var anchor_node = AnchorNode(false, 0u, vec3<f32>(0.0), 0.0, INVALID_U32);
  if (use_anchor) {
    anchor_node = anchor_from_key(ch.node_base, root_bmin, root_size, anchor_key);
  }

  // Only probe grass when we are in small-enough air leaves
  let grass_probe_max_leaf = vs;

  let origin_vox_i = vec3<i32>(ch.origin.x, ch.origin.y, ch.origin.z);
  let time_s       = cam.voxel_params.y;
  let strength     = cam.voxel_params.z;

  var have_leaf: bool = false;
  var leaf: LeafState;

  let MAX_ITERS: u32 = 128u + cam.chunk_size * 2u; // 256 for 64³

  for (var it: u32 = 0u; it < MAX_ITERS; it = it + 1u) {
    if (tcur > t_exit) { break; }

    // ------------------------------------------------------------
    // COARSE: macro empty jump using per-step macro occupancy
    // ------------------------------------------------------------
    if (macro_state.enabled) {
      macro_dda_sync(&macro_state, tcur);
    }

    let macro_step = macro_dda_current(macro_state, ch.macro_base);

    if (macro_step.valid && macro_step.empty) {
      // Jump across empty macro cell
      let t_macro_exit = min(t_exit, macro_step.t_exit);
      tcur = max(t_macro_exit, tcur) + eps_step;
      have_leaf = false;
      if (macro_state.enabled && tcur >= macro_step.t_exit) {
        macro_dda_step(&macro_state);
      }
      continue;
    }

    // ------------------------------------------------------------
    // FINE: leaf traversal (rope traversal only on true leaf exit)
    // ------------------------------------------------------------
    let p  = ro + tcur * rd;
    let pq = p + ray_eps_vec(rd, 1e-4 * vs);

    if (!have_leaf || !point_in_cube(pq, leaf.bmin, leaf.size)) {
      if (anchor_node.valid && point_in_cube(pq, anchor_node.bmin, anchor_node.size)) {
        leaf = descend_leaf_sparse(pq, ch.node_base, anchor_node.idx, anchor_node.bmin, anchor_node.size);
      } else {
        leaf = descend_leaf_sparse(pq, ch.node_base, 0u, root_bmin, root_size);
      }
      let next_anchor = anchor_from_leaf(leaf, ch.node_base);
      if (next_anchor.valid && (!anchor_node.valid || next_anchor.key != anchor_node.key)) {
        anchor_node = next_anchor;
      }
      have_leaf = true;
    }

    let slab    = cube_slab_inv(ro, inv, leaf.bmin, leaf.size);
    let t_leave = slab.t_exit;

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

      if (ENABLE_GRASS && lod_probe != 2u && leaf.size <= grass_leaf_limit) {
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
            return ChunkTraceResult(outg, anchor_node.valid, anchor_node.key);
          }
        }
      }

      // True leaf exit => rope traversal (but bail out on edge/corner exits)
      let ex = exit_face_from_slab_safe(rd, slab);

      tcur = max(t_leave, tcur) + eps_step;
      if (tcur > t_exit) { break; }

      if (ex.amb != 0u) {
        // Edge/corner exit: ropes are unstable here; re-descend next iter.
        have_leaf = false;

        continue;
      }

      let face = ex.face;

      let p_next  = ro + tcur * rd;
      let pq_next = p_next + ray_eps_vec(rd, 1e-4 * vs);

      // CASE A: real AIR node exists -> use its ropes
      if (leaf.has_node) {
        let nidx = rope_next_local(ch.node_base, leaf.idx_local, face);
        if (nidx == INVALID_U32) { have_leaf = false; continue; }

        let nk = node_at(ch.node_base, nidx).key;
        let c  = node_cube_from_key(root_bmin, root_size, nk);
        leaf = descend_leaf_sparse(pq_next, ch.node_base, nidx, c.xyz, c.w);
        let next_anchor = anchor_from_leaf(leaf, ch.node_base);
        if (next_anchor.valid && (!anchor_node.valid || next_anchor.key != anchor_node.key)) {
          anchor_node = next_anchor;
        }
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
        let next_anchor = anchor_from_leaf(leaf, ch.node_base);
        if (next_anchor.valid && (!anchor_node.valid || next_anchor.key != anchor_node.key)) {
          anchor_node = next_anchor;
        }
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
      let tmin_iter = max(t_enter, tcur - eps_step);
      let h2 = leaf_displaced_cube_hit(
        ro, rd,
        leaf.bmin, leaf.size,
        time_s, strength,
        tmin_iter, t_exit
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
        return ChunkTraceResult(out, anchor_node.valid, anchor_node.key);
      }

      // Miss: step out of this leaf
      tcur = max(t_leave, tcur) + eps_step;
      have_leaf = false;
      continue;
    }

    // Solid hit
    let tmin_iter = max(t_enter, tcur - eps_step); // keep it inside the current chunk interval
    let bh = cube_hit_normal_from_slab(rd, slab, tmin_iter, t_exit);
    if (bh.hit) {
      // Optional grass-on-solid-face probe when solid voxel is grass
      if (ENABLE_GRASS && leaf.mat == MAT_GRASS) {
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
          return ChunkTraceResult(outg, anchor_node.valid, anchor_node.key);
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
      return ChunkTraceResult(out, anchor_node.valid, anchor_node.key);
    }

    // Miss solid: step out of leaf
    tcur = max(t_leave, tcur) + eps_step;
    have_leaf = false;
  }

  return ChunkTraceResult(miss_hitgeom(), anchor_node.valid, anchor_node.key);
}

// --------------------------------------------------------------------------
// Scene voxel tracing over streamed chunk grid (for main_primary)
// --------------------------------------------------------------------------

struct VoxTraceResult {
  in_grid : bool,
  best    : HitGeom,
  t_exit  : f32,
  anchor_valid: bool,
  anchor_key  : u32,
  anchor_chunk: vec3<i32>,
};

fn chunk_coords_neighbor(a: vec3<i32>, b: vec3<i32>) -> bool {
  let dx = abs(a.x - b.x);
  let dy = abs(a.y - b.y);
  let dz = abs(a.z - b.z);
  return (dx <= 1 && dy <= 1 && dz <= 1);
}

fn trace_scene_voxels_interval(
  ro: vec3<f32>,
  rd: vec3<f32>,
  t_min: f32,
  t_max: f32,
  anchor_valid_in: bool,
  anchor_chunk_in: vec3<i32>,
  anchor_key_in: u32
) -> VoxTraceResult {
  if (cam.chunk_count == 0u) {
    return VoxTraceResult(false, miss_hitgeom(), 0.0, false, INVALID_U32, vec3<i32>(0));
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
  var t_enter = max(max(rtg.x, 0.0), t_min);
  let t_exit  = min(min(rtg.y, FOG_MAX_DIST), t_max);

  if (t_exit < t_enter) {
    return VoxTraceResult(false, miss_hitgeom(), 0.0, false, INVALID_U32, vec3<i32>(0));
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
    return VoxTraceResult(false, miss_hitgeom(), 0.0, false, INVALID_U32, vec3<i32>(0));
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

  let eps_step = 1e-4 * voxel_size;

  var best = miss_hitgeom();
  var best_anchor_valid = false;
  var best_anchor_key = INVALID_U32;
  var best_anchor_chunk = vec3<i32>(0);
  var anchor_valid = anchor_valid_in;
  var anchor_coord = anchor_chunk_in;
  var anchor_key = anchor_key_in;
  var anchor_slot: u32 = INVALID_U32;

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

  var found_hit = false;

  for (var s: u32 = 0u; s < max_chunk_steps; s = s + 1u) {
    if (t_local > t_exit_local) { break; }

    let tNextLocal = min(tMaxX, min(tMaxY, tMaxZ));
    if (best.hit != 0u && (start_t + tNextLocal) >= best.t) { break; }

    let slot = chunk_grid[u32(idx_i)];
    if (slot != INVALID_U32 && slot < cam.chunk_count) {
      let ch = chunks[slot];

      // Skip trivially empty chunks (macro occupancy says "no bits set")
      if (ch.macro_empty == 0u) {
        let cell_enter = start_t + t_local;
        let cell_exit  = start_t + min(tNextLocal, t_exit_local);

        let cur_coord = vec3<i32>(go.x + lcx, go.y + lcy, go.z + lcz);

        let root_bmin = vec3<f32>(f32(ch.origin.x), f32(ch.origin.y), f32(ch.origin.z)) * voxel_size;
        let root_size = f32(cam.chunk_size) * voxel_size;

        var t_scan = cell_enter;
        let MAX_COARSE_ITERS: u32 = 64u;

        for (var c: u32 = 0u; c < MAX_COARSE_ITERS; c = c + 1u) {
          if (t_scan >= cell_exit) { break; }

          let coarse = macro_coarse_interval(
            ro,
            rd,
            inv,
            root_bmin,
            root_size,
            ch.macro_base,
            t_scan,
            cell_exit,
            eps_step
          );

          if (!coarse.valid) { break; }

          let use_anchor = anchor_valid &&
            (slot == anchor_slot || chunk_coords_neighbor(cur_coord, anchor_coord));
          let anchor_key_use = select(INVALID_U32, anchor_key, use_anchor);

          let h = trace_chunk_rope_interval(
            ro,
            rd,
            ch,
            coarse.t0,
            coarse.t1,
            use_anchor,
            anchor_key_use
          );
          if (h.hit.hit != 0u && h.hit.t < best.t) {
            best = h.hit;
            best_anchor_valid = h.anchor_valid;
            best_anchor_key = h.anchor_key;
            best_anchor_chunk = cur_coord;
            found_hit = true;
            break;
          }

          anchor_valid = h.anchor_valid;
          anchor_key = h.anchor_key;
          anchor_coord = cur_coord;
          anchor_slot = slot;

          t_scan = coarse.t1 + eps_step;
        }
      }
    }


    // --- tie-aware DDA advance (prevents skipping on edges/corners) ---
    let epsTie = 1e-6 * max(1.0, abs(tNextLocal));

    // Step all axes that match tNextLocal
    if (abs(tMaxX - tNextLocal) <= epsTie) {
      lcx += step_x; idx_i += didx_x;
      tMaxX += tDeltaX;
    }
    if (abs(tMaxY - tNextLocal) <= epsTie) {
      lcy += step_y; idx_i += didx_y;
      tMaxY += tDeltaY;
    }
    if (abs(tMaxZ - tNextLocal) <= epsTie) {
      lcz += step_z; idx_i += didx_z;
      tMaxZ += tDeltaZ;
    }

    t_local = tNextLocal;
    // ---------------------------------------------------------------

    if (found_hit) { break; }

    if (lcx < 0 || lcy < 0 || lcz < 0 || lcx >= nx || lcy >= ny || lcz >= nz) { break; }
  }

  return VoxTraceResult(true, best, t_exit, best_anchor_valid, best_anchor_key, best_anchor_chunk);
}

fn trace_scene_voxels(ro: vec3<f32>, rd: vec3<f32>) -> VoxTraceResult {
  return trace_scene_voxels_interval(ro, rd, 0.0, FOG_MAX_DIST, false, vec3<i32>(0), INVALID_U32);
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

      // ------------------------------------------------------------
      // NEW: ultra-cheap vertical overlap test (kills most calls)
      // Grass slab Y range:
      let slab_y0 = cell_bmin_m.y + vs;
      let slab_y1 = slab_y0 + (GRASS_LAYER_HEIGHT_VOX * vs);

      // Segment y range over [t, tNext]
      // y(t) = ro.y + rd.y * t
      let yA = ro.y + rd.y * t;
      let yB = ro.y + rd.y * tNext;
      let seg_y0 = min(yA, yB);
      let seg_y1 = max(yA, yB);

      // If segment doesn't cross the grass slab vertically, skip immediately.
      if (seg_y1 < slab_y0 || seg_y0 > slab_y1) {
        // no overlap => cannot hit grass in this column interval
      } else {
        // Tight per-column interval [t, tNext]
        let gh = try_grass_slab_hit(
          ro, rd,
          t, tNext,
          cell_bmin_m, id_vox,
          vs, time_s, strength
        );

        if (gh.hit) { return gh; }
      }
      // ------------------------------------------------------------
    }


    // Move to next column
    if (tNext >= t1) { break; }

    // Step across whichever boundary is hit first
    let tStep = min(tMaxX, tMaxZ);
    let epsTie = 1e-6 * max(1.0, abs(tStep));

    if (abs(tMaxX - tStep) <= epsTie) {
      lx += stepX;
      tMaxX += tDeltaX;
    }
    if (abs(tMaxZ - tStep) <= epsTie) {
      lz += stepZ;
      tMaxZ += tDeltaZ;
    }

    // advance t (keep your eps nudge)
    t = tStep + eps;


    // Stop if we walked out of chunk xz bounds
    if (lx < 0 || lx > 63 || lz < 0 || lz > 63) { break; }

    p = ro + rd * t;
  }

  return GrassHit(false, BIG_F32, vec3<f32>(0.0));
}
