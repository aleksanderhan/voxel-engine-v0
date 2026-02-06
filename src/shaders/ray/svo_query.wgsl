//// --------------------------------------------------------------------------
//// SVO query (leaf query + convenience predicates)
//// --------------------------------------------------------------------------

struct LeafQuery {
  bmin : vec3<f32>,
  size : f32,
  mat  : u32,
};

fn query_leaf_at(
  p_in: vec3<f32>,
  root_bmin: vec3<f32>,
  root_size: f32,
  node_base: u32,
  macro_base: u32
) -> LeafQuery {
  var idx: u32 = node_base;
  var bmin: vec3<f32> = root_bmin;
  var size: f32 = root_size;

  // ------------------------------------------------------------------------
  // Macro occupancy early-out (8x8x8). If the macro cell is empty => MAT_AIR.
  // ------------------------------------------------------------------------
  if (macro_base != INVALID_U32) {
  let cell = macro_cell_size(root_size);

  let lp = p_in - root_bmin;

  // If query point is outside this chunk cube, DO NOT clamp into macro cell 0.
  // Treat as "not empty" so traversal continues normally, or return air safely.
  if (lp.x < 0.0 || lp.y < 0.0 || lp.z < 0.0 ||
      lp.x >= root_size || lp.y >= root_size || lp.z >= root_size) {
    // safest: outside chunk => air
    return LeafQuery(root_bmin, root_size, MAT_AIR);
  }

  let mx_i = i32(floor(lp.x / cell));
  let my_i = i32(floor(lp.y / cell));
  let mz_i = i32(floor(lp.z / cell));

  if (mx_i < 0 || my_i < 0 || mz_i < 0 ||
      mx_i >= i32(MACRO_DIM) || my_i >= i32(MACRO_DIM) || mz_i >= i32(MACRO_DIM)) {
    return LeafQuery(root_bmin, root_size, MAT_AIR);
  }

  let mx = u32(mx_i);
  let my = u32(my_i);
  let mz = u32(mz_i);

  let bit = macro_bit_index(mx, my, mz);
  if (!macro_test(macro_base, bit)) {
    // macro cell empty => definitely air
    return LeafQuery(root_bmin, root_size, MAT_AIR);
  }
}



  let min_leaf: f32 = cam.voxel_params.x;
  var p = p_in;

  for (var d: u32 = 0u; d < 32u; d = d + 1u) {
    let n = nodes[idx];

    if (n.child_base == LEAF_U32) {
      return LeafQuery(bmin, size, n.material);
    }

    let max_d = chunk_max_depth();
    if (d >= max_d) {
      // We are at voxel resolution; if the node still isn't a LEAF_U32, treat missing child as air,
      // but DO NOT blanket-return air here earlier than max depth.
      // The loop will keep descending until leaf/missing child; this is just a safety stop.
      return LeafQuery(bmin, size, MAT_AIR);
    }


    let half = size * 0.5;
    let mid  = bmin + vec3<f32>(half);

    let e = 1e-6 * size;

    let hx = select(0u, 1u, p.x > mid.x + e);
    let hy = select(0u, 1u, p.y > mid.y + e);
    let hz = select(0u, 1u, p.z > mid.z + e);
    let ci = hx | (hy << 1u) | (hz << 2u);

    let child_bmin = bmin + vec3<f32>(
      select(0.0, half, hx != 0u),
      select(0.0, half, hy != 0u),
      select(0.0, half, hz != 0u)
    );

    let bit = 1u << ci;
    if ((n.child_mask & bit) == 0u) {
      return LeafQuery(child_bmin, half, MAT_AIR);
    }

    let rank = child_rank(n.child_mask, ci);
    idx = node_base + (n.child_base + rank);

    bmin = child_bmin;
    size = half;
  }

  return LeafQuery(bmin, size, MAT_AIR);
}

fn make_air_leaf(bmin: vec3<f32>, size: f32) -> LeafQuery {
  return LeafQuery(bmin, size, MAT_AIR);
}

fn leaf_face_normal(p: vec3<f32>, leaf: LeafQuery) -> vec3<f32> {
  let size = leaf.size;
  let local = clamp(p - leaf.bmin, vec3<f32>(0.0), vec3<f32>(size));

  let dnx = local.x;
  let dpx = size - local.x;
  let dny = local.y;
  let dpy = size - local.y;
  let dnz = local.z;
  let dpz = size - local.z;

  var dmin = dpx;
  var n = vec3<f32>(1.0, 0.0, 0.0);

  if (dnx < dmin) { dmin = dnx; n = vec3<f32>(-1.0, 0.0, 0.0); }
  if (dny < dmin) { dmin = dny; n = vec3<f32>(0.0, -1.0, 0.0); }
  if (dpy < dmin) { dmin = dpy; n = vec3<f32>(0.0, 1.0, 0.0); }
  if (dnz < dmin) { dmin = dnz; n = vec3<f32>(0.0, 0.0, -1.0); }
  if (dpz < dmin) { n = vec3<f32>(0.0, 0.0, 1.0); }

  return n;
}

// Returns leaf material at world position by first locating the chunk.
fn query_leaf_world(p: vec3<f32>) -> LeafQuery {
  let vs = cam.voxel_params.x;
  let chunk_size_m = f32(cam.chunk_size) * vs;

  // Convert world meters -> chunk coords
  let c = chunk_coord_from_pos(p, chunk_size_m);

  // Look up streamed chunk slot
  let slot = grid_lookup_slot(c.x, c.y, c.z);
  if (slot == INVALID_U32 || slot >= cam.chunk_count) {
    // Outside loaded grid => air
    return make_air_leaf(p, vs);
  }

  let ch = chunks[slot];

  // Build the chunk's root box in meters
  let root_bmin = vec3<f32>(f32(ch.origin.x), f32(ch.origin.y), f32(ch.origin.z)) * vs;
  let root_size = f32(cam.chunk_size) * vs;

  return query_leaf_at(p, root_bmin, root_size, ch.node_base, ch.macro_base);
}
