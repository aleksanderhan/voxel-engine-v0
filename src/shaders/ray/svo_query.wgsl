// src/shaders/ray/svo_query.wgsl
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
  node_base: u32
) -> LeafQuery {
  var idx: u32 = node_base;
  var bmin: vec3<f32> = root_bmin;
  var size: f32 = root_size;

  let min_leaf: f32 = cam.voxel_params.x;
  var p = p_in;

  for (var d: u32 = 0u; d < 32u; d = d + 1u) {
    let n = nodes[idx];

    if (n.child_base == LEAF_U32) {
      return LeafQuery(bmin, size, n.material);
    }

    if (size <= min_leaf) {
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

fn is_air(p: vec3<f32>, root_bmin: vec3<f32>, root_size: f32, node_base: u32) -> bool {
  return query_leaf_at(p, root_bmin, root_size, node_base).mat == MAT_AIR;
}

fn is_grass(p: vec3<f32>, root_bmin: vec3<f32>, root_size: f32, node_base: u32) -> bool {
  return query_leaf_at(p, root_bmin, root_size, node_base).mat == MAT_GRASS;
}
