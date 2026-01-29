// src/shaders/ray_core.wgsl
//
// Consolidated core:
// - SVO queries + hybrid traversal
// - Leaf wind + displaced hit
// - Shadows + sun transmittance
// - Material palette + shading (spec + fresnel + AO)
// - Procedural grass blades on exposed top grass voxel surfaces (FIXED: stable voxel picking across chunk boundaries)

fn safe_inv(x: f32) -> f32 {
  return select(1.0 / x, BIG_F32, abs(x) < EPS_INV);
}

// ------------------------------------------------------------
// Leaf query: point -> leaf cell in SVO
// ------------------------------------------------------------

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

// ------------------------------------------------------------
// Fast stepping: cube exit time with inv dir
// ------------------------------------------------------------

fn exit_time_from_cube_inv(
  ro: vec3<f32>,
  rd: vec3<f32>,
  inv: vec3<f32>,
  bmin: vec3<f32>,
  size: f32
) -> f32 {
  let bmax = bmin + vec3<f32>(size);

  let tx = (select(bmin.x, bmax.x, rd.x > 0.0) - ro.x) * inv.x;
  let ty = (select(bmin.y, bmax.y, rd.y > 0.0) - ro.y) * inv.y;
  let tz = (select(bmin.z, bmax.z, rd.z > 0.0) - ro.z) * inv.z;

  return min(tx, min(ty, tz));
}

// ------------------------------------------------------------
// AABB hit + stable normal selection
// ------------------------------------------------------------

struct BoxHit {
  hit : bool,
  t   : f32,
  n   : vec3<f32>,
};

fn aabb_hit_normal_inv(
  ro: vec3<f32>,
  rd: vec3<f32>,
  inv: vec3<f32>,
  bmin: vec3<f32>,
  size: f32,
  t_min: f32,
  t_max: f32
) -> BoxHit {
  let bmax = bmin + vec3<f32>(size);

  let tx0 = (bmin.x - ro.x) * inv.x;
  let tx1 = (bmax.x - ro.x) * inv.x;
  let ty0 = (bmin.y - ro.y) * inv.y;
  let ty1 = (bmax.y - ro.y) * inv.y;
  let tz0 = (bmin.z - ro.z) * inv.z;
  let tz1 = (bmax.z - ro.z) * inv.z;

  let tminx = min(tx0, tx1);
  let tmaxx = max(tx0, tx1);
  let tminy = min(ty0, ty1);
  let tmaxy = max(ty0, ty1);
  let tminz = min(tz0, tz1);
  let tmaxz = max(tz0, tz1);

  let t_enter = max(tminx, max(tminy, tminz));
  let t_exit  = min(tmaxx, min(tmaxy, tmaxz));

  let t0 = max(t_enter, t_min);

  if (t_exit < t0 || t0 > t_max) {
    return BoxHit(false, BIG_F32, vec3<f32>(0.0));
  }

  let eps = 1e-6 * size;

  var best_abs = -1.0;
  var pick: u32 = 0u;

  if (abs(t_enter - tminx) <= eps) {
    let a = abs(rd.x);
    if (a > best_abs) { best_abs = a; pick = 0u; }
  }
  if (abs(t_enter - tminy) <= eps) {
    let a = abs(rd.y);
    if (a > best_abs) { best_abs = a; pick = 1u; }
  }
  if (abs(t_enter - tminz) <= eps) {
    let a = abs(rd.z);
    if (a > best_abs) { best_abs = a; pick = 2u; }
  }

  var n = vec3<f32>(0.0);
  if (pick == 0u) { n = vec3<f32>(select( 1.0, -1.0, rd.x > 0.0), 0.0, 0.0); }
  if (pick == 1u) { n = vec3<f32>(0.0, select( 1.0, -1.0, rd.y > 0.0), 0.0); }
  if (pick == 2u) { n = vec3<f32>(0.0, 0.0, select( 1.0, -1.0, rd.z > 0.0)); }

  return BoxHit(true, t0, n);
}

// ------------------------------------------------------------
// Leaf wind field + displaced cube hit
// ------------------------------------------------------------

fn hash1(p: vec3<f32>) -> f32 {
  let h = dot(p, vec3<f32>(127.1, 311.7, 74.7));
  return fract(sin(h) * 43758.5453);
}

fn wind_field(pos_m: vec3<f32>, t: f32) -> vec3<f32> {
  let cell = floor(pos_m * WIND_CELL_FREQ);

  let ph0 = hash1(cell);
  let ph1 = hash1(cell + WIND_PHASE_OFF_1);

  let dir = normalize(WIND_DIR_XZ);

  let h = clamp((pos_m.y - WIND_RAMP_Y0) / max(WIND_RAMP_Y1 - WIND_RAMP_Y0, 1e-3), 0.0, 1.0);

  let gust = sin(
    t * WIND_GUST_TIME_FREQ +
    dot(pos_m.xz, WIND_GUST_XZ_FREQ) +
    ph0 * TAU
  );

  let flutter = sin(
    t * WIND_FLUTTER_TIME_FREQ +
    dot(pos_m.xz, WIND_FLUTTER_XZ_FREQ) +
    ph1 * TAU
  );

  let xz = dir * (WIND_GUST_WEIGHT * gust + WIND_FLUTTER_WEIGHT * flutter) * h;
  let y  = WIND_VERTICAL_SCALE * flutter * h;

  return vec3<f32>(xz.x, y, xz.y);
}

fn clamp_len(v: vec3<f32>, max_len: f32) -> vec3<f32> {
  let l2 = dot(v, v);
  if (l2 <= max_len * max_len) { return v; }
  return v * (max_len / sqrt(l2));
}

fn leaf_cube_offset(bmin: vec3<f32>, size: f32, time_s: f32, strength: f32) -> vec3<f32> {
  let center = bmin + vec3<f32>(0.5 * size);

  var w = wind_field(center, time_s) * strength;
  w = vec3<f32>(w.x, LEAF_VERTICAL_REDUCE * w.y, w.z);

  let amp = LEAF_OFFSET_AMP * size;
  return clamp_len(w * amp, LEAF_OFFSET_MAX_FRAC * size);
}

struct LeafCubeHit {
  hit  : bool,
  t    : f32,
  n    : vec3<f32>,
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
  let off   = leaf_cube_offset(bmin, size, time_s, strength);
  let bmin2 = bmin + off;

  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));
  let bh  = aabb_hit_normal_inv(ro, rd, inv, bmin2, size, t_min, t_max);

  return LeafCubeHit(bh.hit, bh.t, bh.n);
}

// ------------------------------------------------------------
// Grass helpers
// ------------------------------------------------------------

fn is_air(p: vec3<f32>, root_bmin: vec3<f32>, root_size: f32, node_base: u32) -> bool {
  return query_leaf_at(p, root_bmin, root_size, node_base).mat == MAT_AIR;
}

fn is_grass(p: vec3<f32>, root_bmin: vec3<f32>, root_size: f32, node_base: u32) -> bool {
  return query_leaf_at(p, root_bmin, root_size, node_base).mat == MAT_GRASS;
}

// --- replace pick_grass_cell_in_chunk with this version ---
struct GrassCell {
  bmin_m : vec3<f32>,  // world meters
  id_vox : vec3<f32>,  // world voxel coords as floats (stable RNG seed)
};

fn pick_grass_cell_in_chunk(
  hp_m: vec3<f32>,
  rd: vec3<f32>,
  root_bmin_m: vec3<f32>,
  ch_origin_vox: vec3<i32>,
  voxel_size_m: f32,
  chunk_size_vox: i32
) -> GrassCell {
  let root_size_m = f32(chunk_size_vox) * voxel_size_m;

  // XZ: bias slightly opposite ray to pick the visible cell at edges, then clamp to chunk.
  let bias = 0.05 * voxel_size_m;
  var local_xz = (hp_m - root_bmin_m) - rd * bias;
  local_xz.x = clamp(local_xz.x, 0.0, root_size_m - 1e-6);
  local_xz.z = clamp(local_xz.z, 0.0, root_size_m - 1e-6);

  // Y: ALWAYS choose the voxel directly *below* the hitpoint (top face hit),
  // independent of SVO leaf size.
  let y_in = hp_m.y - 1e-4 * voxel_size_m;
  var local_y = clamp(y_in - root_bmin_m.y, 0.0, root_size_m - 1e-6);

  var ix = i32(floor(local_xz.x / voxel_size_m));
  var iy = i32(floor(local_y    / voxel_size_m));
  var iz = i32(floor(local_xz.z / voxel_size_m));

  ix = clamp(ix, 0, chunk_size_vox - 1);
  iy = clamp(iy, 0, chunk_size_vox - 1);
  iz = clamp(iz, 0, chunk_size_vox - 1);

  let bmin_m = root_bmin_m + vec3<f32>(f32(ix), f32(iy), f32(iz)) * voxel_size_m;
  let id_vox = vec3<f32>(
    f32(ch_origin_vox.x + ix),
    f32(ch_origin_vox.y + iy),
    f32(ch_origin_vox.z + iz)
  );

  return GrassCell(bmin_m, id_vox);
}


// ------------------------------------------------------------
// Procedural grass SDF + raymarch (COMPLETE BODIES)
// ------------------------------------------------------------

// Signed distance to a capsule (segment AB with radius r).
fn sdf_capsule(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-8), 0.0, 1.0);
  return length(pa - ba * h) - r;
}

fn hash13(p: vec3<f32>) -> f32 {
  return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
}

// Generate a blade root inside the voxel top face (local xz in [0,1]).
fn grass_root_uv(cell_id_vox: vec3<f32>, i: u32) -> vec2<f32> {
  let fi = f32(i);
  let u = hash13(cell_id_vox + vec3<f32>(fi, 0.0, 0.0));
  let v = hash13(cell_id_vox + vec3<f32>(0.0, fi, 0.0));
  return vec2<f32>(u, v);
}

// Bending function: uses wind_field() so grass matches leaves.
fn grass_bend_offset(root_m: vec3<f32>, t: f32, height01: f32, strength: f32) -> vec3<f32> {
  let w = wind_field(root_m, t) * strength;
  let tip = w * (0.55 + 0.45 * height01);
  return vec3<f32>(tip.x, 0.15 * tip.y, tip.z);
}

// Distance to nearest grass blade in this voxel.
fn grass_sdf(
  p_m: vec3<f32>,
  cell_bmin_m: vec3<f32>,
  cell_id_vox: vec3<f32>,
  time_s: f32,
  strength: f32
) -> f32 {
  let vs = cam.voxel_params.x;

  let top_y   = cell_bmin_m.y + vs;
  let layer_h = GRASS_LAYER_HEIGHT_VOX * vs;

  // only evaluate within [top, top+layer_h]
  let y01 = (p_m.y - top_y) / max(layer_h, 1e-6);
  if (y01 < 0.0 || y01 > 1.0) { return BIG_F32; }

  let blade_len = layer_h * (0.65 + 0.35 * hash13(cell_id_vox + vec3<f32>(9.1, 3.7, 5.2)));
  let r = GRASS_BLADE_RADIUS_VOX * vs;

  var dmin = BIG_F32;

  for (var i: u32 = 0u; i < GRASS_BLADE_COUNT; i = i + 1u) {
    let uv = grass_root_uv(cell_id_vox, i);

    let inset = 0.12;
    let ux = mix(inset, 1.0 - inset, uv.x);
    let uz = mix(inset, 1.0 - inset, uv.y);

    let root = vec3<f32>(
      cell_bmin_m.x + ux * vs,
      top_y,
      cell_bmin_m.z + uz * vs
    );

    let base_tip = root + vec3<f32>(0.0, blade_len, 0.0);

    let ph = hash13(cell_id_vox + vec3<f32>(f32(i) * 7.3, 1.1, 2.9));
    let bend = grass_bend_offset(root + vec3<f32>(0.0, ph, 0.0), time_s, y01, strength);

    let tip = base_tip + bend * blade_len;

    dmin = min(dmin, sdf_capsule(p_m, root, tip, r));
  }

  return dmin;
}

fn grass_sdf_normal(
  p_m: vec3<f32>,
  cell_bmin_m: vec3<f32>,
  cell_id_vox: vec3<f32>,
  time_s: f32,
  strength: f32
) -> vec3<f32> {
  let e = 0.02 * cam.voxel_params.x;

  let dx =
    grass_sdf(p_m + vec3<f32>(e, 0.0, 0.0), cell_bmin_m, cell_id_vox, time_s, strength) -
    grass_sdf(p_m - vec3<f32>(e, 0.0, 0.0), cell_bmin_m, cell_id_vox, time_s, strength);

  let dy =
    grass_sdf(p_m + vec3<f32>(0.0, e, 0.0), cell_bmin_m, cell_id_vox, time_s, strength) -
    grass_sdf(p_m - vec3<f32>(0.0, e, 0.0), cell_bmin_m, cell_id_vox, time_s, strength);

  let dz =
    grass_sdf(p_m + vec3<f32>(0.0, 0.0, e), cell_bmin_m, cell_id_vox, time_s, strength) -
    grass_sdf(p_m - vec3<f32>(0.0, 0.0, e), cell_bmin_m, cell_id_vox, time_s, strength);

  return normalize(vec3<f32>(dx, dy, dz));
}

struct GrassHit {
  hit: bool,
  t: f32,
  n: vec3<f32>,
};

fn grass_layer_trace(
  ro: vec3<f32>,
  rd: vec3<f32>,
  t_start: f32,
  t_end: f32,
  cell_bmin_m: vec3<f32>,
  cell_id_vox: vec3<f32>,
  cell_size_m: f32,
  time_s: f32,
  strength: f32
) -> GrassHit {
  let vs = cam.voxel_params.x;
  var t = t_start;

  for (var i: u32 = 0u; i < GRASS_TRACE_STEPS; i = i + 1u) {
    if (t > t_end) { break; }

    let p = ro + rd * t;

    let d = grass_sdf(p, cell_bmin_m, cell_id_vox, time_s, strength);

    let hit_eps = GRASS_HIT_EPS_VOX * vs;
    if (d < hit_eps) {
      let n = grass_sdf_normal(p, cell_bmin_m, cell_id_vox, time_s, strength);
      return GrassHit(true, t, n);
    }

    let step_min = GRASS_STEP_MIN_VOX * vs;
    t += max(d, step_min);
  }

  return GrassHit(false, BIG_F32, vec3<f32>(0.0));
}


// ------------------------------------------------------------
// Chunk tracing: hybrid point-query + interval stepping
// ------------------------------------------------------------

struct HitGeom {
  hit      : u32,       // 0/1
  mat      : u32,
  _pad0    : u32,
  _pad1    : u32,

  t        : f32,
  _pad2    : vec3<f32>,

  n        : vec3<f32>,
  _pad3    : f32,

  // AO sampling in same chunk
  root_bmin : vec3<f32>,
  root_size : f32,
  node_base : u32,
  _pad4     : vec3<u32>,
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
  return h;
}

fn trace_chunk_hybrid_interval(
  ro: vec3<f32>,
  rd: vec3<f32>,
  ch: ChunkMeta,
  t_enter: f32,
  t_exit: f32
) -> HitGeom {
  let voxel_size = cam.voxel_params.x;

  let root_bmin_vox = vec3<f32>(f32(ch.origin.x), f32(ch.origin.y), f32(ch.origin.z));
  let root_bmin = root_bmin_vox * voxel_size;
  let root_size = f32(cam.chunk_size) * voxel_size;

  let eps_step = 1e-4 * voxel_size;
  var tcur = max(t_enter, 0.0) + eps_step;

  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  for (var step_i: u32 = 0u; step_i < cam.max_steps; step_i = step_i + 1u) {
    if (tcur > t_exit) { break; }

    let p  = ro + tcur * rd;
    let pq = p + rd * (1e-4 * voxel_size);

    let q = query_leaf_at(pq, root_bmin, root_size, ch.node_base);

    if (q.mat != MAT_AIR) {
      if (q.mat == MAT_LEAF) {
        let time_s   = cam.voxel_params.y;
        let strength = cam.voxel_params.z;

        let h2 = leaf_displaced_cube_hit(
          ro, rd,
          q.bmin, q.size,
          time_s, strength,
          t_enter,
          t_exit
        );

        if (h2.hit) {
          var out = miss_hitgeom();
          out.hit = 1u;
          out.t   = h2.t;
          out.mat = MAT_LEAF;
          out.n   = h2.n;
          out.root_bmin = root_bmin;
          out.root_size = root_size;
          out.node_base = ch.node_base;
          return out;
        }

        let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
        tcur = max(t_leave, tcur) + eps_step;
        continue;
      }

      let bh = aabb_hit_normal_inv(
        ro, rd, inv,
        q.bmin, q.size,
        t_enter,
        t_exit
      );

      if (bh.hit) {
        // --- Procedural blades refinement for GRASS (SLAB INTERSECTION: works from side views) ---
        if (q.mat == MAT_GRASS) {
          let time_s   = cam.voxel_params.y;
          let strength = cam.voxel_params.z;
          let vs       = cam.voxel_params.x;

          let hp = ro + bh.t * rd;

          // Pick voxel XZ under the hit (chunk-local, clamped)
          let bias = 0.05 * vs;
          var local = (hp - root_bmin) - rd * bias;
          local.x = clamp(local.x, 0.0, root_size - 1e-6);
          local.z = clamp(local.z, 0.0, root_size - 1e-6);

          var ix = i32(floor(local.x / vs));
          var iz = i32(floor(local.z / vs));

          ix = clamp(ix, 0, i32(cam.chunk_size) - 1);
          iz = clamp(iz, 0, i32(cam.chunk_size) - 1);

          // Y: use the TOP of the *grass leaf cube* (q.bmin+q.size), then take the voxel just under it.
          // This avoids needing "air above" queries that break with coarse leaves.
          let y_leaf_top = q.bmin.y + q.size;
          let local_y_top = clamp((y_leaf_top - 1e-4 * vs) - root_bmin.y, 0.0, root_size - 1e-6);
          var iy = i32(floor(local_y_top / vs));
          iy = clamp(iy, 0, i32(cam.chunk_size) - 1);

          let cell_bmin = root_bmin + vec3<f32>(f32(ix), f32(iy), f32(iz)) * vs;

          // stable RNG seed = global voxel id
          let cell_id_vox = vec3<f32>(
            f32(ch.origin.x + ix),
            f32(ch.origin.y + iy),
            f32(ch.origin.z + iz)
          );

          // Build the grass slab AABB: [top of voxel .. top+layer_h] over this voxel's XZ
          let layer_h = GRASS_LAYER_HEIGHT_VOX * vs;

          let slab_bmin = vec3<f32>(cell_bmin.x, cell_bmin.y + vs, cell_bmin.z);
          let slab_bmax = vec3<f32>(cell_bmin.x + vs, cell_bmin.y + vs + layer_h, cell_bmin.z + vs);

          // Intersect ray with slab, and only march the portion that is BEFORE the solid hit (bh.t)
          let rt = intersect_aabb(ro, rd, slab_bmin, slab_bmax);
          var t0 = max(rt.x, t_enter);
          var t1 = min(rt.y, bh.t - 0.01 * vs); // keep strictly in front of voxel surface hit

          if (t1 > t0) {
            let gh = grass_layer_trace(ro, rd, t0, t1, cell_bmin, cell_id_vox, vs, time_s, strength);
            if (gh.hit) {
              var out = miss_hitgeom();
              out.hit = 1u;
              out.t   = gh.t;
              out.mat = MAT_GRASS;
              out.n   = gh.n;
              out.root_bmin = root_bmin;
              out.root_size = root_size;
              out.node_base = ch.node_base;
              return out;
            }
          }
        }




        // Fall back to voxel face hit
        var out = miss_hitgeom();
        out.hit = 1u;
        out.t   = bh.t;
        out.mat = q.mat;
        out.n   = bh.n;
        out.root_bmin = root_bmin;
        out.root_size = root_size;
        out.node_base = ch.node_base;
        return out;
      }

      let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
      tcur = max(t_leave, tcur) + eps_step;
      continue;
    }

    let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
    tcur = max(t_leave, tcur) + eps_step;
  }

  return miss_hitgeom();
}

// ------------------------------------------------------------
// Shadow traversal
// ------------------------------------------------------------

fn trace_chunk_shadow_interval(
  ro: vec3<f32>,
  rd: vec3<f32>,
  ch: ChunkMeta,
  t_enter: f32,
  t_exit: f32
) -> bool {
  let voxel_size = cam.voxel_params.x;
  let nudge_s = 0.18 * voxel_size;

  let root_bmin_vox = vec3<f32>(f32(ch.origin.x), f32(ch.origin.y), f32(ch.origin.z));
  let root_bmin = root_bmin_vox * voxel_size;
  let root_size = f32(cam.chunk_size) * voxel_size;

  var tcur = max(t_enter, 0.0) + nudge_s;
  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  for (var step_i: u32 = 0u; step_i < SHADOW_STEPS; step_i = step_i + 1u) {
    if (tcur > t_exit) { break; }

    let p = ro + tcur * rd;
    let q = query_leaf_at(p, root_bmin, root_size, ch.node_base);

    if (q.mat != MAT_AIR) {
      if (q.mat == MAT_LEAF) {
        if (!SHADOW_DISPLACED_LEAVES) {
          return true;
        }

        let time_s   = cam.voxel_params.y;
        let strength = cam.voxel_params.z;

        let h2 = leaf_displaced_cube_hit(
          ro, rd,
          q.bmin, q.size,
          time_s, strength,
          tcur - nudge_s,
          t_exit
        );

        if (h2.hit) { return true; }

        let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
        tcur = max(t_leave, tcur) + nudge_s;
        continue;
      }

      return true;
    }

    let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
    tcur = max(t_leave, tcur) + nudge_s;
  }

  return false;
}

fn in_shadow(p: vec3<f32>, sun_dir: vec3<f32>) -> bool {
  let voxel_size   = cam.voxel_params.x;
  let nudge_s      = 0.18 * voxel_size;
  let chunk_size_m = f32(cam.chunk_size) * voxel_size;

  let go = cam.grid_origin_chunk;
  let gd = cam.grid_dims;

  let grid_bmin = vec3<f32>(
    f32(go.x) * chunk_size_m,
    f32(go.y) * chunk_size_m,
    f32(go.z) * chunk_size_m
  );

  let grid_bmax = grid_bmin + vec3<f32>(
    f32(gd.x) * chunk_size_m,
    f32(gd.y) * chunk_size_m,
    f32(gd.z) * chunk_size_m
  );

  let bias = max(SHADOW_BIAS, 0.50 * voxel_size);
  let ro   = p + sun_dir * bias;
  let rd   = sun_dir;

  let rtg = intersect_aabb(ro, rd, grid_bmin, grid_bmax);
  let t_enter = max(rtg.x, 0.0);
  let t_exit  = rtg.y;
  if (t_exit < t_enter) { return false; }

  let start_t = t_enter + nudge_s;
  let p0 = ro + start_t * rd;

  var t_local: f32 = 0.0;
  let t_exit_local = max(t_exit - start_t, 0.0);

  var c = chunk_coord_from_pos(p0, chunk_size_m);
  var cx: i32 = c.x;
  var cy: i32 = c.y;
  var cz: i32 = c.z;

  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  let step_x: i32 = select(-1, 1, rd.x > 0.0);
  let step_y: i32 = select(-1, 1, rd.y > 0.0);
  let step_z: i32 = select(-1, 1, rd.z > 0.0);

  let bx = select(f32(cx) * chunk_size_m, f32(cx + 1) * chunk_size_m, rd.x > 0.0);
  let by = select(f32(cy) * chunk_size_m, f32(cy + 1) * chunk_size_m, rd.y > 0.0);
  let bz = select(f32(cz) * chunk_size_m, f32(cz + 1) * chunk_size_m, rd.z > 0.0);

  var tMaxX: f32 = (bx - p0.x) * inv.x;
  var tMaxY: f32 = (by - p0.y) * inv.y;
  var tMaxZ: f32 = (bz - p0.z) * inv.z;

  let tDeltaX: f32 = abs(chunk_size_m * inv.x);
  let tDeltaY: f32 = abs(chunk_size_m * inv.y);
  let tDeltaZ: f32 = abs(chunk_size_m * inv.z);

  if (abs(rd.x) < EPS_INV) { tMaxX = BIG_F32; }
  if (abs(rd.y) < EPS_INV) { tMaxY = BIG_F32; }
  if (abs(rd.z) < EPS_INV) { tMaxZ = BIG_F32; }

  let max_chunk_steps = min((gd.x + gd.y + gd.z) * 6u + 8u, 1024u);

  for (var s: u32 = 0u; s < max_chunk_steps; s = s + 1u) {
    if (t_local > t_exit_local) { break; }

    let tNextLocal = min(tMaxX, min(tMaxY, tMaxZ));

    let slot = grid_lookup_slot(cx, cy, cz);
    if (slot != INVALID_U32 && slot < cam.chunk_count) {
      let ch2 = chunks[slot];

      let cell_enter = start_t + t_local;
      let cell_exit2 = start_t + min(tNextLocal, t_exit_local);

      if (trace_chunk_shadow_interval(ro, rd, ch2, cell_enter, cell_exit2)) {
        return true;
      }
    }

    if (tMaxX < tMaxY) {
      if (tMaxX < tMaxZ) { cx += step_x; t_local = tMaxX; tMaxX += tDeltaX; }
      else               { cz += step_z; t_local = tMaxZ; tMaxZ += tDeltaZ; }
    } else {
      if (tMaxY < tMaxZ) { cy += step_y; t_local = tMaxY; tMaxY += tDeltaY; }
      else               { cz += step_z; t_local = tMaxZ; tMaxZ += tDeltaZ; }
    }

    let ox = cam.grid_origin_chunk.x;
    let oy = cam.grid_origin_chunk.y;
    let oz = cam.grid_origin_chunk.z;

    let nx = i32(cam.grid_dims.x);
    let ny = i32(cam.grid_dims.y);
    let nz = i32(cam.grid_dims.z);

    if (cx < ox || cy < oy || cz < oz || cx >= ox + nx || cy >= oy + ny || cz >= oz + nz) {
      break;
    }
  }

  return false;
}

// Transmittance step inside a chunk (respects displaced leaves option)
fn trace_chunk_shadow_trans_interval(
  ro: vec3<f32>,
  rd: vec3<f32>,
  ch: ChunkMeta,
  t_enter: f32,
  t_exit: f32
) -> f32 {
  let voxel_size = cam.voxel_params.x;
  let nudge_s = 0.18 * voxel_size;

  let root_bmin_vox = vec3<f32>(f32(ch.origin.x), f32(ch.origin.y), f32(ch.origin.z));
  let root_bmin = root_bmin_vox * voxel_size;
  let root_size = f32(cam.chunk_size) * voxel_size;

  var tcur = max(t_enter, 0.0) + nudge_s;
  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  var trans = 1.0;

  for (var step_i: u32 = 0u; step_i < VSM_STEPS; step_i = step_i + 1u) {
    if (tcur > t_exit) { break; }
    if (trans < MIN_TRANS) { break; }

    let p = ro + tcur * rd;
    let qeps = 1e-4 * cam.voxel_params.x;
    let pq   = p + rd * qeps;

    let q = query_leaf_at(pq, root_bmin, root_size, ch.node_base);

    if (q.mat != MAT_AIR) {
      if (q.mat == MAT_LEAF) {
        if (VOLUME_DISPLACED_LEAVES) {
          let time_s   = cam.voxel_params.y;
          let strength = cam.voxel_params.z;

          let h2 = leaf_displaced_cube_hit(
            ro, rd,
            q.bmin, q.size,
            time_s, strength,
            tcur - nudge_s,
            t_exit
          );

          if (h2.hit) { trans *= LEAF_LIGHT_TRANSMIT; }

          let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
          tcur = max(t_leave, tcur) + nudge_s;
          continue;
        } else {
          trans *= LEAF_LIGHT_TRANSMIT;
          let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
          tcur = max(t_leave, tcur) + nudge_s;
          continue;
        }
      }

      if (q.mat == MAT_GRASS) {
        trans *= GRASS_LIGHT_TRANSMIT;
        let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
        tcur = max(t_leave, tcur) + nudge_s;
        continue;
      }

      return 0.0;
    }

    let t_leave = exit_time_from_cube_inv(ro, rd, inv, q.bmin, q.size);
    tcur = max(t_leave, tcur) + nudge_s;
  }

  return trans;
}

fn sun_transmittance_geom_only(p: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
  let voxel_size   = cam.voxel_params.x;
  let nudge_s      = 0.18 * voxel_size;
  let chunk_size_m = f32(cam.chunk_size) * voxel_size;

  let go = cam.grid_origin_chunk;
  let gd = cam.grid_dims;

  let grid_bmin = vec3<f32>(
    f32(go.x) * chunk_size_m,
    f32(go.y) * chunk_size_m,
    f32(go.z) * chunk_size_m
  );

  let grid_bmax = grid_bmin + vec3<f32>(
    f32(gd.x) * chunk_size_m,
    f32(gd.y) * chunk_size_m,
    f32(gd.z) * chunk_size_m
  );

  let bias = max(SHADOW_BIAS, 0.50 * voxel_size);
  let ro   = p + sun_dir * bias;
  let rd   = sun_dir;

  let rtg = intersect_aabb(ro, rd, grid_bmin, grid_bmax);
  let t_enter = max(rtg.x, 0.0);
  let t_exit  = rtg.y;
  if (t_exit < t_enter) { return 1.0; }

  let start_t = t_enter + nudge_s;
  let p0 = ro + start_t * rd;

  var t_local: f32 = 0.0;
  let t_exit_local = max(t_exit - start_t, 0.0);

  var c = chunk_coord_from_pos(p0, chunk_size_m);
  var cx: i32 = c.x;
  var cy: i32 = c.y;
  var cz: i32 = c.z;

  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  let step_x: i32 = select(-1, 1, rd.x > 0.0);
  let step_y: i32 = select(-1, 1, rd.y > 0.0);
  let step_z: i32 = select(-1, 1, rd.z > 0.0);

  let bx = select(f32(cx) * chunk_size_m, f32(cx + 1) * chunk_size_m, rd.x > 0.0);
  let by = select(f32(cy) * chunk_size_m, f32(cy + 1) * chunk_size_m, rd.y > 0.0);
  let bz = select(f32(cz) * chunk_size_m, f32(cz + 1) * chunk_size_m, rd.z > 0.0);

  var tMaxX: f32 = (bx - p0.x) * inv.x;
  var tMaxY: f32 = (by - p0.y) * inv.y;
  var tMaxZ: f32 = (bz - p0.z) * inv.z;

  let tDeltaX: f32 = abs(chunk_size_m * inv.x);
  let tDeltaY: f32 = abs(chunk_size_m * inv.y);
  let tDeltaZ: f32 = abs(chunk_size_m * inv.z);

  if (abs(rd.x) < EPS_INV) { tMaxX = BIG_F32; }
  if (abs(rd.y) < EPS_INV) { tMaxY = BIG_F32; }
  if (abs(rd.z) < EPS_INV) { tMaxZ = BIG_F32; }

  var trans = 1.0;

  let max_chunk_steps = min((gd.x + gd.y + gd.z) * 6u + 8u, 512u);

  for (var s: u32 = 0u; s < max_chunk_steps; s = s + 1u) {
    if (t_local > t_exit_local) { break; }
    if (trans < MIN_TRANS) { break; }

    let tNextLocal = min(tMaxX, min(tMaxY, tMaxZ));
    let slot = grid_lookup_slot(cx, cy, cz);

    if (slot != INVALID_U32 && slot < cam.chunk_count) {
      let ch2 = chunks[slot];

      let cell_enter = start_t + t_local;
      let cell_exit2 = start_t + min(tNextLocal, t_exit_local);

      trans *= trace_chunk_shadow_trans_interval(ro, rd, ch2, cell_enter, cell_exit2);
      if (trans < MIN_TRANS) { break; }
    }

    if (tMaxX < tMaxY) {
      if (tMaxX < tMaxZ) { cx += step_x; t_local = tMaxX; tMaxX += tDeltaX; }
      else               { cz += step_z; t_local = tMaxZ; tMaxZ += tDeltaZ; }
    } else {
      if (tMaxY < tMaxZ) { cy += step_y; t_local = tMaxY; tMaxY += tDeltaY; }
      else               { cz += step_z; t_local = tMaxZ; tMaxZ += tDeltaZ; }
    }

    let ox = cam.grid_origin_chunk.x;
    let oy = cam.grid_origin_chunk.y;
    let oz = cam.grid_origin_chunk.z;

    let nx = i32(cam.grid_dims.x);
    let ny = i32(cam.grid_dims.y);
    let nz = i32(cam.grid_dims.z);

    if (cx < ox || cy < oy || cz < oz || cx >= ox + nx || cy >= oy + ny || cz >= oz + nz) {
      break;
    }
  }

  return trans;
}

fn sun_transmittance(p: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
  let Tc = cloud_sun_transmittance(p, sun_dir);

  let voxel_size   = cam.voxel_params.x;
  let nudge_s      = 0.18 * voxel_size;
  let chunk_size_m = f32(cam.chunk_size) * voxel_size;

  let go = cam.grid_origin_chunk;
  let gd = cam.grid_dims;

  let grid_bmin = vec3<f32>(
    f32(go.x) * chunk_size_m,
    f32(go.y) * chunk_size_m,
    f32(go.z) * chunk_size_m
  );

  let grid_bmax = grid_bmin + vec3<f32>(
    f32(gd.x) * chunk_size_m,
    f32(gd.y) * chunk_size_m,
    f32(gd.z) * chunk_size_m
  );

  let bias = max(SHADOW_BIAS, 0.50 * voxel_size);
  let ro   = p + sun_dir * bias;
  let rd   = sun_dir;

  let rtg = intersect_aabb(ro, rd, grid_bmin, grid_bmax);
  let t_enter = max(rtg.x, 0.0);
  let t_exit  = rtg.y;
  if (t_exit < t_enter) { return Tc; }

  let start_t = t_enter + nudge_s;
  let p0 = ro + start_t * rd;

  var t_local: f32 = 0.0;
  let t_exit_local = max(t_exit - start_t, 0.0);

  var c = chunk_coord_from_pos(p0, chunk_size_m);
  var cx: i32 = c.x;
  var cy: i32 = c.y;
  var cz: i32 = c.z;

  let inv = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));

  let step_x: i32 = select(-1, 1, rd.x > 0.0);
  let step_y: i32 = select(-1, 1, rd.y > 0.0);
  let step_z: i32 = select(-1, 1, rd.z > 0.0);

  let bx = select(f32(cx) * chunk_size_m, f32(cx + 1) * chunk_size_m, rd.x > 0.0);
  let by = select(f32(cy) * chunk_size_m, f32(cy + 1) * chunk_size_m, rd.y > 0.0);
  let bz = select(f32(cz) * chunk_size_m, f32(cz + 1) * chunk_size_m, rd.z > 0.0);

  var tMaxX: f32 = (bx - p0.x) * inv.x;
  var tMaxY: f32 = (by - p0.y) * inv.y;
  var tMaxZ: f32 = (bz - p0.z) * inv.z;

  let tDeltaX: f32 = abs(chunk_size_m * inv.x);
  let tDeltaY: f32 = abs(chunk_size_m * inv.y);
  let tDeltaZ: f32 = abs(chunk_size_m * inv.z);

  if (abs(rd.x) < EPS_INV) { tMaxX = BIG_F32; }
  if (abs(rd.y) < EPS_INV) { tMaxY = BIG_F32; }
  if (abs(rd.z) < EPS_INV) { tMaxZ = BIG_F32; }

  var trans = 1.0;

  let max_chunk_steps = min((gd.x + gd.y + gd.z) * 6u + 8u, 512u);

  for (var s: u32 = 0u; s < max_chunk_steps; s = s + 1u) {
    if (t_local > t_exit_local) { break; }
    if (trans < MIN_TRANS) { break; }

    let tNextLocal = min(tMaxX, min(tMaxY, tMaxZ));
    let slot = grid_lookup_slot(cx, cy, cz);

    if (slot != INVALID_U32 && slot < cam.chunk_count) {
      let ch2 = chunks[slot];

      let cell_enter = start_t + t_local;
      let cell_exit2 = start_t + min(tNextLocal, t_exit_local);

      trans *= trace_chunk_shadow_trans_interval(ro, rd, ch2, cell_enter, cell_exit2);
      if (trans < MIN_TRANS) { break; }
    }

    if (tMaxX < tMaxY) {
      if (tMaxX < tMaxZ) { cx += step_x; t_local = tMaxX; tMaxX += tDeltaX; }
      else               { cz += step_z; t_local = tMaxZ; tMaxZ += tDeltaZ; }
    } else {
      if (tMaxY < tMaxZ) { cy += step_y; t_local = tMaxY; tMaxY += tDeltaY; }
      else               { cz += step_z; t_local = tMaxZ; tMaxZ += tDeltaZ; }
    }

    let ox = cam.grid_origin_chunk.x;
    let oy = cam.grid_origin_chunk.y;
    let oz = cam.grid_origin_chunk.z;

    let nx = i32(cam.grid_dims.x);
    let ny = i32(cam.grid_dims.y);
    let nz = i32(cam.grid_dims.z);

    if (cx < ox || cy < oy || cz < oz || cx >= ox + nx || cy >= oy + ny || cz >= oz + nz) {
      break;
    }
  }

  return trans * Tc;
}

// ------------------------------------------------------------
// Shading (spec + fresnel + AO)
// ------------------------------------------------------------

fn color_for_material(m: u32) -> vec3<f32> {
  if (m == MAT_AIR)   { return vec3<f32>(0.0); }

  if (m == MAT_GRASS) { return vec3<f32>(0.18, 0.75, 0.18); }
  if (m == MAT_DIRT)  { return vec3<f32>(0.45, 0.30, 0.15); }
  if (m == MAT_STONE) { return vec3<f32>(0.50, 0.50, 0.55); }
  if (m == MAT_WOOD)  { return vec3<f32>(0.38, 0.26, 0.14); }
  if (m == MAT_LEAF)  { return vec3<f32>(0.10, 0.55, 0.12); }

  return vec3<f32>(1.0, 0.0, 1.0);
}

fn hemi_ambient(n: vec3<f32>) -> vec3<f32> {
  let upw = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
  let sky = sky_color(vec3<f32>(0.0, 1.0, 0.0));
  let grd = FOG_COLOR_GROUND;
  return mix(grd, sky, upw);
}

fn hash31(p: vec3<f32>) -> f32 {
  let h = dot(p, vec3<f32>(127.1, 311.7, 74.7));
  return fract(sin(h) * 43758.5453);
}

fn material_variation(world_p: vec3<f32>, cell_size_m: f32) -> f32 {
  let cell = floor(world_p / cell_size_m);
  return (hash31(cell) - 0.5) * 2.0; // [-1,+1]
}

fn apply_material_variation(base: vec3<f32>, mat: u32, hp: vec3<f32>) -> vec3<f32> {
  var c = base;

  let v = material_variation(hp, 0.05);

  if (mat == MAT_GRASS) {
    c += vec3<f32>(0.02 * v, 0.05 * v, 0.01 * v);
    c *= (1.0 + 0.06 * v);
  } else if (mat == MAT_DIRT) {
    c += vec3<f32>(0.04 * v, 0.02 * v, 0.01 * v);
    c *= (1.0 + 0.08 * v);
  } else if (mat == MAT_STONE) {
    c *= (1.0 + 0.10 * v);
  } else if (mat == MAT_WOOD) {
    c += vec3<f32>(0.05 * v, 0.02 * v, 0.00 * v);
    c *= (1.0 + 0.07 * v);
  } else if (mat == MAT_LEAF) {
    c += vec3<f32>(0.00 * v, 0.03 * v, 0.00 * v);
    c *= (1.0 + 0.04 * v);
  }

  return clamp(c, vec3<f32>(0.0), vec3<f32>(1.5));
}

// Cheap local AO around hit point (6 taps)
fn voxel_ao_local(
  hp: vec3<f32>,
  n: vec3<f32>,
  root_bmin: vec3<f32>,
  root_size: f32,
  node_base: u32
) -> f32 {
  let r = 0.75 * cam.voxel_params.x;

  let up_ref = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
  let t = normalize(cross(up_ref, n));
  let b = normalize(cross(n, t));

  var occ = 0.0;

  let q0 = query_leaf_at(hp + t * r, root_bmin, root_size, node_base);
  occ += select(0.0, 1.0, q0.mat != MAT_AIR);

  let q1 = query_leaf_at(hp - t * r, root_bmin, root_size, node_base);
  occ += select(0.0, 1.0, q1.mat != MAT_AIR);

  let q2 = query_leaf_at(hp + b * r, root_bmin, root_size, node_base);
  occ += select(0.0, 1.0, q2.mat != MAT_AIR);

  let q3 = query_leaf_at(hp - b * r, root_bmin, root_size, node_base);
  occ += select(0.0, 1.0, q3.mat != MAT_AIR);

  let h0 = normalize(n + 0.65 * t + 0.35 * b);
  let q4 = query_leaf_at(hp + h0 * r, root_bmin, root_size, node_base);
  occ += select(0.0, 1.0, q4.mat != MAT_AIR);

  let h1 = normalize(n - 0.65 * t + 0.35 * b);
  let q5 = query_leaf_at(hp + h1 * r, root_bmin, root_size, node_base);
  occ += select(0.0, 1.0, q5.mat != MAT_AIR);

  let occ_n = occ * (1.0 / 6.0);
  return clamp(1.0 - 0.70 * occ_n, 0.35, 1.0);
}

fn fresnel_schlick(ndv: f32, f0: f32) -> f32 {
  return f0 + (1.0 - f0) * pow(1.0 - clamp(ndv, 0.0, 1.0), 5.0);
}

fn material_roughness(mat: u32) -> f32 {
  if (mat == MAT_STONE) { return 0.45; }
  if (mat == MAT_WOOD)  { return 0.70; }
  if (mat == MAT_LEAF)  { return 0.80; }
  if (mat == MAT_GRASS) { return 0.85; }
  if (mat == MAT_DIRT)  { return 0.90; }
  return 0.90;
}

fn material_f0(mat: u32) -> f32 {
  if (mat == MAT_STONE) { return 0.04; }
  if (mat == MAT_WOOD)  { return 0.03; }
  if (mat == MAT_LEAF)  { return 0.05; }
  if (mat == MAT_GRASS) { return 0.04; }
  if (mat == MAT_DIRT)  { return 0.02; }
  return 0.02;
}

fn shade_hit(ro: vec3<f32>, rd: vec3<f32>, hg: HitGeom) -> vec3<f32> {
  let hp = ro + hg.t * rd;

  var base = color_for_material(hg.mat);
  base = apply_material_variation(base, hg.mat, hp);

  if (hg.mat == MAT_GRASS) {
    let vs = cam.voxel_params.x;
    let tip = clamp(fract(hp.y / max(vs, 1e-6)), 0.0, 1.0);

    base = mix(base, base + vec3<f32>(0.10, 0.10, 0.02), 0.35 * tip);

    let back = pow(clamp(dot(-SUN_DIR, hg.n), 0.0, 1.0), 2.0);
    base += 0.18 * back * vec3<f32>(0.20, 0.35, 0.08);
  }

  let vs = cam.voxel_params.x;
  let hp_shadow  = hp + hg.n * (0.75 * vs);

  let vis = sun_transmittance(hp_shadow, SUN_DIR);
  let diff = max(dot(hg.n, SUN_DIR), 0.0);

  let ao = select(1.0, voxel_ao_local(hp, hg.n, hg.root_bmin, hg.root_size, hg.node_base), hg.hit != 0u);

  let amb_col = hemi_ambient(hg.n);
  let amb_strength = select(0.10, 0.14, hg.mat == MAT_LEAF);
  let ambient = amb_col * amb_strength * ao;

  var dapple = 1.0;
  if (hg.mat == MAT_LEAF) {
    let time_s = cam.voxel_params.y;
    let d0 = sin(dot(hp.xz, vec2<f32>(3.0, 2.2)) + time_s * 3.5);
    let d1 = sin(dot(hp.xz, vec2<f32>(6.5, 4.1)) - time_s * 6.0);
    dapple = 0.90 + 0.10 * (0.6 * d0 + 0.4 * d1);
  }

  let direct = SUN_COLOR * SUN_INTENSITY * (diff * diff) * vis * dapple;

  let v = normalize(-rd);
  let h = normalize(v + SUN_DIR);

  let ndv = max(dot(hg.n, v), 0.0);
  let ndh = max(dot(hg.n, h), 0.0);

  let rough = material_roughness(hg.mat);
  let shininess = mix(8.0, 96.0, 1.0 - rough);
  let spec = pow(ndh, shininess);

  let f0 = material_f0(hg.mat);
  let fres = fresnel_schlick(ndv, f0);

  let spec_col = SUN_COLOR * SUN_INTENSITY * spec * fres * vis;

  return base * (ambient + direct) + 0.20 * spec_col;
}

const GRASS_Y_SCAN_STEPS: u32 = 12u;

struct GrassCellPick {
  ok       : bool,
  bmin_m   : vec3<f32>,
  id_vox   : vec3<f32>,
};

fn find_exposed_grass_cell_in_chunk(
  hp_m: vec3<f32>,
  rd: vec3<f32>,
  root_bmin_m: vec3<f32>,
  root_size_m: f32,
  ch_origin_vox: vec3<i32>,
  voxel_size_m: f32,
  chunk_size_vox: i32,
  node_base: u32
) -> GrassCellPick {
  // Stable XZ pick (chunk-local, clamped so it never spills to neighbor chunks)
  let bias = 0.05 * voxel_size_m;
  var local = (hp_m - root_bmin_m) - rd * bias;

  local.x = clamp(local.x, 0.0, root_size_m - 1e-6);
  local.z = clamp(local.z, 0.0, root_size_m - 1e-6);

  var ix = i32(floor(local.x / voxel_size_m));
  var iz = i32(floor(local.z / voxel_size_m));

  ix = clamp(ix, 0, chunk_size_vox - 1);
  iz = clamp(iz, 0, chunk_size_vox - 1);

  // Start Y from just below the hitpoint (but don’t trust SVO leaf size)
  var local_y = clamp((hp_m.y - 1e-4 * voxel_size_m) - root_bmin_m.y, 0.0, root_size_m - 1e-6);
  var iy0 = i32(floor(local_y / voxel_size_m));
  iy0 = clamp(iy0, 0, chunk_size_vox - 1);

  // Scan downward to find the topmost exposed grass voxel: grass below + air above
  for (var k: u32 = 0u; k < GRASS_Y_SCAN_STEPS; k = k + 1u) {
    let iy = iy0 - i32(k);
    if (iy < 0) { break; }

    let bmin_m = root_bmin_m + vec3<f32>(f32(ix), f32(iy), f32(iz)) * voxel_size_m;

    let below_c = bmin_m + vec3<f32>(0.5 * voxel_size_m, 0.5 * voxel_size_m, 0.5 * voxel_size_m);
    let above_c = bmin_m + vec3<f32>(0.5 * voxel_size_m, 1.5 * voxel_size_m, 0.5 * voxel_size_m);

    if (is_grass(below_c, root_bmin_m, root_size_m, node_base) &&
        is_air  (above_c, root_bmin_m, root_size_m, node_base)) {

      let id_vox = vec3<f32>(
        f32(ch_origin_vox.x + ix),
        f32(ch_origin_vox.y + iy),
        f32(ch_origin_vox.z + iz)
      );

      return GrassCellPick(true, bmin_m, id_vox);
    }
  }

  return GrassCellPick(false, vec3<f32>(0.0), vec3<f32>(0.0));
}
