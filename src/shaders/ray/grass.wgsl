



struct GrassCell {
  bmin_m : vec3<f32>,
  id_vox : vec3<f32>,
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

  let bias = 0.05 * voxel_size_m;
  var local_xz = (hp_m - root_bmin_m) - rd * bias;
  local_xz.x = clamp(local_xz.x, 0.0, root_size_m - 1e-6);
  local_xz.z = clamp(local_xz.z, 0.0, root_size_m - 1e-6);

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

fn sdf_box(p: vec3<f32>, c: vec3<f32>, b: vec3<f32>) -> f32 {
  let q = abs(p - c) - b;
  let outside = length(max(q, vec3<f32>(0.0)));
  let inside  = min(max(q.x, max(q.y, q.z)), 0.0);
  return outside + inside;
}

fn grass_root_uv(cell_id_vox: vec3<f32>, i: u32) -> vec2<f32> {
  let fi = f32(i);
  let u = hash31(cell_id_vox + vec3<f32>(fi, 0.0, 0.0));
  let v = hash31(cell_id_vox + vec3<f32>(0.0, fi, 0.0));
  return vec2<f32>(u, v);
}

fn grass_wind_xz(root_m: vec3<f32>, t: f32, strength: f32) -> vec2<f32> {
  
  let w = wind_field(root_m, t) * strength;
  return vec2<f32>(w.x, w.z);
}


fn grass_sdf_lod(
  p_m: vec3<f32>,
  cell_bmin_m: vec3<f32>,
  cell_id_vox: vec3<f32>,
  time_s: f32,
  strength: f32,
  lod: u32
) -> f32 {
  let vs = cam.voxel_params.x;

  let top_y   = cell_bmin_m.y + vs;
  let layer_h = GRASS_LAYER_HEIGHT_VOX * vs;

  
  let y01 = (p_m.y - top_y) / max(layer_h, 1e-6);
  if (y01 < 0.0 || y01 > 1.0) { return BIG_F32; }

  
  let over = GRASS_OVERHANG_VOX * vs;
  if (p_m.x < cell_bmin_m.x - over || p_m.x > cell_bmin_m.x + vs + over ||
      p_m.z < cell_bmin_m.z - over || p_m.z > cell_bmin_m.z + vs + over) {
    return BIG_F32;
  }

  let blade_len = layer_h * (0.65 + 0.35 * hash31(cell_id_vox + vec3<f32>(9.1, 3.7, 5.2)));

  
  var blade_count: u32 = GRASS_BLADE_COUNT;
  var segs: u32 = u32(max(3.0, floor(GRASS_VOXEL_SEGS)));
  if (lod == 1u) {
    blade_count = min(blade_count, GRASS_BLADE_COUNT_MID);
    segs = min(segs, GRASS_SEGS_MID);
  } else if (lod == 2u) {
    blade_count = min(blade_count, GRASS_BLADE_COUNT_FAR);
    segs = min(segs, GRASS_SEGS_FAR);
  }

  let half_xz = GRASS_VOXEL_THICKNESS_VOX * vs;
  let inv_segs = 1.0 / max(f32(segs), 1.0);
  let seg_h   = blade_len * inv_segs;
  let half_y  = 0.5 * seg_h;

  var dmin = BIG_F32;

  
  let inset = 0.12;

  for (var i: u32 = 0u; i < blade_count; i = i + 1u) {
    
    let uvp = grass_blade_params(cell_id_vox, i);

    let ux = mix(inset, 1.0 - inset, uvp.x);
    let uz = mix(inset, 1.0 - inset, uvp.y);

    let root = vec3<f32>(
      cell_bmin_m.x + ux * vs,
      top_y,
      cell_bmin_m.z + uz * vs
    );

    
    let ph = uvp.z;

    
    let w_xz = grass_wind_xz(root + vec3<f32>(0.0, ph, 0.0), time_s, strength);

    
    for (var s: u32 = 0u; s < segs; s = s + 1u) {
      let t01 = (f32(s) + 0.5) * inv_segs;
      let y   = t01 * blade_len;

      
      
      
      let height_factor = (0.55 + 0.45 * t01);
      let bend_mag      = (blade_len * t01) * height_factor;

      let off_xz = w_xz * bend_mag;

      let c = root + vec3<f32>(off_xz.x, y, off_xz.y);

      let taper = mix(1.0, GRASS_VOXEL_TAPER, t01);
      let bxz = half_xz * taper;
      let b = vec3<f32>(bxz, half_y, bxz);

      dmin = min(dmin, sdf_box(p_m, c, b));
    }
  }

  return dmin;
}


fn grass_blade_params(cell_id_vox: vec3<f32>, i: u32) -> vec3<f32> {
  
  let fi = f32(i);

  
  let h = hash31(cell_id_vox + vec3<f32>(fi * 7.3, 1.1, 2.9));

  
  let u = fract(h * 1.61803398875);  
  let v = fract(h * 2.41421356237);  
  let p = fract(h * 3.14159265359);  
  return vec3<f32>(u, v, p);
}

fn grass_sdf_normal_lod(
  p_m: vec3<f32>,
  cell_bmin_m: vec3<f32>,
  cell_id_vox: vec3<f32>,
  time_s: f32,
  strength: f32,
  lod: u32
) -> vec3<f32> {
  
  if (lod != 0u) {
    return vec3<f32>(0.0, 1.0, 0.0);
  }

  let e = 0.02 * cam.voxel_params.x;

  let dx =
    grass_sdf_lod(p_m + vec3<f32>(e, 0.0, 0.0), cell_bmin_m, cell_id_vox, time_s, strength, lod) -
    grass_sdf_lod(p_m - vec3<f32>(e, 0.0, 0.0), cell_bmin_m, cell_id_vox, time_s, strength, lod);

  let dy =
    grass_sdf_lod(p_m + vec3<f32>(0.0, e, 0.0), cell_bmin_m, cell_id_vox, time_s, strength, lod) -
    grass_sdf_lod(p_m - vec3<f32>(0.0, e, 0.0), cell_bmin_m, cell_id_vox, time_s, strength, lod);

  let dz =
    grass_sdf_lod(p_m + vec3<f32>(0.0, 0.0, e), cell_bmin_m, cell_id_vox, time_s, strength, lod) -
    grass_sdf_lod(p_m - vec3<f32>(0.0, 0.0, e), cell_bmin_m, cell_id_vox, time_s, strength, lod);

  return normalize(vec3<f32>(dx, dy, dz));
}


struct GrassHit {
  hit: bool,
  t: f32,
  n: vec3<f32>,
};

fn grass_layer_trace_lod(
  ro: vec3<f32>,
  rd: vec3<f32>,
  t_start: f32,
  t_end: f32,
  cell_bmin_m: vec3<f32>,
  cell_id_vox: vec3<f32>,
  time_s: f32,
  strength: f32,
  lod: u32
) -> GrassHit {
  let vs = cam.voxel_params.x;
  var t = t_start;

  var steps: u32 = GRASS_TRACE_STEPS;
  if (lod == 1u) { steps = GRASS_TRACE_STEPS_MID; }
  if (lod == 2u) { steps = GRASS_TRACE_STEPS_FAR; }

  for (var i: u32 = 0u; i < steps; i = i + 1u) {
    if (t > t_end) { break; }

    let p = ro + rd * t;
    let d = grass_sdf_lod(p, cell_bmin_m, cell_id_vox, time_s, strength, lod);

    let hit_eps = GRASS_HIT_EPS_VOX * vs;
    if (d < hit_eps) {
      let n = grass_sdf_normal_lod(p, cell_bmin_m, cell_id_vox, time_s, strength, lod);
      return GrassHit(true, t, n);
    }

    let step_min = GRASS_STEP_MIN_VOX * vs;
    t += max(d, step_min);
  }

  return GrassHit(false, BIG_F32, vec3<f32>(0.0));
}


fn try_grass_slab_hit(
  ro: vec3<f32>,
  rd: vec3<f32>,
  t_min: f32,
  t_max: f32,
  cell_bmin: vec3<f32>,
  cell_id_vox: vec3<f32>,
  vs: f32,
  time_s: f32,
  strength: f32
) -> GrassHit {
  let layer_h = GRASS_LAYER_HEIGHT_VOX * vs;
  let over    = GRASS_OVERHANG_VOX * vs;

  let slab_bmin = vec3<f32>(cell_bmin.x - over,      cell_bmin.y + vs,            cell_bmin.z - over);
  let slab_bmax = vec3<f32>(cell_bmin.x + vs + over, cell_bmin.y + vs + layer_h,  cell_bmin.z + vs + over);

  let rt_slab = intersect_aabb(ro, rd, slab_bmin, slab_bmax);

  var t0 = max(rt_slab.x, t_min);
  var t1 = min(rt_slab.y, t_max);

  
  let vox_bmin = cell_bmin;
  let vox_bmax = cell_bmin + vec3<f32>(vs);
  let rt_vox   = intersect_aabb(ro, rd, vox_bmin, vox_bmax);

  let clip_eps = 0.01 * vs;
  if (rt_vox.y > rt_vox.x) {
    let t_enter_vox = rt_vox.x;
    if (t_enter_vox > t0) {
      t1 = min(t1, t_enter_vox - clip_eps);
    }
  }

  if (t1 <= t0) {
    return GrassHit(false, BIG_F32, vec3<f32>(0.0));
  }

  
  let lod = grass_lod_from_t(t0);
  

  return grass_layer_trace_lod(ro, rd, t0, t1, cell_bmin, cell_id_vox, time_s, strength, lod);
}

fn grass_lod_from_t(t: f32) -> u32 {
  
  if (t >= GRASS_LOD_FAR_START) { return 2u; }
  if (t >= GRASS_LOD_MID_START) { return 1u; }
  return 0u;
}

fn grass_allowed_primary(t: f32, n: vec3<f32>, seed: u32) -> bool {
  if (t > GRASS_PRIMARY_MAX_DIST) { return false; }
  if (n.y < GRASS_PRIMARY_MIN_NY) { return false; }

  
  if ((seed & GRASS_PRIMARY_RATE_MASK) != 0u) { return false; }

  return true;
}
