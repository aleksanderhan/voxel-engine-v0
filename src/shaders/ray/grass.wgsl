//// --------------------------------------------------------------------------
//// Grass: SDF blades + tracing
//// --------------------------------------------------------------------------

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

fn grass_cell_from_world(
  hp_m: vec3<f32>,
  rd: vec3<f32>,
  root_bmin_m: vec3<f32>,
  voxel_size_m: f32,
  chunk_size_vox: i32
) -> GrassCell {
  let ch_origin_vox = vec3<i32>(
    i32(floor(root_bmin_m.x / voxel_size_m)),
    i32(floor(root_bmin_m.y / voxel_size_m)),
    i32(floor(root_bmin_m.z / voxel_size_m))
  );
  return pick_grass_cell_in_chunk(
    hp_m,
    rd,
    root_bmin_m,
    ch_origin_vox,
    voxel_size_m,
    chunk_size_vox
  );
}

fn sdf_box(p: vec3<f32>, c: vec3<f32>, b: vec3<f32>) -> f32 {
  let q = abs(p - c) - b;
  let outside = length(max(q, vec3<f32>(0.0)));
  let inside  = min(max(q.x, max(q.y, q.z)), 0.0);
  return outside + inside;
}

fn sdf_capsule(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
  return length(pa - ba * h) - r;
}

fn make_orthonormal_basis(n: vec3<f32>) -> mat3x3<f32> {
  // n becomes Y axis in this basis; returns columns (x,y,z)
  let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.999);
  let x = normalize(cross(up, n));
  let z = cross(n, x);
  return mat3x3<f32>(x, n, z);
}

// distance to a "rounded rectangle" in 2D (Inigo Quilez style)
fn sd_round_rect_2d(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

// A flat blade segment: closest point on segment, then distance in the segment's local frame
// width = half-width (flat axis), thick = half-thickness (thin axis), edge_r = rounding
fn sdf_blade_segment(
  p: vec3<f32>,
  a: vec3<f32>,
  b: vec3<f32>,
  width: f32,
  thick: f32,
  edge_r: f32,
  // stable per-blade "side" axis orientation control
  side_hint: vec3<f32>
) -> f32 {
  let ab = b - a;
  let ab2 = max(dot(ab, ab), 1e-6);
  let t = clamp(dot(p - a, ab) / ab2, 0.0, 1.0);
  let c = a + ab * t;

  let T = normalize(ab);

  // Build a stable frame: choose side axis from side_hint, made orthogonal to T.
  var S = side_hint - T * dot(side_hint, T);
  let s2 = dot(S, S);
  if (s2 < 1e-6) {
    // fallback if hint is parallel to tangent
    let B = make_orthonormal_basis(T);
    S = B[0]; // X axis
  } else {
    S = S * inverseSqrt(s2);
  }
  let N = cross(T, S); // thin axis (normal to blade face), orthonormal

  let d = p - c;

  // local coords around the segment centerline
  let x = dot(d, S); // across blade width
  let y = dot(d, N); // blade thickness direction

  // We want a slab around the segment, so clamp z into segment half-length implicitly
  // Using closest point already handles along-segment distance -> z ~= 0 at closest.
  // Now do a rounded-rect in (x,y) and combine with a small rounding.
  let rr = max(edge_r, 0.0);
  return sd_round_rect_2d(vec2<f32>(x, y), vec2<f32>(width - rr, thick - rr), rr);
}

fn grass_root_uv(cell_id_vox: vec3<f32>, i: u32) -> vec2<f32> {
  let fi = f32(i);
  let u = hash31(cell_id_vox + vec3<f32>(fi, 0.0, 0.0));
  let v = hash31(cell_id_vox + vec3<f32>(0.0, fi, 0.0));
  return vec2<f32>(u, v);
}

fn grass_wind_xz(root_m: vec3<f32>, t: f32, strength: f32) -> vec2<f32> {
  // wind_field returns vec3; we only use XZ for bending
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

  // Quick vertical reject
  let y01 = (p_m.y - top_y) / max(layer_h, 1e-6);
  if (y01 < 0.0 || y01 > 1.0) { return BIG_F32; }

  // Cheap horizontal reject
  let over = GRASS_OVERHANG_VOX * vs;
  if (p_m.x < cell_bmin_m.x - over || p_m.x > cell_bmin_m.x + vs + over ||
      p_m.z < cell_bmin_m.z - over || p_m.z > cell_bmin_m.z + vs + over) {
    return BIG_F32;
  }

  let blade_len = layer_h * (0.65 + 0.35 * hash31(cell_id_vox + vec3<f32>(9.1, 3.7, 5.2)));

  // LOD selection (as before)
  var blade_count: u32 = GRASS_BLADE_COUNT;
  var segs: u32 = u32(max(3.0, floor(GRASS_VOXEL_SEGS)));
  if (lod == 1u) {
    blade_count = min(blade_count, GRASS_BLADE_COUNT_MID);
    segs = min(segs, GRASS_SEGS_MID);
  } else if (lod == 2u) {
    blade_count = min(blade_count, GRASS_BLADE_COUNT_FAR);
    segs = min(segs, GRASS_SEGS_FAR);
  }

  let inv_segs = 1.0 / max(f32(segs), 1.0);

  var dmin = BIG_F32;

  // --- CLUMPING: a few tuft centers per cell ---
  // (small number keeps it cheap; makes “lush tufts” instead of uniform lawn)
  let CLUMPS: u32 = 5u;

  // tuft radius in-cell (in meters)
  let clump_r = 0.18 * vs;

  // base thickness in meters (this is your “lushness” knob)
  let base_r = (GRASS_VOXEL_THICKNESS_VOX * vs) * 0.85;

  for (var i: u32 = 0u; i < blade_count; i = i + 1u) {
    // (u,v,phase,widthRand)
    let uvpw = grass_blade_params(cell_id_vox, i);
    let u0 = uvpw.x;
    let v0 = uvpw.y;
    let ph = uvpw.z;
    let wR = uvpw.w;

    // pick clump index and its center (stable per cell)
    let ci = i % CLUMPS;
    let cfi = f32(ci);
    let cu = hash31(cell_id_vox + vec3<f32>(13.7 + cfi, 2.1, 9.9));
    let cv = hash31(cell_id_vox + vec3<f32>( 4.2 + cfi, 7.3, 1.4));

    // clump center in [inset..1-inset]
    let inset = 0.10;
    let cux = mix(inset, 1.0 - inset, cu);
    let cvz = mix(inset, 1.0 - inset, cv);

    let clump_center = vec2<f32>(
      cell_bmin_m.x + cux * vs,
      cell_bmin_m.z + cvz * vs
    );

    // blade root starts near the clump center, but jittered inside a tuft radius
    // (pulls blades into thick patches)
    let ang = TAU * u0;
    let rr  = clump_r * (0.15 + 0.85 * v0);
    let root_xz = clump_center + vec2<f32>(cos(ang), sin(ang)) * rr;

    let root = vec3<f32>(root_xz.x, top_y, root_xz.y);

    // wind (XZ only), phase-shifted
    let w_xz = grass_wind_xz(root + vec3<f32>(0.0, ph, 0.0), time_s, strength);

    // per-blade lean direction (adds “messy lush” look even with low wind)
    let lean_ang = TAU * fract(u0 * 1.73 + v0 * 2.11);
    let lean_dir = vec2<f32>(cos(lean_ang), sin(lean_ang));
    let lean_amt = 0.10 + 0.25 * (v0 * v0);   // stable bend

    // radius variance: thicker overall, plus per-blade
    let r0 = base_r * mix(0.85, 1.35, wR);

    // segment loop: capsules (rounded blades)
    for (var s: u32 = 0u; s < segs; s = s + 1u) {
      let t01a = (f32(s)      ) * inv_segs;
      let t01b = (f32(s) + 1.0) * inv_segs;

      let ya = t01a * blade_len;
      let yb = t01b * blade_len;

      // bending increases with height, with a little extra lean
      let bend_a = (blade_len * t01a) * (0.55 + 0.45 * t01a);
      let bend_b = (blade_len * t01b) * (0.55 + 0.45 * t01b);

      // wind + stable lean combined
      let bend_vec = (w_xz + lean_dir * lean_amt);

      let offa = bend_vec * bend_a;
      let offb = bend_vec * bend_b;

      let pa = root + vec3<f32>(offa.x, ya, offa.y);
      let pb = root + vec3<f32>(offb.x, yb, offb.y);

      // --- taper to a point ---
      // t in [0..1], 0=base, 1=tip
      let tmid = 0.5 * (t01a + t01b);

      // Nonlinear taper so most narrowing happens near the top.
      // k bigger => pointier. Try 2.0..4.0.
      let k = 3.0;
      let taper_shape = pow(clamp(1.0 - tmid, 0.0, 1.0), k);

      // Treat GRASS_VOXEL_TAPER as *minimum tip fraction* (set it small, e.g. 0.08..0.20)
      let taper = mix(GRASS_VOXEL_TAPER, 1.0, taper_shape);

      // --- flat blade profile (width tapers hard, thickness follows) ---
      let half_w = r0 * taper;

      // thickness: much smaller than width (ribbon-like blade)
      // keep a tiny floor to avoid shimmering / disappearing
      let half_t = max(0.03 * half_w, 0.006 * vs);

      // edge rounding scales down with thickness
      let edge_r = 0.35 * half_t;

      // Roll/twist changes the ribbon orientation along the blade, giving richer silhouettes.
      // (tmid is 0..1 along blade)
      let roll = TAU * (0.15 * ph + 0.35 * tmid + 0.10 * wR);
      let roll_dir = vec2<f32>(cos(roll), sin(roll));

      let side_hint = normalize(vec3<f32>(
        (lean_dir.x + 0.35 * w_xz.x) * roll_dir.x - (lean_dir.y + 0.35 * w_xz.y) * roll_dir.y,
        0.12,
        (lean_dir.x + 0.35 * w_xz.x) * roll_dir.y + (lean_dir.y + 0.35 * w_xz.y) * roll_dir.x
      ));

      dmin = min(dmin, sdf_blade_segment(p_m, pa, pb, half_w, half_t, edge_r, side_hint));
    }
  }

  return dmin;
}


fn grass_blade_params(cell_id_vox: vec3<f32>, i: u32) -> vec4<f32> {
  // returns (u, v, phase, w) where w is a width rand
  let fi = f32(i);

  let h0 = hash31(cell_id_vox + vec3<f32>(fi * 7.3, 1.1, 2.9));
  let h1 = hash31(cell_id_vox + vec3<f32>(fi * 3.7, 8.4, 0.6));

  let u = fract(h0 * 1.61803398875);
  let v = fract(h0 * 2.41421356237);
  let p = fract(h0 * 3.14159265359);

  // width rand (biased towards thicker)
  let w = 0.35 + 0.65 * (h1 * h1);

  return vec4<f32>(u, v, p, w);
}

fn grass_sdf_normal_lod(
  p_m: vec3<f32>,
  cell_bmin_m: vec3<f32>,
  cell_id_vox: vec3<f32>,
  time_s: f32,
  strength: f32,
  lod: u32
) -> vec3<f32> {
  // Mid/far: cheap approximation is good enough visually
  if (lod != 0u) {
    // Approximate: "up" tilted by local wind direction (stable, cheap)
    let vs = cam.voxel_params.x;
    let top_y = cell_bmin_m.y + vs;

    // pick a representative root in this cell (center)
    let root = vec3<f32>(cell_bmin_m.x + 0.5 * vs, top_y, cell_bmin_m.z + 0.5 * vs);
    let w = grass_wind_xz(root, time_s, strength);

    // tilt amount tuned small
    let tilt = 0.35;
    return normalize(vec3<f32>(-w.x * tilt, 1.0, -w.y * tilt));
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

fn grass_self_shadow(
  hp: vec3<f32>,
  cell_bmin_m: vec3<f32>,
  cell_id_vox: vec3<f32>,
  time_s: f32,
  strength: f32,
  lod: u32
) -> f32 {
  // March a few short steps toward the sun through the grass volume.
  // Returns transmittance in [0..1].
  let vs = cam.voxel_params.x;

  // start a bit off the surface
  var p = hp + SUN_DIR * (0.02 * vs);

  // tune distance: longer = more shadowing
  let max_dist = (0.55 + 0.35 * f32(lod)) * vs;
  let steps: u32 = select(6u, 4u, lod != 0u);

  let dt = max_dist / f32(steps);

  var occ: f32 = 0.0;
  for (var i: u32 = 0u; i < steps; i = i + 1u) {
    let d = grass_sdf_lod(p, cell_bmin_m, cell_id_vox, time_s, strength, lod);

    // Convert SDF “inside/near” into occlusion
    // (soft ramp, not binary)
    let x = clamp(1.0 - d / (0.05 * vs), 0.0, 1.0);
    occ += x;

    p += SUN_DIR * dt;
  }

  // Map accumulated occlusion to transmittance
  // stronger exponent = darker cores, nicer depth
  let k = 0.35;
  return exp(-k * occ);
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

  // clip by true solid voxel cube (avoid "inside voxel" artifacts)
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

  // ---- NEW: pick LOD based on distance along the ray
  let lod = grass_lod_from_t(t0);
  // -----------------------------------------------

  return grass_layer_trace_lod(ro, rd, t0, t1, cell_bmin, cell_id_vox, time_s, strength, lod);
}

fn grass_lod_from_t(t: f32) -> u32 {
  // 0 = near, 1 = mid, 2 = far
  if (t >= GRASS_LOD_FAR_START) { return 2u; }
  if (t >= GRASS_LOD_MID_START) { return 1u; }
  return 0u;
}

fn grass_allowed_primary(_t: f32, _n: vec3<f32>, _rd: vec3<f32>, _seed: u32) -> bool {
  return ENABLE_GRASS;
}
