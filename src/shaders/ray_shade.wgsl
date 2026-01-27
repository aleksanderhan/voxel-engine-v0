// ray_shade.wgsl
//
// Material palette + final surface shading.
//
// Responsibilities:
// - Map material ids (u32) to base albedo colors.
// - Compute simple direct + ambient lighting from a single directional sun.
// - Query hard shadows via in_shadow().
// - Apply a small "dapple" modulation for leaf material to fake translucency / flutter.
//
// Dependencies:
// - HitGeom from ray_core.wgsl (hit record: t, mat, normal).
// - cam.voxel_params:
//     x = voxel_size_m
//     y = time_seconds
// - SUN_DIR, SUN_COLOR, SUN_INTENSITY from common.wgsl.
// - in_shadow() defined elsewhere (likely a shadow ray marcher through the SVO).

/// Return a base RGB color for a given material id.
///
/// Convention:
/// - 0 = air/empty (not normally shaded; returns black)
/// - 1..N = solid materials
/// - Unknown ids return magenta as a debug "missing material" marker.
fn color_for_material(m: u32) -> vec3<f32> {
  if (m == 0u) { return vec3<f32>(0.0); } // AIR (shouldn't be hit)

  if (m == 1u) { return vec3<f32>(0.18, 0.75, 0.18); } // GRASS
  if (m == 2u) { return vec3<f32>(0.45, 0.30, 0.15); } // DIRT
  if (m == 3u) { return vec3<f32>(0.50, 0.50, 0.55); } // STONE

  if (m == 4u) { return vec3<f32>(0.38, 0.26, 0.14); } // WOOD
  if (m == 5u) { return vec3<f32>(0.10, 0.55, 0.12); } // LEAF

  // Debug: "unknown material"
  return vec3<f32>(1.0, 0.0, 1.0);
}

/// Shade a hit point with a single directional sun + hard shadow.
///
/// Inputs:
/// - ro/rd: camera ray (used to reconstruct hit position hp)
/// - hg   : hit record (t, mat, normal)
///
/// Lighting model (deliberately cheap):
/// - Ambient term: constant per-material (slightly higher for leaves).
/// - Direct term : Lambert (N·L) times hard shadow mask.
/// - Leaf-only "dapple": time-varying modulation to break up flat shading.
///
/// Notes on shadow acne:
/// Shadow rays starting exactly on the surface tend to self-intersect due to
/// limited precision and voxel/grid discretization. We push the shadow start
/// point outward along the geometric normal by ~0.75 voxels.
fn shade_hit(ro: vec3<f32>, rd: vec3<f32>, hg: HitGeom) -> vec3<f32> {
  // Hit point in world space.
  let hp = ro + hg.t * rd;

  // Base albedo for the material.
  let base = color_for_material(hg.mat);

  // Start shadow ray clearly off the surface to reduce acne / grid striping.
  // Offset is expressed in world meters via voxel_size.
  let voxel_size = cam.voxel_params.x;
  let hp_shadow  = hp + hg.n * (0.75 * voxel_size); // tune range: ~0.5..1.5 voxels

  // Hard shadow: 1 = lit, 0 = in shadow.
  // in_shadow() returns bool; select converts it into a scalar.
  let shadow = select(1.0, 0.0, in_shadow(hp_shadow, SUN_DIR));

  // Lambert diffuse term from sun direction.
  let diff = max(dot(hg.n, SUN_DIR), 0.0);

  // Ambient floor. Leaves get a bit more ambient to feel brighter/translucent-ish.
  let ambient = select(0.22, 0.28, hg.mat == 5u);

  // Leaf-only dapple:
  // A cheap moving sinusoidal pattern in XZ that adds subtle brightness variation.
  // This helps leaves feel less like flat cubes and more like noisy foliage.
  var dapple = 1.0;
  if (hg.mat == 5u) {
    let time_s = cam.voxel_params.y;
    let d0 = sin(dot(hp.xz, vec2<f32>(3.0, 2.2)) + time_s * 3.5);
    let d1 = sin(dot(hp.xz, vec2<f32>(6.5, 4.1)) - time_s * 6.0);
    dapple = 0.90 + 0.10 * (0.6 * d0 + 0.4 * d1); // keep within ~[0.8..1.0+]
  }

  // Direct sunlight contribution (color/intensity constants come from common.wgsl).
  let direct = SUN_COLOR * SUN_INTENSITY * diff * shadow;

  // Combine:
  // - Ambient contributes a constant fraction of base color.
  // - (1 - ambient) scales how strongly direct lighting modulates the remainder.
  // - dapple is applied at the end (leaf-only modulation).
  let lit = base * (ambient + (1.0 - ambient) * direct) * dapple;

  return lit;
}
