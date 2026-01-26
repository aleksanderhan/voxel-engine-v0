// ray_shade.wgsl
//
// Material colors + final shading (calls in_shadow()).

fn color_for_material(m: u32) -> vec3<f32> {
  if (m == 0u) { return vec3<f32>(0.0); } // AIR

  if (m == 1u) { return vec3<f32>(0.18, 0.75, 0.18); } // GRASS
  if (m == 2u) { return vec3<f32>(0.45, 0.30, 0.15); } // DIRT
  if (m == 3u) { return vec3<f32>(0.50, 0.50, 0.55); } // STONE

  if (m == 4u) { return vec3<f32>(0.38, 0.26, 0.14); } // WOOD
  if (m == 5u) { return vec3<f32>(0.10, 0.55, 0.12); } // LEAF

  return vec3<f32>(1.0, 0.0, 1.0);
}

fn shade_hit(ro: vec3<f32>, rd: vec3<f32>, hg: HitGeom) -> vec3<f32> {
  let hp = ro + hg.t * rd;
  let base = color_for_material(hg.mat);

  let shadow = select(1.0, 0.0, in_shadow(hp, SUN_DIR));
  let diff = max(dot(hg.n, SUN_DIR), 0.0);

  let ambient = select(0.22, 0.28, hg.mat == 5u);

  var dapple = 1.0;
  if (hg.mat == 5u) {
    let time_s = cam.voxel_params.y;
    let d0 = sin(dot(hp.xz, vec2<f32>(3.0, 2.2)) + time_s * 3.5);
    let d1 = sin(dot(hp.xz, vec2<f32>(6.5, 4.1)) - time_s * 6.0);
    dapple = 0.90 + 0.10 * (0.6 * d0 + 0.4 * d1);
  }

  let direct = SUN_COLOR * SUN_INTENSITY * diff * shadow;
  let lit = base * (ambient + (1.0 - ambient) * direct) * dapple;

  return lit;
}
