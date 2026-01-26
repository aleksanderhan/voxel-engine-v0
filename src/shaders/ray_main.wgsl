// ray_main.wgsl
//
// Entry point: per-pixel ray, test all chunks, shade nearest hit.

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(out_img);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let res = vec2<f32>(f32(dims.x), f32(dims.y));
  let px  = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

  let ro = cam.cam_pos.xyz;
  let rd = ray_dir_from_pixel(px, res);

  // Background sky gradient.
  let tsky = clamp(0.5 * (rd.y + 1.0), 0.0, 1.0);
  let sky = mix(
    vec3<f32>(0.05, 0.08, 0.12),
    vec3<f32>(0.6, 0.8, 1.0),
    tsky
  );

  var best = HitGeom(false, BIG_F32, 0u, vec3<f32>(0.0));

  // Chunk AABB early-out.
  let voxel_size = cam.voxel_params.x;
  let chunk_size_m = f32(cam.chunk_size) * voxel_size;

  for (var i: u32 = 0u; i < cam.chunk_count; i = i + 1u) {
    let ch = chunks[i];

    let root_bmin = vec3<f32>(f32(ch.origin.x), f32(ch.origin.y), f32(ch.origin.z)) * voxel_size;
    let root_bmax = root_bmin + vec3<f32>(chunk_size_m);

    let rt = intersect_aabb(ro, rd, root_bmin, root_bmax);
    let t_enter = max(rt.x, 0.0);
    let t_exit  = rt.y;

    if (t_exit < t_enter) { continue; }
    if (t_enter >= best.t) { continue; }

    let h = trace_chunk_hybrid(ro, rd, ch);
    if (h.hit && h.t < best.t) {
      best = h;
    }
  }

  let col = select(sky, shade_hit(ro, rd, best), best.hit);
  textureStore(out_img, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(col, 1.0));
}
