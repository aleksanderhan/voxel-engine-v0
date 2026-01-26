// ray_shadow.wgsl
//
// Shadow ray traversal.

const SHADOW_STEPS : u32 = 32u;

// If false: leaves cast shadows using their undisplaced voxel cube (faster).
// If true: shadows match displaced leaf cubes (slower).
const SHADOW_DISPLACED_LEAVES : bool = true;

fn trace_chunk_shadow(ro: vec3<f32>, rd: vec3<f32>, ch: ChunkMeta, t_min: f32) -> bool {
  let voxel_size = cam.voxel_params.x;

  let root_bmin_vox = vec3<f32>(f32(ch.origin.x), f32(ch.origin.y), f32(ch.origin.z));
  let root_bmin = root_bmin_vox * voxel_size;
  let root_size = f32(cam.chunk_size) * voxel_size;
  let root_bmax = root_bmin + vec3<f32>(root_size);

  let rt = intersect_aabb(ro, rd, root_bmin, root_bmax);
  var t_enter = max(rt.x, t_min);
  let t_exit  = rt.y;

  if (t_exit < t_enter) { return false; }

  var tcur = t_enter;

  for (var step_i: u32 = 0u; step_i < SHADOW_STEPS; step_i = step_i + 1u) {
    if (tcur > t_exit) { break; }

    let p = ro + tcur * rd;
    let q = query_leaf_at(p, root_bmin, root_size, ch.node_base);

    if (q.mat != 0u) {
      if (q.mat == 5u) {
        if (!SHADOW_DISPLACED_LEAVES) {
          return true;
        }

        let time_s   = cam.voxel_params.y;
        let strength = cam.voxel_params.z;

        let h2 = leaf_displaced_cube_hit(
          ro, rd,
          q.bmin, q.size,
          time_s, strength,
          tcur - 1e-5,
          t_exit
        );

        if (h2.hit) { return true; }

        let t_leave = exit_time_from_cube(ro, rd, q.bmin, q.size);
        tcur = max(t_leave, tcur) + 1e-4;
        continue;
      }

      return true;
    }

    let t_leave = exit_time_from_cube(ro, rd, q.bmin, q.size);
    tcur = max(t_leave, tcur) + 1e-4;
  }

  return false;
}

fn in_shadow(p: vec3<f32>, sun_dir: vec3<f32>) -> bool {
  let ro = p + sun_dir * SHADOW_BIAS;
  let rd = sun_dir;

  for (var i: u32 = 0u; i < cam.chunk_count; i = i + 1u) {
    if (trace_chunk_shadow(ro, rd, chunks[i], 0.0)) {
      return true;
    }
  }
  return false;
}
