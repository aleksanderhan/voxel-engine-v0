// ray_cs.wgsl
//
// Minimal compute shader for:
// - generating a ray direction per pixel from inverse matrices
// - intersecting the ray with a chunk AABB
// - 3D DDA stepping through voxels inside the chunk
// - per-voxel SVO point lookup (descend by coordinate bits)
// - returning first hit with simple lighting

struct Node {
  // Index of first child if internal, 0xFFFF_FFFF if leaf.
  child_base : u32,
  // Child presence bitmask (bit 0..7).
  child_mask : u32,
  // Material id for leaf nodes (0 = empty/air).
  material   : u32,
  _pad       : u32,
};

struct Camera {
  // Inverse view/projection so we can reconstruct world rays per pixel.
  view_inv     : mat4x4<f32>,
  proj_inv     : mat4x4<f32>,
  cam_pos      : vec4<f32>,

  // Chunk origin in world voxel coordinates (integer).
  chunk_origin : vec4<i32>,
  chunk_size   : u32,

  // Safety limit for DDA steps.
  max_steps    : u32,
  _pad0        : u32,
  _pad1        : u32,
};

@group(0) @binding(0) var<uniform> cam : Camera;
@group(0) @binding(1) var<storage, read> nodes : array<Node>;
@group(0) @binding(2) var out_img : texture_storage_2d<rgba16float, write>;

// Convert pixel -> world ray direction using inverse matrices.
fn ray_dir_from_pixel(px: vec2<f32>, res: vec2<f32>) -> vec3<f32> {
  // Normalized Device Coordinates (NDC): x,y in [-1,1]
  let ndc = vec4<f32>(
    (2.0 * px.x / res.x - 1.0),
    (1.0 - 2.0 * px.y / res.y),
    1.0,
    1.0
  );

  // Project back to view space.
  let view = cam.proj_inv * ndc;
  let v = vec4<f32>(view.xyz / view.w, 0.0);

  // Then to world space.
  let w = (cam.view_inv * v).xyz;
  return normalize(w);
}

// Ray vs axis-aligned bounding box intersection.
// Returns [t_enter, t_exit].
fn intersect_aabb(ro: vec3<f32>, rd: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
  let eps = 1e-8;

  var t_enter = -1e30;
  var t_exit  =  1e30;

  // X slab
  if (abs(rd.x) < eps) {
    if (ro.x < bmin.x || ro.x > bmax.x) { return vec2<f32>(1.0, 0.0); } // miss
  } else {
    let inv = 1.0 / rd.x;
    let t0 = (bmin.x - ro.x) * inv;
    let t1 = (bmax.x - ro.x) * inv;
    t_enter = max(t_enter, min(t0, t1));
    t_exit  = min(t_exit,  max(t0, t1));
  }

  // Y slab
  if (abs(rd.y) < eps) {
    if (ro.y < bmin.y || ro.y > bmax.y) { return vec2<f32>(1.0, 0.0); }
  } else {
    let inv = 1.0 / rd.y;
    let t0 = (bmin.y - ro.y) * inv;
    let t1 = (bmax.y - ro.y) * inv;
    t_enter = max(t_enter, min(t0, t1));
    t_exit  = min(t_exit,  max(t0, t1));
  }

  // Z slab
  if (abs(rd.z) < eps) {
    if (ro.z < bmin.z || ro.z > bmax.z) { return vec2<f32>(1.0, 0.0); }
  } else {
    let inv = 1.0 / rd.z;
    let t0 = (bmin.z - ro.z) * inv;
    let t1 = (bmax.z - ro.z) * inv;
    t_enter = max(t_enter, min(t0, t1));
    t_exit  = min(t_exit,  max(t0, t1));
  }

  return vec2<f32>(t_enter, t_exit);
}


// SVO point query: returns material at local voxel coord v in [0..chunk_size).
fn svo_lookup(v: vec3<u32>) -> u32 {
  // Root is node 0.
  var idx: u32 = 0u;
  var size: u32 = cam.chunk_size;

  // Descend until leaf or voxel level. 32 is a safe upper bound.
  for (var d: u32 = 0u; d < 32u; d = d + 1u) {
    let n = nodes[idx];

    // Leaf -> material
    if (n.child_base == 0xFFFFFFFFu) {
      return n.material;
    }

    // If size already voxel-level, treat empty.
    if (size <= 1u) {
      return 0u;
    }

    // Next level cube half-size in voxels
    size = size >> 1u;

    // Octant select:
    // If v.x has the current bit set, it is in the high half along X, etc.
    let cx = select(0u, 1u, (v.x & size) != 0u);
    let cy = select(0u, 2u, (v.y & size) != 0u);
    let cz = select(0u, 4u, (v.z & size) != 0u);
    let ci = cx | cy | cz;

    // Missing child means empty.
    if (((n.child_mask >> ci) & 1u) == 0u) {
      return 0u;
    }

    idx = n.child_base + ci;
  }

  return 0u;
}

// Material palette.
fn color_for_material(m: u32) -> vec3<f32> {
  if (m == 1u) { return vec3<f32>(0.20, 0.80, 0.20); } // floor green
  if (m == 2u) { return vec3<f32>(0.80, 0.30, 0.10); } // blocks orange/red
  return vec3<f32>(0.0);
}


const BIG_F32 : f32 = 1e30;
const EPS_INV : f32 = 1e-8;

fn safe_inv(x: f32) -> f32 {
  // Return a huge number for near-zero components so DDA/AABB stays finite-ish.
  return select(1.0 / x, BIG_F32, abs(x) < EPS_INV);
}


// 3D DDA stepping through voxels inside the chunk AABB.
// Returns the first hit as RGBA.
fn trace(ro: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {
  // Chunk bounds in world voxel coordinates.
  let origin = vec3<f32>(
    f32(cam.chunk_origin.x),
    f32(cam.chunk_origin.y),
    f32(cam.chunk_origin.z)
  );
  let bmin = origin;
  let bmax = origin + vec3<f32>(f32(cam.chunk_size));

  // Intersect ray with the chunk AABB.
  let t = intersect_aabb(ro, rd, bmin, bmax);
  if (t.y < max(t.x, 0.0)) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }

  // Start just inside the boundary to avoid precision sticking.
  var tcur = max(t.x, 0.0) + 1e-4;
  var p = ro + tcur * rd;

  // Bias slightly along the ray to avoid landing exactly on boundaries.
  p = p + rd * 1e-4;

  // Current voxel (world integer cell).
  var v = vec3<i32>(floor(p));


  let chunk_min = vec3<i32>(cam.chunk_origin.x, cam.chunk_origin.y, cam.chunk_origin.z);
  let chunk_max = chunk_min + vec3<i32>(i32(cam.chunk_size));

  // DDA step direction per axis.
  let step = vec3<i32>(
    select(-1, 1, rd.x > 0.0),
    select(-1, 1, rd.y > 0.0),
    select(-1, 1, rd.z > 0.0)
  );

  // Next voxel boundary planes from current voxel.
  let next_boundary = vec3<f32>(
    f32(v.x + select(0, 1, rd.x > 0.0)),
    f32(v.y + select(0, 1, rd.y > 0.0)),
    f32(v.z + select(0, 1, rd.z > 0.0))
  );

  let inv_rd = vec3<f32>(safe_inv(rd.x), safe_inv(rd.y), safe_inv(rd.z));
  
  // tMax/tDelta using inv_rd
  var tMax   = (next_boundary - ro) * inv_rd;
  var tDelta = abs(inv_rd); // because voxel size = 1

  // Tracks which axis we last stepped (for a cheap face normal).
  var last_axis = 0u;

  var last_lv = vec3<u32>(0u);
  var last_m: u32 = 0u;
  var have_last = false;

  for (var i = 0u; i < cam.max_steps; i = i + 1u) {
    if (tcur > t.y) { break; }

    // If voxel inside chunk, query local coordinates in the SVO.
    if (all(v >= chunk_min) && all(v < chunk_max)) {
      let lv = vec3<u32>(
        u32(v.x - chunk_min.x),
        u32(v.y - chunk_min.y),
        u32(v.z - chunk_min.z)
      );

      var m: u32;
      if (have_last && all(lv == last_lv)) {
        m = last_m;
      } else {
        m = svo_lookup(lv);
        last_lv = lv;
        last_m = m;
        have_last = true;
      }

      if (m != 0u) {
        let base = color_for_material(m);

        // Face normal from last stepped axis.
        var n = vec3<f32>(0.0);
        if (last_axis == 0u) { n = vec3<f32>(-f32(step.x), 0.0, 0.0); }
        if (last_axis == 1u) { n = vec3<f32>(0.0, -f32(step.y), 0.0); }
        if (last_axis == 2u) { n = vec3<f32>(0.0, 0.0, -f32(step.z)); }

        // Simple directional light.
        let sun = normalize(vec3<f32>(0.6, 1.0, 0.2));
        let diff = max(dot(n, sun), 0.0);
        let col = base * (0.25 + 0.75 * diff);

        return vec4<f32>(col, 1.0);
      }
    }

    // Step to next voxel boundary.
    if (tMax.x < tMax.y) {
      if (tMax.x < tMax.z) {
        v.x = v.x + step.x;
        tcur = tMax.x;
        tMax.x = tMax.x + tDelta.x;
        last_axis = 0u;
      } else {
        v.z = v.z + step.z;
        tcur = tMax.z;
        tMax.z = tMax.z + tDelta.z;
        last_axis = 2u;
      }
    } else {
      if (tMax.y < tMax.z) {
        v.y = v.y + step.y;
        tcur = tMax.y;
        tMax.y = tMax.y + tDelta.y;
        last_axis = 1u;
      } else {
        v.z = v.z + step.z;
        tcur = tMax.z;
        tMax.z = tMax.z + tDelta.z;
        last_axis = 2u;
      }
    }
  }

  return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(out_img);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let res = vec2<f32>(f32(dims.x), f32(dims.y));
  let px = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

  let ro = cam.cam_pos.xyz;
  let rd = ray_dir_from_pixel(px, res);

  let c = trace(ro, rd);
  textureStore(out_img, vec2<i32>(i32(gid.x), i32(gid.y)), c);
}
