// light.wgsl
//
// Cheap & simple deterministic voxel light gather:
//
// Idea:
// - Cast a *small fixed set* of hemisphere directions (no Hammersley, no CP shift, no rotation).
// - Step voxel-by-voxel using a tiny 3D DDA so we don't "skip" 1-voxel lights.
// - First MAT_LIGHT hit per ray contributes with a simple lambert + softened inverse-square falloff.
// - No backside wrap, no minimum fill.
//
// Expected external symbols (defined elsewhere in your shader project):
// - cam.voxel_params.x : voxel size in world meters
// - query_leaf_world(p_ws) -> leaf
// - leaf.mat material id
// - MAT_AIR, MAT_LIGHT material ids
// - hash_u32(u32) -> u32 (optional; used only for stable per-surface rotation if you enable it)
//
// This file only contains lighting code; it does not declare bindings.

// -----------------------------------------------------------------------------
// Tunables
// -----------------------------------------------------------------------------

const LIGHT_MAX_DIST_VOX : u32 = 72u;   // shorter is cheaper (try 24..64)
const LIGHT_RAYS         : u32 = 12u;    // 6..12 is usually enough
const LIGHT_INTENSITY    : f32 = 60.0;  // tune for your scene scale

// Softens 1/r^2 so it doesn't blow up near the source.
const LIGHT_SOFT_RADIUS_VOX : f32 = 3.0; // in voxels

// -----------------------------------------------------------------------------
// Emission
// -----------------------------------------------------------------------------

fn light_emission_color() -> vec3<f32> {
  return vec3<f32>(1.0, 0.95, 0.75);
}

// -----------------------------------------------------------------------------
// Small helpers
// -----------------------------------------------------------------------------

fn hash3_i32(p: vec3<i32>) -> u32 {
  let ux = u32(p.x) * 73856093u;
  let uy = u32(p.y) * 19349663u;
  let uz = u32(p.z) * 83492791u;
  return hash_u32(ux ^ uy ^ uz);
}

fn make_tbn(n: vec3<f32>) -> mat3x3<f32> {
  let up_ref = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
  let t = normalize(cross(up_ref, n));
  let b = normalize(cross(n, t));
  return mat3x3<f32>(t, b, n);
}

// Cheap rotation around normal (optional). Kept here but you can remove it for max simplicity.
fn rot_about_axis(v: vec3<f32>, axis: vec3<f32>, ang: f32) -> vec3<f32> {
  let a = normalize(axis);
  let s = sin(ang);
  let c = cos(ang);
  return v * c + cross(a, v) * s + a * dot(a, v) * (1.0 - c);
}

// -----------------------------------------------------------------------------
// Fixed direction set (very cheap, deterministic)
// -----------------------------------------------------------------------------

// A tiny fixed direction set over the *sphere* (cheap, deterministic).
// 12 dirs = 6 axis + 6 diagonals. Normalize anyway.
fn sphere_dir_local(i: u32) -> vec3<f32> {
  switch(i) {
    default: { return vec3<f32>(0.0, 0.0, 1.0); }

    // axis
    case 0u: { return vec3<f32>( 1.0,  0.0,  0.0); }
    case 1u: { return vec3<f32>(-1.0,  0.0,  0.0); }
    case 2u: { return vec3<f32>( 0.0,  1.0,  0.0); }
    case 3u: { return vec3<f32>( 0.0, -1.0,  0.0); }
    case 4u: { return vec3<f32>( 0.0,  0.0,  1.0); }
    case 5u: { return vec3<f32>( 0.0,  0.0, -1.0); }

    // diagonals (roughly distributed)
    case 6u:  { return vec3<f32>( 1.0,  1.0,  1.0); }
    case 7u:  { return vec3<f32>(-1.0,  1.0,  1.0); }
    case 8u:  { return vec3<f32>( 1.0, -1.0,  1.0); }
    case 9u:  { return vec3<f32>(-1.0, -1.0,  1.0); }
    case 10u: { return vec3<f32>( 1.0,  1.0, -1.0); }
    case 11u: { return vec3<f32>(-1.0,  1.0, -1.0); }
  }
}


// -----------------------------------------------------------------------------
// 3D DDA: walk voxels along ray without skipping 1-voxel lights
// -----------------------------------------------------------------------------

// Returns (hitLight, hitPosWS).
fn dda_hit_light(p0: vec3<f32>, dir: vec3<f32>, max_steps: u32, vs: f32) -> vec4<f32> {
  // Voxel coordinates in "grid units" where 1.0 = 1 voxel.
  // Keep this in float space; we only need stepping consistency.
  var gpos = p0 / max(vs, 1e-6);
  var cell = vec3<i32>(floor(gpos));

  let d = dir;

  // Step direction in each axis
  let step = vec3<i32>(
    select(-1, 1, d.x >= 0.0),
    select(-1, 1, d.y >= 0.0),
    select(-1, 1, d.z >= 0.0)
  );

  // Next boundary (in grid coords) we will cross in each axis
  let next_boundary = vec3<f32>(
    f32(cell.x + select(0, 1, d.x >= 0.0)),
    f32(cell.y + select(0, 1, d.y >= 0.0)),
    f32(cell.z + select(0, 1, d.z >= 0.0))
  );

  // tMax: distance along ray (in "grid units") to the next boundary
  // tDelta: distance along ray between crossings of voxel boundaries
  let inv = vec3<f32>(
    select(1e30, 1.0 / abs(d.x), abs(d.x) > 1e-6),
    select(1e30, 1.0 / abs(d.y), abs(d.y) > 1e-6),
    select(1e30, 1.0 / abs(d.z), abs(d.z) > 1e-6)
  );

  var tMax = vec3<f32>(
    (next_boundary.x - gpos.x) * inv.x,
    (next_boundary.y - gpos.y) * inv.y,
    (next_boundary.z - gpos.z) * inv.z
  );

  let tDelta = inv; // since grid cell size is 1

  // Visit starting cell too (helps when p0 is already inside a light voxel)
  for (var s: u32 = 0u; s < max_steps; s = s + 1u) {
    // Sample at the center of the current voxel cell
    let center_ws = (vec3<f32>(f32(cell.x) + 0.5, f32(cell.y) + 0.5, f32(cell.z) + 0.5)) * vs;
    let leaf = query_leaf_world(center_ws);

    if (leaf.mat == MAT_LIGHT) {
      return vec4<f32>(center_ws, 1.0);
    }
    if (leaf.mat != MAT_AIR) {
      // Solid blocks the ray
      return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Step to next voxel boundary: choose smallest tMax
    if (tMax.x < tMax.y) {
      if (tMax.x < tMax.z) {
        cell.x += step.x;
        tMax.x += tDelta.x;
      } else {
        cell.z += step.z;
        tMax.z += tDelta.z;
      }
    } else {
      if (tMax.y < tMax.z) {
        cell.y += step.y;
        tMax.y += tDelta.y;
      } else {
        cell.z += step.z;
        tMax.z += tDelta.z;
      }
    }
  }

  return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

// -----------------------------------------------------------------------------
// Main local light gather from MAT_LIGHT voxels (cheap version)
// -----------------------------------------------------------------------------

fn gather_voxel_lights(
  hp: vec3<f32>,
  n: vec3<f32>,
  root_bmin: vec3<f32>,
  root_size: f32,
  node_base: u32,
  macro_base: u32
) -> vec3<f32> {
  let vs = cam.voxel_params.x;

  // Nudge off surface to reduce self hits
  let p0 = hp + n * (0.75 * vs);

  // TBN basis (only used to orient our fixed hemisphere set)
  let tbn = make_tbn(n);

  // Optional stable per-surface rotation to reduce visible direction patterns.
  // If you want it even cheaper, delete these 3 lines and the rotation below.
  let surf_v = vec3<i32>(floor((hp - root_bmin) / max(vs, 1e-6)));
  let h = hash3_i32(surf_v);
  let rot = 6.28318530718 * (f32(h & 1023u) / 1024.0);

  var sum = vec3<f32>(0.0);

  // Distance softening in world units
  let soft_r = LIGHT_SOFT_RADIUS_VOX * vs;
  let soft_r2 = soft_r * soft_r;

  for (var i: u32 = 0u; i < LIGHT_RAYS; i = i + 1u) {
    var ldir = normalize(tbn * normalize(sphere_dir_local(i)));

    // optional: rotate around normal for de-patterning
    ldir = normalize(rot_about_axis(ldir, n, rot));

    let hit = dda_hit_light(p0, ldir, LIGHT_MAX_DIST_VOX, vs);
    if (hit.w > 0.5) {
      let pL = hit.xyz;

      let L = pL - hp;
      let r2 = max(dot(L, L), (1.25 * vs) * (1.25 * vs));
      let r = sqrt(r2);
      let ldir_ws = L / r;

      // Half-lambert: keeps some energy even when the light is slightly behind the normal.
      // This fakes a bit of bounce in caves without doing GI.
      let ndl_raw = dot(n, ldir_ws);
      let ndl = clamp(0.5 * ndl_raw + 0.5, 0.0, 1.0);
      if (ndl > 0.0) {
        // Softened inverse-square
        let falloff = 1.0 / (r2 + soft_r2);

        // Very simple energy scale
        sum += light_emission_color() * (LIGHT_INTENSITY * ndl * falloff);
      }
    }
  }

  return sum * (1.0 / f32(LIGHT_RAYS));
}
