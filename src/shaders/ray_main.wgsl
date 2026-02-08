// src/shaders/ray_main.wgsl
//
// Compute entrypoints + pass bindings only.
// Depends on: common.wgsl + ray_core.wgsl + clipmap.wgsl
//
// Changes:
// - Added local light output target (local_img) for shade_hit_split().local_hdr
// - main_primary now:
//     * computes ShadeOut for voxel hits
//     * fogs ONLY base_hdr (and emissive voxels are already in base_hdr)
//     * writes local_hdr UNFOGGED into local_img (for later temporal accumulation)
// - For non-voxel cases (heightfield/sky), local_img is written as 0.
//
// NOTE: This file assumes you have:
// - shade_hit_split() in shading.wgsl (and ShadeOut struct)
// - shade_clip_hit() unchanged
// - apply_fog() unchanged
// - Your temporal accumulation pass is separate (not shown here).

struct PrimaryDispatch {
  base_y: u32,
  max_y: u32,
  _pad: vec2<u32>,
};

struct TileCandidates {
  count: u32,
  slots: array<u32, PRIMARY_MAX_TILE_CHUNKS>,
  enters: array<f32, PRIMARY_MAX_TILE_CHUNKS>,
};

@group(0) @binding(4) var color_img : texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var depth_img : texture_storage_2d<r32float, write>;

// local (noisy) voxel-light term output (HDR, NOT fogged)
@group(0) @binding(6) var local_img : texture_storage_2d<rgba32float, write>;

// primary hit history (temporal reprojection)
@group(0) @binding(12) var primary_hist_tex  : texture_2d<f32>;
@group(0) @binding(13) var primary_hist_out  : texture_storage_2d<rgba32float, write>;
@group(0) @binding(14) var primary_hist_samp : sampler;

@group(0) @binding(15) var shadow_hist_in : texture_2d<f32>;
@group(0) @binding(16) var<storage, read_write> shadow_hist_out : array<f32>;

@group(1) @binding(0) var depth_tex       : texture_2d<f32>;
@group(1) @binding(1) var godray_hist_tex : texture_2d<f32>;
@group(1) @binding(2) var godray_out      : texture_storage_2d<rgba32float, write>;
@group(1) @binding(3) var godray_hist_samp: sampler;

@group(2) @binding(0) var color_tex  : texture_2d<f32>;
@group(2) @binding(1) var godray_tex : texture_2d<f32>;
@group(2) @binding(2) var out_img    : texture_storage_2d<rgba32float, write>;
@group(2) @binding(3) var depth_full : texture_2d<f32>;
@group(2) @binding(4) var godray_samp: sampler;

// accumulated local lighting (HDR, same res as internal render)
@group(2) @binding(5) var local_hist_tex : texture_2d<f32>;
@group(2) @binding(6) var local_samp     : sampler;

var<workgroup> WG_SKY_UP : vec3<f32>;
var<workgroup> WG_TILE_COUNT_CACHED : u32;

fn pack_i16x2(a: i32, b: i32) -> u32 {
  return (u32(a) & 0xFFFFu) | ((u32(b) & 0xFFFFu) << 16u);
}

fn unpack_i16(v: u32) -> i32 {
  let s = i32(v & 0xFFFFu);
  return select(s, s - 0x10000, (v & 0x8000u) != 0u);
}

fn unpack_i16x2(v: u32) -> vec2<i32> {
  return vec2<i32>(unpack_i16(v), unpack_i16(v >> 16u));
}

struct HfShadeOut {
  col: vec3<f32>,
  t_scene: f32,
};

@group(0) @binding(17) var<uniform> primary_dispatch : PrimaryDispatch;
@group(3) @binding(0) var<storage, read_write> tile_candidates : array<TileCandidates>;

fn shade_heightfield(
  ro: vec3<f32>,
  rd: vec3<f32>,
  hf: ClipHit,
  sky_up: vec3<f32>,
  seed: u32
) -> HfShadeOut {
  let surface = shade_clip_hit(ro, rd, hf, sky_up, seed);
  let t_scene = min(hf.t, FOG_MAX_DIST);
  let sky_bg_rd = sky_bg(rd);
  let col = apply_fog(surface, ro, rd, t_scene, sky_bg_rd);
  return HfShadeOut(col, t_scene);
}

fn trace_primary_voxels(
  ro: vec3<f32>,
  rd: vec3<f32>,
  t_start: f32,
  t_end: f32,
  has_tile_candidates: bool,
  hist_valid: bool,
  hist_anchor_key: u32,
  hist_anchor_coord: vec3<i32>,
  tile_candidate_count: u32,
  seed: u32
) -> VoxTraceResult {
  var vt = VoxTraceResult(false, miss_hitgeom(), 0.0, false, INVALID_U32, vec3<i32>(0));
  if (has_tile_candidates) {
    if (hist_valid) {
      let vt_hint = trace_scene_voxels_candidates(
        ro,
        rd,
        t_start,
        t_end,
        hist_anchor_key != INVALID_U32,
        hist_anchor_coord,
        hist_anchor_key,
        tile_candidate_count,
        seed
      );
      if (vt_hint.best.hit != 0u) {
        return vt_hint;
      }
    }
    vt = trace_scene_voxels_candidates(
      ro,
      rd,
      0.0,
      FOG_MAX_DIST,
      false,
      vec3<i32>(0),
      INVALID_U32,
      tile_candidate_count,
      seed
    );
  }
  return vt;
}

@compute @workgroup_size(1, 1, 1)
fn main_primary_tiles(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tiles_x = (cam.render_present_px.x + TILE_SIZE - 1u) / TILE_SIZE;
  let tiles_y = (cam.render_present_px.y + TILE_SIZE - 1u) / TILE_SIZE;
  if (gid.x >= tiles_x || gid.y >= tiles_y) { return; }

  atomicStore(&WG_TILE_COUNT, 0u);
  if (cam.chunk_count != 0u) {
    let tile_base = vec2<f32>(
      f32(gid.x * TILE_SIZE),
      f32(gid.y * TILE_SIZE)
    );
    let ro_tile = cam.cam_pos.xyz;

    // Two candidate rays per tile to stabilize distant voxel selection.
    var px = tile_base + vec2<f32>(4.5, 4.5);
    tile_append_candidates_for_ray(ro_tile, ray_dir_from_pixel(px), 0.0, FOG_MAX_DIST);
    px = tile_base + vec2<f32>(1.5, 1.5);
    tile_append_candidates_for_ray(ro_tile, ray_dir_from_pixel(px), 0.0, FOG_MAX_DIST);
  }
  let raw_count = min(atomicLoad(&WG_TILE_COUNT), MAX_TILE_CHUNKS);
  tile_sort_candidates_by_enter(raw_count);
  let count = min(raw_count, PRIMARY_MAX_TILE_CHUNKS);

  let tile_index = gid.y * tiles_x + gid.x;
  tile_candidates[tile_index].count = count;
  for (var i: u32 = 0u; i < PRIMARY_MAX_TILE_CHUNKS; i = i + 1u) {
    if (i < count) {
      tile_candidates[tile_index].slots[i] = WG_TILE_SLOTS[i];
      tile_candidates[tile_index].enters[i] = WG_TILE_ENTER[i];
    } else {
      tile_candidates[tile_index].slots[i] = INVALID_U32;
      tile_candidates[tile_index].enters[i] = BIG_F32;
    }
  }
}

@compute @workgroup_size(8, 8, 1)
fn main_primary(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wg_id: vec3<u32>
) {
  let dims = textureDimensions(color_img);
  let global_y = gid.y + primary_dispatch.base_y;
  if (gid.x >= dims.x || global_y >= primary_dispatch.max_y) { return; }

  // Compute once per 8x8 workgroup (already cheap: sky_bg only)
  if (lid == 0u) {
    WG_SKY_UP = sky_bg(vec3<f32>(0.0, 1.0, 0.0));
  }
  workgroupBarrier();
  let sky_up = WG_SKY_UP;

  if (lid == 0u) {
    let tiles_x = (u32(dims.x) + TILE_SIZE - 1u) / TILE_SIZE;
    let base_tile_y = primary_dispatch.base_y / TILE_SIZE;
    let tile_index = (base_tile_y + wg_id.y) * tiles_x + wg_id.x;
    let count = min(tile_candidates[tile_index].count, PRIMARY_MAX_TILE_CHUNKS);
    WG_TILE_COUNT_CACHED = count;
    for (var i: u32 = 0u; i < count; i = i + 1u) {
      WG_TILE_SLOTS[i] = tile_candidates[tile_index].slots[i];
      WG_TILE_ENTER[i] = tile_candidates[tile_index].enters[i];
    }
  }
  workgroupBarrier();

  let res = vec2<f32>(f32(dims.x), f32(dims.y));
  let px  = vec2<f32>(f32(gid.x) + 0.5, f32(global_y) + 0.5);
  let frame = cam.frame_index;
  let seed  = (u32(gid.x) * 1973u) ^ (u32(global_y) * 9277u) ^ (frame * 26699u);

  let ro  = cam.cam_pos.xyz;
  let rd  = ray_dir_from_pixel(px);

  let ip = vec2<i32>(i32(gid.x), i32(global_y));

  // Local output defaults: invalid (alpha=0) so TAA keeps history instead of blending black.
  var local_out = vec3<f32>(0.0);
  var local_w   : f32 = 0.0;
  var t_store   : f32 = 0.0;
  var shadow_out: f32 = 1.0;

  // ------------------------------------------------------------
  // Case 1: no voxel chunks => heightfield or sky
  // ------------------------------------------------------------
  let bytes_per_row = dims.x * 4u;
  let padded_bpr = (bytes_per_row + 255u) & ~255u;
  let shadow_stride = padded_bpr / 4u;
  let shadow_idx = u32(ip.y) * shadow_stride + u32(ip.x);

  if (cam.chunk_count == 0u) {
    var hf = clip_trace_heightfield(ro, rd, 0.0, FOG_MAX_DIST);

    if (hf.hit) {
      if (hf.mat == MAT_GRASS && ENABLE_GRASS && grass_allowed_primary(hf.t, hf.n, rd, seed)) {
        let voxel_size = cam.voxel_params.x;
        let hp = ro + rd * hf.t;
        let cell = grass_cell_from_world_global(hp, rd, voxel_size);
        let grass_bmin = vec3<f32>(cell.bmin_m.x, hp.y - voxel_size, cell.bmin_m.z);
        let gh = try_grass_slab_hit(
          ro,
          rd,
          0.0,
          hf.t,
          grass_bmin,
          cell.id_vox,
          voxel_size,
          cam.voxel_params.y,
          cam.voxel_params.z
        );
        if (gh.hit) {
          hf.t = gh.t;
          hf.n = gh.n;
        }
      }

      let shade = shade_heightfield(ro, rd, hf, sky_up, seed);
      t_store = shade.t_scene;
      textureStore(color_img, ip, vec4<f32>(shade.col, 1.0));
      textureStore(depth_img, ip, vec4<f32>(shade.t_scene, 0.0, 0.0, 0.0));
      textureStore(local_img, ip, vec4<f32>(local_out, local_w)); // alpha=0
      textureStore(primary_hist_out, ip, vec4<f32>(t_store, 0.0, 0.0, 0.0));
      shadow_hist_out[shadow_idx] = shadow_out;
      return;
    }

    // True sky pixel: now pay for full sky (clouds + sun)
    let sky = sky_color(rd);
    textureStore(color_img, ip, vec4<f32>(sky, 1.0));
    textureStore(depth_img, ip, vec4<f32>(FOG_MAX_DIST, 0.0, 0.0, 0.0));
    textureStore(local_img, ip, vec4<f32>(local_out, local_w)); // alpha=0
    textureStore(primary_hist_out, ip, vec4<f32>(t_store, 0.0, 0.0, 0.0));
    shadow_hist_out[shadow_idx] = shadow_out;
    return;
  }

  // ------------------------------------------------------------
  // Case 2: voxel grid present => voxels, then heightfield fallback, then sky
  // ------------------------------------------------------------
  var t_hist     : f32 = 0.0;
  var hist_valid : bool = false;
  var hist_anchor_key  : u32 = INVALID_U32;
  var hist_anchor_coord: vec3<i32> = vec3<i32>(0);
  let tile_candidate_count = WG_TILE_COUNT_CACHED;
  let has_tile_candidates = tile_candidate_count != 0u;
  let uv  = px / res;

  if (has_tile_candidates) {
    let hist_guess = textureLoad(primary_hist_tex, ip, 0);
    let t_hist_guess = hist_guess.x;
    if (t_hist_guess > 1e-3) {
      let p_ws = ro + rd * t_hist_guess;
      let uv_prev = prev_uv_from_world(p_ws);

      if (in_unit_square(uv_prev)) {
        let hist_prev = textureSampleLevel(primary_hist_tex, primary_hist_samp, uv_prev, 0.0);
        let t_prev = hist_prev.x;
        let rel = abs(t_prev - t_hist_guess) / max(t_hist_guess, 1e-3);
        let depth_ok = 1.0 - smoothstep(PRIMARY_HIT_DEPTH_REL0, PRIMARY_HIT_DEPTH_REL1, rel);
        let vel_px = length((uv_prev - uv) * res);
        let motion_ok = 1.0 - smoothstep(PRIMARY_HIT_MOTION_PX0, PRIMARY_HIT_MOTION_PX1, vel_px);
        if (t_prev > 1e-3 && depth_ok > 0.5 && motion_ok > 0.5) {
          t_hist = t_prev;
          hist_valid = true;
          let packed_z = bitcast<u32>(hist_prev.w);
          if ((packed_z & 0x80000000u) != 0u) {
            hist_anchor_key = bitcast<u32>(hist_prev.y);
            let packed_xy = bitcast<u32>(hist_prev.z);
            let xy = unpack_i16x2(packed_xy);
            let z = unpack_i16(packed_z);
            hist_anchor_coord = vec3<i32>(xy.x, xy.y, z);
          }
        }
      }
    }
  }

  let t_start = max(t_hist - PRIMARY_HIT_MARGIN, 0.0);
  let t_end   = min(t_hist + PRIMARY_HIT_WINDOW, FOG_MAX_DIST);
  var vt = VoxTraceResult(false, miss_hitgeom(), 0.0, false, INVALID_U32, vec3<i32>(0));
  if (hist_valid) {
    let vt_anchor = trace_scene_voxels_anchor(
      ro,
      rd,
      t_start,
      t_end,
      hist_anchor_coord,
      hist_anchor_key,
      seed
    );
    if (vt_anchor.best.hit != 0u) {
      vt = vt_anchor;
    }
  }

  if (vt.best.hit == 0u) {
    vt = trace_primary_voxels(
      ro,
      rd,
      t_start,
      t_end,
      has_tile_candidates,
      hist_valid,
      hist_anchor_key,
      hist_anchor_coord,
      tile_candidate_count,
      seed
    );
  }

  // Outside streamed grid => heightfield or sky
  if (!vt.in_grid) {
    var hf = clip_trace_heightfield(ro, rd, 0.0, FOG_MAX_DIST);

    if (hf.hit) {
      if (hf.mat == MAT_GRASS && ENABLE_GRASS && grass_allowed_primary(hf.t, hf.n, rd, seed)) {
        let voxel_size = cam.voxel_params.x;
        let hp = ro + rd * hf.t;
        let cell = grass_cell_from_world_global(hp, rd, voxel_size);
        let grass_bmin = vec3<f32>(cell.bmin_m.x, hp.y - voxel_size, cell.bmin_m.z);
        let gh = try_grass_slab_hit(
          ro,
          rd,
          0.0,
          hf.t,
          grass_bmin,
          cell.id_vox,
          voxel_size,
          cam.voxel_params.y,
          cam.voxel_params.z
        );
        if (gh.hit) {
          hf.t = gh.t;
          hf.n = gh.n;
        }
      }

      let shade = shade_heightfield(ro, rd, hf, sky_up, seed);
      t_store = shade.t_scene;
      textureStore(color_img, ip, vec4<f32>(shade.col, 1.0));
      textureStore(depth_img, ip, vec4<f32>(shade.t_scene, 0.0, 0.0, 0.0));
      textureStore(local_img, ip, vec4<f32>(local_out, local_w)); // alpha=0
      textureStore(primary_hist_out, ip, vec4<f32>(t_store, 0.0, 0.0, 0.0));
      shadow_hist_out[shadow_idx] = shadow_out;
      return;
    }

    let sky = sky_color(rd);
    textureStore(color_img, ip, vec4<f32>(sky, 1.0));
    textureStore(depth_img, ip, vec4<f32>(FOG_MAX_DIST, 0.0, 0.0, 0.0));
    textureStore(local_img, ip, vec4<f32>(local_out, local_w)); // alpha=0
    textureStore(primary_hist_out, ip, vec4<f32>(t_store, 0.0, 0.0, 0.0));
    shadow_hist_out[shadow_idx] = shadow_out;
    return;
  }

  // In grid: voxel hit?
  if (vt.best.hit != 0u) {
    if (vt.best.mat == MAT_GRASS && ENABLE_GRASS) {
      if (grass_allowed_primary(vt.best.t, vt.best.n, rd, seed)) {
        let voxel_size = cam.voxel_params.x;
        let chunk_size_vox = i32(cam.chunk_size);
        let ch_origin_vox = vec3<i32>(
          i32(floor(vt.best.root_bmin.x / voxel_size + 0.5)),
          i32(floor(vt.best.root_bmin.y / voxel_size + 0.5)),
          i32(floor(vt.best.root_bmin.z / voxel_size + 0.5))
        );
        let hp = ro + rd * vt.best.t;
        let cell = pick_grass_cell_in_chunk(
          hp,
          rd,
          vt.best.root_bmin,
          ch_origin_vox,
          voxel_size,
          chunk_size_vox
        );
        let gh = try_grass_slab_hit(
          ro,
          rd,
          0.0,
          vt.best.t,
          cell.bmin_m,
          cell.id_vox,
          voxel_size,
          cam.voxel_params.y,
          cam.voxel_params.z
        );
        if (gh.hit) {
          vt.best.t = gh.t;
          vt.best.n = gh.n;
        }
      }
    }

    let hp = ro + vt.best.t * rd;
    let hp_shadow = hp + vt.best.n * (0.75 * cam.voxel_params.x);

    let shadow_do = (seed & SHADOW_SUBSAMPLE_MASK) == 0u;
    if (shadow_do) {
      var shadow_hist = textureLoad(shadow_hist_in, ip, 0).x;
      let uv_prev = prev_uv_from_world(hp);
      if (in_unit_square(uv_prev)) {
        let prev_px = vec2<i32>(
          clamp(i32(uv_prev.x * f32(dims.x)), 0, i32(dims.x) - 1),
          clamp(i32(uv_prev.y * f32(dims.y)), 0, i32(dims.y) - 1)
        );
        shadow_hist = textureLoad(shadow_hist_in, prev_px, 0).x;
      }
      shadow_hist = clamp(shadow_hist, 0.0, 1.0);
      let shadow_cur = sun_transmittance_geom_only(hp_shadow, SUN_DIR);
      shadow_out = mix(shadow_hist, shadow_cur, SHADOW_TAA_ALPHA);
    } else {
      shadow_out = clamp(textureLoad(shadow_hist_in, ip, 0).x, 0.0, 1.0);
    }

    // Split shading (base + local)
    let sh = shade_hit_split(ro, rd, vt.best, sky_up, seed, shadow_out);

    let t_scene = min(vt.best.t, FOG_MAX_DIST);

    // Fog only the base surface term (view-space medium)
    let sky_bg_rd = sky_bg(rd);
    let col_base = apply_fog(sh.base_hdr, ro, rd, t_scene, sky_bg_rd);

    // Local is stored UNFOGGED for temporal accumulation
    local_out = sh.local_hdr;
    local_w   = sh.local_w;
    t_store   = t_scene;

    textureStore(color_img, ip, vec4<f32>(col_base, 1.0));
    textureStore(depth_img, ip, vec4<f32>(t_scene, 0.0, 0.0, 0.0));
    textureStore(local_img, ip, vec4<f32>(local_out, local_w)); // alpha = validity
    var packed_xy: u32 = 0u;
    var packed_z: u32 = 0u;
    var anchor_key: u32 = 0u;
    if (vt.anchor_valid) {
      packed_xy = pack_i16x2(vt.anchor_chunk.x, vt.anchor_chunk.y);
      packed_z = pack_i16x2(vt.anchor_chunk.z, 0);
      packed_z |= 0x80000000u;
      anchor_key = vt.anchor_key;
    }
    textureStore(
      primary_hist_out,
      ip,
      vec4<f32>(t_store, bitcast<f32>(anchor_key), bitcast<f32>(packed_xy), bitcast<f32>(packed_z))
    );
    shadow_hist_out[shadow_idx] = shadow_out;
    return;
  }

  // Voxel miss: try heightfield
  var hf = clip_trace_heightfield(ro, rd, 0.0, FOG_MAX_DIST);

  if (hf.hit) {
    if (hf.mat == MAT_GRASS && ENABLE_GRASS && grass_allowed_primary(hf.t, hf.n, rd, seed)) {
      let voxel_size = cam.voxel_params.x;
      let hp = ro + rd * hf.t;
      let cell = grass_cell_from_world_global(hp, rd, voxel_size);
      let grass_bmin = vec3<f32>(cell.bmin_m.x, hp.y - voxel_size, cell.bmin_m.z);
      let gh = try_grass_slab_hit(
        ro,
        rd,
        0.0,
        hf.t,
        grass_bmin,
        cell.id_vox,
        voxel_size,
        cam.voxel_params.y,
        cam.voxel_params.z
      );
      if (gh.hit) {
        hf.t = gh.t;
        hf.n = gh.n;
      }
    }

    let shade = shade_heightfield(ro, rd, hf, sky_up, seed);
    t_store = shade.t_scene;
    textureStore(color_img, ip, vec4<f32>(shade.col, 1.0));
    textureStore(depth_img, ip, vec4<f32>(shade.t_scene, 0.0, 0.0, 0.0));
    textureStore(local_img, ip, vec4<f32>(local_out, local_w)); // alpha=0
    textureStore(primary_hist_out, ip, vec4<f32>(t_store, 0.0, 0.0, 0.0));
    shadow_hist_out[shadow_idx] = shadow_out;
    return;
  }

  // True sky pixel: now compute full sky (clouds + sun)
  let sky = sky_color(rd);
  textureStore(color_img, ip, vec4<f32>(sky, 1.0));
  textureStore(depth_img, ip, vec4<f32>(FOG_MAX_DIST, 0.0, 0.0, 0.0));
  textureStore(local_img, ip, vec4<f32>(local_out, local_w)); // alpha=0
  textureStore(primary_hist_out, ip, vec4<f32>(t_store, 0.0, 0.0, 0.0));
  shadow_hist_out[shadow_idx] = shadow_out;
}


@compute @workgroup_size(8, 8, 1)
fn main_godray(@builtin(global_invocation_id) gid3: vec3<u32>) {
  let qdims = textureDimensions(godray_out);
  if (gid3.x >= qdims.x || gid3.y >= qdims.y) { return; }

  let gid = vec2<u32>(gid3.x, gid3.y);
  let hip = vec2<i32>(i32(gid.x), i32(gid.y));

  if (!ENABLE_GODRAYS) {
    textureStore(godray_out, hip, vec4<f32>(0.0));
    return;
  }

  let out_rgba = compute_godray_pixel(gid, depth_tex, godray_hist_tex, godray_hist_samp);
  textureStore(godray_out, hip, out_rgba);
}

@compute @workgroup_size(8, 8, 1)
fn main_composite(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_dims = textureDimensions(out_img);
  if (gid.x >= out_dims.x || gid.y >= out_dims.y) { return; }

  // present pixel center
  let px_present = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

  // mapped render integer pixel (used by depth-aware mapping logic)
  let ip_render = ip_render_from_present_px(px_present);

  // mapped render pixel center (float, in internal render space)
  let px_render = px_render_from_present_px(px_present);

  // Base composite (already fogged inside composite_pixel_mapped / via color_tex content)
  let outc = composite_pixel_mapped(
    ip_render, px_render,
    color_tex, godray_tex, godray_samp,
    depth_full
  );

  // Sample accumulated local lighting in the SAME internal render UV space.
  // local_hist_tex is expected to be internal render resolution (same as color_tex/depth_full).
  let dims_r = textureDimensions(color_tex);
  let inv_r  = vec2<f32>(1.0 / f32(dims_r.x), 1.0 / f32(dims_r.y));
  let uv_r   = px_render * inv_r;

  // local_hist holds HDR RGB (alpha ignored here, or you can use it as confidence later)
  let local_rgb = textureSampleLevel(local_hist_tex, local_samp, uv_r, 0.0).xyz;

  // WGSL can't assign to swizzles, so rebuild the vec4
  let rgb_final = outc.xyz + local_rgb;
  let outc_final = vec4<f32>(rgb_final, outc.w);

  let ip_out = vec2<i32>(i32(gid.x), i32(gid.y));
  textureStore(out_img, ip_out, outc_final);
}
