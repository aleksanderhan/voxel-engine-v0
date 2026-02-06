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

@group(0) @binding(4) var color_img : texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var depth_img : texture_storage_2d<r32float, write>;

// local (noisy) voxel-light term output (HDR, NOT fogged)
@group(0) @binding(6) var local_img : texture_storage_2d<rgba32float, write>;

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

@compute @workgroup_size(8, 8, 1)
fn main_primary(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32
) {
  let dims = textureDimensions(color_img);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  // Compute once per 8x8 workgroup (already cheap: sky_bg only)
  if (lid == 0u) {
    WG_SKY_UP = sky_bg(vec3<f32>(0.0, 1.0, 0.0));
  }
  workgroupBarrier();
  let sky_up = WG_SKY_UP;

  let res = vec2<f32>(f32(dims.x), f32(dims.y));
  let px  = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

  let ro  = cam.cam_pos.xyz;
  let rd  = ray_dir_from_pixel(px, res);

  // CHEAP sky used for fog coloration (no clouds/sun march)
  let sky_bg_rd = sky_bg(rd);

  let ip = vec2<i32>(i32(gid.x), i32(gid.y));

  let frame = cam.frame_index;
  let seed  = (u32(gid.x) * 1973u) ^ (u32(gid.y) * 9277u) ^ (frame * 26699u);

  // Temporal max distance hint from last frame depth proxy.
  let uv_cur = px / res;
  let hist0 = textureSampleLevel(godray_hist_tex, godray_hist_samp, uv_cur, 0.0);
  var t_prev = hist0.w;

  if (t_prev > 1e-3) {
    let p_ws = ro + rd * t_prev;
    let uv_prev = prev_uv_from_world(p_ws);
    if (in_unit_square(uv_prev)) {
      let hist1 = textureSampleLevel(godray_hist_tex, godray_hist_samp, uv_prev, 0.0);
      if (hist1.w > 1e-3) {
        t_prev = hist1.w;
      }
    }
  }

  let voxel_size = cam.voxel_params.x;
  var vel_px = 0.0;
  var t_max_hint = FOG_MAX_DIST;
  if (t_prev > 1e-3) {
    vel_px = length((prev_uv_from_world(ro + rd * t_prev) - uv_cur) * res);
    let safety_margin = 4.0 * voxel_size + 0.5 * voxel_size * vel_px;
    let t_hint = min(FOG_MAX_DIST, t_prev + safety_margin);
    let trust = 1.0 - smoothstep(0.6, 2.2, vel_px);
    let clamp_ok = select(0.0, 1.0, t_prev < (0.95 * FOG_MAX_DIST));
    t_max_hint = mix(FOG_MAX_DIST, t_hint, trust * clamp_ok);
  }

  // Local output defaults: invalid (alpha=0) so TAA keeps history instead of blending black.
  var local_out = vec3<f32>(0.0);
  var local_w   : f32 = 0.0;

  // ------------------------------------------------------------
  // Case 1: no voxel chunks => heightfield or sky
  // ------------------------------------------------------------
  if (cam.chunk_count == 0u) {
    let hf = clip_trace_heightfield(ro, rd, 0.0, FOG_MAX_DIST);

    if (hf.hit) {
      let surface = shade_clip_hit(ro, rd, hf, sky_up, seed);
      let t_scene = min(hf.t, FOG_MAX_DIST);
      let col = apply_fog(surface, ro, rd, t_scene, sky_bg_rd);

      textureStore(color_img, ip, vec4<f32>(col, 1.0));
      textureStore(depth_img, ip, vec4<f32>(t_scene, 0.0, 0.0, 0.0));
      textureStore(local_img, ip, vec4<f32>(local_out, local_w)); // alpha=0
      return;
    }

    // True sky pixel: now pay for full sky (clouds + sun)
    let sky = sky_color(rd);
    textureStore(color_img, ip, vec4<f32>(sky, 1.0));
    textureStore(depth_img, ip, vec4<f32>(FOG_MAX_DIST, 0.0, 0.0, 0.0));
    textureStore(local_img, ip, vec4<f32>(local_out, local_w)); // alpha=0
    return;
  }

  // ------------------------------------------------------------
  // Case 2: voxel grid present => voxels, then heightfield fallback, then sky
  // ------------------------------------------------------------
  let vt = trace_scene_voxels(ro, rd, t_max_hint);

  // Outside streamed grid => heightfield or sky
  if (!vt.in_grid) {
    let hf = clip_trace_heightfield(ro, rd, 0.0, FOG_MAX_DIST);

    if (hf.hit) {
      let surface = shade_clip_hit(ro, rd, hf, sky_up, seed);
      let t_scene = min(hf.t, FOG_MAX_DIST);
      let col = apply_fog(surface, ro, rd, t_scene, sky_bg_rd);

      textureStore(color_img, ip, vec4<f32>(col, 1.0));
      textureStore(depth_img, ip, vec4<f32>(t_scene, 0.0, 0.0, 0.0));
      textureStore(local_img, ip, vec4<f32>(local_out, local_w)); // alpha=0
      return;
    }

    let sky = sky_color(rd);
    textureStore(color_img, ip, vec4<f32>(sky, 1.0));
    textureStore(depth_img, ip, vec4<f32>(FOG_MAX_DIST, 0.0, 0.0, 0.0));
    textureStore(local_img, ip, vec4<f32>(local_out, local_w)); // alpha=0
    return;
  }

  // In grid: voxel hit?
  if (vt.best.hit != 0u) {
    // Split shading (base + local)
    let sh = shade_hit_split(ro, rd, vt.best, sky_up, seed);

    let t_scene = min(vt.best.t, FOG_MAX_DIST);

    // Fog only the base surface term (view-space medium)
    let col_base = apply_fog(sh.base_hdr, ro, rd, t_scene, sky_bg_rd);

    // Local is stored UNFOGGED for temporal accumulation
    local_out = sh.local_hdr;
    local_w   = sh.local_w;

    textureStore(color_img, ip, vec4<f32>(col_base, 1.0));
    textureStore(depth_img, ip, vec4<f32>(t_scene, 0.0, 0.0, 0.0));
    textureStore(local_img, ip, vec4<f32>(local_out, local_w)); // alpha = validity
    return;
  }

  // Voxel miss: try heightfield
  let hf = clip_trace_heightfield(ro, rd, 0.0, FOG_MAX_DIST);

  if (hf.hit) {
    let surface = shade_clip_hit(ro, rd, hf, sky_up, seed);
    let t_scene = min(hf.t, FOG_MAX_DIST);
    let col = apply_fog(surface, ro, rd, t_scene, sky_bg_rd);

    textureStore(color_img, ip, vec4<f32>(col, 1.0));
    textureStore(depth_img, ip, vec4<f32>(t_scene, 0.0, 0.0, 0.0));
    textureStore(local_img, ip, vec4<f32>(local_out, local_w)); // alpha=0
    return;
  }

  // True sky pixel: now compute full sky (clouds + sun)
  let sky = sky_color(rd);
  textureStore(color_img, ip, vec4<f32>(sky, 1.0));
  textureStore(depth_img, ip, vec4<f32>(FOG_MAX_DIST, 0.0, 0.0, 0.0));
  textureStore(local_img, ip, vec4<f32>(local_out, local_w)); // alpha=0
}


@compute @workgroup_size(8, 8, 1)
fn main_godray(@builtin(global_invocation_id) gid3: vec3<u32>) {
  let qdims = textureDimensions(godray_out);
  if (gid3.x >= qdims.x || gid3.y >= qdims.y) { return; }

  let gid = vec2<u32>(gid3.x, gid3.y);
  let hip = vec2<i32>(i32(gid.x), i32(gid.y));

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
