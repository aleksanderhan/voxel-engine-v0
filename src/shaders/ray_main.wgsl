// src/shaders/ray_main.wgsl
//
// Compute entrypoints + pass bindings only.
// Depends on: common.wgsl + ray_core.wgsl + clipmap.wgsl

@group(0) @binding(4) var color_img : texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var depth_img : texture_storage_2d<r32float, write>;

@group(1) @binding(0) var depth_tex       : texture_2d<f32>;
@group(1) @binding(1) var godray_hist_tex : texture_2d<f32>;
@group(1) @binding(2) var godray_out      : texture_storage_2d<rgba16float, write>;
@group(1) @binding(3) var godray_hist_samp: sampler;

@group(2) @binding(0) var color_tex  : texture_2d<f32>;
@group(2) @binding(1) var godray_tex : texture_2d<f32>;
@group(2) @binding(2) var out_img    : texture_storage_2d<rgba16float, write>;
@group(2) @binding(3) var depth_full : texture_2d<f32>;
@group(2) @binding(4) var godray_samp: sampler;

var<workgroup> WG_SKY_UP : vec3<f32>;

@compute @workgroup_size(8, 8, 1)
fn main_primary(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32
) {
  let dims = textureDimensions(color_img);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  // Compute once per 8x8 workgroup
  if (lid == 0u) {
    WG_SKY_UP = sky_color_base(vec3<f32>(0.0, 1.0, 0.0));
  }
  workgroupBarrier();
  let sky_up = WG_SKY_UP;

  let res = vec2<f32>(f32(dims.x), f32(dims.y));
  let px  = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

  let ro  = cam.cam_pos.xyz;
  let rd  = ray_dir_from_pixel(px, res);

  let sky = sky_color(rd);

  let ip = vec2<i32>(i32(gid.x), i32(gid.y));

  // If no SVO chunks => only heightfield.
  if (cam.chunk_count == 0u) {
    let hf = clip_trace_heightfield(ro, rd, 0.0, FOG_MAX_DIST);
    let surface = select(sky, shade_clip_hit(ro, rd, hf, sky_up), hf.hit);
    let t_scene = select(FOG_MAX_DIST, min(hf.t, FOG_MAX_DIST), hf.hit);
    let col = apply_fog(surface, ro, rd, t_scene, sky);
    textureStore(color_img, ip, vec4<f32>(col, 1.0));
    textureStore(depth_img, ip, vec4<f32>(t_scene, 0.0, 0.0, 0.0));
    return;
  }

  let vt = trace_scene_voxels(ro, rd);

  // Outside streamed grid => only heightfield.
  if (!vt.in_grid) {
    let hf = clip_trace_heightfield(ro, rd, 0.0, FOG_MAX_DIST);
    let surface = select(sky, shade_clip_hit(ro, rd, hf, sky_up), hf.hit);
    let t_scene = select(FOG_MAX_DIST, min(hf.t, FOG_MAX_DIST), hf.hit);
    let col = apply_fog(surface, ro, rd, t_scene, sky);
    textureStore(color_img, ip, vec4<f32>(col, 1.0));
    textureStore(depth_img, ip, vec4<f32>(t_scene, 0.0, 0.0, 0.0));
    return;
  }

  // In grid: heightfield is ONLY needed if voxel miss.
  let use_vox = (vt.best.hit != 0u);

  var hf: ClipHit = ClipHit(false, BIG_F32, vec3<f32>(0.0), MAT_AIR);
  var use_hf: bool = false;
  if (!use_vox) {
    hf = clip_trace_heightfield(ro, rd, 0.0, FOG_MAX_DIST);
    use_hf = hf.hit;
  }

  let surface =
    select(sky,
      select(shade_clip_hit(ro, rd, hf, sky_up),
             shade_hit(ro, rd, vt.best, sky_up),
             use_vox),
      (use_vox || use_hf));

  let t_scene =
    select(min(vt.t_exit, FOG_MAX_DIST),
      select(min(hf.t, FOG_MAX_DIST),
             min(vt.best.t, FOG_MAX_DIST),
             use_vox),
      (use_vox || use_hf));

  let col = apply_fog(surface, ro, rd, t_scene, sky);

  textureStore(color_img, ip, vec4<f32>(col, 1.0));
  textureStore(depth_img, ip, vec4<f32>(t_scene, 0.0, 0.0, 0.0));
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

  // mapped render integer pixel
  let ip_render = ip_render_from_present_px(px_present);

  // mapped render pixel center (float)
  let px_render = px_render_from_present_px(px_present);

  let outc = composite_pixel_mapped(
    ip_render, px_render,
    color_tex, godray_tex, godray_samp,
    depth_full
  );

  let ip_out = vec2<i32>(i32(gid.x), i32(gid.y));
  textureStore(out_img, ip_out, outc);
}
