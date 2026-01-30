// src/shaders/ray_main.wgsl
//
// Compute entrypoints + pass bindings only.
// Depends on: common.wgsl + ray_core.wgsl + clipmap.wgsl

@group(0) @binding(4) var color_img : texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var depth_img : texture_storage_2d<r32float, write>;

@group(1) @binding(0) var depth_tex       : texture_2d<f32>;
@group(1) @binding(1) var godray_hist_tex : texture_2d<f32>;
@group(1) @binding(2) var godray_out      : texture_storage_2d<rgba16float, write>;

@group(2) @binding(0) var color_tex  : texture_2d<f32>;
@group(2) @binding(1) var godray_tex : texture_2d<f32>;
@group(2) @binding(2) var out_img    : texture_storage_2d<rgba16float, write>;
@group(2) @binding(3) var depth_full : texture_2d<f32>;
@group(2) @binding(4) var godray_samp: sampler;


@compute @workgroup_size(8, 8, 1)
fn main_primary(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(color_img);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let res = vec2<f32>(f32(dims.x), f32(dims.y));
  let px  = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

  let ro  = cam.cam_pos.xyz;
  let rd  = ray_dir_from_pixel(px, res);
  let sky = sky_color(rd);
  let sky_up = sky_color_base(vec3<f32>(0.0, 1.0, 0.0));

  let ip = vec2<i32>(i32(gid.x), i32(gid.y));

  // If no SVO chunks, still render clipmap terrain.
  if (cam.chunk_count == 0u) {
    let hf = clip_trace_heightfield(ro, rd, 0.0, FOG_MAX_DIST);
    let surface = select(sky, shade_clip_hit(ro, rd, hf, sky_up), hf.hit);
    let t_scene = select(FOG_MAX_DIST, min(hf.t, FOG_MAX_DIST), hf.hit);

    let col = apply_fog(surface, ro, rd, t_scene, sky);

    textureStore(color_img, ip, vec4<f32>(col, 1.0));
    textureStore(depth_img, ip, vec4<f32>(t_scene, 0.0, 0.0, 0.0));
    return;
  }

  // Streamed voxel grid trace
  let vt = trace_scene_voxels(ro, rd);

  // Outside streamed grid => clipmap fallback.
  if (!vt.in_grid) {
    let hf = clip_trace_heightfield(ro, rd, 0.0, FOG_MAX_DIST);
    let surface = select(sky, shade_clip_hit(ro, rd, hf, sky_up), hf.hit);
    let t_scene = select(FOG_MAX_DIST, min(hf.t, FOG_MAX_DIST), hf.hit);

    let col = apply_fog(surface, ro, rd, t_scene, sky);

    textureStore(color_img, ip, vec4<f32>(col, 1.0));
    textureStore(depth_img, ip, vec4<f32>(t_scene, 0.0, 0.0, 0.0));
    return;
  }

  // If no voxel hit, try heightfield clipmap fallback.
  let hf = clip_trace_heightfield(ro, rd, 0.0, FOG_MAX_DIST);

  let use_vox = (vt.best.hit != 0u);
  let use_hf  = (!use_vox) && hf.hit;

  let surface = select(
    sky,
    select(shade_clip_hit(ro, rd, hf, sky_up), shade_hit(ro, rd, vt.best, sky), use_vox),
    (use_vox || use_hf)
  );

  let t_scene = select(
    min(vt.t_exit, FOG_MAX_DIST),
    select(min(hf.t, FOG_MAX_DIST), min(vt.best.t, FOG_MAX_DIST), use_vox),
    (use_vox || use_hf)
  );

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

  let blended = compute_godray_quarter_pixel(gid, depth_tex, godray_hist_tex);
  textureStore(godray_out, hip, vec4<f32>(blended, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn main_composite(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(out_img);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let ip = vec2<i32>(i32(gid.x), i32(gid.y));
  let px = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

  let outc = composite_pixel(ip, px, color_tex, godray_tex, godray_samp, depth_full);
  textureStore(out_img, ip, outc);
}
