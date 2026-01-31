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
fn main_primary(@builtin(global_invocation_id) gid: vec3<u32>) {
  // pixel coords
  let ip: vec2<i32> = vec2<i32>(i32(gid.x), i32(gid.y));
  let dims: vec2<i32> = vec2<i32>(textureDimensions(color_img));

  if (ip.x < 0 || ip.y < 0 || ip.x >= dims.x || ip.y >= dims.y) {
    return;
  }

  // normalized pixel coords
  let uv: vec2<f32> = (vec2<f32>(f32(ip.x) + 0.5, f32(ip.y) + 0.5) / vec2<f32>(f32(dims.x), f32(dims.y)));

  // build primary ray (no get_primary_ray needed)
  let ro: vec3<f32> = cam.cam_pos.xyz;
  let res: vec2<f32> = vec2<f32>(f32(dims.x), f32(dims.y));
  let px_center: vec2<f32> = uv * res;              // == (ip + 0.5)
  let rd: vec3<f32> = ray_dir_from_pixel(px_center, res);


  // sky
  let sky_up: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);
 // or whatever you use (fallback: vec3(0,1,0))
  let sky: vec3<f32>    = sky_color(rd);    // <-- must exist

  // Trace the streamed voxel scene (your existing function)
  let vt = trace_scene_voxels(ro, rd);      // <-- must exist; should provide vt.in_grid, vt.t_exit, vt.best.hit, vt.best.t

  // Always trace balls so they can appear both inside and outside the streamed grid
  let bh = trace_balls(ro, rd, 0.0, FOG_MAX_DIST); // <-- must exist (from earlier)
  let use_ball: bool = bh.hit;

  // If we're outside the streamed grid, you typically used heightfield only.
  // We now pick nearest of: heightfield, balls, or sky.
  if (!vt.in_grid) {
    let hf = clip_trace_heightfield(ro, rd, 0.0, FOG_MAX_DIST); // <-- must exist; should provide hf.hit, hf.t, hf.n, hf.mat

    var t_scene: f32 = FOG_MAX_DIST;
    var surface: vec3<f32> = sky;

    // heightfield candidate
    if (hf.hit && hf.t < t_scene) {
      t_scene = hf.t;
      surface = shade_clip_hit(ro, rd, hf, sky_up); // <-- must exist
    }

    // ball candidate
    if (use_ball && bh.t < t_scene) {
      t_scene = bh.t;
      surface = shade_ball_hit(ro, rd, bh, sky_up); // <-- must exist (from earlier)
    }

    let col = apply_fog(surface, ro, rd, t_scene, sky); // <-- must exist
    textureStore(color_img, ip, vec4<f32>(col, 1.0));
    textureStore(depth_img, ip, vec4<f32>(t_scene, 0.0, 0.0, 0.0));
    return;
  }

  // Inside streamed grid:
  // You previously resolved: voxel hit if any, else heightfield, else sky.
  // Now we resolve nearest of: voxel, heightfield (only if no voxel or if you want both), balls, sky.

  let use_vox: bool = (vt.best.hit != 0u);

  // Only trace heightfield when you need it (matches your old behavior).
  // If you want heightfield to compete even when a voxel hit exists, remove the "if (!use_vox)" guard.
  var hf: ClipHit = ClipHit(false, BIG_F32, vec3<f32>(0.0), MAT_AIR);
  var use_hf: bool = false;
  if (!use_vox) {
    hf = clip_trace_heightfield(ro, rd, 0.0, FOG_MAX_DIST);
    use_hf = hf.hit;
  }

  // Start with sky as default. Keep your old "t_exit" behavior if that matters for fog/depth.
  // If you used vt.t_exit as "scene depth" for fog when no hit, preserve it:
  var t_scene: f32 = min(vt.t_exit, FOG_MAX_DIST);
  var surface: vec3<f32> = sky;

  // voxel candidate
  if (use_vox) {
    let tv = min(vt.best.t, FOG_MAX_DIST);
    if (tv < t_scene) {
      t_scene = tv;
      surface = shade_hit(ro, rd, vt.best, sky_up); // <-- must exist
    }
  }

  // heightfield candidate
  if (use_hf) {
    let th = min(hf.t, FOG_MAX_DIST);
    if (th < t_scene) {
      t_scene = th;
      surface = shade_clip_hit(ro, rd, hf, sky_up);
    }
  }

  // ball candidate
  if (use_ball) {
    let tb = min(bh.t, FOG_MAX_DIST);
    if (tb < t_scene) {
      t_scene = tb;
      surface = shade_ball_hit(ro, rd, bh, sky_up);
    }
  }

  // fog + store
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
