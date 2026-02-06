


















@group(0) @binding(4) var color_img : texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var depth_img : texture_storage_2d<r32float, write>;


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

  
  if (lid == 0u) {
    WG_SKY_UP = sky_bg(vec3<f32>(0.0, 1.0, 0.0));
  }
  workgroupBarrier();
  let sky_up = WG_SKY_UP;

  let res = vec2<f32>(f32(dims.x), f32(dims.y));
  let px  = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

  let ro  = cam.cam_pos.xyz;
  let rd  = ray_dir_from_pixel(px, res);

  
  let sky_bg_rd = sky_bg(rd);

  let ip = vec2<i32>(i32(gid.x), i32(gid.y));

  let frame = cam.frame_index;
  let seed  = (u32(gid.x) * 1973u) ^ (u32(gid.y) * 9277u) ^ (frame * 26699u);

  
  var local_out = vec3<f32>(0.0);
  var local_w   : f32 = 0.0;

  
  
  
  if (cam.chunk_count == 0u) {
    let hf = clip_trace_heightfield(ro, rd, 0.0, FOG_MAX_DIST);

    if (hf.hit) {
      let surface = shade_clip_hit(ro, rd, hf, sky_up, seed);
      let t_scene = min(hf.t, FOG_MAX_DIST);
      let col = apply_fog(surface, ro, rd, t_scene, sky_bg_rd);

      textureStore(color_img, ip, vec4<f32>(col, 1.0));
      textureStore(depth_img, ip, vec4<f32>(t_scene, 0.0, 0.0, 0.0));
      textureStore(local_img, ip, vec4<f32>(local_out, local_w)); 
      return;
    }

    
    let sky = sky_color(rd);
    textureStore(color_img, ip, vec4<f32>(sky, 1.0));
    textureStore(depth_img, ip, vec4<f32>(FOG_MAX_DIST, 0.0, 0.0, 0.0));
    textureStore(local_img, ip, vec4<f32>(local_out, local_w)); 
    return;
  }

  
  
  
  let vt = trace_scene_voxels(ro, rd);

  
  if (!vt.in_grid) {
    let hf = clip_trace_heightfield(ro, rd, 0.0, FOG_MAX_DIST);

    if (hf.hit) {
      let surface = shade_clip_hit(ro, rd, hf, sky_up, seed);
      let t_scene = min(hf.t, FOG_MAX_DIST);
      let col = apply_fog(surface, ro, rd, t_scene, sky_bg_rd);

      textureStore(color_img, ip, vec4<f32>(col, 1.0));
      textureStore(depth_img, ip, vec4<f32>(t_scene, 0.0, 0.0, 0.0));
      textureStore(local_img, ip, vec4<f32>(local_out, local_w)); 
      return;
    }

    let sky = sky_color(rd);
    textureStore(color_img, ip, vec4<f32>(sky, 1.0));
    textureStore(depth_img, ip, vec4<f32>(FOG_MAX_DIST, 0.0, 0.0, 0.0));
    textureStore(local_img, ip, vec4<f32>(local_out, local_w)); 
    return;
  }

  
  if (vt.best.hit != 0u) {
    
    let sh = shade_hit_split(ro, rd, vt.best, sky_up, seed);

    let t_scene = min(vt.best.t, FOG_MAX_DIST);

    
    let col_base = apply_fog(sh.base_hdr, ro, rd, t_scene, sky_bg_rd);

    
    local_out = sh.local_hdr;
    local_w   = sh.local_w;

    textureStore(color_img, ip, vec4<f32>(col_base, 1.0));
    textureStore(depth_img, ip, vec4<f32>(t_scene, 0.0, 0.0, 0.0));
    textureStore(local_img, ip, vec4<f32>(local_out, local_w)); 
    return;
  }

  
  let hf = clip_trace_heightfield(ro, rd, 0.0, FOG_MAX_DIST);

  if (hf.hit) {
    let surface = shade_clip_hit(ro, rd, hf, sky_up, seed);
    let t_scene = min(hf.t, FOG_MAX_DIST);
    let col = apply_fog(surface, ro, rd, t_scene, sky_bg_rd);

    textureStore(color_img, ip, vec4<f32>(col, 1.0));
    textureStore(depth_img, ip, vec4<f32>(t_scene, 0.0, 0.0, 0.0));
    textureStore(local_img, ip, vec4<f32>(local_out, local_w)); 
    return;
  }

  
  let sky = sky_color(rd);
  textureStore(color_img, ip, vec4<f32>(sky, 1.0));
  textureStore(depth_img, ip, vec4<f32>(FOG_MAX_DIST, 0.0, 0.0, 0.0));
  textureStore(local_img, ip, vec4<f32>(local_out, local_w)); 
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

  
  let px_present = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

  
  let ip_render = ip_render_from_present_px(px_present);

  
  let px_render = px_render_from_present_px(px_present);

  
  let outc = composite_pixel_mapped(
    ip_render, px_render,
    color_tex, godray_tex, godray_samp,
    depth_full
  );

  
  
  let dims_r = textureDimensions(color_tex);
  let inv_r  = vec2<f32>(1.0 / f32(dims_r.x), 1.0 / f32(dims_r.y));
  let uv_r   = px_render * inv_r;

  
  let local_rgb = textureSampleLevel(local_hist_tex, local_samp, uv_r, 0.0).xyz;

  
  let rgb_final = outc.xyz + local_rgb;
  let outc_final = vec4<f32>(rgb_final, outc.w);

  let ip_out = vec2<i32>(i32(gid.x), i32(gid.y));
  textureStore(out_img, ip_out, outc_final);
}
