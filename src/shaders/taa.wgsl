// src/shaders/taa.wgsl
// --------------------
// Temporal accumulation passes.
//
// Passes:
// - local_taa: accumulates the local voxel light term.
// - composite_taa: full-frame reprojection + history blend.
//
// Input:
//   local_in_tex  = current frame noisy local term (rgb) + validity (a = local_w)
//   local_hist_tex = previous accumulated local history
//
// Output:
//   local_hist_out = new accumulated history
//
// KEY FIX:
//   When local_w == 0, do NOT decay/overwrite history. Keep history as-is.

@group(0) @binding(0) var local_in_tex   : texture_2d<f32>;
@group(0) @binding(1) var local_hist_in_tex : texture_2d<f32>;
@group(0) @binding(2) var local_hist_out : texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var local_taa_samp : sampler;

// Tunables live in common.wgsl.

@compute @workgroup_size(8, 8, 1)
fn main_local_taa(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(local_hist_out);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let ip = vec2<i32>(i32(gid.x), i32(gid.y));
  let uv = (vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5)) / vec2<f32>(f32(dims.x), f32(dims.y));

  let cur4  = textureSampleLevel(local_in_tex,      local_taa_samp, uv, 0.0);
  let hist4 = textureSampleLevel(local_hist_in_tex, local_taa_samp, uv, 0.0);

  let cur   = cur4.xyz;
  let w_in  = cur4.w;          // local_w in [0,1] meaning “valid this frame”
  let hist_ok = !(is_bad_vec3(hist4.xyz));
  let hist  = select(vec3<f32>(0.0), hist4.xyz, hist_ok);
  let conf0 = select(0.0, hist4.w, hist_ok); // history confidence in [0,1]

  if (!LOCAL_TAA_ENABLED) {
    let out = vec4<f32>(cur, select(0.0, 1.0, w_in > 0.0));
    textureStore(local_hist_out, ip, out);
    return;
  }

  // 1) Strong validity gate: reject bogus inputs even if w_in > 0
  //    (NaNs/Infs propagate horribly through TAA).
  let cur_ok = !(is_bad_vec3(cur));
  let has_sample = (w_in > 0.0) && cur_ok;

  // 2) Adaptive blend amount scaled by sample strength.
  //    If w_in is small (distance fade), integrate less.
  let k = select(0.0, LOCAL_TAA_ALPHA * clamp(w_in, 0.0, 1.0), has_sample);

  // 3) Only update color if we have a valid sample.
  let out_local = mix(hist, cur, k);

  // 4) Maintain confidence:
  //    - If we had a valid sample: confidence rises toward 1.
  //    - If not: confidence stays EXACTLY the same (critical correctness).
  //
  // The “rise” is matched to k so it behaves intuitively.
  let conf1 = select(conf0, min(1.0, conf0 + k), has_sample);

  textureStore(local_hist_out, ip, vec4<f32>(out_local, conf1));
}

// -----------------------------------------------------------------------------
// Composite TAA (full-frame)
// -----------------------------------------------------------------------------

@group(3) @binding(0) var composite_in_tex : texture_2d<f32>;
@group(3) @binding(1) var depth_full_tex   : texture_2d<f32>;
@group(3) @binding(2) var hist_in_tex      : texture_2d<f32>;
@group(3) @binding(3) var hist_samp        : sampler;
@group(3) @binding(4) var output_tex       : texture_storage_2d<rgba32float, write>;
@group(3) @binding(5) var hist_out_tex     : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main_composite_taa(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(output_tex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let ip_out = vec2<i32>(i32(gid.x), i32(gid.y));
  let px_present = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

  let ip_render = ip_render_from_present_px(px_present);
  let px_render = px_render_from_present_px(px_present);

  let cur = textureLoad(composite_in_tex, ip_out, 0).xyz;
  var taa_rgb = cur;

  if (COMPOSITE_TAA_ENABLED && cam.frame_index > 1u) {
    let depth = textureLoad(depth_full_tex, ip_render, 0).x;
    let depth_ok = depth > 1e-3 && depth < (FOG_MAX_DIST - 1e-3);

    let rd = ray_dir_from_pixel(px_render);
    let p_ws = cam.cam_pos.xyz + rd * depth;
    let uv_prev = prev_uv_from_world(p_ws);

    let dims_r = render_dims_f();
    let uv_cur = px_render / dims_r;
    let motion_px = max(abs(uv_prev.x - uv_cur.x) * dims_r.x, abs(uv_prev.y - uv_cur.y) * dims_r.y);
    let motion_ok = motion_px <= 1.5;

    if (depth_ok && motion_ok && in_unit_square(uv_prev)) {
      let hist_rgb = textureSampleLevel(hist_in_tex, hist_samp, uv_prev, 0.0).xyz;
      taa_rgb = mix(hist_rgb, taa_rgb, COMPOSITE_TAA_ALPHA);
    }
  }

  textureStore(output_tex, ip_out, vec4<f32>(taa_rgb, 1.0));
  textureStore(hist_out_tex, ip_out, vec4<f32>(taa_rgb, 1.0));
}
