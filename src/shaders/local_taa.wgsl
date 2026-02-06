// src/shaders/ray/local_taa.wgsl
// ------------------------------------
// Temporal accumulation for local voxel light term.
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

@group(1) @binding(0) var local_in_tex   : texture_2d<f32>;
@group(1) @binding(2) var local_hist_out : texture_storage_2d<rgba32float, write>;

// Tunables live in common.wgsl.

@compute @workgroup_size(8, 8, 1)
fn main_local_taa(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(local_hist_out);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let ip = vec2<i32>(i32(gid.x), i32(gid.y));
  let uv = (vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5)) / vec2<f32>(f32(dims.x), f32(dims.y));

  let cur4  = textureSampleLevel(local_in_tex,   local_samp, uv, 0.0);
  let hist4 = textureSampleLevel(local_hist_tex, local_samp, uv, 0.0);

  let cur   = cur4.xyz;
  let w_in  = cur4.w;          // local_w in [0,1] meaning “valid this frame”
  let hist  = hist4.xyz;
  let conf0 = hist4.w;         // history confidence in [0,1] (we’ll maintain it)

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
