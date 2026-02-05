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
@group(1) @binding(1) var local_hist_tex : texture_2d<f32>;
@group(1) @binding(2) var local_hist_out : texture_storage_2d<rgba16float, write>;
@group(1) @binding(3) var local_samp     : sampler;

// Tune this: lower = steadier but slower response
const LOCAL_TAA_ALPHA : f32 = 0.12;

@compute @workgroup_size(8, 8, 1)
fn main_local_taa(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(local_hist_out);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let ip = vec2<i32>(i32(gid.x), i32(gid.y));
  let uv = (vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5)) / vec2<f32>(f32(dims.x), f32(dims.y));

  let cur4  = textureSampleLevel(local_in_tex,   local_samp, uv, 0.0);
  let hist4 = textureSampleLevel(local_hist_tex, local_samp, uv, 0.0);

  let cur   = cur4.xyz;
  let w     = cur4.w;          // local_w in [0,1]
  let hist  = hist4.xyz;

  // Gate history updates: if w == 0 => keep history (k = 0)
  // If w == 1 => normal blend (k = alpha)
  let has_sample = (w > 0.0);
  let k = select(0.0, LOCAL_TAA_ALPHA, has_sample);

  let out_local = mix(hist, cur, k);

  // Store alpha=1 just so history tex stays "valid" for sampling
  textureStore(local_hist_out, ip, vec4<f32>(out_local, 1.0));
}
