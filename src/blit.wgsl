// blit.wgsl
//
// Minimal fullscreen triangle blit:
// - Vertex shader outputs a single triangle covering the screen.
// - Fragment shader samples the compute output texture and applies a tiny tonemap.

@group(0) @binding(0) var img : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;

struct Overlay {
  fps    : u32,
  width  : u32,
  height : u32,
  _pad0  : u32,
};
@group(0) @binding(2) var<uniform> overlay : Overlay;


fn digit_mask(d: u32) -> u32 {
  // 3x5 digits packed into 15 bits (row-major), bit0 = top-left.
  // Correct masks for this packing (no X-flip needed).
  if (d == 0u) { return 0x7B6Fu; } // 111 101 101 101 111
  if (d == 1u) { return 0x749Au; } // 010 110 010 010 111
  if (d == 2u) { return 0x73E7u; } // 111 001 111 100 111
  if (d == 3u) { return 0x79E7u; } // 111 001 111 001 111
  if (d == 4u) { return 0x49EDu; } // 101 101 111 001 001
  if (d == 5u) { return 0x79CFu; } // 111 100 111 001 111
  if (d == 6u) { return 0x7BCFu; } // 111 100 111 101 111
  if (d == 7u) { return 0x4927u; } // 111 001 001 001 001
  if (d == 8u) { return 0x7BEFu; } // 111 101 111 101 111
  if (d == 9u) { return 0x79EFu; } // 111 101 111 001 111
  return 0u;
}


fn mask_bit(mask: u32, x: u32, y: u32) -> bool {
  // x in [0..2], y in [0..4]
  let bit = y * 3u + x;
  return ((mask >> bit) & 1u) != 0u;
}


struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VSOut {
  // Fullscreen triangle in clip space.
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );

  // UVs chosen so the triangle maps to [0..1] over the screen.
  var uv = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(2.0, 1.0),
    vec2<f32>(0.0, -1.0)
  );

  var o: VSOut;
  o.pos = vec4<f32>(p[i], 0.0, 1.0);
  o.uv = uv[i];
  return o;
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>, @location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {

  let c = textureSample(img, samp, uv);

  // Tiny tonemap-ish curve to keep HDR-ish values visible.
  var rgb = c.rgb / (c.rgb + vec3<f32>(1.0));

  // ---- FPS overlay (top-right) ----
  let dims = vec2<u32>(overlay.width, overlay.height);

  // Fragment pixel coordinates (top-left origin).
  let px = vec2<u32>(u32(frag_pos.x), u32(frag_pos.y));

  // Layout parameters
  let scale: u32 = 8u;     // change this to taste
  let digit_w: u32 = 3u * scale;
  let digit_h: u32 = 5u * scale;
  let gap: u32 = 1u * scale;
  let margin: u32 = 12u;

  let num_digits: u32 = 4u;
  let total_w: u32 = num_digits * digit_w + (num_digits - 1u) * gap;

  // Compute origin in signed space to avoid u32 underflow/wrap.
  let ox_i: i32 = i32(dims.x) - i32(margin) - i32(total_w);
  let oy_i: i32 = i32(margin);

  let origin_x: u32 = u32(max(ox_i, 0));
  let origin_y: u32 = u32(max(oy_i, 0));

  // Extract 4 digits from FPS (clamp to 9999)
  var v: u32 = min(overlay.fps, 9999u);
  let d0: u32 = v % 10u; v = v / 10u;
  let d1: u32 = v % 10u; v = v / 10u;
  let d2: u32 = v % 10u; v = v / 10u;
  let d3: u32 = v % 10u;


  // Check if we're inside the overlay rectangle
  if (px.x >= origin_x && px.x < origin_x + total_w && px.y >= origin_y && px.y < origin_y + digit_h) {
    // Determine which digit column we're in
    let local_x = px.x - origin_x;
    let local_y = px.y - origin_y;

    let stride = digit_w + gap;
    let digit_i = local_x / stride;

    // Inside the gap area? skip
    let in_digit_x = local_x % stride;
    if (in_digit_x < digit_w) {
      // map to 3x5 cell
      let cell_x = (in_digit_x / scale); // 0..2
      let cell_y = (local_y / scale);    // 0..4

      var dig: u32 = 0u;
      // digit_i = 0 is leftmost, so use d3..d0
      if (digit_i == 0u) { dig = d3; }
      if (digit_i == 1u) { dig = d2; }
      if (digit_i == 2u) { dig = d1; }
      if (digit_i == 3u) { dig = d0; }


      let m = digit_mask(dig);
      if (mask_bit(m, cell_x, cell_y)) {
        // Draw white digits (you can change this)
        rgb = vec3<f32>(1.0, 1.0, 1.0);
      }
    }
  }

  return vec4<f32>(rgb, 1.0);
}

