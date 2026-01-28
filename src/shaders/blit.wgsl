// src/shaders/blit.wgsl
// ---------------------
// blit.wgsl
//
// Minimal fullscreen blit + tiny HUD:
//
// 1) Vertex shader emits a single fullscreen triangle (no vertex buffers).
// 2) Fragment shader:
//    - samples the compute output texture
//    - (NO extra tonemap here; composite already outputs LDR)
//    - optionally draws a tiny 3x5-pixel-font FPS overlay in the top-right.
//
// Bindings (must match Rust bind group layout `layouts.blit`):
//   @group(0) @binding(0): sampled output texture (img)
//   @group(0) @binding(1): sampler (samp)
//   @group(0) @binding(2): uniform overlay struct (fps + dimensions)

@group(0) @binding(0) var img : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;

struct Overlay {
  fps    : u32,
  width  : u32,
  height : u32,
  _pad0  : u32,
};
@group(0) @binding(2) var<uniform> overlay : Overlay;

// -----------------------------------------------------------------------------
// 3x5 digit font helpers
// -----------------------------------------------------------------------------

fn digit_mask(d: u32) -> u32 {
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
  let bit = y * 3u + x;
  return ((mask >> bit) & 1u) != 0u;
}

// -----------------------------------------------------------------------------
// Fullscreen triangle vertex shader
// -----------------------------------------------------------------------------

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VSOut {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );

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

// -----------------------------------------------------------------------------
// Fragment shader: sample + FPS overlay
// -----------------------------------------------------------------------------

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>, @location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  // Composite already outputs LDR; don't tonemap again.
  let c = textureSample(img, samp, uv);
  var rgb = c.rgb;

  // ---- FPS overlay (top-right) ----

  let dims = vec2<u32>(overlay.width, overlay.height);
  let px = vec2<u32>(u32(frag_pos.x), u32(frag_pos.y));

  let scale: u32 = 8u;
  let digit_w: u32 = 3u * scale;
  let digit_h: u32 = 5u * scale;
  let gap: u32 = 1u * scale;
  let margin: u32 = 12u;

  let num_digits: u32 = 4u;
  let total_w: u32 = num_digits * digit_w + (num_digits - 1u) * gap;

  let ox_i: i32 = i32(dims.x) - i32(margin) - i32(total_w);
  let oy_i: i32 = i32(margin);

  let origin_x: u32 = u32(max(ox_i, 0));
  let origin_y: u32 = u32(max(oy_i, 0));

  var v: u32 = min(overlay.fps, 9999u);
  let d0: u32 = v % 10u; v = v / 10u;
  let d1: u32 = v % 10u; v = v / 10u;
  let d2: u32 = v % 10u; v = v / 10u;
  let d3: u32 = v % 10u;

  if (px.x >= origin_x && px.x < origin_x + total_w && px.y >= origin_y && px.y < origin_y + digit_h) {
    let local_x = px.x - origin_x;
    let local_y = px.y - origin_y;

    let stride = digit_w + gap;
    let digit_i = local_x / stride;
    let in_digit_x = local_x % stride;

    if (in_digit_x < digit_w) {
      let cell_x = (in_digit_x / scale);
      let cell_y = (local_y / scale);

      var dig: u32 = 0u;
      if (digit_i == 0u) { dig = d3; }
      if (digit_i == 1u) { dig = d2; }
      if (digit_i == 2u) { dig = d1; }
      if (digit_i == 3u) { dig = d0; }

      let m = digit_mask(dig);
      if (mask_bit(m, cell_x, cell_y)) {
        rgb = vec3<f32>(1.0, 1.0, 1.0);
      }
    }
  }

  return vec4<f32>(rgb, 1.0);
}
