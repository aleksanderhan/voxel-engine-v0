// blit.wgsl
//
// Minimal fullscreen blit + tiny HUD:
//
// 1) Vertex shader emits a single fullscreen triangle (no vertex buffers).
//    This is a common trick to avoid cracks/precision issues on a fullscreen quad.
// 2) Fragment shader:
//    - samples the HDR-ish compute output texture
//    - applies a simple tonemap-like curve: c / (c + 1)
//    - optionally draws a tiny 3x5-pixel-font FPS overlay in the top-right.
//
// Bindings (must match Rust bind group layout `layouts.blit`):
//   @group(0) @binding(0): sampled output texture (img)
//   @group(0) @binding(1): sampler (samp)
//   @group(0) @binding(2): uniform overlay struct (fps + dimensions)

@group(0) @binding(0) var img : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;

// Overlay uniform block written from CPU each frame (or periodically for fps).
// `width/height` are used to place the overlay in screen space.
// `_pad0` keeps the struct 16-byte aligned (nice for uniform layout rules).
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
//
// Digits are encoded as a 3x5 bitmap packed into 15 bits, row-major:
//   bit = y*3 + x
// with (x,y) = (0,0) being the top-left pixel of the digit cell.
//
// Example for "0":
//   111
//   101
//   101
//   101
//   111
//
// The returned constants are the 15-bit masks for digits 0..9.

fn digit_mask(d: u32) -> u32 {
  // 3x5 digits packed into 15 bits (row-major), bit0 = top-left.
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

// Return true if the digit mask has a "pixel on" at (x,y) in a 3x5 grid.
fn mask_bit(mask: u32, x: u32, y: u32) -> bool {
  // x in [0..2], y in [0..4]
  let bit = y * 3u + x;
  return ((mask >> bit) & 1u) != 0u;
}


// -----------------------------------------------------------------------------
// Fullscreen triangle vertex shader
// -----------------------------------------------------------------------------
//
// Output:
// - @builtin(position): clip-space position
// - @location(0): UV in [0..1] (with overshoot values on the other two verts)
//   that interpolates correctly across the fullscreen triangle.

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VSOut {
  // Fullscreen triangle in clip space.
  //
  // (-1,-1) is bottom-left, (3,-1) and (-1,3) extend beyond the screen so the
  // triangle fully covers the viewport after clipping.
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );

  // UVs chosen so the interpolated UV lands in [0..1] over the visible region.
  // This avoids needing a quad/indices/vertex buffer.
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
// Fragment shader: sample + tonemap + FPS overlay
// -----------------------------------------------------------------------------

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>, @location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {

  // Sample the renderer's final output texture.
  let c = textureSample(img, samp, uv);

  // Simple tonemap-ish curve:
  // - keeps values in [0..1)
  // - preserves highlight detail somewhat for HDR-ish inputs
  var rgb = c.rgb / (c.rgb + vec3<f32>(1.0));

  // ---- FPS overlay (top-right) ----
  //
  // The overlay draws up to 4 digits (clamped to 9999) using the 3x5 font,
  // scaled by `scale`, with a small margin from the top-right corner.
  //
  // Coordinate note:
  // `@builtin(position) frag_pos` is in framebuffer pixel coordinates.
  // This code treats (0,0) as the *top-left* for overlay placement by using
  // the provided overlay.height/width and direct pixel comparisons.

  // Screen size passed from CPU.
  let dims = vec2<u32>(overlay.width, overlay.height);

  // Current fragment coordinates as integer pixels.
  let px = vec2<u32>(u32(frag_pos.x), u32(frag_pos.y));

  // Layout parameters (edit to taste)
  let scale: u32 = 8u;     // size multiplier for each font pixel
  let digit_w: u32 = 3u * scale;
  let digit_h: u32 = 5u * scale;
  let gap: u32 = 1u * scale;
  let margin: u32 = 12u;

  // We always reserve 4 digits worth of space (right-aligned by placement).
  let num_digits: u32 = 4u;
  let total_w: u32 = num_digits * digit_w + (num_digits - 1u) * gap;

  // Compute overlay origin in signed integer space first to avoid u32 underflow.
  // This allows small windows where (dims.x - margin - total_w) would go negative.
  let ox_i: i32 = i32(dims.x) - i32(margin) - i32(total_w);
  let oy_i: i32 = i32(margin);

  // Clamp origin to >= 0 to keep comparisons valid.
  let origin_x: u32 = u32(max(ox_i, 0));
  let origin_y: u32 = u32(max(oy_i, 0));

  // Extract 4 digits from FPS (clamp to 0..9999).
  // d0 is ones, d3 is thousands.
  var v: u32 = min(overlay.fps, 9999u);
  let d0: u32 = v % 10u; v = v / 10u;
  let d1: u32 = v % 10u; v = v / 10u;
  let d2: u32 = v % 10u; v = v / 10u;
  let d3: u32 = v % 10u;

  // Overlay bounds test: only do digit math if we're inside the overlay rectangle.
  if (px.x >= origin_x && px.x < origin_x + total_w && px.y >= origin_y && px.y < origin_y + digit_h) {
    // Local pixel coords relative to the overlay origin.
    let local_x = px.x - origin_x;
    let local_y = px.y - origin_y;

    // Each digit occupies digit_w, followed by gap (except after last digit).
    let stride = digit_w + gap;

    // Which digit column is this pixel in? (0 = leftmost).
    let digit_i = local_x / stride;

    // Position within the digit+gap region.
    let in_digit_x = local_x % stride;

    // If we're in the gap column area, do nothing (leave underlying rgb).
    if (in_digit_x < digit_w) {
      // Map pixel coords into the 3x5 cell coordinates.
      let cell_x = (in_digit_x / scale); // 0..2
      let cell_y = (local_y / scale);    // 0..4

      // Choose the digit value for this column.
      // digit_i = 0 is leftmost, so we use d3..d0 to display thousands..ones.
      var dig: u32 = 0u;
      if (digit_i == 0u) { dig = d3; }
      if (digit_i == 1u) { dig = d2; }
      if (digit_i == 2u) { dig = d1; }
      if (digit_i == 3u) { dig = d0; }

      // If the corresponding font bit is set, paint the pixel white.
      let m = digit_mask(dig);
      if (mask_bit(m, cell_x, cell_y)) {
        rgb = vec3<f32>(1.0, 1.0, 1.0);
      }
    }
  }

  // Opaque output.
  return vec4<f32>(rgb, 1.0);
}
