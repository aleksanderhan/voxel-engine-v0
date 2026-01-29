// src/shaders/blit.wgsl
// ---------------------
// Minimal fullscreen blit + tiny HUD (FPS) optimized for speed:
// A) No digit work unless pixel is inside HUD rect
// B) HUD layout + digits are precomputed on CPU and passed in a uniform
// C) Digit masks are a LUT (no if-chain)
//
// Bindings (match Rust bind group layout `layouts.blit`):
//   @group(0) @binding(0): sampled output texture (img)
//   @group(0) @binding(1): sampler (samp)
//   @group(0) @binding(2): uniform overlay struct (packed digits + HUD layout)

@group(0) @binding(0) var img : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;

// CPU-precomputed overlay:
// - digits_packed = d0 | d1<<8 | d2<<16 | d3<<24  (d0=ones, d3=thousands)
// - origin_x/origin_y/total_w/digit_h/scale/stride are computed on CPU
struct Overlay {
  digits_packed : u32,
  origin_x : u32,
  origin_y : u32,
  total_w  : u32,

  digit_h  : u32,
  scale    : u32,
  stride   : u32,
  _pad0    : u32,
};
@group(0) @binding(2) var<uniform> overlay : Overlay;


// -----------------------------------------------------------------------------
// 3x5 digit font helpers (LUT)
// -----------------------------------------------------------------------------

fn digit_mask(d: u32) -> u32 {
  switch (d) {
    case 0u: { return 0x7B6Fu; }
    case 1u: { return 0x749Au; }
    case 2u: { return 0x73E7u; }
    case 3u: { return 0x79E7u; }
    case 4u: { return 0x49EDu; }
    case 5u: { return 0x79CFu; }
    case 6u: { return 0x7BCFu; }
    case 7u: { return 0x4927u; }
    case 8u: { return 0x7BEFu; }
    case 9u: { return 0x79EFu; }
    default: { return 0u; }
  }
}

fn mask_bit(mask: u32, x: u32, y: u32) -> bool {
  let bit = y * 3u + x;           // x in [0..2], y in [0..4]
  return ((mask >> bit) & 1u) != 0u;
}

fn unpack_digit(packed: u32, i: u32) -> u32 {
  // i: 0=d0 ones, 1=d1 tens, 2=d2 hundreds, 3=d3 thousands
  return (packed >> (8u * i)) & 0xFFu;
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
    vec2<f32>(0.0,  1.0),
    vec2<f32>(2.0,  1.0),
    vec2<f32>(0.0, -1.0)
  );

  var o: VSOut;
  o.pos = vec4<f32>(p[i], 0.0, 1.0);
  o.uv  = uv[i];
  return o;
}

// -----------------------------------------------------------------------------
// Fragment shader: sample + tonemap + FPS HUD
// -----------------------------------------------------------------------------

@fragment
fn fs_main(
  @builtin(position) frag_pos: vec4<f32>,
  @location(0) uv: vec2<f32>
) -> @location(0) vec4<f32> {

  // Sample the renderer's final output texture.
  let c = textureSample(img, samp, uv);

  // Simple tonemap-ish curve: c / (c + 1)
  var rgb = c.rgb / (c.rgb + vec3<f32>(1.0));

  // ---- FPS overlay ----
  // Early bounds test: if not inside the HUD rect, do nothing (fast path).
  let px = vec2<u32>(u32(frag_pos.x), u32(frag_pos.y));

  let ox = overlay.origin_x;
  let oy = overlay.origin_y;

  if (px.x >= ox && px.x < (ox + overlay.total_w) &&
      px.y >= oy && px.y < (oy + overlay.digit_h)) {

    let local_x = px.x - ox;
    let local_y = px.y - oy;

    let scale   = overlay.scale;
    let stride  = overlay.stride;      // digit_w + gap
    let digit_w = 3u * scale;

    // digit_i: 0..3 left->right
    let digit_i = local_x / stride;

    // x within digit+gap region
    let in_x = local_x - digit_i * stride;

    // inside digit area (not gap) and in range
    if (digit_i < 4u && in_x < digit_w) {
      let cell_x = in_x    / scale;    // 0..2
      let cell_y = local_y / scale;    // 0..4

      if (cell_y < 5u) {
        // thousands..ones left->right: digit_i=0 => d3, digit_i=3 => d0
        let which = 3u - digit_i;      // maps to packed index (0..3)
        let dig   = unpack_digit(overlay.digits_packed, which);
        let m     = digit_mask(dig);

        if (mask_bit(m, cell_x, cell_y)) {
          rgb = vec3<f32>(1.0, 1.0, 1.0);
        }
      }
    }
  }

  return vec4<f32>(rgb, 1.0);
}
