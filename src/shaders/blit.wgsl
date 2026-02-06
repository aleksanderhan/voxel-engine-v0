












@group(0) @binding(0) var img : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;




struct Overlay {
  
  digits_packed : u32,
  origin_x : u32,
  origin_y : u32,
  total_w  : u32,

  digit_h  : u32,
  scale    : u32,
  stride   : u32,

  
  text_len   : u32, 
  text_p0    : u32, 
  text_p1    : u32, 
  text_p2    : u32, 
};

@group(0) @binding(2) var<uniform> overlay : Overlay;





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
  let bit = y * 3u + x; 
  return ((mask >> bit) & 1u) != 0u;
}

fn unpack_digit(packed: u32, i: u32) -> u32 {
  
  return (packed >> (8u * i)) & 0xFFu;
}





fn pack_get_byte(p: u32, i: u32) -> u32 {
  
  return (p >> (8u * i)) & 0xFFu;
}

fn unpack_char(i: u32) -> u32 {
  
  if (i < 4u)  { return pack_get_byte(overlay.text_p0, i); }
  if (i < 8u)  { return pack_get_byte(overlay.text_p1, i - 4u); }
  if (i < 12u) { return pack_get_byte(overlay.text_p2, i - 8u); }
  return 0u;
}

fn pack3x5(r0: u32, r1: u32, r2: u32, r3: u32, r4: u32) -> u32 {
  
  
  
  
  
  
  
  return (r0 & 7u) | ((r1 & 7u) << 3u) | ((r2 & 7u) << 6u) | ((r3 & 7u) << 9u) | ((r4 & 7u) << 12u);
}

fn glyph_mask(code: u32) -> u32 {
  
  var c = code;
  if (c >= 97u && c <= 122u) { c = c - 32u; }

  switch (c) {
    
    case 32u: { return pack3x5(0u,0u,0u,0u,0u); }                
    case 45u: { return pack3x5(0u,0u,7u,0u,0u); }                
    case 58u: { return pack3x5(0u,2u,0u,2u,0u); }                

    
    case 48u: { return pack3x5(2u,5u,5u,5u,2u); }                
    case 49u: { return pack3x5(2u,6u,2u,2u,7u); }                
    case 50u: { return pack3x5(6u,1u,2u,4u,7u); }                
    case 51u: { return pack3x5(6u,1u,2u,1u,6u); }                
    case 52u: { return pack3x5(5u,5u,7u,1u,1u); }                
    case 53u: { return pack3x5(7u,4u,6u,1u,6u); }                
    case 54u: { return pack3x5(3u,4u,6u,5u,2u); }                
    case 55u: { return pack3x5(7u,1u,2u,2u,2u); }                
    case 56u: { return pack3x5(2u,5u,2u,5u,2u); }                
    case 57u: { return pack3x5(2u,5u,3u,1u,6u); }                

    
    case 65u: { return pack3x5(2u,5u,7u,5u,5u); }                
    case 67u: { return pack3x5(3u,4u,4u,4u,3u); }                
    case 68u: { return pack3x5(6u,5u,5u,5u,6u); }                
    case 69u: { return pack3x5(7u,4u,6u,4u,7u); }                
    case 71u: { return pack3x5(3u,4u,5u,5u,3u); }                
    case 72u: { return pack3x5(5u,5u,7u,5u,5u); }                
    case 73u: { return pack3x5(7u,2u,2u,2u,7u); }                
    case 76u: { return pack3x5(4u,4u,4u,4u,7u); }                
    case 79u: { return pack3x5(2u,5u,5u,5u,2u); }                
    case 80u: { return pack3x5(6u,5u,6u,4u,4u); }                
    case 82u: { return pack3x5(6u,5u,6u,5u,5u); }                
    case 83u: { return pack3x5(3u,4u,2u,1u,6u); }                
    case 84u: { return pack3x5(7u,2u,2u,2u,2u); }                
    case 78u: { return pack3x5(5u, 7u, 7u, 7u, 5u); } 
    case 87u: { return pack3x5(5u, 5u, 5u, 7u, 7u); } 
    
    default: { return 0u; }
  }
}

fn glyph_bit(mask: u32, x: u32, y: u32) -> bool {
  
  let bit = y * 3u + (2u - x);
  return ((mask >> bit) & 1u) != 0u;
}







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





@fragment
fn fs_main(
  @builtin(position) frag_pos: vec4<f32>,
  @location(0) uv: vec2<f32>
) -> @location(0) vec4<f32> {

  let uv_c = clamp(uv, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
  let c = textureSampleLevel(img, samp, uv_c, 0.0);

  var rgb = c.rgb; 

  
  
  let px = vec2<u32>(u32(frag_pos.x), u32(frag_pos.y));

  let ox = overlay.origin_x;
  let oy = overlay.origin_y;

  if (px.x >= ox && px.x < (ox + overlay.total_w) &&
      px.y >= oy && px.y < (oy + overlay.digit_h)) {

    let local_x = px.x - ox;
    let local_y = px.y - oy;

    let scale   = overlay.scale;
    let stride  = overlay.stride; 
    let digit_w = 3u * scale;

    
    let digit_i = local_x / stride;

    
    let in_x = local_x - digit_i * stride;

    
    if (digit_i < 4u && in_x < digit_w) {
      let cell_x = in_x    / scale; 
      let cell_y = local_y / scale; 

      if (cell_y < 5u) {
        
        let which = 3u - digit_i; 
        let dig   = unpack_digit(overlay.digits_packed, which);
        let m     = digit_mask(dig);

        if (mask_bit(m, cell_x, cell_y)) {
          rgb = vec3<f32>(1.0, 1.0, 1.0);
        }
      }
    }
  }

  
  let text_len_raw = overlay.text_len;
  let text_len = min(text_len_raw, 12u);

  if (text_len > 0u) {
    let scale  = overlay.scale;
    let char_w = 3u * scale;
    let char_h = 5u * scale;
    let gap    = scale;          
    let stride = char_w + gap;
    let margin = 2u * scale;

    let toy = overlay.origin_y + overlay.digit_h + margin;

    
    let text_w = text_len * stride - gap;

    
    let fps_right_i = i32(overlay.origin_x + overlay.total_w);
    let tox_i = max(0, fps_right_i - i32(text_w));
    let tox = u32(tox_i);

    if (px.x >= tox && px.x < (tox + text_w) &&
        px.y >= toy && px.y < (toy + char_h)) {

      let lx = px.x - tox;
      let ly = px.y - toy;

      let ci   = lx / stride;        
      let in_x = lx - ci * stride;   

      if (ci < text_len && in_x < char_w) {
        let cell_x = in_x / scale;   
        let cell_y = ly   / scale;   

        if (cell_y < 5u) {
          let cch = unpack_char(ci); 
          let m   = glyph_mask(cch);

          if (glyph_bit(m, cell_x, cell_y)) {
            rgb = vec3<f32>(1.0, 1.0, 1.0);
          }
        }
      }
    }
  }



  
  
  
  
  let dims = textureDimensions(img);
  let cx = i32(dims.x) / 2;
  let cy = i32(dims.y) / 2;

  let xi = i32(frag_pos.x);
  let yi = i32(frag_pos.y);

  
  let half_len  : i32 = 10; 
  let thickness : i32 = 1;  
  let gap       : i32 = 3;  

  let dx = abs(xi - cx);
  let dy = abs(yi - cy);

  
  if (dx <= half_len && dy <= half_len) {
    let on_vert = (dx <= thickness) && (dy <= half_len) && (dy >= gap);
    let on_horz = (dy <= thickness) && (dx <= half_len) && (dx >= gap);

    if (on_vert || on_horz) {
      rgb = vec3<f32>(1.0, 1.0, 1.0);
    }
  }

  return vec4<f32>(rgb, 1.0);
}
