



fn phase_mie(costh: f32) -> f32 {
  let g = PHASE_G;
  let gg = g * g;
  let denom = pow(1.0 + gg - 2.0 * g * costh, 1.5);
  return INV_4PI * (1.0 - gg) / max(denom, 1e-3);
}

fn phase_blended(costh: f32) -> f32 {
  let mie = phase_mie(costh);     
  let iso = INV_4PI;              
  return mix(iso, mie, PHASE_MIE_W);
}
