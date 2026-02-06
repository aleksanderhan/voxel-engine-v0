
#[inline]
pub fn hash_u32(mut v: u32) -> u32 {
    v ^= v >> 16;
    v = v.wrapping_mul(0x7feb_352d);
    v ^= v >> 15;
    v = v.wrapping_mul(0x846c_a68b);
    v ^= v >> 16;
    v
}

#[inline]
pub fn hash2(seed: u32, x: i32, z: i32) -> u32 {
    let a = (x as u32).wrapping_mul(0x9e37_79b1);
    let b = (z as u32).wrapping_mul(0x85eb_ca6b);
    hash_u32(seed ^ a ^ b)
}

#[inline]
pub fn hash3(seed: u32, x: i32, y: i32, z: i32) -> u32 {
    let a = (x as u32).wrapping_mul(0x9e37_79b1);
    let b = (y as u32).wrapping_mul(0x85eb_ca6b);
    let c = (z as u32).wrapping_mul(0xc2b2_ae35);
    hash_u32(seed ^ a ^ b ^ c)
}

#[inline]
pub fn u01(v: u32) -> f32 {
    (v as f32) * (1.0 / 4294967296.0)
}

#[inline(always)]
pub fn s11(n: u32) -> f32 {
    (u01(n) - 0.5) * 2.0
}
