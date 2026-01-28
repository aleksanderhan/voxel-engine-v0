// src/render/shaders.rs
// ---------------------

pub const RAY_CS_WGSL: &str = concat!(
    include_str!("../shaders/common.wgsl"),
    "\n",
    include_str!("../shaders/world_svo.wgsl"),
    "\n",
    include_str!("../shaders/ray_main.wgsl"),
    "\n",
);

pub const BLIT_WGSL: &str = include_str!("../shaders/blit.wgsl");

#[inline]
pub fn ray_cs_wgsl() -> &'static str {
    RAY_CS_WGSL
}

#[inline]
pub fn blit_wgsl() -> &'static str {
    BLIT_WGSL
}
