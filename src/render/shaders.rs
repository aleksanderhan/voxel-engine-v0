// src/render/shaders.rs
//
// Centralized shader sources. WGSL has no native include mechanism in wgpu,
// so we concatenate multiple WGSL files into a single source string.

pub const RAY_CS_WGSL: &str = concat!(
    include_str!("../shaders/common.wgsl"),
    "\n",
    include_str!("../shaders/ray_core.wgsl"),
    "\n",
    include_str!("../shaders/ray_main.wgsl"),
    "\n",
);

pub const BLIT_WGSL: &str = include_str!("../shaders/blit.wgsl");

// Optional function wrappers (keeps call sites like shaders::ray_cs_wgsl()).
#[inline]
pub fn ray_cs_wgsl() -> &'static str {
    RAY_CS_WGSL
}

#[inline]
pub fn blit_wgsl() -> &'static str {
    BLIT_WGSL
}
