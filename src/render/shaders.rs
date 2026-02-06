




pub const RAY_CS_WGSL: &str = concat!(
    include_str!("../shaders/common.wgsl"),
    "\n",

    
    include_str!("../shaders/ray/clouds.wgsl"),
    "\n",
    include_str!("../shaders/ray/phase.wgsl"),
    "\n",
    include_str!("../shaders/ray/sky.wgsl"),
    "\n",
    include_str!("../shaders/ray/fog.wgsl"),
    "\n",
    include_str!("../shaders/ray/aabb.wgsl"),
    "\n",
    include_str!("../shaders/ray/wind.wgsl"),
    "\n",
    include_str!("../shaders/ray/svo_query.wgsl"),
    "\n",
    include_str!("../shaders/ray/leaves.wgsl"),
    "\n",
    include_str!("../shaders/ray/grass.wgsl"),
    "\n",
    include_str!("../shaders/ray/chunk_trace.wgsl"),
    "\n",
    include_str!("../shaders/ray/shadows.wgsl"),
    "\n",
    include_str!("../shaders/ray/shading.wgsl"),
    "\n",
    include_str!("../shaders/ray/godrays.wgsl"),
    "\n",
    include_str!("../shaders/ray/composite.wgsl"),
    "\n",
    include_str!("../shaders/ray/light.wgsl"),
    "\n",

    include_str!("../shaders/clipmap.wgsl"),
    "\n",
    include_str!("../shaders/ray_main.wgsl"),
    "\n",
    include_str!("../shaders/local_taa.wgsl"),
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
