// src/render/state/bindgroups.rs
//
// Bind group creation.
//
// Bind groups are the concrete "argument packs" you bind before dispatching a
// compute pass or drawing a render pass. Each bind group must match a specific
// BindGroupLayout (the pipeline's expectation).
//
// This file is split out so resize() only needs to recreate textures and then
// rebuild bind groups (since bind groups reference texture views).

use super::{buffers::Buffers, layout::Layouts, textures::TextureSet};

/// All bind groups used by the renderer.
///
/// Naming convention:
/// - primary/scene are group(0) alternatives for different pipelines.
/// - godray/composite are additional groups for post processing.
/// - empty is a placeholder when a pipeline layout expects groups that a pass
///   doesn't logically use.
/// - blit is the final render pass bind group (sample output to swapchain).
pub struct BindGroups {
    /// group(0) for the primary compute pass.
    /// Includes storage outputs (color/depth) so the compute shader can write to them.
    pub primary: wgpu::BindGroup,

    /// group(0) for the godray compute pass.
    /// Excludes storage outputs to avoid binding the same textures as both storage + sampled.
    pub scene: wgpu::BindGroup,

    /// group(1) for godray ping-pong history:
    /// index selects which "direction" we are doing this frame (A->B or B->A).
    pub godray: [wgpu::BindGroup; 2],

    /// group(2) for composite:
    /// index selects which godray texture (A or B) we read when composing into `output`.
    pub composite: [wgpu::BindGroup; 2],

    /// Empty bind group used as a placeholder for group(0/1) in the composite pass.
    /// Some pipeline layouts are fixed to expect groups that a pass doesn't use.
    pub empty: wgpu::BindGroup,

    /// Bind group for the final blit render pass (texture + sampler + overlay uniform).
    pub blit: wgpu::BindGroup,
}

/// Create bind group for the primary compute pass.
///
/// Layout expectations_toggle (by binding index):
/// 0: camera uniform/storage buffer
/// 1: chunk buffer
/// 2: node buffer
/// 3: chunk grid buffer
/// 4: color texture view (storage texture the compute shader writes into)
/// 5: depth texture view (storage texture the compute shader writes into)
fn make_primary_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffers: &Buffers,
    textures: &TextureSet,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("primary_bg"),
        layout,
        entries: &[
            // Camera parameters (matrices, position, etc.)
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers.camera.as_entire_binding(),
            },
            // Chunk metadata buffer (streamed voxel/chunk data indexing)
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffers.chunk.as_entire_binding(),
            },
            // Node/acceleration structure data (whatever "node" represents in your scene)
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffers.node.as_entire_binding(),
            },
            // Grid mapping from world chunk coords -> buffer indices / residency info
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffers.chunk_grid.as_entire_binding(),
            },
            // Primary color output (storage texture view)
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&textures.color.view),
            },
            // Primary depth output (storage texture view)
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&textures.depth.view),
            },
        ],
    })
}

/// Create bind group for "scene" inputs that are common across passes.
///
/// This intentionally *does not* include storage outputs, so it can be reused in
/// compute passes that only read scene data and sample textures elsewhere.
fn make_scene_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffers: &Buffers,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scene_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers.camera.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffers.chunk.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffers.node.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffers.chunk_grid.as_entire_binding(),
            },
        ],
    })
}

/// Create bind group for the godray compute pass.
///
/// Bindings (by index):
/// 0: depth (read-only) - used for occlusion / ray marching limits.
/// 1: history texture (read-only) - previous godray accumulation.
/// 2: output texture (write) - current frame godray result.
///
/// The ping-pong scheme swaps which texture is "history" and which is "out"
/// each frame to avoid read/write hazards on the same texture.
fn make_godray_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    depth_view: &wgpu::TextureView,
    hist_view: &wgpu::TextureView,
    out_view: &wgpu::TextureView,
    label: &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(hist_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(out_view),
            },
        ],
    })
}

/// Create bind group for the composite compute pass.
///
/// Bindings (by index):
/// 0: base color texture (read-only) - the main scene color.
/// 1: godray texture (read-only) - chosen ping-pong buffer (A or B).
/// 2: output texture (write) - final composited frame texture.
fn make_composite_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    color_view: &wgpu::TextureView,
    godray_view: &wgpu::TextureView,
    output_view: &wgpu::TextureView,
    label: &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(color_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(godray_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(output_view),
            },
        ],
    })
}

/// Create bind group for the final blit render pass.
///
/// Bindings (by index):
/// 0: output texture (read-only) - full-screen texture to present.
/// 1: sampler - filtering and addressing for sampling output.
/// 2: overlay uniform buffer - small UI data (FPS/size) consumed by fragment shader.
fn make_blit_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    output_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    overlay_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blit_bg"),
        layout,
        entries: &[
            // Fullscreen texture to sample in the fragment shader.
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(output_view),
            },
            // Sampler used when sampling the output texture.
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            // Overlay uniforms (FPS, screen size, etc.).
            wgpu::BindGroupEntry {
                binding: 2,
                resource: overlay_buf.as_entire_binding(),
            },
        ],
    })
}

/// Top-level helper to (re)create all bind groups.
///
/// Called during initial renderer creation, and again after resize since texture
/// views change (new textures -> new views -> old bind groups become invalid).
pub fn create_bind_groups(
    device: &wgpu::Device,
    layouts: &Layouts,
    buffers: &Buffers,
    textures: &TextureSet,
    sampler: &wgpu::Sampler,
) -> BindGroups {
    // Primary includes storage outputs (color/depth), so it must NOT be used in godray.
    // Otherwise you'd risk binding the same resource in incompatible ways across passes.
    let primary = make_primary_bg(device, &layouts.primary, buffers, textures);

    // Scene excludes storage outputs; safe for passes that only need buffers.
    let scene = make_scene_bg(device, &layouts.scene, buffers);

    // Empty bind group used as placeholder for composite pipeline layout group(0/1).
    // (Composite pass likely only uses group(2), but the pipeline layout includes 0..2.)
    let empty = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("empty_bg"),
        layout: &layouts.empty,
        entries: &[],
    });

    // Godray ping-pong:
    // - godray[0] reads history=A, writes out=B
    // - godray[1] reads history=B, writes out=A
    //
    // `textures.godray` is a pair of textures used to avoid reading and writing the same
    // texture in a single dispatch.
    let godray = [
        make_godray_bg(
            device,
            &layouts.godray,
            &textures.depth.view,
            &textures.godray[0].view,
            &textures.godray[1].view,
            "godray_bg_a_to_b",
        ),
        make_godray_bg(
            device,
            &layouts.godray,
            &textures.depth.view,
            &textures.godray[1].view,
            &textures.godray[0].view,
            "godray_bg_b_to_a",
        ),
    ];

    // Composite bind groups:
    // Same pipeline/layout, but choose which godray texture to read (A or B).
    let composite = [
        make_composite_bg(
            device,
            &layouts.composite,
            &textures.color.view,
            &textures.godray[0].view,
            &textures.output.view,
            "composite_bg_read_a",
        ),
        make_composite_bg(
            device,
            &layouts.composite,
            &textures.color.view,
            &textures.godray[1].view,
            &textures.output.view,
            "composite_bg_read_b",
        ),
    ];

    // Blit bind group:
    // Samples final output into the swapchain; overlay buffer provides UI data.
    let blit = make_blit_bg(
        device,
        &layouts.blit,
        &textures.output.view,
        sampler,
        &buffers.overlay,
    );

    BindGroups {
        primary,
        scene,
        godray,
        composite,
        empty,
        blit,
    }
}
