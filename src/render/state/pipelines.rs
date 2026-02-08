// src/render/state/pipelines.rs
//
// Pipeline creation.
// This is intentionally isolated so the renderer logic (per-frame encoding) isn't
// buried under wgpu setup boilerplate.
//
// Terminology:
// - BindGroupLayout (BGL): describes what resources exist at @group/@binding.
// - PipelineLayout (PL): ordered list of BGLs for group(0), group(1), ...
// - Pipeline: compiled/validated shader entry point + fixed state + pipeline layout.
//
// Rule of thumb:
// The order of BGLs in `bind_group_layouts` must match the group indices used in WGSL.
// If a shader references @group(2), then the pipeline layout must include entries
// for group(0) and group(1) as well (even if they're "empty" placeholders).

use super::layout::Layouts;

pub struct Pipelines {
    /// Compute pipeline for the primary full-resolution pass (writes color/depth).
    pub primary: wgpu::ComputePipeline,

    /// Compute pipeline for the quarter-resolution godray pass (ping-pong temporal).
    pub godray: wgpu::ComputePipeline,

    /// Compute pipeline for local lighting temporal accumulation.
    pub local_taa: wgpu::ComputePipeline,

    /// Compute pipeline for the full-resolution composite pass (writes final output).
    pub composite: wgpu::ComputePipeline,

    /// Render pipeline for the final blit to the swapchain (fullscreen triangle).
    pub blit: wgpu::RenderPipeline,
}

/// Helper to build a compute pipeline with a specific entry point and bind group layout list.
///
/// `bgls` order defines the pipeline layout's group indices:
/// - bgls[0] => group(0)
/// - bgls[1] => group(1)
/// - ...
fn make_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    entry: &str,
    bgls: &[&wgpu::BindGroupLayout],
) -> wgpu::ComputePipeline {
    // Create a pipeline layout named "{label}_pl" that fixes the bind group schema.
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}_pl")),
        bind_group_layouts: bgls,
        // No push constants used by these shaders.
        push_constant_ranges: &[],
    });

    // Create the compute pipeline referencing the WGSL entry point.
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        module,
        entry_point: entry,
        compilation_options: Default::default(),
    })
}

/// Create all pipelines (compute + blit).
///
/// Inputs:
/// - `cs_module`: WGSL module containing compute entry points.
/// - `fs_module`: WGSL module containing vertex/fragment entry points for blit.
/// - `surface_format`: swapchain format used for the final render target.
pub fn create_pipelines(
    device: &wgpu::Device,
    layouts: &Layouts,
    cs_module: &wgpu::ShaderModule,
    fs_module: &wgpu::ShaderModule,
    surface_format: wgpu::TextureFormat,
) -> Pipelines {
    // -------------------------------------------------------------------------
    // Compute pipelines
    // -------------------------------------------------------------------------

    // Primary pass:
    // Uses group(0) = layouts.primary, which includes:
    // - camera + scene buffers
    // - storage outputs for color/depth
    let primary = make_compute_pipeline(
        device,
        "primary_pipeline",
        cs_module,
        "main_primary",
        &[&layouts.primary],
    );

    // Godray pass:
    // Uses:
    //   group(0) = layouts.scene  (camera + scene buffers only)
    //   group(1) = layouts.godray (depth sample + history sample + out storage)
    let godray = make_compute_pipeline(
        device,
        "godray_pipeline",
        cs_module,
        "main_godray",
        &[&layouts.scene, &layouts.godray],
    );

    let local_taa = make_compute_pipeline(
        device,
        "local_taa_pipeline",
        cs_module,
        "main_local_taa",
        &[&layouts.local_taa],
    );

    // Composite pass:
    // Shader reads from @group(2) (color + godray + output storage).
    // wgpu requires the pipeline layout to include group(0) and group(1) slots too,
    // so we provide empty placeholder layouts for those indices.
    let composite = make_compute_pipeline(
        device,
        "composite_pipeline",
        cs_module,
        "main_composite",
        // group(0)=scene (cam + buffers), group(1)=empty, group(2)=composite textures
        &[&layouts.scene, &layouts.empty, &layouts.composite],
    );

    // -------------------------------------------------------------------------
    // Render pipeline: blit
    // -------------------------------------------------------------------------
    //
    // Full-screen triangle approach:
    // - No vertex buffers.
    // - Vertex shader generates positions from vertex_index.
    // - Fragment shader samples the renderer output texture.

    // Pipeline layout for blit uses a single bind group: group(0) = layouts.blit.
    let blit_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("blit_pl"),
        bind_group_layouts: &[&layouts.blit],
        push_constant_ranges: &[],
    });

    // Render pipeline state:
    // - Targets the swapchain format.
    // - Uses REPLACE blending (overwrite framebuffer).
    // - Default primitive/multisample state is fine for a simple fullscreen draw.
    let blit = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("blit_pipeline"),
        layout: Some(&blit_pl),
        vertex: wgpu::VertexState {
            module: fs_module,
            entry_point: "vs_main",
            // No vertex buffers; vertices are synthesized in the vertex shader.
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: fs_module,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                // Overwrite swapchain pixel with sampled color.
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        // Default triangle list, CCW front face, etc. (fullscreen triangle doesn't care much).
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    Pipelines {
        primary,
        godray,
        local_taa,
        composite,
        blit,
    }
}
