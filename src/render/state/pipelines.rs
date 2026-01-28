// src/render/state/pipelines.rs
// -----------------------------

use super::layout::Layouts;

pub struct Pipelines {
    // Height acceleration passes
    pub build_height_l0: wgpu::ComputePipeline,
    pub build_height_mips: wgpu::ComputePipeline,

    pub build_leaves: wgpu::ComputePipeline,
    pub build_l4: wgpu::ComputePipeline,
    pub build_l3: wgpu::ComputePipeline,
    pub build_l2: wgpu::ComputePipeline,
    pub build_l1: wgpu::ComputePipeline,
    pub build_l0: wgpu::ComputePipeline,

    pub clear_dirty: wgpu::ComputePipeline,

    pub primary: wgpu::ComputePipeline,
    pub godray: wgpu::ComputePipeline,
    pub composite: wgpu::ComputePipeline,

    pub blit: wgpu::RenderPipeline,
}

fn make_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    entry: &str,
    bgls: &[&wgpu::BindGroupLayout],
) -> wgpu::ComputePipeline {
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}_pl")),
        bind_group_layouts: bgls,
        push_constant_ranges: &[],
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        module,
        entry_point: entry,
        compilation_options: Default::default(),
    })
}

pub fn create_pipelines(
    device: &wgpu::Device,
    layouts: &Layouts,
    cs_module: &wgpu::ShaderModule,
    fs_module: &wgpu::ShaderModule,
    surface_format: wgpu::TextureFormat,
) -> Pipelines {
    // Height build
    let build_height_l0 =
        make_compute_pipeline(device, "build_height_l0", cs_module, "build_height_l0", &[&layouts.scene]);
    let build_height_mips =
        make_compute_pipeline(device, "build_height_mips", cs_module, "build_height_mips", &[&layouts.scene]);

    // SVO build
    let build_leaves =
        make_compute_pipeline(device, "build_leaves", cs_module, "build_leaves", &[&layouts.scene]);
    let build_l4 =
        make_compute_pipeline(device, "build_l4", cs_module, "build_L4", &[&layouts.scene]);
    let build_l3 =
        make_compute_pipeline(device, "build_l3", cs_module, "build_L3", &[&layouts.scene]);
    let build_l2 =
        make_compute_pipeline(device, "build_l2", cs_module, "build_L2", &[&layouts.scene]);
    let build_l1 =
        make_compute_pipeline(device, "build_l1", cs_module, "build_L1", &[&layouts.scene]);
    let build_l0 =
        make_compute_pipeline(device, "build_l0", cs_module, "build_L0", &[&layouts.scene]);

    let clear_dirty =
        make_compute_pipeline(device, "clear_dirty", cs_module, "clear_dirty", &[&layouts.scene]);

    let primary = make_compute_pipeline(
        device,
        "primary_pipeline",
        cs_module,
        "main_primary",
        &[&layouts.primary],
    );

    let godray = make_compute_pipeline(
        device,
        "godray_pipeline",
        cs_module,
        "main_godray",
        &[&layouts.scene, &layouts.godray],
    );

    let composite = make_compute_pipeline(
        device,
        "composite_pipeline",
        cs_module,
        "main_composite",
        &[&layouts.empty, &layouts.empty, &layouts.composite],
    );

    let blit_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("blit_pl"),
        bind_group_layouts: &[&layouts.blit],
        push_constant_ranges: &[],
    });

    let blit = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("blit_pipeline"),
        layout: Some(&blit_pl),
        vertex: wgpu::VertexState {
            module: fs_module,
            entry_point: "vs_main",
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: fs_module,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    Pipelines {
        build_height_l0,
        build_height_mips,
        build_leaves,
        build_l4,
        build_l3,
        build_l2,
        build_l1,
        build_l0,
        clear_dirty,
        primary,
        godray,
        composite,
        blit,
    }
}
