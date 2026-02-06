















use super::layout::Layouts;

pub struct Pipelines {
    
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
        
        &[&layouts.scene, &layouts.empty, &layouts.composite],
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
        primary,
        godray,
        composite,
        blit,
    }
}
