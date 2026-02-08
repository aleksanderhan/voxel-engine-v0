// src/render/state/bindgroups.rs
//
// Bind group creation.

use super::{buffers::Buffers, layout::Layouts, textures::TextureSet};

pub struct BindGroups {
    pub primary: [wgpu::BindGroup; 2],
    pub scene: wgpu::BindGroup,
    pub godray: [wgpu::BindGroup; 2],
    pub local_taa: [wgpu::BindGroup; 2],
    pub composite: [wgpu::BindGroup; 2],
    pub composite_taa: [wgpu::BindGroup; 2],
    pub empty: wgpu::BindGroup,
    pub blit: wgpu::BindGroup,
}

fn make_primary_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffers: &Buffers,
    textures: &TextureSet,
    hist_in: &wgpu::TextureView,
    hist_out: &wgpu::TextureView,
    shadow_hist_in: &wgpu::TextureView,
    shadow_hist_out: &wgpu::Buffer,
    sampler: &wgpu::Sampler,
    label: &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
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
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&textures.color.view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&textures.depth.view),
            },
            // NEW: local lighting output
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&textures.local.view),
            },
            // shifted clipmap params uniform
            wgpu::BindGroupEntry {
                binding: 7,
                resource: buffers.clipmap.as_entire_binding(),
            },
            // shifted clipmap height texture array
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(&textures.clip_height.view),
            },
            // shifted storage buffers
            wgpu::BindGroupEntry {
                binding: 9,
                resource: buffers.macro_occ.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: buffers.node_ropes.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: buffers.colinfo.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 12,
                resource: wgpu::BindingResource::TextureView(hist_in),
            },
            wgpu::BindGroupEntry {
                binding: 13,
                resource: wgpu::BindingResource::TextureView(hist_out),
            },
            wgpu::BindGroupEntry {
                binding: 14,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 15,
                resource: wgpu::BindingResource::TextureView(shadow_hist_in),
            },
            wgpu::BindGroupEntry {
                binding: 16,
                resource: shadow_hist_out.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 17,
                resource: buffers.primary_profile.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 18,
                resource: wgpu::BindingResource::TextureView(&textures.grass.view),
            },


        ],
    })
}

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
            wgpu::BindGroupEntry { 
                binding: 8, 
                resource: buffers.macro_occ.as_entire_binding() 
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: buffers.node_ropes.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: buffers.colinfo.as_entire_binding(),
            },

        ],
    })
}

fn make_godray_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    depth_view: &wgpu::TextureView,
    hist_view: &wgpu::TextureView,
    out_view: &wgpu::TextureView,
    hist_sampler: &wgpu::Sampler,   // NEW
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
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(hist_sampler),
            },
        ],
    })
}

fn make_local_taa_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    local_in_view: &wgpu::TextureView,
    local_hist_in_view: &wgpu::TextureView,
    local_hist_out_view: &wgpu::TextureView,
    local_sampler: &wgpu::Sampler,
    label: &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(local_in_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(local_hist_in_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(local_hist_out_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(local_sampler),
            },
        ],
    })
}

fn make_composite_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    color_view: &wgpu::TextureView,
    godray_view: &wgpu::TextureView,
    output_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    godray_sampler: &wgpu::Sampler,
    local_hist_view: &wgpu::TextureView,
    local_sampler: &wgpu::Sampler,
    grass_view: &wgpu::TextureView,
    grass_sampler: &wgpu::Sampler,
    label: &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(color_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(godray_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(output_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(depth_view) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(godray_sampler) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(local_hist_view) },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(local_sampler) },
            wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(grass_view) },
            wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(grass_sampler) },
        ],
    })
}

fn make_composite_taa_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    composite_in_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    hist_in_view: &wgpu::TextureView,
    hist_sampler: &wgpu::Sampler,
    output_view: &wgpu::TextureView,
    hist_out_view: &wgpu::TextureView,
    label: &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(composite_in_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(hist_in_view) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(hist_sampler) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(output_view) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(hist_out_view) },
        ],
    })
}



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
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(output_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: overlay_buf.as_entire_binding(),
            },
        ],
    })
}

pub fn create_bind_groups(
    device: &wgpu::Device,
    layouts: &Layouts,
    buffers: &Buffers,
    textures: &TextureSet,
    sampler: &wgpu::Sampler,
) -> BindGroups {
    let primary = [
        make_primary_bg(
            device,
            &layouts.primary,
            buffers,
            textures,
            &textures.primary_hit_hist[0].view,
            &textures.primary_hit_hist[1].view,
            &textures.shadow_hist.view,
            &textures.shadow_hist_buf,
            sampler,
            "primary_bg_hist_a_to_b",
        ),
        make_primary_bg(
            device,
            &layouts.primary,
            buffers,
            textures,
            &textures.primary_hit_hist[1].view,
            &textures.primary_hit_hist[0].view,
            &textures.shadow_hist.view,
            &textures.shadow_hist_buf,
            sampler,
            "primary_bg_hist_b_to_a",
        ),
    ];
    let scene = make_scene_bg(device, &layouts.scene, buffers);

    let empty = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("empty_bg"),
        layout: &layouts.empty,
        entries: &[],
    });

    let godray = [
        make_godray_bg(
            device,
            &layouts.godray,
            &textures.depth.view,
            &textures.godray[0].view,
            &textures.godray[1].view,
            sampler,
            "godray_bg_a_to_b",
        ),
        make_godray_bg(
            device,
            &layouts.godray,
            &textures.depth.view,
            &textures.godray[1].view,
            &textures.godray[0].view,
            sampler,
            "godray_bg_b_to_a",
        ),
    ];

    let local_taa = [
        make_local_taa_bg(
            device,
            &layouts.local_taa,
            &textures.local.view,
            &textures.local_hist[0].view,
            &textures.local_hist[1].view,
            sampler,
            "local_taa_bg_a_to_b",
        ),
        make_local_taa_bg(
            device,
            &layouts.local_taa,
            &textures.local.view,
            &textures.local_hist[1].view,
            &textures.local_hist[0].view,
            sampler,
            "local_taa_bg_b_to_a",
        ),
    ];

    let composite = [
        make_composite_bg(
            device,
            &layouts.composite,
            &textures.color.view,
            &textures.godray[0].view,
            &textures.output_pre_taa.view,
            &textures.depth.view,
            sampler,
            &textures.local_hist[0].view,
            sampler,
            &textures.grass.view,
            sampler,
            "composite_bg_read_a",
        ),
        make_composite_bg(
            device,
            &layouts.composite,
            &textures.color.view,
            &textures.godray[1].view,
            &textures.output_pre_taa.view,
            &textures.depth.view,
            sampler,
            &textures.local_hist[1].view,
            sampler,
            &textures.grass.view,
            sampler,
            "composite_bg_read_b",
        ),
    ];

    let composite_taa = [
        make_composite_taa_bg(
            device,
            &layouts.composite_taa,
            &textures.output_pre_taa.view,
            &textures.depth.view,
            &textures.output_hist[0].view,
            sampler,
            &textures.output.view,
            &textures.output_hist[1].view,
            "composite_taa_bg_a_to_b",
        ),
        make_composite_taa_bg(
            device,
            &layouts.composite_taa,
            &textures.output_pre_taa.view,
            &textures.depth.view,
            &textures.output_hist[1].view,
            sampler,
            &textures.output.view,
            &textures.output_hist[0].view,
            "composite_taa_bg_b_to_a",
        ),
    ];


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
        local_taa,
        composite,
        composite_taa,
        empty,
        blit,
    }
}
