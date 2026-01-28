// src/render/state/bindgroups.rs
//
// Bind group creation.

use super::{buffers::Buffers, layout::Layouts, textures::TextureSet};

pub struct BindGroups {
    pub primary: wgpu::BindGroup,
    pub scene: wgpu::BindGroup,
    pub godray: [wgpu::BindGroup; 2],
    pub composite: [wgpu::BindGroup; 2],
    pub empty: wgpu::BindGroup,
    pub blit: wgpu::BindGroup,
}

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
            // Clipmap params uniform
            wgpu::BindGroupEntry {
                binding: 6,
                resource: buffers.clipmap.as_entire_binding(),
            },
            // Clipmap height texture array
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(&textures.clip_height.view),
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
        ],
    })
}

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

fn make_composite_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    color_view: &wgpu::TextureView,
    godray_view: &wgpu::TextureView,
    output_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView, // NEW
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
    let primary = make_primary_bg(device, &layouts.primary, buffers, textures);
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

    let composite = [
        make_composite_bg(
            device,
            &layouts.composite,
            &textures.color.view,
            &textures.godray[0].view,
            &textures.output.view,
            &textures.depth.view,
            "composite_bg_read_a",
        ),
        make_composite_bg(
            device,
            &layouts.composite,
            &textures.color.view,
            &textures.godray[1].view,
            &textures.output.view,
            &textures.depth.view,
            "composite_bg_read_b",
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
        composite,
        empty,
        blit,
    }
}
