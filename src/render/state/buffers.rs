



use crate::app::config;
use crate::{
    render::gpu_types::{ChunkMetaGpu, ClipmapGpu, NodeGpu, NodeRopesGpu},
};

pub struct Buffers {
    
    pub camera: wgpu::Buffer,
    pub overlay: wgpu::Buffer,

    
    pub clipmap: wgpu::Buffer,

    
    pub node: wgpu::Buffer,
    pub chunk: wgpu::Buffer,
    pub chunk_grid: wgpu::Buffer,

    
    pub node_capacity: u32,
    pub chunk_capacity: u32,
    pub grid_capacity: u32,

    pub macro_occ: wgpu::Buffer,
    pub macro_capacity_u32: u32,

    pub node_ropes: wgpu::Buffer,
    pub rope_capacity: u32, 

    pub colinfo: wgpu::Buffer,
    pub colinfo_capacity_u32: u32,

}

fn make_uniform_buffer<T: Sized>(device: &wgpu::Device, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<T>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn make_storage_buffer(device: &wgpu::Device, label: &str, size_bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

pub fn create_persistent_buffers(device: &wgpu::Device) -> Buffers {
    let camera = make_uniform_buffer::<crate::render::gpu_types::CameraGpu>(device, "camera_buf");
    let overlay = make_uniform_buffer::<crate::render::gpu_types::OverlayGpu>(device, "overlay_buf");

    let clipmap = make_uniform_buffer::<ClipmapGpu>(device, "clipmap_buf");

    let node_capacity = (config::NODE_BUDGET_BYTES / std::mem::size_of::<NodeGpu>()) as u32;

    let node = make_storage_buffer(
        device,
        "svo_nodes_arena",
        (node_capacity as u64) * (std::mem::size_of::<NodeGpu>() as u64),
    );

    let chunk_capacity =
        (2 * config::KEEP_RADIUS + 1) as u32 * 4u32 * (2 * config::KEEP_RADIUS + 1) as u32;

    let chunk = make_storage_buffer(
        device,
        "chunk_meta_persistent",
        (chunk_capacity as u64) * (std::mem::size_of::<ChunkMetaGpu>() as u64),
    );

    let grid_capacity = chunk_capacity;

    let chunk_grid = make_storage_buffer(
        device,
        "chunk_grid_buf",
        (grid_capacity as u64) * (std::mem::size_of::<u32>() as u64),
    );

    
    const MACRO_WORDS_PER_CHUNK: u32 = 16;
    let macro_capacity_u32 = chunk_capacity * MACRO_WORDS_PER_CHUNK;
    let macro_occ = make_storage_buffer(
        device,
        "macro_occ_buf",
        (macro_capacity_u32 as u64) * (std::mem::size_of::<u32>() as u64),
    );

    let rope_capacity = node_capacity;
    let node_ropes = make_storage_buffer(
        device,
        "svo_node_ropes",
        (rope_capacity as u64) * (std::mem::size_of::<NodeRopesGpu>() as u64),
    );

    
    const COLINFO_WORDS_PER_CHUNK: u32 = 2048;
    let colinfo_capacity_u32 = chunk_capacity * COLINFO_WORDS_PER_CHUNK;
    let colinfo = make_storage_buffer(
        device,
        "chunk_colinfo_buf",
        (colinfo_capacity_u32 as u64) * (std::mem::size_of::<u32>() as u64),
    );

    Buffers {
        camera,
        overlay,
        clipmap,
        node,
        chunk,
        chunk_grid,
        node_capacity,
        chunk_capacity,
        grid_capacity,
        macro_occ,
        macro_capacity_u32,
        node_ropes,
        rope_capacity,
        colinfo,
        colinfo_capacity_u32,
    }
}
