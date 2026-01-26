use std::hash::Hash;

use crate::render::NodeGpu;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkKey {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

pub struct ChunkCpu {
    pub nodes: Vec<NodeGpu>,
}

pub enum ChunkState {
    Missing,
    Queued,
    Building,
    Ready(ChunkCpu),
}
