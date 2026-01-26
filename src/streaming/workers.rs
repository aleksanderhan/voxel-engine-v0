use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};

use crate::{svo::build_chunk_svo_sparse, world::WorldGen, config, render::NodeGpu};
use super::ChunkKey;

#[derive(Clone, Copy, Debug)]
pub struct BuildJob {
    pub key: ChunkKey,
}

pub struct BuildDone {
    pub key: ChunkKey,
    pub nodes: Vec<NodeGpu>,
}

pub fn spawn_workers(gen: Arc<WorldGen>, rx_job: Receiver<BuildJob>, tx_done: Sender<BuildDone>) {
    for _ in 0..config::WORKER_THREADS {
        let gen = gen.clone();
        let rx_job = rx_job.clone();
        let tx_done = tx_done.clone();

        std::thread::spawn(move || {
            while let Ok(job) = rx_job.recv() {
                let k = job.key;
                let origin = [
                    k.x * config::CHUNK_SIZE as i32,
                    k.y * config::CHUNK_SIZE as i32,
                    k.z * config::CHUNK_SIZE as i32,
                ];

                let nodes = build_chunk_svo_sparse(&gen, origin, config::CHUNK_SIZE);

                if tx_done.send(BuildDone { key: k, nodes }).is_err() {
                    break;
                }
            }
        });
    }
}
