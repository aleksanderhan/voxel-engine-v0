// src/streaming/workers.rs
use std::sync::{Arc, atomic::Ordering};
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};

use crate::app::config;
use crate::{
    render::gpu_types::{NodeGpu, NodeRopesGpu},
    svo::{build_chunk_svo_sparse_cancelable_with_scratch, BuildScratch},
    world::WorldGen,
};
use crate::svo::builder::{BuildOutput, BuildTimingsMs};

use super::types::{BuildDone, BuildJob};

pub fn spawn_workers(gen: Arc<WorldGen>, rx_job: Receiver<BuildJob>, tx_done: Sender<BuildDone>) {
    for _ in 0..config::WORKER_THREADS {
        let gen = gen.clone();
        let rx_job = rx_job.clone();
        let tx_done = tx_done.clone();

        std::thread::spawn(move || {
            let mut scratch = BuildScratch::new();

            while let Ok(job) = rx_job.recv() {
                let k = job.key;

                let t_start = Instant::now();
                let queue_ms = (t_start - job.enqueued_at).as_secs_f64() * 1000.0;

                if job.cancel.load(Ordering::Relaxed) {
                    let _ = tx_done.send(BuildDone {
                        key: k,
                        cancel: job.cancel,
                        canceled: true,
                        nodes: Vec::new(),
                        macro_words: Vec::new(),
                        ropes: Vec::new(),
                        colinfo_words: Vec::new(),
                        tim: BuildTimingsMs::default(),
                        queue_ms,
                        build_ms: 0.0,
                    });
                    continue;
                }

                let origin = [
                    k.x * config::CHUNK_SIZE as i32,
                    k.y * config::CHUNK_SIZE as i32,
                    k.z * config::CHUNK_SIZE as i32,
                    0,
                ];

                let BuildOutput {
                    nodes,
                    macro_words,
                    ropes,
                    colinfo_words,
                    timings: tim,
                } = build_chunk_svo_sparse_cancelable_with_scratch(
                    &gen,
                    [origin[0], origin[1], origin[2]],
                    config::CHUNK_SIZE,
                    &job.cancel,
                    &mut scratch,
                    &job.edits,
                );



                let canceled = job.cancel.load(Ordering::Relaxed);
                let (nodes, macro_words, ropes) = if canceled {
                    (Vec::new(), Vec::new(), Vec::new())
                } else {
                    (nodes, macro_words, ropes)
                };

                let build_ms = t_start.elapsed().as_secs_f64() * 1000.0;
                let _ = tx_done.send(BuildDone {
                    key: k,
                    cancel: job.cancel,
                    canceled,
                    nodes,
                    macro_words,
                    ropes,
                    colinfo_words,
                    tim,
                    queue_ms,
                    build_ms,
                });
            }
        });
    }
}
