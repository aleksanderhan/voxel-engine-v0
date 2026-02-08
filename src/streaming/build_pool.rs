// src/streaming/build_pool.rs
use once_cell::sync::Lazy;
use rayon::ThreadPool;
use std::cell::RefCell;
use std::sync::atomic::AtomicBool;

use crate::svo::builder::build_chunk_svo_sparse_cancelable_with_scratch;
use crate::svo::builder::BuildOutput;
use crate::world::edits::EditEntry;
use crate::world::WorldGen;
use crate::svo::BuildScratch;
use crate::app::config;
use rayon::ThreadPoolBuilder;

thread_local! {
    static TLS_SCRATCH: RefCell<Vec<BuildScratch>> = RefCell::new(Vec::new());
}

pub static BUILD_POOL: Lazy<ThreadPool> = Lazy::new(|| {
    ThreadPoolBuilder::new()
        .num_threads(config::WORKER_THREADS) // tune: start with 2..(num cores / 2)
        .thread_name(|i| format!("chunk-build-{}", i))
        .build()
        .expect("failed to build chunk build thread pool")
});

pub fn build_chunk_svo_sparse_cancelable_tls(
    gen: &WorldGen,
    chunk_origin: [i32; 3],
    chunk_size: u32,
    cancel: &AtomicBool,
    edits: &[EditEntry],
) -> BuildOutput {
    // Take a scratch out of TLS WITHOUT holding the RefCell borrow during the build.
    let mut scratch = TLS_SCRATCH.with(|cell| {
        cell.borrow_mut().pop().unwrap_or_else(BuildScratch::new)
    });

    let out = build_chunk_svo_sparse_cancelable_with_scratch(
        gen,
        chunk_origin,
        chunk_size,
        cancel,
        &mut scratch,
        edits,
    );

    // Return scratch to the per-thread pool.
    TLS_SCRATCH.with(|cell| {
        cell.borrow_mut().push(scratch);
    });

    out
}
