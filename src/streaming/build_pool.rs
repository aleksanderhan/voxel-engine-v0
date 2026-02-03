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
    static TLS_SCRATCH: RefCell<BuildScratch> = RefCell::new(BuildScratch::new());
}

pub static BUILD_POOL: Lazy<ThreadPool> = Lazy::new(|| {
    ThreadPoolBuilder::new()
        .num_threads(4) // tune: start with 2..(num cores / 2)
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
    TLS_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        build_chunk_svo_sparse_cancelable_with_scratch(
            gen,
            chunk_origin,
            chunk_size,
            cancel,
            &mut *scratch,
            edits,
        )
    })
}