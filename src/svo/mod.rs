// src/svo/mod.rs
pub mod builder;
pub mod mips;
pub mod raycast;

pub use builder::{
    BuildScratch,
    build_chunk_svo_sparse_cancelable_with_scratch,
};


#[inline(always)]
fn split_euclid(v: i32, size: i32) -> (i32, i32) {
    debug_assert!(size > 0);
    (v.div_euclid(size), v.rem_euclid(size))
}

#[inline(always)]
fn chunk_origin(v: i32, size: i32) -> i32 {
    v.div_euclid(size) * size
}
