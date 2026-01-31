use std::{collections::VecDeque, mem::size_of, sync::Arc};
use rustc_hash::FxHashMap as HashMap;

use crate::{config, render::gpu_types::{NodeGpu, NodeRopesGpu}};
use crate::streaming::types::ChunkKey;

#[derive(Clone)]
pub struct CachedChunk {
    pub nodes: Arc<[NodeGpu]>,
    pub ropes: Arc<[NodeRopesGpu]>,
    pub macro_words: Arc<[u32]>,
    pub colinfo_words: Arc<[u32]>,
    pub bytes: usize,
    pub stamp: u64,
}

pub struct ChunkCache {
    map: HashMap<ChunkKey, CachedChunk>,
    lru: VecDeque<(ChunkKey, u64)>,
    stamp: u64,
    bytes: usize,
}

impl ChunkCache {
    pub fn new() -> Self {
        Self { map: HashMap::default(), lru: VecDeque::default(), stamp: 1, bytes: 0 }
    }

    pub fn stats(&self) -> (usize, usize, usize) {
        (self.bytes, self.map.len(), self.lru.len())
    }

    pub fn get(&self, key: &ChunkKey) -> Option<&CachedChunk> { self.map.get(key) }

    pub fn touch(&mut self, key: ChunkKey) {
        if let Some(e) = self.map.get_mut(&key) {
            self.stamp = self.stamp.wrapping_add(1).max(1);
            e.stamp = self.stamp;
            self.lru.push_back((key, e.stamp));
            self.maybe_compact_lru();
        }
    }

    pub fn put(
        &mut self,
        key: ChunkKey,
        nodes: Arc<[NodeGpu]>,
        macro_words: Arc<[u32]>,
        ropes: Arc<[NodeRopesGpu]>,
        colinfo_words: Arc<[u32]>,
    ) {
        if let Some(old) = self.map.remove(&key) {
            self.bytes = self.bytes.saturating_sub(old.bytes);
        }

        self.stamp = self.stamp.wrapping_add(1).max(1);
        let stamp = self.stamp;

        let bytes =
            nodes.len() * size_of::<NodeGpu>()
            + ropes.len() * size_of::<NodeRopesGpu>()
            + macro_words.len() * size_of::<u32>()
            + colinfo_words.len() * size_of::<u32>();

        self.map.insert(
            key,
            CachedChunk { nodes, ropes, macro_words, colinfo_words, bytes, stamp },
        );

        self.bytes = self.bytes.saturating_add(bytes);
        self.lru.push_back((key, stamp));
        self.evict_as_needed();
    }

    pub fn remove(&mut self, key: &ChunkKey) {
        if let Some(old) = self.map.remove(key) {
            self.bytes = self.bytes.saturating_sub(old.bytes);
        }
    }

    fn evict_as_needed(&mut self) {
        let budget = config::CHUNK_CACHE_BUDGET_BYTES;
        while self.bytes > budget {
            let Some((k, stamp)) = self.lru.pop_front() else { break; };

            let should_evict = self.map.get(&k).map(|e| e.stamp == stamp).unwrap_or(false);
            if !should_evict { continue; }

            if let Some(ev) = self.map.remove(&k) {
                self.bytes = self.bytes.saturating_sub(ev.bytes);
            }
        }
    }

    fn maybe_compact_lru(&mut self) {
        let max = self.map.len().saturating_mul(8).max(1024);
        if self.lru.len() <= max { return; }

        let mut new = VecDeque::with_capacity(self.map.len());
        for (k, e) in self.map.iter() {
            new.push_back((*k, e.stamp));
        }
        self.lru = new;
    }
}
