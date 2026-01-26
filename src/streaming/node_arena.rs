// src/streaming/node_arena.rs
//
// Very simple free-list arena for node ranges (in units of NodeGpu elements).
// First-fit is fine; KEEP set is small.

#[derive(Clone, Copy, Debug)]
struct Range {
    start: u32,
    len: u32,
}

pub struct NodeArena {
    capacity: u32,
    free: Vec<Range>, // kept sorted by start
}

impl NodeArena {
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity,
            free: vec![Range { start: 0, len: capacity }],
        }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn alloc(&mut self, len: u32) -> Option<u32> {
        if len == 0 {
            return Some(0);
        }

        for i in 0..self.free.len() {
            let r = self.free[i];
            if r.len >= len {
                let start = r.start;

                if r.len == len {
                    self.free.remove(i);
                } else {
                    self.free[i] = Range {
                        start: r.start + len,
                        len: r.len - len,
                    };
                }
                return Some(start);
            }
        }
        None
    }

    pub fn free(&mut self, start: u32, len: u32) {
        if len == 0 {
            return;
        }

        // insert sorted
        let mut idx = 0usize;
        while idx < self.free.len() && self.free[idx].start < start {
            idx += 1;
        }
        self.free.insert(idx, Range { start, len });

        // merge neighbors
        self.merge_at(idx);
        if idx > 0 {
            self.merge_at(idx - 1);
        }
    }

    fn merge_at(&mut self, i: usize) {
        if i + 1 >= self.free.len() {
            return;
        }
        let a = self.free[i];
        let b = self.free[i + 1];
        if a.start + a.len == b.start {
            self.free[i] = Range {
                start: a.start,
                len: a.len + b.len,
            };
            self.free.remove(i + 1);
        }
    }
}
