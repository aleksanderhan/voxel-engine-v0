// src/streaming/node_arena.rs
//
// Very simple free-list arena for node ranges (in units of NodeGpu elements).
// Improvements:
// - free() now fully coalesces adjacent ranges (fixes long-run fragmentation).
// - alloc() uses best-fit (smallest range that fits) to reduce fragmentation further.

#[derive(Clone, Copy, Debug)]
struct Range {
    start: u32,
    len: u32,
}

pub struct NodeArena {
    free: Vec<Range>, // kept sorted by start
}

impl NodeArena {
    pub fn new(capacity: u32) -> Self {
        Self {
            free: vec![Range {
                start: 0,
                len: capacity,
            }],
        }
    }

    /// Allocate a contiguous range of `len` elements.
    /// Returns the start index in the arena, or None if no free range fits.
    pub fn alloc(&mut self, len: u32) -> Option<u32> {
        if len == 0 {
            return Some(0);
        }

        // Best-fit: choose the smallest free range that still fits.
        let mut best_i: Option<usize> = None;
        let mut best_len: u32 = u32::MAX;

        for (i, r) in self.free.iter().enumerate() {
            if r.len >= len && r.len < best_len {
                best_len = r.len;
                best_i = Some(i);
                if r.len == len {
                    break; // perfect fit
                }
            }
        }

        let i = best_i?;
        let r = self.free[i];
        let start = r.start;

        if r.len == len {
            self.free.remove(i);
        } else {
            self.free[i] = Range {
                start: r.start + len,
                len: r.len - len,
            };
        }

        Some(start)
    }

    /// Free a previously allocated range.
    pub fn free(&mut self, start: u32, len: u32) {
        if len == 0 {
            return;
        }

        // Insert sorted by start (log n search).
        let idx = self
            .free
            .binary_search_by_key(&start, |r| r.start)
            .unwrap_or_else(|i| i);

        self.free.insert(idx, Range { start, len });

        // Fully coalesce with neighbors (both directions).
        self.coalesce_at(idx);
    }

    fn coalesce_at(&mut self, mut i: usize) {
        // Merge backward as long as possible.
        while i > 0 {
            let a = self.free[i - 1];
            let b = self.free[i];
            if a.start + a.len == b.start {
                self.free[i - 1] = Range {
                    start: a.start,
                    len: a.len + b.len,
                };
                self.free.remove(i);
                i -= 1;
            } else {
                break;
            }
        }

        // Merge forward as long as possible.
        while i + 1 < self.free.len() {
            let a = self.free[i];
            let b = self.free[i + 1];
            if a.start + a.len == b.start {
                self.free[i] = Range {
                    start: a.start,
                    len: a.len + b.len,
                };
                self.free.remove(i + 1);
            } else {
                break;
            }
        }
    }

    // Optional: quick stats for debugging.
    pub fn free_range_count(&self) -> usize {
        self.free.len()
    }

    pub fn largest_free_range(&self) -> u32 {
        self.free.iter().map(|r| r.len).max().unwrap_or(0)
    }

    pub fn total_free(&self) -> u32 {
        self.free.iter().map(|r| r.len).sum()
    }
}
