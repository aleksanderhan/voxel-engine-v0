






#[derive(Clone, Copy, Debug)]
struct Range {
    start: u32,
    len: u32,
}

pub struct NodeArena {
    free: Vec<Range>, 
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

    
    
    pub fn alloc(&mut self, len: u32) -> Option<u32> {
        if len == 0 {
            return Some(0);
        }

        
        let mut best_i: Option<usize> = None;
        let mut best_len: u32 = u32::MAX;

        for (i, r) in self.free.iter().enumerate() {
            if r.len >= len && r.len < best_len {
                best_len = r.len;
                best_i = Some(i);
                if r.len == len {
                    break; 
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

    
    pub fn free(&mut self, start: u32, len: u32) {
        if len == 0 {
            return;
        }

        
        let idx = self
            .free
            .binary_search_by_key(&start, |r| r.start)
            .unwrap_or_else(|i| i);

        self.free.insert(idx, Range { start, len });

        
        self.coalesce_at(idx);
    }

    fn coalesce_at(&mut self, mut i: usize) {
        
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
}
