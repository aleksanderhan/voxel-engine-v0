use std::time::{Duration, Instant};

pub struct FrameProf {
    pub frame: u64,
    pub last_print: Instant,
    pub print_every: Duration,

    pub n_frames: u64,
    pub t_cam: f64,
    pub t_stream: f64,
    pub t_clipmap_update: f64,
    pub t_cam_write: f64,
    pub t_overlay: f64,
    pub t_chunk_uploads: f64,
    pub t_acquire: f64,
    pub t_encode_clipmap: f64,
    pub t_encode_compute: f64,
    pub t_encode_blit: f64,
    pub t_submit: f64,
    pub t_poll: f64,
    pub t_present: f64,

    pub clip_uploads: u64,
    pub clip_bytes: u64,
    pub chunk_uploads: u64,

    pub max_frame_ms: f64,

    pub t_poll_wait: f64,

}

impl FrameProf {
    pub fn new() -> Self {
        Self {
            frame: 0,
            last_print: Instant::now(),
            print_every: Duration::from_millis(500),

            n_frames: 0,
            t_cam: 0.0,
            t_stream: 0.0,
            t_clipmap_update: 0.0,
            t_cam_write: 0.0,
            t_overlay: 0.0,
            t_chunk_uploads: 0.0,
            t_acquire: 0.0,
            t_encode_clipmap: 0.0,
            t_encode_compute: 0.0,
            t_encode_blit: 0.0,
            t_submit: 0.0,
            t_poll: 0.0,
            t_present: 0.0,

            clip_uploads: 0,
            clip_bytes: 0,
            chunk_uploads: 0,

            max_frame_ms: 0.0,

            t_poll_wait: 0.0,
        }
    }

    #[inline]
    pub fn mark_ms(t0: Instant) -> f64 {
        t0.elapsed().as_secs_f64() * 1000.0
    }

    // --- per-slot adders (avoid borrow conflicts) ---
    #[inline] pub fn cam(&mut self, ms: f64) { self.t_cam += ms; }
    #[inline] pub fn stream(&mut self, ms: f64) { self.t_stream += ms; }
    #[inline] pub fn clip_update(&mut self, ms: f64) { self.t_clipmap_update += ms; }
    #[inline] pub fn cam_write(&mut self, ms: f64) { self.t_cam_write += ms; }
    #[inline] pub fn overlay(&mut self, ms: f64) { self.t_overlay += ms; }
    #[inline] pub fn chunk_up(&mut self, ms: f64) { self.t_chunk_uploads += ms; }
    #[inline] pub fn acquire(&mut self, ms: f64) { self.t_acquire += ms; }
    #[inline] pub fn enc_clip(&mut self, ms: f64) { self.t_encode_clipmap += ms; }
    #[inline] pub fn enc_comp(&mut self, ms: f64) { self.t_encode_compute += ms; }
    #[inline] pub fn enc_blit(&mut self, ms: f64) { self.t_encode_blit += ms; }
    #[inline] pub fn submit(&mut self, ms: f64) { self.t_submit += ms; }
    #[inline] pub fn poll(&mut self, ms: f64) { self.t_poll += ms; }
    #[inline] pub fn present(&mut self, ms: f64) { self.t_present += ms; }

    pub fn add_clip_uploads(&mut self, n: usize, bytes: usize) {
        self.clip_uploads += n as u64;
        self.clip_bytes += bytes as u64;
    }

    pub fn add_chunk_uploads(&mut self, n: usize) {
        self.chunk_uploads += n as u64;
    }

    pub fn end_frame(&mut self, frame_ms: f64) {
        self.frame += 1;
        self.n_frames += 1;
        self.max_frame_ms = self.max_frame_ms.max(frame_ms);

        if self.last_print.elapsed() >= self.print_every {
            let nf = self.n_frames.max(1) as f64;
            let avg = |x: f64| x / nf;

            let avg_frame = avg(
                self.t_cam + self.t_stream + self.t_clipmap_update + self.t_cam_write + self.t_overlay
                    + self.t_chunk_uploads + self.t_acquire + self.t_encode_clipmap + self.t_encode_compute
                    + self.t_encode_blit + self.t_submit + self.t_poll + self.t_poll_wait + self.t_present
            );

            println!(
                concat!(
                    "\n[prof] frames={} avg_frame={:.2}ms max_frame={:.2}ms\n",
                    "  cam={:.2} stream={:.2} clip_update={:.2} cam_write={:.2} overlay={:.2}\n",
                    "  chunk_up={:.2} acquire={:.2} enc_clip={:.2} enc_comp={:.2} enc_blit={:.2}\n",
                    "  submit={:.2} poll={:.2} poll_wait={:.2} present={:.2}\n",
                    "  clip_uploads/frame={:.1} clip_kb/frame={:.1} chunk_uploads/frame={:.1}\n"
                ),
                self.frame,
                avg_frame,
                self.max_frame_ms,
                avg(self.t_cam),
                avg(self.t_stream),
                avg(self.t_clipmap_update),
                avg(self.t_cam_write),
                avg(self.t_overlay),
                avg(self.t_chunk_uploads),
                avg(self.t_acquire),
                avg(self.t_encode_clipmap),
                avg(self.t_encode_compute),
                avg(self.t_encode_blit),
                avg(self.t_submit),
                avg(self.t_poll),
                avg(self.t_poll_wait),
                avg(self.t_present),
                (self.clip_uploads as f64) / nf,
                (self.clip_bytes as f64) / 1024.0 / nf,
                (self.chunk_uploads as f64) / nf,
            );

            // reset window counters, keep frame + print_every
            self.last_print = Instant::now();
            self.n_frames = 0;
            self.t_cam = 0.0;
            self.t_stream = 0.0;
            self.t_clipmap_update = 0.0;
            self.t_cam_write = 0.0;
            self.t_overlay = 0.0;
            self.t_chunk_uploads = 0.0;
            self.t_acquire = 0.0;
            self.t_encode_clipmap = 0.0;
            self.t_encode_compute = 0.0;
            self.t_encode_blit = 0.0;
            self.t_submit = 0.0;
            self.t_poll = 0.0;
            self.t_present = 0.0;
            self.clip_uploads = 0;
            self.clip_bytes = 0;
            self.chunk_uploads = 0;
            self.max_frame_ms = 0.0;
        }
    }

    #[inline]
    pub fn poll_wait(&mut self, ms: f64) {
        self.t_poll_wait += ms;
    }

}
