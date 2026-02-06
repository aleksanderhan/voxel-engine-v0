

use std::time::{Duration, Instant};

use crate::render::state::GpuTimingsMs; 
use crate::streaming::types::StreamStats;

pub struct FrameProf {
    enabled: bool,

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
    pub t_acq_swapchain: f64,

    pub t_prof_overhead: f64,
    pub max_prof_overhead_ms: f64,

    
    pub max_present_ms: f64,
    pub max_acq_swapchain_ms: f64,
    pub max_submit_ms: f64,
    pub max_poll_wait_ms: f64,
}

impl FrameProf {
    pub fn new(enabled: bool, print_every: Duration) -> Self {
        Self {
            enabled,

            frame: 0,
            last_print: Instant::now(),
            print_every,

            n_frames: 0,
            t_cam: 0.0,
            t_stream: 0.0,
            t_clipmap_update: 0.0,
            t_cam_write: 0.0,
            t_overlay: 0.0,
            t_chunk_uploads: 0.0,
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
            t_acq_swapchain: 0.0,

            t_prof_overhead: 0.0,
            max_prof_overhead_ms: 0.0,

            max_present_ms: 0.0,
            max_acq_swapchain_ms: 0.0,
            max_submit_ms: 0.0,
            max_poll_wait_ms: 0.0,
        }
    }

    #[inline]
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    
    #[inline]
    pub fn start(&self) -> Option<Instant> {
        if self.enabled {
            Some(Instant::now())
        } else {
            None
        }
    }

    #[inline]
    pub fn end_ms(t0: Option<Instant>) -> f64 {
        match t0 {
            Some(t) => t.elapsed().as_secs_f64() * 1000.0,
            None => 0.0,
        }
    }

    
    #[inline]
    pub fn cam(&mut self, ms: f64) {
        if self.enabled {
            self.t_cam += ms;
        }
    }
    #[inline]
    pub fn stream(&mut self, ms: f64) {
        if self.enabled {
            self.t_stream += ms;
        }
    }
    #[inline]
    pub fn clip_update(&mut self, ms: f64) {
        if self.enabled {
            self.t_clipmap_update += ms;
        }
    }
    #[inline]
    pub fn cam_write(&mut self, ms: f64) {
        if self.enabled {
            self.t_cam_write += ms;
        }
    }
    #[inline]
    pub fn overlay(&mut self, ms: f64) {
        if self.enabled {
            self.t_overlay += ms;
        }
    }
    #[inline]
    pub fn chunk_up(&mut self, ms: f64) {
        if self.enabled {
            self.t_chunk_uploads += ms;
        }
    }
    #[inline]
    pub fn enc_clip(&mut self, ms: f64) {
        if self.enabled {
            self.t_encode_clipmap += ms;
        }
    }
    #[inline]
    pub fn enc_comp(&mut self, ms: f64) {
        if self.enabled {
            self.t_encode_compute += ms;
        }
    }
    #[inline]
    pub fn enc_blit(&mut self, ms: f64) {
        if self.enabled {
            self.t_encode_blit += ms;
        }
    }
    #[inline]
    pub fn poll(&mut self, ms: f64) {
        if self.enabled {
            self.t_poll += ms;
        }
    }

    #[inline]
    pub fn present(&mut self, ms: f64) {
        if self.enabled {
            self.t_present += ms;
            self.max_present_ms = self.max_present_ms.max(ms);
        }
    }

    #[inline]
    pub fn acq_swapchain(&mut self, ms: f64) {
        if self.enabled {
            self.t_acq_swapchain += ms;
            self.max_acq_swapchain_ms = self.max_acq_swapchain_ms.max(ms);
        }
    }

    #[inline]
    pub fn submit(&mut self, ms: f64) {
        if self.enabled {
            self.t_submit += ms;
            self.max_submit_ms = self.max_submit_ms.max(ms);
        }
    }

    #[inline]
    pub fn poll_wait(&mut self, ms: f64) {
        if self.enabled {
            self.t_poll_wait += ms;
            self.max_poll_wait_ms = self.max_poll_wait_ms.max(ms);
        }
    }

    pub fn add_clip_uploads(&mut self, n: usize, bytes: usize) {
        if !self.enabled {
            return;
        }
        self.clip_uploads += n as u64;
        self.clip_bytes += bytes as u64;
    }

    pub fn add_chunk_uploads(&mut self, n: usize) {
        if !self.enabled {
            return;
        }
        self.chunk_uploads += n as u64;
    }

    #[inline]
    pub fn should_print(&self) -> bool {
        self.enabled && self.last_print.elapsed() >= self.print_every
    }

    pub fn end_frame(
        &mut self,
        render_ms: f64,
        prof_overhead_ms: f64,
        stream: Option<StreamStats>,
        gpu: Option<GpuTimingsMs>,
    ) {
        if !self.enabled {
            return;
        }

        self.frame += 1;
        self.n_frames += 1;

        
        self.max_frame_ms = self.max_frame_ms.max(render_ms);

        
        self.t_prof_overhead += prof_overhead_ms;
        self.max_prof_overhead_ms = self.max_prof_overhead_ms.max(prof_overhead_ms);

        
        
        let do_print = self.last_print.elapsed() >= self.print_every;
        if !do_print {
            return;
        }

        let nf = self.n_frames.max(1) as f64;
        let avg = |x: f64| x / nf;

        
        let avg_frame = avg(
            self.t_cam
                + self.t_stream
                + self.t_clipmap_update
                + self.t_cam_write
                + self.t_overlay
                + self.t_chunk_uploads
                + self.t_acq_swapchain
                + self.t_encode_clipmap
                + self.t_encode_compute
                + self.t_encode_blit
                + self.t_submit
                + self.t_poll
                + self.t_poll_wait
                + self.t_present,
        );

        println!(
            concat!(
                "\n[prof] frames={} avg_frame={:.2}ms max_frame={:.2}ms\n",
                "  cam={:.2} stream={:.2} clip_update={:.2} cam_write={:.2} overlay={:.2}\n",
                "  chunk_up={:.2} acq_sc={:.2} enc_clip={:.2} enc_comp={:.2} enc_blit={:.2}\n",
                "  submit={:.2} poll={:.2} poll_wait={:.2} present={:.2}\n",
                "  clip_uploads/frame={:.1} clip_kb/frame={:.1} chunk_uploads/frame={:.1}\n",
                "  prof_overhead(avg/max)={:.2}/{:.2}ms\n",
                "  max: acq_sc={:.2} submit={:.2} poll_wait={:.2} present={:.2}\n",
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
            avg(self.t_acq_swapchain),
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
            avg(self.t_prof_overhead),
            self.max_prof_overhead_ms,
            self.max_acq_swapchain_ms,
            self.max_submit_ms,
            self.max_poll_wait_ms,
            self.max_present_ms,
        );

        if let Some(s) = stream {
            println!(
                concat!(
                    "  [stream] center=({},{},{}) slots={}/{} chunks={} ",
                    "Q={} B={} U={} R={} in_flight={} done_backlog={}\n",
                    "           uploads: rw={} act={} oth={} | cache: {:.1}MB entries={} lru={}\n",
                    "           build_q={} queued_set={} cancels={} orphanQ={}\n",
                    "           builds: done={} canceled={} | queue_ms(avg/max)={:.2}/{:.2} ",
                    "build_ms(avg/max)={:.2}/{:.2} | nodes(avg/max)={:.0}/{:.0}\n",
                ),
                s.center.0,
                s.center.1,
                s.center.2,
                s.resident_slots,
                s.total_slots,
                s.chunks_map,
                s.st_queued,
                s.st_building,
                s.st_uploading,
                s.st_resident,
                s.in_flight,
                s.done_backlog,
                s.up_rewrite,
                s.up_active,
                s.up_other,
                (s.cache_bytes as f64) / (1024.0 * 1024.0),
                s.cache_entries,
                s.cache_lru,
                s.build_queue_len,
                s.queued_set_len,
                s.cancels_len,
                s.orphan_queued,
                s.builds_done,
                s.builds_canceled,
                s.queue_ms_avg,
                s.queue_ms_max,
                s.build_ms_avg,
                s.build_ms_max,
                s.nodes_avg,
                s.nodes_max as f64,
            );

            if s.builds_done > 0 {
                println!(
                    concat!(
                        "           bt_ms(avg/max): total={:.2}/{:.2} height_cache={:.2}/{:.2} tree_mask={:.2}/{:.2}\n",
                        "                         ground_2d={:.2}/{:.2} ground_mip={:.2}/{:.2} tree_top={:.2}/{:.2} tree_mip={:.2}/{:.2}\n",
                        "                         cave_mask={:.2}/{:.2} material_fill={:.2}/{:.2} colinfo={:.2}/{:.2} colinfo_pack={:.2}/{:.2}\n",
                        "                         prefix_x={:.2}/{:.2} prefix_y={:.2}/{:.2} prefix_z={:.2}/{:.2}\n",
                        "                         macro_occ={:.2}/{:.2} svo_build={:.2}/{:.2} ropes={:.2}/{:.2}\n",
                        "           bt_cnt(max): cache_w={} cache_h={} tree_cells_tested={} tree_instances={} solid_voxels={} nodes={}\n"
                    ),
                    s.bt_avg.total,          s.bt_max.total,
                    s.bt_avg.height_cache,   s.bt_max.height_cache,
                    s.bt_avg.tree_mask,      s.bt_max.tree_mask,
                    s.bt_avg.ground_2d,      s.bt_max.ground_2d,
                    s.bt_avg.ground_mip,     s.bt_max.ground_mip,
                    s.bt_avg.tree_top,       s.bt_max.tree_top,
                    s.bt_avg.tree_mip,       s.bt_max.tree_mip,
                    s.bt_avg.cave_mask,      s.bt_max.cave_mask,
                    s.bt_avg.material_fill,  s.bt_max.material_fill,
                    s.bt_avg.colinfo,        s.bt_max.colinfo,
                    s.bt_avg.colinfo_pack,   s.bt_max.colinfo_pack,
                    s.bt_avg.prefix_x,       s.bt_max.prefix_x,
                    s.bt_avg.prefix_y,       s.bt_max.prefix_y,
                    s.bt_avg.prefix_z,       s.bt_max.prefix_z,
                    s.bt_avg.macro_occ,      s.bt_max.macro_occ,
                    s.bt_avg.svo_build,      s.bt_max.svo_build,
                    s.bt_avg.ropes,          s.bt_max.ropes,
                    s.bt_max.cache_w,
                    s.bt_max.cache_h,
                    s.bt_max.tree_cells_tested,
                    s.bt_max.tree_instances,
                    s.bt_max.solid_voxels,
                    s.bt_max.nodes,
                );
            }
        }

        if let Some(g) = gpu {
            println!(
                "  [gpu] primary={:.2}ms godray={:.2}ms composite={:.2}ms blit={:.2}ms total={:.2}ms",
                g.primary, g.godray, g.composite, g.blit, g.total
            );
        }

        
        self.last_print = Instant::now();
        self.n_frames = 0;

        self.t_cam = 0.0;
        self.t_stream = 0.0;
        self.t_clipmap_update = 0.0;
        self.t_cam_write = 0.0;
        self.t_overlay = 0.0;
        self.t_chunk_uploads = 0.0;
        self.t_encode_clipmap = 0.0;
        self.t_encode_compute = 0.0;
        self.t_encode_blit = 0.0;
        self.t_submit = 0.0;
        self.t_poll = 0.0;
        self.t_poll_wait = 0.0;
        self.t_present = 0.0;
        self.t_acq_swapchain = 0.0;

        self.clip_uploads = 0;
        self.clip_bytes = 0;
        self.chunk_uploads = 0;

        self.max_frame_ms = 0.0;

        self.t_prof_overhead = 0.0;
        self.max_prof_overhead_ms = 0.0;

        self.max_acq_swapchain_ms = 0.0;
        self.max_submit_ms = 0.0;
        self.max_poll_wait_ms = 0.0;
        self.max_present_ms = 0.0;
    }
}




pub fn settings_from_args() -> (bool, Duration) {
    #[cfg(target_arch = "wasm32")]
    {
        return (false, Duration::from_millis(500));
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut enabled = false;
        let mut every_ms: Option<u64> = None;

        let mut it = std::env::args().skip(1);
        while let Some(a) = it.next() {
            if a == "--profile" || a == "--profiling" {
                enabled = true;
                continue;
            }

            if let Some(v) = a.strip_prefix("--profile-every-ms=") {
                if let Ok(n) = v.parse::<u64>() {
                    every_ms = Some(n);
                }
                continue;
            }

            if a == "--profile-every-ms" {
                if let Some(v) = it.next() {
                    if let Ok(n) = v.parse::<u64>() {
                        every_ms = Some(n);
                    }
                }
                continue;
            }
        }

        let print_every = Duration::from_millis(every_ms.unwrap_or(2000));
        (enabled, print_every)
    }
}
