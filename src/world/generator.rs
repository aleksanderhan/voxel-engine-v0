// src/world/generator.rs

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::app::config;
use crate::world::materials::{AIR, DIRT, GRASS, STONE, WOOD, LEAF};
use crate::world::hash::{hash2, hash_u32, u01, s11};
use crate::world::edits::{EditStore, voxel_to_chunk_local};

#[derive(Clone)]
pub struct WorldGen {
    pub seed: u32,
    height: Fbm<Perlin>,
    detail: Fbm<Perlin>,

    // Caves:
    // - cave_a + cave_b: two independent fields; tunnels are their intersection (AND)
    // - cave_warp: domain warp for twist
    // - cave_room: low-frequency modulation for chambers
    cave_a: Fbm<Perlin>,
    cave_b: Fbm<Perlin>,
    cave_warp: Fbm<Perlin>,

    // 2.5D tunnel shaping:
    cave_level: Fbm<Perlin>,
    cave_warp2: Fbm<Perlin>,

    voxels_per_meter: f32,
}


impl WorldGen {
    pub fn new(seed: u32) -> Self {
        let height = Fbm::<Perlin>::new(seed).set_octaves(7).set_frequency(0.010);
        let detail = Fbm::<Perlin>::new(seed ^ 0xA5A5_A5A5).set_octaves(3).set_frequency(0.02);

        // Cave knobs: smaller, tunnel-only
        // Higher frequency => smaller features (tighter tunnels).
        let cave_a = Fbm::<Perlin>::new(seed ^ 0xB4B4_B4B4).set_octaves(4).set_frequency(0.135);
        let cave_b = Fbm::<Perlin>::new(seed ^ 0xD1D1_D1D1).set_octaves(4).set_frequency(0.148);

        // Warp: keep twist but prevent cavern blow-ups.
        let cave_warp = Fbm::<Perlin>::new(seed ^ 0xC3C3_C3C3).set_octaves(2).set_frequency(0.030);

        // 2.5D tunnel layer controls:
        // Low frequency => long meanders and stable “tunnel layer” depth.
        let cave_level = Fbm::<Perlin>::new(seed ^ 0x9C9C_9C9C).set_octaves(2).set_frequency(0.010);
        let cave_warp2 = Fbm::<Perlin>::new(seed ^ 0x7D7D_7D7D).set_octaves(2).set_frequency(0.020);


        let voxels_per_meter = (1.0 / config::VOXEL_SIZE_M_F64) as f32;

        Self {
            seed,
            height,
            detail,
            cave_a,
            cave_b,
            cave_warp,
            cave_level,
            cave_warp2,
            voxels_per_meter,
        }


    }

    #[inline(always)]
    pub fn ground_height_m(&self, xm: f64, zm: f64) -> i32 {
        let h0 = self.height.get([xm, zm]) as f32;
        let h1 = self.detail.get([xm, zm]) as f32;

        let base_m = 10.0;
        let amp_m  = 18.0;
        let hills_m = h0 * amp_m + h1 * 3.0;

        ((base_m + hills_m) * self.voxels_per_meter).round() as i32
    }

    #[inline]
    pub fn ground_height(&self, x_vox: i32, z_vox: i32) -> i32 {
        let xm = (x_vox as f64) * config::VOXEL_SIZE_M_F64;
        let zm = (z_vox as f64) * config::VOXEL_SIZE_M_F64;
        self.ground_height_m(xm, zm)
    }


    #[inline(always)]
    fn cave_ridged_pair_scaled(&self, wx: i32, wy: i32, wz: i32, scale: f64) -> (f32, f32) {
        // Sample in meters so frequencies are stable even if VOXEL_SIZE changes.
        let xm = (wx as f64) * config::VOXEL_SIZE_M_F64 * scale;
        let ym = (wy as f64) * config::VOXEL_SIZE_M_F64 * scale;
        let zm = (wz as f64) * config::VOXEL_SIZE_M_F64 * scale;

        // Domain warp (shared), but decorrelate the two fields slightly.
        let w0 = self.cave_warp.get([xm, ym, zm]) as f32; // ~[-1,1]
        let w1 = self.cave_warp.get([xm + 31.7, ym - 12.4, zm + 8.9]) as f32;

        let warp_m = 2.8; // meters; lower => fewer huge open spaces
        let xw0 = xm + (w0 as f64) * (warp_m as f64);
        let yw0 = ym + (w0 as f64) * (warp_m as f64) * 0.30; // less vertical warp
        let zw0 = zm - (w0 as f64) * (warp_m as f64);

        let xw1 = xm + (w1 as f64) * (warp_m as f64);
        let yw1 = ym + (w1 as f64) * (warp_m as f64) * 0.30;
        let zw1 = zm - (w1 as f64) * (warp_m as f64);

        let n0 = self.cave_a.get([xw0, yw0, zw0]) as f32; // ~[-1,1]
        let n1 = self.cave_b.get([xw1, yw1, zw1]) as f32;

        // Ridged: high near zero-crossings.
        let r0 = 1.0 - n0.abs();
        let r1 = 1.0 - n1.abs();
        (r0, r1)
    }

    #[inline(always)]
    fn cave_ridged_pair(&self, wx: i32, wy: i32, wz: i32) -> (f32, f32) {
        self.cave_ridged_pair_scaled(wx, wy, wz, 1.0)
    }

    #[inline(always)]
    pub fn carve_cave(&self, wx: i32, wy: i32, wz: i32, ground_y_vox: i32) -> bool {
        let vpm = config::VOXELS_PER_METER as f32;
        let vpm_i = config::VOXELS_PER_METER;

        // Depth below surface
        let depth_vox = (ground_y_vox - wy) as f32;
        let depth_m = depth_vox / vpm;

        // -----------------------------------------------------------------------------
        // 0) Quick reject: never carve above ground
        // -----------------------------------------------------------------------------
        if wy >= ground_y_vox {
            return false;
        }

        // -----------------------------------------------------------------------------
        // 1) Compute the main tunnel layer center height (2.5D)
        //    Tunnels mostly live around: ground - layer_depth(x,z)
        // -----------------------------------------------------------------------------
        let xm = (wx as f64) * config::VOXEL_SIZE_M_F64;
        let zm = (wz as f64) * config::VOXEL_SIZE_M_F64;

        // layer depth varies smoothly: 4m..14m below ground (much closer to surface)
        let lv = self.cave_level.get([xm, zm]) as f32; // ~[-1,1]
        let lv01 = 0.5 + 0.5 * lv;
        let layer_depth_m = 4.0 + 10.0 * lv01;


        let tunnel_center_y = (ground_y_vox as f32) - layer_depth_m * vpm;
        let dy_to_center = (wy as f32) - tunnel_center_y;

        // -----------------------------------------------------------------------------
        // 2) Tunnel “map” in XZ (ridged intersection => long meandering networks)
        // -----------------------------------------------------------------------------
        let (r0, r1) = self.cave_ridged_pair_2d(wx, wz);
        let ridge2d = r0.min(r1);

        // Threshold controls how sparse the tunnel network is.
        // Higher => fewer, more distinct tunnels.
        let thr2d = 0.72;

        // -----------------------------------------------------------------------------
        // 3) Tunnel radius: narrower near surface layer, slightly wider deeper
        // -----------------------------------------------------------------------------
        // Make the vertical tube half-thickness depend on depth a bit (in meters).
        // This is NOT making caverns; it's just "comfortable tunnels".
        let depth_widen = ((layer_depth_m - 10.0) / 16.0).clamp(0.0, 1.0);
        let radius_m = 1.8 + 0.7 * depth_widen; // ~1.1m .. ~1.8m
        let radius_vox = radius_m * vpm;

        // Soft center bias: stronger ridge => more likely to be “in the tunnel core”.
        // This avoids thick sheets and gives a more tube-like feel.
        let core = (ridge2d - thr2d) * (1.0 / (1.0 - thr2d)); // ~0..1 when above thr2d
        let core = core.clamp(0.0, 1.0);
        let effective_r = radius_vox * (0.55 + 0.60 * core); // thinner on edges

        let in_main_tunnel = ridge2d > thr2d && dy_to_center.abs() <= effective_r;

        // -----------------------------------------------------------------------------
        // 4) Entrances / shafts: only where the main tunnel actually exists below
        // -----------------------------------------------------------------------------
        let entrance_band_m = 6.0;
        let roof_m = 3.0;

        let entrance_band_m = 6.0;
        let roof_m = 3.0;

        if depth_m < entrance_band_m {
            // Probe: only open entrances where the 2D tunnel line is strong
            let (pr0, pr1) = self.cave_ridged_pair_2d(wx, wz);
            let probe_ridge = pr0.min(pr1);

            let thr2d = 0.72; // MUST match the thr2d you use for in_main_tunnel

            let probe_core = ((probe_ridge - thr2d) * (1.0 / (1.0 - thr2d))).clamp(0.0, 1.0);
            let has_tunnel_here = probe_ridge > (thr2d + 0.06) && probe_core > 0.35;

            // Entrance placement grid (in meters)
            let cell_m: i32 = 14;
            let gate_mask: u32 = 3; // 1/4 for testing; use 7 for 1/8, 15 for 1/16

            let xm_i = wx.div_euclid(vpm_i);
            let zm_i = wz.div_euclid(vpm_i);
            let gx = xm_i.div_euclid(cell_m);
            let gz = zm_i.div_euclid(cell_m);

            let h = hash2(self.seed ^ 0xE17A_0001, gx, gz);

            if has_tunnel_here && (h & gate_mask) == 0 {
                // Cell center in meters
                let cxm0 = gx * cell_m + cell_m / 2;
                let czm0 = gz * cell_m + cell_m / 2;

                // Jitter inside the cell so entrances aren’t a perfect grid
                let jitter_m = 4.0;
                let jx = s11(hash_u32(h ^ 0x51A1_0001)) * jitter_m;
                let jz = s11(hash_u32(h ^ 0x51A1_0002)) * jitter_m;

                let cxm = (cxm0 as f32 + jx).round() as i32;
                let czm = (czm0 as f32 + jz).round() as i32;

                // Convert entrance center to voxels
                let cxv = cxm * vpm_i + vpm_i / 2;
                let czv = czm * vpm_i + vpm_i / 2;

                // Cheap reject: only carve when we're near this entrance center
                let dx0 = (wx - cxv) as f32;
                let dz0 = (wz - czv) as f32;
                let max_r_vox = (2.2 * vpm).max(8.0);

                if dx0 * dx0 + dz0 * dz0 <= max_r_vox * max_r_vox {
                    // Shaft radii
                    let mouth_r_m = 2.2;
                    let shaft_r_m = 1.5;

                    let mouth_r = mouth_r_m * vpm;
                    let shaft_r = shaft_r_m * vpm;

                    // Carve down to (a bit below) the tunnel layer center
                    let shaft_bottom = (tunnel_center_y - 2.0 * vpm).min((ground_y_vox as f32) - roof_m * vpm);

                    if (wy as f32) >= shaft_bottom {
                        // Smooth mouth -> shaft radius through entrance band
                        let t = (depth_m / entrance_band_m).clamp(0.0, 1.0);
                        let smooth = t * t * (3.0 - 2.0 * t); // smoothstep
                        let r = mouth_r + (shaft_r - mouth_r) * smooth;

                        // Actual carve test
                        let dx = (wx - cxv) as f32;
                        let dz = (wz - czv) as f32;
                        if dx * dx + dz * dz <= r * r {
                            return true;
                        }
                    }
                }
            }

            // Roof protection: don't swiss-cheese the surface unless we returned true above
            if depth_m < roof_m {
                return false;
            }
        }


        // -----------------------------------------------------------------------------
        // 5) Final carve: main tunnel layer
        // -----------------------------------------------------------------------------
        in_main_tunnel
    }



    pub fn material_at_voxel(&self, wx: i32, wy: i32, wz: i32) -> u32 {
        // --- terrain ---
        let g = self.ground_height(wx, wz);

        // match builder’s “dirt_depth = 3 * vpm”
        let vpm = config::VOXELS_PER_METER;
        let dirt_depth = 3 * vpm;

        let mut m = if wy < g {
            if wy >= g - dirt_depth { DIRT } else { STONE }
        } else if wy == g {
            GRASS
        } else {
            AIR
        };

        // --- caves carve terrain/stone into AIR ---
        // Only carve underground solids.
        if m != AIR {
            if self.carve_cave(wx, wy, wz, g) {
                m = AIR;
            }
        }

        // --- trees overlay (so rays can hit them) ---
        // Trees only write into AIR, so caves won’t delete trees (good).
        if m == AIR {
            if let Some(tm) = self.tree_material_at_voxel(wx, wy, wz) {
                m = tm;
            }
        }

        m
    }

    /// Query world material with edits taking priority (edits can “delete” tree voxels).
    /// This is the one you want for picking / interaction.
    pub fn material_at_voxel_with_edits(
        &self,
        edits: &EditStore,
        wx: i32,
        wy: i32,
        wz: i32,
    ) -> u32 {
        // 1) edits override everything
        let (ck, lx, ly, lz) = voxel_to_chunk_local(&self, wx, wy, wz);
        if let Some(mat) = edits.get_override(ck, lx, ly, lz) {
            return mat;
        }

        // 2) otherwise query generated world (terrain + trees)
        self.material_at_voxel(wx, wy, wz)
    }

    /// Returns Some(WOOD/LEAF) if (wx,wy,wz) is inside an approximate tree volume.
    #[inline]
    fn tree_material_at_voxel(&self, wx: i32, wy: i32, wz: i32) -> Option<u32> {
        let vpm = config::VOXELS_PER_METER;

        // Your trees are placed on a meter grid cell. Reconstruct trunk position.
        let xm = wx.div_euclid(vpm);
        let zm = wz.div_euclid(vpm);

        // Match trees.rs placement: hash2(seed, xm, zm) and TREE_CELL_MOD gating.
        // NOTE: keep TREE_CELL_MOD in sync with trees.rs (currently 256).
        const TREE_CELL_MOD: u32 = 256;

        let seed = hash2(self.seed, xm, zm);
        if (seed % TREE_CELL_MOD) != 0 {
            return None;
        }

        // Trunk position in voxel coords:
        let tx = xm * vpm;
        let tz = zm * vpm;

        // Ground at trunk
        let ground = self.ground_height(tx, tz);

        // Match trees.rs: trunk base is ground + vpm
        let base_y = ground + vpm;

        // Match trees.rs: trunk_h_m = 5..10m, crown_r_m = 3..6m
        let trunk_h_m = 5 + (hash_u32(seed) % 6) as i32;
        let crown_r_m = 3 + (hash_u32(seed ^ 0xBEEF) % 4) as i32;

        let trunk_h = trunk_h_m * vpm;
        let crown_r = crown_r_m * vpm;

        let top_y = base_y + trunk_h;

        // Outside tree vertical span (with a bit of canopy headroom)
        if wy < base_y - 2 || wy > top_y + 8 * vpm {
            return None;
        }

        // Approx trunk radius (match trees.rs)
        let vpm_f = vpm as f32;
        let r0 = 0.45 * vpm_f + u01(hash_u32(seed ^ 0x1111)) * (0.30 * vpm_f);
        let r1 = 0.18 * vpm_f + u01(hash_u32(seed ^ 0x2222)) * (0.12 * vpm_f);

        // Linear taper along trunk
        if wy >= base_y && wy <= top_y {
            let t = ((wy - base_y) as f32 / (trunk_h as f32)).clamp(0.0, 1.0);
            let rr = (r0 + (r1 - r0) * t).max(1.0);

            let dx = (wx - tx) as f32;
            let dz = (wz - tz) as f32;

            if dx * dx + dz * dz <= rr * rr {
                return Some(WOOD);
            }
        }

        // Approx canopy: a soft sphere-ish region around trunk tip.
        // This is intentionally generous so leaves become “hittable”.
        let cy = top_y + 5 * vpm; // similar to builder’s canopy_h_vox
        let dx = (wx - tx) as f32;
        let dy = (wy - cy) as f32;
        let dz = (wz - tz) as f32;

        let cr = (crown_r + 2 * vpm) as f32;
        if dx * dx + (dy * 0.85) * (dy * 0.85) + dz * dz <= cr * cr {
            return Some(LEAF);
        }

        None
    }

    #[inline(always)]
    fn cave_ridged_pair_2d(&self, wx: i32, wz: i32) -> (f32, f32) {
        // Sample in meters.
        let xm = (wx as f64) * config::VOXEL_SIZE_M_F64;
        let zm = (wz as f64) * config::VOXEL_SIZE_M_F64;

        // Gentle 2D warp in XZ (keeps tunnels meandering, not “grid-noise”).
        let w = self.cave_warp2.get([xm, zm]) as f32;           // ~[-1,1]
        let w2 = self.cave_warp2.get([xm + 23.7, zm - 11.3]) as f32;

        let warp_m = 6.0; // meters (bigger => more meander)
        let xw0 = xm + (w as f64) * (warp_m as f64);
        let zw0 = zm - (w as f64) * (warp_m as f64);

        let xw1 = xm + (w2 as f64) * (warp_m as f64);
        let zw1 = zm - (w2 as f64) * (warp_m as f64);

        // Use your existing cave_a/cave_b but sample them as “2D” by fixing Y=0.
        let n0 = self.cave_a.get([xw0, 0.0, zw0]) as f32;
        let n1 = self.cave_b.get([xw1, 0.0, zw1]) as f32;

        // Ridged: high near zero-crossings => “lines”.
        (1.0 - n0.abs(), 1.0 - n1.abs())
    }

}

