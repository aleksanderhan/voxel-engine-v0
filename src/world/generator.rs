// src/world/generator.rs

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::app::config;
use crate::world::materials::{AIR, DIRT, GRASS, STONE, WOOD, LEAF};
use crate::world::hash::{hash2, hash_u32, u01};
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

        let voxels_per_meter = (1.0 / config::VOXEL_SIZE_M_F64) as f32;

        Self { seed, height, detail, cave_a, cave_b, cave_warp, voxels_per_meter }

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

        // Depth below surface in voxels/meters
        let depth_vox = (ground_y_vox - wy) as f32;
        let depth_m = depth_vox / vpm;

        // Small helper: map hash -> [-1,1]
        #[inline(always)]
        fn s11(n: u32) -> f32 {
            (u01(n) - 0.5) * 2.0
        }

        // -----------------------------------------------------------------------------
        // 1) Surface connectivity: shafts / entrances (NOW voxel-precise + irregular)
        // -----------------------------------------------------------------------------
        let roof_m = 3.0;
        let roof_vox = roof_m * vpm;

        let entrance_band_m = 6.0;
        let entrance_band_vox = entrance_band_m * vpm;

        if depth_vox < entrance_band_vox {
            // Meter coords (used ONLY for gating / placement)
            let xm = wx.div_euclid(vpm_i);
            let zm = wz.div_euclid(vpm_i);

            // --- TUNING (entrances / shafts) ---
            let cell_m: i32 = 10;         // was 8 (spread them out)
            let gate_mask: u32 = 31;      // was 15 (rarer: 1/32)

            let mouth_r_m: f32 = 1.6;     // was 3.0
            let shaft_r_m: f32 = 1.1;     // was 2.0
            let shaft_depth_m: f32 = 14.0; // was 22.0

            let probe_depth_m: f32 = 12.0; // was 10.0 (avoid surface swiss-cheese)
            let entrance_thr: f32 = 0.62;  // was 0.46 (much stricter: only real tunnels connect)

            // ----------------------------------

            let gx = xm.div_euclid(cell_m);
            let gz = zm.div_euclid(cell_m);

            let h = hash2(self.seed ^ 0xE17A_0001, gx, gz);
            if (h & gate_mask) == 0 {
                // Entrance cell center in *meters*
                let cxm = gx * cell_m + cell_m / 2;
                let czm = gz * cell_m + cell_m / 2;

                // Convert that to a voxel-space center (sub-meter precision comes from jitter/meander)
                let cxv0 = cxm * vpm_i + vpm_i / 2;
                let czv0 = czm * vpm_i + vpm_i / 2;

                // Per-entrance variability (radius + center jitter)
                let mouth_r_vox = (mouth_r_m * (0.85 + 0.45 * u01(hash_u32(h ^ 0xA11C_0001))) * vpm).max(2.0);
                let shaft_r_vox = (shaft_r_m * (0.85 + 0.35 * u01(hash_u32(h ^ 0xA11C_0002))) * vpm).max(1.5);

                // Center jitter up to ~1.2m
                let jitter_amp_vox = 1.2 * vpm;
                let jx = s11(hash_u32(h ^ 0x51A1_0001)) * jitter_amp_vox;
                let jz = s11(hash_u32(h ^ 0x51A1_0002)) * jitter_amp_vox;

                // Only consider voxels reasonably near this entrance (cheap reject)
                let dx0 = (wx - cxv0) as f32;
                let dz0 = (wz - czv0) as f32;
                let near_r = (mouth_r_vox + 2.0 * vpm).max(shaft_r_vox + 2.0 * vpm);
                if dx0 * dx0 + dz0 * dz0 <= near_r * near_r {
                    // Probe below to ensure we don’t open dead shafts
                    let probe_y = (ground_y_vox as f32 - probe_depth_m * vpm).round() as i32;
                    let (p0, p1) = self.cave_ridged_pair(wx, probe_y, wz);
                    let (q0, q1) = self.cave_ridged_pair_scaled(wx, probe_y, wz, 0.55);

                    let has_cave_below =
                        (p0 > entrance_thr && p1 > entrance_thr) ||
                        (q0 > entrance_thr && q1 > entrance_thr);

                    if has_cave_below {
                        // Depth along the shaft (0 at surface → 1 at shaft bottom)
                        let shaft_depth_vox = (shaft_depth_m * vpm).round();
                        let shaft_bottom = ground_y_vox as f32 - shaft_depth_vox;

                        // Meander: shift the shaft center slightly as we go down
                        // (use cave_warp as a smooth 3D field keyed by entrance cell + depth)
                        let d01 = ((ground_y_vox as f32 - wy as f32) / shaft_depth_vox.max(1.0)).clamp(0.0, 1.0);
                        let meander_amp = (0.9 * vpm) * (d01 * d01); // grows with depth
                        let nmx = self.cave_warp.get([
                            (cxm as f64) * 0.11,
                            (depth_m as f64) * 0.23,
                            (czm as f64) * 0.11,
                        ]) as f32;
                        let nmz = self.cave_warp.get([
                            (cxm as f64) * 0.11 + 19.7,
                            (depth_m as f64) * 0.23 - 6.3,
                            (czm as f64) * 0.11 + 8.9,
                        ]) as f32;

                        let cx = (cxv0 as f32) + jx + nmx * meander_amp;
                        let cz = (czv0 as f32) + jz + nmz * meander_amp;

                        // Radius transitions: mouth → shaft (no hard step)
                        let t = (depth_m / entrance_band_m).clamp(0.0, 1.0);
                        let smooth = t * t * (3.0 - 2.0 * t); // smoothstep
                        let r_base = mouth_r_vox + (shaft_r_vox - mouth_r_vox) * smooth;

                        // Boundary noise so the mouth isn’t a perfect circle
                        let wxm = (wx as f64) * config::VOXEL_SIZE_M_F64;
                        let wym = (wy as f64) * config::VOXEL_SIZE_M_F64;
                        let wzm = (wz as f64) * config::VOXEL_SIZE_M_F64;

                        let bn = self.cave_warp.get([wxm * 0.55, wym * 0.22, wzm * 0.55]) as f32; // ~[-1,1]
                        let rough = 1.0 + 0.22 * bn * (1.0 - smooth); // stronger near the mouth
                        let r = (r_base * rough).max(1.5);

                        // Carve only above shaft bottom
                        if (wy as f32) >= shaft_bottom {
                            let dx = (wx as f32) - cx;
                            let dz = (wz as f32) - cz;
                            if dx * dx + dz * dz <= r * r {
                                return true;
                            }
                        }
                    }
                }
            }

            // Keep the roof intact very near the surface unless we’re inside an accepted shaft carve
            if depth_vox < roof_vox {
                return false;
            }
        }

        // -----------------------------------------------------------------------------
        // 2) Underground: tunnel-only (single scale, no chambers)
        // -----------------------------------------------------------------------------
        let max_depth_m = 48.0; // was 55; shallower caves feel less "world-spanning"
        if depth_m > max_depth_m {
            return false;
        }

        // Depth ramp 0..1 below roof
        let t = ((depth_m - roof_m) / (max_depth_m - roof_m)).clamp(0.0, 1.0);

        // Single scale: use the ridged intersection, but make it THIN.
        // Using min() forces BOTH ridges to be near their zero-crossing simultaneously -> tube centerline.
        let (r0, r1) = self.cave_ridged_pair(wx, wy, wz);
        let mut ridge = r0.min(r1);

        // Slight vertical breakup so we don't get long planar strata
        ridge *= 0.96 + 0.04 * (((wy & 31) as f32) * (1.0 / 31.0));

        // Tight threshold: small tunnels near surface, only slightly wider deeper.
        // (Higher threshold => thinner tunnels.)
        let mut thr = 0.78 + (0.70 - 0.78) * t; // 0.78 -> 0.70

        // Tiny threshold noise breaks uniform walls WITHOUT creating caverns.
        let wxm = (wx as f64) * config::VOXEL_SIZE_M_F64;
        let wym = (wy as f64) * config::VOXEL_SIZE_M_F64;
        let wzm = (wz as f64) * config::VOXEL_SIZE_M_F64;
        let tn = self.cave_warp.get([wxm * 0.40, wym * 0.24, wzm * 0.40]) as f32; // ~[-1,1]
        thr += 0.010 * tn;

        // Final carve: tunnel-only
        ridge > thr

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
}
