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
    // - cave_room: low-frequency modulation for chambers
    cave_a: Fbm<Perlin>,
    cave_b: Fbm<Perlin>,

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
    pub fn carve_cave(&self, wx: i32, wy: i32, wz: i32, ground_y_vox: i32) -> bool {
        let vpm = config::VOXELS_PER_METER as f32;
        let vpm_i = config::VOXELS_PER_METER;

        // Depth below surface
        let depth_vox = (ground_y_vox - wy) as f32;
        let depth_m = depth_vox / vpm;

        // -----------------------------------------------------------------------------
        // 0) Quick reject: never carve above ground
        // -----------------------------------------------------------------------------
        if wy > ground_y_vox {
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
        // 4) Entrances: walk-in meandering tunnel from surface to the main tunnel layer
        // -----------------------------------------------------------------------------
        let entrance_band_m = 10.0; // allow entrances closer to surface for visibility
        let roof_m = 3.0;

        if depth_m < entrance_band_m {
            // Grid in meters
            let cell_m: i32 = 32;

            // Entrance density gate:
            // 0 => debugging (every cell). For production try e.g. 31 (~1/32), 63 (~1/64), 127...
            let gate_mask: u32 = 0;

            let xm_i = wx.div_euclid(vpm_i);
            let zm_i = wz.div_euclid(vpm_i);
            let gx = xm_i.div_euclid(cell_m);
            let gz = zm_i.div_euclid(cell_m);

            let h = hash2(self.seed ^ 0xE17A_0001, gx, gz);

            if (h & gate_mask) == 0 {
                // Cell center (meters) + jitter
                let cxm0 = gx * cell_m + cell_m / 2;
                let czm0 = gz * cell_m + cell_m / 2;

                let jitter_m = 4.0;
                let cxm = (cxm0 as f32 + s11(hash_u32(h ^ 0x51A1_0001)) * jitter_m).round() as i32;
                let czm = (czm0 as f32 + s11(hash_u32(h ^ 0x51A1_0002)) * jitter_m).round() as i32;

                // Entrance "mouth" start in voxels
                let sxv = cxm * vpm_i + vpm_i / 2;
                let szv = czm * vpm_i + vpm_i / 2;

                // Only place an entrance where the 2D tunnel network exists under it.
                let (er0, er1) = self.cave_ridged_pair_2d(sxv, szv);
                let entrance_ridge2d = er0.min(er1);

                // Slightly stricter than thr2d to avoid borderline "almost tunnels"
                if entrance_ridge2d > (thr2d + 0.05) {
                    // Bottom of the entrance should meet (or slightly overshoot) the tunnel layer
                    let bottom_y = (tunnel_center_y - 1.5 * vpm) as i32;
                    let top_y = ground_y_vox;

                    if wy >= bottom_y && wy <= top_y {
                        // 0 at surface, 1 at bottom
                        let denom = (top_y - bottom_y).max(1) as f32;
                        let t = ((top_y - wy) as f32 / denom).clamp(0.0, 1.0);

                        // Smoothstep for nicer shaping
                        let ts = t * t * (3.0 - 2.0 * t);

                        // Choose a “walk-in” slope direction + horizontal reach (in meters)
                        // Longer reach => gentler slope.
                        let ang = (u01(hash_u32(h ^ 0x51A1_1000)) * std::f32::consts::TAU) as f32;
                        let (dirx, dirz) = (ang.cos(), ang.sin());

                        let layer_depth_m_clamped = layer_depth_m.clamp(4.0, 18.0);
                        let horiz_m =
                            (8.0 + 0.75 * layer_depth_m_clamped + 6.0 * u01(hash_u32(h ^ 0x51A1_1003)))
                                .clamp(10.0, 24.0);

                        let ex = (sxv as f32) + dirx * (horiz_m * vpm);
                        let ez = (szv as f32) + dirz * (horiz_m * vpm);

                        // Base centerline (simple lerp)
                        let mut cx = (sxv as f32) * (1.0 - ts) + ex * ts;
                        let mut cz = (szv as f32) * (1.0 - ts) + ez * ts;

                        // Add meander using your existing 2D warp noise (cheap, stable in meters)
                        // Make wiggle strongest mid-way, weaker at the endpoints.
                        let mid_boost = (1.0 - (2.0 * t - 1.0).abs()).clamp(0.0, 1.0);

                        let cxm_f = (cx as f64) * config::VOXEL_SIZE_M_F64;
                        let czm_f = (cz as f64) * config::VOXEL_SIZE_M_F64;

                        let me0 = self.cave_warp2.get([cxm_f + (t as f64) * 17.0, czm_f - (t as f64) * 11.0]) as f32;
                        let me1 = self.cave_warp2.get([cxm_f - (t as f64) * 13.0, czm_f + (t as f64) * 19.0]) as f32;

                        let wiggle_m = (0.9 + 0.8 * u01(hash_u32(h ^ 0x51A1_1004))) * mid_boost;
                        cx += me0 * (wiggle_m * vpm);
                        cz += me1 * (wiggle_m * vpm);

                        // Radius tapers from a wider mouth to a tunnel-ish width
                        let mouth_r = 1.7 * vpm;
                        let tunnel_r = 1.1 * vpm;
                        let r = mouth_r * (1.0 - ts) + tunnel_r * ts;

                        let dx = (wx as f32) - cx;
                        let dz = (wz as f32) - cz;

                        if dx * dx + dz * dz <= r * r {
                            return true;
                        }

                        // Optional: a small “threshold” widening right at the entrance
                        // so it feels like a natural mouth you can see.
                        if t <= 0.12 {
                            let r2 = (mouth_r * 1.25) * (mouth_r * 1.25);
                            if dx * dx + dz * dz <= r2 {
                                return true;
                            }
                        }
                    }
                } else {
                    // no tunnel network here => no entrance; keep surface intact
                    // (don’t return false here; just fall through to roof protection)
                }
            }

            // Roof protection: keep near-surface intact unless we returned true above
            if depth_m < roof_m {
                return false;
            }
        }





        // -----------------------------------------------------------------------------
        // 5) Final carve: main tunnel layer
        // -----------------------------------------------------------------------------
        // If we're near the surface, ONLY entrances/shafts are allowed to open.
        // Prevent the main tunnel field from "perforating" the ground into swiss cheese.
        if depth_m < entrance_band_m {
            return false;
        }

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
