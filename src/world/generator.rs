// src/world/generator.rs

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use crate::app::config;
use crate::world::materials::{AIR, DIRT, GRASS, STONE, WOOD, LEAF};
use crate::world::hash::{hash2, hash_u32, u01};
use crate::world::edits::{EditStore, voxel_to_chunk_local};
use crate::world::hash::hash3;

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
    cave_room: Fbm<Perlin>,

    voxels_per_meter: f32,
}


impl WorldGen {
    pub fn new(seed: u32) -> Self {
        let height = Fbm::<Perlin>::new(seed).set_octaves(7).set_frequency(0.010);
        let detail = Fbm::<Perlin>::new(seed ^ 0xA5A5_A5A5).set_octaves(3).set_frequency(0.02);

        // Cave knobs:
        // Higher frequency => smaller features (tighter tunnels).
        let cave_a = Fbm::<Perlin>::new(seed ^ 0xB4B4_B4B4).set_octaves(3).set_frequency(0.085);
        let cave_b = Fbm::<Perlin>::new(seed ^ 0xD1D1_D1D1).set_octaves(3).set_frequency(0.093);

        // Warp: keep twist, but reduce "cathedral blobs" by not over-warping vertically.
        let cave_warp = Fbm::<Perlin>::new(seed ^ 0xC3C3_C3C3).set_octaves(3).set_frequency(0.024);

        // Low-frequency modulation for rooms/chambers (big features but gated rare).
        let cave_room = Fbm::<Perlin>::new(seed ^ 0xE6E6_E6E6).set_octaves(2).set_frequency(0.018);


        let voxels_per_meter = (1.0 / config::VOXEL_SIZE_M_F64) as f32;

        Self { seed, height, detail, cave_a, cave_b, cave_warp, cave_room, voxels_per_meter }

    }


    #[inline]
    pub fn ground_height(&self, x_vox: i32, z_vox: i32) -> i32 {
        let xm = (x_vox as f64) * config::VOXEL_SIZE_M_F64;
        let zm = (z_vox as f64) * config::VOXEL_SIZE_M_F64;

        let h0 = self.height.get([xm, zm]) as f32;
        let h1 = self.detail.get([xm, zm]) as f32;

        let base_m = 10.0;
        let amp_m  = 18.0;
        let hills_m = h0 * amp_m + h1 * 3.0;

        ((base_m + hills_m) * self.voxels_per_meter).round() as i32
    }

    #[inline(always)]
    fn cave_ridged_pair(&self, wx: i32, wy: i32, wz: i32) -> (f32, f32) {
        // Sample in meters so frequencies are stable even if VOXEL_SIZE changes.
        let xm = (wx as f64) * config::VOXEL_SIZE_M_F64;
        let ym = (wy as f64) * config::VOXEL_SIZE_M_F64;
        let zm = (wz as f64) * config::VOXEL_SIZE_M_F64;

        // Domain warp (shared), but decorrelate the two fields slightly.
        let w0 = self.cave_warp.get([xm, ym, zm]) as f32; // ~[-1,1]
        let w1 = self.cave_warp.get([xm + 31.7, ym - 12.4, zm + 8.9]) as f32;

        let warp_m = 4.5; // meters; lower => fewer huge open spaces
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
    pub fn carve_cave(&self, wx: i32, wy: i32, wz: i32, ground_y_vox: i32) -> bool {
        // Depth below surface in meters
        let vpm = config::VOXELS_PER_METER as f32;
        let depth_vox = (ground_y_vox - wy) as f32;

        // Don’t open caves at/near the surface by default (prevents swiss-cheese terrain),
        // BUT allow rare entrances that connect to cave noise below.
        let roof_m = 3.0;
        let roof_vox = roof_m * vpm;

        if depth_vox < roof_vox {
            let vpm_i = config::VOXELS_PER_METER;
            let xm = wx.div_euclid(vpm_i);
            let zm = wz.div_euclid(vpm_i);

            // ---- TUNING ----
            // Smaller cell => more candidate cells
            let cell_m: i32 = 6;           // was 12
            // Lower mask => higher probability (mask=127 => 1/128)
            let gate_mask: u32 = 31;      // was 511
            // Wider mouth radius so you can actually *see* the entrance
            let mouth_r_m: i32 = 4;        // 3m radius (~7m diameter)
            // Slightly easier threshold at the probe depth
            let entrance_thr: f32 = 0.50;  // was 0.58
            // ----------------

            let gx = xm.div_euclid(cell_m);
            let gz = zm.div_euclid(cell_m);

            let h = hash2(self.seed ^ 0xE17A_0001, gx, gz);
            if (h & gate_mask) != 0 {
                return false;
            }

            // Carve a visible "mouth" disk around the cell center in meters.
            let cxm = gx * cell_m + cell_m / 2;
            let czm = gz * cell_m + cell_m / 2;
            let dxm = xm - cxm;
            let dzm = zm - czm;
            if dxm * dxm + dzm * dzm > mouth_r_m * mouth_r_m {
                return false;
            }

            // Probe below the roof so we only open when there's a cave worth connecting to.
            let probe_y = ground_y_vox - roof_vox as i32;
            let (r0p, r1p) = self.cave_ridged_pair(wx, probe_y, wz);
            // Entrance opens only if a *tunnel* exists below (prevents random surface holes).
            return (r0p > entrance_thr) && (r1p > entrance_thr);

        }



        // If you only stream a small vertical band, keep caves inside it:
        // (This still works without this clamp; it just reduces “wasted” patterns.)
        let max_depth_m = 40.0;
        if depth_vox > max_depth_m * vpm {
            return false;
        }

        // Make caves more likely deeper down.
        let depth_m = depth_vox / vpm;
        let t = ((depth_m - roof_m) / (max_depth_m - roof_m)).clamp(0.0, 1.0);

        // --- TUNNELS (Minecraft-y) ---
        // Two ridged fields; require BOTH high -> tunnels instead of giant sheets/cathedrals.
        let (r0, r1) = self.cave_ridged_pair(wx, wy, wz);

        // Threshold ramps with depth: shallow => stricter, deep => looser.
        // (Slightly stricter overall than before to avoid huge voids.)
        let thr_shallow = 0.62;
        let thr_deep = 0.52;
        let thr = thr_shallow + (thr_deep - thr_shallow) * t;

        // Vertical anti-sheet jitter (keeps things from becoming huge planar caverns)
        let y_bias = ((wy as i32) & 7) as f32 * (1.0 / 7.0);
        let r0 = r0 * (0.94 + 0.06 * y_bias);
        let r1 = r1 * (0.94 + 0.06 * (1.0 - y_bias));

        let carve_tunnel = (r0 > thr) && (r1 > thr);

        // --- CHAMBERS (rare bigger interiors) ---
        // Seed occasional spheres in a 3D meter-grid so you get rooms connected by tunnels.
        let vpm_i = config::VOXELS_PER_METER;
        let xm = wx.div_euclid(vpm_i);
        let ym = wy.div_euclid(vpm_i);
        let zm = wz.div_euclid(vpm_i);

        // Room grid in meters: bigger = rarer.
        let room_cell_m: i32 = 18;
        let rx = xm.div_euclid(room_cell_m);
        let ry = ym.div_euclid(room_cell_m);
        let rz = zm.div_euclid(room_cell_m);

        // 1/(mask+1) chance per cell.
        let room_gate_mask: u32 = 63; // 1/64
        let h = hash3(self.seed ^ 0xCAFE_600D, rx, ry, rz);

        let mut carve_room = false;
        if (h & room_gate_mask) == 0 {
            // Cell center (meters)
            let cxm = rx * room_cell_m + room_cell_m / 2;
            let cym = ry * room_cell_m + room_cell_m / 2;
            let czm = rz * room_cell_m + room_cell_m / 2;

            let dxm = xm - cxm;
            let dym = ym - cym;
            let dzm = zm - czm;

            // Radius in meters (Minecraft-ish "small room" range)
            let r_m = 4 + (hash_u32(h ^ 0x1234) % 6) as i32; // 4..9m

            // Slightly squash vertically so rooms feel cave-like, not domes.
            let d2 = (dxm * dxm) + (dzm * dzm) + ((dym * dym) * 2);

            if d2 <= r_m * r_m {
                // Modulate with a low-frequency noise so rooms aren't perfect spheres.
                let wxm = (wx as f64) * config::VOXEL_SIZE_M_F64;
                let wym = (wy as f64) * config::VOXEL_SIZE_M_F64;
                let wzm = (wz as f64) * config::VOXEL_SIZE_M_F64;
                let rn = self.cave_room.get([wxm, wym, wzm]) as f32; // ~[-1,1]
                let room_r = 1.0 - rn.abs();

                // Only keep the "core" of the chamber; prevents giant blown-out volumes.
                carve_room = room_r > 0.55;
            }
        }

        // Final carve: tunnels everywhere, rooms only where the rare chamber condition hits.
        carve_tunnel || carve_room

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
