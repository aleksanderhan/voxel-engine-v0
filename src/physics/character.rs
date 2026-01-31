use glam::Vec3;

use crate::physics::voxel::VoxelQuery;

#[derive(Clone, Copy, Debug)]
pub struct CharacterState {
    pub pos: Vec3,   // meters
    pub vel: Vec3,   // m/s
    pub on_ground: bool,
}

pub struct CharacterController {
    pub radius: f32,        // meters
    pub height: f32,        // meters (capsule total height)
    pub gravity: f32,       // m/s^2 (positive value)
    pub max_slope_cos: f32, // not used yet; placeholder for slope limit
}

impl Default for CharacterController {
    fn default() -> Self {
        Self {
            radius: 0.35,
            height: 1.7,
            gravity: 9.81,
            max_slope_cos: 0.7,
        }
    }
}

impl CharacterController {
    #[inline]
    fn feet_y(&self, pos: Vec3) -> f32 {
        // capsule bottom center is at pos.y - (height*0.5 - radius)
        pos.y - (self.height * 0.5 - self.radius)
    }

    #[inline]
    fn set_feet_y(&self, pos: &mut Vec3, feet_y: f32) {
        pos.y = feet_y + (self.height * 0.5 - self.radius);
    }

    /// Step the controller:
    /// - apply gravity
    /// - integrate velocity
    /// - resolve ground penetration using ground_height_m
    pub fn step(&self, st: &mut CharacterState, q: &dyn VoxelQuery, dt: f32) {
        st.on_ground = false;

        // gravity
        st.vel.y -= self.gravity * dt;

        // integrate
        st.pos += st.vel * dt;

        // ground resolve
        let feet_y = self.feet_y(st.pos);
        if let Some(ground_y) = q.ground_height_m(st.pos.x, st.pos.z) {
            // treat ground as plane at ground_y
            let min_feet = ground_y;

            if feet_y < min_feet {
                // snap up and kill downward velocity
                let mut p = st.pos;
                self.set_feet_y(&mut p, min_feet);
                st.pos = p;

                if st.vel.y < 0.0 {
                    st.vel.y = 0.0;
                }
                st.on_ground = true;
            }
        }
    }
}
