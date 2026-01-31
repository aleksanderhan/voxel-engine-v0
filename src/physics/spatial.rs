use glam::{Mat3, Quat, Vec3};

/// A 6D spatial motion vector: angular (w) and linear (v).
#[derive(Clone, Copy, Debug, Default)]
pub struct SpatialMotion {
    pub w: Vec3,
    pub v: Vec3,
}

/// A 6D spatial force vector: torque (n) and force (f).
#[derive(Clone, Copy, Debug, Default)]
pub struct SpatialForce {
    pub n: Vec3,
    pub f: Vec3,
}

impl SpatialMotion {
    #[inline] pub fn zero() -> Self { Self { w: Vec3::ZERO, v: Vec3::ZERO } }

    #[inline]
    pub fn cross(self, b: SpatialMotion) -> SpatialMotion {
        // motion cross motion
        SpatialMotion {
            w: self.w.cross(b.w),
            v: self.w.cross(b.v) + self.v.cross(b.w),
        }
    }
}

impl SpatialForce {
    #[inline] pub fn zero() -> Self { Self { n: Vec3::ZERO, f: Vec3::ZERO } }

    #[inline]
    pub fn dot_motion(self, a: SpatialMotion) -> f32 {
        // power = n·w + f·v
        self.n.dot(a.w) + self.f.dot(a.v)
    }

    #[inline]
    pub fn cross_motion(self, v: SpatialMotion) -> SpatialForce {
        // force cross motion: v x* f
        SpatialForce {
            n: v.w.cross(self.n) + v.v.cross(self.f),
            f: v.w.cross(self.f),
        }
    }
}

/// Spatial transform X that maps motion vectors from parent frame to child frame.
/// Featherstone motion transform: v_child = X * v_parent
///
/// X(E, r): E rotation, r translation from parent origin to child origin expressed in parent.
#[derive(Clone, Copy, Debug)]
pub struct SpatialTransform {
    pub E: Mat3,
    pub r: Vec3,
}

impl SpatialTransform {
    #[inline]
    pub fn identity() -> Self {
        Self { E: Mat3::IDENTITY, r: Vec3::ZERO }
    }

    #[inline]
    pub fn from_rotation_translation(q: Quat, r_parent_to_child_in_parent: Vec3) -> Self {
        Self { E: Mat3::from_quat(q), r: r_parent_to_child_in_parent }
    }

    #[inline]
    pub fn inv(&self) -> Self {
        // Inverse: E^T, r' = -E^T * r
        let Et = self.E.transpose();
        Self { E: Et, r: -(Et * self.r) }
    }

    /// Compose: X_ab * X_bc = X_ac
    #[inline]
    pub fn mul(self, b: Self) -> Self {
        // E = Ea * Eb
        // r = r_a + Ea * r_b
        Self {
            E: self.E * b.E,
            r: self.r + self.E * b.r,
        }
    }

    #[inline]
    pub fn apply_motion(&self, v: SpatialMotion) -> SpatialMotion {
        // w' = E*w
        // v' = E*(v + w×r)
        let w = self.E * v.w;
        let vlin = self.E * (v.v + v.w.cross(self.r));
        SpatialMotion { w, v: vlin }
    }

    #[inline]
    pub fn apply_force(&self, f: SpatialForce) -> SpatialForce {
        // force transform (dual): f' = X^{-T} f
        // n' = E*(n + r×f)
        // f' = E*f
        let n = self.E * (f.n + self.r.cross(f.f));
        let ff = self.E * f.f;
        SpatialForce { n, f: ff }
    }

    /// Apply transpose mapping for forces when accumulating to parent:
    /// f_parent += X^T * f_child   (in Featherstone recursions)
    ///
    /// Implemented via the inverse force transform:
    /// X^T on force is equivalent to inv(X) applied as a force transform.
    #[inline]
    pub fn apply_force_T(&self, f_child: SpatialForce) -> SpatialForce {
        self.inv().apply_force(f_child)
    }
}

/// Spatial inertia for a rigid body.
/// Stored as mass, center of mass (com) in body frame, and rotational inertia about COM in body frame.
#[derive(Clone, Copy, Debug)]
pub struct SpatialInertia {
    pub mass: f32,
    pub com: Vec3,
    pub I_com: Mat3,
}

impl SpatialInertia {
    #[inline]
    pub fn mul_motion(&self, v: SpatialMotion) -> SpatialForce {
        // Convert to spatial force: f = I * v
        // Using standard spatial inertia block form:
        // [ I + m*rcx*rcx^T   m*rcx ]
        // [ m*rcx^T           m*I3 ]
        // where rcx is skew(com).
        let m = self.mass;
        let c = self.com;

        let wc = v.w;
        let vc = v.v;

        let c_x_v = c.cross(vc);
        let c_x_w = c.cross(wc);

        // torque: (Icom * w) + c×(m*v) + c×(c×(m*w))
        let n = self.I_com * wc + c.cross(m * vc) + c.cross(c.cross(m * wc));
        // force: m*v + (m*w)×c
        let f = m * vc + (m * wc).cross(c);

        SpatialForce { n, f }
    }
}
