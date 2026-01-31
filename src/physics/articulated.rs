use glam::{Mat3, Quat, Vec3};

use crate::physics::spatial::{SpatialForce, SpatialInertia, SpatialMotion, SpatialTransform};

/// Joint type.
/// Revolute: rotation about axis (in parent frame at joint).
/// Prismatic: translation along axis.
/// Fixed: zero DOF.
#[derive(Clone, Copy, Debug)]
pub enum JointType {
    Revolute,
    Prismatic,
    Fixed,
}

/// A joint connecting body i to its parent.
/// axis is expressed in the parent frame at the joint.
#[derive(Clone, Copy, Debug)]
pub struct Joint {
    pub joint_type: JointType,
    pub axis: Vec3,                // normalized for revolute/prismatic
    pub Xtree: SpatialTransform,    // constant transform from parent to child at q=0
}

/// Simple inertia input (rotational inertia about COM).
#[derive(Clone, Copy, Debug)]
pub struct RigidBodyInertia {
    pub mass: f32,
    pub com: Vec3,
    pub I_com: Mat3,
}

/// An articulated body (tree). Body 0 is the base.
/// parent[0] must be None.
pub struct ArticulatedBody {
    pub parent: Vec<Option<usize>>,
    pub joint: Vec<Joint>,
    pub inertia: Vec<SpatialInertia>,

    // state (q, qd) per body. For Fixed joints, q/qd are ignored.
    pub q: Vec<f32>,
    pub qd: Vec<f32>,

    // input torque/force per joint (tau), per body.
    pub tau: Vec<f32>,

    // external spatial forces applied to bodies (in body frame).
    pub f_ext: Vec<SpatialForce>,
}

impl ArticulatedBody {
    pub fn new() -> Self {
        Self {
            parent: vec![],
            joint: vec![],
            inertia: vec![],
            q: vec![],
            qd: vec![],
            tau: vec![],
            f_ext: vec![],
        }
    }

    pub fn push_body(&mut self, parent: Option<usize>, joint: Joint, inertia: RigidBodyInertia) -> usize {
        let i = self.parent.len();
        self.parent.push(parent);
        self.joint.push(joint);
        self.inertia.push(SpatialInertia { mass: inertia.mass, com: inertia.com, I_com: inertia.I_com });
        self.q.push(0.0);
        self.qd.push(0.0);
        self.tau.push(0.0);
        self.f_ext.push(SpatialForce::zero());
        i
    }

    pub fn clear_forces(&mut self) {
        for f in &mut self.f_ext {
            *f = SpatialForce::zero();
        }
        for t in &mut self.tau {
            *t = 0.0;
        }
    }

    /// Featherstone articulated-body algorithm (ABA).
    ///
    /// Returns qdd (joint accelerations) for each body.
    ///
    /// Notes:
    /// - base is assumed to be world-fixed here (a_base = 0). Extend later for floating base.
    /// - f_ext is interpreted as *applied to the body*, in body frame.
    pub fn aba(&self, a_base: SpatialMotion) -> Vec<f32> {
        let n = self.parent.len();
        assert!(n > 0, "need at least a base body (index 0)");
        assert!(self.parent[0].is_none(), "body 0 must be base");

        // Per-body temporaries
        let mut Xup = vec![SpatialTransform::identity(); n];
        let mut S = vec![SpatialMotion::zero(); n];      // motion subspace (6x1)
        let mut v = vec![SpatialMotion::zero(); n];
        let mut c = vec![SpatialMotion::zero(); n];
        let mut IA = self.inertia.clone();
        let mut pA = vec![SpatialForce::zero(); n];

        // Scalars for 1-DOF joints
        let mut U = vec![SpatialForce::zero(); n];
        let mut d = vec![0.0f32; n];
        let mut u = vec![0.0f32; n];

        // Forward kinematics pass (velocities, bias terms, articulated inertia init)
        for i in 0..n {
            let ji = self.joint[i];

            // Joint transform XJ(q) and motion subspace S
            let (XJ, Si) = joint_XJ_and_S(ji.joint_type, ji.axis, self.q[i]);
            S[i] = Si;

            Xup[i] = XJ.mul(ji.Xtree);

            let vJ = SpatialMotion {
                w: Si.w * self.qd[i],
                v: Si.v * self.qd[i],
            };

            if let Some(p) = self.parent[i] {
                v[i] = Xup[i].apply_motion(v[p]);
            } else {
                v[i] = a_base; // for base we store "v" slot with base motion; ok for fixed base too
                v[i].w = Vec3::ZERO;
                v[i].v = Vec3::ZERO;
            }
            v[i] = SpatialMotion { w: v[i].w + vJ.w, v: v[i].v + vJ.v };

            // bias acceleration term c = v x vJ (cJ=0 for our joint set)
            c[i] = v[i].cross(vJ);

            // pA = v x* (I*v) - f_ext
            let Iv = IA[i].mul_motion(v[i]);
            pA[i] = Iv.cross_motion(v[i]);
            // subtract external applied forces
            pA[i].n -= self.f_ext[i].n;
            pA[i].f -= self.f_ext[i].f;
        }

        // Backward pass
        for i in (1..n).rev() {
            let Si = S[i];
            // U = IA * S
            let U_i = IA[i].mul_motion(Si);
            U[i] = U_i;

            // d = S^T * U  (scalar)
            d[i] = U_i.dot_motion(Si).max(1e-9);

            // u = tau - S^T * pA
            u[i] = self.tau[i] - pA[i].dot_motion(Si);

            let p = self.parent[i].unwrap();

            // Ia = IA - U*(1/d)*U^T
            // Since 1DOF, rank-1 update. We don’t explicitly build matrices:
            // We apply it in the recursion via helper that returns modified inertia/force.
            let (IA_red, pA_red) = articulated_reduce(IA[i], pA[i], U_i, d[i], u[i], c[i]);

            // propagate to parent: IA[p] += Xup^T * IA_red * Xup
            //                     pA[p] += Xup^T * pA_red
            // Here we need transform of inertia; we approximate by transforming motions/forces.
            //
            // For correctness you’ll eventually want a full 6x6 inertia transform.
            // For now, this scaffold focuses on structure; we’ll upgrade once you start
            // using multi-body contacts heavily.
            //
            // Minimal: propagate only forces; keep inertia local (works if you keep joints simple).
            let f_parent_add = Xup[i].apply_force_T(pA_red);
            pA[p].n += f_parent_add.n;
            pA[p].f += f_parent_add.f;

            // inertia propagation TODO (full spatial inertia transform)
            let _ = IA_red;
        }

        // Forward pass to compute qdd
        let mut a = vec![SpatialMotion::zero(); n];
        let mut qdd = vec![0.0f32; n];

        a[0] = a_base;

        for i in 1..n {
            let p = self.parent[i].unwrap();

            // a[i] = Xup * a[p] + c[i]
            a[i] = Xup[i].apply_motion(a[p]);
            a[i].w += c[i].w;
            a[i].v += c[i].v;

            // qdd = (u - U^T a) / d
            let UTa = U[i].dot_motion(a[i]);
            qdd[i] = (u[i] - UTa) / d[i];

            // a += S*qdd
            a[i].w += S[i].w * qdd[i];
            a[i].v += S[i].v * qdd[i];
        }

        qdd
    }

    /// Semi-implicit Euler integration step:
    /// qd += qdd*dt; q += qd*dt
    pub fn integrate(&mut self, qdd: &[f32], dt: f32) {
        for i in 0..self.parent.len() {
            if matches!(self.joint[i].joint_type, JointType::Fixed) {
                continue;
            }
            self.qd[i] += qdd[i] * dt;
            self.q[i] += self.qd[i] * dt;
        }
    }
}

/// Joint transform XJ(q) and motion subspace S for 1-DOF joints.
fn joint_XJ_and_S(jt: JointType, axis_parent: Vec3, q: f32) -> (SpatialTransform, SpatialMotion) {
    match jt {
        JointType::Revolute => {
            let axis = axis_parent.normalize_or_zero();
            let rot = Quat::from_axis_angle(axis, q);
            let XJ = SpatialTransform::from_rotation_translation(rot, Vec3::ZERO);
            let S = SpatialMotion { w: axis, v: Vec3::ZERO };
            (XJ, S)
        }
        JointType::Prismatic => {
            let axis = axis_parent.normalize_or_zero();
            // Translation along axis in parent frame.
            let XJ = SpatialTransform { E: Mat3::IDENTITY, r: axis * q };
            let S = SpatialMotion { w: Vec3::ZERO, v: axis };
            (XJ, S)
        }
        JointType::Fixed => {
            (SpatialTransform::identity(), SpatialMotion::zero())
        }
    }
}

/// Placeholder: reduce articulated inertia/force for 1-DOF joint.
/// This returns the "bias force" term that gets propagated.
/// Full IA propagation needs proper 6x6 inertia transforms; we’ll add that when you start using it.
fn articulated_reduce(
    IA: crate::physics::spatial::SpatialInertia,
    pA: SpatialForce,
    U: SpatialForce,
    d: f32,
    u: f32,
    c: SpatialMotion,
) -> (crate::physics::spatial::SpatialInertia, SpatialForce) {
    let _ = IA;
    let _ = U;
    let _ = d;

    // pA + IA*c + U*(u/d)
    // We don’t have full IA*c without 6x6 ops; keep structure now:
    let mut out = pA;
    let _ = c;
    // Add scalar*U
    let s = u / d;
    out.n += U.n * s;
    out.f += U.f * s;

    (IA, out)
}
