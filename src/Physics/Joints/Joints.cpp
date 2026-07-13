// Joints.cpp -- the concrete joint set + factory (M6, Task P2.5).
//
// See Joints.hpp / Joint.hpp for the contract + the PORT-vs-MODERNIZE framing.
// The five ported joints (Distance/Revolute/Weld/Prismatic/Mouse) are faithful
// ports of Client/src/physics/Joints.lua; Wheel + Motor are Box2D-derived
// additions (b2WheelJoint / b2MotorJoint-simplified) with no Lua oracle. All
// point-constraint math routes through JointMath.hpp (the ported vat/applyAt/
// solvePoint helpers).
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.

#include <Manifold2D/Physics/Joints/Joints.hpp>

#include <algorithm>
#include <cmath>

#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Joints/JointMath.hpp>
#include <Manifold2D/Physics/Solver/SoftCoeffs.hpp> // shared MakeSoft + SoftCoeffs

namespace Manifold2D
{
    namespace Physics
    {
        namespace
        {
            // 1/dt guarded against dt == 0 (the Lua BETA/dt; a zero dt -> 0 bias).
            Real InvDt(Real dt) noexcept
            {
                return dt > Real(0) ? Real(1) / dt : Real(0);
            }

            // Inverse-rotate a world anchor into body-local space (bodies usually unrotated at creation).
            Vec2 WorldToLocal(const PhysicsWorld& w, BodyHandle h, Vec2 worldPt) noexcept
            {
                const Vec2 p   = w.Position(h);
                const Real ang = w.IsValid(h) ? w.GetAngle(h) : Real(0);
                const Real dx  = worldPt.x - p.x;
                const Real dy  = worldPt.y - p.y;
                const Real c   = std::cos(-ang);
                const Real s   = std::sin(-ang);
                return Vec2(dx * c - dy * s, dx * s + dy * c);
            }
        } // namespace

        // =================================================================
        // DistanceJoint -- PORT of Joints.lua Distance (lines 44-67).
        // =================================================================

        void DistanceJoint::Prepare(PhysicsWorld& w, Real dt)
        {
            m_ia = w.IsValid(m_hA) ? m_hA.index : kInvalidSlot;
            m_ib = w.IsValid(m_hB) ? m_hB.index : kInvalidSlot;

            const Vec2 a = JointMath::Pos(w, m_ia);
            const Vec2 b = JointMath::Pos(w, m_ib);
            const Real dx = b.x - a.x;
            const Real dy = b.y - a.y;
            const Real d = std::sqrt(dx * dx + dy * dy);
            if (d > Real(1e-9))
            {
                m_ux = dx / d;
                m_uy = dy / d;
            }
            else
            {
                m_ux = Real(1);
                m_uy = Real(0);
            }
            m_bias = kJointBeta * InvDt(dt) * (d - m_length);

            const Real m = JointMath::InvMass(w, m_ia) + JointMath::InvMass(w, m_ib);
            m_mass = m > Real(0) ? Real(1) / m : Real(0);
        }

        void DistanceJoint::SolveVelocity(PhysicsWorld& w)
        {
            const Vec2 va = JointMath::Vat(w, m_ia, Real(0), Real(0));
            const Vec2 vb = JointMath::Vat(w, m_ib, Real(0), Real(0));
            const Real vn = (vb.x - va.x) * m_ux + (vb.y - va.y) * m_uy;
            const Real j = -(vn + m_bias) * m_mass;
            JointMath::ApplyAt(w, m_ia, -m_ux * j, -m_uy * j, Real(0), Real(0));
            JointMath::ApplyAt(w, m_ib, m_ux * j, m_uy * j, Real(0), Real(0));
        }

        // =================================================================
        // RevoluteJoint -- PORT of Joints.lua Revolute (lines 69-99).
        // =================================================================

        void RevoluteJoint::Prepare(PhysicsWorld& w, Real dt)
        {
            m_ia = w.IsValid(m_hA) ? m_hA.index : kInvalidSlot;
            m_ib = w.IsValid(m_hB) ? m_hB.index : kInvalidSlot;

            m_rA = JointMath::Rotated(w, m_ia, m_localA.x, m_localA.y);
            m_rB = JointMath::Rotated(w, m_ib, m_localB.x, m_localB.y);

            const Vec2 pa = JointMath::Pos(w, m_ia);
            const Vec2 pb = JointMath::Pos(w, m_ib);
            const Real pax = pa.x + m_rA.x;
            const Real pay = pa.y + m_rA.y;
            const Real pbx = pb.x + m_rB.x;
            const Real pby = pb.y + m_rB.y;
            const Real beta = kJointBeta * InvDt(dt);
            m_biasX = beta * (pbx - pax);
            m_biasY = beta * (pby - pay);
        }

        void RevoluteJoint::SolveVelocity(PhysicsWorld& w)
        {
            const Vec2 va = JointMath::Vat(w, m_ia, m_rA.x, m_rA.y);
            const Vec2 vb = JointMath::Vat(w, m_ib, m_rB.x, m_rB.y);
            const Real dvx = -(vb.x - va.x + m_biasX);
            const Real dvy = -(vb.y - va.y + m_biasY);
            const Vec2 j = JointMath::SolvePoint(w, m_ia, m_ib,
                                                 m_rA.x, m_rA.y, m_rB.x, m_rB.y,
                                                 dvx, dvy);
            JointMath::ApplyAt(w, m_ib, j.x, j.y, m_rB.x, m_rB.y);
            JointMath::ApplyAt(w, m_ia, -j.x, -j.y, m_rA.x, m_rA.y);
        }

        // =================================================================
        // WeldJoint -- PORT of Joints.lua Weld (lines 101-118).
        // =================================================================

        void WeldJoint::Prepare(PhysicsWorld& w, Real dt)
        {
            RevoluteJoint::Prepare(w, dt);
            const Real da = JointMath::Angle(w, m_ib) - JointMath::Angle(w, m_ia);
            m_angBias = kJointBeta * InvDt(dt) * (da - m_refAngle);
            const Real k = JointMath::InvInertia(w, m_ia) + JointMath::InvInertia(w, m_ib);
            m_angMass = k > Real(0) ? Real(1) / k : Real(0);
        }

        void WeldJoint::SolveVelocity(PhysicsWorld& w)
        {
            RevoluteJoint::SolveVelocity(w);
            const Real dw = JointMath::AngVel(w, m_ib) - JointMath::AngVel(w, m_ia) + m_angBias;
            const Real j = -dw * m_angMass;
            JointMath::ApplyAngular(w, m_ia, -j);
            JointMath::ApplyAngular(w, m_ib, j);
        }

        // =================================================================
        // PrismaticJoint -- PORT of Joints.lua Prismatic (lines 120-151).
        // =================================================================

        void PrismaticJoint::Prepare(PhysicsWorld& w, Real dt)
        {
            m_ia = w.IsValid(m_hA) ? m_hA.index : kInvalidSlot;
            m_ib = w.IsValid(m_hB) ? m_hB.index : kInvalidSlot;

            // Perpendicular of the (already normalized) axis: (-ay, ax).
            m_px = -m_axis.y;
            m_py = m_axis.x;

            const Vec2 pa = JointMath::Pos(w, m_ia);
            const Vec2 pb = JointMath::Pos(w, m_ib);
            const Real sep = (pb.x - pa.x - m_orig.x) * m_px
                           + (pb.y - pa.y - m_orig.y) * m_py;
            const Real beta = kJointBeta * InvDt(dt);
            m_bias = beta * sep;

            const Real m = JointMath::InvMass(w, m_ia) + JointMath::InvMass(w, m_ib);
            m_mass = m > Real(0) ? Real(1) / m : Real(0);

            const Real da = JointMath::Angle(w, m_ib) - JointMath::Angle(w, m_ia) - m_refAngle;
            m_angBias = beta * da;
            const Real k = JointMath::InvInertia(w, m_ia) + JointMath::InvInertia(w, m_ib);
            m_angMass = k > Real(0) ? Real(1) / k : Real(0);
        }

        void PrismaticJoint::SolveVelocity(PhysicsWorld& w)
        {
            const Vec2 va = JointMath::Vat(w, m_ia, Real(0), Real(0));
            const Vec2 vb = JointMath::Vat(w, m_ib, Real(0), Real(0));
            const Real vp = (vb.x - va.x) * m_px + (vb.y - va.y) * m_py;
            const Real j = -(vp + m_bias) * m_mass;
            JointMath::ApplyAt(w, m_ia, -m_px * j, -m_py * j, Real(0), Real(0));
            JointMath::ApplyAt(w, m_ib, m_px * j, m_py * j, Real(0), Real(0));

            const Real dw = JointMath::AngVel(w, m_ib) - JointMath::AngVel(w, m_ia) + m_angBias;
            const Real ja = -dw * m_angMass;
            JointMath::ApplyAngular(w, m_ia, -ja);
            JointMath::ApplyAngular(w, m_ib, ja);
        }

        // =================================================================
        // MouseJoint -- PORT of Joints.lua Mouse (lines 153-179).
        // =================================================================

        void MouseJoint::Prepare(PhysicsWorld& w, Real dt)
        {
            m_ib = w.IsValid(m_hB) ? m_hB.index : kInvalidSlot;
            const Real iM = JointMath::InvMass(w, m_ib);
            const Real m = iM > Real(0) ? Real(1) / iM : Real(0);
            const Real omega = Real(2) * kPi * kMouseFreq;
            m_k = m * omega * omega;
            m_d = Real(2) * m * kMouseZeta * omega;
            m_dt = dt;
        }

        void MouseJoint::SolveVelocity(PhysicsWorld& w)
        {
            if (!JointMath::Valid(m_ib))
            {
                return;
            }
            const Vec2 p = w.PosSlot(m_ib);
            const Vec2 v = w.VelSlot(m_ib);
            Real fx = m_k * (m_target.x - p.x) - m_d * v.x;
            Real fy = m_k * (m_target.y - p.y) - m_d * v.y;
            const Real f = std::sqrt(fx * fx + fy * fy);
            if (f > m_maxForce && f > Real(0))
            {
                fx = fx / f * m_maxForce;
                fy = fy / f * m_maxForce;
            }
            JointMath::ApplyAt(w, m_ib, fx * m_dt, fy * m_dt, Real(0), Real(0));
        }

        // =================================================================
        // WheelJoint (NEW; b2WheelJoint).
        //
        // Body B rides on A along a suspension axis through an anchor:
        //   * a PERPENDICULAR-to-axis point constraint, held rigidly (Baumgarte
        //     bias): B stays on the axis line through A's anchor.
        //   * a SOFT spring (b2MakeSoft from frequency/dampingRatio) along the
        //     axis: the suspension travel.
        //   * FREE rotation (no angular point constraint) + an optional rotation
        //     motor that drives B's spin relative to A toward motorSpeed.
        // Anchors are stored in local frames (rotated each Prepare); the axis is
        // stored in A's local frame and rotated into world each Prepare.
        // =================================================================

        void WheelJoint::Prepare(PhysicsWorld& w, Real dt)
        {
            m_ia = w.IsValid(m_hA) ? m_hA.index : kInvalidSlot;
            m_ib = w.IsValid(m_hB) ? m_hB.index : kInvalidSlot;

            m_rA = JointMath::Rotated(w, m_ia, m_localA.x, m_localA.y);
            m_rB = JointMath::Rotated(w, m_ib, m_localB.x, m_localB.y);

            // World axis (rotate the local-A axis by A's angle) + its perpendicular.
            m_axis = JointMath::Rotated(w, m_ia, m_localAxisA.x, m_localAxisA.y);
            const Real al = std::sqrt(m_axis.x * m_axis.x + m_axis.y * m_axis.y);
            if (al > Real(1e-9))
            {
                m_axis.x /= al;
                m_axis.y /= al;
            }
            m_perp = Vec2(-m_axis.y, m_axis.x);

            const Vec2 pa = JointMath::Pos(w, m_ia);
            const Vec2 pb = JointMath::Pos(w, m_ib);
            // d = (pb + rB) - (pa + rA): the separation vector between the anchors.
            const Real dx = (pb.x + m_rB.x) - (pa.x + m_rA.x);
            const Real dy = (pb.y + m_rB.y) - (pa.y + m_rA.y);

            const Real iMa = JointMath::InvMass(w, m_ia);
            const Real iIa = JointMath::InvInertia(w, m_ia);
            const Real iMb = JointMath::InvMass(w, m_ib);
            const Real iIb = JointMath::InvInertia(w, m_ib);

            // ---- perpendicular (rigid point) constraint ----------------------
            // Jacobian arms: sA = (rA + d) x perp, sB = rB x perp (b2WheelJoint).
            m_sAp = (m_rA.x + dx) * m_perp.y - (m_rA.y + dy) * m_perp.x;
            m_sBp = m_rB.x * m_perp.y - m_rB.y * m_perp.x;
            const Real kPerp = iMa + iMb + iIa * m_sAp * m_sAp + iIb * m_sBp * m_sBp;
            m_perpMass = kPerp > Real(0) ? Real(1) / kPerp : Real(0);
            const Real perpSep = dx * m_perp.x + dy * m_perp.y;
            m_perpBias = kJointBeta * InvDt(dt) * perpSep;

            // ---- axis (soft spring) constraint -------------------------------
            m_sAa = (m_rA.x + dx) * m_axis.y - (m_rA.y + dy) * m_axis.x;
            m_sBa = m_rB.x * m_axis.y - m_rB.y * m_axis.x;
            const Real kAxis = iMa + iMb + iIa * m_sAa * m_sAa + iIb * m_sBa * m_sBa;
            m_axMass = kAxis > Real(0) ? Real(1) / kAxis : Real(0);
            m_springSep = dx * m_axis.x + dy * m_axis.y; // current suspension offset
            const SoftCoeffs soft = MakeSoft(m_freqHz, m_damping, dt);
            m_springBiasRate     = soft.biasRate;
            m_springMassScale    = soft.massScale;
            m_springImpulseScale = soft.impulseScale;
            m_springImpulse = Real(0); // re-seeded each Prepare (no cross-step warm start)

            // ---- motor -------------------------------------------------------
            const Real kMotor = iIa + iIb;
            m_motorMass = kMotor > Real(0) ? Real(1) / kMotor : Real(0);
            m_motorImpulse = Real(0);
            m_maxMotorImpulse = m_maxMotorTorque * dt;
        }

        void WheelJoint::SolveVelocity(PhysicsWorld& w)
        {
            const Real iMa = JointMath::InvMass(w, m_ia);
            const Real iIa = JointMath::InvInertia(w, m_ia);
            const Real iMb = JointMath::InvMass(w, m_ib);
            const Real iIb = JointMath::InvInertia(w, m_ib);

            // Helper to read body B / A velocities + angular velocities.
            Vec2 vA = JointMath::Valid(m_ia) ? w.VelSlot(m_ia) : Vec2(Real(0), Real(0));
            Real wA = JointMath::AngVel(w, m_ia);
            Vec2 vB = JointMath::Valid(m_ib) ? w.VelSlot(m_ib) : Vec2(Real(0), Real(0));
            Real wB = JointMath::AngVel(w, m_ib);

            // ---- motor (drive the relative spin) -----------------------------
            if (m_enableMotor && m_motorMass > Real(0))
            {
                const Real cdot = wB - wA - m_motorSpeed;
                Real impulse = -m_motorMass * cdot;
                const Real old = m_motorImpulse;
                m_motorImpulse = std::clamp(old + impulse, -m_maxMotorImpulse, m_maxMotorImpulse);
                impulse = m_motorImpulse - old;
                wA -= iIa * impulse;
                wB += iIb * impulse;
            }

            // ---- axis spring (soft suspension) -------------------------------
            if (m_axMass > Real(0))
            {
                const Real cdot = m_axis.x * (vB.x - vA.x) + m_axis.y * (vB.y - vA.y)
                                + m_sBa * wB - m_sAa * wA;
                const Real bias = m_springBiasRate * m_springSep;
                Real impulse = -m_axMass * m_springMassScale * (cdot + bias)
                             - m_springImpulseScale * m_springImpulse;
                m_springImpulse += impulse;
                const Vec2 P(impulse * m_axis.x, impulse * m_axis.y);
                vA.x -= iMa * P.x;
                vA.y -= iMa * P.y;
                wA -= iIa * impulse * m_sAa;
                vB.x += iMb * P.x;
                vB.y += iMb * P.y;
                wB += iIb * impulse * m_sBa;
            }

            // ---- perpendicular (rigid) constraint ----------------------------
            if (m_perpMass > Real(0))
            {
                const Real cdot = m_perp.x * (vB.x - vA.x) + m_perp.y * (vB.y - vA.y)
                                + m_sBp * wB - m_sAp * wA;
                const Real impulse = -m_perpMass * (cdot + m_perpBias);
                const Vec2 P(impulse * m_perp.x, impulse * m_perp.y);
                vA.x -= iMa * P.x;
                vA.y -= iMa * P.y;
                wA -= iIa * impulse * m_sAp;
                vB.x += iMb * P.x;
                vB.y += iMb * P.y;
                wB += iIb * impulse * m_sBp;
            }

            // Write back (no-op on static / invalid through the guards).
            if (JointMath::Valid(m_ia) && iMa > Real(0))
            {
                w.SetVelSlot(m_ia, vA);
            }
            if (JointMath::Valid(m_ia))
            {
                w.SetAngVelSlot(m_ia, wA);
            }
            if (JointMath::Valid(m_ib) && iMb > Real(0))
            {
                w.SetVelSlot(m_ib, vB);
            }
            if (JointMath::Valid(m_ib))
            {
                w.SetAngVelSlot(m_ib, wB);
            }
        }

        // =================================================================
        // MotorJoint (NEW; b2MotorJoint-simplified): drive (angVelB - angVelA)
        // toward motorSpeed, clamped to +-maxMotorTorque*dt.
        // =================================================================

        void MotorJoint::Prepare(PhysicsWorld& w, Real dt)
        {
            m_ia = w.IsValid(m_hA) ? m_hA.index : kInvalidSlot;
            m_ib = w.IsValid(m_hB) ? m_hB.index : kInvalidSlot;
            const Real k = JointMath::InvInertia(w, m_ia) + JointMath::InvInertia(w, m_ib);
            m_mass = k > Real(0) ? Real(1) / k : Real(0);
            m_impulse = Real(0);
            m_maxImpulse = m_maxMotorTorque * dt;
        }

        void MotorJoint::SolveVelocity(PhysicsWorld& w)
        {
            if (m_mass <= Real(0))
            {
                return;
            }
            const Real cdot = JointMath::AngVel(w, m_ib) - JointMath::AngVel(w, m_ia)
                            - m_motorSpeed;
            Real impulse = -m_mass * cdot;
            const Real old = m_impulse;
            m_impulse = std::clamp(old + impulse, -m_maxImpulse, m_maxImpulse);
            impulse = m_impulse - old;
            JointMath::ApplyAngular(w, m_ia, -impulse);
            JointMath::ApplyAngular(w, m_ib, impulse);
        }

        // =================================================================
        // MakeJoint -- PORT of Joints.make (lines 188-228) + Wheel/Motor.
        // =================================================================

        std::unique_ptr<Joint> MakeJoint(const PhysicsWorld& w, const JointDef& def)
        {
            switch (def.kind)
            {
            case JointKind::Distance:
            {
                Real length = def.length;
                if (length <= Real(0))
                {
                    const Vec2 pa = w.Position(def.a);
                    const Vec2 pb = w.Position(def.b);
                    const Real dx = pb.x - pa.x;
                    const Real dy = pb.y - pa.y;
                    length = std::sqrt(dx * dx + dy * dy);
                }
                return std::make_unique<DistanceJoint>(def.a, def.b, length);
            }
            case JointKind::Revolute:
            case JointKind::Weld:
            {
                // Inverse-rotate the world anchor into each body's local frame
                // (ports the Lua toLocal closure). For unrotated bodies (the
                // common creation case) this is just anchor - pos.
                const Vec2 localA = WorldToLocal(w, def.a, def.anchor);
                const Vec2 localB = WorldToLocal(w, def.b, def.anchor);
                const Real refAngle =
                    (w.IsValid(def.b) ? w.GetAngle(def.b) : Real(0)) -
                    (w.IsValid(def.a) ? w.GetAngle(def.a) : Real(0));
                if (def.kind == JointKind::Weld)
                {
                    return std::make_unique<WeldJoint>(def.a, def.b, localA, localB, refAngle);
                }
                return std::make_unique<RevoluteJoint>(def.a, def.b, localA, localB);
            }
            case JointKind::Prismatic:
            {
                Vec2 axis = def.axis;
                const Real len = std::sqrt(axis.x * axis.x + axis.y * axis.y);
                if (len > Real(1e-9))
                {
                    axis.x /= len;
                    axis.y /= len;
                }
                const Vec2 pa = w.Position(def.a);
                const Vec2 pb = w.Position(def.b);
                const Vec2 orig(pb.x - pa.x, pb.y - pa.y);
                const Real refAngle =
                    (w.IsValid(def.b) ? w.GetAngle(def.b) : Real(0)) -
                    (w.IsValid(def.a) ? w.GetAngle(def.a) : Real(0));
                return std::make_unique<PrismaticJoint>(def.a, def.b, axis, orig, refAngle);
            }
            case JointKind::Mouse:
            {
                const Real maxForce = def.maxForce > Real(0) ? def.maxForce : Real(1e6); // "0 -> engine default" fallback mirrors JointDef::maxForce -- see Joint.hpp
                return std::make_unique<MouseJoint>(def.b, def.target, maxForce);
            }
            case JointKind::Wheel:
            {
                // Local anchors (world anchor inverse-rotated into each frame).
                const Vec2 localA = WorldToLocal(w, def.a, def.anchor);
                const Vec2 localB = WorldToLocal(w, def.b, def.anchor);
                // Suspension axis in A's local frame.
                Vec2 axis = def.axis;
                const Real len = std::sqrt(axis.x * axis.x + axis.y * axis.y);
                if (len > Real(1e-9))
                {
                    axis.x /= len;
                    axis.y /= len;
                }
                const Real angA = w.IsValid(def.a) ? w.GetAngle(def.a) : Real(0);
                const Real c = std::cos(-angA);
                const Real s = std::sin(-angA);
                const Vec2 localAxisA(axis.x * c - axis.y * s, axis.x * s + axis.y * c);
                return std::make_unique<WheelJoint>(
                    def.a, def.b, localA, localB, localAxisA,
                    def.frequencyHz, def.dampingRatio,
                    def.enableMotor, def.motorSpeed, def.maxMotorTorque);
            }
            case JointKind::Motor:
                return std::make_unique<MotorJoint>(def.a, def.b, def.motorSpeed,
                                                    def.maxMotorTorque);
            }
            return nullptr;
        }

    } // namespace Physics
} // namespace Manifold2D
