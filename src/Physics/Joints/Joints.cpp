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
#include <limits>

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

            Vec2 RotateBy(Real a, Vec2 v) noexcept
            {
                const Real c = std::cos(a), s = std::sin(a);
                return Vec2(c * v.x - s * v.y, s * v.x + c * v.y);
            }
        } // namespace

        // =================================================================
        // DistanceJoint -- PORT of Joints.lua Distance (lines 44-67), plus
        // Box2D v3 b2DistanceJoint (distance_joint.c).
        // =================================================================

        void DistanceJoint::SetLength(Real length) noexcept
        {
            m_length = std::clamp(length, kLinearSlop, kDistanceJointHuge);
            m_impulse = m_lowerImpulse = m_upperImpulse = Real(0);
        }

        void DistanceJoint::SetLengthRange(Real minLength, Real maxLength) noexcept
        {
            minLength = std::clamp(minLength, kLinearSlop, kDistanceJointHuge);
            maxLength = std::clamp(maxLength, kLinearSlop, kDistanceJointHuge);
            m_minLength = std::min(minLength, maxLength);
            m_maxLength = std::max(minLength, maxLength);
            m_impulse = m_lowerImpulse = m_upperImpulse = Real(0);
        }

        Real DistanceJoint::GetCurrentLength(const PhysicsWorld& w) const noexcept
        {
            // Position() reads (0, 0) for an invalid handle, as JointMath::Pos does
            const Real angA = w.IsValid(m_hA) ? w.GetAngle(m_hA) : Real(0);
            const Real angB = w.IsValid(m_hB) ? w.GetAngle(m_hB) : Real(0);
            const Vec2 pa = w.Position(m_hA), pb = w.Position(m_hB);
            const Vec2 ra = RotateBy(angA, m_localA), rb = RotateBy(angB, m_localB);
            const Real dx = (pb.x + rb.x) - (pa.x + ra.x);
            const Real dy = (pb.y + rb.y) - (pa.y + ra.y);
            return std::sqrt(dx * dx + dy * dy);
        }

        void DistanceJoint::Prepare(PhysicsWorld& w, Real dt)
        {
            m_ia = w.IsValid(m_hA) ? m_hA.index : kInvalidSlot;
            m_ib = w.IsValid(m_hB) ? m_hB.index : kInvalidSlot;
            m_substepping = false;
            m_pass = 0;

            if (Legacy())
            {
                // The Lua port, unchanged (see the class comment).
                m_impulse = m_lowerImpulse = m_upperImpulse = Real(0);
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
                return;
            }

            // b2PrepareDistanceJoint: the start-of-step arms from each centre of
            // mass (Box2D: rotate(q, localOriginAnchor - localCenter)), the centre
            // offset, and the axial mass along the start-of-step axis.
            m_h = dt;
            const Real angA = JointMath::Angle(w, m_ia), angB = JointMath::Angle(w, m_ib);
            const Vec2 lcA = JointMath::Valid(m_ia) ? w.LocalCenterSlot(m_ia) : Vec2(Real(0), Real(0));
            const Vec2 lcB = JointMath::Valid(m_ib) ? w.LocalCenterSlot(m_ib) : Vec2(Real(0), Real(0));
            m_rA0 = RotateBy(angA, Vec2(m_localA.x - lcA.x, m_localA.y - lcA.y));
            m_rB0 = RotateBy(angB, Vec2(m_localB.x - lcB.x, m_localB.y - lcB.y));
            const Vec2 cA0 = RotateBy(angA, lcA), cB0 = RotateBy(angB, lcB);
            const Vec2 pa = JointMath::Pos(w, m_ia), pb = JointMath::Pos(w, m_ib);
            m_dc0 = Vec2((pb.x + cB0.x) - (pa.x + cA0.x), (pb.y + cB0.y) - (pa.y + cA0.y));
            m_dpA = m_dpB = Vec2(Real(0), Real(0));
            m_daA = m_daB = Real(0);

            const Vec2 sep(m_dc0.x + m_rB0.x - m_rA0.x, m_dc0.y + m_rB0.y - m_rA0.y);
            const Real len = std::sqrt(sep.x * sep.x + sep.y * sep.y);
            const Vec2 axis = len > std::numeric_limits<Real>::epsilon() ? Vec2(sep.x / len, sep.y / len) : Vec2(Real(0), Real(0));
            const Real crA = m_rA0.x * axis.y - m_rA0.y * axis.x;
            const Real crB = m_rB0.x * axis.y - m_rB0.y * axis.x;
            const Real k = JointMath::InvMass(w, m_ia) + JointMath::InvMass(w, m_ib)
                         + JointMath::InvInertia(w, m_ia) * crA * crA + JointMath::InvInertia(w, m_ib) * crB * crB;
            m_axialMass = k > Real(0) ? Real(1) / k : Real(0);

            const SoftCoeffs spring = MakeSoft(m_hertz, m_dampingRatio, dt);
            m_springBiasRate = spring.biasRate; m_springMassScale = spring.massScale; m_springImpulseScale = spring.impulseScale;
            const Real constraintHertz = dt > Real(0) ? std::min(kJointConstraintHertz, Real(0.25) / dt) : Real(0);
            const SoftCoeffs soft = MakeSoft(constraintHertz, kJointConstraintDampingRatio, dt);
            m_softBiasRate = soft.biasRate; m_softMassScale = soft.massScale; m_softImpulseScale = soft.impulseScale;

            // The accumulated impulses warm start this step, as Box2D's do -- but
            // only those of the parts that will be solved: Box2D's warm start
            // re-applies all three whatever is enabled, so a part switched off
            // would keep pushing with its last impulse.
            const bool springMode = m_enableSpring && (m_minLength < m_maxLength || !m_enableLimit);
            if (springMode && m_hertz <= Real(0)) { m_impulse = Real(0); }
            if (!springMode || !m_enableLimit) { m_lowerImpulse = m_upperImpulse = Real(0); }
        }

        void DistanceJoint::WarmStart(PhysicsWorld& w)
        {
            // b2WarmStartDistanceJoint: the accumulated axial impulse along the
            // current axis at the current arms, re-applied in full at the start of
            // the sub-step (delivered again, so it is reaction too)
            const Real j = m_impulse + m_lowerImpulse - m_upperImpulse;
            if (j == Real(0)) { return; }
            const Vec2 rA = RotateBy(m_daA, m_rA0), rB = RotateBy(m_daB, m_rB0);
            const Vec2 sep(m_dc0.x + (m_dpB.x - m_dpA.x) + (rB.x - rA.x),
                           m_dc0.y + (m_dpB.y - m_dpA.y) + (rB.y - rA.y));
            const Real len = std::sqrt(sep.x * sep.x + sep.y * sep.y);
            if (!(len > std::numeric_limits<Real>::epsilon())) { return; }
            const Vec2 P(sep.x / len * j, sep.y / len * j);
            JointMath::ApplyAt(w, m_ia, -P.x, -P.y, rA.x, rA.y);
            JointMath::ApplyAt(w, m_ib, P.x, P.y, rB.x, rB.y);
            AddReaction(P.x, P.y);
        }

        void DistanceJoint::SolveSoft(PhysicsWorld& w, bool useBias)
        {
            // b2SolveDistanceJoint: the current arms and separation (start offset
            // + in-flight motion + arms)
            const Vec2 rA = RotateBy(m_daA, m_rA0), rB = RotateBy(m_daB, m_rB0);
            const Vec2 sep(m_dc0.x + (m_dpB.x - m_dpA.x) + (rB.x - rA.x),
                           m_dc0.y + (m_dpB.y - m_dpA.y) + (rB.y - rA.y));
            const Real length = std::sqrt(sep.x * sep.x + sep.y * sep.y);
            const Vec2 axis = length > std::numeric_limits<Real>::epsilon()
                            ? Vec2(sep.x / length, sep.y / length) : Vec2(Real(0), Real(0));
            const auto cdot = [&]()
            {
                const Vec2 va = JointMath::Vat(w, m_ia, rA.x, rA.y);
                const Vec2 vb = JointMath::Vat(w, m_ib, rB.x, rB.y);
                return (vb.x - va.x) * axis.x + (vb.y - va.y) * axis.y;
            };
            const auto apply = [&](Real impulse)
            {
                const Vec2 P(axis.x * impulse, axis.y * impulse);
                JointMath::ApplyAt(w, m_ia, -P.x, -P.y, rA.x, rA.y);
                JointMath::ApplyAt(w, m_ib, P.x, P.y, rB.x, rB.y);
                AddReaction(P.x, P.y);
            };
            // Box2D's b2CreateDistanceJoint clamps the length to the linear slop
            const Real restLength = std::max(m_length, kLinearSlop);

            // soft if the spring is on and the limit is off or spans a range
            if (m_enableSpring && (m_minLength < m_maxLength || !m_enableLimit))
            {
                if (m_hertz > Real(0))
                {
                    const Real C = length - restLength;
                    const Real bias = m_springBiasRate * C;
                    const Real impulse = -m_springMassScale * m_axialMass * (cdot() + bias) - m_springImpulseScale * m_impulse;
                    m_impulse += impulse;
                    apply(impulse);
                }

                if (m_enableLimit)
                {
                    const Real invH = InvDt(m_h);
                    // lower: C = length - minLength >= 0
                    {
                        const Real C = length - m_minLength;
                        Real bias = Real(0), massScale = Real(1), impulseScale = Real(0);
                        if (C > Real(0)) { bias = C * invH; } // speculative
                        else if (useBias) { bias = m_softBiasRate * C; massScale = m_softMassScale; impulseScale = m_softImpulseScale; }
                        const Real old = m_lowerImpulse;
                        const Real impulse = -massScale * m_axialMass * (cdot() + bias) - impulseScale * old;
                        m_lowerImpulse = std::max(Real(0), old + impulse);
                        apply(m_lowerImpulse - old);
                    }
                    // upper: C = maxLength - length >= 0 (signs flipped to keep C and the impulse positive)
                    {
                        const Real C = m_maxLength - length;
                        Real bias = Real(0), massScale = Real(1), impulseScale = Real(0);
                        if (C > Real(0)) { bias = C * invH; }
                        else if (useBias) { bias = m_softBiasRate * C; massScale = m_softMassScale; impulseScale = m_softImpulseScale; }
                        const Real old = m_upperImpulse;
                        const Real impulse = -massScale * m_axialMass * (-cdot() + bias) - impulseScale * old;
                        m_upperImpulse = std::max(Real(0), old + impulse);
                        apply(-(m_upperImpulse - old));
                    }
                }
            }
            else
            {
                // rigid: soft at the joint softness on the biased pass, rigid and
                // bias-free on the relax pass
                const Real C = length - restLength;
                Real bias = Real(0), massScale = Real(1), impulseScale = Real(0);
                if (useBias) { bias = m_softBiasRate * C; massScale = m_softMassScale; impulseScale = m_softImpulseScale; }
                const Real impulse = -massScale * m_axialMass * (cdot() + bias) - impulseScale * m_impulse;
                m_impulse += impulse;
                apply(impulse);
            }
        }

        void DistanceJoint::SolveVelocity(PhysicsWorld& w)
        {
            if (Legacy())
            {
                SolveLegacy(w);
                return;
            }
            // Under SoftStep, pass #2 of a sub-step follows the position
            // integration: take in the motion it just integrated first (as the
            // revolute joint does). Pass #1 is the biased pass and warm starts.
            // Without sub-steps (no BeginSubstep) every pass is biased and the
            // impulses start from zero each step.
            bool useBias = true;
            if (m_substepping)
            {
                if (m_pass == 1)
                {
                    const Real wa = JointMath::AngVel(w, m_ia), wb = JointMath::AngVel(w, m_ib);
                    const Vec2 va = JointMath::Vat(w, m_ia, Real(0), Real(0)), vb = JointMath::Vat(w, m_ib, Real(0), Real(0));
                    m_daA += wa * m_h; m_daB += wb * m_h;
                    m_dpA = Vec2(m_dpA.x + va.x * m_h, m_dpA.y + va.y * m_h);
                    m_dpB = Vec2(m_dpB.x + vb.x * m_h, m_dpB.y + vb.y * m_h);
                }
                useBias = m_pass == 0;
                if (m_pass == 0) { WarmStart(w); }
                ++m_pass;
            }
            else if (m_pass++ == 0)
            {
                m_impulse = m_lowerImpulse = m_upperImpulse = Real(0);
            }
            SolveSoft(w, useBias);
        }

        void DistanceJoint::SolveLegacy(PhysicsWorld& w)
        {
            const Vec2 va = JointMath::Vat(w, m_ia, Real(0), Real(0));
            const Vec2 vb = JointMath::Vat(w, m_ib, Real(0), Real(0));
            const Real vn = (vb.x - va.x) * m_ux + (vb.y - va.y) * m_uy;
            const Real j = -(vn + m_bias) * m_mass;
            JointMath::ApplyAt(w, m_ia, -m_ux * j, -m_uy * j, Real(0), Real(0));
            JointMath::ApplyAt(w, m_ib, m_ux * j, m_uy * j, Real(0), Real(0));
            AddReaction(m_ux * j, m_uy * j);
        }

        // =================================================================
        // RevoluteJoint -- PORT of Joints.lua Revolute (lines 69-99).
        // =================================================================

        namespace
        {
            // b2UnwindAngle: into (-pi, pi]
            Real Unwind(Real a) noexcept
            {
                if (a < -kPi) { return a + Real(2) * kPi * std::ceil((-kPi - a) / (Real(2) * kPi)); }
                if (a > kPi) { return a - Real(2) * kPi * std::ceil((a - kPi) / (Real(2) * kPi)); }
                return a;
            }
        } // namespace

        Real RevoluteJoint::JointAngle(const PhysicsWorld& w) const noexcept
        {
            const Real a = w.IsValid(m_hA) ? w.GetAngle(m_hA) : Real(0);
            const Real b = w.IsValid(m_hB) ? w.GetAngle(m_hB) : Real(0);
            return Unwind(b - a - m_referenceAngle);
        }

        void RevoluteJoint::Prepare(PhysicsWorld& w, Real dt)
        {
            m_ia = w.IsValid(m_hA) ? m_hA.index : kInvalidSlot;
            m_ib = w.IsValid(m_hB) ? m_hB.index : kInvalidSlot;

            // the angular parts (b2PrepareRevoluteJoint + b2PrepareJoint)
            m_h = dt;
            const Real k = JointMath::InvInertia(w, m_ia) + JointMath::InvInertia(w, m_ib);
            m_axialMass = k > Real(0) ? Real(1) / k : Real(0);
            m_angle0 = Unwind(JointMath::Angle(w, m_ib) - JointMath::Angle(w, m_ia) - m_referenceAngle);
            m_deltaAngle = Real(0);
            m_substepping = false;
            m_pass = 0;
            const SoftCoeffs spring = MakeSoft(m_hertz, m_dampingRatio, dt);
            m_springBiasRate = spring.biasRate; m_springMassScale = spring.massScale; m_springImpulseScale = spring.impulseScale;
            const Real constraintHertz = dt > Real(0) ? std::min(kJointConstraintHertz, Real(0.25) / dt) : Real(0);
            const SoftCoeffs soft = MakeSoft(constraintHertz, kJointConstraintDampingRatio, dt);
            m_softBiasRate = soft.biasRate; m_softMassScale = soft.massScale; m_softImpulseScale = soft.impulseScale;
            m_springImpulse = m_motorImpulse = m_lowerImpulse = m_upperImpulse = Real(0);

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
            // the soft point constraint's start-of-step state (m_linearImpulse is kept:
            // it warm-starts the next step, as Box2D's does)
            m_rA0 = m_rA; m_rB0 = m_rB;
            m_dc0 = Vec2(pb.x - pa.x, pb.y - pa.y);
            m_dpA = m_dpB = Vec2(Real(0), Real(0));
            m_daA = m_daB = Real(0);
        }

        void RevoluteJoint::SolveVelocity(PhysicsWorld& w)
        {
            // pass #2 of a sub-step follows the position integration: take in the
            // rotation it just integrated before solving (see the class comment)
            bool useBias = true;
            if (m_substepping)
            {
                if (m_pass == 1)
                {
                    const Real wa = JointMath::AngVel(w, m_ia), wb = JointMath::AngVel(w, m_ib);
                    const Vec2 va = JointMath::Vat(w, m_ia, Real(0), Real(0)), vb = JointMath::Vat(w, m_ib, Real(0), Real(0));
                    m_deltaAngle += (wb - wa) * m_h;
                    m_daA += wa * m_h; m_daB += wb * m_h;
                    m_dpA = Vec2(m_dpA.x + va.x * m_h, m_dpA.y + va.y * m_h);
                    m_dpB = Vec2(m_dpB.x + vb.x * m_h, m_dpB.y + vb.y * m_h);
                }
                useBias = m_pass == 0;
                if (m_pass == 0) { WarmStartPoint(w); }
                ++m_pass;
            }
            SolveAngular(w, useBias);
            if (m_substepping) { SolvePointSoft(w, useBias); }
            else { SolvePoint(w); }
        }

        void RevoluteJoint::WarmStartPoint(PhysicsWorld& w)
        {
            const Vec2 rA = RotateBy(m_daA, m_rA0), rB = RotateBy(m_daB, m_rB0);
            JointMath::ApplyAt(w, m_ib, m_linearImpulse.x, m_linearImpulse.y, rB.x, rB.y);
            JointMath::ApplyAt(w, m_ia, -m_linearImpulse.x, -m_linearImpulse.y, rA.x, rA.y);
            AddReaction(m_linearImpulse.x, m_linearImpulse.y); // re-applied in full: delivered again this sub-step
        }

        void RevoluteJoint::SolvePointSoft(PhysicsWorld& w, bool useBias)
        {
            // b2SolveRevoluteJoint's point-to-point block: the current arms, the
            // current separation (start offset + in-flight motion + arms), soft on
            // the biased pass, rigid and bias-free on the relax pass
            const Vec2 rA = RotateBy(m_daA, m_rA0), rB = RotateBy(m_daB, m_rB0);
            const Vec2 va = JointMath::Vat(w, m_ia, rA.x, rA.y);
            const Vec2 vb = JointMath::Vat(w, m_ib, rB.x, rB.y);
            Vec2 bias(Real(0), Real(0));
            Real massScale = Real(1), impulseScale = Real(0);
            if (useBias)
            {
                const Vec2 sep(m_dc0.x + (m_dpB.x - m_dpA.x) + (rB.x - rA.x),
                               m_dc0.y + (m_dpB.y - m_dpA.y) + (rB.y - rA.y));
                bias = Vec2(m_softBiasRate * sep.x, m_softBiasRate * sep.y);
                massScale = m_softMassScale;
                impulseScale = m_softImpulseScale;
            }
            const Vec2 b = JointMath::SolvePoint(w, m_ia, m_ib, rA.x, rA.y, rB.x, rB.y,
                                                 vb.x - va.x + bias.x, vb.y - va.y + bias.y);
            const Vec2 impulse(-massScale * b.x - impulseScale * m_linearImpulse.x,
                               -massScale * b.y - impulseScale * m_linearImpulse.y);
            m_linearImpulse = Vec2(m_linearImpulse.x + impulse.x, m_linearImpulse.y + impulse.y);
            JointMath::ApplyAt(w, m_ib, impulse.x, impulse.y, rB.x, rB.y);
            JointMath::ApplyAt(w, m_ia, -impulse.x, -impulse.y, rA.x, rA.y);
            AddReaction(impulse.x, impulse.y);
        }

        void RevoluteJoint::SolveAngular(PhysicsWorld& w, bool useBias)
        {
            if (!(m_enableSpring || m_enableMotor || m_enableLimit) || m_axialMass <= Real(0))
            {
                return; // nothing to drive, or neither body can turn (Box2D: fixedRotation)
            }
            const Real jointAngle = m_angle0 + m_deltaAngle;
            const auto cdot = [&]() { return JointMath::AngVel(w, m_ib) - JointMath::AngVel(w, m_ia); };
            const auto apply = [&](Real impulse)
            {
                JointMath::ApplyAngular(w, m_ia, -impulse);
                JointMath::ApplyAngular(w, m_ib, impulse);
                AddReactionTorque(impulse);
            };

            if (m_enableSpring)
            {
                const Real C = Unwind(jointAngle - m_targetAngle);
                const Real bias = m_springBiasRate * C;
                const Real impulse = -m_springMassScale * m_axialMass * (cdot() + bias) - m_springImpulseScale * m_springImpulse;
                m_springImpulse += impulse;
                apply(impulse);
            }

            if (m_enableMotor)
            {
                const Real maxImpulse = m_h * m_maxMotorTorque;
                const Real impulse = -m_axialMass * (cdot() - m_motorSpeed);
                const Real old = m_motorImpulse;
                m_motorImpulse = std::clamp(old + impulse, -maxImpulse, maxImpulse);
                apply(m_motorImpulse - old);
            }

            if (m_enableLimit)
            {
                const Real invH = m_h > Real(0) ? Real(1) / m_h : Real(0);
                // lower: C = angle - lower >= 0
                {
                    const Real C = jointAngle - m_lower;
                    Real bias = Real(0), massScale = Real(1), impulseScale = Real(0);
                    if (C > Real(0)) { bias = C * invH; } // speculation
                    else if (useBias) { bias = m_softBiasRate * C; massScale = m_softMassScale; impulseScale = m_softImpulseScale; }
                    const Real old = m_lowerImpulse;
                    const Real impulse = -massScale * m_axialMass * (cdot() + bias) - impulseScale * old;
                    m_lowerImpulse = std::max(old + impulse, Real(0));
                    apply(m_lowerImpulse - old);
                }
                // upper: C = upper - angle >= 0 (signs flipped to keep C and the impulse positive)
                {
                    const Real C = m_upper - jointAngle;
                    Real bias = Real(0), massScale = Real(1), impulseScale = Real(0);
                    if (C > Real(0)) { bias = C * invH; }
                    else if (useBias) { bias = m_softBiasRate * C; massScale = m_softMassScale; impulseScale = m_softImpulseScale; }
                    const Real old = m_upperImpulse;
                    const Real impulse = -massScale * m_axialMass * (-cdot() + bias) - impulseScale * old;
                    m_upperImpulse = std::max(old + impulse, Real(0));
                    apply(-(m_upperImpulse - old));
                }
            }
        }

        void RevoluteJoint::SolvePoint(PhysicsWorld& w)
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
            AddReaction(j.x, j.y);
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
            AddReactionTorque(j);
        }

        // =================================================================
        // PrismaticJoint -- PORT of Joints.lua Prismatic (lines 120-151), plus
        // Box2D v3 b2PrismaticJoint's motor and translation limit.
        // =================================================================

        Real PrismaticJoint::GetTranslation(const PhysicsWorld& w) const noexcept
        {
            // Position() reads (0, 0) for an invalid handle, as JointMath::Pos does
            const Vec2 pa = w.Position(m_hA);
            const Vec2 pb = w.Position(m_hB);
            return (pb.x - pa.x - m_orig.x) * m_axis.x + (pb.y - pa.y - m_orig.y) * m_axis.y;
        }

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

            // The motor acts along the axis through the same origins as the
            // perpendicular constraint, so its effective mass is the same.
            m_maxMotorImpulse = m_maxMotorForce * dt;

            // The limit (b2PreparePrismaticJoint + b2PrepareJoint): the start-of-
            // step translation, tracked through the sub-steps, and the joint
            // constraint softness its push-out uses. Its axial mass is m_mass
            // too (no lever arms: the axis runs through the centres). The
            // accumulated impulses are kept -- they warm start this step, as
            // Box2D's do -- unless the limit is off.
            m_h = dt;
            m_translation0 = (pb.x - pa.x - m_orig.x) * m_axis.x + (pb.y - pa.y - m_orig.y) * m_axis.y;
            m_deltaTranslation = Real(0);
            m_substepping = false;
            m_pass = 0;
            const Real constraintHertz = dt > Real(0) ? std::min(kJointConstraintHertz, Real(0.25) / dt) : Real(0);
            const SoftCoeffs soft = MakeSoft(constraintHertz, kJointConstraintDampingRatio, dt);
            m_softBiasRate = soft.biasRate; m_softMassScale = soft.massScale; m_softImpulseScale = soft.impulseScale;
            if (!m_enableLimit) { m_lowerImpulse = m_upperImpulse = Real(0); }
        }

        void PrismaticJoint::WarmStartLimit(PhysicsWorld& w)
        {
            // b2WarmStartPrismaticJoint's limit share: the accumulated
            // lowerImpulse - upperImpulse along the axis, re-applied in full at
            // the start of the sub-step (delivered again, so it is reaction too)
            const Real j = m_lowerImpulse - m_upperImpulse;
            if (j == Real(0)) { return; }
            JointMath::ApplyAt(w, m_ia, -m_axis.x * j, -m_axis.y * j, Real(0), Real(0));
            JointMath::ApplyAt(w, m_ib, m_axis.x * j, m_axis.y * j, Real(0), Real(0));
            AddReaction(m_axis.x * j, m_axis.y * j);
        }

        void PrismaticJoint::SolveLimit(PhysicsWorld& w, bool useBias)
        {
            // b2SolvePrismaticJoint's limit block: two one-sided constraints on
            // the current translation, speculative while open (bias C / h), soft
            // when breached and only on the biased pass, each accumulated
            // impulse clamped >= 0
            const Real translation = m_translation0 + m_deltaTranslation;
            const Real invH = InvDt(m_h);
            const auto cdot = [&]()
            {
                const Vec2 va = JointMath::Vat(w, m_ia, Real(0), Real(0));
                const Vec2 vb = JointMath::Vat(w, m_ib, Real(0), Real(0));
                return (vb.x - va.x) * m_axis.x + (vb.y - va.y) * m_axis.y;
            };
            const auto apply = [&](Real impulse)
            {
                JointMath::ApplyAt(w, m_ia, -m_axis.x * impulse, -m_axis.y * impulse, Real(0), Real(0));
                JointMath::ApplyAt(w, m_ib, m_axis.x * impulse, m_axis.y * impulse, Real(0), Real(0));
                AddReaction(m_axis.x * impulse, m_axis.y * impulse);
            };
            // lower: C = translation - lower >= 0
            {
                const Real C = translation - m_lower;
                Real bias = Real(0), massScale = Real(1), impulseScale = Real(0);
                if (C > Real(0)) { bias = C * invH; } // speculation
                else if (useBias) { bias = m_softBiasRate * C; massScale = m_softMassScale; impulseScale = m_softImpulseScale; }
                const Real old = m_lowerImpulse;
                const Real impulse = -massScale * m_mass * (cdot() + bias) - impulseScale * old;
                m_lowerImpulse = std::max(old + impulse, Real(0));
                apply(m_lowerImpulse - old);
            }
            // upper: C = upper - translation >= 0 (signs flipped to keep C and the impulse positive)
            {
                const Real C = m_upper - translation;
                Real bias = Real(0), massScale = Real(1), impulseScale = Real(0);
                if (C > Real(0)) { bias = C * invH; }
                else if (useBias) { bias = m_softBiasRate * C; massScale = m_softMassScale; impulseScale = m_softImpulseScale; }
                const Real old = m_upperImpulse;
                const Real impulse = -massScale * m_mass * (-cdot() + bias) - impulseScale * old;
                m_upperImpulse = std::max(old + impulse, Real(0));
                apply(-(m_upperImpulse - old));
            }
        }

        void PrismaticJoint::SolveVelocity(PhysicsWorld& w)
        {
            // Under SoftStep, pass #2 of a sub-step follows the position
            // integration: take in the slide it just integrated first (as the
            // revolute joint does its angle). Pass #1 is the biased pass and
            // warm starts the limit. Without sub-steps (no BeginSubstep) the
            // limit is a plain sequential impulse over the step's iterations.
            bool useBias = true;
            if (m_substepping)
            {
                if (m_pass == 1)
                {
                    const Vec2 va = JointMath::Vat(w, m_ia, Real(0), Real(0));
                    const Vec2 vb = JointMath::Vat(w, m_ib, Real(0), Real(0));
                    m_deltaTranslation += ((vb.x - va.x) * m_axis.x + (vb.y - va.y) * m_axis.y) * m_h;
                }
                useBias = m_pass == 0;
                if (m_pass == 0 && m_enableLimit) { WarmStartLimit(w); }
                ++m_pass;
            }
            else if (m_pass++ == 0)
            {
                m_lowerImpulse = m_upperImpulse = Real(0);
            }

            // ---- motor (drive the slide; b2PrismaticJoint's motor) ----------
            // Solved first, like Box2D, so the constraints below have the last
            // word. The accumulated impulse is clamped to maxMotorForce * dt per
            // sub-step: against a load it cannot move, the motor stalls.
            if (m_enableMotor && m_mass > Real(0) && m_maxMotorImpulse > Real(0))
            {
                const Vec2 va = JointMath::Vat(w, m_ia, Real(0), Real(0));
                const Vec2 vb = JointMath::Vat(w, m_ib, Real(0), Real(0));
                const Real cdot = (vb.x - va.x) * m_axis.x + (vb.y - va.y) * m_axis.y - m_motorSpeed;
                Real impulse = -m_mass * cdot;
                const Real old = m_motorImpulse;
                m_motorImpulse = std::clamp(old + impulse, -m_maxMotorImpulse, m_maxMotorImpulse);
                impulse = m_motorImpulse - old;
                JointMath::ApplyAt(w, m_ia, -m_axis.x * impulse, -m_axis.y * impulse, Real(0), Real(0));
                JointMath::ApplyAt(w, m_ib, m_axis.x * impulse, m_axis.y * impulse, Real(0), Real(0));
                AddReaction(m_axis.x * impulse, m_axis.y * impulse);
            }

            // ---- limit (b2PrismaticJoint's lower/upper translation) ---------
            // After the motor, as in Box2D: a motor driving into the limit stops
            // there (the limit has the last word along the axis).
            if (m_enableLimit && m_mass > Real(0)) { SolveLimit(w, useBias); }

            const Vec2 va = JointMath::Vat(w, m_ia, Real(0), Real(0));
            const Vec2 vb = JointMath::Vat(w, m_ib, Real(0), Real(0));
            const Real vp = (vb.x - va.x) * m_px + (vb.y - va.y) * m_py;
            const Real j = -(vp + m_bias) * m_mass;
            JointMath::ApplyAt(w, m_ia, -m_px * j, -m_py * j, Real(0), Real(0));
            JointMath::ApplyAt(w, m_ib, m_px * j, m_py * j, Real(0), Real(0));
            AddReaction(m_px * j, m_py * j);

            const Real dw = JointMath::AngVel(w, m_ib) - JointMath::AngVel(w, m_ia) + m_angBias;
            const Real ja = -dw * m_angMass;
            JointMath::ApplyAngular(w, m_ia, -ja);
            JointMath::ApplyAngular(w, m_ib, ja);
            AddReactionTorque(ja);
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
            AddReaction(fx * m_dt, fy * m_dt);
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
                AddReactionTorque(impulse);
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
                AddReaction(P.x, P.y);
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
                AddReaction(P.x, P.y);
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
            AddReactionTorque(impulse);
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
                const Vec2 la = def.localAnchorA, lb = def.localAnchorB;
                const bool centres = la.x == Real(0) && la.y == Real(0) && lb.x == Real(0) && lb.y == Real(0);
                Real length = def.length;
                if (length <= Real(0))
                {
                    Vec2 pa = w.Position(def.a);
                    Vec2 pb = w.Position(def.b);
                    if (!centres)
                    {
                        // the current distance between the anchor points
                        const Vec2 ra = RotateBy(w.IsValid(def.a) ? w.GetAngle(def.a) : Real(0), la);
                        const Vec2 rb = RotateBy(w.IsValid(def.b) ? w.GetAngle(def.b) : Real(0), lb);
                        pa = Vec2(pa.x + ra.x, pa.y + ra.y);
                        pb = Vec2(pb.x + rb.x, pb.y + rb.y);
                    }
                    const Real dx = pb.x - pa.x;
                    const Real dy = pb.y - pa.y;
                    length = std::sqrt(dx * dx + dy * dy);
                }
                auto j = std::make_unique<DistanceJoint>(def.a, def.b, length, la, lb);
                j->EnableSpring(def.enableSpring);
                j->SetSpring(def.frequencyHz, def.dampingRatio);
                j->EnableLimit(def.enableLimit);
                j->SetLengthRange(def.minLength, def.maxLength);
                return j;
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
                auto j = std::make_unique<RevoluteJoint>(def.a, def.b, localA, localB);
                j->SetReferenceAngle(def.referenceAngle);
                j->EnableLimit(def.enableLimit);
                j->SetLimits(def.lowerAngle, def.upperAngle);
                j->EnableSpring(def.enableSpring);
                j->SetSpring(def.frequencyHz, def.dampingRatio);
                j->SetTargetAngle(def.targetAngle);
                j->EnableMotor(def.enableMotor);
                j->SetMotorSpeed(def.motorSpeed);
                j->SetMaxMotorTorque(def.maxMotorTorque);
                return j;
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
                auto j = std::make_unique<PrismaticJoint>(def.a, def.b, axis, orig, refAngle,
                                                          def.enableMotor, def.motorSpeed, def.maxMotorForce);
                j->EnableLimit(def.enableLimit);
                j->SetLimits(def.lowerTranslation, def.upperTranslation);
                return j;
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
