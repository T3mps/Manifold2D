#pragma once

// Joints: the concrete joint set + factory (M6, Task P2.5).
//
// PORT + MODERNIZE. Ports the five Joints.lua joints (Distance/Revolute/Weld/
// Prismatic/Mouse) and ADDS two Box2D-derived joints (Wheel/Motor). Each is a
// velocity constraint solved in the solver's iteration loop (see Joint.hpp for
// the lifecycle + the soft-constraint framing). The shared point-constraint
// math lives in JointMath.hpp (a faithful port of the Lua vat/applyAt/solvePoint
// helpers). All state is captured at Prepare into the joint object -> zero
// steady-state allocation in the solve loop.
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.
// namespace Manifold2D::Physics, Core style.

#include <algorithm>
#include <cstdint>
#include <memory>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Joints/Joint.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        class PhysicsWorld;

        // Box2D's B2_HUGE (100000 length units): an unbounded maxLength.
        inline constexpr Real kDistanceJointHuge = Real(100000);

        // ================================================================
        // DistanceJoint -- hold the distance between an anchor on A and an
        // anchor on B.
        // PORT: Joints.lua Distance (lines 44-67), plus Box2D v3
        // b2DistanceJoint (distance_joint.c b2SolveDistanceJoint):
        //   * anchors: localAnchorA / localAnchorB in each body's frame
        //     (relative to the body origin, as Box2D's localAnchor), so the
        //     constraint acts between the anchor points and torques both bodies;
        //   * rigid (the default): the length is a hard constraint, soft at the
        //     joint constraint softness (60 Hz clamped to 0.25 / h, damping
        //     ratio 2) on the biased pass, rigid on the relax pass, warm started;
        //   * spring (enableSpring): the length becomes a spring of (hertz,
        //     dampingRatio) about `length`; hertz 0 is NO length constraint;
        //   * limit (enableLimit, with the spring on): minLength <= length <=
        //     maxLength as two one-sided constraints, speculative while open
        //     (bias C / h), soft at the joint softness when breached and only on
        //     the biased pass, clamped >= 0, warm started. With the spring OFF,
        //     or minLength == maxLength, the joint is rigid and the limit is not
        //     solved (Box2D's rule);
        //   * rope: enableSpring with hertz 0 + enableLimit [0, maxLength] --
        //     holds only when taut, never pushes.
        // Box2D clamps length, minLength and maxLength to [kLinearSlop,
        // kDistanceJointHuge]; so does this joint (a rope's minLength 0 becomes
        // 5 mm). No motor.
        //
        // LEGACY PATH: a joint with both anchors at the body origins and neither
        // the spring nor the limit enabled -- what the original JointDef built --
        // runs the Lua port unchanged (a rigid centre-to-centre rod with a
        // Baumgarte bias computed once per step), bit for bit, so existing scenes
        // and the determinism goldens do not move. Everything else runs Box2D's
        // formulation, which measures the arms from each body's centre of mass.
        //
        // In-flight state through a step: as the revolute joint, this joint
        // integrates the bodies' motion through the sub-steps itself (pass #2 of
        // each sub-step follows the position integration).
        // ================================================================
        class DistanceJoint final : public Joint
        {
        public:
            DistanceJoint(BodyHandle a, BodyHandle b, Real length,
                          Vec2 localAnchorA = Vec2(Real(0), Real(0)),
                          Vec2 localAnchorB = Vec2(Real(0), Real(0)))
                : m_hA(a), m_hB(b), m_length(length), m_localA(localAnchorA), m_localB(localAnchorB)
            {
            }
            void Prepare(PhysicsWorld& w, Real dt) override;
            void SolveVelocity(PhysicsWorld& w) override;
            // The accumulated impulses are kept across sub-steps (warm start).
            void BeginSubstep() noexcept override
            {
                m_substepping = true;
                m_pass = 0;
            }
            [[nodiscard]] std::uint32_t BodyA() const noexcept override { return m_ia; }
            [[nodiscard]] std::uint32_t BodyB() const noexcept override { return m_ib; }
            [[nodiscard]] BodyHandle HandleA() const noexcept override { return m_hA; }
            [[nodiscard]] BodyHandle HandleB() const noexcept override { return m_hB; }

            // ---- Box2D b2DistanceJoint_* controls (take effect from the next
            // Step; a sleeping body is not woken here: the caller wakes it) ----
            // The rest length (rigid length, or the spring's rest length);
            // clamped to [kLinearSlop, kDistanceJointHuge]; clears the impulses.
            void SetLength(Real length) noexcept;
            [[nodiscard]] Real GetLength() const noexcept { return m_length; }
            void EnableSpring(bool on) noexcept { m_enableSpring = on; }
            void SetSpring(Real hertz, Real dampingRatio) noexcept { m_hertz = hertz; m_dampingRatio = dampingRatio; }
            [[nodiscard]] bool IsSpringEnabled() const noexcept { return m_enableSpring; }
            [[nodiscard]] Real GetSpringHertz() const noexcept { return m_hertz; }
            [[nodiscard]] Real GetSpringDampingRatio() const noexcept { return m_dampingRatio; }
            void EnableLimit(bool on) noexcept { m_enableLimit = on; }
            // Both clamped to [kLinearSlop, kDistanceJointHuge] and sorted; clears
            // the impulses (b2DistanceJoint_SetLengthRange).
            void SetLengthRange(Real minLength, Real maxLength) noexcept;
            [[nodiscard]] bool IsLimitEnabled() const noexcept { return m_enableLimit; }
            [[nodiscard]] Real GetMinLength() const noexcept { return m_minLength; }
            [[nodiscard]] Real GetMaxLength() const noexcept { return m_maxLength; }
            [[nodiscard]] Vec2 LocalAnchorA() const noexcept { return m_localA; }
            [[nodiscard]] Vec2 LocalAnchorB() const noexcept { return m_localB; }
            // The current anchor-to-anchor distance (b2DistanceJoint_GetCurrentLength).
            [[nodiscard]] Real GetCurrentLength(const PhysicsWorld& w) const noexcept;

        private:
            [[nodiscard]] bool Legacy() const noexcept
            {
                return !m_enableSpring && !m_enableLimit
                    && m_localA.x == Real(0) && m_localA.y == Real(0)
                    && m_localB.x == Real(0) && m_localB.y == Real(0);
            }
            void SolveLegacy(PhysicsWorld& w);
            void WarmStart(PhysicsWorld& w);
            void SolveSoft(PhysicsWorld& w, bool useBias);

            BodyHandle    m_hA, m_hB;
            std::uint32_t m_ia = kInvalidSlot, m_ib = kInvalidSlot;
            Real          m_length = Real(0);
            Vec2          m_localA{ Real(0), Real(0) }; // anchor in A's frame (from its origin)
            Vec2          m_localB{ Real(0), Real(0) }; // anchor in B's frame (from its origin)
            // Box2D b2DistanceJoint parameters.
            bool m_enableSpring = false;
            Real m_hertz = Real(0), m_dampingRatio = Real(0);
            bool m_enableLimit = false;
            Real m_minLength = kLinearSlop, m_maxLength = kDistanceJointHuge;
            // Prepared per step (legacy path).
            Real m_ux = Real(1), m_uy = Real(0); // unit A->B axis
            Real m_bias = Real(0);
            Real m_mass = Real(0);
            // Prepared per step / tracked per sub-step (Box2D path).
            Real m_h = Real(0);
            Real m_axialMass = Real(0);
            Real m_springBiasRate = Real(0), m_springMassScale = Real(1), m_springImpulseScale = Real(0);
            Real m_softBiasRate = Real(0), m_softMassScale = Real(1), m_softImpulseScale = Real(0);
            // start-of-step arms (from the centres of mass) and centre offset
            Vec2 m_rA0{ Real(0), Real(0) }, m_rB0{ Real(0), Real(0) }, m_dc0{ Real(0), Real(0) };
            Vec2 m_dpA{ Real(0), Real(0) }, m_dpB{ Real(0), Real(0) };
            Real m_daA = Real(0), m_daB = Real(0);
            // accumulated (warm-started across sub-steps and steps)
            Real m_impulse = Real(0), m_lowerImpulse = Real(0), m_upperImpulse = Real(0);
            bool m_substepping = false; // the solver calls BeginSubstep (SoftStep)
            int  m_pass = 0;            // joint pass within the sub-step
        };

        // ================================================================
        // RevoluteJoint -- pin A and B at a shared anchor (a point constraint).
        // PORT: Joints.lua Revolute (lines 69-99), plus Box2D v3 b2RevoluteJoint's
        // angular parts (revolute_joint.c b2SolveRevoluteJoint), solved before
        // the point constraint as there:
        //   * spring: drive the joint angle to targetAngle, soft at (hertz,
        //     dampingRatio) (b2MakeSoft);
        //   * motor: drive the relative angular velocity to motorSpeed, the
        //     impulse clamped to h * maxMotorTorque;
        //   * limit: two one-sided constraints, speculative while open (bias
        //     C / h), soft at the joint constraint softness when breached
        //     (b2PrepareJoint: 60 Hz clamped to 0.25 / h, damping ratio 2), the
        //     push-out only on the biased pass (useBias).
        //
        // The point constraint under SoftStep is Box2D's too: soft (the joint
        // constraint softness), solved against the CURRENT separation each
        // sub-step, its push-out on the biased pass only, and warm started from
        // its accumulated impulse. (The Lua port's Baumgarte bias was computed
        // once per step from the start-of-step separation and applied again in
        // every sub-step: a light chain under a heavy load stretched 3 cm a joint
        // and zig-zagged.) Without sub-steps (no BeginSubstep) the Lua point
        // constraint runs unchanged.
        //
        // In-flight state through a step: Box2D reads the bodies' deltaPosition
        // and deltaRotation; this engine's joints see velocities only, so the
        // joint integrates them itself. SoftStep runs joint pass #1 after the
        // warm start (the biased pass) and pass #2 right after integrating
        // positions (the relax pass): the velocities pass #2 sees are the ones
        // just integrated, so it adds v * h and w * h first.
        // ================================================================
        class RevoluteJoint : public Joint
        {
        public:
            RevoluteJoint(BodyHandle a, BodyHandle b, Vec2 localA, Vec2 localB)
                : m_hA(a), m_hB(b), m_localA(localA), m_localB(localB)
            {
            }
            void Prepare(PhysicsWorld& w, Real dt) override;
            void SolveVelocity(PhysicsWorld& w) override;
            void BeginSubstep() noexcept override
            {
                m_substepping = true;
                m_pass = 0;
                m_springImpulse = m_motorImpulse = m_lowerImpulse = m_upperImpulse = Real(0);
            }
            [[nodiscard]] std::uint32_t BodyA() const noexcept override { return m_ia; }
            [[nodiscard]] std::uint32_t BodyB() const noexcept override { return m_ib; }
            [[nodiscard]] BodyHandle HandleA() const noexcept override { return m_hA; }
            [[nodiscard]] BodyHandle HandleB() const noexcept override { return m_hB; }

            // ---- Box2D b2RevoluteJoint_* controls (take effect from the next Step;
            // a sleeping body is not woken here: the caller wakes it) ----
            // The current joint angle: angleB - angleA - referenceAngle, in (-pi, pi].
            [[nodiscard]] Real JointAngle(const PhysicsWorld& w) const noexcept;
            void SetReferenceAngle(Real a) noexcept { m_referenceAngle = a; }
            void EnableLimit(bool on) noexcept { m_enableLimit = on; }
            void SetLimits(Real lower, Real upper) noexcept { m_lower = std::min(lower, upper); m_upper = std::max(lower, upper); }
            void EnableSpring(bool on) noexcept { m_enableSpring = on; }
            void SetSpring(Real hertz, Real dampingRatio) noexcept { m_hertz = hertz; m_dampingRatio = dampingRatio; }
            void SetTargetAngle(Real a) noexcept { m_targetAngle = a; }
            void EnableMotor(bool on) noexcept { m_enableMotor = on; }
            void SetMotorSpeed(Real speed) noexcept { m_motorSpeed = speed; }
            void SetMaxMotorTorque(Real torque) noexcept { m_maxMotorTorque = torque > Real(0) ? torque : Real(0); }
            [[nodiscard]] Real ReferenceAngle() const noexcept { return m_referenceAngle; }
            [[nodiscard]] bool IsLimitEnabled() const noexcept { return m_enableLimit; }
            [[nodiscard]] Real LowerLimit() const noexcept { return m_lower; }
            [[nodiscard]] Real UpperLimit() const noexcept { return m_upper; }
            [[nodiscard]] bool IsSpringEnabled() const noexcept { return m_enableSpring; }
            [[nodiscard]] Real TargetAngle() const noexcept { return m_targetAngle; }
            [[nodiscard]] bool IsMotorEnabled() const noexcept { return m_enableMotor; }
            [[nodiscard]] Real MotorSpeed() const noexcept { return m_motorSpeed; }

        protected:
            // the angular parts (spring, motor, limit), then the point constraint: Box2D's order
            void SolveAngular(PhysicsWorld& w, bool useBias);
            void SolvePoint(PhysicsWorld& w);                     // the Lua port (no sub-steps)
            void SolvePointSoft(PhysicsWorld& w, bool useBias);   // Box2D's (under SoftStep)
            void WarmStartPoint(PhysicsWorld& w);

            BodyHandle    m_hA, m_hB;
            std::uint32_t m_ia = kInvalidSlot, m_ib = kInvalidSlot;
            Vec2          m_localA{ Real(0), Real(0) }; // anchor in A's local frame
            Vec2          m_localB{ Real(0), Real(0) }; // anchor in B's local frame
            // Prepared per step.
            Vec2 m_rA{ Real(0), Real(0) }; // world arm A
            Vec2 m_rB{ Real(0), Real(0) }; // world arm B
            Real m_biasX = Real(0), m_biasY = Real(0);

            // Box2D b2RevoluteJoint parameters.
            Real m_referenceAngle = Real(0);
            bool m_enableLimit = false;
            Real m_lower = Real(0), m_upper = Real(0);
            bool m_enableSpring = false;
            Real m_hertz = Real(0), m_dampingRatio = Real(0), m_targetAngle = Real(0);
            bool m_enableMotor = false;
            Real m_motorSpeed = Real(0), m_maxMotorTorque = Real(0);
            // Prepared per step / tracked per sub-step.
            Real m_h = Real(0);              // the (sub-)step the joint is prepared for
            Real m_axialMass = Real(0);      // 1 / (iA + iB)
            Real m_angle0 = Real(0);         // the joint angle at the start of the step
            Real m_deltaAngle = Real(0);     // its change, integrated through the sub-steps
            Real m_springBiasRate = Real(0), m_springMassScale = Real(1), m_springImpulseScale = Real(0);
            Real m_softBiasRate = Real(0), m_softMassScale = Real(1), m_softImpulseScale = Real(0);
            Real m_springImpulse = Real(0), m_motorImpulse = Real(0);
            Real m_lowerImpulse = Real(0), m_upperImpulse = Real(0);
            // the soft point constraint: start-of-step arms and origin offset, the
            // bodies' in-flight motion, and the accumulated (warm-started) impulse
            Vec2 m_rA0{ Real(0), Real(0) }, m_rB0{ Real(0), Real(0) }, m_dc0{ Real(0), Real(0) };
            Vec2 m_dpA{ Real(0), Real(0) }, m_dpB{ Real(0), Real(0) };
            Real m_daA = Real(0), m_daB = Real(0);
            Vec2 m_linearImpulse{ Real(0), Real(0) };
            bool m_substepping = false;      // the solver calls BeginSubstep (SoftStep)
            int  m_pass = 0;                 // joint pass within the sub-step
        };

        // ================================================================
        // WeldJoint -- revolute + a relative-angle lock (rigid join).
        // PORT: Joints.lua Weld (lines 101-118; derives from Revolute).
        // ================================================================
        class WeldJoint final : public RevoluteJoint
        {
        public:
            WeldJoint(BodyHandle a, BodyHandle b, Vec2 localA, Vec2 localB, Real refAngle)
                : RevoluteJoint(a, b, localA, localB), m_refAngle(refAngle)
            {
            }
            void Prepare(PhysicsWorld& w, Real dt) override;
            void SolveVelocity(PhysicsWorld& w) override;

        private:
            Real m_refAngle = Real(0); // locked relative angle (angleB - angleA)
            // Prepared per step.
            Real m_angBias = Real(0);
            Real m_angMass = Real(0);
        };

        // ================================================================
        // PrismaticJoint -- slide along a fixed world axis: no perpendicular
        // drift, no relative rotation.
        // PORT: Joints.lua Prismatic (lines 120-151), plus Box2D v3
        // b2PrismaticJoint's motor and translation limit (prismatic_joint.c
        // b2SolvePrismaticJoint), solved before the perpendicular and angular
        // constraints as there:
        //   * motor: drive B's speed along the axis toward motorSpeed with at
        //     most maxMotorForce (a fresh h * maxMotorForce budget each sub-step);
        //   * limit: lower <= translation <= upper, two one-sided constraints,
        //     each with its own accumulated impulse: speculative while open
        //     (bias C / h), soft at the joint constraint softness when breached
        //     (b2PrepareJoint: 60 Hz clamped to 0.25 / h, damping ratio 2), the
        //     push-out only on the biased pass (useBias), clamped >= 0. Box2D
        //     warm starts them (b2WarmStartPrismaticJoint): their accumulated
        //     impulses carry across sub-steps and steps and are re-applied at
        //     the start of each sub-step (joint pass #1), as the revolute point
        //     constraint's are.
        //
        // Translation (b2PrismaticJoint_GetTranslation, dot(axis, d)): the slide
        // of B's centre relative to A's along the axis, measured from the pose at
        // creation -- this joint's "anchors" are the two centres and the origin
        // offset m_orig, so d = (pB - pA) - m_orig and the translation is zero at
        // creation. Through a step the joint integrates its own translation from
        // the velocities the solver integrated (pass #2), as the revolute joint
        // does its angle.
        // ================================================================
        class PrismaticJoint final : public Joint
        {
        public:
            PrismaticJoint(BodyHandle a, BodyHandle b, Vec2 axis, Vec2 origin, Real refAngle,
                           bool enableMotor = false, Real motorSpeed = Real(0), Real maxMotorForce = Real(0))
                : m_hA(a), m_hB(b), m_axis(axis), m_orig(origin), m_refAngle(refAngle),
                  m_enableMotor(enableMotor), m_motorSpeed(motorSpeed),
                  m_maxMotorForce(maxMotorForce > Real(0) ? maxMotorForce : Real(0))
            {
            }
            void Prepare(PhysicsWorld& w, Real dt) override;
            void SolveVelocity(PhysicsWorld& w) override;
            // Each sub-step gets the full motor force budget (see
            // Joint::BeginSubstep); the limit impulses are kept (warm start).
            void BeginSubstep() noexcept override
            {
                m_substepping = true;
                m_pass = 0;
                m_motorImpulse = Real(0);
            }

            // The translation motor (b2PrismaticJoint's): drives B's speed along
            // the axis, relative to A, toward motorSpeed (m/s) with at most
            // maxMotorForce (N). A force-limited motor stalls against a load it
            // cannot move instead of forcing through it -- the safe way to build
            // a press, a piston or a gate. Driven into an enabled limit, it stops
            // there.
            void EnableMotor(bool on) noexcept { m_enableMotor = on; }
            void SetMotorSpeed(Real speed) noexcept { m_motorSpeed = speed; }
            void SetMaxMotorForce(Real force) noexcept { m_maxMotorForce = force > Real(0) ? force : Real(0); }
            [[nodiscard]] bool IsMotorEnabled() const noexcept { return m_enableMotor; }
            [[nodiscard]] Real MotorSpeed() const noexcept { return m_motorSpeed; }
            [[nodiscard]] Real MaxMotorForce() const noexcept { return m_maxMotorForce; }

            // ---- Box2D b2PrismaticJoint_* limit controls (take effect from the
            // next Step; a sleeping body is not woken here: the caller wakes it).
            // As Box2D's, a change clears the limit's accumulated impulses.
            void EnableLimit(bool on) noexcept
            {
                if (on != m_enableLimit) { m_enableLimit = on; m_lowerImpulse = m_upperImpulse = Real(0); }
            }
            void SetLimits(Real lower, Real upper) noexcept
            {
                const Real lo = std::min(lower, upper), hi = std::max(lower, upper);
                if (lo != m_lower || hi != m_upper) { m_lower = lo; m_upper = hi; m_lowerImpulse = m_upperImpulse = Real(0); }
            }
            [[nodiscard]] bool IsLimitEnabled() const noexcept { return m_enableLimit; }
            [[nodiscard]] Real GetLowerLimit() const noexcept { return m_lower; }
            [[nodiscard]] Real GetUpperLimit() const noexcept { return m_upper; }
            // The current translation (m) of B relative to A along the axis,
            // zero at creation (b2PrismaticJoint_GetTranslation).
            [[nodiscard]] Real GetTranslation(const PhysicsWorld& w) const noexcept;

            [[nodiscard]] std::uint32_t BodyA() const noexcept override { return m_ia; }
            [[nodiscard]] std::uint32_t BodyB() const noexcept override { return m_ib; }
            [[nodiscard]] BodyHandle HandleA() const noexcept override { return m_hA; }
            [[nodiscard]] BodyHandle HandleB() const noexcept override { return m_hB; }

        private:
            void WarmStartLimit(PhysicsWorld& w);
            void SolveLimit(PhysicsWorld& w, bool useBias);

            BodyHandle    m_hA, m_hB;
            std::uint32_t m_ia = kInvalidSlot, m_ib = kInvalidSlot;
            Vec2          m_axis{ Real(1), Real(0) }; // normalized slide axis
            Vec2          m_orig{ Real(0), Real(0) }; // B - A at creation
            Real          m_refAngle = Real(0);
            // Prepared per step.
            Real m_px = Real(0), m_py = Real(1); // perpendicular of the axis
            Real m_bias = Real(0);
            Real m_mass = Real(0);
            Real m_angBias = Real(0);
            Real m_angMass = Real(0);
            // Motor.
            bool m_enableMotor     = false;
            Real m_motorSpeed      = Real(0);
            Real m_maxMotorForce   = Real(0);
            Real m_maxMotorImpulse = Real(0); // maxMotorForce * dt
            Real m_motorImpulse    = Real(0); // accumulated this sub-step
            // Limit (b2PrismaticJointDef enableLimit / lowerTranslation / upperTranslation).
            bool m_enableLimit = false;
            Real m_lower = Real(0), m_upper = Real(0);
            // Prepared per step / tracked per sub-step.
            Real m_h = Real(0);                // the (sub-)step the joint is prepared for
            Real m_translation0 = Real(0);     // the translation at the start of the step
            Real m_deltaTranslation = Real(0); // its change, integrated through the sub-steps
            Real m_softBiasRate = Real(0), m_softMassScale = Real(1), m_softImpulseScale = Real(0);
            Real m_lowerImpulse = Real(0), m_upperImpulse = Real(0); // accumulated (warm-started)
            bool m_substepping = false;        // the solver calls BeginSubstep (SoftStep)
            int  m_pass = 0;                   // joint pass within the sub-step
        };

        // ================================================================
        // MouseJoint -- a soft critically-damped spring dragging body B to a
        // target with a force clamp (editor drag).
        // PORT: Joints.lua Mouse (lines 153-179). Has body B only; A is invalid.
        // ================================================================
        class MouseJoint final : public Joint
        {
        public:
            MouseJoint(BodyHandle b, Vec2 target, Real maxForce)
                : m_hB(b), m_target(target), m_maxForce(maxForce)
            {
            }
            void Prepare(PhysicsWorld& w, Real dt) override;
            void SolveVelocity(PhysicsWorld& w) override;
            [[nodiscard]] std::uint32_t BodyA() const noexcept override { return kInvalidSlot; }
            [[nodiscard]] std::uint32_t BodyB() const noexcept override { return m_ib; }
            [[nodiscard]] BodyHandle HandleA() const noexcept override { return BodyHandle{}; }
            [[nodiscard]] BodyHandle HandleB() const noexcept override { return m_hB; }

            // Move the spring target (editor drag). Ports Mouse:setTarget.
            void SetTarget(Vec2 t) noexcept { m_target = t; }
            [[nodiscard]] Vec2 Target() const noexcept { return m_target; }

        private:
            BodyHandle    m_hB;
            std::uint32_t m_ib = kInvalidSlot;
            Vec2          m_target{ Real(0), Real(0) };
            Real          m_maxForce = Real(1e6); // dead default (ctor always sets it); mirrors JointDef::maxForce -- see Joint.hpp
            // Prepared per step.
            Real m_k  = Real(0); // spring stiffness
            Real m_d  = Real(0); // spring damping
            Real m_dt = Real(0); // step dt (force -> impulse)
        };

        // ================================================================
        // WheelJoint (NEW; b2WheelJoint) -- body B attached to A along a
        // suspension axis through an anchor: a PERPENDICULAR-to-axis point
        // constraint held rigidly (B stays on the axis line), a SOFT spring
        // along the axis (suspension travel), FREE rotation, and an optional
        // rotation motor. Box2D-derived (no Lua oracle).
        // ================================================================
        class WheelJoint final : public Joint
        {
        public:
            WheelJoint(BodyHandle a, BodyHandle b, Vec2 localA, Vec2 localB,
                       Vec2 localAxisA, Real freqHz, Real dampingRatio,
                       bool enableMotor, Real motorSpeed, Real maxMotorTorque)
                : m_hA(a), m_hB(b), m_localA(localA), m_localB(localB),
                  m_localAxisA(localAxisA), m_freqHz(freqHz), m_damping(dampingRatio),
                  m_enableMotor(enableMotor), m_motorSpeed(motorSpeed),
                  m_maxMotorTorque(maxMotorTorque)
            {
            }
            void Prepare(PhysicsWorld& w, Real dt) override;
            void SolveVelocity(PhysicsWorld& w) override;
            void BeginSubstep() noexcept override { m_motorImpulse = Real(0); }

            // Live motor control (takes effect from the next Step; the torque
            // clamp is re-derived in Prepare). A vehicle throttles, brakes or
            // reverses through these instead of rebuilding its wheel joints.
            void EnableMotor(bool on) noexcept { m_enableMotor = on; }
            void SetMotorSpeed(Real speed) noexcept { m_motorSpeed = speed; }
            void SetMaxMotorTorque(Real torque) noexcept { m_maxMotorTorque = torque > Real(0) ? torque : Real(0); }
            [[nodiscard]] bool IsMotorEnabled() const noexcept { return m_enableMotor; }
            [[nodiscard]] Real MotorSpeed() const noexcept { return m_motorSpeed; }
            [[nodiscard]] Real MaxMotorTorque() const noexcept { return m_maxMotorTorque; }
            [[nodiscard]] std::uint32_t BodyA() const noexcept override { return m_ia; }
            [[nodiscard]] std::uint32_t BodyB() const noexcept override { return m_ib; }
            [[nodiscard]] BodyHandle HandleA() const noexcept override { return m_hA; }
            [[nodiscard]] BodyHandle HandleB() const noexcept override { return m_hB; }

        private:
            BodyHandle    m_hA, m_hB;
            std::uint32_t m_ia = kInvalidSlot, m_ib = kInvalidSlot;
            Vec2          m_localA{ Real(0), Real(0) };
            Vec2          m_localB{ Real(0), Real(0) };
            Vec2          m_localAxisA{ Real(1), Real(0) }; // suspension axis in A's frame
            Real          m_freqHz = Real(4);
            Real          m_damping = Real(0.7);
            bool          m_enableMotor = false;
            Real          m_motorSpeed = Real(0);
            Real          m_maxMotorTorque = Real(0);
            // Prepared per step.
            Vec2 m_rA{ Real(0), Real(0) };
            Vec2 m_rB{ Real(0), Real(0) };
            Vec2 m_perp{ Real(0), Real(1) }; // world perpendicular axis (rigid)
            Vec2 m_axis{ Real(1), Real(0) }; // world suspension axis (soft spring)
            // Perpendicular (rigid) constraint.
            Real m_perpMass = Real(0);
            Real m_perpBias = Real(0);
            Real m_sAp = Real(0), m_sBp = Real(0); // perp lever arms r x perp
            // Spring (axis) soft coefficients (b2MakeSoft).
            Real m_axMass = Real(0);
            Real m_sAa = Real(0), m_sBa = Real(0); // axis lever arms r x axis
            Real m_springBiasRate = Real(0);
            Real m_springMassScale = Real(1);
            Real m_springImpulseScale = Real(0);
            Real m_springSep = Real(0);          // axis separation (for the soft bias)
            Real m_springImpulse = Real(0);       // accumulated (warm across sub-steps)
            // Motor.
            Real m_motorMass = Real(0);
            Real m_motorImpulse = Real(0);        // accumulated
            Real m_maxMotorImpulse = Real(0);     // maxMotorTorque * dt
        };

        // ================================================================
        // MotorJoint (NEW; b2MotorJoint-simplified) -- drive the RELATIVE
        // angular velocity of B vs A toward motorSpeed, clamped to
        // +-maxMotorTorque*dt. A standalone angular motor. Box2D-derived.
        // ================================================================
        class MotorJoint final : public Joint
        {
        public:
            MotorJoint(BodyHandle a, BodyHandle b, Real motorSpeed, Real maxMotorTorque)
                : m_hA(a), m_hB(b), m_motorSpeed(motorSpeed), m_maxMotorTorque(maxMotorTorque)
            {
            }
            void Prepare(PhysicsWorld& w, Real dt) override;
            void SolveVelocity(PhysicsWorld& w) override;
            void BeginSubstep() noexcept override { m_impulse = Real(0); }

            // Live motor control (takes effect from the next Step).
            void SetMotorSpeed(Real speed) noexcept { m_motorSpeed = speed; }
            void SetMaxMotorTorque(Real torque) noexcept { m_maxMotorTorque = torque > Real(0) ? torque : Real(0); }
            [[nodiscard]] Real MotorSpeed() const noexcept { return m_motorSpeed; }
            [[nodiscard]] Real MaxMotorTorque() const noexcept { return m_maxMotorTorque; }
            [[nodiscard]] std::uint32_t BodyA() const noexcept override { return m_ia; }
            [[nodiscard]] std::uint32_t BodyB() const noexcept override { return m_ib; }
            [[nodiscard]] BodyHandle HandleA() const noexcept override { return m_hA; }
            [[nodiscard]] BodyHandle HandleB() const noexcept override { return m_hB; }

        private:
            BodyHandle    m_hA, m_hB;
            std::uint32_t m_ia = kInvalidSlot, m_ib = kInvalidSlot;
            Real          m_motorSpeed = Real(0);
            Real          m_maxMotorTorque = Real(0);
            // Prepared per step.
            Real m_mass = Real(0);
            Real m_impulse = Real(0);      // accumulated
            Real m_maxImpulse = Real(0);   // maxMotorTorque * dt
        };

        // ----------------------------------------------------------------
        // MakeJoint: the factory (ports Joints.make). Builds the concrete joint
        // from a JointDef, resolving creation-time geometry (local anchors,
        // reference angle, default distance length) against the world's current
        // body transforms. Returns nullptr for an unknown kind (never thrown so
        // AddJoint stays noexcept-friendly). The caller owns the returned joint.
        // ----------------------------------------------------------------
        [[nodiscard]] std::unique_ptr<Joint> MakeJoint(const PhysicsWorld& w,
                                                       const JointDef& def);

    } // namespace Physics
} // namespace Manifold2D
