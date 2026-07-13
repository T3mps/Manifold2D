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

#include <cstdint>
#include <memory>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Joints/Joint.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        class PhysicsWorld;

        // ================================================================
        // DistanceJoint -- hold a fixed separation between A and B.
        // PORT: Joints.lua Distance (lines 44-67).
        // ================================================================
        class DistanceJoint final : public Joint
        {
        public:
            DistanceJoint(BodyHandle a, BodyHandle b, Real length)
                : m_hA(a), m_hB(b), m_length(length)
            {
            }
            void Prepare(PhysicsWorld& w, Real dt) override;
            void SolveVelocity(PhysicsWorld& w) override;
            [[nodiscard]] std::uint32_t BodyA() const noexcept override { return m_ia; }
            [[nodiscard]] std::uint32_t BodyB() const noexcept override { return m_ib; }
            [[nodiscard]] BodyHandle HandleA() const noexcept override { return m_hA; }
            [[nodiscard]] BodyHandle HandleB() const noexcept override { return m_hB; }

        private:
            BodyHandle    m_hA, m_hB;
            std::uint32_t m_ia = kInvalidSlot, m_ib = kInvalidSlot;
            Real          m_length = Real(0);
            // Prepared per step.
            Real m_ux = Real(1), m_uy = Real(0); // unit A->B axis
            Real m_bias = Real(0);
            Real m_mass = Real(0);
        };

        // ================================================================
        // RevoluteJoint -- pin A and B at a shared anchor (a point constraint).
        // PORT: Joints.lua Revolute (lines 69-99).
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
            [[nodiscard]] std::uint32_t BodyA() const noexcept override { return m_ia; }
            [[nodiscard]] std::uint32_t BodyB() const noexcept override { return m_ib; }
            [[nodiscard]] BodyHandle HandleA() const noexcept override { return m_hA; }
            [[nodiscard]] BodyHandle HandleB() const noexcept override { return m_hB; }

        protected:
            BodyHandle    m_hA, m_hB;
            std::uint32_t m_ia = kInvalidSlot, m_ib = kInvalidSlot;
            Vec2          m_localA{ Real(0), Real(0) }; // anchor in A's local frame
            Vec2          m_localB{ Real(0), Real(0) }; // anchor in B's local frame
            // Prepared per step.
            Vec2 m_rA{ Real(0), Real(0) }; // world arm A
            Vec2 m_rB{ Real(0), Real(0) }; // world arm B
            Real m_biasX = Real(0), m_biasY = Real(0);
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
        // PORT: Joints.lua Prismatic (lines 120-151).
        // ================================================================
        class PrismaticJoint final : public Joint
        {
        public:
            PrismaticJoint(BodyHandle a, BodyHandle b, Vec2 axis, Vec2 origin, Real refAngle)
                : m_hA(a), m_hB(b), m_axis(axis), m_orig(origin), m_refAngle(refAngle)
            {
            }
            void Prepare(PhysicsWorld& w, Real dt) override;
            void SolveVelocity(PhysicsWorld& w) override;
            [[nodiscard]] std::uint32_t BodyA() const noexcept override { return m_ia; }
            [[nodiscard]] std::uint32_t BodyB() const noexcept override { return m_ib; }
            [[nodiscard]] BodyHandle HandleA() const noexcept override { return m_hA; }
            [[nodiscard]] BodyHandle HandleB() const noexcept override { return m_hB; }

        private:
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
