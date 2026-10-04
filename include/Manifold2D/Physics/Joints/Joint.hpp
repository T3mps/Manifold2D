#pragma once

// Joint: the polymorphic joint base + JointDef (M6, Task P2.5).
//
// PORT + MODERNIZE: ports the joint set behind Client/src/physics/Joints.lua
// (Distance / Revolute / Weld / Prismatic / Mouse), each a velocity constraint
// with a Baumgarte positional bias (BETA = 0.2) solved inside the solver's
// velocity-iteration loop. The Lua `:init(w, dt)` becomes `Prepare(w, dt)` and
// `:solve(w)` becomes `SolveVelocity(w)`. MODERNIZED to "completeness to a T"
// by ADDING two Box2D-derived joints not present in the Lua: an angular Motor
// (b2MotorJoint-simplified) and a Wheel suspension joint (b2WheelJoint).
//
// This file DEFINES the `struct Joint` that Solver/Solver.hpp forward-declared
// since P2.1 (its JointConstraint holds a `Joint*`). The two body slots are the
// ISLAND seam: BodyA()/BodyB() return SoA slot indices (kInvalidSlot for a
// static-anchor or a missing body, e.g. Mouse has no body A) so the island pass
// can keep jointed dynamic bodies awake.
//
// SOFT-CONSTRAINT FORMULATION (the plan's title): these are Baumgarte-bias
// VELOCITY constraints. Under the SoftStep solver they are Prepared with the
// SUB-STEP dt (subDt) and SolveVelocity'd once per sub-step's velocity solve;
// the sub-stepping is what "softens" them (the bias BETA/subDt is applied over
// substepCount small steps, the proven-stable TGS pattern). Under the Baumgarte
// oracle they are Prepared once with the full dt and solved each velocity
// iteration, exactly the Lua SequentialImpulse ordering.
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.
// No SDL3/NVRHI/Batcher2D/ImGui. namespace Manifold2D::Physics, Core style.

#include <cstdint>

#include <Manifold2D/Physics/PhysicsTypes.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        class PhysicsWorld;

        // Baumgarte positional-correction factor folded into the joint velocity
        // constraints (Joints.lua BETA = 0.2). Shared by every joint type.
        inline constexpr Real kJointBeta = Real(0.2);
        // Box2D v3 joint constraint softness (b2DefaultJointDef: constraintHertz 60,
        // constraintDampingRatio 2; b2PrepareJoint clamps the hertz to 0.25 / h).
        // Used by the revolute limit's soft push-out.
        inline constexpr Real kJointConstraintHertz = Real(60);
        inline constexpr Real kJointConstraintDampingRatio = Real(2);

        // Mouse-joint critically-damped spring constants (Joints.lua FREQ/ZETA).
        inline constexpr Real kMouseFreq = Real(5);
        inline constexpr Real kMouseZeta = Real(1);

        // ----------------------------------------------------------------
        // JointKind: which joint a JointDef builds (the Lua def.type string).
        // ----------------------------------------------------------------
        enum class JointKind : std::uint8_t
        {
            Distance  = 0, // hold a fixed separation between A and B (Lua "distance")
            Revolute  = 1, // pin A and B at a shared world anchor (Lua "revolute")
            Weld      = 2, // revolute + relative-angle lock (Lua "weld")
            Prismatic = 3, // slide along a world axis, no perp drift / rel rotation (Lua "prismatic")
            Mouse     = 4, // soft spring dragging body B to a target (Lua "mouse")
            Wheel     = 5, // NEW: b2WheelJoint suspension (perp rigid + axis spring + motor)
            Motor     = 6, // NEW: b2MotorJoint-simplified angular motor (rel angVel -> motorSpeed)
        };

        // ----------------------------------------------------------------
        // JointDef: tagged parameters for PhysicsWorld::AddJoint (ports the Lua
        // def table; each kind reads the fields documented below).
        // ----------------------------------------------------------------
        //
        // PORT mapping (Joints.make, Joints.lua:182-227):
        //   Distance : length (defaults to the current anchor-to-anchor distance
        //              if <= 0); Box2D v3 b2DistanceJointDef's local anchors
        //              (localAnchorA, localAnchorB), limit (enableLimit,
        //              minLength, maxLength) and spring (enableSpring,
        //              frequencyHz, dampingRatio). A rope is enableSpring with
        //              frequencyHz 0 plus enableLimit [0, maxLength].
        //   Revolute : anchor (world point shared by A and B at creation).
        //   Weld     : anchor (as revolute) + the relative angle is locked to
        //              its value at creation.
        //   Prismatic: axis (world direction; B may only slide along it, with no
        //              perpendicular drift and no relative rotation); Box2D v3
        //              b2PrismaticJointDef's optional limit (enableLimit,
        //              lowerTranslation, upperTranslation) and force-limited motor
        //              (enableMotor, motorSpeed, maxMotorForce).
        //   Mouse    : target + maxForce (body B only; A is kInvalidSlot).
        // NEW:
        //   Wheel    : axis (suspension direction, local to A's frame at
        //              creation, normalized) + anchor (world attach point) +
        //              suspension spring (frequencyHz, dampingRatio) + an
        //              optional rotation motor (enableMotor, motorSpeed,
        //              maxMotorTorque).
        //   Motor    : motorSpeed (target relative angular velocity of B vs A) +
        //              maxMotorTorque (impulse clamp).
        //   Revolute (Box2D v3 b2RevoluteJointDef): referenceAngle + optional
        //              limit (enableLimit, lowerAngle, upperAngle), spring
        //              (enableSpring, frequencyHz, dampingRatio, targetAngle) and
        //              motor (enableMotor, motorSpeed, maxMotorTorque as a torque
        //              in N m: the impulse is clamped to h * maxMotorTorque).
        struct JointDef
        {
            JointKind  kind = JointKind::Distance;
            BodyHandle a{};            // body A (kInvalidBody for Mouse)
            BodyHandle b{};            // body B

            // Distance.
            Real length = Real(-1);    // <= 0 -> use the current separation
            // Distance anchors, in each body's local frame (relative to its
            // origin; Box2D localAnchorA/B). (0, 0) is the body origin.
            Vec2 localAnchorA{ Real(0), Real(0) };
            Vec2 localAnchorB{ Real(0), Real(0) };
            // Distance limit (with enableLimit, and only while the spring is on):
            // minLength <= length <= maxLength. Clamped to [kLinearSlop, 1e5].
            Real minLength = Real(0);
            Real maxLength = Real(100000);

            // Revolute / Weld / Wheel: world anchor point at creation.
            Vec2 anchor{ Real(0), Real(0) };

            // Revolute: the joint angle is angleB - angleA - referenceAngle
            // (Box2D's angle between the joint frames). Box2D's default is 0, so the
            // joint angle is the raw relative angle; pass the creation-time relative
            // angle to measure the limits and the spring from the pose at creation.
            Real referenceAngle = Real(0);
            // Revolute limit: lowerAngle <= joint angle <= upperAngle (radians;
            // Box2D documents a usable range of about +-0.99 pi). enableLimit also
            // switches on the Prismatic limit (lower/upperTranslation below) and
            // the Distance limit (minLength/maxLength above).
            bool enableLimit = false;
            Real lowerAngle  = Real(0);
            Real upperAngle  = Real(0);
            // Revolute spring: drives the joint angle to targetAngle, soft at
            // (frequencyHz, dampingRatio) below. enableSpring also switches on the
            // Distance spring (rest length `length`; frequencyHz 0 = no length
            // constraint).
            bool enableSpring = false;
            Real targetAngle  = Real(0);

            // Prismatic / Wheel: world axis (direction). Normalized at Prepare.
            Vec2 axis{ Real(1), Real(0) };
            // Prismatic limit (b2PrismaticJointDef), on when enableLimit:
            // lowerTranslation <= translation <= upperTranslation (m along the
            // axis; the translation of B relative to A is zero at creation).
            Real lowerTranslation = Real(0);
            Real upperTranslation = Real(0);

            // Mouse: target + force clamp. Default is an MKS-honest "effectively unclamped"
            // value = Box2D's drag-sample convention 1000*mass*g (samples/sample.cpp:338)
            // at the heaviest in-range body (~100 kg, g=10). Real callers set this per-body;
            // the only in-repo MouseJoint (PhysicsJointsTest) overrides it explicitly.
            Vec2 target{ Real(0), Real(0) };
            Real maxForce = Real(1e6);

            // Wheel suspension spring (b2WheelJoint), and the Revolute and
            // Distance springs when enableSpring (NOTE: the default 4 Hz applies to
            // them too; a Distance rope sets frequencyHz = 0). frequencyHz <= 0 -> a rigid
            // axis constraint (no suspension travel). dampingRatio is the spring's
            // zeta (1 = critically damped).
            Real frequencyHz  = Real(4);
            Real dampingRatio = Real(0.7);

            // Wheel / Motor rotation drive; Prismatic translation drive.
            bool enableMotor    = false;     // Wheel: drive the wheel's spin; Prismatic: drive the slide
            Real motorSpeed     = Real(0);   // Wheel/Motor: rad/s; Prismatic: m/s along the axis
            Real maxMotorTorque = Real(0);    // impulse clamp magnitude (torque * dt)
            Real maxMotorForce  = Real(0);    // Prismatic: the motor's force limit (N)
        };

        // ----------------------------------------------------------------
        // Joint: the polymorphic constraint base (defines Solver.hpp's fwd decl).
        // ----------------------------------------------------------------
        //
        // Lifecycle: AddJoint constructs the concrete joint from a JointDef +
        // resolves its body HANDLES to SoA slots (captured at Prepare so a slot
        // recycle is observed). Prepare(w, dt) precomputes the per-step constants
        // (anchors, effective mass, Baumgarte bias) -- the Lua :init. SolveVelocity
        // (w) applies one velocity-constraint pass -- the Lua :solve. The solver
        // calls Prepare once (or per sub-step for SoftStep) and SolveVelocity each
        // velocity iteration / sub-step.
        struct Joint
        {
            virtual ~Joint() = default;

            // Precompute this step's (or sub-step's) constants: resolve body
            // slots, world anchors, effective masses, and the Baumgarte bias
            // (which uses `dt` -- the full dt for Baumgarte, the sub-step dt for
            // SoftStep). Ports the Lua :init(w, dt).
            virtual void Prepare(PhysicsWorld& w, Real dt) = 0;

            // Apply one velocity-constraint solve pass. Ports the Lua :solve(w).
            virtual void SolveVelocity(PhysicsWorld& w) = 0;

            // Called by the solver at the start of every sub-step, before that
            // sub-step's first SolveVelocity. Impulse-clamped drives (the Motor
            // and Wheel motors) reset their accumulated impulse here so the
            // clamp maxMotorTorque * subDt is a per-SUB-STEP budget: the motor
            // then delivers maxMotorTorque over the whole step, as Box2D's
            // per-sub-step motor does. Default: nothing to reset.
            virtual void BeginSubstep() noexcept {}

            // The two body slots (kInvalidSlot for a static-anchor / missing
            // body). The ISLAND pass reads these to keep jointed dynamic bodies
            // awake. Resolved at Prepare; kInvalidSlot before the first Prepare.
            [[nodiscard]] virtual std::uint32_t BodyA() const noexcept = 0;
            [[nodiscard]] virtual std::uint32_t BodyB() const noexcept = 0;

            // The two body HANDLES this joint was created with (kInvalidBody for
            // a missing body, e.g. Mouse's A). Stable from construction (NOT
            // dependent on Prepare) so RemoveBody can drop joints referencing a
            // destroyed body by handle index. Ports the Lua j.a.idx / j.b.idx
            // membership test (PhysicsWorld.lua:281-286).
            [[nodiscard]] virtual BodyHandle HandleA() const noexcept = 0;
            [[nodiscard]] virtual BodyHandle HandleB() const noexcept = 0;

            // ---- reaction (Box2D v3 b2Joint_GetConstraintForce / _GetConstraintTorque)
            //
            // The force (N, world frame) and the torque (N m) this joint applied
            // to body B over the last step it was solved in -- what a breakable
            // joint compares against its threshold. The force is every linear
            // impulse the joint's constraints put on B (point, axis, motor,
            // spring); the torque is only its PURE angular impulses (an angle
            // lock, a motor, a spring, a limit), not the moment r x F of the
            // force about B's centre -- Box2D's split.
            //
            // DEFINITION: the total impulse delivered to B over the step, divided
            // by the step dt. Box2D reports its accumulated impulse J times
            // inv_h (the SUB-step inverse); that J is warm-started, i.e. applied
            // in full again every sub-step, so over one step of N sub-steps the
            // joint delivers sum_k J_k and sum_k J_k / (N h) is the mean of
            // Box2D's per-sub-step reading -- equal to it when J is steady (a
            // load at rest, a stalled motor) and smoothed over the step in a
            // transient. This engine's Baumgarte joints carry no accumulated
            // impulse (each pass applies a fresh one), so the delivered sum is
            // the one definition that holds for every kind; it is pure
            // bookkeeping and changes no solve.
            //
            // The world opens the window (BeginReactionWindow) for every joint it
            // hands the solver; a sleeping joint keeps its last awake reading, as
            // Box2D's keeps its impulses. Zero before the first solved step.
            [[nodiscard]] Vec2 ReactionForce() const noexcept
            {
                const Real inv = m_reactionDt > Real(0) ? Real(1) / m_reactionDt : Real(0);
                return Vec2(m_reactionImpulse.x * inv, m_reactionImpulse.y * inv);
            }
            [[nodiscard]] Real ReactionTorque() const noexcept
            {
                return m_reactionDt > Real(0) ? m_reactionAngularImpulse / m_reactionDt : Real(0);
            }
            // Reset the per-step sums for a step of length `dt` (called by
            // PhysicsWorld before the solve, not by the solver).
            void BeginReactionWindow(Real dt) noexcept
            {
                m_reactionImpulse = Vec2(Real(0), Real(0));
                m_reactionAngularImpulse = Real(0);
                m_reactionDt = dt;
            }

        protected:
            // Record an impulse the joint just applied to body B (A gets the
            // opposite). Every SolveVelocity path calls these beside its apply.
            void AddReaction(Real jx, Real jy) noexcept
            {
                m_reactionImpulse = Vec2(m_reactionImpulse.x + jx, m_reactionImpulse.y + jy);
            }
            void AddReactionTorque(Real j) noexcept { m_reactionAngularImpulse += j; }

        private:
            Vec2 m_reactionImpulse{ Real(0), Real(0) }; // linear impulse on B, this step
            Real m_reactionAngularImpulse = Real(0);    // pure angular impulse on B, this step
            Real m_reactionDt = Real(0);                // the step the sums cover
        };

    } // namespace Physics
} // namespace Manifold2D
