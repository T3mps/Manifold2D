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
        //   Distance : length (defaults to current |B - A| if <= 0).
        //   Revolute : anchor (world point shared by A and B at creation).
        //   Weld     : anchor (as revolute) + the relative angle is locked to
        //              its value at creation.
        //   Prismatic: axis (world direction; B may only slide along it, with no
        //              perpendicular drift and no relative rotation).
        //   Mouse    : target + maxForce (body B only; A is kInvalidSlot).
        // NEW:
        //   Wheel    : axis (suspension direction, local to A's frame at
        //              creation, normalized) + anchor (world attach point) +
        //              suspension spring (frequencyHz, dampingRatio) + an
        //              optional rotation motor (enableMotor, motorSpeed,
        //              maxMotorTorque).
        //   Motor    : motorSpeed (target relative angular velocity of B vs A) +
        //              maxMotorTorque (impulse clamp).
        struct JointDef
        {
            JointKind  kind = JointKind::Distance;
            BodyHandle a{};            // body A (kInvalidBody for Mouse)
            BodyHandle b{};            // body B

            // Distance.
            Real length = Real(-1);    // <= 0 -> use the current separation

            // Revolute / Weld / Wheel: world anchor point at creation.
            Vec2 anchor{ Real(0), Real(0) };

            // Prismatic / Wheel: world axis (direction). Normalized at Prepare.
            Vec2 axis{ Real(1), Real(0) };

            // Mouse: target + force clamp. Default is an MKS-honest "effectively unclamped"
            // value = Box2D's drag-sample convention 1000*mass*g (samples/sample.cpp:338)
            // at the heaviest in-range body (~100 kg, g=10). Real callers set this per-body;
            // the only in-repo MouseJoint (PhysicsJointsTest) overrides it explicitly.
            Vec2 target{ Real(0), Real(0) };
            Real maxForce = Real(1e6);

            // Wheel suspension spring (b2WheelJoint). frequencyHz <= 0 -> a rigid
            // axis constraint (no suspension travel). dampingRatio is the spring's
            // zeta (1 = critically damped).
            Real frequencyHz  = Real(4);
            Real dampingRatio = Real(0.7);

            // Wheel / Motor rotation drive.
            bool enableMotor    = false;     // Wheel: drive the wheel's spin
            Real motorSpeed     = Real(0);   // target relative angular velocity (rad/s)
            Real maxMotorTorque = Real(0);    // impulse clamp magnitude (torque * dt)
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
        };

    } // namespace Physics
} // namespace Manifold2D
