#pragma once

// StepTrace: the per-solver-stage snapshot record PhysicsWorld::StepTraced fills.
//
// WHY: the step is the product. A consumer (the Manifold hero's WebAssembly
// build, an editor inspector) wants to freeze one Step and walk the TGS Soft
// stage sequence the solver actually runs -- five stages per sub-step
// (IntegrateVelocities / WarmStart / Solve / IntegratePositions / Relax) then
// Restitution and StoreImpulses once -- reading the engine's OWN numbers at
// each stop. StepTraced is a sibling entry point on PhysicsWorld::StepImpl, so
// the untraced Step(dt) path pays one never-taken branch per stage group.
//
// MID-STEP POSES are reconstructed, not read from the world: the solver
// accumulates TGS deltas in its own BodyState rows (dpx/dpy/dq) and commits
// them to the world ONCE at the end (SoftStep::FinalizePositionsSoA). A
// snapshot therefore composes the pose the step STARTED from with the deltas
// accumulated so far, which makes the final snapshot bit-identical to the pose
// the Step leaves behind.
//
// PRESENTATION-FREE + dependency-free: Geometry::Vec2 + std only.

#include <cstdint>
#include <vector>

#include <Manifold2D/Physics/PhysicsTypes.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        // Opaque re-declaration of Solver/SolverStages.hpp's stage enum. An
        // enumeration with a FIXED underlying type is a complete type from the
        // end of its enum-base ([dcl.enum]/7), so it can be a struct member
        // here without dragging the SIMD solver headers into every consumer of
        // PhysicsWorld.hpp. To NAME an enumerator (StageType::Solve) a TU must
        // also include <Manifold2D/Physics/Solver/SolverStages.hpp>.
        enum class StageType : std::uint8_t;

        // One body's state at one stage. `body` is the WORLD SLOT
        // (BodyHandle::index) so a consumer can index a slot-ordered body array
        // with it directly.
        struct StepTraceBody
        {
            std::uint32_t body = kInvalidSlot;
            Vec2          position{ Real(0), Real(0) };
            Real          angle = Real(0);
            Vec2          velocity{ Real(0), Real(0) };
            Real          angularVelocity = Real(0);
        };

        // One MANIFOLD POINT of one active contact constraint at one stage.
        // bodyA/bodyB are body SLOTS (bodyB == kInvalidSlot for a tile-span
        // constraint); fixtureA/fixtureB are FixtureHandle::index values
        // resolved through the persistent contact pool (kInvalidSlot when the
        // constraint has no pool home). `point` is world space; `normal` points
        // from B toward A; `separation` is the TGS current separation (> 0 is a
        // gap); the impulses are the ACCUMULATED normal/tangent impulses at
        // that instant.
        struct StepTraceContact
        {
            std::uint32_t bodyA    = kInvalidSlot;
            std::uint32_t bodyB    = kInvalidSlot;
            std::uint32_t fixtureA = kInvalidSlot;
            std::uint32_t fixtureB = kInvalidSlot;
            Vec2 point{ Real(0), Real(0) };
            Vec2 normal{ Real(0), Real(0) };
            Real separation     = Real(0);
            Real normalImpulse  = Real(0);
            Real tangentImpulse = Real(0);
        };

        // One stop. `bodies` holds every solver row in dense solver order: the
        // awake dynamics in AwakeBodies() order first, then the kinematics in
        // KinematicBodies() order. `substep` is the sub-step index for the five
        // in-sub-step stages and substepCount for Restitution/StoreImpulses.
        struct StepTraceSnapshot
        {
            StageType    stage{};
            std::uint8_t substep = 0;
            std::vector<StepTraceBody>    bodies;
            std::vector<StepTraceContact> contacts;
        };

        // The whole capture. StepTraced APPENDS one step's stops; the caller
        // clears it to keep only the latest step.
        struct StepTrace
        {
            std::vector<StepTraceSnapshot> snapshots;
        };

    } // namespace Physics
} // namespace Manifold2D
