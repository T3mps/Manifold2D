#pragma once

// BodyState.hpp: 32-byte AoS body-state row for the SIMD constraint solver.
//
// The lane-wide (graph-colored, SoA) solve gathers/scatters body velocities by
// the constraint's body-index lanes. The world stores velocity as a per-
// component Vec2 SoA (m_velX/m_velY/m_angVel) that is NOT gather-friendly, so
// per Step we mirror a PACKED copy of the body state into the AoS rows below,
// indexed BY SOLVERINDEX -- a DENSE per-step index space (Phase C, Task 2),
// NOT the sparse world slot. The solver sizes them to solverCount where
// solverCount = AwakeCount() + KinematicCount():
//   * awake dynamics occupy [0, AwakeCount())              at AwakeIndexOf(slot)
//   * kinematics    occupy [AwakeCount(), solverCount)     at AwakeCount()+KinematicIndexOf(slot)
//   * statics / spans / padding have NO row -- they pack kNullBodyIndex (-1) and
//     the lane-wide gather injects a shared zero IDENTITY row for them (Task 3,
//     Gap 2.2; see ContactConstraintSimd's NULL-INDEX BRANCH note). This REPLACES
//     the old "+1" scatter-safe dummy tail (no dummy slot, no write contention).
// The packed scratch is dense (no per-world-slot holes), so the gathers stay
// cache-local (the WIN -- not gather elimination).
//
// LAYOUT: one 32-byte AoS row per solver slot (8 floats: vx/vy/w/dpx/dpy/dq +
// 2 padding floats). alignas(32) places rows on cache-line boundaries.
//
// SCALAR CHOICE: fields are `float` (NOT Real) deliberately -- they feed the
// SIMD f32w path directly. Vec2/Real/std::uint32_t come from PhysicsTypes.
//
// SCOPE: the struct + its sync bridge. SoftStep::Solve consumes the DENSE
// SyncInCompacted/SyncOutCompacted methods (solverIndex-indexed -- the lane-wide
// colored SoA contact solve gathers/scatters through them). The legacy world-slot
// SyncIn/SyncOut survive ONLY for the standalone round-trip contract test in
// PhysicsSimdSolverTest.cpp (they are NOT on the solver hot path). All four sync
// helpers are DECLARED here (lightweight: forward-declare PhysicsWorld, include
// only PhysicsTypes) and DEFINED in SoftStep.cpp where the PhysicsWorld accessors
// are in scope.
//
// PRESENTATION-FREE + C++20-clean.

#include <cstdint>
#include <vector>

#include <Manifold2D/Physics/PhysicsTypes.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        class PhysicsWorld;

        // Solver-local packed body-state row, indexed BY SOLVERINDEX -- a DENSE
        // per-step index space (Phase C, Task 2), NOT the sparse world slot. Eight
        // floats: velocity (vx,vy,w), TGS position/angle delta (dpx,dpy,dq), and two
        // padding floats to reach 32 bytes (one full cache line, 32-byte aligned).
        // alignas(32) ensures each row starts on a 32-byte boundary and sizeof==32.
        struct alignas(32) BodyState
        {
            float vx, vy, w;      // linear + angular velocity
            float dpx, dpy, dq;  // accumulated position/angle delta (TGS)
            float pad0, pad1;    // padding: sizeof(BodyState) == 32
        };
        static_assert(sizeof(BodyState) == 32, "BodyState must be 32 bytes");
        static_assert(alignof(BodyState) == 32, "BodyState must be 32-byte aligned");

        // Container for the solver's AoS body-state scratch. Sized to solverCount
        // per Solve (Task 3 dropped the old "+1" scatter-safe dummy tail for the
        // null-index branch). MSVC's std::allocator honors 32-byte over-alignment for
        // BodyState (C++17 over-aligned new), so the default vector is fine -- the
        // transpose gather does an aligned 256-bit load per row, which requires it.
        //
        // Declared here, sync defs in SoftStep.cpp (needs PhysicsWorld accessors).
        class BodyStateStore
        {
        public:
            // Resize to n zero-initialized rows. The solver passes solverCount.
            void Resize(std::uint32_t n)
            {
                m_states.assign(static_cast<std::size_t>(n), BodyState{});
            }

            BodyState* data() noexcept { return m_states.data(); }
            const BodyState* data() const noexcept { return m_states.data(); }
            std::size_t size() const noexcept { return m_states.size(); }

            BodyState& operator[](std::size_t i) { return m_states[i]; }
            const BodyState& operator[](std::size_t i) const { return m_states[i]; }

            // Declared here, defined in SoftStep.cpp (needs PhysicsWorld accessors).
            //
            // SyncInCompacted (Phase C, Task 2): fill the DENSE scratch -- awake
            // dynamics at AwakeIndexOf(slot) (vel mirrored, dp/dq zeroed), kinematics
            // at AwakeCount()+KinematicIndexOf(slot) (vel mirrored; dp/dq stay zero,
            // they never integrate). Statics/spans have no row (they pack the null
            // index and gather a zero identity). The caller Resize()s to solverCount.
            //
            // SyncOut writes awake dynamics' velocity back to the world (kinematics are
            // read-only -> never written). The caller passes the same dense index map
            // by re-deriving AwakeIndexOf(slot) per awake slot.
            //
            // SyncIn (legacy, world-slot-indexed): the original sparse fill -- world
            // vel -> row by WORLD SLOT; dp/dq zeroed for awake dynamics. NO LONGER on
            // the solver hot path (SyncInCompacted replaced it in Solve), but kept for
            // the standalone SyncIn/SyncOut round-trip contract test which pins the
            // world-slot bridge in isolation. Caller Resize()s to world.Count() first.
            void SyncIn(const PhysicsWorld& world);          // legacy world-slot fill
            void SyncInCompacted(const PhysicsWorld& world); // dense fill (solverIndex)
            void SyncOut(PhysicsWorld& world) const;         // legacy world-slot vel write-back
            // Dense write-back (Phase C, Task 2): awake dynamics' vel at
            // AwakeIndexOf(slot) -> world (kinematics read-only -> never written).
            void SyncOutCompacted(PhysicsWorld& world) const;

        private:
            std::vector<BodyState> m_states;
        };

    } // namespace Physics
} // namespace Manifold2D
