#pragma once

// SoftStep: the Box2D-v3-class TGS Soft constraint solver (M6, Task P2.2).
//
// MODERNIZATION CENTERPIECE -- NOT a port. The Lua engine's solver
// (SequentialImpulse.lua) was a Baumgarte sequential-impulse PGS solver; it was
// ported as the P2.3 A/B cross-check oracle, since retired (MKS P2). SoftStep is a
// from-scratch implementation of the Box2D v3 "Solver2D" / "TGS Soft"
// algorithm (https://box2d.org/posts/2024/02/solver2d/,
// https://box2d.org/posts/2024/08/releasing-box2d-3.0/). It is validated by
// BEHAVIORAL INVARIANTS (a stack settles, a ball rests, restitution rebounds,
// friction stops a slide, kinematic pushes dynamic, energy stays bounded,
// determinism holds), NOT bit-matched to the Lua.
//
// ALGORITHM (per full Step(dt), substepCount sub-steps, h = dt/substepCount):
//   1. PrepareContacts (once): per contact point compute anchors, normal +
//      tangent effective masses, the rest-relative normal velocity (for
//      restitution), and the SOFT (biasRate, massScale, impulseScale) from
//      b2MakeSoft(contactHertz, dampingRatio, h). Warm-start impulses are NOT
//      seeded here -- they arrive already set on each ContactConstraintPoint
//      (EmitContactConstraints copies them off the persistent Contact's
//      manifold point). PrepareContacts must NOT reset them.
//   2. for substep in [0, substepCount):
//        a. integrate velocities (gravity + linear damping) for awake dynamics
//        b. warm start: apply accumulated normal + tangent impulses
//        c. biased solve (useBias=true): TGS soft normal solve (re-evaluate
//           separation from the per-body position deltas accumulated this step)
//           + friction
//        d. integrate positions: accumulate dp += h*v, dq += h*w
//        e. relax pass (useBias=false): same solve with bias=0, massScale=1,
//           impulseScale=0 -- removes the bias-injected energy
//   3. ApplyRestitution (once): add restitution impulse for points whose
//      approach speed exceeded the threshold.
// IMPLEMENTATION: each contact pass (b/c/e + restitution) runs LANE-WIDE over a
// per-step graph-colored AoS+SoA (BodyStateStore + ContactConstraintSimd,
// the SimdSolve passes) plus a width-1 scalar Overflow* pass for the un-colorable
// bucket; the integrate passes (a/d) iterate the SoA scalar. Joints stay scalar
// against the world (velocities bridged SoA<->world only around the joint passes).
// The converged per-point impulses are stored back onto ctx.contacts; after Solve()
// returns, PhysicsWorld::Step writes each ContactConstraintPoint's accumulated
// (normal, tangent) impulse back onto its source persistent Contact's manifold
// point (via ContactConstraint::sourceContactId), where they persist to next
// step. This replaced the old solver-owned m_cache.
// At the end the solver commits each awake dynamic body's accumulated dp/dq to the
// world position/angle (the world's stage-4 inline dynamic position integrate is
// SUPERSEDED -- the solver owns dynamic integration).
//
// The world just calls Solve once per Step (it does NOT drive a phase sequence).
// The world owns the ContactConstraint pool; the warm-start impulses live on the
// graph-owned persistent Contact. The solver owns only the per-step SoA scratch (m_bodyState
// dp/dq + the coloring buffers), reused across steps -> zero steady-state
// allocation. Determinism: fixed iteration order -- the coloring + each color's
// batch packing are deterministic functions of the world-built (already sorted)
// contact array; no wall-clock; no fast-math (the workspace builds /fp:precise).
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers
// only. No SDL3/NVRHI/ImGui. namespace Manifold2D::Physics, Core style.

#include <array>
#include <cstdint>
#include <vector>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Solver/Solver.hpp>
#include <Manifold2D/Physics/Solver/BodyState.hpp>            // SIMD solve body state
#include <Manifold2D/Physics/Solver/ContactColoring.hpp>     // kColorCount (color bucket count)
#include <Manifold2D/Physics/Solver/ContactConstraintSimd.hpp> // SoA batches + lane solve
#include <Manifold2D/Physics/Solver/SolverStages.hpp>         // SolverStage/SolverBlock + region driver

namespace Manifold2D
{
    namespace Physics
    {
        // ----------------------------------------------------------------
        // SoftStep: ISolver implementation (Box2D v3 TGS Soft).
        // ----------------------------------------------------------------
        class SoftStep final : public ISolver
        {
        public:
            SoftStep() = default;
            ~SoftStep() override = default;

            // ---- ISolver entry (the single per-Step driver) ----------------
            //
            // Runs the full TGS Soft pipeline as a LANE-WIDE COLORED-SoA solve:
            //   sync world velocities -> BodyStateStore AoS rows (sized solverCount;
            //     read-only-B/padding lanes pack the null index, not a dummy slot) ;
            //     PrepareContacts/PrepareJoints (scalar)
            //   -> bucket the touching contacts by their PERSISTENT contact color
            //      (assigned incrementally at create/destroy -- no per-step recolor)
            //   -> Build one ContactConstraintSimd batch list per color
            //   -> sub-step loop: integrate-vel(SoA) -> warm-start -> solve(bias)
            //      -> integrate-pos(SoA) -> relax(no-bias), each contact pass run
            //      lane-wide over every color (SimdSolve::*) plus a width-1 scalar
            //      Overflow* pass for the un-colorable bucket; joints stay scalar
            //      (velocities bridged SoA<->world only around the joint passes)
            //   -> restitution (lane-wide + overflow) -> store impulses back onto
            //      ctx.contacts -> SyncOut velocities -> commit positions (SoA).
            // The solver owns the dynamic velocity + position integration the v3
            // algorithm folds into the sub-steps; the world just calls Solve once
            // per Step. This driver was REWRITTEN from the scalar Array-of-Structs
            // sequential Gauss-Seidel to the colored-SoA SIMD solve above, with a
            // deliberate numerics re-baseline (validated by the behavioral invariants
            // + the lane-width-invariance / scalar-oracle bit-match tests, NOT by
            // byte-identity to the old scalar path).
            void Solve(SolverContext& ctx) override;

            // Drop a body's warm-start state on slot recycle (RemoveBody). SoftStep
            // keeps NO solver-side cache anymore -- warm-start impulses live on the
            // persistent Contact, which the broadphase destroys when a fixture/body
            // is removed (DestroyContactsForBody/Fixture), so a recycled slot
            // inherits nothing. This override stays a no-op purely to honor the
            // ISolver contract. (WarmStartCacheSize() is intentionally NOT
            // overridden: it falls back to ISolver's default 0 -- there is no cache
            // to report. The hook + PhysicsWorld::SolverWarmStartCacheSize remain
            // as a seam for a future solver that keeps its own cache.)
            void DropBody(std::uint32_t slot) override;

            // Un-colorable constraint count from the LAST Solve (test/inspection
            // hook -- see ISolver::LastOverflowCount). Returns m_overflowRefs's size:
            // Phase C, Task 5 deleted the per-step greedy recolor; the driver now
            // buckets the emitted constraints by their PERSISTENT contact color and
            // collects every kInvalidColor constraint (an overflow contact OR a span)
            // into m_overflowRefs, refilled every Solve and never cleared afterward,
            // so it accurately reflects the most recent completed Solve's spill set.
            // The overflow-settle test asserts this peaks above 0 to PROVE the dense
            // dynamic-dynamic hub really did spill past kColorCount into the scalar
            // OverflowSolve tail (else that path would go silently untested).
            [[nodiscard]] std::size_t LastOverflowCount() const noexcept override
            {
                return m_overflowRefs.size();
            }

        private:
            // ---- scalar prepare + joint helpers (called by Solve) ----------
            //
            // Solve() sequences these. PrepareContacts/PrepareJoints run once per
            // Step (scalar on ctx.contacts/ctx.joints); SolveJoints runs one joint
            // velocity pass per sub-step (joints stay scalar against the world --
            // Part 2 SIMD-izes them). They are SoftStep-private non-virtual helpers
            // (ISolver is slimmed to Solve()+DropBody(); SoftStep owns its own phase
            // ordering internally).
            void PrepareContacts(SolverContext& ctx);
            void PrepareJoints(SolverContext& ctx);            // joint Prepare (subDt bias)
            void SolveJoints(SolverContext& ctx);              // one joint velocity pass (per sub-step)

            // ---- SIMD lane-wide contact-solve helpers (Part 1) -------------
            //
            // The contact solve is the hot path; Solve() runs it lane-wide over a
            // graph-colored AoS+SoA (BodyStateStore + ContactConstraintSimd) via the
            // SimdSolve passes. These helpers integrate velocity/position over the
            // AoS (scalar O(n)), bridge velocities to/from the world for the scalar
            // joint passes, and solve the un-colorable overflow bucket scalar (the
            // width-1 sequential twin of the SimdSolve passes -- they share one
            // BodyStateStore so they MUST stay numerically in lockstep).

            // Integrate-range helpers: the per-body loop bodies of the two integrate
            // passes (velocity = gravity + linear damping; position = dp/dq += v*h),
            // callable over AwakeBodies()[begin,end). The persistent solver region
            // drives these directly via the IntegrateVelocities / IntegratePositions
            // body stages, one sub-range per block. Per-body writes are to disjoint
            // dense rows (AwakeIndexOf is a bijection), so any partition -- hence any
            // worker count -- is scatter-safe + byte-identical.
            void IntegrateVelocitiesRange(SolverContext& ctx, Real h, std::size_t begin, std::size_t end);
            void IntegratePositionsRange(SolverContext& ctx, Real h, std::size_t begin, std::size_t end);

            // Build the flat per-step stage/block list into m_stages/m_blocks (reused
            // scratch; resize-not-realloc). One stage per non-empty color for each
            // colored type + one stage each for the two body types, blocks partitioned
            // per spec 5.2 (contacts min ~4 batches, bodies min ~32, target
            // ~4*sizingWorkers). Resets every stage's completionCount/syncIndex and
            // every block's syncIndex. Returns the active (non-empty) color count C.
            int BuildStages(std::uint32_t awakeCount, std::uint32_t sizingWorkers);

            // Commit the SoA dp/dq onto the world position/angle (compound-COM
            // commit: integrate the COM along its inertial path, then reconstruct
            // the origin from new COM + new angle).
            void FinalizePositionsSoA(SolverContext& ctx);

            // Velocity bridge for the scalar joint passes: copy awake-dynamic
            // velocities SoA->world (before a joint pass) / world->SoA (after).
            // Skipped entirely when there are no joints (the hot-path stays pure
            // SoA). dp/dq are NOT touched (joints do not read them).
            void SyncVelToWorld(SolverContext& ctx);
            void SyncVelFromWorld(SolverContext& ctx);

            // Solve the overflow (un-colorable) refs SEQUENTIALLY, width-1 scalar,
            // over the BodyStateStore AoS rows -- one constraint at a time so they are scatter-
            // safe even though they may share bodies. The WIDTH-1 SEQUENTIAL twin of
            // the lane-wide SimdSolve::WarmStart/SolveNormalAndFriction/ApplyRestitution
            // passes (ContactConstraintSimd.hpp), reading/writing the SAME SoA -- the
            // two MUST stay numerically in lockstep (see SoftStep.cpp's LOCKSTEP note).
            void OverflowWarmStart(SolverContext& ctx);
            void OverflowSolve(SolverContext& ctx, Real h, bool useBias);
            void OverflowRestitution(SolverContext& ctx);

            // ---- SIMD solve state (reused across steps) --------------------
            //
            // m_bodyState: AoS packed velocity + TGS dp/dq (BodyStateStore: 32-byte
            //   aligned rows), indexed by dense solverIndex (Phase C, Task 2), sized
            //   to solverCount (Task 3 dropped the "+1" dummy slot for the null-index
            //   branch -- a read-only-B/padding lane packs kNullBodyIndex and the
            //   gather injects a zero identity; see ContactConstraintSimd::Build). The
            //   lane-wide solve gathers/scatters through it via .data(); world<->AoS
            //   sync happens at the Step boundary + around joint passes.
            // m_colorRefs/m_overflowRefs: per-step bucketing of the emitted
            //   constraints by their PERSISTENT contact color (Phase C, Task 5; the
            //   per-step greedy recolor + its m_edges/m_coloring scratch are gone).
            //   m_colorRefs[k] holds the ctx.contacts indices whose color == k (a
            //   valid coloring -> within a color no two constraints share an awake
            //   dynamic body, so each is one scatter-safe lane-wide batch);
            //   m_overflowRefs holds the kInvalidColor indices (an overflow contact
            //   that found no free color at create, OR a span) -> the scalar tail.
            // m_colorBatches: per-color SoA batch lists built from m_colorRefs[k],
            //   kept per color so the overflow scalar pass composes after.
            BodyStateStore                  m_bodyState;
            std::array<std::vector<std::uint32_t>, kColorCount> m_colorRefs;
            std::vector<std::uint32_t>      m_overflowRefs;
            std::vector<std::vector<ContactConstraintSimd>> m_colorBatches;

            // Gap 1.1 (solver-MT scaling): the flat per-step stage list + its block
            // pool, reused across steps (resize-not-realloc -> zero steady-state
            // alloc). m_stages holds one SolverStage per non-empty color per colored
            // type + the two body stages; each stage's `blocks` points into a
            // contiguous slice of m_blocks. SolverWorker walks m_stages in D1's exact
            // substep order, so the stage runner is byte-identical to the per-color
            // ParallelFor it replaces. SolverStage/SolverBlock carry std::atomic
            // control words (claim/completion), copied relaxed only during the single-
            // threaded rebuild (see SolverStages.hpp).
            std::vector<SolverStage>        m_stages;
            std::vector<SolverBlock>        m_blocks;
        };

    } // namespace Physics
} // namespace Manifold2D
