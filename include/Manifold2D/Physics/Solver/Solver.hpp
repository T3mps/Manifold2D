#pragma once

// ISolver: the velocity/position constraint-solver interface (M6, Tasks P2.1-3).
//
// PORT + MODERNIZE: the Lua engine routed all dynamics response through a single
// swappable `solver` seam (PhysicsWorld.lua:181 `solver = SequentialImpulse`,
// called once per Step at line 389 `self.solver.solve(self, contacts, nC,
// joints, dt)`). The C++ engine keeps that single-entry seam: ISolver exposes
// ONE Solve(SolverContext&) entry that owns the whole per-Step pipeline. One
// implementation lives behind it today:
//   * SoftStep   -- the Box2D-v3 TGS Soft modernization centerpiece (P2.2),
//                   the world's only solver. The P2.3 Baumgarte PGS A/B oracle
//                   was retired 2026-07-03 (MKS P2); the seam stays so a future
//                   solver (an XPBD experiment, say) drops in behind ISolver
//                   without touching the world.
//
// HISTORY (P2.1 -> P2.3): the P2.1 scaffolding split the monolithic solve into
// six Box2D-v3 sub-step phases (PrepareContacts/PrepareJoints/SolveVelocity/
// Relax/ApplyRestitution/SolvePosition) as PURE virtuals. P2.2's SoftStep made
// those its OWN orchestration (it owns the sub-step loop ordering internally),
// so for the P2.3 A/B seam the six phase methods are DEMOTED to SoftStep-private
// helpers and ISolver is slimmed to the shared entry below. Each solver owns its
// whole pipeline (integrate velocities -> solve -> integrate positions); the
// world no longer drives a phase sequence. A future XPBD experiment drops in the
// same way -- it just implements Solve.
//
// INTEGRATION OWNERSHIP: since P2.2 the world's Step no longer integrates
// dynamic velocity/position inline (the old stage-1/stage-4 dynamic branches are
// gone). The installed solver OWNS dynamic integration: SoftStep folds it into
// its sub-step loop (gravity/damping per sub-step, position per sub-step). It
// consumes the world-generated raw ContactConstraint array
// (GenerateContacts is solver-AGNOSTIC: it produces geometry only -- normal,
// per-point {anchorA/B, baseSeparation, id}, friction, restitution, the cached
// invMass/invInertia of A + B -- NOT solver-specific solved data). Each solver
// computes its own effective masses / bias in its own prepare pass.
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.
// No SDL3/NVRHI/Batcher2D/ImGui. namespace Manifold2D::Physics, Core style.

#include <cstddef>
#include <cstdint>

#include <Manifold2D/Physics/Contact.hpp>   // kInvalidColor (persistent contact coloring, Phase C Task 5)
#include <Manifold2D/Physics/Narrowphase/NarrowphaseTrace.hpp>
#include <Manifold2D/Physics/PhysicsTypes.hpp>

namespace Mosaic { struct IWorkScheduler; }

namespace Manifold2D
{
    namespace Physics
    {
        class PhysicsWorld; // the SoA owner; the solver mutates velocities through it
        struct Manifold;    // Narrowphase/Manifold.hpp -- per-pair contact points
        struct Joint;       // Joints/Joint.hpp (P2.5) -- forward-declared so JointConstraint
                            // can hold a typed pointer without pulling the full definition.

        // ----------------------------------------------------------------
        // ContactConstraintPoint: one solvable contact point (P2.2 Soft Step).
        // ----------------------------------------------------------------
        //
        // Per Box2D v3 b2ContactConstraintPoint: the per-point state the TGS
        // Soft solver carries across the sub-step loop. Anchors are relative to
        // each body's CENTER at Prepare time; the solver re-derives the current
        // separation each sub-step from the accumulated position deltas (the TGS
        // "track per-body deltaPos/deltaRot, re-evaluate separation" scheme).
        struct ContactConstraintPoint
        {
            // Anchors from each body center to the contact point (Prepare-time).
            Vec2 anchorA{ Real(0), Real(0) };
            Vec2 anchorB{ Real(0), Real(0) };
            // Base separation captured at Prepare (manifold separation; positive
            // = penetration in this engine's convention -> stored as Box2D's
            // signed separation = -penetration so s>0 means a gap).
            Real baseSeparation = Real(0);
            // Effective masses along the normal + tangent (inverse-K scalars).
            Real normalMass  = Real(0);
            Real tangentMass = Real(0);
            // Accumulated impulses. Warm-start-seeded at emit from the persistent
            // Contact's manifold point (NOT a solver cache anymore); the world
            // writes the converged values back onto that Contact after Solve().
            Real normalImpulse  = Real(0);
            Real tangentImpulse = Real(0);
            // Relative normal velocity at Prepare (for the restitution pass).
            Real relativeVelocity = Real(0);
            // Stable warm-start key (the manifold point id).
            std::uint32_t id = 0;
        };

        // ----------------------------------------------------------------
        // ContactConstraint: one solvable contact pair (P2.2 Soft Step).
        // ----------------------------------------------------------------
        //
        // Built by PhysicsWorld's stage-4 contact generation (Part A) into a
        // world-owned pool (PhysicsWorld::m_contactConstraints) -> zero
        // steady-state alloc, exactly like the Lua self._contacts scratch.
        // bodyA is ALWAYS the dynamic body (the Lua "A must be dynamic"
        // orientation, PhysicsWorld.lua:377-378). bodyB may be dynamic,
        // kinematic, static, or a tile-span VIRTUAL fixture (invMassB =
        // invInertiaB = 0, bodyB = kInvalidSlot). The normal points from B
        // toward A (push A out of B), matching ManifoldPoint::normal.
        struct ContactConstraint
        {
            // Sentinel for sourceContactId below: this constraint has NO persistent
            // ContactPool home (a transient tile span), so warm-start impulses are
            // not written back to a pool Contact -- it cold-starts every step.
            static constexpr std::uint32_t kNoContact = 0xFFFFFFFFu;

            std::uint32_t bodyA = 0;            // SoA slot index (always dynamic)
            std::uint32_t bodyB = 0;            // SoA slot index, or kInvalidSlot for a span
            bool          bodyBIsBody = false;  // false -> tile-span virtual fixture

            // The persistent ContactPool id this constraint was emitted from
            // (EmitContactConstraints sets it for a pool contact); kNoContact for a
            // transient tile span (no warm-start home). Warm-start impulses on the
            // persistent Contact replaced the solver's old m_cache: emit seeds the
            // ContactConstraintPoint from the source Contact's manifold, and after
            // Solve() the world writes the accumulated impulses back onto that
            // Contact via this id. Travels with the constraint through the canonical
            // sort (it is a field on the struct, not a parallel array), so the
            // post-sort permutation does not desync it.
            std::uint32_t sourceContactId = kNoContact;

            // Cached inverse mass/inertia (copied at Prepare so the solver does
            // not re-touch the world SoA per sub-step iteration). For spans /
            // statics / kinematics these are 0 (immovable: push, not pushed).
            Real invMassA = Real(0), invInertiaA = Real(0);
            Real invMassB = Real(0), invInertiaB = Real(0);

            // Representative normal (B -> A) + its tangent (perp, = (-ny, nx)).
            Vec2 normal{ Real(0), Real(0) };

            // Combined material coefficients (set by GenerateContacts: friction
            // = geometric mean sqrt(fricA*fricB); restitution = max). Both
            // solvers consume these raw values.
            Real friction    = Real(0);
            Real restitution = Real(0);

            // Soft coefficients (b2MakeSoft from contact hertz/dampingRatio/h).
            // Filled by SoftStep::Prepare. GenerateContacts leaves them at the
            // neutral defaults (massScale=1, biasRate=impulseScale=0) so they are
            // solver-agnostic until a solver overwrites them.
            Real biasRate     = Real(0);
            Real massScale    = Real(1);
            Real impulseScale = Real(0);

            int pointCount = 0;
            ContactConstraintPoint points[2]{};

            // DISPLAY-ONLY narrowphase tag (debug-viz Slice A): copied from the
            // producing Manifold by GenerateContacts' emit. The solver NEVER
            // reads it -- it is a pure inspection field (determinism unaffected).
            NarrowphaseKind kind = NarrowphaseKind::Separated;

            // PERSISTENT CONTACT COLOR (Phase C, Stage 2, Task 5). Copied at emit
            // from the source persistent Contact's `color` (EmitContactConstraints).
            // The solver CONSUMES it: it buckets the emitted constraints by this
            // color (deleting the old per-step greedy recolor), feeding each color
            // as one scatter-safe lane-wide batch. kInvalidColor means "no color":
            // a transient tile span (kNoContact home) OR an overflow contact (the
            // source Contact found no free color at create) -> the scalar overflow
            // path. Travels with the constraint through the canonical sort (a struct
            // field, copied by the post-sort permutation), like sourceContactId.
            std::uint8_t color = kInvalidColor;
        };

        // ----------------------------------------------------------------
        // JointConstraint: one solvable joint (scaffolding).
        // ----------------------------------------------------------------
        //
        // P2.2 replaces this with the soft-constraint joint formulation
        // (Distance/Revolute/Prismatic/Weld/Mouse/Wheel/Motor). P2.1 keeps an
        // opaque handle so PrepareJoints/SolveVelocity have something to name.
        struct JointConstraint
        {
            // P2.2: joint kind + anchors + soft (hertz, dampingRatio) + state.
            // Typed pointer (forward-declared above) so P2.5's joint module (which
            // will define `struct Joint` in Physics/Joints/) slots in without a
            // struct-layout break. A different name in P2.5 is a trivial rename here.
            Joint* joint = nullptr;
        };

        // ----------------------------------------------------------------
        // SolverContext: everything a Soft Step solver needs for one Step.
        // ----------------------------------------------------------------
        //
        // Bundles the world (the SoA the solver reads/writes through), the
        // constraint arrays built this step, the timestep, and the sub-step
        // schedule. Passed by reference to every ISolver phase so the methods
        // stay parameter-light and the world owns all backing storage (no solver-
        // side allocation -> the zero-steady-state-alloc contract holds).
        //
        // PORT mapping: the Lua passed (world, contacts, nC, joints, dt) to
        // solver.solve. This is the same five things, plus the sub-step schedule
        // the Soft Step modernization adds (the Lua single-shot solve had no
        // sub-steps). counts are explicit (nContacts/nJoints) so the arrays can
        // be over-capacity pooled buffers, mirroring nC over self._contacts.
        struct SolverContext
        {
            PhysicsWorld* world = nullptr;

            // Constraint arrays for this step (world-owned, pooled). The *Count
            // fields are the live prefix length (the arrays may be larger).
            ContactConstraint* contacts = nullptr;
            std::uint32_t       contactCount = 0;
            JointConstraint*    joints = nullptr;
            std::uint32_t       jointCount = 0;

            // Timestep + Soft Step schedule.
            Real          dt = Real(0);            // full Step dt
            std::uint32_t substepCount = 4;        // dt split into this many sub-steps
            Real          invDt = Real(0);         // 1/dt (0 if dt==0)
            Real          subDt = Real(0);         // dt / substepCount
            Real          invSubDt = Real(0);      // 1/subDt (0 if subDt==0)

            // Gravity (read for soft-constraint bias terms that reference it).
            Vec2          gravity{ Real(0), Real(0) };

            // Task-parallelism seam (Phase D1). Always non-null when the solver runs
            // (PhysicsWorld resolves it to its serial default if none was injected).
            Mosaic::IWorkScheduler* executor = nullptr;
        };

        // ----------------------------------------------------------------
        // ISolver: the swappable constraint-solver contract.
        // ----------------------------------------------------------------
        //
        // ONE entry: Solve(ctx) runs the installed solver's WHOLE per-Step
        // pipeline (it owns dynamic integration + the constraint solve -- see the
        // INTEGRATION OWNERSHIP note up top). SoftStep stores the converged
        // impulses on the persistent Contact (the world seeds + writes them back),
        // so it keeps no cache at all (WarmStartCacheSize() returns 0). The
        // world holds a std::unique_ptr<ISolver> and calls Solve once per Step; it
        // does NOT drive a phase sequence (the six P2.1 phase pure-virtuals were
        // demoted to SoftStep-private helpers when the seam was slimmed). The one
        // impl today is SoftStep (Box2D-v3 TGS Soft); the seam stays for a future
        // solver.
        class ISolver
        {
        public:
            virtual ~ISolver() = default;

            // Advance the dynamics for one full Step: integrate dynamic
            // velocities, solve the contact constraints in `ctx`, integrate
            // dynamic positions, and persist warm-start state. The world owns the
            // ContactConstraint pool (ctx.contacts); the solver owns its own
            // effective-mass/bias prepare + its warm-start cache.
            virtual void Solve(SolverContext& ctx) = 0;

            // Drop a body's warm-start state when its slot is recycled
            // (RemoveBody) so a reused slot inherits no stale impulses. Default
            // no-op: a solver whose cache self-evicts by stamp need not act.
            virtual void DropBody(std::uint32_t slot) { (void)slot; }

            // Current warm-start cache entry count (test/inspection hook; the
            // harness asserts the cache stays bounded). Default 0 for a solver
            // that keeps no cache.
            [[nodiscard]] virtual std::size_t WarmStartCacheSize() const noexcept
            {
                return 0;
            }

            // Number of un-colorable (overflow) constraints the most recent Solve
            // spilled to its width-1 scalar tail (test/inspection hook). EXISTS so
            // the overflow-settle test can PROVE the spill path actually ran: the
            // SIMD solve graph-colors contacts and lane-solves the first kColorCount
            // colors, spilling any excess (a body that is an endpoint of more than
            // kColorCount dynamic-dynamic contacts) to the scalar OverflowSolve. A
            // test that drops > kColorCount disks on one hub asserts this is > 0 so a
            // future coloring change (raising kColorCount, a different spill rule)
            // that stops the scene overflowing fails LOUD instead of silently no
            // longer exercising the overflow code. Default 0: a solver with no
            // overflow concept (a scalar PGS colors nothing, spills nothing)
            // inherits this and reports 0.
            [[nodiscard]] virtual std::size_t LastOverflowCount() const noexcept
            {
                return 0;
            }
        };

    } // namespace Physics
} // namespace Manifold2D
