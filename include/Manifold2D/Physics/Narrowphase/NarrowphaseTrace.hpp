#pragma once

// NarrowphaseTrace: the opt-in per-call narrowphase recorder (debug-viz Slice B
// inspector seam, Task 3).
//
// The NarrowphaseKind tag this file once held (Task 1) was extracted to the
// sibling NarrowphaseKind.hpp to break an include cycle: Manifold.hpp needs the
// ENUM (Manifold::kind) and this file's NarrowphaseTrace struct needs the full
// Manifold (its `manifold` member). NarrowphaseKind.hpp carries the enum alone;
// Manifold.hpp includes it; this file includes both. Use
// <Manifold2D/Physics/Narrowphase/NarrowphaseTrace.hpp> for the recorder (it
// re-exports the enum via NarrowphaseKind.hpp), or include NarrowphaseKind.hpp
// directly for the tag only.
//
// This file carries the richer per-call NarrowphaseTrace recorder -- an OPT-IN
// structure that records the intermediate geometry one Collide() invocation
// computes (SAT candidate axes, GJK simplices, EPA polytope snapshots, MPR
// portals) so the editor's physics inspector can draw the algorithm
// step-by-step.
//
// CRITICAL CONTRACT: the recorder is OPT-IN. Every simulation Step-path call to
// Collide passes a nullptr recorder (the trailing parameter defaults null), so
// the narrowphase is BYTE-IDENTICAL to a trace-free build on the Step path --
// the recorder only appends already-computed values behind `if (trace)` guards
// at the natural points; it NEVER changes the math or control flow. Only
// PhysicsWorld::DebugCollide (a side-effect-free re-run) passes a non-null
// recorder. Determinism is therefore unaffected.
//
// PRESENTATION-FREE + C++20-clean: std + Geometry::Vec2 + sibling Physics headers
// (NarrowphaseKind + Manifold + Shapes + PhysicsTypes) only. All POD / Vec2 /
// std::vector. No SDL3/NVRHI/Batcher2D/ImGui. Compiles both /MD (Arcane.dll)
// and pure-compute (Arcane-only, /MD).

#include <cstdint>
#include <vector>

#include <Manifold2D/Physics/Narrowphase/Manifold.hpp>
#include <Manifold2D/Physics/Narrowphase/NarrowphaseKind.hpp>
#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        // ----------------------------------------------------------------
        // SatAxis: one candidate separating axis tested during the poly-poly
        // SAT reference-face search.
        // ----------------------------------------------------------------
        //
        //   dir  : the candidate axis (a reference-shape edge outward normal,
        //          world space, unit).
        //   minRef/maxRef : projection interval of the REFERENCE shape's core
        //                   verts onto `dir`.
        //   minInc/maxInc : projection interval of the INCIDENT shape's core
        //                   verts onto `dir`.
        //   chosen        : true for the axis the SAT search selected as the
        //                   minimum-penetration reference axis (its normal equals
        //                   the manifold normal, up to the B->A orientation flip).
        //
        // The SAT reference-face search is inherently reference/incident: which
        // PHYSICAL shape is the reference flips between the two SAT passes (A as
        // reference, then B as reference), so these intervals are labelled by
        // SEARCH ROLE (reference/incident), not by shape A/B.
        struct SatAxis
        {
            Vec2 dir{ Real(0), Real(0) };
            Real minRef = Real(0);
            Real maxRef = Real(0);
            Real minInc = Real(0);
            Real maxInc = Real(0);
            bool chosen = false;
        };

        // ----------------------------------------------------------------
        // SimplexSnapshot: one GJK iteration's simplex, in MINKOWSKI space.
        // ----------------------------------------------------------------
        //
        //   verts/count   : the 1..3 current simplex vertices (MD points).
        //   searchDir     : the search direction for the next support query.
        //   containsOrigin: true once the simplex encloses the origin (overlap).
        struct SimplexSnapshot
        {
            Vec2 verts[3]{};
            int  count = 0;
            Vec2 searchDir{ Real(0), Real(0) };
            bool containsOrigin = false;
        };

        // ----------------------------------------------------------------
        // PolytopeSnapshot: one EPA iteration's expanding polytope, in
        // MINKOWSKI space.
        // ----------------------------------------------------------------
        //
        //   verts      : the ordered (CCW) polytope MD boundary vertices.
        //   edgeA/edgeB: the closest-edge endpoint indices into `verts`.
        //   edgeNormal : the closest edge's outward normal (MD space, unit).
        //   edgeDist   : the closest edge's distance from the origin.
        struct PolytopeSnapshot
        {
            std::vector<Vec2> verts;
            int  edgeA = 0;
            int  edgeB = 0;
            Vec2 edgeNormal{ Real(0), Real(0) };
            Real edgeDist = Real(0);
        };

        // ----------------------------------------------------------------
        // MprSnapshot: one MPR iteration's portal triangle, in MINKOWSKI space.
        // ----------------------------------------------------------------
        //
        //   v0     : the interior MD seed point (centroid difference).
        //   v1/v2  : the current closest-edge portal endpoints (MD points).
        //   rayDir : the support search direction this iteration (the closest
        //            edge's outward normal).
        struct MprSnapshot
        {
            Vec2 v0{ Real(0), Real(0) };
            Vec2 v1{ Real(0), Real(0) };
            Vec2 v2{ Real(0), Real(0) };
            Vec2 rayDir{ Real(0), Real(0) };
        };

        // ----------------------------------------------------------------
        // NarrowphaseTrace: the OPT-IN recorder for one Collide() invocation.
        // ----------------------------------------------------------------
        //
        // Holds the intermediate geometry of a single narrowphase call so the
        // debug inspector can draw each algorithm step (the per-iteration
        // vectors are what the inspector's step slider scrubs). Populated only
        // when a non-null NarrowphaseTrace* is threaded through Collide (i.e.
        // only by PhysicsWorld::DebugCollide); the Step path passes nullptr and
        // records nothing.
        //
        // FIELDS:
        //   kind     : the narrowphase path that produced the manifold (mirrors
        //              Manifold::kind).
        //   manifold : the FINAL world-space manifold this call resolved.
        //   shapeA/B + xfA/xfB : the two world-space shapes + transforms (so the
        //              inspector inset can draw the colliding shapes).
        //   satAxes        : poly-poly SAT candidate axes (ordered; one chosen).
        //   gjkSnapshots   : per-iteration GJK simplices (Minkowski space).
        //   epaSnapshots   : per-iteration EPA polytopes (Minkowski space).
        //   mprSnapshots   : per-iteration MPR portals (Minkowski space).
        //
        // REUSE: Clear() resets the kind + all vectors (preserving capacity) so
        // a caller may reuse ONE instance per inspect call -- no per-call heap
        // churn after warmup.
        struct NarrowphaseTrace
        {
            NarrowphaseKind kind = NarrowphaseKind::Separated;

            Manifold  manifold{};

            // The two world-space shapes + transforms this call collided.
            Shape     shapeA{};
            Shape     shapeB{};
            Transform xfA{};
            Transform xfB{};

            std::vector<SatAxis>          satAxes;
            std::vector<SimplexSnapshot>  gjkSnapshots;
            std::vector<PolytopeSnapshot> epaSnapshots;
            std::vector<MprSnapshot>      mprSnapshots;

            // Reset the kind + clear all per-iteration vectors (capacity kept so
            // reuse across inspect calls allocates nothing after warmup).
            void Clear() noexcept
            {
                kind     = NarrowphaseKind::Separated;
                manifold = Manifold{};
                shapeA   = Shape{};
                shapeB   = Shape{};
                xfA      = Transform{};
                xfB      = Transform{};
                satAxes.clear();
                gjkSnapshots.clear();
                epaSnapshots.clear();
                mprSnapshots.clear();
            }
        };
    } // namespace Physics
} // namespace Manifold2D
