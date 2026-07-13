#pragma once

// 2D MPR (Minkowski Portal Refinement / XenoCollide) for the Manifold2D physics
// narrowphase (Physics v2 EPA workstream, Task 4).
//
// MPR is a FAST single-point deep-overlap solver over two OVERLAPPING convex
// cores. It is EPA's convergence FALLBACK in Collide.cpp: when EPA fails to
// converge within its caps (rare, degenerate input), MPR produces a single
// deep-overlap contact instead. Unlike EPA (which expands a full polytope to
// the EXACT minimum-penetration edge), MPR refines a single "portal" (a 2D
// portal is one Minkowski-difference boundary segment) until it lies on the
// MD boundary along the origin ray -- one contact point, one normal, fast.
//
// Reference: Gary Snethen, XenoCollide (GDC 2008); dyn4j
// MinkowskiPenetrationSolver (2D). The 2D specialization: v0 = interior MD
// point (MD of the two core centroids), build a portal segment (v1, v2) that
// the origin ray (v0 -> origin) crosses, then iteratively replace the portal
// vertex on the wrong side with a fresh support along the portal's outward
// normal until the support stops advancing past the portal edge.
//
// CORES: as with GJK / EPA, MPR operates on convex CORES (radii NOT applied).
// The returned `depth` is the CORE penetration; the caller adds rA + rB. The
// `point` is on the MD boundary in world space (the witness midpoint of the
// portal edge nearest the origin).
//
// NORMAL CONVENTION: `normal` points B -> A -- the direction to push A out of
// B -- matching ShapeDistance / Collide / Epa across the narrowphase. The
// converged portal edge's outward normal points AWAY from the interior v0
// (i.e. away from the origin into the MD boundary = the "push B out of A"
// direction over D = A - B); the returned `normal` is its NEGATION, the same
// flip Epa applies. Pinned by the test (circle-core-in-box -> (+1,0)).
//
// PRECISION: the portal math is carried in f64 (matching the GJK / EPA
// precision pattern); MprResult is narrowed to Real (f32) on return.
//
// DETERMINISM: fixed max refinement iterations (kMprMaxIters) + fixed epsilon
// (kMprEps) + a fixed perp-sign / bracket decision; no wall-clock; no heap
// (fixed stack scratch).
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only
// (Gjk.hpp, PhysicsTypes.hpp). No SDL3/NVRHI/Batcher2D/ImGui/Astra. Compiles
// /MD (Arcane.dll) and pure-compute (Arcane-only, /MD).

#include <Manifold2D/Physics/PhysicsTypes.hpp>

// Only the NarrowphaseTrace* parameter type is named here (an incomplete type
// suffices for a pointer); the full definition is needed only in Mpr.cpp, which
// includes NarrowphaseTrace.hpp directly to push_back snapshots.
namespace Manifold2D { namespace Physics { struct NarrowphaseTrace; } }

namespace Manifold2D
{
    namespace Physics
    {
        // ----------------------------------------------------------------
        // MprResult: a single-point deep-overlap contact for two convex cores.
        // ----------------------------------------------------------------
        //
        //   ok     : false if MPR failed to converge within caps (rare) or the
        //            cores are degenerate / not actually overlapping.
        //   normal : unit contact normal, B -> A.
        //   depth  : penetration depth along `normal` (>= 0). CORE depth -- the
        //            caller adds rA + rB.
        //   point  : single contact point on the MD boundary, in world space.
        struct MprResult
        {
            bool ok = false;
            Vec2 normal{ Real(0), Real(0) };
            Real depth = Real(0);
            Vec2 point{ Real(0), Real(0) };
        };

        // ----------------------------------------------------------------
        // Mpr: single-point deep overlap via MPR on two world cores.
        // ----------------------------------------------------------------
        //
        // va[0..na-1] / vb[0..nb-1] : world-space convex core vertices (radii
        //   NOT applied -- the caller rotated each shape's local core verts into
        //   world space before calling; same input contract as GjkDistanceCore /
        //   GjkOriginSimplex / Epa).
        //
        // Returns MprResult. ok==false when the cores are degenerate or MPR
        // exceeded its caps -- the caller emits no contact for that frame.
        //
        // Deterministic: fixed max refinement iterations + fixed epsilon; f64
        // internal; no wall-clock; no heap.
        //
        // trace (debug-viz, OPT-IN): when NON-null, each refinement iteration
        // appends the current portal (v0 interior seed + the closest edge
        // endpoints v1/v2 + the support ray direction) as an MprSnapshot to
        // trace->mprSnapshots. When null (the Step-path default) NOTHING is
        // recorded and the computation is byte-identical -- the recorder only
        // reads already-computed values behind `if (trace)`.
        [[nodiscard]] MprResult Mpr(const Vec2* va, int na,
                                    const Vec2* vb, int nb,
                                    NarrowphaseTrace* trace = nullptr);

    } // namespace Physics
} // namespace Manifold2D
