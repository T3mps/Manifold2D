#pragma once

// 2D EPA (Expanding Polytope Algorithm) for the Manifold2D physics narrowphase
// (Physics v2 EPA workstream, Task 2).
//
// EPA resolves the EXACT minimum-penetration data for two OVERLAPPING convex
// cores. It is seeded by the terminal origin-enclosing simplex GJK produces on
// overlap (GjkOriginSimplex, Task 1) and expands a convex polytope of
// Minkowski-difference boundary points until the closest polytope edge to the
// origin stops moving. The converged edge's outward normal + origin-distance
// are the penetration axis + depth; projecting the origin onto that edge and
// interpolating the carried A/B support witnesses gives the witness pair.
//
// Reference: standard 2D EPA (Box2D v3 / dyn4j Epa.java / Bullet btGjkEpa 2D
// analog). This is the deep-overlap counterpart to GJK distance: GJK gives the
// closest features when SEPARATED; EPA gives the penetration when OVERLAPPING.
//
// CORES: as with GJK, EPA operates on convex CORES (radii NOT applied). The
// returned `depth` is the CORE penetration; the caller adds rA + rB. Witnesses
// are on the cores' surfaces; the caller pulls them in/out by the radii.
//
// NORMAL CONVENTION: `normal` points B -> A -- the direction to push A out of
// B -- matching ShapeDistance / Collide / Manifold across the narrowphase.
//
// PRECISION: the internal polytope is carried in f64 (matching the GJK simplex
// precision pattern); the EpaResult is narrowed to Real (f32) on return.
//
// DETERMINISM: fixed max iterations + max polytope size + fixed epsilon;
// deterministic closest-edge selection (smallest origin-distance, lowest edge
// index tie-break); no wall-clock; no heap (fixed stack arrays).
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only
// (Gjk.hpp, PhysicsTypes.hpp). No SDL3/NVRHI/Batcher2D/ImGui/Astra. Compiles
// /MD (Arcane.dll) and pure-compute (Arcane-only, /MD).

#include <Manifold2D/Physics/PhysicsTypes.hpp>

// Only the NarrowphaseTrace* parameter type is named here (an incomplete type
// suffices for a pointer); the full definition is needed only in Epa.cpp, which
// includes NarrowphaseTrace.hpp directly to push_back snapshots.
namespace Manifold2D { namespace Physics { struct NarrowphaseTrace; } }

namespace Manifold2D
{
    namespace Physics
    {
        // ----------------------------------------------------------------
        // EpaResult: exact minimum-penetration data for two overlapping cores.
        // ----------------------------------------------------------------
        //
        //   ok      : false if the inputs were not actually overlapping (caller
        //             error -- GjkOriginSimplex reported enclosesOrigin==false)
        //             or EPA failed to converge within caps -> caller falls
        //             back to MPR (later task).
        //   normal  : unit penetration axis, B -> A (the direction to push A
        //             out of B).
        //   depth   : penetration depth along `normal` (>= 0). CORE depth --
        //             the caller adds rA + rB.
        //   witnessA: witness point on A's core surface (radius NOT applied;
        //             caller adds rA along the normal).
        //   witnessB: witness point on B's core surface (radius NOT applied;
        //             caller adds rB along the normal).
        struct EpaResult
        {
            bool ok = false;
            Vec2 normal{ Real(0), Real(0) };
            Real depth = Real(0);
            Vec2 witnessA{ Real(0), Real(0) };
            Vec2 witnessB{ Real(0), Real(0) };
        };

        // ----------------------------------------------------------------
        // Epa: exact deep-overlap penetration via EPA on two world cores.
        // ----------------------------------------------------------------
        //
        // va[0..na-1] / vb[0..nb-1] : world-space convex core vertices (radii
        //   NOT applied -- the caller rotated each shape's local core verts into
        //   world space before calling, same input contract as GjkDistanceCore /
        //   GjkOriginSimplex).
        //
        // Returns EpaResult. ok==false when the cores were not actually
        // overlapping or EPA exceeded its caps -- the caller falls back to MPR.
        //
        // Deterministic: fixed max iterations + max polytope size + fixed
        // epsilon; no wall-clock; no heap.
        //
        // trace (debug-viz, OPT-IN): when NON-null, each expansion iteration
        // appends the current polytope + the chosen closest edge as a
        // PolytopeSnapshot to trace->epaSnapshots. When null (the Step-path
        // default) NOTHING is recorded and the computation is byte-identical --
        // the recorder only reads already-computed values behind `if (trace)`.
        [[nodiscard]] EpaResult Epa(const Vec2* va, int na,
                                    const Vec2* vb, int nb,
                                    NarrowphaseTrace* trace = nullptr);

    } // namespace Physics
} // namespace Manifold2D
