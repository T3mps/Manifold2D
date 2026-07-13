#pragma once

// Unified narrowphase entry for Arcane Physics v2 (Phase A, Task 3).
//
// Collide(a, xfA, b, xfB, speculativeMargin) -> Manifold
//
// This is the one rotation-aware narrowphase entry that replaces the old
// kind-dispatched CollidePolygons / CollideRoundPolygon / GJK-speculative
// split.  It was introduced additively at Task 3, wired into GenerateContacts
// at Task 5, and became the SOLE narrowphase path once T7 dropped the last
// CollideShapes caller.  T8 then strangled the dead old paths entirely
// (Dispatch/Specialized/Sat and Manifold.cpp's CollidePolygons are gone).
//
// ALGORITHM (Box2D v3 b2CollidePolygons lineage):
//   1. Rotate each shape's unified core verts (Shape::verts) into world space
//      using RotateInto(xf). Radii come from Shape::radius.
//   2. Run GjkDistanceCore on the two world cores -> core distance + closest
//      points + feature indices (featureA, featureB).
//   3. SEPARATED / SPECULATIVE:
//        coreDist >= rA+rB+speculativeMargin -> empty manifold.
//        coreDist in [rA+rB, rA+rB+margin]   -> 1 speculative point,
//          separation = coreDist - rA - rB (negative gap), normal = B->A,
//          id from GJK feature pair.
//   4. OVERLAPPING (coreDist == 0) -- SAT reference-face clip path:
//        Find the reference face via SAT over both shapes' CCW edge normals
//        (minimum-penetration axis; tie-break: lowest edge index wins).
//        Clip the incident edge against the two side planes of the reference
//        face; keep clipped points that project inside the reference face.
//        Emit up to 2 contact points with:
//          separation  = signed penetration depth per point (positive = overlap).
//          normal      = reference face's outward normal, oriented B->A.
//          id          = packed stable feature key: (refIsA:1b, refEdge, incVtx).
//   5. ROUND FAST PATH: circle/capsule cores (1 or 2 verts) use the
//        segment-distance path instead of the clip path. The core witness
//        pair from GJK gives up to 2 contact points offset by rA+rB.
//
// STABLE FEATURE IDS:
//   ManifoldPoint::id encodes the contacting feature pair as a stable uint32:
//     bits [31]    = refIsA  (1 if the reference face is on shape A, 0 if B)
//     bits [30:16] = refEdgeIdx (15 bits, index into the reference shape's verts)
//     bits [15: 0] = incFeature (vertex/edge index on the incident shape)
//   For the round path the feature pair comes from GjkDistanceCore::featureA/B.
//   This is the same packing as Box2D v3 b2MakeId(a,b) (two 8-bit fields)
//   extended to the wider index ranges this engine supports.
//
// NORMAL CONVENTION:
//   Manifold::normal and ManifoldPoint::normal both point FROM B TOWARD A --
//   "push A out of B". Matching the existing Manifold.hpp doc.
//   separation > 0 = penetrating; < 0 = speculative gap.
//
// SAT TIE-BREAK:
//   When two edge axes give equal minimum penetration the LOWEST edge INDEX
//   wins (deterministic; documented rule).
//
// DETERMINISM:
//   No wall-clock, no fast-math (/fp:precise everywhere).  Iteration order is
//   fixed (edge index ascending). Zero per-call heap (stack scratch only).
//
// PRESENTATION-FREE + dual CRT:
//   Geometry::Vec2 + std + sibling Physics headers only. Compiles /MD (Arcane.dll) and
//   pure-compute (Arcane-only, /MD).

#include <Manifold2D/Physics/Narrowphase/Manifold.hpp>
#include <Manifold2D/Physics/Narrowphase/NarrowphaseTrace.hpp>
#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        // The unified rotation-aware narrowphase entry (v2, Task 3).
        //
        // Returns a Manifold with 0, 1, or 2 contact points.
        // speculativeMargin: a gap in [0, speculativeMargin) produces a
        //   speculative contact with negative separation (pre-impact warning).
        //   Pass Real(0) for hard-contact-only behaviour.
        //
        // trace (debug-viz Slice B, OPT-IN): when NON-null, Collide records the
        //   intermediate narrowphase geometry (SAT axes, GJK/EPA/MPR snapshots,
        //   the final manifold + the two world shapes/transforms) into *trace at
        //   the natural points the algorithm already computes them. When null
        //   (the DEFAULT, taken by every Step-path call) NOTHING is recorded and
        //   the computation is byte-identical to a trace-free build -- the
        //   recorder only reads already-computed values behind `if (trace)`
        //   guards; it never alters the math or control flow. Only
        //   PhysicsWorld::DebugCollide (a side-effect-free re-run) passes a
        //   non-null pointer.
        [[nodiscard]] Manifold Collide(const Shape& a, const Transform& xfA,
                                       const Shape& b, const Transform& xfB,
                                       Real speculativeMargin = Real(0),
                                       NarrowphaseTrace* trace = nullptr);

    } // namespace Physics
} // namespace Manifold2D
