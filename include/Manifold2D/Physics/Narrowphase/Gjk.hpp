#pragma once

// 2D GJK distance + closest points, surface distance, and a conservative-
// advancement shape cast for the Manifold2D 2D physics narrowphase (M6, Task P1.4).
//
// PORT NOTE: a faithful port of Client/src/physics/GJK.lua (the distance core,
// shapeDistance, shapePolyDistance) plus the conservative-advancement `advance`
// primitive from Client/src/physics/Cast.lua. The Lua module was the behavioral
// reference; the v2 gate is the analytic PhysicsGjkV2Test (the physics_oracle
// gjk.json bit-match gate was retired in Physics v2 Phase A). GJK serves
// DISTANCE + closest features (and the CCD/cast primitive); penetration
// manifolds are now produced by the unified Collide (Narrowphase/Collide.cpp,
// GJK->SAT ref-face clip), which replaced the retired Sat/Specialized path.
//
// CORES: GJK operates on convex CORES with the radius subtracted afterward
// (a circle is a 1-vert point of radius r; a capsule is a 2-vert segment of
// radius r; an AABB is its 4 corners with radius 0; a polygon is its world
// verts with radius 0). BuildCore (Gjk.cpp) ports GJK.lua:buildCore. The
// AABB-core radius (0) differs from nothing here, but note the core layout
// matches GeometryKernel::WorldPoly's corner order so the two stay consistent.
//
// CONVENTION (matching the Lua source): y-down screen space, no rotation.
// ShapeDistance's normal points FROM the second shape (B) TOWARD the first (A)
// -- i.e. the direction that pushes A out -- matching the Geometry-kernel and
// SAT convention used across the narrowphase. When the cores intersect the
// distance is 0 and the witnesses/normal are meaningless (returned as zeros),
// exactly as GJK.lua does.
//
// SHAPECAST (Cast.lua:advance, lines 20-29): a DECOUPLED conservative-advance
// primitive, NOT coupled to the world. Translation-only: advancing by
// dist/|translation| can never overshoot a convex obstacle. TOL = 0.05,
// MAX_ITER = 32 (ported verbatim). The world-level cast that collects tile /
// body candidates (Cast.shapeCast / rayVsBody) is P1.9 and will reuse this
// primitive; here we expose only the per-obstacle TOI.
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.
// No SDL3/NVRHI/Batcher2D/ImGui, no C++23-only features (this module is also
// Arcane-only, /MD. namespace Manifold2D::Physics.

#include <cstdint>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        // ----------------------------------------------------------------
        // GjkResult: GJK.distance return tuple.
        // ----------------------------------------------------------------
        //
        // distance : Euclidean distance between the two convex cores. 0 means
        //            the cores intersect -- the witnesses are then meaningless
        //            and returned as zeros (matching GJK.lua).
        // witnessA : closest point on core A.
        // witnessB : closest point on core B.
        struct GjkResult
        {
            Real distance = Real(0);
            Vec2 witnessA{ Real(0), Real(0) };
            Vec2 witnessB{ Real(0), Real(0) };
        };

        // ----------------------------------------------------------------
        // GjkCoreResult: v2 core-based GJK return (Task 2, Phase A).
        // ----------------------------------------------------------------
        //
        // distance : Euclidean distance between the two convex CORES (radius
        //            NOT subtracted -- callers add rA+rB themselves). 0 means
        //            cores intersect; EPA (Task 4) resolves penetration.
        // pointA   : closest point on core A (in world space).
        // pointB   : closest point on core B (in world space).
        // featureA : witness vertex/edge on A encoded as:
        //              vertex: plain index into the input va[] array (< kEdgeMask).
        //              edge  : kEdgeMask | (idxLo << 16) | idxHi
        //                      where idxLo and idxHi are the two simplex vertex
        //                      indices (input array) in ascending order.
        // featureB : same encoding for the B core.
        //
        // Feature semantics (port of Box2D v3 b2SimplexVertex.indexA/indexB):
        //   The simplex produced by GJK has 1 or 2 vertices after convergence
        //   (a 2D GJK never needs more than 3, but the closest-feature is always
        //   1 or 2 because a 3-simplex means the origin is INSIDE -- overlap).
        //   * n==1 simplex: closest feature is a single vertex; feature = idx.
        //   * n==2 simplex: closest feature is an edge; feature = edge encoding.
        //   When cores overlap (distance==0) features are 0 (meaningless).
        //
        // This is the v2 entry (rotation-aware -- caller rotates core verts into
        // world space before calling). The M6 GjkDistance / ShapeDistance /
        // ShapeCast path is RETAINED unchanged; this is an ADDITIVE new entry.
        struct GjkCoreResult
        {
            Real     distance = Real(0);
            Vec2     pointA{ Real(0), Real(0) };
            Vec2     pointB{ Real(0), Real(0) };
            uint32_t featureA = 0u;
            uint32_t featureB = 0u;

            // Sentinel bit: set in featureA/featureB when the closest feature
            // is an EDGE (two input indices packed as (idxLo<<16)|idxHi).
            // Clear when the feature is a single VERTEX (the value IS the index).
            //
            // Field layout when kEdgeMask is SET:
            //   bit  31    = 1 (kEdgeMask sentinel)
            //   bits [30:16] = idxLo (15 bits); decode: lo = (id >> 16) & 0x7FFFu
            //   bits [15: 0] = idxHi (16 bits); decode: hi = id & 0xFFFFu
            //   Invariant: idxLo <= idxHi (canonical ascending order).
            //
            // When kEdgeMask is CLEAR the full value IS the vertex index:
            //   vertex check: !(id & kEdgeMask)  -- id is the simplex vertex index.
            static constexpr uint32_t kEdgeMask = 0x8000'0000u;
        };

        // ----------------------------------------------------------------
        // ShapeDistanceResult: GJK.shapeDistance / shapePolyDistance tuple.
        // ----------------------------------------------------------------
        //
        // distance : SURFACE distance (core distance minus both radii).
        //            Negative => the shapes overlap within their radii.
        // witnessA : surface witness on A (core witness pulled in by A's radius
        //            along the normal).
        // witnessB : surface witness on B (core witness pushed out by B's
        //            radius along the normal).
        // normal   : unit normal from B toward A (push A out). Zeros when the
        //            cores intersect (distance computed as 0 by GJK).
        struct ShapeDistanceResult
        {
            Real distance = Real(0);
            Vec2 witnessA{ Real(0), Real(0) };
            Vec2 witnessB{ Real(0), Real(0) };
            Vec2 normal{ Real(0), Real(0) };
        };

        // ----------------------------------------------------------------
        // ShapeCastResult: conservative-advancement shape-cast tuple.
        // ----------------------------------------------------------------
        //
        // hit      : the swept shape reached within TOL of the obstacle before
        //            the end of its travel (t in [0,1]).
        // t        : parametric time of impact along the cast translation. The
        //            shape CENTER at TOI is start + translation * t.
        // normal   : surface normal at the impact (from the obstacle toward the
        //            moving shape -- the push-back direction).
        // distance : surface distance at the stopping point (< TOL when hit).
        struct ShapeCastResult
        {
            bool hit = false;
            Real t = Real(0);
            Vec2 normal{ Real(0), Real(0) };
            Real distance = Real(0);
        };

        // ----------------------------------------------------------------
        // Coupled to kLinearSlop per Box2D v3.1.1 (distance.c:610-614,641: target =
        // max(linearSlop, totalRadius - linearSlop), tolerance = 0.25*linearSlop --
        // effective near-zero-radius hit band = 1.25*linearSlop). Arcane compares a
        // single flat tolerance against the POST-RADII surface distance (Gjk.cpp
        // Advance), so the flat equivalent of Box2D's band is used; the radius-aware
        // target split is a recorded parity note, not adopted (MKS P4).
        // ----------------------------------------------------------------
        inline constexpr Real kShapeCastTol = Real(1.25) * kLinearSlop; // = 0.00625
        inline constexpr int  kShapeCastMaxIter = 32;

        // ----------------------------------------------------------------
        // GjkDistance: distance + closest points between two convex CORES.
        // ----------------------------------------------------------------
        //
        // PORT of GJK.distance(va, vb): va/vb are world-space vertex spans
        // (point=1, segment=2, polygon=n) -- the raw cores, radii NOT applied.
        // The simplex loop ports the Lua exactly (initial support along (1,0),
        // up to 64 iterations, the n==1/2/3 closest-point-to-origin reduction,
        // barycentric witness interpolation, the (s.d)-(p.d) < 1e-9 convergence
        // test, and the origin-reached early-out). The arithmetic is carried in
        // f64 internally to track the Lua's f64 distance core (the f32 core
        // inputs are widened on read); results are returned as f32. distance==0
        // means the cores intersect (witnesses returned as zeros).
        [[nodiscard]] GjkResult GjkDistance(const Vec2* va, int na,
                                            const Vec2* vb, int nb);

        // ----------------------------------------------------------------
        // GjkDistanceCore: v2 core-based GJK with closest-feature output.
        // ----------------------------------------------------------------
        //
        // ROTATION-AWARE entry (Phase A, Task 2). Operates on world-space core
        // vertex spans produced by the caller rotating each shape's local core
        // verts by their body transform. Radius is NOT applied here -- the
        // returned `distance` is core-to-core; the caller subtracts rA+rB to
        // get surface distance. Rotation enters ONLY via the input vertex arrays;
        // GJK itself is rotation-agnostic.
        //
        // va[0..na-1] : world-space core vertices of shape A (1=circle, 2=capsule,
        //               4=aabb, n=polygon).
        // vb[0..nb-1] : world-space core vertices of shape B.
        //
        // Returns GjkCoreResult with distance + closest points + feature indices.
        // Feature encoding: see GjkCoreResult::kEdgeMask.
        //
        // Precision: carries the same f64 internal simplex as GjkDistance
        // (preserving the M6 robustness guarantee). Max 64 iterations (same cap).
        // Deterministic: fixed iteration order + no wall-clock.
        //
        // ADDITIVE: the M6 GjkDistance / ShapeDistance / ShapeCast path is
        // UNCHANGED; this entry exists ALONGSIDE it. Tasks 5/6 migrate callers.
        [[nodiscard]] GjkCoreResult GjkDistanceCore(const Vec2* va, int na,
                                                    const Vec2* vb, int nb);

        // ----------------------------------------------------------------
        // MdSupport: Minkowski-difference support over two world cores (the EPA
        // / MPR seam, Physics v2 EPA workstream Task 1).
        // ----------------------------------------------------------------
        //
        // The MD boundary point farthest along a search direction `dir`, plus
        // the support points that produced it on A's core (along +dir) and B's
        // core (along -dir) and their input vertex indices.
        //
        // md == support(A,+dir) - support(B,-dir). pa is the witness on A's
        // core, pb the witness on B's core (both world space). ia/ib index into
        // the va[]/vb[] arrays passed to SupportMinkowski.
        //
        // NOTE: Gjk.cpp's anonymous namespace carries an f64 internal MdSupport
        // (the simplex math runs in double, mirroring GJK.lua). This PUBLIC type
        // is the f32 (Real) narrowing of that internal struct; SupportMinkowski
        // calls the internal f64 SupportMd and narrows on return.
        struct MdSupport
        {
            Vec2 md{ Real(0), Real(0) };  // support on A minus support on B
            Vec2 pa{ Real(0), Real(0) };  // support point on A's core (+dir)
            Vec2 pb{ Real(0), Real(0) };  // support point on B's core (-dir)
            int  ia = 0;                  // A vertex index
            int  ib = 0;                  // B vertex index
        };

        // ----------------------------------------------------------------
        // SupportMinkowski: the MD support over two world cores along `dir`.
        // ----------------------------------------------------------------
        //
        // Argmax over A along +dir, over B along -dir; md = pa - pb. Same math
        // as Gjk.cpp's internal SupportMd (carried in f64, narrowed to Real on
        // return). ADDITIVE: existing GJK entries unchanged.
        [[nodiscard]] MdSupport SupportMinkowski(const Vec2* va, int na,
                                                 const Vec2* vb, int nb,
                                                 Vec2 dir);

        // ----------------------------------------------------------------
        // GjkSimplexVertex / GjkSimplex: the terminal GJK simplex (EPA seed,
        // Physics v2 EPA workstream Task 1).
        // ----------------------------------------------------------------
        //
        // GjkSimplexVertex carries one simplex vertex: the MD point plus the A/B
        // support pair (wa on A, wb on B) and their input vertex indices that
        // produced it.
        struct GjkSimplexVertex
        {
            Vec2 md{ Real(0), Real(0) };
            Vec2 wa{ Real(0), Real(0) };
            Vec2 wb{ Real(0), Real(0) };
            int  ia = 0;
            int  ib = 0;
        };

        // The terminal GJK simplex. enclosesOrigin == true means the cores
        // OVERLAP and v[0..2] form a triangle containing the origin (EPA's seed,
        // count == 3). enclosesOrigin == false means the cores are separated --
        // count (1..3) + v describe the closest simplex.
        struct GjkSimplex
        {
            GjkSimplexVertex v[3];
            int  count = 0;             // 1..3
            bool enclosesOrigin = false;
        };

        // ----------------------------------------------------------------
        // GjkOriginSimplex: run GJK on two world cores and return the terminal
        // simplex (EPA seed).
        // ----------------------------------------------------------------
        //
        // Runs the SAME simplex loop as GjkDistanceCore (same f64 internal
        // simplex, same 64-iteration cap, same convergence / overlap tests). On
        // OVERLAP the n==3 origin-inside branch returns the 3 terminal MD verts
        // as a triangle enclosing the origin (enclosesOrigin == true, count ==
        // 3). On convergence WITHOUT enclosing the origin it returns the closest
        // simplex (count = 1/2/3 as built, enclosesOrigin == false).
        //
        // Carries f64 internally, narrows to Real on return. Deterministic
        // (fixed cap + iteration order, no heap, no wall-clock). ADDITIVE:
        // GjkDistanceCore and all existing GJK outputs are UNCHANGED.
        [[nodiscard]] GjkSimplex GjkOriginSimplex(const Vec2* va, int na,
                                                  const Vec2* vb, int nb);

        // ----------------------------------------------------------------
        // BuildCore: PORT of GJK.lua:buildCore. Writes the convex core verts of
        // `s` at world transform `xf` (ROTATION + translation: each local core
        // vert is rotated by xf.rotation, then offset by xf.position -- T7 made
        // this rotation-aware so ShapeDistance/ShapeCast honor body angle) into
        // `out` and returns the core RADIUS (circle/capsule -> s.radius;
        // aabb/polygon -> 0).
        // Returns the vertex count via `outCount`. `out` must have room for at
        // least kMaxPolyVerts vertices.
        // ----------------------------------------------------------------
        [[nodiscard]] Real BuildCore(const Shape& s, const Transform& xf,
                                     Vec2* out, int& outCount);

        // ----------------------------------------------------------------
        // ShapeDistance: PORT of GJK.shapeDistance. Surface distance between two
        // shapes at world transforms. Subtracts both radii; the normal points
        // from B toward A. Negative distance => overlapping within radii.
        // ----------------------------------------------------------------
        [[nodiscard]] ShapeDistanceResult ShapeDistance(const Shape& a,
                                                        const Transform& xfA,
                                                        const Shape& b,
                                                        const Transform& xfB);

        // ----------------------------------------------------------------
        // ShapePolyDistance: PORT of GJK.shapePolyDistance. Surface distance
        // between shape `a` and a RAW world-space polygon span (e.g. a tile
        // span); the poly's radius is 0. Same return shape as ShapeDistance.
        // ----------------------------------------------------------------
        [[nodiscard]] ShapeDistanceResult ShapePolyDistance(const Shape& a,
                                                            const Transform& xfA,
                                                            const Vec2* poly,
                                                            int n);

        // ----------------------------------------------------------------
        // ShapeCast: conservative-advancement TOI of a moving shape against a
        // single static obstacle shape. PORT of Cast.lua:advance using
        // ShapeDistance as the per-step distance function. `start` is the
        // moving shape's transform at t=0; `translation` is its displacement
        // over the cast. Returns the center-at-TOI parametric t with the
        // surface normal/distance at the stopping point. No hit (zero/near-zero
        // translation, or the obstacle is never reached within travel) -> hit
        // == false. Deterministic: fixed iteration count, no wall-clock.
        // ----------------------------------------------------------------
        [[nodiscard]] ShapeCastResult ShapeCast(const Shape& moving,
                                                const Transform& start,
                                                const Vec2& translation,
                                                const Shape& obstacle,
                                                const Transform& obstacleXf);

        // ----------------------------------------------------------------
        // ShapeCastPoly: ShapeCast against a RAW world-space polygon span
        // (ShapePolyDistance as the distance function). Same semantics.
        // ----------------------------------------------------------------
        [[nodiscard]] ShapeCastResult ShapeCastPoly(const Shape& moving,
                                                    const Transform& start,
                                                    const Vec2& translation,
                                                    const Vec2* poly, int n);

    } // namespace Physics
} // namespace Manifold2D
