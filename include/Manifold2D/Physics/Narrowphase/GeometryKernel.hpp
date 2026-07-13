#pragma once

// Shared pure geometry kernels for the Manifold2D 2D physics narrowphase (M6).
//
// PORT NOTE: a faithful port of the pure, coordinate-agnostic kernels in
// Client/src/physics/Geometry.lua. The Lua module was the behavioral reference;
// the v2 gate is the analytic Physics tests (the physics_oracle geometry.json
// bit-match gate was retired in Physics v2 Phase A).
//
// Convention (matching the Lua source): y-down screen space, no rotation.
// Normals point FROM the second/static shape TOWARD the first/queried shape --
// i.e. the direction that pushes the queried shape OUT. Polygons are flat
// world-vertex arrays (Vec2*, count), convex, ANY winding (outward edge
// normals are resolved against the centroid, so the result is winding-agnostic
// -- it does NOT depend on whether MakePolygon reordered the verts to CCW).
//
// DRY: these kernels are the single source of truth for the shared geometry
// utilities (AabbOverlap, AabbToCorners, ClosestOnSegment, Depenetrate, etc.)
// consumed by PhysicsWorld/Queries/CharacterController. Narrowphase contact
// generation itself now lives in the unified Collide (Collide.cpp); the M6-era
// poly/round/SAT split (Manifold.cpp/Specialized.cpp/Sat.hpp) was retired in
// Physics v2 Phase A.
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.
// No SDL3/NVRHI/Batcher2D/ImGui, no C++23-only features (this module is also
// Arcane-only, /MD. namespace Manifold2D::Physics.

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        // ----------------------------------------------------------------
        // Hit: the (hit, normal, depth) tuple the Lua kernels return.
        // ----------------------------------------------------------------
        //
        // hit    : the shapes overlap (false => clear; normal/depth carry no
        //          meaning and are left at zero).
        // normal : unit push-out normal. For circle*/capsule* kernels it points
        //          FROM the polygon/other shape TOWARD the queried circle
        //          (push the circle out), matching the Lua convention.
        // depth  : penetration depth (positive when hit), the Lua's `depth`.
        struct Hit
        {
            bool hit   = false;
            Vec2 normal{ Real(0), Real(0) };
            Real depth = Real(0);
        };

        // ----------------------------------------------------------------
        // Boundary: closestOnPolyBoundary result.
        // ----------------------------------------------------------------
        //
        // point  : closest point on the polygon BOUNDARY to the query point.
        // dist   : distance from the query point to `point` (>= 0).
        // normal : the supporting edge's OUTWARD unit normal (resolved against
        //          the polygon centroid so it always points away from the
        //          interior, independent of winding).
        struct Boundary
        {
            Vec2 point{ Real(0), Real(0) };
            Real dist = Real(0);
            Vec2 normal{ Real(0), Real(0) };
        };

        // ----------------------------------------------------------------
        // ClosestSegSeg result: the closest pair of points (p on seg P,
        // q on seg Q). PORT of Geometry.closestSegSeg.
        // ----------------------------------------------------------------
        struct SegSeg
        {
            Vec2 p{ Real(0), Real(0) };
            Vec2 q{ Real(0), Real(0) };
        };

        // Expand an axis-aligned box to its 4 world corners in the canonical
        // order (min,min)(max,min)(max,max)(min,max) -- the Lua aabbPoly order
        // used by the narrowphase/CC/casts. `out` must hold 4 Vec2.
        // This is the single source of truth for AABB-to-corner expansion,
        // shared by CharacterController (Depenetrate), PhysicsWorld (ShapeCast),
        // and WorldPoly's AABB branch to prevent corner-order drift.
        // (Aabb2 = Aabb from Broadphase.hpp; we use Aabb here to avoid
        // pulling a broadphase header into the narrowphase layer.)
        void AabbToCorners(const Aabb& box, Vec2 out[4]) noexcept;

        // LEGACY translate-only poly/aabb world-vert expansion (port of
        // Manifold.lua worldPoly). NOTE: this helper is NO LONGER on the live
        // narrowphase path -- the unified rotation-aware Collide builds world
        // cores via RotateInto (Collide.cpp), so WorldPoly is now UNUSED by the
        // engine and is a candidate for removal; it is retained only for its
        // canonical AABB corner order (shared with AabbToCorners / the GJK
        // ShapeCastPoly span order). The AABB branch emits the four corners in
        // the Lua's order:
        //   (x-hw,y-hh),(x+hw,y-hh),(x+hw,y+hh),(x-hw,y+hh).
        // `out` must have room for at least kMaxPolyVerts vertices. Returns the
        // vertex count written. The shape MUST be a Polygon or Aabb (callers
        // route round shapes elsewhere).
        [[nodiscard]] int WorldPoly(const Shape& s, const Transform& xf,
                                    Vec2* out);

        // PORT of Geometry.pointInPoly(px, py, verts): winding-agnostic
        // convex containment via consistent edge-cross signs. A point exactly
        // on an edge (cross == 0) does NOT flip the verdict (matches the Lua).
        [[nodiscard]] bool PointInPoly(const Vec2& p, const Vec2* verts, int n);

        // PORT of Geometry.polyCentroid(verts): the arithmetic mean of the
        // vertices (NOT the area centroid). closestOnPolyBoundary uses this to
        // orient the outward edge normal.
        [[nodiscard]] Vec2 PolyCentroid(const Vec2* verts, int n);

        // PORT of Geometry.closestSegSeg(p1,p2, q1,q2): the closest pair of
        // points between segments P(p1->p2) and Q(q1->q2). Exact for
        // non-degenerate segments (endpoint candidates + the proper-crossing
        // 2x2 solve). Returns the intersection point in both fields when the
        // segments properly cross.
        [[nodiscard]] SegSeg ClosestSegSeg(const Vec2& p1, const Vec2& p2,
                                           const Vec2& q1, const Vec2& q2);

        // PORT of Geometry.closestOnPolyBoundary(px,py,verts): the closest
        // point on the polygon boundary to p, the distance, and the supporting
        // edge's outward normal (resolved against the centroid).
        [[nodiscard]] Boundary ClosestOnPolyBoundary(const Vec2& p,
                                                     const Vec2* verts, int n);

        // PORT of Geometry.circleCircle(ax,ay,ar, bx,by,br): circle A queried
        // against circle B. Normal points from B toward A (push A out).
        // Degenerate (coincident centers) -> normal (1,0), the Lua fallback.
        [[nodiscard]] Hit CircleCircle(const Vec2& a, Real ar,
                                       const Vec2& b, Real br);

        // PORT of Geometry.circlePoly(cx,cy,r,verts): circle vs convex poly.
        // Normal pushes the circle out. Contained center -> push out along the
        // nearest edge's outward normal (depth = r + dist). Outside -> push
        // along the boundary-to-center direction (depth = r - dist). Center
        // exactly on the boundary (dist ~ 0) -> borrow the edge normal.
        [[nodiscard]] Hit CirclePoly(const Vec2& c, Real r,
                                     const Vec2* verts, int n);

        // PORT of Geometry.capsulePoly(ax,ay,bx,by,r,verts): capsule (segment
        // a-b inflated by r) vs convex poly. Endpoint-containment falls back to
        // CirclePoly at the contained endpoint; otherwise the closest pair
        // between the segment and each boundary edge (closestSegSeg) gives the
        // push-out. Normal pushes the capsule out.
        //
        // NOTE: CapsulePoly is the direct port of Geometry.lua:capsulePoly and
        // produces a SINGLE push-out Hit for the whole capsule. It is still LIVE
        // as a single-contact depenetration helper (CharacterController; the
        // Phase-B swept controller). It is INTENTIONALLY NOT the contact-manifold
        // path: solver contact generation routes ALL shapes through the unified
        // rotation-aware Collide (Collide.cpp), whose round fast-path can yield
        // TWO contact points with DIFFERENT per-point normals from a capsule.
        // So CapsulePoly (single Hit) and Collide's round path (full manifold)
        // are two algorithms for the same problem, used at different sites.
        [[nodiscard]] Hit CapsulePoly(const Vec2& a, const Vec2& b, Real r,
                                      const Vec2* verts, int n);

    } // namespace Physics
} // namespace Manifold2D
