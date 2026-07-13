// Shapes.cpp -- collider geometry + rotation-aware AABB + mass/inertia (v2).
//
// See Shapes.hpp for the v2 unified-core+radius contract. Changes vs M6:
//
//   MakeCircle / MakeCapsule / MakeAabb now populate the unified core verts
//   (and normals for Aabb) so that ComputeAABB can work uniformly over all
//   four kinds via rotation. MakePolygon is unchanged (already populated).
//
//   ComputeAABB is now ROTATION-AWARE (v2 Phase A, Task 1):
//     - Rotate each core vert by xf.rotation (2D rotation matrix).
//     - Bound the rotated verts (min/max x and y).
//     - Expand by radius on all sides.
//   This replaces the old translation-only per-kind switch.
//
//   ComputeMass is UNCHANGED from M6 (closed forms per kind; validated against
//   hand-derived analytic values). The existing test assertions still pass.
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.

#include <Manifold2D/Physics/Shapes.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Manifold2D/Physics/Math.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        // --------------------------------------------------------------------
        // ComputeAABB -- rotation-aware (v2 Phase A, Task 1).
        //
        // Algorithm:
        //   1. Build a 2D rotation matrix from xf.rotation: c = cos(theta),
        //      s = sin(theta). Then for a local vert (lx, ly):
        //        world_x = c*lx - s*ly + pos.x
        //        world_y = s*lx + c*ly + pos.y
        //   2. Walk the unified core verts (verts[]), rotate each, accumulate
        //      min/max.
        //   3. Expand the bounding box by `radius` on all sides.
        //
        // For circle (1 vert at origin) the rotation step is a no-op and the
        // result is [pos - r, pos + r] -- identical to the old formula.
        // For capsule at rotation 0 the two verts (-hl,0) and (+hl,0) give the
        // same result as the old `x-hl-r ... x+hl+r` formula.
        // For aabb at rotation 0 the four corner verts give the same result as
        // the old `x-hw ... x+hw, y-hh ... y+hh` formula.
        // So at zero rotation all results are identical to M6; the rotation path
        // is the new capability.
        // --------------------------------------------------------------------
        Aabb Shape::ComputeAABB(const Transform& xf) const
        {
            const Real cx = xf.position.x;
            const Real cy = xf.position.y;
            const Real c  = std::cos(xf.rotation);
            const Real s  = std::sin(xf.rotation);

            Real x0 =  std::numeric_limits<Real>::max();
            Real y0 =  std::numeric_limits<Real>::max();
            Real x1 = -std::numeric_limits<Real>::max();
            Real y1 = -std::numeric_limits<Real>::max();

            for (const Vec2& v : verts)
            {
                // Rotate the local vert by xf.rotation, then translate.
                const Real wx = c * v.x - s * v.y + cx;
                const Real wy = s * v.x + c * v.y + cy;
                if (wx < x0) x0 = wx;
                if (wx > x1) x1 = wx;
                if (wy < y0) y0 = wy;
                if (wy > y1) y1 = wy;
            }

            // Expand by the round-inflation radius.
            return Aabb{
                Vec2(x0 - radius, y0 - radius),
                Vec2(x1 + radius, y1 + radius)
            };
        }

        // --------------------------------------------------------------------
        // ComputeMass -- standard Box2D-style mass properties. NEW (no Lua
        // oracle). Inertia is about the shape's centroid. Per-kind derivation:
        // --------------------------------------------------------------------
        MassData Shape::ComputeMass(Real density) const
        {
            MassData md{};

            switch (kind)
            {
                case ShapeKind::Circle:
                {
                    // Solid disc, radius r.
                    //   area     = pi * r^2
                    //   mass     = density * area
                    //   centroid = (0,0)  (local origin)
                    //   I_centroid = mass * (r^2 / 2)   [solid disc]
                    const Real r    = radius;
                    const Real area = kPi * r * r;
                    md.mass     = density * area;
                    md.centroid = Vec2(Real(0), Real(0));
                    md.inertia  = md.mass * (Real(0.5) * r * r);
                    return md;
                }

                case ShapeKind::Aabb:
                {
                    // Solid box, full dimensions w = 2*halfW, h = 2*halfH.
                    //   area     = w * h
                    //   mass     = density * area
                    //   centroid = (0,0)
                    //   I_centroid = mass * (w^2 + h^2) / 12   [solid rectangle]
                    const Real w    = Real(2) * halfW;
                    const Real h    = Real(2) * halfH;
                    const Real area = w * h;
                    md.mass     = density * area;
                    md.centroid = Vec2(Real(0), Real(0));
                    md.inertia  = md.mass * (w * w + h * h) / Real(12);
                    return md;
                }

                case ShapeKind::Capsule:
                {
                    // Horizontal capsule: a rectangle (the segment swept by
                    // the radius) capped by two half-discs (= one full circle).
                    // Box2D v3 b2ComputeCapsuleMass decomposition, with the two
                    // segment endpoints at p1=(-halfLen,0), p2=(+halfLen,0):
                    //   length = |p2 - p1| = 2 * halfLen   (segment length)
                    //   h      = 0.5 * length = halfLen    (half-segment)
                    //
                    // Mass:
                    //   rectangle: dims length x (2r) -> area = 2 * r * length
                    //   end caps : one full circle      -> area = pi * r^2
                    //   total mass = density * (2*r*length + pi*r^2)
                    //   centroid = midpoint = (0,0)
                    //
                    // Inertia about the centroid (sum of the two parts):
                    //   box part  : rectangle (w=length, h=2r) about its center
                    //               I_box = boxMass * (w^2 + h^2)/12
                    //                     = boxMass * (length^2 + (2r)^2)/12
                    //                     = boxMass * (4*r^2 + length^2)/12
                    //   cap part  : the two half-discs together carry circleMass
                    //               and sit offset +/- h from the center. Each
                    //               half-disc's own centroid is lc = 4r/(3*pi)
                    //               from the flat edge. Box2D's closed form for
                    //               the combined two-cap inertia about the
                    //               capsule center:
                    //               I_circle = circleMass *
                    //                          (0.5*r^2 + h^2 + 2*h*lc)
                    //     (0.5*r^2 is the full-disc self inertia; h^2 is the
                    //      parallel-axis shift of the cap mass to +/-h; 2*h*lc
                    //      corrects for the half-disc centroid offset.)
                    //   I_centroid = I_box + I_circle
                    const Real r       = radius;
                    const Real length  = Real(2) * halfLen;
                    const Real h       = Real(0.5) * length; // == halfLen
                    const Real rr      = r * r;
                    const Real ll      = length * length;

                    const Real circleMass = density * (kPi * rr);
                    const Real boxMass    = density * (Real(2) * r * length);

                    md.mass     = circleMass + boxMass;
                    md.centroid = Vec2(Real(0), Real(0));

                    const Real lc = (Real(4) * r) / (Real(3) * kPi);
                    const Real circleInertia =
                        circleMass * (Real(0.5) * rr + h * h + Real(2) * h * lc);
                    const Real boxInertia =
                        boxMass * (Real(4) * rr + ll) / Real(12);

                    md.inertia = circleInertia + boxInertia;
                    return md;
                }

                case ShapeKind::Polygon:
                {
                    // Standard Box2D b2ComputePolygonMass (radius = 0). Walk
                    // the polygon as a fan of triangles from a reference vertex
                    // (verts[0]); accumulate signed area, area-weighted
                    // centroid, and the second moment of inertia, then translate
                    // the inertia to the computed centroid via parallel axis.
                    //
                    // verts are stored CCW (MakePolygon normalizes), so the
                    // signed area is positive.
                    const std::size_t n = verts.size();
                    if (n < 3)
                    {
                        return md; // degenerate; defensive (factory rejects)
                    }

                    // Reference origin: verts[0]. Subtracting it improves
                    // numerical conditioning (Box2D does the same) without
                    // changing the result -- we add it back to the centroid.
                    const Vec2 ref = verts[0];

                    Vec2 center(Real(0), Real(0)); // area-weighted, ref-relative
                    Real area = Real(0);
                    Real inertia = Real(0);        // about ref
                    const Real inv3 = Real(1) / Real(3);

                    for (std::size_t i = 1; i + 1 < n; ++i)
                    {
                        // Triangle (ref, verts[i], verts[i+1]) as edge vectors
                        // e1, e2 from ref.
                        const Vec2 e1 = verts[i]     - ref;
                        const Vec2 e2 = verts[i + 1] - ref;

                        const Real d = Math::Cross2(e1, e2);
                        const Real triArea = Real(0.5) * d;
                        area += triArea;

                        // Triangle centroid (ref-relative) = (e1 + e2)/3.
                        center += triArea * inv3 * (e1 + e2);

                        // Second moment of the triangle about ref (Box2D form).
                        const Real intx2 =
                            e1.x * e1.x + e2.x * e1.x + e2.x * e2.x;
                        const Real inty2 =
                            e1.y * e1.y + e2.y * e1.y + e2.y * e2.y;
                        inertia += (Real(0.25) * inv3 * d) * (intx2 + inty2);
                    }

                    md.mass = density * area;

                    // Centroid relative to ref, then back to local space.
                    if (area > Real(0))
                    {
                        center /= area;
                    }
                    md.centroid = ref + center;

                    // inertia is about ref; shift to centroid via parallel
                    // axis: I_c = I_ref - m * |center|^2 (center is the centroid
                    // measured from ref). Box2D scales by density here.
                    md.inertia = density * inertia;
                    md.inertia -= md.mass * (center.x * center.x +
                                             center.y * center.y);
                    return md;
                }
            }
            // unreachable: all ShapeKind cases handled
            return {};
        }

        // --------------------------------------------------------------------
        // Factory functions
        // --------------------------------------------------------------------

        Shape MakeCircle(Real r)
        {
            Shape s;
            s.kind   = ShapeKind::Circle;
            s.radius = r;
            // Unified core: 1 vertex at the local origin.
            // ComputeAABB rotates this point (a no-op for the origin) then
            // expands by radius, giving the correct disc AABB at any rotation.
            s.verts = { Vec2(Real(0), Real(0)) };
            // Circles have no edges, so normals stays empty.
            // polyCentroid is the local origin (default Vec2(0,0) is correct).
            return s;
        }

        Shape MakeCapsule(Real halfLen, Real r)
        {
            Shape s;
            s.kind    = ShapeKind::Capsule;
            s.halfLen = halfLen;
            s.radius  = r;
            // Unified core: 2 verts at the segment endpoints (-halfLen,0) and
            // (+halfLen,0). ComputeAABB rotates both endpoints by xf.rotation
            // then expands by radius, giving the geometrically-correct capsule
            // AABB under arbitrary orientation.
            s.verts = {
                Vec2(-halfLen, Real(0)),
                Vec2(+halfLen, Real(0)),
            };
            // Capsule has no polygon edges (the shape is the Minkowski sum of
            // the segment with a disc); normals stays empty.
            // polyCentroid = midpoint = (0,0) (default is correct).
            return s;
        }

        Shape MakeAabb(Real hw, Real hh)
        {
            Shape s;
            s.kind  = ShapeKind::Aabb;
            s.halfW = hw;
            s.halfH = hh;
            // Unified core: 4 corner verts in CCW winding (standard for a
            // y-down coordinate space where positive y is downward and CCW is
            // the winding that gives a positive shoelace area). The box
            // corners in local space are:
            //   v0 = (-hw, -hh)   bottom-left
            //   v1 = (+hw, -hh)   bottom-right
            //   v2 = (+hw, +hh)   top-right
            //   v3 = (-hw, +hh)   top-left
            // Edge normals (outward, one per edge i -> i+1):
            //   e0 v0->v1: right edge along +x bottom  -> normal (0,-1)
            //   e1 v1->v2: right edge along +y         -> normal (+1,0)
            //   e2 v2->v3: top edge along -x            -> normal (0,+1)
            //   e3 v3->v0: left edge along -y           -> normal (-1,0)
            // (These match the outward-normal convention used by MakePolygon:
            // Perp(b-a) = (dy,-dx) for CCW-in-y-down winding.)
            s.verts = {
                Vec2(-hw, -hh),
                Vec2(+hw, -hh),
                Vec2(+hw, +hh),
                Vec2(-hw, +hh),
            };
            s.normals = {
                Vec2(Real(0),  Real(-1)),  // bottom face: outward = -y
                Vec2(Real(1),  Real(0)),   // right face:  outward = +x
                Vec2(Real(0),  Real(1)),   // top face:    outward = +y
                Vec2(Real(-1), Real(0)),   // left face:   outward = -x
            };
            // polyCentroid = (0,0) by symmetry (default is correct).
            return s;
        }

        Shape MakePolygon(const std::vector<Vec2>& verts)
        {
            // PORT: shapes.polygon assert -- need 3..kMaxPolyVerts (x,y) verts.
            // The Lua counted a flat {x,y,...} array (>= 6 and even and
            // <= MAX_POLY_VERTS*2); here the input is already a Vec2 list so
            // the equivalent bound is on the vertex count directly.
            if (verts.size() < 3 || verts.size() > kMaxPolyVerts)
            {
                throw std::invalid_argument(
                    "MakePolygon: need 3..kMaxPolyVerts vertices");
            }

            Shape s;
            s.kind  = ShapeKind::Polygon;
            s.verts = verts;

            // Normalize to CCW winding. Signed area via the shoelace sum: with
            // y-down screen space (the Lua/renderer convention) a CCW polygon
            // has POSITIVE signed area under cross2 accumulation. If negative,
            // the input is CW -- reverse it so the baked normals point outward
            // for the SAT port.
            Real signedArea2 = Real(0);
            const std::size_t n = s.verts.size();
            for (std::size_t i = 0; i < n; ++i)
            {
                const Vec2& a = s.verts[i];
                const Vec2& b = s.verts[(i + 1) % n];
                signedArea2 += Math::Cross2(a, b);
            }
            if (signedArea2 < Real(0))
            {
                std::reverse(s.verts.begin(), s.verts.end());
            }

            // Outward unit edge normals (CCW winding). For edge a->b the
            // outward normal is Perp(b - a) = (dy, -dx) normalized -- the
            // right-hand perpendicular, which points away from the interior for
            // CCW-in-y-down winding. (Same Perp the Geometry SAT axes use.)
            s.normals.clear();
            s.normals.reserve(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                const Vec2& a = s.verts[i];
                const Vec2& b = s.verts[(i + 1) % n];
                const Vec2 edge = b - a;
                Vec2 nrm = Math::Perp(edge); // (edge.y, -edge.x)
                const Real len = std::sqrt(nrm.x * nrm.x + nrm.y * nrm.y);
                if (len > Real(0))
                {
                    nrm /= len;
                }
                s.normals.push_back(nrm);
            }

            // Bake the area centroid (cached for the SAT port + diagnostics).
            s.polyCentroid = s.ComputeMass(Real(1)).centroid;

            return s;
        }

    } // namespace Physics
} // namespace Manifold2D
