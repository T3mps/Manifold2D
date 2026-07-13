// GeometryKernel.cpp -- shared pure narrowphase kernels (port of Geometry.lua).
//
// See GeometryKernel.hpp for the contract. Every function is a faithful port of
// the corresponding Geometry.lua kernel (the physics_oracle geometry.json
// bit-match gate was retired in Physics v2 Phase A; the v2 gate is the analytic
// Physics tests). The epsilons (1e-9, 1e-12) are carried over verbatim from the
// Lua so the branch selection matches.
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.

#include <Manifold2D/Physics/Narrowphase/GeometryKernel.hpp>

#include <cmath>
#include <limits>

#include <Manifold2D/Physics/Math.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        namespace
        {
            // Lua math.huge sentinel for the "best so far" accumulators.
            constexpr Real kHuge = std::numeric_limits<Real>::infinity();
        } // namespace

        // --------------------------------------------------------------------
        // AabbToCorners -- single source of truth for AABB-to-4-corner
        // expansion in the canonical Lua aabbPoly order. Called by WorldPoly's
        // AABB branch, CharacterController::Depenetrate, and PhysicsWorld::
        // ShapeCast so corner order can never drift between the three sites.
        // --------------------------------------------------------------------
        void AabbToCorners(const Aabb& box, Vec2 out[4]) noexcept
        {
            out[0] = Vec2(box.min.x, box.min.y);
            out[1] = Vec2(box.max.x, box.min.y);
            out[2] = Vec2(box.max.x, box.max.y);
            out[3] = Vec2(box.min.x, box.max.y);
        }

        // --------------------------------------------------------------------
        // worldPoly -- expand a poly/aabb into world verts (Manifold.lua:19-39,
        // translation-only branch). AABB branch delegates to AabbToCorners so
        // corner order is guaranteed identical across all call sites.
        // --------------------------------------------------------------------
        int WorldPoly(const Shape& s, const Transform& xf, Vec2* out)
        {
            const Real x = xf.position.x;
            const Real y = xf.position.y;

            if (s.kind == ShapeKind::Aabb)
            {
                const Real hw = s.halfW;
                const Real hh = s.halfH;
                const Aabb box{
                    Vec2(x - hw, y - hh),
                    Vec2(x + hw, y + hh)
                };
                AabbToCorners(box, out);
                return 4;
            }

            // Polygon: translate the baked local verts (rotation identity).
            const int n = static_cast<int>(s.verts.size());
            for (int i = 0; i < n; ++i)
            {
                out[i] = Vec2(x + s.verts[i].x, y + s.verts[i].y);
            }
            return n;
        }

        // --------------------------------------------------------------------
        // pointInPoly -- winding-agnostic containment (Geometry.lua:52-65).
        // --------------------------------------------------------------------
        bool PointInPoly(const Vec2& p, const Vec2* verts, int n)
        {
            int sign = 0;
            for (int i = 0; i < n; ++i)
            {
                const int j = (i + 1 < n) ? (i + 1) : 0;
                const Real ex = verts[j].x - verts[i].x;
                const Real ey = verts[j].y - verts[i].y;
                const Real cross =
                    ex * (p.y - verts[i].y) - ey * (p.x - verts[i].x);
                if (cross != Real(0))
                {
                    const int s = (cross > Real(0)) ? 1 : -1;
                    if (sign == 0) sign = s;
                    else if (s != sign) return false;
                }
            }
            return true;
        }

        // --------------------------------------------------------------------
        // polyCentroid -- arithmetic mean of the verts (Geometry.lua:67-71).
        // --------------------------------------------------------------------
        Vec2 PolyCentroid(const Vec2* verts, int n)
        {
            Real cx = Real(0), cy = Real(0);
            for (int i = 0; i < n; ++i)
            {
                cx += verts[i].x;
                cy += verts[i].y;
            }
            const Real inv = (n > 0) ? (Real(1) / static_cast<Real>(n)) : Real(0);
            return Vec2(cx * inv, cy * inv);
        }

        // --------------------------------------------------------------------
        // closestSegSeg -- closest pair between two segments (Geometry.lua:19-50).
        // --------------------------------------------------------------------
        SegSeg ClosestSegSeg(const Vec2& p1, const Vec2& p2,
                             const Vec2& q1, const Vec2& q2)
        {
            Real bestDSq = kHuge; // SQUARED distance accumulator
            Vec2 bp{ Real(0), Real(0) }, bq{ Real(0), Real(0) };

            auto consider = [&](const Vec2& a, const Vec2& b)
            {
                const Real dx = a.x - b.x;
                const Real dy = a.y - b.y;
                const Real d = dx * dx + dy * dy;
                if (d < bestDSq) { bestDSq = d; bp = a; bq = b; }
            };

            // Each endpoint vs the other segment (point as p, closest as q),
            // matching the Lua's `consider(p, closest)` argument order so the
            // returned (p,q) pairing reproduces the oracle.
            Math::ClosestPoint c = Math::ClosestOnSegment(p1, q1, q2);
            consider(p1, c.point);
            c = Math::ClosestOnSegment(p2, q1, q2);
            consider(p2, c.point);
            c = Math::ClosestOnSegment(q1, p1, p2);
            consider(c.point, q1);
            c = Math::ClosestOnSegment(q2, p1, p2);
            consider(c.point, q2);

            // Proper crossing: endpoints alone miss it. Solve the 2x2 system.
            const Real rx = p2.x - p1.x, ry = p2.y - p1.y;
            const Real sx = q2.x - q1.x, sy = q2.y - q1.y;
            const Real denom = rx * sy - ry * sx;
            if (std::fabs(denom) > Real(1e-12))
            {
                const Real t = ((q1.x - p1.x) * sy - (q1.y - p1.y) * sx) / denom;
                const Real u = ((q1.x - p1.x) * ry - (q1.y - p1.y) * rx) / denom;
                if (t >= Real(0) && t <= Real(1) && u >= Real(0) && u <= Real(1))
                {
                    const Vec2 ix(p1.x + rx * t, p1.y + ry * t);
                    return SegSeg{ ix, ix };
                }
            }
            return SegSeg{ bp, bq };
        }

        // --------------------------------------------------------------------
        // closestOnPolyBoundary -- closest boundary point + outward edge normal
        // resolved vs the centroid (Geometry.lua:75-98).
        // --------------------------------------------------------------------
        Boundary ClosestOnPolyBoundary(const Vec2& p, const Vec2* verts, int n)
        {
            Real bestDSq = kHuge; // SQUARED distance accumulator
            Vec2 bp{ Real(0), Real(0) };
            Vec2 bn{ Real(0), Real(0) };
            const Vec2 cc = PolyCentroid(verts, n);

            for (int i = 0; i < n; ++i)
            {
                const int j = (i + 1 < n) ? (i + 1) : 0;
                const Vec2 a = verts[i];
                const Vec2 q = verts[j];
                const Math::ClosestPoint c = Math::ClosestOnSegment(p, a, q);
                const Real dx = p.x - c.point.x;
                const Real dy = p.y - c.point.y;
                const Real d = dx * dx + dy * dy;
                if (d < bestDSq)
                {
                    bestDSq = d;
                    bp = c.point;
                    // outward edge normal
                    const Real ex = q.x - a.x;
                    const Real ey = q.y - a.y;
                    Real nx = ey, ny = -ex;
                    const Real len = std::sqrt(nx * nx + ny * ny);
                    if (len > Real(0)) { nx /= len; ny /= len; }
                    const Real mx = (a.x + q.x) * Real(0.5) - cc.x;
                    const Real my = (a.y + q.y) * Real(0.5) - cc.y;
                    if (nx * mx + ny * my < Real(0)) { nx = -nx; ny = -ny; }
                    bn = Vec2(nx, ny);
                }
            }
            return Boundary{ bp, std::sqrt(bestDSq), bn };
        }

        // --------------------------------------------------------------------
        // circleCircle (Geometry.lua:148-155).
        // --------------------------------------------------------------------
        Hit CircleCircle(const Vec2& a, Real ar, const Vec2& b, Real br)
        {
            Hit h{};
            const Real dx = a.x - b.x;
            const Real dy = a.y - b.y;
            const Real d = std::sqrt(dx * dx + dy * dy);
            const Real rsum = ar + br;
            if (d >= rsum) return h; // clear
            h.hit = true;
            if (d > Real(1e-9))
            {
                h.normal = Vec2(dx / d, dy / d);
                h.depth = rsum - d;
            }
            else
            {
                h.normal = Vec2(Real(1), Real(0));
                h.depth = rsum;
            }
            return h;
        }

        // --------------------------------------------------------------------
        // circlePoly (Geometry.lua:101-112).
        // --------------------------------------------------------------------
        Hit CirclePoly(const Vec2& c, Real r, const Vec2* verts, int n)
        {
            Hit h{};
            const Boundary b = ClosestOnPolyBoundary(c, verts, n);
            if (PointInPoly(c, verts, n))
            {
                // contained: push out along the nearest edge's outward normal
                h.hit = true;
                h.normal = b.normal;
                h.depth = r + b.dist;
                return h;
            }
            if (b.dist >= r) return h; // clear
            h.hit = true;
            if (b.dist > Real(1e-9))
            {
                h.normal = Vec2((c.x - b.point.x) / b.dist,
                                (c.y - b.point.y) / b.dist);
                h.depth = r - b.dist;
            }
            else
            {
                // degenerate: center on boundary
                h.normal = b.normal;
                h.depth = r;
            }
            return h;
        }

        // --------------------------------------------------------------------
        // capsulePoly (Geometry.lua:115-145).
        // --------------------------------------------------------------------
        Hit CapsulePoly(const Vec2& a, const Vec2& b, Real r,
                        const Vec2* verts, int n)
        {
            // Containment (either endpoint inside): treat as a deep circle at
            // the contained endpoint.
            if (PointInPoly(a, verts, n))
            {
                return CirclePoly(a, r, verts, n);
            }
            if (PointInPoly(b, verts, n))
            {
                return CirclePoly(b, r, verts, n);
            }

            // Shallow case: closest pair between the segment and the boundary
            // edges; keep the minimum-distance edge.
            Real bestD = kHuge;
            Real snx = Real(0), sny = Real(0);
            for (int i = 0; i < n; ++i)
            {
                const int j = (i + 1 < n) ? (i + 1) : 0;
                const SegSeg cs = ClosestSegSeg(a, b, verts[i], verts[j]);
                const Real dx = cs.p.x - cs.q.x;
                const Real dy = cs.p.y - cs.q.y;
                const Real d = std::sqrt(dx * dx + dy * dy);
                if (d < bestD)
                {
                    bestD = d;
                    // The Lua assigns snx,sny ONLY when d > 1e-9; when the
                    // closest pair is (near-)coincident it leaves them at their
                    // prior value (initially 0,0). We reproduce that exactly --
                    // do NOT zero them in the else branch.
                    if (d > Real(1e-9)) { snx = dx / d; sny = dy / d; }
                }
            }

            Hit h{};
            if (bestD >= r) return h; // clear
            if (snx == Real(0) && sny == Real(0))
            {
                // segment touches boundary exactly: borrow the edge normal
                const Boundary bd = ClosestOnPolyBoundary(a, verts, n);
                snx = bd.normal.x;
                sny = bd.normal.y;
            }
            h.hit = true;
            h.normal = Vec2(snx, sny);
            h.depth = r - bestD;
            return h;
        }

    } // namespace Physics
} // namespace Manifold2D
