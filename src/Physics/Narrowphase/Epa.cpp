// Epa.cpp -- 2D EPA (Expanding Polytope Algorithm): exact deep-overlap
// penetration for two convex cores. See Epa.hpp for the contract.
//
// Standard 2D EPA (Box2D v3 / dyn4j Epa.java / Bullet btGjkEpa 2D analog):
//   1. Seed the polytope with the 3 GjkOriginSimplex MD verts, forced CCW.
//   2. Each iteration: find the polytope EDGE closest to the origin (smallest
//      dot(outwardNormal, edgeStart); lowest-index tie-break). Query the MD
//      support along that edge's outward normal. If the support is farther out
//      than the edge (by > kEpaEps), INSERT it splitting the edge and continue;
//      else CONVERGE on that edge.
//   3. The converged edge's outward normal + origin-distance are the MD
//      penetration axis + depth; projecting the origin onto the edge gives the
//      barycentric t to lerp the carried A/B support witnesses.
//
// NORMAL CONVENTION: the polytope is the Minkowski difference D = A - B (the
// support pairs carry md = wa - wb). For D, the closest-edge OUTWARD normal
// points away from the origin toward the nearest D boundary -- which is the
// "push B out of A" direction; the contract wants B -> A, so the returned
// `normal` is the NEGATION of the converged edge's outward normal. Pinned by
// the test: circle-core-in-box yields outward (-1,0) -> returned (+1,0).
//
// PRECISION: the polytope is carried in f64 (matching the GJK simplex pattern);
// the EpaResult is narrowed to Real on return.
//
// DETERMINISM: fixed caps (kEpaMaxIters / kEpaMaxVerts), fixed epsilon
// (kEpaEps), deterministic closest-edge selection, no wall-clock, no heap
// (fixed stack arrays sized to kEpaMaxVerts).
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.

#include <Manifold2D/Physics/Narrowphase/Epa.hpp>

#include <array>
#include <cmath>
#include <limits>

#include <Manifold2D/Physics/Narrowphase/Gjk.hpp>
#include <Manifold2D/Physics/Narrowphase/NarrowphaseTrace.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        namespace
        {
            // EPA tuning constants (named per the Task contract).
            //
            // kEpaEps  : progress epsilon. A support whose distance along an
            //            edge's outward normal exceeds the edge's origin-distance
            //            by MORE than this is "new" -> insert + continue. Within
            //            it -> converged. Sized for the f64 polytope at the world
            //            scales the engine uses (matches the GJK 1e-9-class
            //            convergence tolerance, loosened slightly so f64 support
            //            roundoff on a settled edge does not oscillate).
            constexpr double kEpaEps = 1e-7;
            // kEpaMaxIters : max expansion iterations (support insertions).
            constexpr int    kEpaMaxIters = 32;
            // kEpaMaxVerts : max polytope vertices (seed 3 + insertions).
            constexpr int    kEpaMaxVerts = 32;

            // f64 polytope vertex: the MD boundary point plus the A/B support
            // pair (wa on A, wb on B) the seed / expansion carried. The witness
            // interpolation lerps wa/wb along the converged edge.
            struct PolyVert
            {
                double mx = 0.0, my = 0.0;   // MD point (wa - wb)
                double ax = 0.0, ay = 0.0;   // witness on A
                double bx = 0.0, by = 0.0;   // witness on B
            };
        } // namespace

        // --------------------------------------------------------------------
        // Epa -- the expanding-polytope loop over the Minkowski difference.
        // --------------------------------------------------------------------
        EpaResult Epa(const Vec2* va, int na, const Vec2* vb, int nb,
                      NarrowphaseTrace* trace)
        {
            EpaResult out;

            // Degenerate cores: SupportMinkowski guards na/nb<=0, but EPA needs
            // a real area MD, so bail to ok=false (caller falls back).
            if (na <= 0 || nb <= 0)
                return out;

            // --- 1. Seed from the GJK origin-enclosing simplex. ---
            const GjkSimplex seed = GjkOriginSimplex(va, na, vb, nb);
            if (!seed.enclosesOrigin || seed.count < 3)
            {
                // Cores were not actually overlapping (caller error) or the
                // seed could not form an origin-enclosing triangle.
                return out;
            }

            std::array<PolyVert, kEpaMaxVerts> poly{};
            int count = 3;
            for (int i = 0; i < 3; ++i)
            {
                poly[i].mx = static_cast<double>(seed.v[i].md.x);
                poly[i].my = static_cast<double>(seed.v[i].md.y);
                poly[i].ax = static_cast<double>(seed.v[i].wa.x);
                poly[i].ay = static_cast<double>(seed.v[i].wa.y);
                poly[i].bx = static_cast<double>(seed.v[i].wb.x);
                poly[i].by = static_cast<double>(seed.v[i].wb.y);
            }

            // Ensure CCW winding (signed area > 0). EPA's outward-normal math
            // (outward normal of CCW edge e is (e.y, -e.x)) assumes CCW; if the
            // seed triangle is CW, reverse it (swap verts 1 and 2).
            {
                const double x0 = poly[0].mx, y0 = poly[0].my;
                const double x1 = poly[1].mx, y1 = poly[1].my;
                const double x2 = poly[2].mx, y2 = poly[2].my;
                const double signedArea2 =
                    (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
                if (signedArea2 < 0.0)
                {
                    const PolyVert tmp = poly[1];
                    poly[1] = poly[2];
                    poly[2] = tmp;
                }
            }

            // --- 2-3. Expand until the closest edge stops moving. ---
            // On each iteration find the closest edge, query the support along
            // its outward normal, and either insert (split the edge) or converge.
            for (int iter = 0; iter < kEpaMaxIters; ++iter)
            {
                // Closest edge: smallest dot(outwardNormal, edgeStart). Outward
                // normal for CCW edge e=(p1-p0) is (e.y, -e.x), normalized. The
                // origin-distance along it is dot(unitOutward, p0). Tie-break:
                // lowest edge index (the loop keeps the FIRST strictly-smaller).
                int    bestEdge = 0;
                double bestDist = std::numeric_limits<double>::infinity();
                double bestNx = 0.0, bestNy = 0.0;
                for (int i = 0; i < count; ++i)
                {
                    const int j = (i + 1) % count;
                    const double ex = poly[j].mx - poly[i].mx;
                    const double ey = poly[j].my - poly[i].my;
                    // Outward normal (CCW): (e.y, -e.x).
                    double nx = ey;
                    double ny = -ex;
                    const double nl = std::sqrt(nx * nx + ny * ny);
                    if (nl > 0.0)
                    {
                        nx /= nl;
                        ny /= nl;
                    }
                    const double dist = nx * poly[i].mx + ny * poly[i].my;
                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        bestEdge = i;
                        bestNx = nx;
                        bestNy = ny;
                    }
                }

                // Debug-viz recording (OPT-IN; nullptr on the Step path).
                // Snapshot the CURRENT polytope + the chosen closest edge for
                // this iteration. Reads already-computed values only -- no
                // effect on the math below.
                if (trace)
                {
                    PolytopeSnapshot snap;
                    snap.verts.reserve(static_cast<std::size_t>(count));
                    for (int i = 0; i < count; ++i)
                    {
                        snap.verts.emplace_back(static_cast<Real>(poly[i].mx),
                                                static_cast<Real>(poly[i].my));
                    }
                    snap.edgeA      = bestEdge;
                    snap.edgeB      = (bestEdge + 1) % count;
                    snap.edgeNormal = Vec2(static_cast<Real>(bestNx),
                                           static_cast<Real>(bestNy));
                    snap.edgeDist   = static_cast<Real>(bestDist);
                    trace->epaSnapshots.push_back(std::move(snap));
                }

                // Support along the closest edge's outward normal.
                const MdSupport s = SupportMinkowski(
                    va, na, vb, nb,
                    Vec2(static_cast<Real>(bestNx), static_cast<Real>(bestNy)));
                const double smx = static_cast<double>(s.md.x);
                const double smy = static_cast<double>(s.md.y);
                const double supportDist = smx * bestNx + smy * bestNy;

                // Converged: the support is no farther out than the edge.
                if (supportDist - bestDist <= kEpaEps)
                {
                    return [&]() -> EpaResult
                    {
                        const int i = bestEdge;
                        const int j = (i + 1) % count;
                        // Project the origin onto the converged edge -> t in
                        // [0,1] (closest point on the segment to the origin).
                        const double abx = poly[j].mx - poly[i].mx;
                        const double aby = poly[j].my - poly[i].my;
                        const double dd = abx * abx + aby * aby;
                        double t = (dd > 0.0)
                            ? (-(poly[i].mx * abx + poly[i].my * aby)) / dd
                            : 0.0;
                        if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;

                        EpaResult r;
                        r.ok = true;
                        // Returned normal is B->A = -(MD outward normal).
                        r.normal = Vec2(static_cast<Real>(-bestNx),
                                        static_cast<Real>(-bestNy));
                        r.depth = static_cast<Real>(bestDist < 0.0 ? 0.0
                                                                   : bestDist);
                        r.witnessA = Vec2(
                            static_cast<Real>(poly[i].ax + (poly[j].ax - poly[i].ax) * t),
                            static_cast<Real>(poly[i].ay + (poly[j].ay - poly[i].ay) * t));
                        r.witnessB = Vec2(
                            static_cast<Real>(poly[i].bx + (poly[j].bx - poly[i].bx) * t),
                            static_cast<Real>(poly[i].by + (poly[j].by - poly[i].by) * t));
                        return r;
                    }();
                }

                // Not converged: insert the new vertex between i and i+1
                // (splitting the closest edge). Cap guard -> fall back.
                if (count >= kEpaMaxVerts)
                    return out; // ok=false: overflow

                const int insertAt = bestEdge + 1;
                for (int k = count; k > insertAt; --k)
                    poly[k] = poly[k - 1];
                poly[insertAt].mx = smx;
                poly[insertAt].my = smy;
                poly[insertAt].ax = static_cast<double>(s.pa.x);
                poly[insertAt].ay = static_cast<double>(s.pa.y);
                poly[insertAt].bx = static_cast<double>(s.pb.x);
                poly[insertAt].by = static_cast<double>(s.pb.y);
                ++count;
            }

            // Iteration budget exhausted without convergence -> fall back.
            return out; // ok=false
        }

    } // namespace Physics
} // namespace Manifold2D
