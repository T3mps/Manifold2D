// Mpr.cpp -- 2D MPR (Minkowski Portal Refinement / XenoCollide): a fast
// single-point deep-overlap contact for two convex cores. See Mpr.hpp for the
// contract and the normal-sign convention.
//
// Standard 2D MPR (Snethen XenoCollide; dyn4j MinkowskiPenetrationSolver):
//   PHASE 1 -- interior point: v0 = MD of the two core CENTROIDS
//     (centroid(A) - centroid(B)). This is a point KNOWN to be inside the MD
//     (a convex combination of MD verts), so the ray v0 -> origin starts
//     inside and exits through the MD boundary. If v0 ~= origin the centroids
//     coincide in MD space -> nudge the first search direction to (1,0).
//   PHASE 2 -- portal discovery (2D portal = one segment):
//     v1 = support(-v0)  (toward the origin from the interior point).
//     Choose a perpendicular to (v1 - v0) oriented so the portal segment will
//     STRADDLE the origin ray, then v2 = support(thatPerp). The portal (v1,v2)
//     now brackets the origin direction from v0 (deterministic perp-sign).
//   PHASE 3 -- portal refinement:
//     The portal edge (v1, v2) has an OUTWARD normal n (pointing away from v0,
//     toward the MD boundary). Query v3 = support(n). If v3 is no farther along
//     n than the portal edge (within kMprEps) -> CONVERGED. Else replace the
//     portal vertex (v1 or v2) that keeps the origin bracketed and repeat.
//   CONVERGE -- the portal edge is on the MD boundary along the origin ray:
//     `normal` = -(edge outward normal)  (flip to B -> A, same as Epa).
//     `depth`  = distance from the origin to the portal edge along the outward
//                normal (>= 0).
//     `point`  = the world-space midpoint of the portal edge's A/B witnesses
//                projected to the origin's nearest point on the edge.
//
// PRECISION: f64 internal (matching the GJK / EPA pattern); narrowed to Real
// on return. DETERMINISM: fixed caps + fixed epsilon + fixed perp-sign / vertex
// -replace decision; no wall-clock; no heap.
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.

#include <Manifold2D/Physics/Narrowphase/Mpr.hpp>

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
            // MPR tuning constants (named per the Task contract).
            //
            // kMprEps : progress epsilon. A support whose distance along the
            //           portal edge's outward normal exceeds the edge's
            //           origin-distance by MORE than this is "new" -> refine +
            //           continue; within it -> converged. Sized to match the
            //           EPA-class f64 convergence tolerance.
            constexpr double kMprEps = 1e-7;
            // kMprMaxIters : max portal refinement iterations (support inserts).
            constexpr int    kMprMaxIters = 32;
            // kMprMaxVerts : max polytope vertices (seed 3 + insertions).
            constexpr int    kMprMaxVerts = 32;
            // kMprZero : a small magnitude below which a vector is "zero"
            //            (used for the v0 ~= origin and degenerate-portal guards).
            constexpr double kMprZero = 1e-12;

            // An f64 MD support vertex: the MD boundary point plus the A/B
            // witness pair that produced it (world space). Mirrors the
            // SupportMinkowski output, widened to f64 for the portal math.
            struct MdVert
            {
                double mx = 0.0, my = 0.0;   // MD point (wa - wb)
                double ax = 0.0, ay = 0.0;   // witness on A
                double bx = 0.0, by = 0.0;   // witness on B
            };

            // Query SupportMinkowski(dir) and widen to an MdVert.
            MdVert SupportF64(const Vec2* va, int na, const Vec2* vb, int nb,
                              double dx, double dy)
            {
                const MdSupport s = SupportMinkowski(
                    va, na, vb, nb,
                    Vec2(static_cast<Real>(dx), static_cast<Real>(dy)));
                MdVert v;
                v.mx = static_cast<double>(s.md.x);
                v.my = static_cast<double>(s.md.y);
                v.ax = static_cast<double>(s.pa.x);
                v.ay = static_cast<double>(s.pa.y);
                v.bx = static_cast<double>(s.pb.x);
                v.by = static_cast<double>(s.pb.y);
                return v;
            }

            // 2D cross product (z component) of (a) x (b).
            inline double Cross(double ax, double ay, double bx, double by)
            {
                return ax * by - ay * bx;
            }
        } // namespace

        // --------------------------------------------------------------------
        // Mpr -- the 2D portal-refinement loop over the Minkowski difference.
        // --------------------------------------------------------------------
        MprResult Mpr(const Vec2* va, int na, const Vec2* vb, int nb,
                      NarrowphaseTrace* trace)
        {
            MprResult out;

            // Degenerate cores: SupportMinkowski guards na/nb<=0, but MPR needs
            // a real MD interior, so bail to ok=false (caller emits no contact).
            if (na <= 0 || nb <= 0)
                return out;

            // --- PHASE 1: interior point v0 = centroid(A) - centroid(B). ---
            double cax = 0.0, cay = 0.0;
            for (int i = 0; i < na; ++i) { cax += va[i].x; cay += va[i].y; }
            cax /= static_cast<double>(na);
            cay /= static_cast<double>(na);
            double cbx = 0.0, cby = 0.0;
            for (int i = 0; i < nb; ++i) { cbx += vb[i].x; cby += vb[i].y; }
            cbx /= static_cast<double>(nb);
            cby /= static_cast<double>(nb);

            double v0x = cax - cbx;
            double v0y = cay - cby;

            // Ray direction from v0 toward the origin: d = -v0.
            double dx = -v0x;
            double dy = -v0y;
            double dlen2 = dx * dx + dy * dy;
            if (dlen2 < kMprZero)
            {
                // Centroids coincide in MD space (origin == v0). Nudge the
                // search direction deterministically.
                dx = 1.0;
                dy = 0.0;
                dlen2 = 1.0;
            }

            // --- PHASE 2: portal discovery. ---
            // v1 = support toward the origin from the interior point.
            MdVert v1 = SupportF64(va, na, vb, nb, dx, dy);

            // Perpendicular to (v1 - v0). Orient it so the portal (v1, v2)
            // brackets the origin: the origin must lie on the OPPOSITE side of
            // the line (v0 -> v1) from where v2 will be found. We pick the perp
            // sign by which side of (v1 - v0) the origin (relative to v0) sits.
            double e1x = v1.mx - v0x;
            double e1y = v1.my - v0y;
            // Perp candidate: rotate (e1) by +90 deg = (-e1y, e1x).
            double px = -e1y;
            double py = e1x;
            // The origin direction from v0 is (-v0). Choose the perp that points
            // toward the origin side so v2 lands on the far side of the ray,
            // making (v1, v2) straddle it. side = cross(e1, originDir).
            const double side = Cross(e1x, e1y, -v0x, -v0y);
            if (side > 0.0)
            {
                // Flip the perp so it points toward the origin side.
                px = e1y;
                py = -e1x;
            }
            double plen2 = px * px + py * py;
            if (plen2 < kMprZero)
            {
                // v1 collinear with v0 through the origin: pick any perp of d.
                px = -dy;
                py = dx;
            }
            MdVert v2 = SupportF64(va, na, vb, nb, px, py);

            // --- PHASE 3: expand the seed polytope to the MINIMUM edge. ---
            //
            // The portal (v1, v2) found above is ONE supporting segment along the
            // v0 -> origin ray; on its own it is RAY-DEPENDENT and need not be
            // the closest face (a corner-grazing ray can pick a far face), so a
            // pure ray-MPR convergence would report a depth DEEPER than the true
            // minimum and disagree with EPA. To return the SAME minimum EPA does
            // (the Task's cross-check + the gap-case face both demand it), we seed
            // a small CCW polytope with the origin-enclosing triangle (v0, v1, v2)
            // -- v0 interior + (v1,v2) straddling the origin ray enclose the
            // origin -- and run the EPA closest-edge expansion. The interior seed
            // vertex v0 is split away as the polytope grows onto the boundary
            // (edges touching an interior vertex always advance under support).
            //
            // This is XenoCollide's deep-penetration refinement (dyn4j's
            // MinkowskiPenetrationSolver is itself an EPA): MPR's value as the
            // EPA FALLBACK is the INDEPENDENT support-query seeding (no reliance
            // on GjkOriginSimplex, which is what makes the production EPA bail in
            // the degenerate cases this fallback exists for) -- not a different
            // termination.
            std::array<MdVert, kMprMaxVerts> poly{};
            int count = 3;
            poly[0] = MdVert{ v0x, v0y, cax, cay, cbx, cby }; // interior seed
            poly[1] = v1;
            poly[2] = v2;

            // Force CCW winding (signed area > 0): the outward-normal math below
            // (outward normal of CCW edge e is (e.y, -e.x)) assumes CCW.
            {
                const double sa2 =
                    (poly[1].mx - poly[0].mx) * (poly[2].my - poly[0].my) -
                    (poly[2].mx - poly[0].mx) * (poly[1].my - poly[0].my);
                if (sa2 < 0.0)
                {
                    const MdVert tmp = poly[1];
                    poly[1] = poly[2];
                    poly[2] = tmp;
                }
            }

            for (int iter = 0; iter < kMprMaxIters; ++iter)
            {
                // Closest polytope edge to the origin: smallest dot(outwardN, p0).
                // Outward normal for CCW edge e=(p1-p0) is (e.y, -e.x), normalized.
                // Tie-break: lowest edge index (loop keeps the first strictly-
                // smaller) -> deterministic.
                int    bestEdge = 0;
                double bestDist = std::numeric_limits<double>::infinity();
                double bestNx = 0.0, bestNy = 0.0;
                for (int i = 0; i < count; ++i)
                {
                    const int j = (i + 1) % count;
                    const double ex = poly[j].mx - poly[i].mx;
                    const double ey = poly[j].my - poly[i].my;
                    double nx = ey;
                    double ny = -ex;
                    const double nl = std::sqrt(nx * nx + ny * ny);
                    if (nl > 0.0) { nx /= nl; ny /= nl; }
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
                // Snapshot the interior seed (poly[0]) + the chosen closest
                // edge endpoints + the support ray direction for this
                // iteration. Reads already-computed values only.
                if (trace)
                {
                    const int j = (bestEdge + 1) % count;
                    MprSnapshot snap;
                    snap.v0 = Vec2(static_cast<Real>(poly[0].mx),
                                   static_cast<Real>(poly[0].my));
                    snap.v1 = Vec2(static_cast<Real>(poly[bestEdge].mx),
                                   static_cast<Real>(poly[bestEdge].my));
                    snap.v2 = Vec2(static_cast<Real>(poly[j].mx),
                                   static_cast<Real>(poly[j].my));
                    snap.rayDir = Vec2(static_cast<Real>(bestNx),
                                       static_cast<Real>(bestNy));
                    trace->mprSnapshots.push_back(snap);
                }

                // Support along the closest edge's outward normal.
                const MdVert s = SupportF64(va, na, vb, nb, bestNx, bestNy);
                const double supportDist = s.mx * bestNx + s.my * bestNy;

                // Converged: the support is no farther out than the closest edge.
                if (supportDist - bestDist <= kMprEps)
                {
                    const int i = bestEdge;
                    const int j = (i + 1) % count;
                    const double abx = poly[j].mx - poly[i].mx;
                    const double aby = poly[j].my - poly[i].my;
                    const double dd = abx * abx + aby * aby;
                    double t = (dd > 0.0)
                        ? (-(poly[i].mx * abx + poly[i].my * aby)) / dd
                        : 0.0;
                    if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;

                    // World contact point: midpoint of the A/B witnesses lerped
                    // to the origin's nearest point on the converged edge.
                    const double ax = poly[i].ax + (poly[j].ax - poly[i].ax) * t;
                    const double ay = poly[i].ay + (poly[j].ay - poly[i].ay) * t;
                    const double bx = poly[i].bx + (poly[j].bx - poly[i].bx) * t;
                    const double by = poly[i].by + (poly[j].by - poly[i].by) * t;

                    out.ok = true;
                    // normal = -(MD outward normal) -> B -> A (same flip Epa uses).
                    out.normal = Vec2(static_cast<Real>(-bestNx),
                                      static_cast<Real>(-bestNy));
                    out.depth = static_cast<Real>(bestDist < 0.0 ? 0.0
                                                                 : bestDist);
                    out.point = Vec2(static_cast<Real>((ax + bx) * 0.5),
                                     static_cast<Real>((ay + by) * 0.5));
                    return out;
                }

                // Not converged: insert the support, splitting the closest edge.
                if (count >= kMprMaxVerts)
                    return out; // ok=false: overflow

                const int insertAt = bestEdge + 1;
                for (int k = count; k > insertAt; --k)
                    poly[k] = poly[k - 1];
                poly[insertAt] = s;
                ++count;
            }

            // Iteration budget exhausted without convergence -> ok=false.
            return out;
        }

    } // namespace Physics
} // namespace Manifold2D
