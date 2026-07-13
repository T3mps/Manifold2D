// Gjk.cpp -- 2D GJK distance + closest points, surface distance, and a
// conservative-advancement shape cast (port of GJK.lua + Cast.lua:advance).
//
// See Gjk.hpp for the contract. Ported from GJK.lua (the physics_oracle
// gjk.json bit-match gate was retired in Physics v2 Phase A; the v2 gate is the
// analytic PhysicsGjkV2Test). The simplex reduction, witness interpolation, the
// 64-iteration cap, and the 1e-9 / 1e-18 epsilons are carried over VERBATIM from
// GJK.lua so the branch selection + convergence match the Lua reference.
//
// PRECISION: GJK.lua runs the distance core in Lua doubles. The stored shape
// geometry here is f32 (PhysicsTypes Real), but to track the f64 distance core
// (so witnesses/distance match the Lua within ~1e-4) the simplex arithmetic is
// carried in `double` internally -- f32 core verts are widened on read; the
// GjkResult is narrowed back to Real on return. This keeps the convergence test
// and the closest-point reductions on the same numeric footing as the Lua
// reference.
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.

#include <Manifold2D/Physics/Narrowphase/Gjk.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace Manifold2D
{
    namespace Physics
    {
        namespace
        {
            // f64 vec2 scratch for the simplex (the Lua distance core is f64).
            struct V2
            {
                double x = 0.0;
                double y = 0.0;
            };

            // support(verts, dx, dy) (GJK.lua:10-17): farthest vert of the
            // f32 world-vert span along (dx,dy), returned widened to f64. The
            // ">" comparison + first-index tie-break match the Lua loop.
            // Returns the index of the support vertex via `outIdx`.
            V2 Support(const Vec2* verts, int n, double dx, double dy,
                       int* outIdx = nullptr)
            {
                int bi = 0;
                double best = -std::numeric_limits<double>::infinity();
                for (int i = 0; i < n; ++i)
                {
                    const double d = static_cast<double>(verts[i].x) * dx +
                                     static_cast<double>(verts[i].y) * dy;
                    if (d > best)
                    {
                        best = d;
                        bi = i;
                    }
                }
                if (outIdx) *outIdx = bi;
                return V2{ static_cast<double>(verts[bi].x),
                           static_cast<double>(verts[bi].y) };
            }

            // supportMD (GJK.lua:22-26): Minkowski-difference support =
            // support(A,d) - support(B,-d). Returns the MD point and the two
            // witness points (pa on A, pb on B).
            //
            // f64 INTERNAL type, distinct from the public Manifold2D::Physics::
            // MdSupport (Vec2/Real). The two are kept separate so the simplex
            // math stays in double (GJK.lua parity); SupportMinkowski narrows
            // this to the public Real struct on return.
            struct MdSupport_f64
            {
                V2  md;   // support(A,d) - support(B,-d)
                V2  pa;   // witness on A
                V2  pb;   // witness on B
                int ia;   // index into va[] for pa
                int ib;   // index into vb[] for pb
            };

            MdSupport_f64 SupportMd(const Vec2* va, int na, const Vec2* vb,
                                    int nb, double dx, double dy)
            {
                int ia = 0, ib = 0;
                const V2 pa = Support(va, na,  dx,  dy, &ia);
                const V2 pb = Support(vb, nb, -dx, -dy, &ib);
                return MdSupport_f64{ V2{ pa.x - pb.x, pa.y - pb.y },
                                      pa, pb, ia, ib };
            }

            // EncloseOrigin -- expand a degenerate origin-touching simplex (n==1
            // or n==2) to a NON-degenerate triangle that encloses the origin,
            // then narrow + return. Used by GjkOriginSimplex when GJK stops
            // early with the origin exactly on an MD vertex/edge (boxes sharing
            // a face direction hit this), so EPA always gets a count==3 seed.
            //
            // Each appended vertex is a true MD support (a real boundary point
            // with its A/B witness pair + indices preserved). The search dirs
            // are deterministic: edge perpendicular for n==2, the +/-X / +/-Y
            // axes for n==1. If no support adds a new direction (the MD is a
            // point or segment -- only possible for point/segment cores that
            // cannot actually enclose the origin in area) we return what we
            // have; that path is unreachable for the overlapping-polygon case
            // this seed feeds (a true 2D overlap MD always has area).
            //
            // The templated Narrow is the GjkOriginSimplex-local lambda that
            // packs the f64 scratch slots [0..count) into the public GjkSimplex.
            template <typename Narrow>
            GjkSimplex EncloseOrigin(const Vec2* va, int na,
                                     const Vec2* vb, int nb, int n,
                                     std::array<double, 3>& smx,
                                     std::array<double, 3>& smy,
                                     std::array<double, 3>& sax,
                                     std::array<double, 3>& say,
                                     std::array<double, 3>& sbx,
                                     std::array<double, 3>& sby,
                                     std::array<int, 3>& sia,
                                     std::array<int, 3>& sib,
                                     Narrow&& narrow)
            {
                // Append the MD support along (dx,dy) into slot `n` if it is a
                // genuinely new MD point (not coincident with an existing vert).
                // Returns true on append (n is then advanced by the caller).
                auto tryAppend = [&](double dx, double dy) -> bool
                {
                    const MdSupport_f64 s = SupportMd(va, na, vb, nb, dx, dy);
                    for (int i = 0; i < n; ++i)
                    {
                        const double ex = s.md.x - smx[i];
                        const double ey = s.md.y - smy[i];
                        if (ex * ex + ey * ey < 1e-18)
                            return false; // coincident with an existing vert
                    }
                    smx[n] = s.md.x; smy[n] = s.md.y;
                    sax[n] = s.pa.x; say[n] = s.pa.y;
                    sbx[n] = s.pb.x; sby[n] = s.pb.y;
                    sia[n] = s.ia;   sib[n] = s.ib;
                    return true;
                };

                // n==1: grow to a segment first. Try +/-X then +/-Y until a
                // distinct second MD vertex is found.
                if (n == 1)
                {
                    if (tryAppend(1.0, 0.0))       n = 2;
                    else if (tryAppend(-1.0, 0.0)) n = 2;
                    else if (tryAppend(0.0, 1.0))  n = 2;
                    else if (tryAppend(0.0, -1.0)) n = 2;
                    else return narrow(1, /*encloses*/ true); // MD is a point
                }

                // n==2: grow to a triangle off the edge line. The edge is
                // v0->v1; query the edge perpendicular (both signs) for the
                // off-line apex. The origin lies ON this edge, so any apex off
                // the line yields a triangle that contains the origin.
                {
                    const double ex = smx[1] - smx[0];
                    const double ey = smy[1] - smy[0];
                    // perp = (-ey, ex); try +perp then -perp.
                    if (tryAppend(-ey, ex))      n = 3;
                    else if (tryAppend(ey, -ex)) n = 3;
                    else return narrow(2, /*encloses*/ true); // MD is a segment
                }

                return narrow(3, /*encloses*/ true);
            }
        } // namespace

        // --------------------------------------------------------------------
        // GjkDistance -- the simplex loop (GJK.lua:31-107), carried in f64.
        // --------------------------------------------------------------------
        GjkResult GjkDistance(const Vec2* va, int na, const Vec2* vb, int nb)
        {
            // Guard: a zero-vert core is degenerate (BuildCore default branch).
            // Documents the precondition and prevents Support() reading verts[-1]
            // or verts[0] of a zero-length span (UB).
            if (na <= 0 || nb <= 0)
                return GjkResult{};

            // Simplex scratch: MD points (smx/smy) + witnesses on A (sax/say)
            // and B (sbx/sby), mirroring the Lua module-local arrays. Capacity 3
            // (a 2D simplex never exceeds 3 verts under this reduction).
            std::array<double, 3> smx{}, smy{};
            std::array<double, 3> sax{}, say{}, sbx{}, sby{};

            int n = 1;
            {
                const MdSupport_f64 s0 = SupportMd(va, na, vb, nb, 1.0, 0.0);
                smx[0] = s0.md.x; smy[0] = s0.md.y;
                sax[0] = s0.pa.x; say[0] = s0.pa.y;
                sbx[0] = s0.pb.x; sby[0] = s0.pb.y;
            }

            // Closest point to origin (px,py) + barycentric weights (l1,l2).
            double px = smx[0], py = smy[0];
            double l1 = 1.0, l2 = 0.0;

            for (int iter = 0; iter < 64; ++iter)
            {
                // Closest point on the simplex to the origin, reducing simplex.
                if (n == 1)
                {
                    px = smx[0]; py = smy[0]; l1 = 1.0; l2 = 0.0;
                }
                else if (n == 2)
                {
                    const double ax_ = smx[0], ay_ = smy[0];
                    const double bx_ = smx[1], by_ = smy[1];
                    const double abx = bx_ - ax_, aby = by_ - ay_;
                    const double dd = abx * abx + aby * aby;
                    const double t =
                        (dd > 0.0) ? (-(ax_ * abx + ay_ * aby)) / dd : 0.0;
                    if (t <= 0.0)
                    {
                        n = 1; px = ax_; py = ay_; l1 = 1.0; l2 = 0.0;
                    }
                    else if (t >= 1.0)
                    {
                        smx[0] = smx[1]; smy[0] = smy[1];
                        sax[0] = sax[1]; say[0] = say[1];
                        sbx[0] = sbx[1]; sby[0] = sby[1];
                        n = 1; px = smx[0]; py = smy[0]; l1 = 1.0; l2 = 0.0;
                    }
                    else
                    {
                        px = ax_ + abx * t; py = ay_ + aby * t;
                        l1 = 1.0 - t; l2 = t;
                    }
                }
                else // n == 3: inside test, else reduce to the closest edge.
                {
                    const double x1 = smx[0], y1 = smy[0];
                    const double x2 = smx[1], y2 = smy[1];
                    const double x3 = smx[2], y3 = smy[2];
                    const double c1 = (x2 - x1) * (-y1) - (y2 - y1) * (-x1);
                    const double c2 = (x3 - x2) * (-y2) - (y3 - y2) * (-x2);
                    const double c3 = (x1 - x3) * (-y3) - (y1 - y3) * (-x3);
                    const bool hasNeg = (c1 < 0.0) || (c2 < 0.0) || (c3 < 0.0);
                    const bool hasPos = (c1 > 0.0) || (c2 > 0.0) || (c3 > 0.0);
                    if (!(hasNeg && hasPos))
                    {
                        // Origin inside the simplex -> the cores intersect.
                        return GjkResult{ Real(0), Vec2(Real(0), Real(0)),
                                          Vec2(Real(0), Real(0)) };
                    }

                    // Reduce to the closest of the three edges to the origin.
                    double bestD = std::numeric_limits<double>::infinity();
                    int e1 = 0, e2 = 1;
                    double bt = 0.0;
                    auto edge = [&](int i, int j)
                    {
                        const double ax_ = smx[i], ay_ = smy[i];
                        const double bx_ = smx[j], by_ = smy[j];
                        const double abx = bx_ - ax_, aby = by_ - ay_;
                        const double dd = abx * abx + aby * aby;
                        double t =
                            (dd > 0.0) ? (-(ax_ * abx + ay_ * aby)) / dd : 0.0;
                        if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;
                        const double cx_ = ax_ + abx * t;
                        const double cy_ = ay_ + aby * t;
                        const double d2 = cx_ * cx_ + cy_ * cy_;
                        if (d2 < bestD) { bestD = d2; e1 = i; e2 = j; bt = t; }
                    };
                    edge(0, 1); edge(1, 2); edge(2, 0);

                    const double t1x = smx[e1], t1y = smy[e1];
                    const double t1ax = sax[e1], t1ay = say[e1];
                    const double t1bx = sbx[e1], t1by = sby[e1];
                    const double t2x = smx[e2], t2y = smy[e2];
                    const double t2ax = sax[e2], t2ay = say[e2];
                    const double t2bx = sbx[e2], t2by = sby[e2];

                    smx[0] = t1x; smy[0] = t1y;
                    sax[0] = t1ax; say[0] = t1ay; sbx[0] = t1bx; sby[0] = t1by;
                    smx[1] = t2x; smy[1] = t2y;
                    sax[1] = t2ax; say[1] = t2ay; sbx[1] = t2bx; sby[1] = t2by;
                    n = 2;
                    px = t1x + (t2x - t1x) * bt;
                    py = t1y + (t2y - t1y) * bt;
                    l1 = 1.0 - bt; l2 = bt;
                    if (bt <= 0.0)
                    {
                        n = 1; l1 = 1.0; l2 = 0.0;
                    }
                    else if (bt >= 1.0)
                    {
                        smx[0] = t2x; smy[0] = t2y;
                        sax[0] = t2ax; say[0] = t2ay;
                        sbx[0] = t2bx; sby[0] = t2by;
                        n = 1; l1 = 1.0; l2 = 0.0;
                    }
                }

                // Origin reached -> the cores intersect (witnesses meaningless).
                if (px * px + py * py < 1e-18)
                {
                    return GjkResult{ Real(0), Vec2(Real(0), Real(0)),
                                      Vec2(Real(0), Real(0)) };
                }

                const double dx_ = -px, dy_ = -py;
                const MdSupport_f64 s = SupportMd(va, na, vb, nb, dx_, dy_);
                // No closer support along the search direction -> converged.
                if ((s.md.x * dx_ + s.md.y * dy_) - (px * dx_ + py * dy_) < 1e-9)
                {
                    break;
                }
                // Append the new support (n was 1 or 2 here; index n).
                smx[n] = s.md.x; smy[n] = s.md.y;
                sax[n] = s.pa.x; say[n] = s.pa.y;
                sbx[n] = s.pb.x; sby[n] = s.pb.y;
                n = n + 1;
            }

            const double dist = std::sqrt(px * px + py * py);
            if (n == 1)
            {
                return GjkResult{
                    static_cast<Real>(dist),
                    Vec2(static_cast<Real>(sax[0]), static_cast<Real>(say[0])),
                    Vec2(static_cast<Real>(sbx[0]), static_cast<Real>(sby[0])) };
            }
            return GjkResult{
                static_cast<Real>(dist),
                Vec2(static_cast<Real>(sax[0] * l1 + sax[1] * l2),
                     static_cast<Real>(say[0] * l1 + say[1] * l2)),
                Vec2(static_cast<Real>(sbx[0] * l1 + sbx[1] * l2),
                     static_cast<Real>(sby[0] * l1 + sby[1] * l2)) };
        }

        // --------------------------------------------------------------------
        // GjkDistanceCore -- v2 core-based GJK with closest-feature output.
        //
        // Same algorithm as GjkDistance (same f64 double-precision simplex,
        // same 64-iteration cap, same convergence / overlap tests), extended
        // to track the INPUT VERTEX INDEX for each simplex vertex on both A
        // and B -- port of Box2D v3 b2ShapeDistance's indexA/indexB on each
        // b2SimplexVertex. Feature encoding: see GjkCoreResult::kEdgeMask.
        // --------------------------------------------------------------------
        GjkCoreResult GjkDistanceCore(const Vec2* va, int na,
                                      const Vec2* vb, int nb)
        {
            // Guard: degenerate cores.
            if (na <= 0 || nb <= 0)
                return GjkCoreResult{};

            // Simplex scratch: MD points (smx/smy), witnesses on A (sax/say)
            // and B (sbx/sby), plus the input vertex indices (sia/sib) for the
            // feature output (port of Box2D v3 b2SimplexVertex.indexA/indexB).
            std::array<double, 3>   smx{}, smy{};
            std::array<double, 3>   sax{}, say{}, sbx{}, sby{};
            std::array<int, 3>      sia{}, sib{};

            int n = 1;
            {
                const MdSupport_f64 s0 = SupportMd(va, na, vb, nb, 1.0, 0.0);
                smx[0] = s0.md.x; smy[0] = s0.md.y;
                sax[0] = s0.pa.x; say[0] = s0.pa.y;
                sbx[0] = s0.pb.x; sby[0] = s0.pb.y;
                sia[0] = s0.ia;   sib[0] = s0.ib;
            }

            double px = smx[0], py = smy[0];
            double l1 = 1.0, l2 = 0.0;

            for (int iter = 0; iter < 64; ++iter)
            {
                if (n == 1)
                {
                    px = smx[0]; py = smy[0]; l1 = 1.0; l2 = 0.0;
                }
                else if (n == 2)
                {
                    const double ax_ = smx[0], ay_ = smy[0];
                    const double bx_ = smx[1], by_ = smy[1];
                    const double abx = bx_ - ax_, aby = by_ - ay_;
                    const double dd = abx * abx + aby * aby;
                    const double t =
                        (dd > 0.0) ? (-(ax_ * abx + ay_ * aby)) / dd : 0.0;
                    if (t <= 0.0)
                    {
                        n = 1; px = ax_; py = ay_; l1 = 1.0; l2 = 0.0;
                    }
                    else if (t >= 1.0)
                    {
                        smx[0] = smx[1]; smy[0] = smy[1];
                        sax[0] = sax[1]; say[0] = say[1];
                        sbx[0] = sbx[1]; sby[0] = sby[1];
                        sia[0] = sia[1]; sib[0] = sib[1];
                        n = 1; px = smx[0]; py = smy[0]; l1 = 1.0; l2 = 0.0;
                    }
                    else
                    {
                        px = ax_ + abx * t; py = ay_ + aby * t;
                        l1 = 1.0 - t; l2 = t;
                    }
                }
                else // n == 3: inside test, else reduce to the closest edge.
                {
                    const double x1 = smx[0], y1 = smy[0];
                    const double x2 = smx[1], y2 = smy[1];
                    const double x3 = smx[2], y3 = smy[2];
                    const double c1 = (x2 - x1) * (-y1) - (y2 - y1) * (-x1);
                    const double c2 = (x3 - x2) * (-y2) - (y3 - y2) * (-x2);
                    const double c3 = (x1 - x3) * (-y3) - (y1 - y3) * (-x3);
                    const bool hasNeg = (c1 < 0.0) || (c2 < 0.0) || (c3 < 0.0);
                    const bool hasPos = (c1 > 0.0) || (c2 > 0.0) || (c3 > 0.0);
                    if (!(hasNeg && hasPos))
                    {
                        // Origin inside the simplex -> cores intersect.
                        return GjkCoreResult{ Real(0), Vec2(Real(0), Real(0)),
                                              Vec2(Real(0), Real(0)), 0u, 0u };
                    }

                    // Reduce to closest of the three edges to origin.
                    double bestD = std::numeric_limits<double>::infinity();
                    int e1 = 0, e2 = 1;
                    double bt = 0.0;
                    auto edge = [&](int i, int j)
                    {
                        const double ax_ = smx[i], ay_ = smy[i];
                        const double bx_ = smx[j], by_ = smy[j];
                        const double abx = bx_ - ax_, aby = by_ - ay_;
                        const double dd = abx * abx + aby * aby;
                        double t =
                            (dd > 0.0) ? (-(ax_ * abx + ay_ * aby)) / dd : 0.0;
                        if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;
                        const double cx_ = ax_ + abx * t;
                        const double cy_ = ay_ + aby * t;
                        const double d2 = cx_ * cx_ + cy_ * cy_;
                        if (d2 < bestD) { bestD = d2; e1 = i; e2 = j; bt = t; }
                    };
                    edge(0, 1); edge(1, 2); edge(2, 0);

                    const double t1x = smx[e1], t1y = smy[e1];
                    const double t1ax = sax[e1], t1ay = say[e1];
                    const double t1bx = sbx[e1], t1by = sby[e1];
                    const int    t1ia = sia[e1], t1ib = sib[e1];
                    const double t2x = smx[e2], t2y = smy[e2];
                    const double t2ax = sax[e2], t2ay = say[e2];
                    const double t2bx = sbx[e2], t2by = sby[e2];
                    const int    t2ia = sia[e2], t2ib = sib[e2];

                    smx[0] = t1x; smy[0] = t1y;
                    sax[0] = t1ax; say[0] = t1ay; sbx[0] = t1bx; sby[0] = t1by;
                    sia[0] = t1ia; sib[0] = t1ib;
                    smx[1] = t2x; smy[1] = t2y;
                    sax[1] = t2ax; say[1] = t2ay; sbx[1] = t2bx; sby[1] = t2by;
                    sia[1] = t2ia; sib[1] = t2ib;
                    n = 2;
                    px = t1x + (t2x - t1x) * bt;
                    py = t1y + (t2y - t1y) * bt;
                    l1 = 1.0 - bt; l2 = bt;
                    if (bt <= 0.0)
                    {
                        n = 1; l1 = 1.0; l2 = 0.0;
                    }
                    else if (bt >= 1.0)
                    {
                        smx[0] = t2x; smy[0] = t2y;
                        sax[0] = t2ax; say[0] = t2ay;
                        sbx[0] = t2bx; sby[0] = t2by;
                        sia[0] = t2ia; sib[0] = t2ib;
                        n = 1; l1 = 1.0; l2 = 0.0;
                    }
                }

                // Origin reached -> cores intersect.
                if (px * px + py * py < 1e-18)
                {
                    return GjkCoreResult{ Real(0), Vec2(Real(0), Real(0)),
                                          Vec2(Real(0), Real(0)), 0u, 0u };
                }

                const double dx_ = -px, dy_ = -py;
                const MdSupport_f64 s = SupportMd(va, na, vb, nb, dx_, dy_);
                // No closer support -> converged.
                if ((s.md.x * dx_ + s.md.y * dy_) - (px * dx_ + py * dy_) < 1e-9)
                {
                    break;
                }
                smx[n] = s.md.x; smy[n] = s.md.y;
                sax[n] = s.pa.x; say[n] = s.pa.y;
                sbx[n] = s.pb.x; sby[n] = s.pb.y;
                sia[n] = s.ia;   sib[n] = s.ib;
                n = n + 1;
            }

            const double dist = std::sqrt(px * px + py * py);

            // Build witness points (barycentric interpolation or single vert).
            Vec2 wA, wB;
            if (n == 1)
            {
                wA = Vec2(static_cast<Real>(sax[0]), static_cast<Real>(say[0]));
                wB = Vec2(static_cast<Real>(sbx[0]), static_cast<Real>(sby[0]));
            }
            else
            {
                wA = Vec2(static_cast<Real>(sax[0] * l1 + sax[1] * l2),
                          static_cast<Real>(say[0] * l1 + say[1] * l2));
                wB = Vec2(static_cast<Real>(sbx[0] * l1 + sbx[1] * l2),
                          static_cast<Real>(sby[0] * l1 + sby[1] * l2));
            }

            // Build feature indices (port of Box2D v3 indexA/indexB logic).
            //   n==1: closest feature is a single vertex; feature = idx.
            //   n==2: closest feature is an edge between two input verts;
            //          feature = kEdgeMask | (idxLo << 16) | idxHi.
            //          We sort the indices (lo <= hi) for a canonical encoding
            //          so the same edge produces the same id regardless of the
            //          order the simplex accumulated the two vertices.
            uint32_t fA = 0u, fB = 0u;
            if (n == 1)
            {
                fA = static_cast<uint32_t>(sia[0]);
                fB = static_cast<uint32_t>(sib[0]);
            }
            else // n == 2
            {
                const int a0 = sia[0], a1 = sia[1];
                const int b0 = sib[0], b1 = sib[1];
                const int alo = (a0 <= a1) ? a0 : a1;
                const int ahi = (a0 <= a1) ? a1 : a0;
                const int blo = (b0 <= b1) ? b0 : b1;
                const int bhi = (b0 <= b1) ? b1 : b0;
                // Edge encoding: kEdgeMask | (lo << 16) | hi.
                // When both edge endpoints are the SAME index (degenerate edge
                // after reduction), emit as vertex feature instead.
                if (alo == ahi)
                    fA = static_cast<uint32_t>(alo);
                else
                    fA = GjkCoreResult::kEdgeMask |
                         (static_cast<uint32_t>(alo) << 16) |
                          static_cast<uint32_t>(ahi);

                if (blo == bhi)
                    fB = static_cast<uint32_t>(blo);
                else
                    fB = GjkCoreResult::kEdgeMask |
                         (static_cast<uint32_t>(blo) << 16) |
                          static_cast<uint32_t>(bhi);
            }

            return GjkCoreResult{
                static_cast<Real>(dist),
                wA, wB,
                fA, fB
            };
        }

        // --------------------------------------------------------------------
        // SupportMinkowski -- public MD support over two world cores (EPA seam).
        //
        // ADDITIVE. Reuses the internal f64 SupportMd (argmax over A along +dir,
        // over B along -dir; md = pa - pb); narrows the f64 result to Real on
        // return so EPA/MPR consumers get the public Vec2 (Real) struct.
        // --------------------------------------------------------------------
        MdSupport SupportMinkowski(const Vec2* va, int na, const Vec2* vb,
                                   int nb, Vec2 dir)
        {
            // Degenerate cores: Support() would read out of a zero-length span.
            if (na <= 0 || nb <= 0)
                return MdSupport{};

            const MdSupport_f64 s = SupportMd(va, na, vb, nb,
                                              static_cast<double>(dir.x),
                                              static_cast<double>(dir.y));
            MdSupport out;
            out.md = Vec2(static_cast<Real>(s.md.x), static_cast<Real>(s.md.y));
            out.pa = Vec2(static_cast<Real>(s.pa.x), static_cast<Real>(s.pa.y));
            out.pb = Vec2(static_cast<Real>(s.pb.x), static_cast<Real>(s.pb.y));
            out.ia = s.ia;
            out.ib = s.ib;
            return out;
        }

        // --------------------------------------------------------------------
        // GjkOriginSimplex -- terminal GJK simplex (EPA seed).
        //
        // Runs the SAME simplex loop as GjkDistanceCore (same f64 internal
        // simplex, same 64-iteration cap, same convergence / overlap tests).
        // The ONLY behavioral difference from GjkDistanceCore is at the n==3
        // origin-inside branch and at convergence: instead of producing a
        // distance + closest feature, it returns the terminal simplex verts
        // (MD point + A/B witness pair + indices). On overlap the returned
        // triangle encloses the origin (enclosesOrigin == true, count == 3).
        // ADDITIVE: GjkDistanceCore is UNCHANGED.
        // --------------------------------------------------------------------
        GjkSimplex GjkOriginSimplex(const Vec2* va, int na, const Vec2* vb,
                                    int nb)
        {
            // Guard: degenerate cores.
            if (na <= 0 || nb <= 0)
                return GjkSimplex{};

            // Simplex scratch: MD points (smx/smy), witnesses on A (sax/say)
            // and B (sbx/sby), plus the input vertex indices (sia/sib) -- the
            // same layout GjkDistanceCore carries.
            std::array<double, 3>   smx{}, smy{};
            std::array<double, 3>   sax{}, say{}, sbx{}, sby{};
            std::array<int, 3>      sia{}, sib{};

            int n = 1;
            {
                const MdSupport_f64 s0 = SupportMd(va, na, vb, nb, 1.0, 0.0);
                smx[0] = s0.md.x; smy[0] = s0.md.y;
                sax[0] = s0.pa.x; say[0] = s0.pa.y;
                sbx[0] = s0.pb.x; sby[0] = s0.pb.y;
                sia[0] = s0.ia;   sib[0] = s0.ib;
            }

            // Narrow the f64 simplex slots [0..count) into the public GjkSimplex.
            auto narrow = [&](int count, bool encloses) -> GjkSimplex
            {
                GjkSimplex out;
                out.count = count;
                out.enclosesOrigin = encloses;
                for (int i = 0; i < count; ++i)
                {
                    out.v[i].md = Vec2(static_cast<Real>(smx[i]),
                                       static_cast<Real>(smy[i]));
                    out.v[i].wa = Vec2(static_cast<Real>(sax[i]),
                                       static_cast<Real>(say[i]));
                    out.v[i].wb = Vec2(static_cast<Real>(sbx[i]),
                                       static_cast<Real>(sby[i]));
                    out.v[i].ia = sia[i];
                    out.v[i].ib = sib[i];
                }
                return out;
            };

            double px = smx[0], py = smy[0];

            for (int iter = 0; iter < 64; ++iter)
            {
                if (n == 1)
                {
                    px = smx[0]; py = smy[0];
                }
                else if (n == 2)
                {
                    const double ax_ = smx[0], ay_ = smy[0];
                    const double bx_ = smx[1], by_ = smy[1];
                    const double abx = bx_ - ax_, aby = by_ - ay_;
                    const double dd = abx * abx + aby * aby;
                    const double t =
                        (dd > 0.0) ? (-(ax_ * abx + ay_ * aby)) / dd : 0.0;
                    if (t <= 0.0)
                    {
                        n = 1; px = ax_; py = ay_;
                    }
                    else if (t >= 1.0)
                    {
                        smx[0] = smx[1]; smy[0] = smy[1];
                        sax[0] = sax[1]; say[0] = say[1];
                        sbx[0] = sbx[1]; sby[0] = sby[1];
                        sia[0] = sia[1]; sib[0] = sib[1];
                        n = 1; px = smx[0]; py = smy[0];
                    }
                    else
                    {
                        px = ax_ + abx * t; py = ay_ + aby * t;
                    }
                }
                else // n == 3: inside test, else reduce to the closest edge.
                {
                    const double x1 = smx[0], y1 = smy[0];
                    const double x2 = smx[1], y2 = smy[1];
                    const double x3 = smx[2], y3 = smy[2];
                    const double c1 = (x2 - x1) * (-y1) - (y2 - y1) * (-x1);
                    const double c2 = (x3 - x2) * (-y2) - (y3 - y2) * (-x2);
                    const double c3 = (x1 - x3) * (-y3) - (y1 - y3) * (-x3);
                    const bool hasNeg = (c1 < 0.0) || (c2 < 0.0) || (c3 < 0.0);
                    const bool hasPos = (c1 > 0.0) || (c2 > 0.0) || (c3 > 0.0);
                    if (!(hasNeg && hasPos))
                    {
                        // Origin inside the simplex -> cores OVERLAP. Return the
                        // 3 terminal MD verts as the EPA seed triangle (vs
                        // GjkDistanceCore which returns distance 0 here).
                        return narrow(3, /*encloses*/ true);
                    }

                    // Reduce to closest of the three edges to origin.
                    double bestD = std::numeric_limits<double>::infinity();
                    int e1 = 0, e2 = 1;
                    double bt = 0.0;
                    auto edge = [&](int i, int j)
                    {
                        const double ax_ = smx[i], ay_ = smy[i];
                        const double bx_ = smx[j], by_ = smy[j];
                        const double abx = bx_ - ax_, aby = by_ - ay_;
                        const double dd = abx * abx + aby * aby;
                        double t =
                            (dd > 0.0) ? (-(ax_ * abx + ay_ * aby)) / dd : 0.0;
                        if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;
                        const double cx_ = ax_ + abx * t;
                        const double cy_ = ay_ + aby * t;
                        const double d2 = cx_ * cx_ + cy_ * cy_;
                        if (d2 < bestD) { bestD = d2; e1 = i; e2 = j; bt = t; }
                    };
                    edge(0, 1); edge(1, 2); edge(2, 0);

                    const double t1x = smx[e1], t1y = smy[e1];
                    const double t1ax = sax[e1], t1ay = say[e1];
                    const double t1bx = sbx[e1], t1by = sby[e1];
                    const int    t1ia = sia[e1], t1ib = sib[e1];
                    const double t2x = smx[e2], t2y = smy[e2];
                    const double t2ax = sax[e2], t2ay = say[e2];
                    const double t2bx = sbx[e2], t2by = sby[e2];
                    const int    t2ia = sia[e2], t2ib = sib[e2];

                    smx[0] = t1x; smy[0] = t1y;
                    sax[0] = t1ax; say[0] = t1ay; sbx[0] = t1bx; sby[0] = t1by;
                    sia[0] = t1ia; sib[0] = t1ib;
                    smx[1] = t2x; smy[1] = t2y;
                    sax[1] = t2ax; say[1] = t2ay; sbx[1] = t2bx; sby[1] = t2by;
                    sia[1] = t2ia; sib[1] = t2ib;
                    n = 2;
                    px = t1x + (t2x - t1x) * bt;
                    py = t1y + (t2y - t1y) * bt;
                    if (bt <= 0.0)
                    {
                        n = 1;
                    }
                    else if (bt >= 1.0)
                    {
                        smx[0] = t2x; smy[0] = t2y;
                        sax[0] = t2ax; say[0] = t2ay;
                        sbx[0] = t2bx; sby[0] = t2by;
                        sia[0] = t2ia; sib[0] = t2ib;
                        n = 1;
                    }
                }

                // Origin reached -> cores OVERLAP. The current simplex is the
                // closest feature touching the origin; GjkDistanceCore returns
                // distance 0 here. GJK can stop early with n<3 when the origin
                // lands exactly on an MD vertex (n==1) or edge (n==2) -- common
                // for boxes that share a face direction. EPA needs a NON-
                // degenerate triangle that encloses the origin, so expand the
                // simplex to 3 verts before returning (each added support is a
                // real MD-boundary vertex with its A/B witness pair preserved).
                if (px * px + py * py < 1e-18)
                {
                    return EncloseOrigin(va, na, vb, nb, n, smx, smy,
                                         sax, say, sbx, sby, sia, sib, narrow);
                }

                const double dx_ = -px, dy_ = -py;
                const MdSupport_f64 s = SupportMd(va, na, vb, nb, dx_, dy_);
                // No closer support -> converged (separated, closest simplex).
                if ((s.md.x * dx_ + s.md.y * dy_) - (px * dx_ + py * dy_) < 1e-9)
                {
                    break;
                }
                smx[n] = s.md.x; smy[n] = s.md.y;
                sax[n] = s.pa.x; say[n] = s.pa.y;
                sbx[n] = s.pb.x; sby[n] = s.pb.y;
                sia[n] = s.ia;   sib[n] = s.ib;
                n = n + 1;
            }

            // Converged without enclosing the origin -> cores are SEPARATED.
            return narrow(n, /*encloses*/ false);
        }

        // --------------------------------------------------------------------
        // BuildCore -- GJK.lua:buildCore (111-131). Returns the core radius.
        // --------------------------------------------------------------------
        Real BuildCore(const Shape& s, const Transform& xf, Vec2* out,
                       int& outCount)
        {
            // ROTATION-AWARE (v2 Task 7, Part 0): the unified core (Shape::verts,
            // populated in LOCAL untransformed space for ALL kinds by T1) is
            // rotated by xf.rotation and translated by xf.position. This replaces
            // the old kind-switch that applied ONLY xf.position (rotation-blind),
            // which made ShapeDistance / ShapeCast / ShapePolyDistance ignore the
            // body angle their callers passed. The radius is the core round-
            // inflation (s.radius for circle/capsule; 0 for aabb/polygon).
            //
            // BYTE-IDENTITY at angle 0: with xf.rotation == 0 -> c=1, s=0 -> R is
            // identity, so out = xf.position + v -- bit-identical to the old
            // translate-only switch (T1 stores the SAME local core verts the old
            // switch hard-coded). GJK is also vertex-order-agnostic, so even
            // where T1's vert order differs the distance result is unchanged.
            //
            // Unknown/empty core: if s.verts is empty (a future ShapeKind that
            // forgot to populate the core) outCount = 0; GjkDistance guards
            // na/nb <= 0 at entry, so this never runs GJK on garbage.
            const int n = static_cast<int>(s.verts.size());
            if (n <= 0)
            {
                outCount = 0;
                return s.radius;
            }
            const Real c = std::cos(xf.rotation);
            const Real sn = std::sin(xf.rotation);
            const Real px = xf.position.x;
            const Real py = xf.position.y;
            for (int i = 0; i < n; ++i)
            {
                const Real lx = s.verts[i].x;
                const Real ly = s.verts[i].y;
                out[i] = Vec2(px + c * lx - sn * ly,
                              py + sn * lx + c * ly);
            }
            outCount = n;
            return s.radius;
        }

        namespace
        {
            // Shared surface-distance assembly (GJK.lua:137-147 / 151-157): take
            // the core GJK result + the two radii and produce the surface
            // distance, surface witnesses, and the B->A normal.
            ShapeDistanceResult AssembleSurface(const GjkResult& g, Real ra,
                                                Real rb)
            {
                ShapeDistanceResult r{};
                Real nx = Real(0), ny = Real(0);
                if (g.distance > Real(1e-9))
                {
                    nx = (g.witnessA.x - g.witnessB.x) / g.distance;
                    ny = (g.witnessA.y - g.witnessB.y) / g.distance;
                }
                r.distance = g.distance - ra - rb;
                r.witnessA = Vec2(g.witnessA.x - nx * ra, g.witnessA.y - ny * ra);
                r.witnessB = Vec2(g.witnessB.x + nx * rb, g.witnessB.y + ny * rb);
                r.normal = Vec2(nx, ny);
                return r;
            }
        } // namespace

        // --------------------------------------------------------------------
        // ShapeDistance -- GJK.lua:shapeDistance (137-147).
        // --------------------------------------------------------------------
        ShapeDistanceResult ShapeDistance(const Shape& a, const Transform& xfA,
                                          const Shape& b, const Transform& xfB)
        {
            // kMaxPolyVerts = 128 -> each buffer is 1 KB on the stack (zero per-call
            // heap, intentional). If kMaxPolyVerts grows substantially, revisit
            // (thread-local scratch or a small-N fast path).
            Vec2 coreA[kMaxPolyVerts];
            Vec2 coreB[kMaxPolyVerts];
            int na = 0, nb = 0;
            const Real ra = BuildCore(a, xfA, coreA, na);
            const Real rb = BuildCore(b, xfB, coreB, nb);
            const GjkResult g = GjkDistance(coreA, na, coreB, nb);
            return AssembleSurface(g, ra, rb);
        }

        // --------------------------------------------------------------------
        // ShapePolyDistance -- GJK.lua:shapePolyDistance (151-157). B is a raw
        // world poly (rb = 0).
        // --------------------------------------------------------------------
        ShapeDistanceResult ShapePolyDistance(const Shape& a,
                                              const Transform& xfA,
                                              const Vec2* poly, int n)
        {
            // kMaxPolyVerts = 128 -> 1 KB on the stack (zero per-call heap,
            // intentional). If kMaxPolyVerts grows substantially, revisit.
            Vec2 coreA[kMaxPolyVerts];
            int na = 0;
            const Real ra = BuildCore(a, xfA, coreA, na);
            const GjkResult g = GjkDistance(coreA, na, poly, n);
            return AssembleSurface(g, ra, Real(0));
        }

        namespace
        {
            // advance (Cast.lua:20-29): march the moving shape's center from
            // `start` along `translation` (length `len`), querying `distFn` at
            // each step center. Stop when the surface distance drops below TOL
            // (hit) or t exceeds 1 (miss). distFn yields (distance, normal).
            template <typename DistFn>
            ShapeCastResult Advance(const Vec2& start, const Vec2& translation,
                                    Real len, DistFn&& distFn)
            {
                ShapeCastResult res{};
                Real t = Real(0);
                for (int i = 0; i < kShapeCastMaxIter; ++i)
                {
                    const Vec2 c(start.x + translation.x * t,
                                 start.y + translation.y * t);
                    const ShapeDistanceResult d = distFn(c);
                    if (d.distance < kShapeCastTol)
                    {
                        res.hit = true;
                        res.t = t;
                        res.normal = d.normal;
                        res.distance = d.distance;
                        return res;
                    }
                    t = t + d.distance / len;
                    if (t > Real(1))
                    {
                        return res; // miss
                    }
                }
                return res; // miss (iteration budget exhausted)
            }
        } // namespace

        // --------------------------------------------------------------------
        // ShapeCast -- conservative advancement vs one obstacle shape.
        // --------------------------------------------------------------------
        ShapeCastResult ShapeCast(const Shape& moving, const Transform& start,
                                  const Vec2& translation, const Shape& obstacle,
                                  const Transform& obstacleXf)
        {
            const Real len = std::sqrt(translation.x * translation.x +
                                       translation.y * translation.y);
            if (len < Real(1e-9))
            {
                return ShapeCastResult{}; // no travel -> no hit (Cast.lua guard)
            }
            return Advance(
                start.position, translation, len, [&](const Vec2& c)
                {
                    const Transform xf{ c, start.rotation };
                    return ShapeDistance(moving, xf, obstacle, obstacleXf);
                });
        }

        // --------------------------------------------------------------------
        // ShapeCastPoly -- conservative advancement vs a raw world poly span.
        // --------------------------------------------------------------------
        ShapeCastResult ShapeCastPoly(const Shape& moving,
                                      const Transform& start,
                                      const Vec2& translation, const Vec2* poly,
                                      int n)
        {
            const Real len = std::sqrt(translation.x * translation.x +
                                       translation.y * translation.y);
            if (len < Real(1e-9))
            {
                return ShapeCastResult{};
            }
            return Advance(
                start.position, translation, len, [&](const Vec2& c)
                {
                    const Transform xf{ c, start.rotation };
                    return ShapePolyDistance(moving, xf, poly, n);
                });
        }

    } // namespace Physics
} // namespace Manifold2D
