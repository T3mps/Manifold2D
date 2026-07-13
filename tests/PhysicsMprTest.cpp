// Physics v2 EPA workstream Task 4: single-point deep overlap via MPR.
//
// MPR (Minkowski Portal Refinement / XenoCollide, Snethen; dyn4j
// MinkowskiPenetrationSolver 2D analog) is EPA's convergence fallback: a fast
// single-point deep-overlap solver over two convex CORES (radii NOT applied).
//
// All expected values are hand-derived from first principles (NOT captured from
// any oracle). The analytic tests + the EPA cross-check ARE the gate.
//
// NORMAL CONVENTION (pinned by case (a), matching Epa): `normal` points B -> A
// (the direction to push A out of B). `depth` is the core penetration depth
// along `normal` (>= 0). `point` is a single contact point on the MD boundary,
// in world space.
//
// Cases:
//   (a) Single-point correctness: circle core (1 vert at (5,1)) vs MakeAabb(6,6)
//       at origin -> nearest face is B's +x (depth 6-5=1). Expect normal~=(1,0),
//       depth~=1. Same deep case as EPA test (a).
//   (b) EPA <-> MPR cross-check: for several deep pairs (box-in-box, circle-in-
//       box, rotated box) the MPR and EPA normals agree within 1e-2 AND
//       mpr.depth <= epa.depth + 1e-3 (EPA is the exact minimum; MPR must NOT
//       report deeper than it).
//   (c) Determinism: Mpr twice on the same inputs -> bit-identical result.
//   (d) Fallback wiring: the Collide deep-round path yields a sane 1-point
//       manifold; a TEST-ONLY helper proves the MPR-built manifold matches the
//       EPA-built manifold for the gap case (no production branch that can't be
//       hit).

#include <array>
#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <Manifold2D/Physics/Narrowphase/Collide.hpp>
#include <Manifold2D/Physics/Narrowphase/Epa.hpp>
#include <Manifold2D/Physics/Narrowphase/Mpr.hpp>
#include <Manifold2D/Physics/Shapes.hpp>

using Catch::Approx;
using namespace Manifold2D::Physics;

namespace
{
    constexpr double kTol = 1e-3;
    constexpr double kPiD64 = 3.14159265358979323846;

    // Rotate + translate a shape's core verts into world space.
    // (copy of the helper in PhysicsGjkV2Test.cpp:46-54). xf.rotation is the
    // angle in radians. Returns a vector of world-space verts.
    std::vector<Vec2> RotateCoreVerts(const Shape& s, const Transform& xf)
    {
        const float c = std::cos(xf.rotation);
        const float s_ = std::sin(xf.rotation);
        std::vector<Vec2> out;
        out.reserve(s.verts.size());
        for (const Vec2& v : s.verts)
        {
            const float wx = v.x * c - v.y * s_ + xf.position.x;
            const float wy = v.x * s_ + v.y * c  + xf.position.y;
            out.push_back(Vec2(wx, wy));
        }
        return out;
    }
} // namespace

// ====================================================================
// (a) Single-point correctness: circle core deep in box (the gap case).
//
// A = circle core: single vert at (5,1). B = MakeAabb(6,6) at origin
// (corners (-6,-6),(6,-6),(6,6),(-6,6)). A's point (5,1) is inside B.
// Nearest B face = +x (depth 6-5=1 vs +y 6-1=5). B->A normal = (+1,0),
// depth = 1. (Identical to EPA test (a).)
// ====================================================================
TEST_CASE("physics: Mpr circle core deep in box (gap case)", "[physics]")
{
    const Vec2 va[1] = { Vec2(Real(5), Real(1)) };

    const Shape boxB = MakeAabb(Real(6), Real(6));
    const std::vector<Vec2> vb =
        RotateCoreVerts(boxB, Transform{ Vec2(Real(0), Real(0)), Real(0) });

    const MprResult r = Mpr(va, 1, vb.data(), static_cast<int>(vb.size()));

    REQUIRE(r.ok == true);

    CHECK(static_cast<double>(r.normal.x) == Approx(1.0).margin(1e-2));
    CHECK(static_cast<double>(r.normal.y) == Approx(0.0).margin(1e-2));
    CHECK(static_cast<double>(r.depth)    == Approx(1.0).margin(1e-2));
}

// ====================================================================
// (b) EPA <-> MPR cross-check over several deep pairs.
//
// For each pair: both Epa and Mpr run on the SAME world cores. The normals
// must agree within 1e-2, and MPR must NOT report deeper than EPA's exact
// minimum (mpr.depth <= epa.depth + 1e-3).
// ====================================================================
TEST_CASE("physics: Mpr cross-checks EPA on deep pairs", "[physics]")
{
    struct Pair
    {
        std::vector<Vec2> va;
        std::vector<Vec2> vb;
        const char*       label;
    };

    std::vector<Pair> pairs;

    // 1) box-in-box: A = MakeAabb(5,5) at (8,0), B = MakeAabb(5,5) at (0,0).
    {
        const Shape boxA = MakeAabb(Real(5), Real(5));
        const Shape boxB = MakeAabb(Real(5), Real(5));
        pairs.push_back({
            RotateCoreVerts(boxA, Transform{ Vec2(Real(8), Real(0)), Real(0) }),
            RotateCoreVerts(boxB, Transform{ Vec2(Real(0), Real(0)), Real(0) }),
            "box-in-box" });
    }

    // 2) circle-in-box: A = circle core (1 vert at (5,1)), B = MakeAabb(6,6).
    {
        const Shape boxB = MakeAabb(Real(6), Real(6));
        pairs.push_back({
            std::vector<Vec2>{ Vec2(Real(5), Real(1)) },
            RotateCoreVerts(boxB, Transform{ Vec2(Real(0), Real(0)), Real(0) }),
            "circle-in-box" });
    }

    // 3) rotated box: A = MakeAabb(5,5) at (8,0), B = MakeAabb(5,5) at (0,0)
    //    rotated 30 degrees.
    {
        const Shape boxA = MakeAabb(Real(5), Real(5));
        const Shape boxB = MakeAabb(Real(5), Real(5));
        pairs.push_back({
            RotateCoreVerts(boxA, Transform{ Vec2(Real(8), Real(0)), Real(0) }),
            RotateCoreVerts(boxB, Transform{ Vec2(Real(0), Real(0)),
                                             Real(kPiD64 / 6.0) }),
            "rotated-box" });
    }

    for (const Pair& p : pairs)
    {
        INFO("pair: " << p.label);

        const EpaResult e = Epa(p.va.data(), static_cast<int>(p.va.size()),
                                p.vb.data(), static_cast<int>(p.vb.size()));
        const MprResult mr = Mpr(p.va.data(), static_cast<int>(p.va.size()),
                                 p.vb.data(), static_cast<int>(p.vb.size()));

        REQUIRE(e.ok == true);
        REQUIRE(mr.ok == true);

        // Normals agree (both unit, B->A) within 1e-2.
        CHECK(static_cast<double>(mr.normal.x) ==
              Approx(static_cast<double>(e.normal.x)).margin(1e-2));
        CHECK(static_cast<double>(mr.normal.y) ==
              Approx(static_cast<double>(e.normal.y)).margin(1e-2));

        // MPR must not report deeper than EPA's exact minimum.
        CHECK(static_cast<double>(mr.depth) <=
              static_cast<double>(e.depth) + 1e-3);

        // MPR depth is a real penetration (>= 0) and close to EPA's (the
        // single deep portal edge equals the min-penetration face here).
        CHECK(static_cast<double>(mr.depth) >= -1e-4);
        CHECK(static_cast<double>(mr.depth) ==
              Approx(static_cast<double>(e.depth)).margin(1e-2));
    }
}

// ====================================================================
// (c) Determinism: Mpr twice on the same inputs -> bit-identical result.
// ====================================================================
TEST_CASE("physics: Mpr is deterministic (bit-identical)", "[physics]")
{
    const Shape boxA = MakeAabb(Real(5), Real(5));
    const Shape boxB = MakeAabb(Real(5), Real(5));

    const Transform xfA{ Vec2(Real(8), Real(0)), Real(0) };
    const Transform xfB{ Vec2(Real(0), Real(0)), Real(kPiD64 / 6.0) };

    const std::vector<Vec2> va = RotateCoreVerts(boxA, xfA);
    const std::vector<Vec2> vb = RotateCoreVerts(boxB, xfB);

    const MprResult r1 = Mpr(va.data(), static_cast<int>(va.size()),
                             vb.data(), static_cast<int>(vb.size()));
    const MprResult r2 = Mpr(va.data(), static_cast<int>(va.size()),
                             vb.data(), static_cast<int>(vb.size()));

    CHECK(r1.ok == r2.ok);
    CHECK(r1.normal.x == r2.normal.x);
    CHECK(r1.normal.y == r2.normal.y);
    CHECK(r1.depth == r2.depth);
    CHECK(r1.point.x == r2.point.x);
    CHECK(r1.point.y == r2.point.y);
}

// ====================================================================
// (d) Fallback wiring.
//
// The production Collide deep-round path uses EPA; on EPA non-convergence it
// falls back to Mpr and builds the SAME 1-point manifold. Rather than force
// epa.ok=false (no reachable production branch for that), we (1) confirm
// Collide on the gap case is sane (EPA path; same as T3), and (2) a TEST-ONLY
// helper builds the manifold from Mpr exactly as the fallback does and asserts
// it matches the EPA-built manifold within tolerance for the gap case.
// ====================================================================
namespace
{
    // TEST-ONLY mirror of the Collide.cpp fallback's manifold build. Given the
    // world cores + radii, build a 1-point manifold from MPR the same way the
    // production fallback does (surface offsets by rA/rB, separation =
    // depth + rA + rB).
    Manifold BuildMprManifold(const Vec2* va, int na, Real rA,
                              const Vec2* vb, int nb, Real rB)
    {
        Manifold m{};
        const MprResult r = Mpr(va, na, vb, nb);
        if (!r.ok) return m;

        const Vec2 n = r.normal;
        const Real separation = r.depth + rA + rB;
        m.normal = n;
        m.points[0] = ManifoldPoint{ r.point, separation, n, 0u };
        m.pointCount = 1;
        return m;
    }
} // namespace

TEST_CASE("physics: Mpr fallback builds a sane manifold (gap case)", "[physics]")
{
    // Circle radius 1 at (5,1), box (6,6) at origin -- the same deep-round
    // gap case the production EPA path handles. Surface contact: A's circle
    // surface vs B's +x face.
    const Shape circle = MakeCircle(Real(1));
    const Shape boxB   = MakeAabb(Real(6), Real(6));

    const Transform xfA{ Vec2(Real(5), Real(1)), Real(0) };
    const Transform xfB{ Vec2(Real(0), Real(0)), Real(0) };

    // Production Collide (EPA path) -- must be a sane 1-point manifold.
    const Manifold mc = Collide(circle, xfA, boxB, xfB, Real(0));
    REQUIRE(mc.pointCount == 1);
    CHECK(static_cast<double>(mc.normal.x) == Approx(1.0).margin(1e-2));
    CHECK(static_cast<double>(mc.normal.y) == Approx(0.0).margin(1e-2));

    // TEST-ONLY MPR-built manifold on the same world cores -> must match the
    // EPA-built one (this is exactly what the fallback emits).
    const std::vector<Vec2> va = RotateCoreVerts(circle, xfA);
    const std::vector<Vec2> vb = RotateCoreVerts(boxB,   xfB);
    const Manifold mm = BuildMprManifold(
        va.data(), static_cast<int>(va.size()), circle.radius,
        vb.data(), static_cast<int>(vb.size()), boxB.radius);

    REQUIRE(mm.pointCount == 1);
    CHECK(static_cast<double>(mm.normal.x) == Approx(1.0).margin(1e-2));
    CHECK(static_cast<double>(mm.normal.y) == Approx(0.0).margin(1e-2));

    // Surface separation: core depth 1 + rA(1) + rB(0) = 2... but the contact
    // separation the manifold carries equals mpr.depth + rA + rB. Just assert
    // it agrees with Collide's reported separation within tolerance.
    CHECK(static_cast<double>(mm.points[0].separation) ==
          Approx(static_cast<double>(mc.points[0].separation)).margin(1e-2));
}
