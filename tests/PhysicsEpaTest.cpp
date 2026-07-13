// Physics v2 EPA workstream Task 2: exact deep-overlap penetration via EPA.
//
// EPA (Expanding Polytope Algorithm) over two world cores: seeded by
// GjkOriginSimplex (Task 1), it expands a convex polytope of Minkowski-
// difference boundary points until the closest edge to the origin stops
// moving, then returns the exact penetration axis + depth + witness pair.
//
// All expected values are hand-derived from first principles (NOT captured
// from any oracle). The analytic test IS the gate for the convention.
//
// NORMAL CONVENTION (pinned by case (a)): `normal` points B -> A (the
// direction to push A out of B). `depth` is the core penetration depth along
// `normal` (>= 0); the caller adds radii. witnessA/witnessB are the core
// surface witnesses (radius NOT applied).
//
// Cases:
//   (a) Circle-core deep in box (THE gap case the centroid approx got wrong):
//       A = circle core (1 vert at (5,1)), B = MakeAabb(6,6) at origin.
//       Nearest face = B's +x (depth 6-5=1, vs +y 6-1=5). Expect
//       normal ~= (1,0), depth ~= 1, witnessB ~= (6,1) on the +x face.
//   (b) Box deep in box (cross-check vs SAT): A = MakeAabb(5,5) at (8,0),
//       B = MakeAabb(5,5) at (0,0) -> x overlap = 2. Expect normal ~= (1,0),
//       depth ~= 2.
//   (c) Rotated box deep: B rotated 30 deg; assert the min-penetration depth
//       matches the SAT-derived value (cross-checked vs the existing Collide).
//   (d) Determinism: Epa twice on the same inputs -> bit-identical result.

#include <array>
#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <Manifold2D/Physics/Narrowphase/Collide.hpp>
#include <Manifold2D/Physics/Narrowphase/Epa.hpp>
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
// (a) Circle-core deep in box (THE gap case).
//
// A = circle core: a single vert at (5,1). B = MakeAabb(6,6) at origin
// (corners (-6,-6),(6,-6),(6,6),(-6,6)). A's point (5,1) is inside B.
// Minkowski difference D = A - B is the box reflected+translated:
//   D = (5,1) - box  ->  spans [-1,11] x [-5,7].
// Origin (0,0) is inside D. Closest D edge to origin is the left edge at
// x=-1 (distance 1), vs right x=11, bottom y=-5, top y=7. So the MD outward
// normal is (-1,0); B->A normal = -(MD outward) = (+1,0), depth = 1.
// Witness: pa = (5,1) (single vert), pb = pa - md = (5,1)-(-1,0) = (6,1).
// ====================================================================
TEST_CASE("physics: Epa circle core deep in box (gap case)", "[physics]")
{
    // A: circle core = single vertex at (5,1).
    const Vec2 va[1] = { Vec2(Real(5), Real(1)) };

    // B: MakeAabb(6,6) at origin.
    const Shape boxB = MakeAabb(Real(6), Real(6));
    const std::vector<Vec2> vb =
        RotateCoreVerts(boxB, Transform{ Vec2(Real(0), Real(0)), Real(0) });

    const EpaResult r = Epa(va, 1, vb.data(), static_cast<int>(vb.size()));

    REQUIRE(r.ok == true);

    // B->A normal must be the exact +x face normal (the OLD centroid approx
    // gave ~(0.98, 0.196) -- this pins that EPA fixes it).
    CHECK(static_cast<double>(r.normal.x) == Approx(1.0).margin(kTol));
    CHECK(static_cast<double>(r.normal.y) == Approx(0.0).margin(kTol));

    // Core penetration depth = 6 - 5 = 1.
    CHECK(static_cast<double>(r.depth) == Approx(1.0).margin(kTol));

    // witnessB on B's +x face at (6,1).
    CHECK(static_cast<double>(r.witnessB.x) == Approx(6.0).margin(1e-2));
    CHECK(static_cast<double>(r.witnessB.y) == Approx(1.0).margin(1e-2));

    // witnessA is A's single core vert (5,1).
    CHECK(static_cast<double>(r.witnessA.x) == Approx(5.0).margin(1e-2));
    CHECK(static_cast<double>(r.witnessA.y) == Approx(1.0).margin(1e-2));
}

// ====================================================================
// (b) Box deep in box (cross-check vs SAT).
//
// A = MakeAabb(5,5) at (8,0): x in [3,13]. B = MakeAabb(5,5) at (0,0):
// x in [-5,5]. Overlap along x = 5 - 3 = 2 (the smaller separation axis;
// y fully overlaps). Push A out of B in +x by 2. normal=(1,0), depth=2.
// ====================================================================
TEST_CASE("physics: Epa box deep in box vs SAT", "[physics]")
{
    const Shape boxA = MakeAabb(Real(5), Real(5));
    const Shape boxB = MakeAabb(Real(5), Real(5));

    const std::vector<Vec2> va =
        RotateCoreVerts(boxA, Transform{ Vec2(Real(8), Real(0)), Real(0) });
    const std::vector<Vec2> vb =
        RotateCoreVerts(boxB, Transform{ Vec2(Real(0), Real(0)), Real(0) });

    const EpaResult r = Epa(va.data(), static_cast<int>(va.size()),
                            vb.data(), static_cast<int>(vb.size()));

    REQUIRE(r.ok == true);
    CHECK(static_cast<double>(r.normal.x) == Approx(1.0).margin(kTol));
    CHECK(static_cast<double>(r.normal.y) == Approx(0.0).margin(kTol));
    CHECK(static_cast<double>(r.depth) == Approx(2.0).margin(kTol));
}

// ====================================================================
// (c) Rotated box deep -- cross-check the min penetration vs Collide's SAT.
//
// A = MakeAabb(5,5) at (8,0) axis-aligned. B = MakeAabb(5,5) at (0,0)
// rotated 30 degrees. Both shapes have radius 0 (aabb cores), so the
// core EPA depth must equal the (positive) separation Collide's SAT
// reference-face path reports for the same pair. We derive the expected
// depth from Collide (the established SAT path) and assert EPA agrees.
// ====================================================================
TEST_CASE("physics: Epa rotated box deep vs Collide SAT", "[physics]")
{
    const Shape boxA = MakeAabb(Real(5), Real(5));
    const Shape boxB = MakeAabb(Real(5), Real(5));

    const Transform xfA{ Vec2(Real(8), Real(0)), Real(0) };
    const Transform xfB{ Vec2(Real(0), Real(0)), Real(kPiD64 / 6.0) }; // 30 deg

    const std::vector<Vec2> va = RotateCoreVerts(boxA, xfA);
    const std::vector<Vec2> vb = RotateCoreVerts(boxB, xfB);

    const EpaResult r = Epa(va.data(), static_cast<int>(va.size()),
                            vb.data(), static_cast<int>(vb.size()));
    REQUIRE(r.ok == true);

    // Cross-check vs the established SAT path (Collide). For two zero-radius
    // boxes, Collide's max positive separation == the min penetration depth
    // EPA finds. The deepest contact point separation is the penetration.
    const Manifold m = Collide(boxA, xfA, boxB, xfB, Real(0));
    REQUIRE(m.pointCount > 0);
    double satDepth = 0.0;
    for (int i = 0; i < m.pointCount; ++i)
        satDepth = std::max(satDepth, static_cast<double>(m.points[i].separation));

    CHECK(static_cast<double>(r.depth) == Approx(satDepth).margin(1e-3));

    // The EPA normal must agree with Collide's normal (B->A) within 1e-2
    // (both point along the minimum-penetration axis).
    CHECK(static_cast<double>(r.normal.x) ==
          Approx(static_cast<double>(m.normal.x)).margin(1e-2));
    CHECK(static_cast<double>(r.normal.y) ==
          Approx(static_cast<double>(m.normal.y)).margin(1e-2));
}

// ====================================================================
// (d) Determinism: Epa twice on the same inputs -> bit-identical result.
// ====================================================================
TEST_CASE("physics: Epa is deterministic (bit-identical)", "[physics]")
{
    const Shape boxA = MakeAabb(Real(5), Real(5));
    const Shape boxB = MakeAabb(Real(5), Real(5));

    const Transform xfA{ Vec2(Real(8), Real(0)), Real(0) };
    const Transform xfB{ Vec2(Real(0), Real(0)), Real(kPiD64 / 6.0) };

    const std::vector<Vec2> va = RotateCoreVerts(boxA, xfA);
    const std::vector<Vec2> vb = RotateCoreVerts(boxB, xfB);

    const EpaResult r1 = Epa(va.data(), static_cast<int>(va.size()),
                             vb.data(), static_cast<int>(vb.size()));
    const EpaResult r2 = Epa(va.data(), static_cast<int>(va.size()),
                             vb.data(), static_cast<int>(vb.size()));

    CHECK(r1.ok == r2.ok);
    CHECK(r1.normal.x == r2.normal.x);
    CHECK(r1.normal.y == r2.normal.y);
    CHECK(r1.depth == r2.depth);
    CHECK(r1.witnessA.x == r2.witnessA.x);
    CHECK(r1.witnessA.y == r2.witnessA.y);
    CHECK(r1.witnessB.x == r2.witnessB.x);
    CHECK(r1.witnessB.y == r2.witnessB.y);
}
