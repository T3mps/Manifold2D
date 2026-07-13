// Physics smoke test: the first ported math kernels + value-type basics.
//
// HISTORY: this file once proved the Lua-oracle fixture pipeline (path + JSON
// loader) against captured reference numbers. Physics v2 Task T8 retired that
// oracle gate (the oracle was an M6 porting scaffold). The smoke coverage that
// MATTERS -- the ClosestOnSegment kernel, the kMaxPolyVerts constant, and the
// BodyHandle / Math basics -- is preserved here with SELF-CONTAINED analytic
// assertions hand-derived from first principles (no captured-oracle dependency).
//
// PRESENTATION-FREE + C++20-clean.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Manifold2D/Physics/Math.hpp>
#include <Manifold2D/Physics/PhysicsTypes.hpp>

using Catch::Approx;
using namespace Manifold2D::Physics;

namespace
{
    // Geometry tolerance for the analytic kernel checks below (f32 rounding at
    // these small pixel-scale magnitudes is well under this).
    constexpr double kTol = 1e-5;
}

TEST_CASE("physics: kMaxPolyVerts is the agreed polygon vertex cap", "[physics]")
{
    // The polygon vertex cap is a hard contract shared by Shapes/Manifold/SAT.
    // (Formerly cross-checked against the Lua MAX_POLY_VERTS via shapes.json;
    // now pinned directly -- the constant is the single source of truth.)
    CHECK(kMaxPolyVerts >= 3u);
    CHECK(kMaxPolyVerts == 128u); // the agreed cap (matches the Lua MAX_POLY_VERTS)
}

TEST_CASE("physics: BodyHandle + types basics", "[physics]")
{
    // Handle equality / null sentinel behavior (the slot-reuse invariant is
    // exercised in full by later World tests; here we pin the value type).
    BodyHandle a{ 3u, 1u };
    BodyHandle b{ 3u, 1u };
    BodyHandle c{ 3u, 2u };   // same slot, bumped generation -> distinct
    CHECK(a == b);
    CHECK(a != c);
    CHECK(kInvalidBody == BodyHandle{ 0u, 0u });
    CHECK(static_cast<int>(BodyType::Dynamic) == 2);
}

TEST_CASE("physics: Math scalar/vector helpers", "[physics]")
{
    CHECK(Math::Clamp(5, 0, 10) == 5);
    CHECK(Math::Clamp(-3, 0, 10) == 0);
    CHECK(Math::Clamp(42, 0, 10) == 10);

    // cross2(a,b) = a.x*b.y - a.y*b.x
    CHECK(Math::Cross2(Vec2(1, 0), Vec2(0, 1)) == Approx(1.0));
    CHECK(Math::Cross2(Vec2(0, 1), Vec2(1, 0)) == Approx(-1.0));

    // perp(v) = (v.y, -v.x)
    const Vec2 p = Math::Perp(Vec2(2, 3));
    CHECK(p.x == Approx(3.0));
    CHECK(p.y == Approx(-2.0));
}

// ---------------------------------------------------------------------------
// Math::ClosestOnSegment -- hand-derived analytic cases (was the geometry
// oracle's closestOnSegment block; the four scenarios are reproduced here from
// first principles, no captured fixture).
// ---------------------------------------------------------------------------
TEST_CASE("physics: Math::ClosestOnSegment analytic cases", "[physics]")
{
    // midpoint: p projects onto the interior of the segment.
    //   segment a=(0,0) -> b=(10,0); p=(5,4). Closest point is the foot of the
    //   perpendicular at (5,0), parametric t = 5/10 = 0.5.
    {
        const Math::ClosestPoint r =
            Math::ClosestOnSegment(Vec2(5, 4), Vec2(0, 0), Vec2(10, 0));
        CHECK(static_cast<double>(r.point.x) == Approx(5.0).margin(kTol));
        CHECK(static_cast<double>(r.point.y) == Approx(0.0).margin(kTol));
        CHECK(static_cast<double>(r.t)       == Approx(0.5).margin(kTol));
    }

    // clamp_lo: p projects BEFORE a -> clamps to a, t = 0.
    //   segment a=(0,0) -> b=(10,0); p=(-3,2). Foot would be x=-3 (t<0); clamp
    //   to the start vertex a=(0,0), t=0.
    {
        const Math::ClosestPoint r =
            Math::ClosestOnSegment(Vec2(-3, 2), Vec2(0, 0), Vec2(10, 0));
        CHECK(static_cast<double>(r.point.x) == Approx(0.0).margin(kTol));
        CHECK(static_cast<double>(r.point.y) == Approx(0.0).margin(kTol));
        CHECK(static_cast<double>(r.t)       == Approx(0.0).margin(kTol));
    }

    // clamp_hi: p projects AFTER b -> clamps to b, t = 1.
    //   segment a=(0,0) -> b=(10,0); p=(14,-2). Foot would be x=14 (t>1); clamp
    //   to the end vertex b=(10,0), t=1.
    {
        const Math::ClosestPoint r =
            Math::ClosestOnSegment(Vec2(14, -2), Vec2(0, 0), Vec2(10, 0));
        CHECK(static_cast<double>(r.point.x) == Approx(10.0).margin(kTol));
        CHECK(static_cast<double>(r.point.y) == Approx(0.0).margin(kTol));
        CHECK(static_cast<double>(r.t)       == Approx(1.0).margin(kTol));
    }

    // degenerate: zero-length segment (a == b). The closest point is that point,
    // t conventionally 0 (no direction to project along).
    {
        const Math::ClosestPoint r =
            Math::ClosestOnSegment(Vec2(5, 5), Vec2(2, 3), Vec2(2, 3));
        CHECK(static_cast<double>(r.point.x) == Approx(2.0).margin(kTol));
        CHECK(static_cast<double>(r.point.y) == Approx(3.0).margin(kTol));
        CHECK(static_cast<double>(r.t)       == Approx(0.0).margin(kTol));
    }
}
