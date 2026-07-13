// Physics v2 EPA workstream Task 3: EPA-exact deep round-in-poly contacts.
//
// The unified Collide()'s ROUND fast path (circle/capsule core, <=2 verts)
// previously approximated a DEEP overlap (cores intersecting, GJK distance ~0)
// with a centroid-to-centroid normal + centroid witnesses. That is wrong for a
// round core buried inside a polygon: the centroid guess points along the line
// of centres, NOT toward the nearest face. Task 3 replaces that branch with a
// call to Epa() -> the exact nearest-face normal/depth.
//
// All expected values are HAND-DERIVED from first principles (the surface
// convention is read off the existing SHALLOW round branch in CollideRound:
//   surfaceA = witnessA - normal*rA,  surfaceB = witnessB + normal*rB,
//   contact  = midpoint(surfaceA, surfaceB),
//   separation positive = penetrating).
//
// Cases:
//   (a) THE gap fix -- circle(2) core at (5,1) deep in MakeAabb(6,6) at origin.
//       EPA gives normal=(1,0) (B->A), core depth=1, witnessA=(5,1),
//       witnessB=(6,1). Surface separation = depth + rA + rB = 1+2+0 = 3.
//       Contrast: the OLD centroid approx gave normal ~ (0.98, 0.196) (the
//       direction from B's centroid (0,0) to A's centroid (5,1)).
//   (b) Capsule deep in box at an angle -> 1-point manifold, normal aligned to
//       the nearest face within 1e-2.
//   (c) Shallow round UNCHANGED -- a circle just outside the box (within
//       specMargin) still goes through the GJK-witness shallow path (1
//       speculative point, NEGATIVE separation): proves ONLY the deep branch
//       changed.

#include <array>
#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <Manifold2D/Physics/Narrowphase/Collide.hpp>
#include <Manifold2D/Physics/Narrowphase/Manifold.hpp>
#include <Manifold2D/Physics/Shapes.hpp>

using Catch::Approx;
using namespace Manifold2D::Physics;

namespace
{
    constexpr double kTol   = 1e-3;
    constexpr double kPiD64 = 3.14159265358979323846;
} // namespace

// ====================================================================
// (a) THE gap fix: circle(2) deep inside MakeAabb(6,6) at origin.
//
// A = MakeCircle(2) at (5,1): core is a single vert at (5,1), radius 2.
// B = MakeAabb(6,6) at origin: x,y in [-6,6], radius 0.
// A's core point (5,1) is inside B -> cores overlap -> the DEEP branch.
//
// EPA (hand-derived in PhysicsEpaTest case (a)):
//   Nearest B face is +x (depth 6-5 = 1, smaller than +y's 6-1 = 5).
//   normal (B->A) = (1, 0).   core depth = 1.
//   witnessA = (5,1) (A's single core vert).   witnessB = (6,1) on B's +x face.
//
// Surface convention (mirrors CollideRound's shallow branch):
//   surfaceA = witnessA - normal*rA = (5,1) - (1,0)*2 = (3,1)
//   surfaceB = witnessB + normal*rB = (6,1) + (1,0)*0 = (6,1)
//   contact  = midpoint = (4.5, 1)
//   separation = depth + rA + rB = 1 + 2 + 0 = 3   (positive = penetrating)
//
// The OLD centroid approx gave normal ~= (0.98, 0.196) (B centroid (0,0) -> A
// centroid (5,1) normalised) -- that is the bug this test pins as fixed.
// ====================================================================
TEST_CASE("physics: CollideDeepRound circle deep in box (gap fix)", "[physics]")
{
    const Shape circle = MakeCircle(Real(2));
    const Shape box    = MakeAabb(Real(6), Real(6));

    const Transform xfA{ Vec2(Real(5), Real(1)), Real(0) };
    const Transform xfB{ Vec2(Real(0), Real(0)), Real(0) };

    const Manifold m = Collide(circle, xfA, box, xfB, Real(0));

    // 1-point manifold (circle core -> single contact).
    REQUIRE(m.pointCount == 1);

    // Normal B->A is the EXACT +x face normal (the old centroid approx gave
    // ~(0.98, 0.196) -- this pins that EPA fixes it).
    CHECK(static_cast<double>(m.normal.x) == Approx(1.0).margin(kTol));
    CHECK(static_cast<double>(m.normal.y) == Approx(0.0).margin(kTol));
    CHECK(static_cast<double>(m.points[0].normal.x) == Approx(1.0).margin(kTol));
    CHECK(static_cast<double>(m.points[0].normal.y) == Approx(0.0).margin(kTol));

    // Surface penetration = core depth + rA + rB = 1 + 2 + 0 = 3 (positive).
    CHECK(static_cast<double>(m.points[0].separation) == Approx(3.0).margin(kTol));
    CHECK(static_cast<double>(m.points[0].separation) > 0.0);

    // Contact point = midpoint of surfaceA (3,1) and surfaceB (6,1) = (4.5, 1).
    CHECK(static_cast<double>(m.points[0].point.x) == Approx(4.5).margin(1e-2));
    CHECK(static_cast<double>(m.points[0].point.y) == Approx(1.0).margin(1e-2));
}

// ====================================================================
// (b) Capsule deep in box at an angle.
//
// A = MakeCapsule(halfLen=2, r=1) at (4, 0.5) rotated 30 deg: its 2-vert core
// segment lies near the box's +x interior. B = MakeAabb(6,6) at origin.
// The capsule core is fully inside B (the rotated segment endpoints are within
// |x|,|y| < 6). The nearest face is B's +x (the capsule centre is at x=4,
// closest to the +x wall at x=6).
//
// We don't pin the exact depth (it depends on the rotated support vertex), but
// the DEEP-branch EPA normal must align with B's +x face normal (1,0) within
// 1e-2, and the manifold must be a single penetrating point. The OLD centroid
// approx would point from B's centroid (0,0) toward the capsule centroid
// (~(4,0.5)) -- normal ~ (0.992, 0.124), wrong on the y-component.
// ====================================================================
TEST_CASE("physics: CollideDeepRound capsule deep in box at angle", "[physics]")
{
    const Shape capsule = MakeCapsule(Real(2), Real(1));
    const Shape box      = MakeAabb(Real(6), Real(6));

    const Transform xfA{ Vec2(Real(4), Real(0.5)), Real(kPiD64 / 6.0) }; // 30 deg
    const Transform xfB{ Vec2(Real(0), Real(0)),   Real(0) };

    const Manifold m = Collide(capsule, xfA, box, xfB, Real(0));

    // Single penetrating contact (round path -> 1 point in the deep branch).
    REQUIRE(m.pointCount == 1);
    CHECK(static_cast<double>(m.points[0].separation) > 0.0);

    // Normal aligned to the nearest face (+x of B) within 1e-2.
    CHECK(static_cast<double>(m.normal.x) == Approx(1.0).margin(1e-2));
    CHECK(static_cast<double>(m.normal.y) == Approx(0.0).margin(1e-2));
}

// ====================================================================
// (c) Shallow round UNCHANGED: circle just OUTSIDE the box, within the
//     speculative margin -> the GJK-witness SHALLOW path must still run.
//
// A = MakeCircle(2) at (8.5, 0): core vert at (8.5, 0), radius 2.
// B = MakeAabb(6,6) at origin: +x face at x=6.
//   coreDist = 8.5 - 6 = 2.5 (witnessB (6,0) to witnessA (8.5,0)).
//   totalR   = rA + rB = 2 + 0 = 2.
//   surface gap = coreDist - totalR = 2.5 - 2 = 0.5 > 0 -> a GAP, not overlap.
// With speculativeMargin = 1.0 (>= 0.5) this is a SPECULATIVE shallow contact:
//   1 point, NEGATIVE separation = totalR - coreDist = 2 - 2.5 = -0.5,
//   normal (B->A) = (1, 0) (from witnessB toward witnessA).
//
// This proves ONLY the deep branch changed: a non-overlapping round case still
// flows through the GJK witness path (NOT EPA -- EPA requires overlap).
// ====================================================================
TEST_CASE("physics: CollideDeepRound shallow round unchanged (speculative)", "[physics]")
{
    const Shape circle = MakeCircle(Real(2));
    const Shape box    = MakeAabb(Real(6), Real(6));

    const Transform xfA{ Vec2(Real(8.5), Real(0)), Real(0) };
    const Transform xfB{ Vec2(Real(0),   Real(0)), Real(0) };

    const Manifold m = Collide(circle, xfA, box, xfB, Real(1.0));

    // Shallow speculative path: exactly 1 point.
    REQUIRE(m.pointCount == 1);

    // NEGATIVE separation (a gap, not penetration) -- the speculative branch.
    CHECK(static_cast<double>(m.points[0].separation) == Approx(-0.5).margin(kTol));
    CHECK(static_cast<double>(m.points[0].separation) < 0.0);

    // Normal still B->A = (+1, 0) (witnessB (6,0) -> witnessA (8.5,0)).
    CHECK(static_cast<double>(m.normal.x) == Approx(1.0).margin(kTol));
    CHECK(static_cast<double>(m.normal.y) == Approx(0.0).margin(kTol));
}
