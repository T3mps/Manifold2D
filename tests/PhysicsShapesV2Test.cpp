// Physics v2 Task 1: Unified shape core + rotation-aware AABB/mass.
//
// All expected values are hand-derived from first principles (NOT captured from
// the Lua oracle). Tolerances are 1e-3 for geometry (f32 rounding at these
// magnitudes) and 1e-2 for mass/inertia analytic cross-checks (larger absolute
// values; relative-epsilon style).
//
// What this file tests (gate for Task 1):
//   (a) MakeAabb(10,5) at rotation 45 degrees: rotation-aware AABB.
//       With c = s = cos(pi/4) = 1/sqrt(2) ~ 0.70711 the AABB half-extents
//       are |10*c| + |5*s| = 10/sqrt(2) + 5/sqrt(2) = 15/sqrt(2) ~ 10.6066
//       on BOTH axes (the rectangle is "tilted square" at 45 degrees).
//   (b) MakeCircle(4) AABB is exactly +-4 on both axes at any rotation
//       (rotation is irrelevant for a disc; test rotation 0 and 1.0 rad).
//   (c) MakeCapsule(6,2) at rotation pi/2 (90 degrees): the segment
//       (+-6,0) rotates to (0,+-6); AABB half-height = 6+2 = 8, half-width = 2.
//   (d) ComputeMass for unit-density circle r=2:
//       mass = pi*r^2 ~ 12.566, inertia = 0.5*m*r^2 ~ 25.13 (within 1e-2).
//   (e) Core vertex counts: circle->1, capsule->2, aabb->4, polygon(tri)->3.

#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <Manifold2D/Physics/Shapes.hpp>

using Catch::Approx;
using namespace Manifold2D::Physics;

namespace
{
    // Tolerance for geometry (f32 rounding at these magnitudes).
    constexpr double kGeomTol = 1e-3;
    // Tolerance for mass/inertia checks.
    constexpr double kMassTol = 1e-2;
    // Pi at f64 for hand-derived expected values.
    constexpr double kPiD = 3.14159265358979323846;
} // namespace

// ====================================================================
// (a) MakeAabb(10, 5) at rotation 45 degrees -- rotation-aware AABB.
//
// Hand derivation:
//   The four corners in local space are (+/-10, +/-5). Under a 45-degree
//   rotation (c = s = 1/sqrt(2) ~ 0.70711):
//     (10, 5) -> (10c - 5s, 10s + 5c) = (3.536, 10.607)
//     (10,-5) -> (10c + 5s, 10s - 5c) = (10.607, 3.536)
//     etc.
//   The AABB half-extents = max |rotated x|, max |rotated y| over all four corners.
//   By symmetry max = |10c| + |5s| = 10/sqrt(2) + 5/sqrt(2) = 15/sqrt(2).
//   15/sqrt(2) = 15 * 0.70711 ... = 10.6066.
//   Both half-extents equal 10.6066 (the tilted rectangle's bounding box is
//   square at 45 degrees).
// ====================================================================
TEST_CASE("physics: ShapesV2 AABB rotation-aware (aabb 45 deg)", "[physics]")
{
    const Shape s = MakeAabb(Real(10), Real(5));
    const Real angle = Real(kPiD / 4.0); // 45 degrees

    const Vec2 center(Real(3), Real(7)); // non-origin position
    const Transform xf{ center, angle };
    const Aabb box = s.ComputeAABB(xf);

    const double expectedHalfExtent = 15.0 / std::sqrt(2.0); // ~ 10.6066

    // Half-extent on x axis
    CHECK(static_cast<double>(box.max.x - center.x) ==
          Approx(expectedHalfExtent).margin(kGeomTol));
    CHECK(static_cast<double>(center.x - box.min.x) ==
          Approx(expectedHalfExtent).margin(kGeomTol));
    // Half-extent on y axis (same magnitude at 45 degrees)
    CHECK(static_cast<double>(box.max.y - center.y) ==
          Approx(expectedHalfExtent).margin(kGeomTol));
    CHECK(static_cast<double>(center.y - box.min.y) ==
          Approx(expectedHalfExtent).margin(kGeomTol));
}

// ====================================================================
// (b) MakeCircle(4): AABB is always +-4 regardless of rotation.
//
// A disc of radius r centered at (cx, cy) always has AABB
// [cx-r, cy-r, cx+r, cy+r] regardless of orientation. Test at
// rotation 0 and 1.0 rad to confirm the rotation path is correct.
// ====================================================================
TEST_CASE("physics: ShapesV2 circle AABB rotation-invariant", "[physics]")
{
    const Shape s = MakeCircle(Real(4));
    const Vec2 center(Real(0), Real(0));

    for (double angle : { 0.0, 1.0 })
    {
        const Aabb box = s.ComputeAABB(Transform{ center, Real(angle) });
        CHECK(static_cast<double>(box.min.x) == Approx(-4.0).margin(kGeomTol));
        CHECK(static_cast<double>(box.min.y) == Approx(-4.0).margin(kGeomTol));
        CHECK(static_cast<double>(box.max.x) == Approx(4.0).margin(kGeomTol));
        CHECK(static_cast<double>(box.max.y) == Approx(4.0).margin(kGeomTol));
    }
}

// ====================================================================
// (c) MakeCapsule(6, 2) at rotation pi/2 (90 degrees).
//
// Hand derivation:
//   Unrotated capsule: segment from (-6,0) to (6,0), radius 2.
//   AABB unrotated: [-8,-2, 8,2] (half-width 8, half-height 2).
//
//   At 90 degrees: the segment (+-6,0) rotates to (0,+-6).
//   Core vertex 1: (-6, 0) -> (  -6*0 - 0*1,  -6*1 + 0*0) = (0, -6)
//   Core vertex 2: ( 6, 0) -> (   6*0 - 0*1,   6*1 + 0*0) = (0,  6)
//   (Using rotation matrix: x' = x*cos - y*sin, y' = x*sin + y*cos
//    with cos(pi/2)=0, sin(pi/2)=1.)
//   After rotation the two core verts are at (0,-6) and (0,6); bound them
//   -> core box [0,−6, 0,6]; expand by radius 2:
//   AABB: [-2, -8, +2, +8] => half-width 2, half-height 8.
// ====================================================================
TEST_CASE("physics: ShapesV2 capsule AABB at 90 deg", "[physics]")
{
    const Shape s = MakeCapsule(Real(6), Real(2));
    const Vec2 center(Real(0), Real(0));
    const Real angle = Real(kPiD / 2.0); // 90 degrees

    const Aabb box = s.ComputeAABB(Transform{ center, angle });

    CHECK(static_cast<double>(box.max.x - center.x) == Approx(2.0).margin(kGeomTol));
    CHECK(static_cast<double>(center.x - box.min.x) == Approx(2.0).margin(kGeomTol));
    CHECK(static_cast<double>(box.max.y - center.y) == Approx(8.0).margin(kGeomTol));
    CHECK(static_cast<double>(center.y - box.min.y) == Approx(8.0).margin(kGeomTol));
}

// ====================================================================
// (d) ComputeMass for unit-density circle r=2.
//
// Hand derivation:
//   area     = pi * r^2     = pi * 4   ~ 12.566
//   mass     = density * area           ~ 12.566   (density=1)
//   centroid = (0, 0)
//   I_c      = 0.5 * mass * r^2 = 0.5 * 12.566 * 4 ~ 25.133
// ====================================================================
TEST_CASE("physics: ShapesV2 ComputeMass circle r=2 density=1", "[physics]")
{
    const double r = 2.0;
    const double expectedMass    = kPiD * r * r;          // ~ 12.566
    const double expectedInertia = 0.5 * expectedMass * r * r; // ~ 25.133

    const MassData md = MakeCircle(Real(r)).ComputeMass(Real(1));

    CHECK(static_cast<double>(md.mass)    == Approx(expectedMass).margin(kMassTol));
    CHECK(static_cast<double>(md.inertia) == Approx(expectedInertia).margin(kMassTol));
    CHECK(static_cast<double>(md.centroid.x) == Approx(0.0).margin(1e-5));
    CHECK(static_cast<double>(md.centroid.y) == Approx(0.0).margin(1e-5));
}

// ====================================================================
// (e) Core vertex counts: circle->1, capsule->2, aabb->4, polygon(tri)->3.
//
// After Task 1 implementation, ALL shape kinds populate the unified
// core verts vector. Circle has exactly 1 vert (the local origin),
// capsule has 2 (the segment endpoints), aabb has 4 (corner verts),
// polygon has N verts.
// ====================================================================
TEST_CASE("physics: ShapesV2 core vertex counts", "[physics]")
{
    // Circle: 1 core vertex at the local origin.
    {
        const Shape s = MakeCircle(Real(5));
        CHECK(s.verts.size() == 1u);
        // The single vert is the local origin.
        CHECK(static_cast<double>(s.verts[0].x) == Approx(0.0).margin(1e-5));
        CHECK(static_cast<double>(s.verts[0].y) == Approx(0.0).margin(1e-5));
    }

    // Capsule: 2 core verts at (-halfLen, 0) and (+halfLen, 0).
    {
        const Shape s = MakeCapsule(Real(6), Real(2));
        CHECK(s.verts.size() == 2u);
        // One vert at -6 and one at +6 on the x axis (order may vary).
        bool foundNeg = false, foundPos = false;
        for (const Vec2& v : s.verts)
        {
            if (std::fabs(static_cast<double>(v.x) - (-6.0)) < 1e-4 &&
                std::fabs(static_cast<double>(v.y)) < 1e-4)
                foundNeg = true;
            if (std::fabs(static_cast<double>(v.x) - 6.0) < 1e-4 &&
                std::fabs(static_cast<double>(v.y)) < 1e-4)
                foundPos = true;
        }
        CHECK(foundNeg);
        CHECK(foundPos);
    }

    // AABB: 4 corner verts.
    {
        const Shape s = MakeAabb(Real(10), Real(5));
        CHECK(s.verts.size() == 4u);
    }

    // Polygon (triangle): 3 verts.
    {
        const Shape s = MakePolygon({
            Vec2(Real(0),  Real(0)),
            Vec2(Real(4),  Real(0)),
            Vec2(Real(0),  Real(3)),
        });
        CHECK(s.verts.size() == 3u);
    }
}
