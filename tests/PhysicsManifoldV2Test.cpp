// Physics v2 Task 3: Unified Collide + reference-face clip manifold with stable feature IDs.
//
// All expected values are hand-derived from first principles. Tolerances
// are 1e-3 for axis-aligned cases and 1e-2 for rotated cases.
//
// Tests:
//   (a) Two MakeAabb(10,10) at (0,0) and (18,0) -- axis-aligned overlap 2.
//       Normal B->A = (-1,0). Two contact points on shared face.
//   (b) Same pair with B rotated 30 degrees -- partial overlap, hand-derived.
//   (c) MakeCircle(4) at (0,0) vs MakeAabb(5,5) at (8,0) -- 1 overlap contact.
//   (d) Feature-id stability sweep: same feature pair across 10 positions.
//
// Collide is the v2 unified narrowphase entry (Task 3). It is now the SOLE
// narrowphase: T5 wired it into PhysicsWorld::GenerateContacts and T8 deleted
// the old CollidePolygons/Sat/Specialized path. The full [physics] suite is green.

#include <array>
#include <cmath>
#include <set>
#include <utility>
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
    constexpr double kTol  = 1e-3; // axis-aligned cases
    constexpr double kTolR = 1e-2; // rotated cases
    // Pi at f64 for hand-derived expected values.
    constexpr double kPiD64 = 3.14159265358979323846;
} // namespace

// ====================================================================
// (a) MakeAabb(10,10) at (0,0) [A] vs MakeAabb(10,10) at (18,0) [B].
//     Axis-aligned, no rotation.
//
// Hand derivation:
//   A spans x in [-10, +10], y in [-10, +10].
//   B spans x in [8,   +28], y in [-10, +10].
//   Overlap on x-axis: A right face = 10, B left face = 8. Overlap = 2.
//   No separation on y-axis (shared extent [-10, +10]).
//   SAT reference axis: x-axis (overlap = 2 < y-overlap = 20).
//
//   Normal B->A: A is LEFT of B. Pushing A out of B means pushing A further
//   LEFT -> normal = (-1, 0).
//
//   Reference-face clip: reference face is A's right face (x=10, spans
//   y=[-10,+10]) or equivalently B's left face (x=8, spans y=[-10,+10]).
//   After clipping the incident edge of B (the left face, y=[-10,+10]) against
//   A's right face side planes (y=-10 and y=+10), 2 contact points survive:
//   at (approximately 9,  -10) and (approximately 9, +10), or some averaging
//   along the face depending on implementation. What we can assert:
//     - pointCount == 2
//     - normal ~ (-1, 0) (within 1e-3)
//     - both points have separation ~ 2.0 (within 1e-3)  [penetration positive]
//     - normal component of contact points is in the overlap band
//
// NOTE: the exact x-coordinate of the contact points depends on implementation
// (B's face at x=8, mid-overlap at x=9, or A's face at x=10). We assert the
// semantically important properties: count, normal, separation, and that the
// y-coordinates are +-10 (the clipped face extents).
// ====================================================================
TEST_CASE("physics: ManifoldV2 axis-aligned box-box overlap", "[physics]")
{
    const Shape boxA = MakeAabb(Real(10), Real(10));
    const Shape boxB = MakeAabb(Real(10), Real(10));

    const Transform xfA{ Vec2(Real(0),  Real(0)),  Real(0) };
    const Transform xfB{ Vec2(Real(18), Real(0)),  Real(0) };

    const Manifold m = Collide(boxA, xfA, boxB, xfB);

    // Must find exactly 2 contact points (two clipped verts on the shared face).
    REQUIRE(m.pointCount == 2);

    // Representative normal must be (-1, 0) within tolerance: push A (left)
    // further left, i.e. away from B (right). B->A direction = (-1, 0).
    CHECK(static_cast<double>(m.normal.x) == Approx(-1.0).margin(kTol));
    CHECK(static_cast<double>(m.normal.y) == Approx(0.0).margin(kTol));

    // Both per-point normals also (-1, 0).
    for (int i = 0; i < m.pointCount; ++i)
    {
        CHECK(static_cast<double>(m.points[i].normal.x) == Approx(-1.0).margin(kTol));
        CHECK(static_cast<double>(m.points[i].normal.y) == Approx(0.0).margin(kTol));
    }

    // Both points should have separation ~ 2.0 (penetration depth = overlap).
    for (int i = 0; i < m.pointCount; ++i)
    {
        CHECK(static_cast<double>(m.points[i].separation) ==
              Approx(2.0).margin(kTol));
    }

    // The two contact points are on opposite ends of the shared face (y ~ +-10).
    // Extract and sort by y to assert them independently.
    std::array<double, 2> ys{
        static_cast<double>(m.points[0].point.y),
        static_cast<double>(m.points[1].point.y)
    };
    if (ys[0] > ys[1]) std::swap(ys[0], ys[1]);

    CHECK(ys[0] == Approx(-10.0).margin(kTol));
    CHECK(ys[1] == Approx(+10.0).margin(kTol));
}

// ====================================================================
// (b1) MakeAabb(10,10) at (0,0) [A] vs MakeAabb(10,10) at (18,0) [B],
//      B rotated 30 degrees CCW — smoke-check that contact is detected
//      and penetration is positive. (The exact SAT winner is
//      implementation-defined; see (b2) for the rotation gate.)
// ====================================================================
TEST_CASE("physics: ManifoldV2 rotated box-box smoke", "[physics]")
{
    const Shape boxA = MakeAabb(Real(10), Real(10));
    const Shape boxB = MakeAabb(Real(10), Real(10));

    const Transform xfA{ Vec2(Real(0),  Real(0)),  Real(0) };
    const Transform xfB{ Vec2(Real(18), Real(0)),  Real(kPiD64 / 6.0) }; // 30 deg

    const Manifold m = Collide(boxA, xfA, boxB, xfB);

    // Must detect overlap.
    REQUIRE(m.pointCount >= 1);

    // Normal x-component negative: A is left of B, B->A direction is leftward.
    CHECK(static_cast<double>(m.normal.x) < 0.0);

    // All contact points must have positive separation (real penetration).
    for (int i = 0; i < m.pointCount; ++i)
    {
        CHECK(static_cast<double>(m.points[i].separation) > 0.0);
    }
}

// ====================================================================
// (b2) MakeCircle(2) [A] vs MakeAabb(5,5) [B] rotated 45 degrees CCW.
//      This is the rotation gate: it pins the contact normal to a
//      non-axis-aligned direction so a non-rotating implementation fails.
//
// Hand derivation:
//   B = MakeAabb(5,5) at origin, rotation = π/4 (45°).
//   B's local verts are the four corners of the 5×5 half-extent box.
//   Under 45° rotation the local +x face (outward normal (1,0) in local
//   space, face at x=+5) maps to:
//     outward normal n = (cos45, sin45) = (√2/2, √2/2) ≈ (0.70711, 0.70711)
//     face vertices in world: (7.0711, 0) and (0, 7.0711)
//                             (the two rotated corners adjacent to that face)
//
//   A = MakeCircle(2) centred at n*6 = (6*0.70711, 6*0.70711) ≈ (4.2426, 4.2426).
//
//   GJK (round-shape path, circle core = 1 vert):
//     Circle core point: (4.2426, 4.2426).
//     Closest point on B's +x face (edge from (7.071,0) to (0,7.071)):
//       face tangent direction: (0-7.071, 7.071-0) normalised = (-0.70711, 0.70711)
//       t = dot((4.2426-7.071, 4.2426-0), (-0.70711, 0.70711))
//         = (-2.8284)*(-0.70711) + 4.2426*(0.70711)
//         = 2.0 + 3.0 = 5.0
//       face length = 10.0, so t=5 places us at the midpoint:
//         (7.071 + 5*(-0.70711), 0 + 5*0.70711) = (3.5355, 3.5355)
//     GJK distance = |(4.2426-3.5355, 4.2426-3.5355)| = |(0.7071, 0.7071)| = 1.0
//
//   totalR = rA + rB = 2 + 0 = 2.
//   coreDist = 1.0 < totalR = 2.0 → real penetration.
//   separation = totalR - coreDist = 2 - 1 = 1.0.
//
//   Normal (B→A, from GJK witness B-point to A-point):
//     (4.2426-3.5355, 4.2426-3.5355) / 1.0 = (0.7071, 0.7071)
//     = n = (cos45, sin45) ≈ (0.70711, 0.70711)
//   BOTH components ≈ 0.70711 within 1e-2 — a non-rotating B would give (1,0).
//
//   Contact point:
//     surfaceA = ptA - normal*rA = (4.2426,4.2426) - (0.7071,0.7071)*2
//              = (4.2426-1.4142, 4.2426-1.4142) = (2.8284, 2.8284)
//     surfaceB = ptB + normal*rB = (3.5355,3.5355) + (0.7071,0.7071)*0
//              = (3.5355, 3.5355)
//     cp = midpoint = ((2.8284+3.5355)/2, (2.8284+3.5355)/2)
//        = (3.1820, 3.1820)
//   We assert each component is in (2.5, 4.0) — wide enough to be robust,
//   tight enough to prove rotation is working.
// ====================================================================
TEST_CASE("physics: ManifoldV2 circle vs 45-deg-rotated box (rotation gate)", "[physics]")
{
    const Shape circle = MakeCircle(Real(2));
    const Shape box    = MakeAabb(Real(5), Real(5));

    // n = (cos45, sin45)
    const double kCos45 = std::sqrt(2.0) / 2.0; // ≈ 0.70711

    const Transform xfA{ Vec2(Real(kCos45 * 6.0), Real(kCos45 * 6.0)), Real(0) };
    const Transform xfB{ Vec2(Real(0),             Real(0)),             Real(kPiD64 / 4.0) }; // 45 deg

    const Manifold m = Collide(circle, xfA, box, xfB);

    // Exactly 1 contact point (circle round path always gives 1).
    REQUIRE(m.pointCount == 1);

    // Normal must be ≈ (0.70711, 0.70711) — B→A direction along B's rotated
    // face normal. BOTH components must be ≈ 0.70711 within 1e-2.
    // (A non-rotating box would yield normal ≈ (1,0), so asserting ny ≈ 0.707
    // is the rotation gate.)
    CHECK(static_cast<double>(m.normal.x) == Approx(kCos45).margin(kTolR));
    CHECK(static_cast<double>(m.normal.y) == Approx(kCos45).margin(kTolR));

    // Separation ≈ 1.0 (penetration depth = totalR − coreDist = 2 − 1).
    CHECK(static_cast<double>(m.points[0].separation) == Approx(1.0).margin(kTolR));

    // Contact point components both in (2.5, 4.0) — on B's rotated face,
    // between the box surface witness (3.535, 3.535) and circle surface (2.828, 2.828).
    CHECK(static_cast<double>(m.points[0].point.x) > 2.5);
    CHECK(static_cast<double>(m.points[0].point.x) < 4.0);
    CHECK(static_cast<double>(m.points[0].point.y) > 2.5);
    CHECK(static_cast<double>(m.points[0].point.y) < 4.0);
}

// ====================================================================
// (c) MakeCircle(4) at (0,0) [A] vs MakeAabb(5,5) at (8,0) [B].
//
// Hand derivation:
//   Circle radius = 4. Box half-extents (5,5) centred at (8,0).
//   Box left face at x = 8-5 = 3.
//   Circle surface toward B: at x = +4.
//   Overlap: circle reaches x=4, box left face at x=3. Overlap = 4-3 = 1.
//   Separation (penetration) = 1.
//
//   Normal B->A: B is to the RIGHT of A -> push A further left -> (-1, 0).
//
//   Contact point: the unique contact on the circle-box interface.
//   The circle core has 1 vert (origin); the box left face spans y in [-5,+5].
//   Circle center (0,0) projects perpendicularly onto box left face at (3, 0).
//   Round-shape fast path: circle (1 core vert) -> single speculative/overlap point.
//   Point on box surface towards circle: (3, 0).
//   Point on circle surface toward box: (4, 0).
//   Contact point: midpoint or one of the two witnesses, at approximately (3.5, 0).
//   We assert x in range (2.5, 4.5) to cover any reasonable placement.
// ====================================================================
TEST_CASE("physics: ManifoldV2 circle vs aabb overlap", "[physics]")
{
    const Shape circle = MakeCircle(Real(4));
    const Shape box    = MakeAabb(Real(5), Real(5));

    const Transform xfA{ Vec2(Real(0), Real(0)), Real(0) };
    const Transform xfB{ Vec2(Real(8), Real(0)), Real(0) };

    const Manifold m = Collide(circle, xfA, box, xfB);

    // Must find exactly 1 contact point (circle is a point core).
    REQUIRE(m.pointCount == 1);

    // Normal: B is right of A -> B->A = (-1, 0).
    CHECK(static_cast<double>(m.normal.x) == Approx(-1.0).margin(kTol));
    CHECK(static_cast<double>(m.normal.y) == Approx(0.0).margin(kTol));

    // Separation = 1.0 (penetration depth = overlap).
    CHECK(static_cast<double>(m.points[0].separation) == Approx(1.0).margin(kTol));

    // Contact point x in (2.5, 4.5): somewhere between box face (x=3) and circle surface (x=4).
    CHECK(static_cast<double>(m.points[0].point.x) > 2.5);
    CHECK(static_cast<double>(m.points[0].point.x) < 4.5);
    // Contact point y ~ 0 (axially symmetric).
    CHECK(static_cast<double>(m.points[0].point.y) == Approx(0.0).margin(kTol));
}

// ====================================================================
// (d) Feature-id stability: sweep B from x=18 to x=17 over 10 steps,
//     both A and B are MakeAabb(10,10). Face-to-face contact throughout.
//
// At every step the SAME physical contact feature pair (A's right face,
// B's left face) is active. The stable-feature-id encoding must yield
// IDENTICAL manifold point ids across all 10 steps.
//
// This is the warm-start gate: the old rank-based id (rank 1 / rank 2)
// changes as the deepest point changes with depth -- it fails this test.
// The new feature-pair id (reference-face-index, incident-vertex-index)
// is stable as long as the same features are in contact.
// ====================================================================
TEST_CASE("physics: ManifoldV2 feature-id stable across sweep", "[physics]")
{
    const Shape boxA = MakeAabb(Real(10), Real(10));
    const Shape boxB = MakeAabb(Real(10), Real(10));
    const Transform xfA{ Vec2(Real(0), Real(0)), Real(0) };

    // Collect {id0, id1} sets (sorted by id value) across 10 steps.
    // All steps must yield the same set of ids.
    bool firstStep = true;
    std::set<uint32_t> expectedIds;

    for (int step = 0; step < 10; ++step)
    {
        // B centre sweeps from x=18 down to x=17 (overlap 2..3).
        const Real bx = Real(18) - Real(step) * Real(0.1f);
        const Transform xfB{ Vec2(bx, Real(0)), Real(0) };

        const Manifold m = Collide(boxA, xfA, boxB, xfB);

        REQUIRE(m.pointCount == 2);

        std::set<uint32_t> thisIds;
        for (int i = 0; i < m.pointCount; ++i)
            thisIds.insert(m.points[i].id);

        if (firstStep)
        {
            expectedIds = thisIds;
            firstStep = false;
        }
        else
        {
            // The id SET must be identical to the first step's set.
            CHECK(thisIds == expectedIds);
        }
    }
}

// ====================================================================
// (d2) Feature-id stability under small constant rotation: same depth-sweep
//      as (d) but with B given a fixed 5° CCW rotation throughout.
//
// The incident-edge selection must map to the same feature at every step
// of the rotated sweep — i.e. the ids must be identical WITHIN this sweep
// (we do NOT require them to match (d)'s ids, since the reference face may
// differ at 5°; we only require STABILITY within the rotated series).
//
// This catches an incident-index-shift bug that the axis-aligned sweep
// cannot: if a small rotation causes the incident-edge finder to alternate
// between two adjacent edges across depth steps, the id set will differ
// between steps and this test fails.
//
// Hand derivation:
//   A spans x=[-10,+10], y=[-10,+10] (axis-aligned, no rotation).
//   B is MakeAabb(10,10) centred at x=18 (overlap ~2), rotated 5° CCW.
//   5° = 0.0873 rad.  At this small angle the left face of B is still
//   approximately perpendicular to the x-axis, so the x-axis SAT axis
//   is still the reference (or B's nearly-vertical face), and the same
//   incident vertex pair on B's near-left face is selected every step.
//   We assert only that the id set is the SAME at each step of this sweep.
// ====================================================================
TEST_CASE("physics: ManifoldV2 feature-id stable under fixed rotation sweep", "[physics]")
{
    const Shape boxA = MakeAabb(Real(10), Real(10));
    const Shape boxB = MakeAabb(Real(10), Real(10));
    const Transform xfA{ Vec2(Real(0), Real(0)), Real(0) };

    // Fixed 5-degree CCW rotation on B throughout the sweep.
    const Real kRot5deg = Real(5.0 * kPiD64 / 180.0); // 0.0873 rad

    bool firstStep = true;
    std::set<uint32_t> expectedIds;

    for (int step = 0; step < 10; ++step)
    {
        // Same depth-sweep as (d): B centre from x=18 down to x=17 (overlap 2..3).
        const Real bx = Real(18) - Real(step) * Real(0.1f);
        const Transform xfB{ Vec2(bx, Real(0)), kRot5deg };

        const Manifold m = Collide(boxA, xfA, boxB, xfB);

        // Must still detect overlap (small rotation doesn't separate the shapes).
        REQUIRE(m.pointCount >= 1);

        std::set<uint32_t> thisIds;
        for (int i = 0; i < m.pointCount; ++i)
            thisIds.insert(m.points[i].id);

        if (firstStep)
        {
            expectedIds = thisIds;
            firstStep   = false;
        }
        else
        {
            // The id SET must be identical across all steps of THIS rotated sweep.
            // (Not required to match the axis-aligned (d) sweep's ids.)
            CHECK(thisIds == expectedIds);
        }
    }
}
