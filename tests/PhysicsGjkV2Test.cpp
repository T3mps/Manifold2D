// Physics v2 Task 2: Rotation-aware GJK distance + closest features on cores.
//
// All expected values are hand-derived from first principles (NOT captured from
// the Lua oracle). Tolerances are 1e-3 for distance/geometry.
//
// The new entry is GjkDistanceCore(va, na, vb, nb) -> GjkCoreResult which adds
// featureA / featureB (simplex witness indices into the input arrays) on top of
// the existing GjkDistance distance + closest-point output. GJK itself is
// rotation-agnostic: rotation enters ONLY via the caller rotating core verts
// into world space before calling.
//
// Tests in this file (gate for Task 2):
//   (a) Two MakeAabb(5,5) cores: left at (0,0) axis-aligned, right at center
//       (20,0) rotated 45 degrees. Core separation along +x:
//       right box nearest extent = 20 - 5*sqrt(2) ~= 12.929, left edge = 5.
//       Separation = 12.929 - 5 = 7.929 (within 1e-3).
//   (b) Circle core (1 vert at (0,0)) vs MakeAabb(5,5) core at (12,0):
//       separation = 12 - 5 = 7.0 (within 1e-3), closest point on box ~= (7,0).
//   (c) Closest-feature indices: circle (1 vert) vs box at (12,0).
//       Circle featureA must be 0 (its only vertex). Box featureB must identify
//       the two verts on the left face (x=7, indices 0 and 3 in world-vert order
//       as built by RotateCoreVerts from MakeAabb(5,5) at (12,0)).
//   (d) Overlapping cores (two MakeAabb(5,5) at (0,0) and (3,0)) report
//       distance == 0 (GJK degenerate; Task 4 EPA handles penetration).

#include <array>
#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <Manifold2D/Physics/Narrowphase/Gjk.hpp>
#include <Manifold2D/Physics/Shapes.hpp>

using Catch::Approx;
using namespace Manifold2D::Physics;

namespace
{
    constexpr double kTol = 1e-3;
    // Pi at f64 for hand-derived values (distinct name to avoid ambiguity
    // with Manifold2D::Physics::kPi which is a Real/f32 in scope via using).
    constexpr double kPiD64 = 3.14159265358979323846;

    // Rotate + translate a shape's core verts into world space.
    // xf.rotation is the angle in radians. Returns a vector of world-space verts.
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
// (a) Two MakeAabb(5,5) cores: left at origin (axis-aligned), right at
//     center (20,0) rotated 45 degrees.
//
// Hand derivation:
//   Left box world verts (axis-aligned): (-5,-5), (5,-5), (5,5), (-5,5).
//   Rightmost extent of left box = +5.
//
//   Right box at center (20,0) rotated 45 deg. In MakeAabb(5,5) local
//   space the verts are: (-5,-5), (5,-5), (5,5), (-5,5).
//   Under 45-deg rotation (c=s=1/sqrt(2)):
//     v0=(-5,-5) -> (-5*c-(-5)*s, -5*s+(-5)*c) = ( 0,  -5*sqrt(2))
//     v1=( 5,-5) -> ( 5*c-(-5)*s,  5*s+(-5)*c) = ( 5*sqrt(2), 0)
//     v2=( 5, 5) -> ( 5*c-  5*s,  5*s+  5*c) = ( 0,  5*sqrt(2))
//     v3=(-5, 5) -> (-5*c-  5*s, -5*s+  5*c) = (-5*sqrt(2), 0)
//   Translated to (20,0):
//     v1 = (20 + 5*sqrt(2), 0) ... leftmost is v3 = (20 - 5*sqrt(2), 0)
//   5*sqrt(2) ~ 7.0711 -> leftmost x ~ 12.929.
//
//   Core separation = 12.929 - 5 = 7.929.
// ====================================================================
TEST_CASE("physics: GjkV2 two rotated aabb cores separation", "[physics]")
{
    const Shape boxA = MakeAabb(Real(5), Real(5));
    const Shape boxB = MakeAabb(Real(5), Real(5));

    // Left box: at origin, axis-aligned.
    const Transform xfA{ Vec2(Real(0), Real(0)), Real(0) };
    // Right box: center (20,0), rotated 45 degrees.
    const Transform xfB{ Vec2(Real(20), Real(0)), Real(kPiD64 / 4.0) };

    const std::vector<Vec2> va = RotateCoreVerts(boxA, xfA);
    const std::vector<Vec2> vb = RotateCoreVerts(boxB, xfB);

    const GjkCoreResult r = GjkDistanceCore(va.data(), static_cast<int>(va.size()),
                                             vb.data(), static_cast<int>(vb.size()));

    // Separation = 20 - 5 - 5*sqrt(2) ~ 7.929.
    const double expected = 20.0 - 5.0 - 5.0 * std::sqrt(2.0); // ~7.929
    CHECK(static_cast<double>(r.distance) == Approx(expected).margin(kTol));

    // Closest point on A is on the right face (x~=5) near y=0.
    CHECK(static_cast<double>(r.pointA.x) == Approx(5.0).margin(kTol));
    CHECK(static_cast<double>(r.pointA.y) == Approx(0.0).margin(kTol));

    // Closest point on B is on the leftmost corner of the rotated box.
    const double bExpX = 20.0 - 5.0 * std::sqrt(2.0); // ~12.929
    CHECK(static_cast<double>(r.pointB.x) == Approx(bExpX).margin(kTol));
    CHECK(static_cast<double>(r.pointB.y) == Approx(0.0).margin(kTol));
}

// ====================================================================
// (b) Circle core (1 vert at origin) vs MakeAabb(5,5) core centered (12,0).
//
// Hand derivation:
//   Circle core: 1 vert at (0,0).
//   Box world verts (axis-aligned at (12,0)): (7,-5),(17,-5),(17,5),(7,5).
//   GJK will find the closest point on the box core to (0,0).
//   The box's left face is at x=7 (verts (7,-5)-(7,5)).
//   Closest point on the left face to (0,0) is (7, 0).
//   Distance = 7.0.
// ====================================================================
TEST_CASE("physics: GjkV2 circle core vs aabb core", "[physics]")
{
    const Shape circle = MakeCircle(Real(1)); // radius=1, but core is just 1 vert
    const Shape box    = MakeAabb(Real(5), Real(5));

    // Circle at origin (radius is irrelevant for core GJK).
    const Transform xfA{ Vec2(Real(0), Real(0)), Real(0) };
    // Box centered at (12, 0), axis-aligned.
    const Transform xfB{ Vec2(Real(12), Real(0)), Real(0) };

    const std::vector<Vec2> va = RotateCoreVerts(circle, xfA);
    const std::vector<Vec2> vb = RotateCoreVerts(box,    xfB);

    const GjkCoreResult r = GjkDistanceCore(va.data(), static_cast<int>(va.size()),
                                             vb.data(), static_cast<int>(vb.size()));

    CHECK(static_cast<double>(r.distance) == Approx(7.0).margin(kTol));

    // Closest point on circle core = (0,0) (the single vert).
    CHECK(static_cast<double>(r.pointA.x) == Approx(0.0).margin(kTol));
    CHECK(static_cast<double>(r.pointA.y) == Approx(0.0).margin(kTol));

    // Closest point on box face ~= (7, 0).
    CHECK(static_cast<double>(r.pointB.x) == Approx(7.0).margin(kTol));
    CHECK(static_cast<double>(r.pointB.y) == Approx(0.0).margin(kTol));
}

// ====================================================================
// (c) Closest-feature indices: circle core vs aabb core.
//
// The circle has exactly 1 vert (index 0). Its featureA must always be 0.
// The aabb box at (12,0) has world verts built from MakeAabb(5,5) local
// verts (CCW: (-5,-5), (5,-5), (5,5), (-5,5)) translated to world:
//   idx 0: (7,  -5)  <- left-bottom
//   idx 1: (17, -5)
//   idx 2: (17,  5)
//   idx 3: (7,   5)  <- left-top
// The left face of the box is the edge from idx 0 to idx 3.
// GJK will reduce to a 2-simplex on this edge: the circle core's single
// vertex at (0,0) projects onto the left face at the interior point (7,0)
// (face spans y=-5..+5), so the closest feature on B is ALWAYS the edge,
// never a degenerate collapse to a single endpoint vertex. The vertex
// branch is unreachable for this geometry; accepting it would mask a
// regression that breaks stable manifold ids in downstream tasks (T3+).
//
// Feature encoding (GjkCoreResult):
//   vertex: feature == plain index (< kEdgeMask).
//   edge  : feature == kEdgeMask | (idxLo << 16) | idxHi
//            where idxLo < idxHi are the two simplex vertex indices
//            in ascending order (matching the Gjk.hpp encode exactly).
//
// Expected featureB = kEdgeMask | (0u << 16) | 3u = 0x80000003u.
// ====================================================================
TEST_CASE("physics: GjkV2 closest-feature indices circle vs aabb", "[physics]")
{
    const Shape circle = MakeCircle(Real(1));
    const Shape box    = MakeAabb(Real(5), Real(5));

    const Transform xfA{ Vec2(Real(0), Real(0)), Real(0) };
    const Transform xfB{ Vec2(Real(12), Real(0)), Real(0) };

    const std::vector<Vec2> va = RotateCoreVerts(circle, xfA);
    const std::vector<Vec2> vb = RotateCoreVerts(box,    xfB);

    const GjkCoreResult r = GjkDistanceCore(va.data(), static_cast<int>(va.size()),
                                             vb.data(), static_cast<int>(vb.size()));

    // The circle has only one vertex; its feature must be index 0.
    CHECK(r.featureA == 0u);

    // The circle's vertex projects onto the interior of the box's left face
    // (y=0 lies strictly between y=-5 and y=+5), so GJK MUST converge to a
    // 2-simplex -> the box's featureB must be an EDGE, not a vertex.
    // The left face connects world-vert indices 0 (7,-5) and 3 (7,+5).
    // Encoding: kEdgeMask | (idxLo << 16) | idxHi  with idxLo=0, idxHi=3.
    constexpr uint32_t kExpectedFeatureB =
        GjkCoreResult::kEdgeMask | (0u << 16) | 3u;

    CHECK((r.featureB & GjkCoreResult::kEdgeMask) != 0u); // must be an edge
    CHECK(r.featureB == kExpectedFeatureB);                // specifically {0,3}
}

// ====================================================================
// (d) Overlapping cores -> distance == 0.
//
// Two MakeAabb(5,5) cores at (0,0) and (3,0): they overlap (distance
// between centers is 3, sum of half-widths is 10). GJK degenerate ->
// origin is inside the Minkowski difference -> returns distance 0.
// Task 4 (EPA) owns penetration resolution. Here we just confirm the
// overlap-flag contract (distance == 0).
// ====================================================================
TEST_CASE("physics: GjkV2 overlapping cores report zero distance", "[physics]")
{
    const Shape box = MakeAabb(Real(5), Real(5));

    const Transform xfA{ Vec2(Real(0), Real(0)), Real(0) };
    const Transform xfB{ Vec2(Real(3), Real(0)), Real(0) };

    const std::vector<Vec2> va = RotateCoreVerts(box, xfA);
    const std::vector<Vec2> vb = RotateCoreVerts(box, xfB);

    const GjkCoreResult r = GjkDistanceCore(va.data(), static_cast<int>(va.size()),
                                             vb.data(), static_cast<int>(vb.size()));

    CHECK(static_cast<double>(r.distance) == Approx(0.0).margin(kTol));
}
