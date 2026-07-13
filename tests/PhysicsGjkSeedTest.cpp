// Physics v2 EPA seam (Task 1): GJK support + origin-enclosing simplex.
//
// EPA and MPR (later tasks) both need two GJK primitives that already exist
// INSIDE Gjk.cpp but were not exposed: (a) the Minkowski-difference support over
// two world cores, and (b) the terminal origin-enclosing simplex GJK builds when
// the cores overlap (EPA's seed). This file is the analytic gate for the two new
// ADDITIVE public entries SupportMinkowski / GjkOriginSimplex.
//
// All expected values are hand-derived from first principles (NOT captured from
// any oracle). Tolerances are 1e-4 for the MD support coordinate.
//
// Cases:
//   (a) SupportMinkowski of two MakeAabb(5,5) cores at centers (0,0) and (3,0)
//       along dir=(1,0): support on A along +x has x=+5; support on B along -x
//       has x = 3-5 = -2; md.x = 5 - (-2) = 7. (within 1e-4)
//   (b) GjkOriginSimplex for those overlapping boxes: enclosesOrigin == true,
//       count == 3, and the triangle v[0].md, v[1].md, v[2].md contains the
//       origin (verified by an inline sign-of-cross-product winding test).
//   (c) GjkOriginSimplex for two MakeAabb(5,5) at (0,0) and (20,0) (separated):
//       enclosesOrigin == false.

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
    constexpr double kTol = 1e-4;

    // Rotate + translate a shape's core verts into world space (copy of the
    // helper in PhysicsGjkV2Test.cpp -- Collide.cpp's RotateInto is private).
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

    // Inline winding test: does triangle (a,b,c) contain the origin? Uses the
    // sign of the cross product of each edge against the vector to the origin;
    // the origin is inside iff all three signs agree (no mixed pos/neg), which
    // is winding-order-agnostic (matches the GJK n==3 inside test).
    bool TriangleContainsOrigin(Vec2 a, Vec2 b, Vec2 c)
    {
        const double c1 = (double(b.x) - a.x) * (-double(a.y)) -
                          (double(b.y) - a.y) * (-double(a.x));
        const double c2 = (double(c.x) - b.x) * (-double(b.y)) -
                          (double(c.y) - b.y) * (-double(b.x));
        const double c3 = (double(a.x) - c.x) * (-double(c.y)) -
                          (double(a.y) - c.y) * (-double(c.x));
        const bool hasNeg = (c1 < 0.0) || (c2 < 0.0) || (c3 < 0.0);
        const bool hasPos = (c1 > 0.0) || (c2 > 0.0) || (c3 > 0.0);
        return !(hasNeg && hasPos);
    }
} // namespace

// ====================================================================
// (a) SupportMinkowski over two MakeAabb(5,5) cores at (0,0) and (3,0)
//     along dir=(1,0): md.x = (+5) - (3-5) = 7.
// ====================================================================
TEST_CASE("physics: GjkSeed SupportMinkowski md along +x", "[physics]")
{
    const Shape boxA = MakeAabb(Real(5), Real(5));
    const Shape boxB = MakeAabb(Real(5), Real(5));

    const std::vector<Vec2> wa =
        RotateCoreVerts(boxA, Transform{ Vec2(Real(0), Real(0)), Real(0) });
    const std::vector<Vec2> wb =
        RotateCoreVerts(boxB, Transform{ Vec2(Real(3), Real(0)), Real(0) });

    const MdSupport s = SupportMinkowski(wa.data(), int(wa.size()),
                                         wb.data(), int(wb.size()),
                                         Vec2(Real(1), Real(0)));

    // Support on A along +x: x = +5. Support on B along -x: x = 3 - 5 = -2.
    // md.x = 5 - (-2) = 7.
    REQUIRE(double(s.md.x) == Approx(7.0).margin(kTol));
    REQUIRE(double(s.pa.x) == Approx(5.0).margin(kTol));
    REQUIRE(double(s.pb.x) == Approx(-2.0).margin(kTol));
}

// ====================================================================
// (b) GjkOriginSimplex for two overlapping MakeAabb(5,5) at (0,0) and
//     (3,0): enclosesOrigin == true, count == 3, triangle contains origin.
// ====================================================================
TEST_CASE("physics: GjkSeed origin simplex on overlap", "[physics]")
{
    const Shape boxA = MakeAabb(Real(5), Real(5));
    const Shape boxB = MakeAabb(Real(5), Real(5));

    const std::vector<Vec2> wa =
        RotateCoreVerts(boxA, Transform{ Vec2(Real(0), Real(0)), Real(0) });
    const std::vector<Vec2> wb =
        RotateCoreVerts(boxB, Transform{ Vec2(Real(3), Real(0)), Real(0) });

    const GjkSimplex s = GjkOriginSimplex(wa.data(), int(wa.size()),
                                          wb.data(), int(wb.size()));

    REQUIRE(s.enclosesOrigin == true);
    REQUIRE(s.count == 3);
    REQUIRE(TriangleContainsOrigin(s.v[0].md, s.v[1].md, s.v[2].md));
}

// ====================================================================
// (c) GjkOriginSimplex for two separated MakeAabb(5,5) at (0,0) and
//     (20,0): enclosesOrigin == false.
// ====================================================================
TEST_CASE("physics: GjkSeed origin simplex on separation", "[physics]")
{
    const Shape boxA = MakeAabb(Real(5), Real(5));
    const Shape boxB = MakeAabb(Real(5), Real(5));

    const std::vector<Vec2> wa =
        RotateCoreVerts(boxA, Transform{ Vec2(Real(0), Real(0)), Real(0) });
    const std::vector<Vec2> wb =
        RotateCoreVerts(boxB, Transform{ Vec2(Real(20), Real(0)), Real(0) });

    const GjkSimplex s = GjkOriginSimplex(wa.data(), int(wa.size()),
                                          wb.data(), int(wb.size()));

    REQUIRE(s.enclosesOrigin == false);
}
