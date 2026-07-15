// Geometry::Vec2<T> contract tests (Manifold2D Phase 1). The layout/trait
// asserts and expression-order KATs here are what let the glm migration be
// value-exact; the f32w case proves the wide instantiation compiles and
// bit-matches scalar lane math.
#include <cstring>
#include <type_traits>
#include <catch2/catch_test_macros.hpp>
#include <Manifold2D/Geometry/Vec2.hpp>
#include <Mosaic/Simd/Wide.hpp>

using Manifold2D::Geometry::Vec2;
using Manifold2D::Geometry::Vec2f;
using Manifold2D::Geometry::Vec2d;

TEST_CASE("Vec2: layout and trait parity with glm::vec2", "[geometry]")
{
    STATIC_CHECK(sizeof(Vec2f) == 8);
    STATIC_CHECK(sizeof(Vec2d) == 16);
    STATIC_CHECK(std::is_standard_layout_v<Vec2f>);
    STATIC_CHECK(std::is_trivially_copyable_v<Vec2f>);
    STATIC_CHECK(std::is_trivially_destructible_v<Vec2f>);
    // x precedes y contiguously (memcpy paths over vertex arrays).
    Vec2f v(1.0f, 2.0f);
    float raw[2];
    std::memcpy(raw, &v, sizeof raw);
    CHECK(raw[0] == 1.0f);
    CHECK(raw[1] == 2.0f);
}

TEST_CASE("Vec2: construction forms used by physics and geometry", "[geometry]")
{
    const Vec2f parens(3.0f, 4.0f);       // Vec2(a, b) -- dominant physics form
    const Vec2f braces{ 3.0f, 4.0f };     // Vec2 name{ a, b } -- member/var init form
    CHECK(parens == braces);
    Vec2f arr[4];                          // default ctor: compiles, uninitialized (glm parity)
    arr[0] = Vec2f(1.0f, 1.0f);
    CHECK(arr[0].x == 1.0f);
    Vec2d fromComponents{ double(parens.x), double(parens.y) };  // ConvexHullTest's Pt<double>{o.x, o.y} idiom
    CHECK(fromComponents.x == 3.0);
}

TEST_CASE("Vec2: operator set matches the surveyed physics usage", "[geometry]")
{
    const Vec2f a(1.0f, 2.0f), b(3.0f, 5.0f);
    CHECK((a + b) == Vec2f(4.0f, 7.0f));
    CHECK((b - a) == Vec2f(2.0f, 3.0f));
    CHECK((a * 2.0f) == Vec2f(2.0f, 4.0f));       // vec * scalar
    CHECK((2.0f * a) == Vec2f(2.0f, 4.0f));       // scalar * vec (Shapes.cpp:217 order)
    CHECK((b / 2.0f) == Vec2f(1.5f, 2.5f));
    CHECK((-a) == Vec2f(-1.0f, -2.0f));
    Vec2f c = a; c += b;  CHECK(c == Vec2f(4.0f, 7.0f));
    Vec2f d = b; d -= a;  CHECK(d == Vec2f(2.0f, 3.0f));
    Vec2f e = a; e *= 3.0f; CHECK(e == Vec2f(3.0f, 6.0f));
    Vec2f f = b; f /= 2.0f; CHECK(f == Vec2f(1.5f, 2.5f));
}

TEST_CASE("Vec2: free-function expression-order KATs", "[geometry]")
{
    using namespace Manifold2D::Geometry;
    const Vec2f a(1.0f, 2.0f), b(3.0f, 5.0f);
    CHECK(Dot(a, b) == 13.0f);            // 1*3 + 2*5
    CHECK(Cross(a, b) == -1.0f);          // 1*5 - 2*3
    CHECK(LengthSq(b) == 34.0f);
    CHECK(Length(Vec2f(3.0f, 4.0f)) == 5.0f);
    CHECK(Perp(Vec2f(2.0f, 7.0f)) == Vec2f(7.0f, -2.0f));   // (v.y, -v.x), Math.hpp parity
    CHECK(Normalized(Vec2f(0.0f, 0.0f)) == Vec2f(0.0f, 0.0f)); // zero-guard contract
    CHECK(Normalized(Vec2f(0.0f, 3.0f)) == Vec2f(0.0f, 1.0f));
    CHECK(Min(a, b) == Vec2f(1.0f, 2.0f));
    CHECK(Max(a, b) == Vec2f(3.0f, 5.0f));
    // double instantiation exercises Vec2d end-to-end
    CHECK(Dot(Vec2d(1.0, 2.0), Vec2d(3.0, 5.0)) == 13.0);
}

TEST_CASE("Vec2<f32w>: wide arithmetic core compiles and bit-matches scalar lanes", "[geometry][simd]")
{
    using Mosaic::Simd::f32w;
    // Wide-lane construction/extraction uses the house Simd API: the lane
    // count is f32w::width and load/store are the free functions
    // Mosaic::Simd::load / Mosaic::Simd::store over alignas(32) arrays
    // (see Simd.hpp + SimdWideTests.inl). The assertion logic is pinned:
    // every lane of the wide result equals the scalar computation on that
    // lane's inputs.
    constexpr int W = f32w::width;
    alignas(32) float ax[W], ay[W], bx[W], by[W];
    for (int i = 0; i < W; ++i)
    {
        ax[i] = 1.0f + float(i); ay[i] = 2.0f - float(i);
        bx[i] = 0.5f * float(i); by[i] = 3.0f + float(i);
    }
    Vec2<f32w> wa(Mosaic::Simd::load(ax), Mosaic::Simd::load(ay));
    Vec2<f32w> wb(Mosaic::Simd::load(bx), Mosaic::Simd::load(by));

    alignas(32) float sv[W];
    for (int i = 0; i < W; ++i)
        sv[i] = 0.25f + float(i);
    const f32w scale = Mosaic::Simd::load(sv);

    const Vec2<f32w> sum  = wa + wb;
    const f32w       dot  = Dot(wa, wb);
    const f32w       crs  = Cross(wa, wb);
    const Vec2<f32w> perp = Perp(wa);
    const Vec2<f32w> neg  = -wa;               // wide unary minus
    const Vec2<f32w> scl  = wa * scale;        // wide vec * scalar
    Vec2<f32w> acc = wa;                       // wide compound +=
    acc += wb;

    // Review fix (Minor #1): assert BOTH lanes of every 2-component result --
    // perp.y is the negation lane ((v.y, -v.x)) and was previously unchecked.
    alignas(32) float oSumX[W], oSumY[W], oDot[W], oCrs[W],
                      oPerpX[W], oPerpY[W], oNegX[W], oNegY[W],
                      oSclX[W], oSclY[W], oAccX[W], oAccY[W];
    Mosaic::Simd::store(oSumX, sum.x);   Mosaic::Simd::store(oSumY, sum.y);
    Mosaic::Simd::store(oDot, dot);
    Mosaic::Simd::store(oCrs, crs);
    Mosaic::Simd::store(oPerpX, perp.x); Mosaic::Simd::store(oPerpY, perp.y);
    Mosaic::Simd::store(oNegX, neg.x);   Mosaic::Simd::store(oNegY, neg.y);
    Mosaic::Simd::store(oSclX, scl.x);   Mosaic::Simd::store(oSclY, scl.y);
    Mosaic::Simd::store(oAccX, acc.x);   Mosaic::Simd::store(oAccY, acc.y);
    for (int i = 0; i < W; ++i)
    {
        const Vec2f sa(ax[i], ay[i]), sb(bx[i], by[i]);
        CHECK(oSumX[i]  == (sa + sb).x);
        CHECK(oSumY[i]  == (sa + sb).y);
        CHECK(oDot[i]   == Dot(sa, sb));
        CHECK(oCrs[i]   == Cross(sa, sb));
        CHECK(oPerpX[i] == Perp(sa).x);
        CHECK(oPerpY[i] == Perp(sa).y);
        CHECK(oNegX[i]  == (-sa).x);
        CHECK(oNegY[i]  == (-sa).y);
        CHECK(oSclX[i]  == (sa * sv[i]).x);
        CHECK(oSclY[i]  == (sa * sv[i]).y);
        Vec2f sacc = sa; sacc += sb;
        CHECK(oAccX[i]  == sacc.x);
        CHECK(oAccY[i]  == sacc.y);
    }
}
