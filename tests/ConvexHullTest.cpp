#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <Manifold2D/Geometry/ConvexHull.hpp>
#include <Manifold2D/Geometry/detail/Predicates.hpp>

using Manifold2D::Geometry::ConvexHull;
using Manifold2D::Geometry::MonotoneChain;
using Pt = Manifold2D::Geometry::Pt<float>;

namespace
{
    std::vector<Pt> Hull(const std::vector<Pt>& in)
    {
        return ConvexHull<MonotoneChain, float>(std::span<const Pt>(in));
    }
    bool Eq(const std::vector<Pt>& a, const std::vector<Pt>& b)
    {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (a[i].x != b[i].x || a[i].y != b[i].y) return false;
        return true;
    }
}

TEST_CASE("ConvexHull canonical contract", "[geometry]")
{
    SECTION("unit square with interior point -> CCW from lex-min")
    {
        std::vector<Pt> in = {{1,1},{0,0},{1,0},{0,1},{0.5f,0.5f}};
        std::vector<Pt> want = {{0,0},{1,0},{1,1},{0,1}};
        REQUIRE(Eq(Hull(in), want));
    }
    SECTION("triangle drops interior points")
    {
        std::vector<Pt> in = {{0,0},{4,0},{0,4},{1,1},{2,1}};
        std::vector<Pt> want = {{0,0},{4,0},{0,4}};
        REQUIRE(Eq(Hull(in), want));
    }
    SECTION("collinear -> two extreme endpoints")
    {
        std::vector<Pt> in = {{0,0},{1,1},{2,2},{3,3}};
        std::vector<Pt> want = {{0,0},{3,3}};
        REQUIRE(Eq(Hull(in), want));
    }
    SECTION("duplicates collapse")
    {
        std::vector<Pt> in = {{0,0},{0,0},{2,0},{2,0},{0,2}};
        std::vector<Pt> want = {{0,0},{2,0},{0,2}};
        REQUIRE(Eq(Hull(in), want));
    }
    SECTION("single point")
    {
        std::vector<Pt> in = {{5,7}};
        REQUIRE(Eq(Hull(in), in));
    }
    SECTION("two points lex-ordered")
    {
        std::vector<Pt> in = {{3,3},{1,1}};
        std::vector<Pt> want = {{1,1},{3,3}};
        REQUIRE(Eq(Hull(in), want));
    }
    SECTION("collinear edge points on a square edge are stripped")
    {
        std::vector<Pt> in = {{0,0},{1,0},{2,0},{2,2},{0,2}};
        std::vector<Pt> want = {{0,0},{2,0},{2,2},{0,2}};
        REQUIRE(Eq(Hull(in), want));
    }
}

namespace
{
    // Fixed point clouds reused by every cross-validation case.
    std::vector<std::vector<Pt>> Clouds()
    {
        return {
            {{0,0},{4,0},{4,4},{0,4},{2,2},{1,3},{3,1}},           // square + interior
            {{0,0},{5,1},{3,5},{-2,4},{-4,-1},{-1,-3},{2,-2},{0,0}},// heptagon-ish + dup
            {{0,0},{10,0},{10,10},{0,10},{5,-3},{13,5},{5,13},{-3,5}}, // star points
            {{1,1},{2,2},{3,3},{4,1},{2,0}},                       // some collinear
        };
    }
    template <class P>
    void AgreesWithMonotone(const char* /*name*/)
    {
        for (const auto& c : Clouds())
        {
            std::span<const Pt> s(c);
            REQUIRE(Eq(ConvexHull<P, float>(s), ConvexHull<MonotoneChain, float>(s)));
        }
    }
}

TEST_CASE("GrahamScan agrees with MonotoneChain", "[geometry]")
{
    AgreesWithMonotone<Manifold2D::Geometry::GrahamScan>("graham");
}

TEST_CASE("JarvisMarch agrees with MonotoneChain", "[geometry]")
{
    AgreesWithMonotone<Manifold2D::Geometry::JarvisMarch>("jarvis");
}

TEST_CASE("QuickHull agrees with MonotoneChain", "[geometry]")
{
    AgreesWithMonotone<Manifold2D::Geometry::QuickHull>("quickhull");
}

TEST_CASE("Chan agrees with MonotoneChain", "[geometry]")
{
    AgreesWithMonotone<Manifold2D::Geometry::Chan>("chan");
}

TEST_CASE("KirkpatrickSeidel agrees with MonotoneChain", "[geometry]")
{
    AgreesWithMonotone<Manifold2D::Geometry::KirkpatrickSeidel>("kps");
}

#include <random>   // (catch2 + Predicates already included at the top of this file)

namespace
{
    template <class T>
    std::vector<Manifold2D::Geometry::Pt<T>> RandomCloud(std::mt19937& rng, int n)
    {
        std::uniform_int_distribution<int> d(-500, 500);   // integer-valued: exact in float
        std::vector<Manifold2D::Geometry::Pt<T>> v;
        v.reserve(n);
        for (int i = 0; i < n; ++i)
            v.push_back(Manifold2D::Geometry::Pt<T>(T(d(rng)), T(d(rng))));
        return v;
    }

    template <class T>
    void AllSixAgree(const std::vector<Manifold2D::Geometry::Pt<T>>& cloud)
    {
        using namespace Manifold2D::Geometry;
        std::span<const Manifold2D::Geometry::Pt<T>> s(cloud);   // fully-qualified: the
        // file-scope `using Pt = Pt<float>` alias otherwise makes `Pt<T>` ambiguous.
        const auto ref = ConvexHull<MonotoneChain, T>(s);
        REQUIRE(ConvexHull<GrahamScan, T>(s)        == ref);
        REQUIRE(ConvexHull<JarvisMarch, T>(s)       == ref);
        REQUIRE(ConvexHull<QuickHull, T>(s)         == ref);
        REQUIRE(ConvexHull<Chan, T>(s)              == ref);
        REQUIRE(ConvexHull<KirkpatrickSeidel, T>(s) == ref);
    }

    // All six agree AND the reference hull obeys the standalone canonical contract,
    // with the no-three-collinear check made through the EXACT Orient2d -- so even if
    // all six agreed on a non-canonical hull (an interior/collinear vertex the plain
    // float Cross failed to strip), this still trips. This is the E01-5 payoff check
    // AllSixAgree alone does not perform.
    template <class T>
    void AllSixAgreeExact(const std::vector<Manifold2D::Geometry::Pt<T>>& cloud)
    {
        using namespace Manifold2D::Geometry;
        AllSixAgree<T>(cloud);
        std::span<const Manifold2D::Geometry::Pt<T>> s(cloud);
        const auto h = ConvexHull<MonotoneChain, T>(s);
        if (h.size() < 3) return;   // all-collinear degenerate draw
        REQUIRE(detail::SignedArea2<T>(h) > T(0));
        for (std::size_t i = 1; i < h.size(); ++i)
            REQUIRE_FALSE(detail::Less<T>(h[i], h[0]));
        for (std::size_t i = 0; i < h.size(); ++i)
            REQUIRE(detail::Orient2d<T>(h[(i + h.size() - 1) % h.size()],
                                        h[i], h[(i + 1) % h.size()]) != 0);
    }
}

TEST_CASE("All six convex-hull algorithms agree on random clouds", "[geometry]")
{
    std::mt19937 rng(0xC0FFEEu);
    for (int trial = 0; trial < 300; ++trial)
    {
        const int n = 3 + (trial % 60);
        AllSixAgree<float>(RandomCloud<float>(rng, n));
        AllSixAgree<double>(RandomCloud<double>(rng, n));
    }
}

TEST_CASE("ConvexHull output obeys the canonical contract", "[geometry]")
{
    using namespace Manifold2D::Geometry;
    std::mt19937 rng(0x1234u);
    for (int trial = 0; trial < 200; ++trial)
    {
        const auto cloud = RandomCloud<float>(rng, 3 + (trial % 50));
        std::span<const Manifold2D::Geometry::Pt<float>> s(cloud);   // fully-qualified (see above)
        const auto h = ConvexHull<MonotoneChain, float>(s);
        if (h.size() < 3) continue;   // degenerate (all-collinear draw)

        // CCW (positive signed area).
        REQUIRE(detail::SignedArea2<float>(h) > 0.0f);
        // Starts at the lexicographically smallest hull vertex.
        for (std::size_t i = 1; i < h.size(); ++i)
            REQUIRE_FALSE(detail::Less<float>(h[i], h[0]));
        // No three consecutive collinear.
        for (std::size_t i = 0; i < h.size(); ++i)
            REQUIRE(detail::Cross<float>(h[(i + h.size() - 1) % h.size()],
                                         h[i], h[(i + 1) % h.size()]) != 0.0f);
        // Every input point is inside-or-on the hull (left of / on every CCW edge).
        for (const auto& p : cloud)
            for (std::size_t i = 0; i < h.size(); ++i)
                REQUIRE(detail::Cross<float>(h[i], h[(i + 1) % h.size()], p) >= 0.0f);
    }
}

// E01-5: the robust orientation predicate must return the EXACT sign.
TEST_CASE("Orient2d is exact for float and double", "[geometry][robust]")
{
    using Manifold2D::Geometry::Pt;
    using Manifold2D::Geometry::detail::Orient2d;

    SECTION("basic float orientation")
    {
        REQUIRE(Orient2d<float>({0,0}, {1,0}, {0,1}) ==  1);   // CCW
        REQUIRE(Orient2d<float>({0,0}, {1,0}, {0,-1}) == -1);  // CW
        REQUIRE(Orient2d<float>({0,0}, {1,0}, {2,0}) ==  0);   // collinear
    }

    SECTION("float exact sign at a 1-ULP-off-collinear point")
    {
        // Three points exactly on a diagonal, then b nudged one ULP off it. This
        // ASSERTION verifies the exact predicate reports collinear (0) then strictly
        // -left (+1) at a 1-ULP-off-collinear point. It does NOT claim naive float
        // mis-signs here -- the plain-float cross keeps the correct sign for this
        // particular +1-ULP point (see the int64-oracle section for inputs that
        // genuinely make the naive computation round).
        const Pt<float> o{0.5f, 0.5f};
        const Pt<float> a{12.0f, 12.0f};
        const Pt<float> b{24.0f, 24.0f};
        REQUIRE(Orient2d<float>(o, a, b) == 0);                // exactly collinear
        Pt<float> b2 = b; b2.y = std::nextafter(b2.y, 100.0f); // one ULP above the line
        REQUIRE(Orient2d<float>(o, a, b2) == 1);               // now strictly left
    }

    SECTION("float path == exact double path on the same points")
    {
        // Double-promotion of float inputs is exact, so it MUST agree with the
        // exact-double predicate applied to the identical (exactly promoted) points.
        // (The float path now promotes and calls the same exact-double kernel, so
        // this is tautological -- it still verifies the lossless promotion + kernel.)
        std::mt19937 rng(0xE0155u);
        std::uniform_real_distribution<float> d(-3.0e5f, 3.0e5f);
        for (int i = 0; i < 20000; ++i)
        {
            const Pt<float> o{d(rng), d(rng)}, a{d(rng), d(rng)}, b{d(rng), d(rng)};
            const Pt<double> od{o.x, o.y}, ad{a.x, a.y}, bd{b.x, b.y};
            REQUIRE(Orient2d<float>(o, a, b) == Orient2d<double>(od, ad, bd));
        }
    }

    SECTION("double exact at 1-ULP-off-collinear points")
    {
        // Points exactly on a diagonal, then b nudged 1 ULP each way: the exact
        // predicate must report 0 then +1 then -1 across the collinear boundary.
        // (These inputs are small enough that naive double is also exact here -- the
        // int64-oracle section below carries the load-bearing "naive double rounds".)
        const double s = 1.0;
        const Pt<double> o{0.5, 0.5};
        const Pt<double> a{s + 0.5, s + 0.5};
        Pt<double> b{2*s + 0.5, 2*s + 0.5};
        REQUIRE(Orient2d<double>(o, a, b) == 0);               // exactly collinear
        b.y = std::nextafter(b.y, 1e9);                        // 1 ULP left of the line
        REQUIRE(Orient2d<double>(o, a, b) == 1);
        b.y = std::nextafter(b.y = 2*s + 0.5, -1e9);           // 1 ULP right
        REQUIRE(Orient2d<double>(o, a, b) == -1);
    }

    SECTION("double exact vs int64 oracle where naive double rounds")
    {
        // To make the double error-free-transform genuinely LOAD-BEARING the inputs
        // must be near-collinear at a magnitude where naive double rounds. Fully
        // random large coords never are (their determinant is astronomically far
        // from zero, so naive double keeps the sign -- a vacuous test). Instead grow
        // a UNIMODULAR difference pair (u x v == 1, an exactly-known tiny
        // determinant) by random shears that preserve the cross product, until both
        // vectors are long and nearly parallel. Then for a = o + u, b = o + v the
        // determinant is exactly +/-1, yet the naive products (ax-ox)(by-oy) and
        // (ay-oy)(bx-ox) each reach ~2^54..2^58 (> 2^53) and ROUND -- so naive double
        // loses the sign while the int64 oracle and the exact EFT predicate keep it.
        // All coords stay <= ~2^30 (double-exact, so Orient2d<double> sees the same
        // points as the oracle) and every int64 product stays <= ~2^58 (< 2^63).
        const auto aabs = [](long long v) noexcept { return v < 0 ? -v : v; };
        const auto mag  = [&](long long x, long long y) noexcept
                          { return aabs(x) > aabs(y) ? aabs(x) : aabs(y); };
        const long long kGrow = 1LL << 27;   // both vectors grown past this
        const long long kCeil = 1LL << 30;   // hard component ceiling (int64-safe products)

        std::mt19937 rng(0xE0157u);
        std::uniform_int_distribution<long long> kd(1, 5);   // random shear multiplier
        std::uniform_int_distribution<long long> od(-(1LL << 20), 1LL << 20); // origin
        std::uniform_int_distribution<int> flip(0, 1);       // vary det sign +/-1

        int naiveDisagreements = 0;
        for (int i = 0; i < 20000; ++i)
        {
            // Shear the shorter vector by the longer (adding an integer multiple of
            // one to the other preserves u x v == 1) until both are long. Cap each
            // grown vector at kCeil so the int64 products cannot overflow.
            long long ux = 1, uy = 0, vx = 0, vy = 1;
            for (int step = 0; step < 300
                 && (mag(ux, uy) < kGrow || mag(vx, vy) < kGrow); ++step)
            {
                long long k = kd(rng);
                if (mag(ux, uy) <= mag(vx, vy))
                {
                    if (mag(ux + k * vx, uy + k * vy) > kCeil) k = 1;
                    ux += k * vx; uy += k * vy;
                }
                else
                {
                    if (mag(vx + k * ux, vy + k * uy) > kCeil) k = 1;
                    vx += k * ux; vy += k * uy;
                }
            }
            if (flip(rng)) { long long t; t = ux; ux = vx; vx = t; t = uy; uy = vy; vy = t; }

            const long long ox = od(rng), oy = od(rng);
            const long long ax = ox + ux, ay = oy + uy, bx = ox + vx, by = oy + vy;

            const long long det = (ax - ox) * (by - oy) - (ay - oy) * (bx - ox); // exactly +/-1
            const int oracle = (det > 0) - (det < 0);
            const Pt<double> o{double(ox), double(oy)}, a{double(ax), double(ay)},
                             b{double(bx), double(by)};
            REQUIRE(Orient2d<double>(o, a, b) == oracle);

            const double naive = (double(ax) - double(ox)) * (double(by) - double(oy))
                               - (double(ay) - double(oy)) * (double(bx) - double(ox));
            if (((naive > 0) - (naive < 0)) != oracle) ++naiveDisagreements;
        }
        // Prove the test actually stresses the exact path: naive double DOES mis-sign
        // a large fraction of these (otherwise this section would be vacuous).
        REQUIRE(naiveDisagreements > 0);
    }

    SECTION("antisymmetry holds for large-coordinate float")
    {
        std::mt19937 rng(0xE0156u);
        std::uniform_real_distribution<float> d(-8.0e5f, 8.0e5f); // large-coord stress; Orient2d is exact for any input
        for (int i = 0; i < 20000; ++i)
        {
            const Pt<float> o{d(rng), d(rng)}, a{d(rng), d(rng)}, b{d(rng), d(rng)};
            REQUIRE(Orient2d<float>(o, a, b) == -Orient2d<float>(o, b, a));
        }
    }
}

// E01-5 acceptance: all 6 policies agree on the EXACT canonical hull for the
// degenerate / near-collinear / large-magnitude regimes the old float Cross
// mis-oriented. Purely additive -- these inputs were never fed before.
TEST_CASE("All six hull policies agree on degenerate + near-collinear clouds",
          "[geometry][robust]")
{
    // Near-collinear float clouds: points on y = m*x + c with sub-ULP jitter,
    // plus a few genuine off-line corners. Scales stay <= ~1e5 so KirkpatrickSeidel's
    // double accumulator is also exact (it, not the exact Orient2d, is the binding
    // constraint on how large these coords may go before the six policies diverge).
    auto nearCollinear = [](std::uint32_t seed, float scale) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> t(-scale, scale);
        std::uniform_int_distribution<int>    jig(-2, 2);
        std::vector<Manifold2D::Geometry::Pt<float>> v;
        const float m = 0.37f, c = 1.5f;
        for (int i = 0; i < 200; ++i)
        {
            const float x = t(rng);
            float y = m * x + c;
            for (int k = 0; k < jig(rng); ++k) y = std::nextafter(y, 1e30f); // sub-ULP off
            v.push_back({x, y});
        }
        v.push_back({-scale, -scale}); v.push_back({scale, scale});           // real corners
        v.push_back({0.0f, scale});    v.push_back({0.0f, -scale});
        return v;
    };
    for (std::uint32_t s = 0; s < 50; ++s)
    {
        AllSixAgreeExact<float>(nearCollinear(0xC0110u + s, 1.0f));       // unit scale
        AllSixAgreeExact<float>(nearCollinear(0xC0220u + s, 1.0e5f));     // large, in-bound
    }

    // Dense collinear runs + duplicates: hull is a small polygon whose edges carry
    // many interior collinear points; StripCollinear must remove all of them and
    // all six must produce the identical canonical corner set.
    auto collinearRuns = [](std::uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> pick(0, 3);
        std::vector<Manifold2D::Geometry::Pt<float>> corners =
            {{0,0},{100,0},{100,100},{0,100}};
        std::vector<Manifold2D::Geometry::Pt<float>> v = corners;
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        for (int i = 0; i < 300; ++i)                    // interior edge points + dups
        {
            const auto& p = corners[pick(rng)];
            const auto& q = corners[(pick(rng)) % 4];
            const float f = u(rng);
            v.push_back({p.x + f * (q.x - p.x), p.y + f * (q.y - p.y)});
        }
        return v;
    };
    for (std::uint32_t s = 0; s < 50; ++s)
        AllSixAgreeExact<float>(collinearRuns(0xC0330u + s));

    // Double instantiation: exact-representable (integer) moderate clouds so the
    // exact-EFT policies and KirkpatrickSeidel's plain-double accumulator both
    // compute the SAME sign (validates the double policy logic without exercising
    // KS's non-EFT limitation, which production never hits -- production is float).
    auto intCloud = [](std::uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> d(-500, 500);
        std::vector<Manifold2D::Geometry::Pt<double>> v;
        for (int i = 0; i < 300; ++i) v.push_back({double(d(rng)), double(d(rng))});
        return v;
    };
    for (std::uint32_t s = 0; s < 50; ++s)
        AllSixAgreeExact<double>(intCloud(0xC0440u + s));
}

// E01-5 Part A: Canonicalize must derive winding EXACTLY. The old shoelace sum
// (SignedArea2 < 0) can flip its sign under large-coordinate cancellation; the new
// path takes the exact Orient2d turn at the lex-min corner (a guaranteed convex
// vertex). Feed a thin ~1e6-scale quad in BOTH windings: the canonical output must
// be CCW either way. (The area of this quad is large, so this may already pass with
// the old code -- its job is to LOCK the exact-winding guarantee going forward.)
TEST_CASE("Canonical winding is exact for large-coordinate hulls", "[geometry][robust]")
{
    using Manifold2D::Geometry::Pt;
    using Manifold2D::Geometry::ConvexHull;
    using Manifold2D::Geometry::MonotoneChain;
    using Manifold2D::Geometry::detail::Orient2d;

    // A thin, large-coordinate quad. Its true signed area is far from zero, but a
    // float shoelace of ~1e6-scale products loses low bits; the winding decision
    // must still be exact. Feed both windings; canonical output must be CCW either way.
    std::vector<Pt<float>> cw   = {{1.0e6f, 1.0e6f}, {1.0e6f, -1.0e6f},
                                   {-1.0e6f, -1.0e6f}, {-1.0e6f, 1.0e6f}};
    std::vector<Pt<float>> ccw  = {{-1.0e6f, 1.0e6f}, {-1.0e6f, -1.0e6f},
                                   {1.0e6f, -1.0e6f}, {1.0e6f, 1.0e6f}};
    for (auto* in : {&cw, &ccw})
    {
        auto h = ConvexHull<MonotoneChain, float>(*in);
        REQUIRE(h.size() == 4);
        // CCW: every corner turns left.
        for (std::size_t i = 0; i < h.size(); ++i)
            REQUIRE(Orient2d<float>(h[(i + h.size() - 1) % h.size()], h[i],
                                    h[(i + 1) % h.size()]) > 0);
    }
}

// E01-5 Part B: QuickHull's farthest-point gate is the EXACT Orient2d, but its
// RANK is an inexact double `mag`. For a strictly-left vertex whose double mag
// rounds to <= 0 (large exponent spread across the base edge), the un-hardened
// update `mag > best` (best initialised 0.0) never fires, so `c` stays unset and
// the gated vertex is DROPPED -- QuickHull returns a smaller hull than the exact
// gate accepted. The Part B fix makes the FIRST gated candidate always set `c`.
//
// This wedge triangle is a hand-verified regression (found by exhaustive search
// over float triples): apex P is strictly LEFT of the A->B base per exact Orient2d,
// yet the naive double mag (Bx-Ax)(Py-Ay)-(By-Ay)(Px-Ax) rounds to exactly 0.0 --
// A is ~1.6e7 while B,P are ~0.1, so the two products reach ~2^53 and the
// near-collinear difference rounds away. Coords are hex-float literals (exact bit
// pattern pinned). Un-hardened QuickHull drops P (2-point hull); MonotoneChain
// (whose stack test is also exact Orient2d) keeps all three -> they disagree until
// the farthest-point gate is hardened.
TEST_CASE("QuickHull keeps every exact-gated vertex at large-coordinate spread",
          "[geometry][robust]")
{
    using Manifold2D::Geometry::Pt;
    using Manifold2D::Geometry::QuickHull;
    using Manifold2D::Geometry::detail::Orient2d;

    const Pt<float> A{-0x1.f3ef380000000p+23f, -0x1.051daa0000000p+23f}; // ~(-1.638e7,-8.556e6)
    const Pt<float> B{-0x1.d0428e0000000p-5f,   0x1.62ce7e0000000p-3f};  // ~(-0.0567, 0.1732)
    const Pt<float> P{-0x1.eb26600000000p-2f,  -0x1.86831c0000000p-5f};  // ~(-0.4796,-0.04767)

    // Self-check that this cloud actually exercises the gate: exact says strictly
    // LEFT (P is a genuine hull apex), but the inexact double rank rounds to <= 0,
    // so the un-hardened `mag > best` would fail to certify P and drop it.
    REQUIRE(Orient2d<float>(A, B, P) > 0);
    const double mag = (double(B.x) - double(A.x)) * (double(P.y) - double(A.y))
                     - (double(B.y) - double(A.y)) * (double(P.x) - double(A.x));
    REQUIRE(mag <= 0.0);

    std::vector<Pt<float>> cloud = {A, B, P};
    std::span<const Pt<float>> s(cloud);
    // Both policies sign every orientation through the exact Orient2d, so on the
    // identical vertex set they must produce byte-identical canonical hulls.
    REQUIRE(ConvexHull<QuickHull, float>(s) == ConvexHull<MonotoneChain, float>(s));
    // ...and the hull is the full triangle (P must not be dropped).
    REQUIRE(ConvexHull<MonotoneChain, float>(s).size() == 3);
}
