// Gravity well + per-body gravity scale (PhysicsWorld::SetGravityWell,
// BodyDef::gravityScale / SetGravityScale).
//
// The well is evaluated per body INSIDE the per-sub-step velocity integration,
// at the body's current in-step position (start-of-step pose + the TGS delta),
// exactly where world gravity is applied. That is semi-implicit Euler at the
// sub-step, which keeps orbits bounded; a once-per-step force held across the
// sub-steps would not (it spirals outward). The headline guards are: orbits stay
// bounded, a body rests on a curved ground under radial gravity AND still
// sleeps, and with no well and scale 1 nothing changes.
// PRESENTATION-FREE + C++23-clean.
#include <cmath>
#include <cstdint>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
using namespace Manifold2D::Physics;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace
{
    constexpr Real kStep = Real(1) / Real(60);

    WorldDef ZeroG()
    {
        WorldDef wd;
        wd.gravityX = Real(0);
        wd.gravityY = Real(0);
        return wd;
    }

    BodyHandle Ball(PhysicsWorld& w, Real x, Real y, Real r = Real(0.1))
    {
        BodyDef d;
        d.type     = BodyType::Dynamic;
        d.position = Vec2(x, y);
        d.shape    = MakeCircle(r);
        d.density  = Real(1);
        return w.AddBody(d);
    }

    GravityWell Well(Real cx, Real cy, Real radius, Real g0)
    {
        GravityWell gw;
        gw.enabled        = true;
        gw.center         = Vec2(cx, cy);
        gw.surfaceRadius  = radius;
        gw.surfaceGravity = g0;
        gw.falloff        = GravityFalloff::InverseSquare;
        return gw;
    }
} // namespace

TEST_CASE("GravityWell: no well and the default gravity scale leave world gravity untouched", "[physics][gravitywell]")
{
    // The baseline the existing suite relies on: a body falls exactly as before.
    PhysicsWorld w(WorldDef{});
    const BodyHandle b = Ball(w, Real(0), Real(0));
    REQUIRE(w.GravityScale(b) == Real(1));
    REQUIRE_FALSE(w.GetGravityWell().enabled);
    w.Step(kStep);
    CHECK_THAT(static_cast<double>(w.Velocity(b).y), WithinAbs(10.0 / 60.0, 1e-6));
}

TEST_CASE("GravityWell: gravityScale scales world gravity per body (0 floats, 2 falls twice as fast)", "[physics][gravitywell]")
{
    PhysicsWorld w(WorldDef{});
    BodyDef d;
    d.type = BodyType::Dynamic; d.shape = MakeCircle(Real(0.1)); d.density = Real(1);
    d.position = Vec2(Real(0), Real(0));     d.gravityScale = Real(0);
    const BodyHandle floater = w.AddBody(d);
    d.position = Vec2(Real(1), Real(0));     d.gravityScale = Real(2);
    const BodyHandle heavy = w.AddBody(d);
    for (int i = 0; i < 60; ++i) { w.Step(kStep); }
    CHECK(w.Velocity(floater).y == Real(0));
    CHECK_THAT(static_cast<double>(w.Velocity(heavy).y), WithinRel(20.0, 1e-4));
}

TEST_CASE("GravityWell: SetGravityScale changes a live body and wakes it", "[physics][gravitywell]")
{
    PhysicsWorld w(WorldDef{});
    const BodyHandle b = Ball(w, Real(0), Real(0));
    w.SetGravityScale(b, Real(0));
    w.Step(kStep);
    CHECK(w.Velocity(b).y == Real(0));
    w.SetGravityScale(b, Real(-1)); // negative: pulled the other way
    w.Step(kStep);
    CHECK_THAT(static_cast<double>(w.Velocity(b).y), WithinAbs(-10.0 / 60.0, 1e-6));
    CHECK(w.IsAwake(b));
    // non-finite input is refused (the scale stays as it was)
    w.SetGravityScale(b, static_cast<Real>(std::nan("")));
    CHECK(w.GravityScale(b) == Real(-1));
}

TEST_CASE("GravityWell: inverse-square field points at the centre and falls off with r^2", "[physics][gravitywell]")
{
    PhysicsWorld w(ZeroG());
    w.SetGravityWell(Well(Real(0), Real(0), Real(2), Real(8)));
    const BodyHandle atSurface = Ball(w, Real(2), Real(0));
    const BodyHandle twice     = Ball(w, Real(0), Real(-4));
    const BodyHandle inside    = Ball(w, Real(-1), Real(0));
    w.Step(kStep);
    // one step from rest: v = g(r) * dt toward the centre (r barely moves in a step)
    CHECK_THAT(static_cast<double>(w.Velocity(atSurface).x), WithinRel(-8.0 / 60.0, 1e-3));
    CHECK(std::abs(static_cast<double>(w.Velocity(atSurface).y)) < 1e-9);
    CHECK_THAT(static_cast<double>(w.Velocity(twice).y), WithinRel(2.0 / 60.0, 1e-3));      // (2/4)^2 * 8, pointing +y
    CHECK_THAT(static_cast<double>(w.Velocity(inside).x), WithinRel(4.0 / 60.0, 1e-3));     // interior: linear in r
}

TEST_CASE("GravityWell: the fade profile holds surface gravity to fadeStart and reaches zero at fadeEnd", "[physics][gravitywell]")
{
    PhysicsWorld w(ZeroG());
    GravityWell gw = Well(Real(0), Real(0), Real(10), Real(10));
    gw.falloff   = GravityFalloff::Fade;
    gw.fadeStart = Real(12);
    gw.fadeEnd   = Real(16);
    w.SetGravityWell(gw);
    const BodyHandle low  = Ball(w, Real(0), Real(-11));
    const BodyHandle mid  = Ball(w, Real(0), Real(-14));
    const BodyHandle high = Ball(w, Real(0), Real(-17));
    w.Step(kStep);
    CHECK_THAT(static_cast<double>(w.Velocity(low).y), WithinRel(10.0 / 60.0, 1e-3));
    CHECK_THAT(static_cast<double>(w.Velocity(mid).y), WithinRel(5.0 / 60.0, 2e-3)); // smoothstep(0.5) = 0.5
    CHECK(w.Velocity(high).y == Real(0));
}

TEST_CASE("GravityWell: an invalid well is refused and the previous one kept", "[physics][gravitywell]")
{
    PhysicsWorld w(ZeroG());
    const GravityWell good = Well(Real(0), Real(0), Real(2), Real(8));
    w.SetGravityWell(good);
    GravityWell bad = good;
    bad.surfaceRadius = Real(0);
    w.SetGravityWell(bad);
    CHECK(w.GetGravityWell().surfaceRadius == Real(2));
    bad = good;
    bad.falloff = GravityFalloff::Fade; bad.fadeStart = Real(5); bad.fadeEnd = Real(4);
    w.SetGravityWell(bad);
    CHECK(w.GetGravityWell().falloff == GravityFalloff::InverseSquare);
    bad = good;
    bad.center = Vec2(static_cast<Real>(std::nan("")), Real(0));
    w.SetGravityWell(bad);
    CHECK(w.GetGravityWell().center.x == Real(0));
    GravityWell off = good;
    off.enabled = false;
    w.SetGravityWell(off); // disabling is always accepted
    CHECK_FALSE(w.GetGravityWell().enabled);
}

TEST_CASE("GravityWell: gravityScale multiplies the well too", "[physics][gravitywell]")
{
    PhysicsWorld w(ZeroG());
    w.SetGravityWell(Well(Real(0), Real(0), Real(2), Real(8)));
    const BodyHandle a = Ball(w, Real(2), Real(0));
    const BodyHandle b = Ball(w, Real(0), Real(2));
    w.SetGravityScale(b, Real(0));
    w.Step(kStep);
    CHECK(w.Velocity(a).x < Real(0));
    CHECK(w.Velocity(b).x == Real(0));
    CHECK(w.Velocity(b).y == Real(0));
}

TEST_CASE("GravityWell: a circular orbit stays bounded for 20 periods (per-sub-step evaluation)", "[physics][gravitywell]")
{
    // r = 10 around a well of surface radius 5 and g0 = 10: g(10) = 2.5,
    // v_circ = sqrt(g r) = 5, period = 2 pi r / v ~ 12.57 s.
    PhysicsWorld w(ZeroG());
    w.SetGravityWell(Well(Real(0), Real(0), Real(5), Real(10)));
    BodyDef d;
    d.type = BodyType::Dynamic; d.shape = MakeCircle(Real(0.1)); d.density = Real(1);
    d.position = Vec2(Real(10), Real(0));
    d.sleepThreshold = Real(0); // an orbit must never be put to sleep by the test itself
    const BodyHandle b = w.AddBody(d);
    w.SetVelocity(b, Vec2(Real(0), Real(5)));
    double rMin = 1e9, rMax = 0;
    const int steps = static_cast<int>(20 * 12.566 * 60);
    for (int i = 0; i < steps; ++i)
    {
        w.Step(kStep);
        const Vec2 p = w.Position(b);
        const double r = std::sqrt(double(p.x) * p.x + double(p.y) * p.y);
        rMin = std::min(rMin, r);
        rMax = std::max(rMax, r);
    }
    INFO("rMin " << rMin << " rMax " << rMax);
    CHECK(rMin > 9.9);
    CHECK(rMax < 10.1);
}

TEST_CASE("GravityWell: a box rests on a segmented curved ground under radial gravity, and sleeps", "[physics][gravitywell]")
{
    // A planet of radius 20 built from 128 static flat segments (two-point
    // manifolds, unlike a single circle), a box placed on it 40 degrees around.
    PhysicsWorld w(ZeroG());
    const Real R = Real(20);
    w.SetGravityWell(Well(Real(0), Real(0), R, Real(10)));
    const int N = 128;
    const Real half = R * std::tan(kPi / Real(N)) + Real(0.02);
    for (int k = 0; k < N; ++k)
    {
        const Real a = (Real(k) + Real(0.5)) * Real(2) * kPi / Real(N);
        BodyDef s;
        s.type     = BodyType::Static;
        s.position = Vec2((R - Real(0.1)) * std::cos(a), (R - Real(0.1)) * std::sin(a));
        s.shape    = MakeAabb(half, Real(0.1));
        s.friction = Real(0.6);
        const BodyHandle h = w.AddBody(s);
        w.SetAngle(h, a + kPi / Real(2));
    }
    const Real at = kPi * Real(40) / Real(180);
    BodyDef d;
    // (a polygon box, not MakeAabb: it must tilt to sit flat 40 degrees round the planet,
    // and a dynamic AABB is fixedRotation by contract -- PhysicsWorld asserts it)
    d.type = BodyType::Dynamic; d.density = Real(1);
    d.shape = MakePolygon(std::vector<Vec2>{ Vec2(Real(-0.3), Real(-0.3)), Vec2(Real(0.3), Real(-0.3)), Vec2(Real(0.3), Real(0.3)), Vec2(Real(-0.3), Real(0.3)) });
    d.friction = Real(0.6);
    d.position = Vec2((R + Real(0.31)) * std::cos(at), (R + Real(0.31)) * std::sin(at));
    const BodyHandle box = w.AddBody(d);
    w.SetAngle(box, at + kPi / Real(2));
    for (int i = 0; i < 6 * 60; ++i) { w.Step(kStep); }
    const Vec2 p = w.Position(box);
    const double r = std::sqrt(double(p.x) * p.x + double(p.y) * p.y);
    const double drift = std::abs(std::atan2(double(p.y), double(p.x)) - double(at)) * double(R);
    INFO("r " << r << " drift " << drift);
    CHECK_THAT(r, WithinAbs(20.3, 0.03));   // resting on the surface
    CHECK(drift < 0.05);                    // not sliding round the planet
    CHECK_FALSE(w.IsAwake(box));            // and asleep: the well does not keep it awake
}

TEST_CASE("GravityWell: two identical worlds under a well stay bit-identical", "[physics][gravitywell][determinism]")
{
    auto run = []() {
        PhysicsWorld w(ZeroG());
        w.SetGravityWell(Well(Real(0), Real(30), Real(25), Real(10)));
        std::vector<BodyHandle> hs;
        for (int i = 0; i < 40; ++i) { hs.push_back(Ball(w, Real(i % 8) * Real(0.5) - Real(2), Real(-2) - Real(i / 8) * Real(0.5), Real(0.2))); }
        for (int i = 0; i < 600; ++i) { w.Step(kStep); }
        std::vector<Real> out;
        for (const BodyHandle h : hs) { const Vec2 p = w.Position(h); out.push_back(p.x); out.push_back(p.y); }
        return out;
    };
    const std::vector<Real> a = run(), b = run();
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) { CHECK(a[i] == b[i]); }
}
