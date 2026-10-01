// PhysicsContinuousTest.cpp
// [physics][ccd][continuous]: continuous collision for FAST non-bullet bodies
// against static geometry (Box2D v3 b2FinalizeBodiesTask + b2SolveContinuous).
//
// After the solve, a dynamic body is "fast" when its motion this step exceeds
// safetyFactor * minExtent (default 0.5; minExtent = the smallest centroid-to-
// surface distance of its shapes). A fast body's step is swept from its start
// pose to its end pose against static bodies and tile spans (filters honoured,
// sensors skipped); at the earliest 0 < fraction it is put back to that pose,
// keeping its velocity but giving back the gravity of the lost time. A sweep
// that starts touching (fraction 0) is retried with a small core circle
// (0.25 * minExtent) about the shape's centroid, so a body sliding along a
// surface is never stopped but its core cannot pass through.
//
// Without it, a hard hit against a static wall overshot the speculative gap in
// the impact step (the sequential two-point solve spins the box) and sank in by
// ~0.1-0.15 m.
//
// PRESENTATION-FREE, ASCII comments, C++23.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>

using namespace Manifold2D::Physics;

namespace
{
    constexpr Real kStep = Real(1) / Real(60);

    Shape Box(Real hx, Real hy)
    {
        return MakePolygon(std::vector<Vec2>{ Vec2(-hx, -hy), Vec2(hx, -hy), Vec2(hx, hy), Vec2(-hx, hy) });
    }

    WorldDef ZeroG(bool continuous = true)
    {
        WorldDef wd;
        wd.gravityX = Real(0);
        wd.gravityY = Real(0);
        wd.enableContinuous = continuous;
        return wd;
    }

    struct Shot { Real maxOverlap = 0; bool through = false; };

    // A projectile fired at +x into a 14 cm static wall at x = 0.
    Shot Fire(PhysicsWorld& w, Shape shape, Real reach, Real speed, Real spin = 0,
              std::uint32_t cat = 1u, std::uint32_t mask = 0xFFFFFFFFu)
    {
        BodyDef wall;
        wall.type = BodyType::Static;
        wall.shape = Box(Real(0.07), Real(2.5));
        w.AddBody(wall);
        BodyDef b;
        b.type = BodyType::Dynamic;
        b.position = Vec2(Real(-3), Real(0));
        b.shape = shape;
        b.restitution = Real(0.1);
        b.categoryBits = cat;
        b.maskBits = mask;
        const BodyHandle p = w.AddBody(b);
        w.SetVelocity(p, Vec2(speed, Real(0)));
        w.SetAngularVelocity(p, spin);
        Shot s;
        for (int i = 0; i < 60; ++i)
        {
            w.Step(kStep);
            const Real x = w.Position(p).x;
            s.maxOverlap = std::max(s.maxOverlap, x + reach + Real(0.07));
            if (x - reach > Real(0.07)) { s.through = true; }
        }
        return s;
    }
} // namespace

TEST_CASE("Continuous: a fast box ends its impact step at a static wall's surface",
          "[physics][ccd][continuous]")
{
    for (const Real speed : { Real(15), Real(40) })
    {
        PhysicsWorld w{ ZeroG() };
        const Shot s = Fire(w, Box(Real(0.15), Real(0.15)), Real(0.15), speed);
        INFO("speed " << speed << " max overlap " << s.maxOverlap);
        CHECK_FALSE(s.through);
        CHECK(s.maxOverlap < Real(0.02)); // was ~0.10 at 15 m/s, ~0.15 at 40 m/s
    }
}

TEST_CASE("Continuous: switched off, the impact step overshoots as before",
          "[physics][ccd][continuous]")
{
    PhysicsWorld w{ ZeroG(false) };
    const Shot s = Fire(w, Box(Real(0.15), Real(0.15)), Real(0.15), Real(15));
    INFO("max overlap " << s.maxOverlap);
    CHECK(s.maxOverlap > Real(0.05));
}

TEST_CASE("Continuous: a fast spinning thin bar does not tunnel a static wall",
          "[physics][ccd][continuous]")
{
    PhysicsWorld w{ ZeroG() };
    const Shot s = Fire(w, Box(Real(0.2), Real(0.05)), Real(0.2), Real(80), Real(25));
    CHECK_FALSE(s.through); // tunnelled before unless flagged a bullet
}

TEST_CASE("Continuous: the sweep honours collision filters",
          "[physics][ccd][continuous][filter]")
{
    PhysicsWorld w{ ZeroG() };
    // category 2 against a wall of category 1 that it does not collide with
    const Shot s = Fire(w, Box(Real(0.15), Real(0.15)), Real(0.15), Real(20), Real(0), 2u, 2u);
    CHECK(s.through);
}

TEST_CASE("Continuous: a fast body sliding along a static floor is not stopped by it",
          "[physics][ccd][continuous]")
{
    // It touches the floor at the start of every sweep (fraction 0); the core-circle
    // retry does not touch, so nothing clamps it.
    PhysicsWorld w{ ZeroG() };
    BodyDef f;
    f.type = BodyType::Static;
    f.position = Vec2(Real(0), Real(0.5)); // top face at y = 0 (+y down)
    f.shape = MakeAabb(Real(40), Real(0.5));
    w.AddBody(f);
    BodyDef b;
    b.type = BodyType::Dynamic;
    b.position = Vec2(Real(-20), Real(-0.15));
    b.shape = Box(Real(0.15), Real(0.15));
    b.friction = Real(0);
    const BodyHandle p = w.AddBody(b);
    w.SetVelocity(p, Vec2(Real(15), Real(0)));
    for (int i = 0; i < 30; ++i) { w.Step(kStep); }
    INFO("x " << w.Position(p).x);
    CHECK(w.Position(p).x > Real(-14)); // ~7.5 m in 0.5 s
}

TEST_CASE("Continuous: a box thrown down onto static ground lands on its surface",
          "[physics][ccd][continuous]")
{
    PhysicsWorld w{ WorldDef{} }; // gravity on
    BodyDef f;
    f.type = BodyType::Static;
    f.position = Vec2(Real(0), Real(0.5));
    f.shape = MakeAabb(Real(10), Real(0.5));
    w.AddBody(f);
    BodyDef b;
    b.type = BodyType::Dynamic;
    b.position = Vec2(Real(0), Real(-5));
    b.shape = Box(Real(0.15), Real(0.15));
    const BodyHandle p = w.AddBody(b);
    w.SetVelocity(p, Vec2(Real(0), Real(30)));
    Real deepest = Real(0);
    for (int i = 0; i < 120; ++i)
    {
        w.Step(kStep);
        deepest = std::max(deepest, w.Position(p).y + Real(0.15));
    }
    INFO("deepest " << deepest << " final y " << w.Position(p).y);
    CHECK(deepest < Real(0.02));
    CHECK(std::abs(w.Position(p).y + Real(0.15)) < Real(0.02)); // resting on the surface
}

TEST_CASE("Continuous: a continuous scene is deterministic (run twice -> identical)",
          "[physics][ccd][continuous][determinism]")
{
    const auto run = []()
    {
        PhysicsWorld w{ WorldDef{} };
        BodyDef f;
        f.type = BodyType::Static;
        f.position = Vec2(Real(0), Real(0.5));
        f.shape = MakeAabb(Real(10), Real(0.5));
        w.AddBody(f);
        std::vector<BodyHandle> bs;
        for (int i = 0; i < 8; ++i)
        {
            BodyDef b;
            b.type = BodyType::Dynamic;
            b.position = Vec2(Real(-4) + Real(i), Real(-4) - Real(i % 3));
            b.shape = (i % 2) ? Box(Real(0.15), Real(0.1)) : MakeCircle(Real(0.12));
            bs.push_back(w.AddBody(b));
            w.SetVelocity(bs.back(), Vec2(Real(i - 4), Real(20 + i)));
        }
        for (int i = 0; i < 120; ++i) { w.Step(kStep); }
        std::vector<Real> out;
        for (const BodyHandle h : bs) { out.push_back(w.Position(h).x); out.push_back(w.Position(h).y); }
        return out;
    };
    CHECK(run() == run());
}

