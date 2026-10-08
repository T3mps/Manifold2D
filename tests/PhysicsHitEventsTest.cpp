// PhysicsHitEventsTest.cpp
// [physics][events]: hit events (spec s6.1, Box2D solver.c:1758-1814; amendment A4).
#include <cmath>
#include <cstddef>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "PhysicsEventTestHelpers.hpp"

using namespace EventTest;

TEST_CASE("A dropped box reports one hit with the free-fall approach speed", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    AddGround(w);
    // Bottom face 2 m above the ground: v = sqrt(2 g h) = sqrt(40) ~ 6.32 m/s.
    AddBox(w, Real(0), Real(-2.5), true, /*hit*/ true);
    EventLog log;
    for (int step = 0; step < 90; ++step)
    {
        w.Step(kStep);
        const ContactEvents c = w.GetContactEvents();
        // One hit per pair per step (Box2D solver.c:1782-1811).
        for (std::size_t i = 0; i < c.hit.size(); ++i)
        {
            for (std::size_t j = i + 1; j < c.hit.size(); ++j)
            {
                const bool samePair = (c.hit[i].a == c.hit[j].a && c.hit[i].b == c.hit[j].b);
                CHECK_FALSE(samePair);
            }
        }
        log.Collect(w);
    }
    REQUIRE(log.hit.size() >= 1);
    const ContactHitEvent& h = log.hit.front();
    CHECK(h.approachSpeed == Catch::Approx(std::sqrt(40.0)).margin(0.35));  // one step of g*dt slack
    CHECK(h.normal.y > Real(0.99));          // A (box) -> B (ground) is +y in a +Y-down world (A8)
    CHECK(h.point.y == Catch::Approx(0.0).margin(0.05));
}

TEST_CASE("No hit below the threshold", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    w.SetHitEventThreshold(Real(50));        // above the ~6.3 m/s landing
    AddGround(w);
    AddBox(w, Real(0), Real(-2.5), true, /*hit*/ true);
    EventLog log;
    log.StepAndCollect(w, 90);               // lands at ~step 38 (t = sqrt(2h/g) = 0.63 s)
    CHECK(log.hit.empty());
}

TEST_CASE("Only the opted-in box reports hits when both drop at the default threshold", "[physics][events]")
{
    // Default is 1 m/s (WorldDef::hitEventThreshold). A 2 m drop lands at ~6.3 m/s,
    // so the threshold does not suppress either box: only hitEvents does.
    PhysicsWorld w{ WorldDef{} };
    CHECK(WorldDef{}.hitEventThreshold == Real(1));
    AddGround(w);
    const BodyHandle opted = AddBox(w, Real(-2), Real(-2.5), true, /*hit*/ true);
    const BodyHandle silent = AddBox(w, Real(2), Real(-2.5), true, /*hit*/ false);
    EventLog log;
    log.StepAndCollect(w, 90);
    REQUIRE(log.hit.size() >= 1);
    for (const ContactHitEvent& h : log.hit)
    {
        CHECK((h.bodyA == opted || h.bodyB == opted));
        CHECK_FALSE(h.bodyA == silent);
        CHECK_FALSE(h.bodyB == silent);
    }
}

TEST_CASE("A resting box produces no hits", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    AddGround(w);
    AddBox(w, Real(0), Real(-0.5), true, true);
    EventLog settle; settle.StepAndCollect(w, 120);
    EventLog log; log.StepAndCollect(w, 60);
    CHECK(log.hit.empty());
}
