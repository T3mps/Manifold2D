// PhysicsHitEventsTest.cpp
// [physics][events]: hit events (spec s6.1, Box2D solver.c:1758-1814; amendment A4).
#include <cmath>
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
    log.StepAndCollect(w, 90);
    REQUIRE(log.hit.size() >= 1);
    const ContactHitEvent& h = log.hit.front();
    CHECK(h.approachSpeed == Catch::Approx(std::sqrt(40.0)).margin(0.35));  // one step of g*dt slack
    CHECK(h.normal.y > Real(0.99));          // A (box) -> B (ground) is +y in a +Y-down world (A8)
    CHECK(h.point.y == Catch::Approx(0.0).margin(0.05));
}

TEST_CASE("No hit below the threshold, and none without the opt-in", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    w.SetHitEventThreshold(Real(50));        // above any speed here
    AddGround(w);
    AddBox(w, Real(-2), Real(-2.5), true, true);
    AddBox(w, Real(2), Real(-2.5), true, false);   // never opted in
    EventLog log; log.StepAndCollect(w, 90); // a 2 m drop lands at ~step 38 (t = sqrt(2h/g) = 0.63 s)
    w.SetHitEventThreshold(Real(1));
    log.StepAndCollect(w, 60);               // both now resting: no approach speed above 1 m/s
    // The first box landed while the threshold was 50; the second never opted in.
    CHECK(log.hit.empty());
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
