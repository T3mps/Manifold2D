// PhysicsContactEventsTest.cpp
// [physics][events]: contact begin/end arrays (spec 2026-10-08 s6.1).
#include <catch2/catch_test_macros.hpp>
#include "PhysicsEventTestHelpers.hpp"

using namespace EventTest;

TEST_CASE("Event flags default off and are captured when a contact is created", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    FixtureDef fd;                       // defaults
    CHECK_FALSE(fd.contactEvents);
    CHECK_FALSE(fd.sensorEvents);
    CHECK_FALSE(fd.hitEvents);
    CHECK(WorldDef{}.hitEventThreshold == Real(1));
    CHECK_FALSE(WorldDef{}.contactEventsRequireBoth);

    const BodyHandle g = AddGround(w, /*contactEvents*/ false);
    const BodyHandle b = AddBox(w, Real(0), Real(-0.49), /*contactEvents*/ true, /*hit*/ true);
    w.Step(kStep);
    const std::uint8_t flags = w.DebugContactEventFlags(w.GetBodyFixture(b, 0), w.GetBodyFixture(g, 0));
    REQUIRE(flags != 0xFF);              // the pair exists
    CHECK(flags == (kEvContact | kEvHit)); // either-fixture rule (Box2D contact.c:253, :535)
}

TEST_CASE("contactEventsRequireBoth needs both fixtures; hit stays either", "[physics][events]")
{
    WorldDef wd; wd.contactEventsRequireBoth = true;
    PhysicsWorld w{ wd };
    const BodyHandle g = AddGround(w, /*contactEvents*/ false);
    const BodyHandle b = AddBox(w, Real(0), Real(-0.49), /*contactEvents*/ true, /*hit*/ true);
    w.Step(kStep);
    CHECK(w.DebugContactEventFlags(w.GetBodyFixture(b, 0), w.GetBodyFixture(g, 0)) == kEvHit);
}

TEST_CASE("A sensor pair captures no contact or hit flags", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    AddGround(w);
    BodyDef d; d.type = BodyType::Dynamic; d.fixedRotation = true; d.position = Vec2(Real(0), Real(-2));
    d.shape = MakeAabb(Real(0.5), Real(0.5)); d.isSensor = true; d.contactEvents = true; d.hitEvents = true;
    const BodyHandle s = w.AddBody(d);
    const BodyHandle o = AddBox(w, Real(0), Real(-2), true, true);   // overlapping mover pair
    w.Step(kStep);
    const std::uint8_t f = w.DebugContactEventFlags(w.GetBodyFixture(s, 0), w.GetBodyFixture(o, 0));
    CHECK((f == 0xFF || f == 0u));       // either no pool contact or a non-solver one
}
