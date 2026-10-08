// PhysicsSensorEventsTest.cpp
// [physics][events]: the end-of-step sensor pass (spec s6.1; Box2D 3.1.1 sensor.c,
// with the sensor-vs-sensor exclusion as amendment A1).
#include <catch2/catch_test_macros.hpp>
#include "PhysicsEventTestHelpers.hpp"

using namespace EventTest;

namespace
{
    BodyHandle AddStaticSensor(PhysicsWorld& w, Real x, Real y)
    {
        BodyDef d; d.type = BodyType::Static; d.position = Vec2(x, y);
        d.shape = MakeAabb(Real(1), Real(1)); d.isSensor = true; d.sensorEvents = true;
        return w.AddBody(d);
    }
}

TEST_CASE("A static sensor sees a falling box enter and leave", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    const BodyHandle s = AddStaticSensor(w, Real(0), Real(-6));   // a zone in mid-air
    const BodyHandle b = AddBox(w, Real(0), Real(-10));
    EventLog log; log.StepAndCollect(w, 120);
    REQUIRE(log.sensorBegin.size() == 1);
    REQUIRE(log.sensorEnd.size() == 1);
    CHECK(log.sensorBegin[0].sensorBody == s);
    CHECK(log.sensorBegin[0].visitorBody == b);
}

TEST_CASE("A dynamic sensor detects a static fixture; sensors never detect sensors", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    AddGround(w);                                                   // static visitor (y 0..1), sensorEvents on
    BodyDef d; d.type = BodyType::Dynamic; d.position = Vec2(Real(0), Real(-2));
    d.shape = MakeAabb(Real(0.5), Real(0.5)); d.isSensor = true; d.sensorEvents = true;
    d.fixedRotation = true;                                         // dynamic AABBs must be fixedRotation
    w.AddBody(d);                                                   // falls THROUGH the ground (no response)
    AddStaticSensor(w, Real(0), Real(4));                           // another sensor on its path (y 3..5), clear of the ground
    EventLog log; log.StepAndCollect(w, 120);                       // ~20 m of fall: passes both
    REQUIRE(log.sensorBegin.size() == 1);                           // the ground only; never sensor-vs-sensor (A1)
}

TEST_CASE("Same-body, filtered and opted-out visitors are skipped", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    BodyDef d; d.type = BodyType::Dynamic; d.position = Vec2(Real(0), Real(-2));
    d.shape = MakeAabb(Real(1), Real(1)); d.isSensor = true; d.sensorEvents = true;
    d.fixedRotation = true;                                         // dynamic AABBs must be fixedRotation
    const BodyHandle s = w.AddBody(d);
    FixtureDef solid; solid.shape = MakeCircle(Real(0.2)); solid.sensorEvents = true;
    w.AddFixture(s, solid);                                         // same body: never a visitor (sensor.c:72)
    AddBox(w, Real(0), Real(-2), true, false, /*sensorEvents*/ false);   // opted out (sensor.c:66)
    const BodyHandle f = AddBox(w, Real(0.5), Real(-2));
    w.SetBodyFilter(f, 2u, 0u);                                     // filtered out (sensor.c:77)
    EventLog log; log.StepAndCollect(w, 1);
    CHECK(log.sensorBegin.empty());
}

TEST_CASE("Turning a sensor's flag off ends its overlaps on the next step", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    const BodyHandle s = AddStaticSensor(w, Real(0), Real(0));
    AddBox(w, Real(0), Real(0));
    EventLog log; log.StepAndCollect(w, 1);
    REQUIRE(log.sensorBegin.size() == 1);
    w.SetFixtureEvents(w.GetBodyFixture(s, 0), false, false, false);
    log.StepAndCollect(w, 1);
    CHECK(log.sensorEnd.size() == 1);                               // Box2D sensor.c:158-165
}

TEST_CASE("a recycled visitor slot ends the old overlap and begins the new", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    AddStaticSensor(w, Real(0), Real(0));
    const BodyHandle old = AddBox(w, Real(0), Real(0));
    EventLog log; log.StepAndCollect(w, 1);
    REQUIRE(log.sensorBegin.size() == 1);
    w.RemoveBody(old);
    const BodyHandle neu = AddBox(w, Real(0), Real(0));             // LIFO recycle: same index, new generation
    REQUIRE(neu.index == old.index);
    log.StepAndCollect(w, 1);
    REQUIRE(log.sensorEnd.size() == 1);
    CHECK(log.sensorEnd[0].visitorBody == old);
    REQUIRE(log.sensorBegin.size() == 2);
    CHECK(log.sensorBegin[1].visitorBody == neu);
}

TEST_CASE("Removing a sensor ends its overlaps", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    const BodyHandle s = AddStaticSensor(w, Real(0), Real(0));
    AddBox(w, Real(0), Real(0));
    EventLog log; log.StepAndCollect(w, 1);
    w.RemoveBody(s);
    log.StepAndCollect(w, 1);
    CHECK(log.sensorEnd.size() == 1);
}

TEST_CASE("a sensor Begin reported while open still gets its End when the gate is closed", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    AddStaticSensor(w, Real(0), Real(0));
    const BodyHandle b = AddBox(w, Real(0), Real(0));
    EventLog log;
    log.StepAndCollect(w, 1);                          // overlap begins while the gate is open
    REQUIRE(log.sensorBegin.size() == 1);
    CHECK(log.sensorEnd.empty());
    w.SetEventsEnabled(false);
    w.SetPosition(b, Vec2(Real(0), Real(-10)));        // leave while gated
    log.StepAndCollect(w, 1);
    CHECK(log.sensorBegin.size() == 1);                // no further Begin
    CHECK(log.sensorEnd.size() == 1);                  // the reported Begin still closes (R12)
}

TEST_CASE("an overlap begun while gated never produces an End after re-enable", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    AddStaticSensor(w, Real(0), Real(0));
    const BodyHandle b = AddBox(w, Real(0), Real(0));
    w.SetEventsEnabled(false);
    EventLog log;
    log.StepAndCollect(w, 1);                          // enters while gated
    CHECK(log.sensorBegin.empty());
    w.SetEventsEnabled(true);
    w.SetPosition(b, Vec2(Real(0), Real(-10)));        // leaves after re-enable
    log.StepAndCollect(w, 1);
    CHECK(log.sensorBegin.empty());                    // no Begin was ever delivered
    CHECK(log.sensorEnd.empty());                      // so no End (R12)
}

TEST_CASE("re-enabling while overlapping emits no burst", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    AddStaticSensor(w, Real(0), Real(0));
    AddBox(w, Real(0), Real(0));
    w.SetEventsEnabled(false);
    EventLog log;
    log.StepAndCollect(w, 1);                          // overlap begins while gated
    CHECK(log.sensorBegin.empty());
    w.SetEventsEnabled(true);
    log.StepAndCollect(w, 1);                          // still overlapping
    CHECK(log.sensorBegin.empty());                    // no burst
    CHECK(log.sensorEnd.empty());
}
