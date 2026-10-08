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

TEST_CASE("A flag change leaves an existing contact's bits alone and applies to the next contact",
          "[physics][events]")
{
    // Zero-g so the authored overlaps stay put (spec s6.1 / amendment A3:
    // contact and hit bits are fixed when the pool contact is created).
    WorldDef wd;
    wd.gravityX = Real(0);
    wd.gravityY = Real(0);
    PhysicsWorld w{ wd };

    const BodyHandle g = AddGround(w, /*contactEvents*/ false);
    const BodyHandle b = AddBox(w, Real(0), Real(-0.49), /*contactEvents*/ true, /*hit*/ true);
    w.Step(kStep);

    const FixtureHandle bf = w.GetBodyFixture(b, 0);
    const FixtureHandle gf = w.GetBodyFixture(g, 0);
    REQUIRE(w.DebugContactEventFlags(bf, gf) == (kEvContact | kEvHit));

    // Ground never opted in, so a per-step recompute would drop kEvContact.
    w.SetFixtureEvents(bf, /*contact*/ false, /*sensor*/ false, /*hit*/ true);
    w.Step(kStep);
    CHECK(w.DebugContactEventFlags(bf, gf) == (kEvContact | kEvHit));

    // New box overlaps b only, and opts out of both. The fresh pair must take
    // b's updated bits (hit, no contact), not the bits captured for b-ground.
    const BodyHandle b2 = AddBox(w, Real(0), Real(-1.4),
                                 /*contactEvents*/ false, /*hit*/ false, /*sensor*/ false);
    w.Step(kStep);
    const std::uint8_t fresh = w.DebugContactEventFlags(bf, w.GetBodyFixture(b2, 0));
    REQUIRE(fresh != 0xFF);
    CHECK(fresh == kEvHit);
    CHECK(w.DebugContactEventFlags(bf, gf) == (kEvContact | kEvHit));
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

TEST_CASE("A box landing on static ground begins once and ends when lifted", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    const BodyHandle g = AddGround(w);
    const BodyHandle b = AddBox(w, Real(0), Real(-2));
    EventLog log;
    log.StepAndCollect(w, 120);
    REQUIRE(log.begin.size() == 1);                 // dynamic-vs-static now reports (the old events skipped it)
    CHECK(log.begin[0].bodyA == b);                 // canonical: A is the dynamic side (Contact.hpp:65)
    CHECK(log.begin[0].bodyB == g);
    CHECK(log.end.empty());
    w.SetPosition(b, Vec2(Real(0), Real(-5)));      // teleport away
    log.StepAndCollect(w, 2);
    CHECK(log.end.size() == 1);
}

TEST_CASE("Two dynamic boxes report begin; kinematic-vs-static reports nothing", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    AddGround(w);
    AddBox(w, Real(0), Real(-0.5));
    AddBox(w, Real(0), Real(-1.6));
    BodyDef k; k.type = BodyType::Kinematic; k.position = Vec2(Real(5), Real(-0.4));
    k.shape = MakeAabb(Real(0.5), Real(0.5)); k.contactEvents = true;
    w.AddBody(k);                                   // overlaps the ground: no solver contact in Box2D terms
    EventLog log;
    log.StepAndCollect(w, 90);
    CHECK(log.begin.size() == 2);                   // box-ground, box-box; never kinematic-ground (A9)
}

TEST_CASE("Opting out on both fixtures silences the pair", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    AddGround(w, false);
    AddBox(w, Real(0), Real(-2), false);
    EventLog log;
    log.StepAndCollect(w, 120);
    CHECK(log.begin.empty());
}

TEST_CASE("Begin arrays are sorted by fixture pair", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    AddGround(w);
    for (int i = 0; i < 6; ++i) AddBox(w, Real(-5 + 2 * i), Real(-0.49));   // all land in step 1
    w.Step(kStep);
    const ContactEvents c = w.GetContactEvents();
    REQUIRE(c.begin.size() == 6);
    for (std::size_t i = 1; i < c.begin.size(); ++i)
        CHECK((FixtureLess(c.begin[i - 1].a, c.begin[i].a) ||
               (c.begin[i - 1].a == c.begin[i].a && FixtureLess(c.begin[i - 1].b, c.begin[i].b))));
}

TEST_CASE("the world gate drops events without a burst on re-enable", "[physics][events]")
{
    PhysicsWorld w{ WorldDef{} };
    AddGround(w);
    AddBox(w, Real(0), Real(-2));
    w.SetEventsEnabled(false);
    EventLog log;
    log.StepAndCollect(w, 120);                     // lands while gated
    CHECK(log.begin.empty());
    w.SetEventsEnabled(true);
    log.StepAndCollect(w, 10);                      // still touching
    CHECK(log.begin.empty());                       // no burst
    CHECK(log.end.empty());                         // and no orphan End
}
