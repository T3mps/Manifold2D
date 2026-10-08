// Physics M6 P1.8: PhysicsWorld core + Body. Contact begin/end and sensor
// overlap moved onto the event arrays (spec s6.5). Stay, the per-body gate,
// and the re-arm burst are gone.
//
// PORT NOTE: a behavioral port of the physics_harness blocks
// (Client/src/tests/physics_harness/main.lua):
//   * "== PhysicsWorld core ==" (~280-337): handle validity, kinematic
//     integration + drawPosition lerp, statics never integrate, QueryAABB,
//     handle generation (removal invalidates; slot reuse keeps stale invalid),
//     run-twice determinism.
//   * "== contact events ==" (~339-379): apart -> 0 begins; overlap -> one
//     Begin; still touching -> no further Begin/End (GetBodyContacts); separate
//     -> End. A static sensor overlap -> one sensor Begin with valid handles.
//
// The expected values are the literals from the harness (coordinate-agnostic,
// no Map/iso needed: the world is built with NO passability source and bodies
// are placed at plain world coords, meters since MKS P4). EventLog copies
// GetContactEvents / GetSensorEvents. The narrowphase / broadphase are the same
// modules the Lua harness exercised, so the overlap decisions match.
//
// PRESENTATION-FREE + C++23-clean.

#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include "PhysicsEventTestHelpers.hpp"

using namespace Manifold2D::Physics;
using Catch::Approx;

namespace
{
    constexpr Real kStep = Real(1) / Real(60);
}

// ---------------------------------------------------------------------------
// PhysicsWorld core
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsWorld: handle validity + kinematic integration", "[physics][world]")
{
    WorldDef wd;
    PhysicsWorld w(wd); // no passability source -> plain Cartesian coords

    BodyDef def;
    def.type     = BodyType::Kinematic;
    def.position = Vec2(Real(1), Real(2));
    def.shape    = MakeCircle(Real(0.5));
    BodyHandle body = w.AddBody(def);

    REQUIRE(w.IsValid(body));
    REQUIRE(w.Position(body).x == Approx(Real(1)));
    REQUIRE(w.Position(body).y == Approx(Real(2)));

    // kinematic integration: velocity moves the body; prev tracks for lerp.
    w.SetVelocity(body, Vec2(Real(6), Real(0)));
    w.Step(kStep);
    REQUIRE(w.Position(body).x == Approx(Real(1.1)));  // 1 + 6*(1/60)
    REQUIRE(w.Position(body).y == Approx(Real(2)));
    REQUIRE(w.DrawPosition(body, Real(0.5)).x == Approx(Real(1.05))); // prev=1,cur=1.1
}

TEST_CASE("PhysicsWorld: statics never integrate", "[physics][world]")
{
    WorldDef wd;
    PhysicsWorld w(wd);

    BodyDef kdef;
    kdef.type     = BodyType::Kinematic;
    kdef.position = Vec2(Real(1), Real(2));
    kdef.shape    = MakeCircle(Real(0.5));
    w.AddBody(kdef);

    BodyDef sdef;
    sdef.type     = BodyType::Static;
    sdef.position = Vec2(Real(10), Real(10));
    sdef.shape    = MakeAabb(Real(1), Real(0.6));
    BodyHandle prop = w.AddBody(sdef);

    w.SetVelocity(prop, Vec2(Real(5), Real(5))); // ignored for statics
    w.Step(kStep);
    REQUIRE(w.Position(prop).x == Approx(Real(10))); // pinned
    REQUIRE(w.Position(prop).y == Approx(Real(10)));
}

TEST_CASE("PhysicsWorld: QueryAABB sees both body kinds", "[physics][world]")
{
    WorldDef wd;
    PhysicsWorld w(wd);

    BodyDef kdef;
    kdef.type     = BodyType::Kinematic;
    kdef.position = Vec2(Real(1), Real(2));
    kdef.shape    = MakeCircle(Real(0.5));
    w.AddBody(kdef);

    BodyDef sdef;
    sdef.type     = BodyType::Static;
    sdef.position = Vec2(Real(10), Real(10));
    sdef.shape    = MakeAabb(Real(1), Real(0.6));
    w.AddBody(sdef);

    std::vector<BodyHandle> hits;
    int n = w.QueryAABB(Aabb{ Vec2(Real(0), Real(0)), Vec2(Real(20), Real(20)) }, hits);
    REQUIRE(n == 2);
}

TEST_CASE("PhysicsWorld: removal invalidates handle + slot reuse bumps generation",
          "[physics][world]")
{
    WorldDef wd;
    PhysicsWorld w(wd);

    BodyDef sdef;
    sdef.type     = BodyType::Static;
    sdef.position = Vec2(Real(10), Real(10));
    sdef.shape    = MakeAabb(Real(1), Real(0.6));
    BodyHandle prop = w.AddBody(sdef);

    REQUIRE(w.IsValid(prop));
    w.RemoveBody(prop);
    REQUIRE_FALSE(w.IsValid(prop)); // removed handle invalid

    // Slot reuse: the next add recycles prop's slot at a bumped generation.
    BodyDef again;
    again.type     = BodyType::Static;
    again.position = Vec2(Real(0.1), Real(0.1));
    again.shape    = MakeCircle(Real(0.2));
    BodyHandle h2 = w.AddBody(again);

    REQUIRE_FALSE(w.IsValid(prop)); // stale handle stays invalid after reuse
    REQUIRE(w.IsValid(h2));
    REQUIRE(h2.index == prop.index);          // same slot
    REQUIRE(h2.generation != prop.generation); // bumped generation
}

TEST_CASE("PhysicsWorld: determinism -- identical input -> identical state",
          "[physics][world]")
{
    auto run = []() -> double
    {
        WorldDef wd;
        PhysicsWorld w(wd);
        BodyDef def;
        def.type     = BodyType::Kinematic;
        def.position = Vec2(Real(0), Real(0));
        def.shape    = MakeCircle(Real(0.4));
        BodyHandle b = w.AddBody(def);

        double acc = 0.0;
        for (int i = 1; i <= 120; ++i)
        {
            const Real vx = Real((i % 7) * 1 - 3);
            const Real vy = Real((i % 5) * 0.8 - 1.6);
            w.SetVelocity(b, Vec2(vx, vy));
            w.Step(kStep);
            const Vec2 p = w.Position(b);
            acc += double(p.x) * 31.0 + double(p.y) * 17.0;
        }
        return acc;
    };
    REQUIRE(run() == run());
}

TEST_CASE("PhysicsWorld: Body view forwards to the world", "[physics][world]")
{
    WorldDef wd;
    PhysicsWorld w(wd);
    BodyDef def;
    def.type     = BodyType::Kinematic;
    def.position = Vec2(Real(1), Real(2));
    def.shape    = MakeCircle(Real(0.5));
    Body body = w.GetBody(w.AddBody(def));

    REQUIRE(body.IsValid());
    REQUIRE(body.GetPosition().x == Approx(Real(1)));
    body.SetVelocity(Vec2(Real(6), Real(0)));
    w.Step(kStep);
    REQUIRE(body.GetPosition().x == Approx(Real(1.1)));
    REQUIRE(body.DrawPosition(Real(0.5)).x == Approx(Real(1.05)));
    REQUIRE(body.GetType() == BodyType::Kinematic);
}

// ---------------------------------------------------------------------------
// Contact begin / end (spec s6.1). The harness's kinematic pair has no solver
// contact (A9), so this is a dynamic box against a static floor. Stay is gone
// (s6.5); "still touching" is GetBodyContacts.
// ---------------------------------------------------------------------------

TEST_CASE("contact begin and end fire when a dynamic pair overlaps and separates",
          "[physics][contacts]")
{
    PhysicsWorld w{ WorldDef{} };

    BodyDef floor;
    floor.type          = BodyType::Static;
    floor.position      = Vec2(Real(0), Real(0.5)); // top face at y = 0
    floor.shape         = MakeAabb(Real(10), Real(0.5));
    floor.contactEvents = true;
    const BodyHandle ground = w.AddBody(floor);

    BodyDef box;
    box.type           = BodyType::Dynamic;
    box.fixedRotation  = true; // dynamic AABBs require fixedRotation
    box.position       = Vec2(Real(0), Real(-3)); // clear of the floor
    box.shape          = MakeAabb(Real(0.5), Real(0.5));
    box.contactEvents  = true;
    box.restitution    = Real(0);
    const BodyHandle body = w.AddBody(box);

    EventTest::EventLog log;
    log.StepAndCollect(w, 1); // gravity cannot close a 2.5 m gap in one step
    REQUIRE(log.begin.empty());
    REQUIRE(log.end.empty());

    // Overlap the floor by 0.1 m (box spans y [-0.9, 0.1], floor starts at 0).
    w.SetPosition(body, Vec2(Real(0), Real(-0.4)));
    log.StepAndCollect(w, 1);
    REQUIRE(log.begin.size() == 1);
    CHECK(log.end.empty());
    CHECK(log.begin[0].bodyA == body);   // canonical: A is the dynamic side
    CHECK(log.begin[0].bodyB == ground);
    CHECK(w.IsValid(log.begin[0].bodyA));
    CHECK(w.IsValid(log.begin[0].bodyB));

    std::vector<BodyContact> touching;
    w.GetBodyContacts(body, touching);
    REQUIRE_FALSE(touching.empty());     // still a contact; there is no Stay event

    log.StepAndCollect(w, 1);            // resting: no second Begin, no End
    CHECK(log.begin.size() == 1);
    CHECK(log.end.empty());

    w.SetPosition(body, Vec2(Real(0), Real(-5)));
    log.StepAndCollect(w, 2);
    CHECK(log.end.size() == 1);
    CHECK(log.end[0].bodyA == body);
    CHECK(log.end[0].bodyB == ground);
}

TEST_CASE("static sensor overlap reports a sensor begin with valid handles",
          "[physics][contacts]")
{
    PhysicsWorld w{ WorldDef{} };

    BodyDef ad;
    ad.type         = BodyType::Kinematic;
    ad.position     = Vec2(Real(0), Real(0));
    ad.shape        = MakeCircle(Real(0.5));
    ad.sensorEvents = true;
    const BodyHandle kin = w.AddBody(ad);

    BodyDef sd;
    sd.type         = BodyType::Static;
    sd.position     = Vec2(Real(0), Real(0));
    sd.shape        = MakeAabb(Real(0.8), Real(0.8));
    sd.isSensor     = true;
    sd.sensorEvents = true;
    const BodyHandle sens = w.AddBody(sd);

    w.Step(kStep);
    const SensorEvents ev = w.GetSensorEvents();
    REQUIRE(ev.begin.size() == 1);                 // the sensor channel is the sensor flag
    CHECK(ev.begin[0].sensorBody == sens);
    CHECK(ev.begin[0].visitorBody == kin);
    CHECK(w.IsValid(ev.begin[0].sensorBody));
    CHECK(w.IsValid(ev.begin[0].visitorBody));
    CHECK(w.GetContactEvents().begin.empty());     // a sensor pair is not a contact begin
}
