// Physics M6 P1.8: PhysicsWorld core + Body + ContactManager + event gating.
//
// PORT NOTE: a behavioral port of the physics_harness blocks
// (Client/src/tests/physics_harness/main.lua):
//   * "== PhysicsWorld core ==" (~280-337): handle validity, kinematic
//     integration + drawPosition lerp, statics never integrate, QueryAABB,
//     handle generation (removal invalidates; slot reuse keeps stale invalid),
//     run-twice determinism.
//   * "== ContactManager events ==" (~339-379): apart -> 0 events; jump-next
//     -> begin; next step -> exactly one stay; move apart -> end; static
//     sensor overlap -> begin with sensor flag; events carry valid handles.
//   * "== event gating ==" (~381-418): baseline begin; per-body mute drops
//     (no synthetic end); re-enable while overlapping -> fresh begin (re-arm);
//     next step -> stay; world gate off -> 0 events; world gate on -> fresh
//     begin (re-arm overlapping).
//
// The expected values are the literals from the harness (coordinate-agnostic,
// no Map/iso needed: the world is built with NO passability source and bodies
// are placed at plain world coords, meters since MKS P4). A std::vector<ContactEvent> capture
// listener records the event stream. The narrowphase / broadphase are the same
// modules the Lua harness exercised, so the overlap decisions match.
//
// PRESENTATION-FREE + C++20-clean.

#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/ContactManager.hpp>

using namespace Manifold2D::Physics;
using Catch::Approx;

namespace
{
    constexpr Real kStep = Real(1) / Real(60);

    // Capture listener: records the event type stream + the last full event.
    struct Capture
    {
        std::vector<ContactEvent::Type> types;
        ContactEvent                    last{};
        bool                            any = false;

        void operator()(const ContactEvent& ev)
        {
            types.push_back(ev.type);
            last = ev;
            any  = true;
        }
    };
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
// ContactManager events
// ---------------------------------------------------------------------------

TEST_CASE("ContactManager: begin/stay/end/sensor events fire deterministically",
          "[physics][contacts]")
{
    WorldDef wd;
    PhysicsWorld w(wd);
    Capture cap;
    w.OnContact([&cap](const ContactEvent& ev) { cap(ev); });

    BodyDef ad;
    ad.type     = BodyType::Kinematic;
    ad.position = Vec2(Real(0), Real(0));
    ad.shape    = MakeCircle(Real(0.5));
    BodyHandle a = w.AddBody(ad);

    BodyDef bd;
    bd.type     = BodyType::Kinematic;
    bd.position = Vec2(Real(3), Real(0));
    bd.shape    = MakeCircle(Real(0.5));
    w.AddBody(bd);

    // apart: no events
    w.Step(kStep);
    REQUIRE(cap.types.empty());

    // jump next to b in one step -> begin
    w.SetVelocity(a, Vec2(Real(6 * 25), Real(0))); // 150 m/s, well under the 400 cap
    w.Step(kStep);
    w.SetVelocity(a, Vec2(Real(0), Real(0)));
    REQUIRE_FALSE(cap.types.empty());
    REQUIRE(cap.types.back() == ContactEvent::Type::Begin);

    // next step -> exactly one stay
    const std::size_t before = cap.types.size();
    w.Step(kStep);
    REQUIRE(cap.types.back() == ContactEvent::Type::Stay);
    REQUIRE(cap.types.size() == before + 1);

    // move apart -> end
    w.SetVelocity(a, Vec2(Real(-6 * 25), Real(0)));
    w.Step(kStep);
    w.SetVelocity(a, Vec2(Real(0), Real(0)));
    REQUIRE(cap.types.back() == ContactEvent::Type::End);
}

TEST_CASE("ContactManager: static sensor overlap carries sensor flag + handles",
          "[physics][contacts]")
{
    WorldDef wd;
    PhysicsWorld w(wd);

    BodyDef ad;
    ad.type     = BodyType::Kinematic;
    ad.position = Vec2(Real(0), Real(0));
    ad.shape    = MakeCircle(Real(0.5));
    w.AddBody(ad);

    // Capture the full last event (sensor flag + handles).
    ContactEvent rec{};
    bool got = false;
    w.OnContact([&rec, &got](const ContactEvent& ev) { rec = ev; got = true; });

    BodyDef sd;
    sd.type     = BodyType::Static;
    sd.position = Vec2(Real(0), Real(0));
    sd.shape    = MakeAabb(Real(0.8), Real(0.8));
    sd.isSensor = true;
    w.AddBody(sd);

    w.Step(kStep);
    REQUIRE(got);
    REQUIRE(rec.type == ContactEvent::Type::Begin);
    REQUIRE(rec.sensor); // sensor flag rides the event
    REQUIRE(w.IsValid(rec.a));
    REQUIRE(w.IsValid(rec.b)); // event carries valid handles
}

// ---------------------------------------------------------------------------
// event gating (two-granularity: per-body + world gate)
// ---------------------------------------------------------------------------

TEST_CASE("Event gating: per-body mute drops + re-arm emits fresh begin",
          "[physics][contacts][gating]")
{
    WorldDef wd;
    PhysicsWorld w(wd);
    Capture cap;
    w.OnContact([&cap](const ContactEvent& ev) { cap(ev); });

    BodyDef ad;
    ad.type     = BodyType::Kinematic;
    ad.position = Vec2(Real(0), Real(0));
    ad.shape    = MakeCircle(Real(0.6));
    BodyHandle a = w.AddBody(ad);

    BodyDef cd;
    cd.type     = BodyType::Kinematic;
    cd.position = Vec2(Real(0.5), Real(0)); // overlapping a (dist 0.5 < 1.2 = sumR)
    cd.shape    = MakeCircle(Real(0.6));
    w.AddBody(cd);

    // baseline begin
    w.Step(kStep);
    REQUIRE(cap.types.back() == ContactEvent::Type::Begin);

    // mute one participant: stays stop, NO synthetic end is queued (drop)
    w.SetBodyEvents(a, false);
    cap.types.clear();
    w.Step(kStep);
    REQUIRE(cap.types.empty());

    // re-enable while still overlapping: fresh begin IMMEDIATELY (re-arm)
    w.SetBodyEvents(a, true);
    REQUIRE_FALSE(cap.types.empty());
    REQUIRE(cap.types.back() == ContactEvent::Type::Begin);

    // next step -> resumes staying
    cap.types.clear();
    w.Step(kStep);
    REQUIRE(cap.types.back() == ContactEvent::Type::Stay);
}

TEST_CASE("Event gating: world gate drops everything + re-arms overlapping",
          "[physics][contacts][gating]")
{
    WorldDef wd;
    PhysicsWorld w(wd);
    Capture cap;
    w.OnContact([&cap](const ContactEvent& ev) { cap(ev); });

    BodyDef ad;
    ad.type     = BodyType::Kinematic;
    ad.position = Vec2(Real(0), Real(0));
    ad.shape    = MakeCircle(Real(0.6));
    w.AddBody(ad);

    BodyDef cd;
    cd.type     = BodyType::Kinematic;
    cd.position = Vec2(Real(0.5), Real(0));
    cd.shape    = MakeCircle(Real(0.6));
    w.AddBody(cd);

    // baseline begin
    w.Step(kStep);
    REQUIRE(cap.types.back() == ContactEvent::Type::Begin);

    // world gate off -> 0 events (drop)
    w.SetEventsEnabled(false);
    cap.types.clear();
    w.Step(kStep);
    REQUIRE(cap.types.empty());

    // world gate on -> fresh begin (re-arm overlapping pairs)
    w.SetEventsEnabled(true);
    REQUIRE_FALSE(cap.types.empty());
    REQUIRE(cap.types.back() == ContactEvent::Type::Begin);
}
