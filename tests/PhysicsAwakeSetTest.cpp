// Physics Phase B: awake-set + sleep-by-migration -- BEHAVIORAL tests.
// Companion to PhysicsIslandTest.cpp / PhysicsPersistentIslandTest.cpp.
// PRESENTATION-FREE + C++23-clean.
#include <cstdint>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/StepProf.hpp>
using namespace Manifold2D::Physics;
namespace {
    constexpr Real kStep = Real(1) / Real(60);
    BodyHandle AddFloor(PhysicsWorld& w, Vec2 pos, Real hw, Real hh) {
        BodyDef d; d.type=BodyType::Static; d.position=pos; d.shape=MakeAabb(hw,hh); d.friction=Real(0.6); return w.AddBody(d);
    }
    BodyHandle AddBox(PhysicsWorld& w, Vec2 pos, Real hw, Real hh) {
        BodyDef d; d.type=BodyType::Dynamic; d.position=pos; d.shape=MakeAabb(hw,hh); d.density=Real(1); d.friction=Real(0.4); d.fixedRotation=true; return w.AddBody(d);
    }
}
// StepProf is a no-op by default: a Step still runs and the scoped timers compile away.
TEST_CASE("PhysicsAwakeSet: StepProf is a no-op when ARCANE_STEPPROF is off", "[physics][awakeset]")
{
    WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
    PhysicsWorld w(wd);
    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(20), Real(0.5));
    AddBox(w, Vec2(Real(0), Real(-2)), Real(0.5), Real(0.5));
    for (int k = 0; k < 10; ++k) { w.Step(kStep); }
    REQUIRE(StepProf::Enabled() == false);
}
// Rerouting the solver loops to the awake-set must not change the result:
// a settle scene is run-twice-identical AND ends fully asleep + frozen.
TEST_CASE("PhysicsAwakeSet: awake-only solve is deterministic + settles identically", "[physics][awakeset]")
{
    auto run = [](std::vector<Vec2>& pos, std::vector<int>& awake) {
        WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
        PhysicsWorld w(wd);
        AddFloor(w, Vec2(Real(0), Real(0.5)), Real(20), Real(0.5));
        std::vector<BodyHandle> boxes;
        for (int i = 0; i < 5; ++i) { boxes.push_back(AddBox(w, Vec2(Real(0), Real(-1.0) - Real(0.9)*static_cast<Real>(i)), Real(0.4), Real(0.4))); }
        for (int k = 0; k < 900; ++k) { w.Step(kStep); }
        pos.clear(); awake.clear();
        for (const BodyHandle b : boxes) { pos.push_back(w.Position(b)); awake.push_back(w.IsAwake(b)?1:0); }
    };
    std::vector<Vec2> p1, p2; std::vector<int> a1, a2;
    run(p1, a1); run(p2, a2);
    REQUIRE(p1.size() == p2.size());
    for (std::size_t i = 0; i < p1.size(); ++i) { REQUIRE(p1[i].x == p2[i].x); REQUIRE(p1[i].y == p2[i].y); REQUIRE(a1[i] == a2[i]); }
}
// The awake-set must, at all times, contain EXACTLY the awake dynamic slots.
TEST_CASE("PhysicsAwakeSet: set membership tracks awake-dynamic slots", "[physics][awakeset]")
{
    auto checkInvariant = [](PhysicsWorld& w) {
        const std::vector<std::uint32_t>& set = w.AwakeBodies();
        std::vector<std::uint8_t> seen(w.Count(), 0u);
        for (const std::uint32_t s : set) {
            REQUIRE(s < w.Count());
            REQUIRE(w.Alive(s));
            REQUIRE(w.TypeSlot(s) == BodyType::Dynamic);
            REQUIRE(w.AwakeSlot(s));
            REQUIRE(seen[s] == 0u); // no duplicates
            seen[s] = 1u;
        }
        for (std::uint32_t i = 0; i < w.Count(); ++i) {
            const bool awakeDyn = w.Alive(i) && w.TypeSlot(i) == BodyType::Dynamic && w.AwakeSlot(i);
            REQUIRE((seen[i] != 0u) == awakeDyn);
        }
    };

    WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
    PhysicsWorld w(wd);
    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(20), Real(0.5));
    const BodyHandle b0 = AddBox(w, Vec2(Real(0), Real(-2)), Real(0.5), Real(0.5));
    const BodyHandle b1 = AddBox(w, Vec2(Real(0), Real(-4)), Real(0.5), Real(0.5));
    checkInvariant(w);                                  // 2 awake dynamics
    for (int k = 0; k < 700; ++k) { w.Step(kStep); }
    checkInvariant(w);                                  // settled -> asleep -> not in set
    // Authored as mass x delta-v (rule 3): box mass = density * 4*hw*hh
    const Real boxMass = Real(1) * Real(4) * Real(0.5) * Real(0.5); // = 1.0 kg
    const Real targetDv = Real(-20); // m/s: well above sleepThreshold 0.05, well under the 400 cap
    w.ApplyImpulse(b1, Vec2(Real(0), boxMass * targetDv));          // wake fan-out
    checkInvariant(w);
    w.RemoveBody(b0);                                   // out of the set
    checkInvariant(w);
    const BodyHandle b2 = AddBox(w, Vec2(Real(0), Real(-6)), Real(0.5), Real(0.5)); // recycle a slot
    (void)b2;
    checkInvariant(w);
}
// A sleeping body must keep prev==pos at all times (render-lerp stays frozen).
// WHY: once Stage 1 only snaps awake dynamics, a sleeping body is never snapped
// each step -- prev-on-sleep (SnapPrevToPos at the sleep seam) is the only
// mechanism that guarantees prev==pos for a body that has settled and frozen.
TEST_CASE("PhysicsAwakeSet: sleeping body render-lerp is frozen (prev==pos)", "[physics][awakeset]")
{
    WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
    PhysicsWorld w(wd);
    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(20), Real(0.5));
    const BodyHandle b = AddBox(w, Vec2(Real(0), Real(-2)), Real(0.5), Real(0.5));
    for (int k = 0; k < 700; ++k) { w.Step(kStep); }
    REQUIRE_FALSE(w.IsAwake(b));
    const Vec2 mid = w.DrawPosition(b, Real(0.5));
    const Vec2 pos = w.Position(b);
    REQUIRE(mid.x == pos.x); REQUIRE(mid.y == pos.y); // prev==pos -> no lerp drift
    for (int k = 0; k < 60; ++k) { w.Step(kStep); }
    const Vec2 mid2 = w.DrawPosition(b, Real(0.5));
    REQUIRE(mid2.x == pos.x); REQUIRE(mid2.y == pos.y); // still frozen
}
// A dense settled pile must fully sleep (the awake-set drains). This is the
// Phase B sleep WIN: a connected pile that quiesces migrates out of the awake
// set. If it FAILS, the soft solver leaves residual jitter above the sleep
// threshold -> the controller decides the minimal quiescence fix.
TEST_CASE("PhysicsAwakeSet: a dense settled pile fully sleeps", "[physics][awakeset]")
{
    WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
    PhysicsWorld w(wd);
    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(40), Real(0.5));
    std::vector<BodyHandle> boxes;
    const Real hw = Real(0.4), hh = Real(0.4);
    // Seating gap re-authored from kSkin (0.02 m = 4x kLinearSlop 0.005), a
    // chosen clearance -- NOT a /10 division of the old 0.2 px gap, even
    // though the two numerically coincide here.
    const Real gap = Real(0.02);
    for (int row = 0; row < 6; ++row) {
        const int n = 10 - row;
        for (int c = 0; c < n; ++c) {
            const Real x = (static_cast<Real>(c) - static_cast<Real>(n)*Real(0.5)) * (Real(2)*hw + gap);
            const Real y = Real(-0.5) - static_cast<Real>(row) * (Real(2)*hh + gap);
            boxes.push_back(AddBox(w, Vec2(x, y), hw, hh));
        }
    }
    for (int k = 0; k < 300; ++k) { w.Step(kStep); }
    INFO("awake bodies remaining: " << w.AwakeBodies().size() << " / " << boxes.size());
    REQUIRE(w.AwakeBodies().empty());
}
// Guard: a body in clear motion must NEVER sleep (protects the sleep threshold
// from being loosened to the point a visibly-moving body would freeze). With
// zero gravity it coasts at constant velocity -- 5.0 m/s is 100x the default
// 0.05 m/s sleepThreshold -- for the whole window.
TEST_CASE("PhysicsAwakeSet: a body in clear motion never sleeps", "[physics][awakeset]")
{
    WorldDef wd;
    wd.gravityX = Real(0); // zero-g: coasts at constant velocity, never settles
    wd.gravityY = Real(0);
    PhysicsWorld w(wd);
    const BodyHandle b = AddBox(w, Vec2(Real(0), Real(0)), Real(0.5), Real(0.5));
    w.SetVelocity(b, Vec2(Real(5.0), Real(0)));   // |v|=5.0 m/s, far above threshold
    for (int k = 0; k < 120; ++k) { w.Step(kStep); REQUIRE(w.IsAwake(b)); }
}
// A stationary (zero-velocity) kinematic must not prevent the scene from
// sleeping, and a settled dynamic still sleeps with such a kinematic present.
// (Regression guard for the Stage-1 zero-velocity kinematic proxy gate.)
TEST_CASE("PhysicsAwakeSet: a stationary kinematic does not prevent the scene from sleeping", "[physics][awakeset]")
{
    WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
    PhysicsWorld w(wd);
    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(20), Real(0.5));
    BodyDef kd; kd.type = BodyType::Kinematic; kd.position = Vec2(Real(15), Real(-5)); kd.shape = MakeAabb(Real(0.5), Real(0.5));
    const BodyHandle k = w.AddBody(kd);          // zero velocity, off to the side
    const BodyHandle b = AddBox(w, Vec2(Real(0), Real(-2)), Real(0.5), Real(0.5));
    for (int n = 0; n < 700; ++n) { w.Step(kStep); }
    REQUIRE_FALSE(w.IsAwake(b));                  // dynamic still sleeps
    (void)k;
}
// Determinism: a scene that exercises create + sleep + wake + remove + slot
// RECYCLE (so the awake-set goes through append + swap-remove cycles) is
// bit-identical across two runs (positions + awake state). The awake-set's
// non-ascending (append/swap-remove) order must not perturb determinism.
TEST_CASE("PhysicsAwakeSet: create/sleep/wake/remove is deterministic across two runs", "[physics][awakeset]")
{
    auto run = [](std::vector<Vec2>& pos, std::vector<int>& awake) {
        WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
        PhysicsWorld w(wd);
        AddFloor(w, Vec2(Real(0), Real(0.5)), Real(20), Real(0.5));
        std::vector<BodyHandle> boxes;
        for (int i = 0; i < 6; ++i) { boxes.push_back(AddBox(w, Vec2(Real(0), Real(-1.0) - Real(0.9)*static_cast<Real>(i)), Real(0.4), Real(0.4))); }
        for (int k = 0; k < 200; ++k) { w.Step(kStep); }
        w.RemoveBody(boxes[2]);                                          // swap-remove from the awake-set mid-life
        const BodyHandle nb = AddBox(w, Vec2(Real(3), Real(-1)), Real(0.4), Real(0.4)); // recycle a slot
        // Authored as mass x delta-v (rule 3): box mass = density * 4*hw*hh
        const Real boxMass = Real(1) * Real(4) * Real(0.4) * Real(0.4); // = 0.64 kg
        const Vec2 targetDv = Vec2(Real(2), Real(-25)); // m/s: up+sideways kick, well under the 400 cap
        w.ApplyImpulse(boxes[5], boxMass * targetDv);          // wake fan-out
        for (int k = 0; k < 500; ++k) { w.Step(kStep); }
        pos.clear(); awake.clear();
        for (std::size_t i = 0; i < boxes.size(); ++i) { if (i==2) continue; pos.push_back(w.Position(boxes[i])); awake.push_back(w.IsAwake(boxes[i])?1:0); }
        pos.push_back(w.Position(nb)); awake.push_back(w.IsAwake(nb)?1:0);
    };
    std::vector<Vec2> p1, p2; std::vector<int> a1, a2;
    run(p1, a1); run(p2, a2);
    REQUIRE(p1.size() == p2.size());
    for (std::size_t i = 0; i < p1.size(); ++i) { REQUIRE(p1[i].x == p2[i].x); REQUIRE(p1[i].y == p2[i].y); REQUIRE(a1[i] == a2[i]); }
}

// REGRESSION (scene-0 "Playground"): a mixed AABB+circle pile must FULLY settle
// and sleep. Bug: WakeMoverPair woke a settled body every sleep cycle from a
// STATIONARY neighbour across a sub-pixel gap (the bodies were in different
// islands, not touching) -> the pile flashed rest/wake forever. Fix: a body
// idle enough to be a sleep candidate must not wake its sleeping neighbours.
// Values mirror the MKS-converted Scenes.cpp BuildPlayground exactly (MKS P6)
// -- this test is that scene's regression twin.
TEST_CASE("PhysicsAwakeSet: scene-0 mixed AABB+circle pile fully settles (no rest/wake flashing)",
          "[physics][awakeset]")
{
    WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
    PhysicsWorld w(wd);
    // Floor + 2 walls (static).
    auto staticBox = [&](Real cx, Real cy, Real hw, Real hh){
        BodyDef d; d.type=BodyType::Static; d.position=Vec2(cx,cy); d.shape=MakeAabb(hw,hh); d.friction=Real(0.4); return w.AddBody(d);
    };
    staticBox(Real(6.4),  Real(8.2), Real(7.6),  Real(0.36));
    staticBox(Real(-0.8), Real(5.6), Real(0.36), Real(3.0));
    staticBox(Real(13.6), Real(5.6), Real(0.36), Real(3.0));
    // 3 dynamic boxes (fixedRotation) + 2 dynamic circles.
    auto dynBox = [&](Real cx, Real cy, Real hw, Real hh){
        BodyDef d; d.type=BodyType::Dynamic; d.position=Vec2(cx,cy); d.shape=MakeAabb(hw,hh);
        d.density=Real(1); d.friction=Real(0.4); d.restitution=Real(0.1); d.fixedRotation=true; return w.AddBody(d);
    };
    auto dynCircle = [&](Real cx, Real cy, Real r){
        BodyDef d; d.type=BodyType::Dynamic; d.position=Vec2(cx,cy); d.shape=MakeCircle(r);
        d.density=Real(1); d.friction=Real(0.4); d.restitution=Real(0.2); return w.AddBody(d);
    };
    std::vector<BodyHandle> dyn;
    dyn.push_back(dynBox(Real(4.4), Real(1.2), Real(0.54), Real(0.54)));
    dyn.push_back(dynBox(Real(6.4), Real(0.4), Real(0.66), Real(0.42)));
    dyn.push_back(dynBox(Real(8.6), Real(0.9), Real(0.46), Real(0.46)));
    dyn.push_back(dynCircle(Real(5.4), Real(2.6), Real(0.50)));
    dyn.push_back(dynCircle(Real(7.6), Real(2.0), Real(0.58)));

    // 10 seconds is ample for a 5-body pile to settle + sleep.
    for (int k = 0; k < 600; ++k) { w.Step(kStep); }
    int awake = 0; for (const BodyHandle b : dyn) { if (w.IsAwake(b)) ++awake; }
    INFO("awake dynamic bodies after settle: " << awake << " / " << dyn.size());
    REQUIRE(w.AwakeBodies().empty());   // the whole pile drained the awake-set
}
