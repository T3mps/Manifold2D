// Physics correctness batch -- Task A: sensor contacts must not couple islands.
//
// A dyn-dyn contact whose ONLY fixture pair is a sensor must:
//   (a) NOT merge the two bodies into the same island (sensor is not an edge), and
//   (b) still produce a pooled contact (event-relevant, DebugHasContact == true).
//
// A non-sensor dyn-dyn contact DOES merge islands (regression guard confirming
// the gate on solverRelevant did not over-fire).
//
// PRESENTATION-FREE + C++23-clean. ASCII comments only.

#include <cstdint>

#include <catch2/catch_test_macros.hpp>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>

using namespace Manifold2D::Physics;

namespace
{
    constexpr Real kStep = Real(1) / Real(60);

    BodyHandle AddSensorBox(PhysicsWorld& w, Vec2 pos, Real hw, Real hh)
    {
        BodyDef def;
        def.type          = BodyType::Dynamic;
        def.position      = pos;
        def.shape         = MakeAabb(hw, hh);
        def.isSensor      = true;
        def.density       = Real(1);
        def.fixedRotation = true;
        return w.AddBody(def);
    }

    BodyHandle AddSolidBox(PhysicsWorld& w, Vec2 pos, Real hw, Real hh)
    {
        BodyDef def;
        def.type          = BodyType::Dynamic;
        def.position      = pos;
        def.shape         = MakeAabb(hw, hh);
        def.isSensor      = false;
        def.density       = Real(1);
        def.friction      = Real(0.4);
        def.fixedRotation = true;
        return w.AddBody(def);
    }

    BodyHandle AddFloor(PhysicsWorld& w, Vec2 pos, Real hw, Real hh)
    {
        BodyDef def;
        def.type     = BodyType::Static;
        def.position = pos;
        def.shape    = MakeAabb(hw, hh);
        def.friction = Real(0.6);
        return w.AddBody(def);
    }
}

// ---------------------------------------------------------------------------
// Task A Step 1+2: a dyn-dyn SENSOR pair must NOT merge islands.
// The sensor contact IS pooled (event-relevant) but the bodies remain in
// separate islands.  IslandRootOf must differ before AND after stepping.
// ---------------------------------------------------------------------------
TEST_CASE("PhysicsSensorIsland: sensor dyn-dyn pair does not merge islands",
          "[physics][island]")
{
    // Zero gravity: both bodies start overlapping, no forces, no solver impulse
    // (sensor -> solverRelevant=false -> no collision response).
    WorldDef wd;
    wd.gravityY = Real(0);
    wd.gravityX = Real(0);
    // sleepThreshold, restitutionThreshold, contactPushMaxVelocity, and
    // hashCellSize all inherit the MKS WorldDef defaults (0.05 / 1.0 / 3.0 / 1.0).
    PhysicsWorld w(wd);

    // Place bA (sensor) and bB (solid) overlapping so the broadphase detects them.
    // Half-width 0.5 each, centres 0.3 apart -> 0.7 m overlap. Re-authored /10
    // (not /100): this is the cluster's most slop-marginal geometry -- a /100
    // centre offset of 0.03 m would sit only 1.5x above kSkin (0.02), inside
    // speculative-margin noise. /10 keeps every quantity 15x+ above kSkin while
    // preserving the "large overlap relative to body size" shape under test.
    const BodyHandle bA = AddSensorBox(w, Vec2(Real(0),   Real(0)), Real(0.5), Real(0.5));
    const BodyHandle bB = AddSolidBox (w, Vec2(Real(0.3), Real(0)), Real(0.5), Real(0.5));

    // Before any Step: distinct islands (one-body each, assigned at AddBody).
    REQUIRE(w.IslandRootOf(bA.index) != w.IslandRootOf(bB.index));

    // Step enough for the broadphase to fire and the contact to be created.
    for (int k = 0; k < 5; ++k) { w.Step(kStep); }

    // The sensor pair produces a pooled contact (event path is live).
    CHECK(w.DebugHasContact(bA, bB));

    // Sensor contact must NOT have merged the islands.
    CHECK(w.IslandRootOf(bA.index) != w.IslandRootOf(bB.index));
}

// ---------------------------------------------------------------------------
// Regression: a NON-sensor dyn-dyn contact DOES merge islands (gate must not
// over-fire).  Mirrors the "touching chain merges to one island" test pattern
// from PhysicsPersistentIslandTest but as a minimal 2-body scene here.
// ---------------------------------------------------------------------------
TEST_CASE("PhysicsSensorIsland: non-sensor dyn-dyn contact merges islands (regression)",
          "[physics][island]")
{
    WorldDef wd;
    wd.gravityY = Real(10);
    // gravityX, sleepThreshold, restitutionThreshold, contactPushMaxVelocity, and
    // hashCellSize all inherit the MKS WorldDef defaults (0 / 0.05 / 1.0 / 3.0 / 1.0).
    PhysicsWorld w(wd);

    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(20), Real(0.5));

    // Two solid boxes stacked: they will settle into contact and merge.
    const Real hw = Real(0.5), hh = Real(0.5);
    const Real gap = Real(0.02); // = kSkin (mirrors MKS P3 Task 4 case 5)
    const BodyHandle bA = AddSolidBox(w, Vec2(Real(0), -hh),                      hw, hh);
    const BodyHandle bB = AddSolidBox(w, Vec2(Real(0), -hh - (Real(2)*hh + gap)), hw, hh);

    REQUIRE(w.IslandRootOf(bA.index) != w.IslandRootOf(bB.index));

    // Settle into contact while still awake.
    for (int k = 0; k < 30; ++k) { w.Step(kStep); }
    REQUIRE(w.IsAwake(bA));
    REQUIRE(w.IsAwake(bB));

    // Non-sensor dyn-dyn contact must have merged the two bodies into ONE island.
    CHECK(w.IslandRootOf(bA.index) == w.IslandRootOf(bB.index));
}
