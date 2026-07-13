// Physics Phase A: persistent incremental islands -- BEHAVIORAL tests.
// Companion to PhysicsIslandTest.cpp (the original 6-case sleep suite,
// which stays UNCHANGED). These cover the persistent-registry facts:
// create-time 1-body islands, merge on touch, deferred split, and the
// wake-the-whole-island fan-out. Determinism is the contract.
//
// PRESENTATION-FREE + C++23-clean.

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

    BodyHandle AddFloor(PhysicsWorld& w, Vec2 pos, Real hw, Real hh)
    {
        BodyDef def;
        def.type     = BodyType::Static;
        def.position = pos;
        def.shape    = MakeAabb(hw, hh);
        def.friction = Real(0.6);
        return w.AddBody(def);
    }
    BodyHandle AddBox(PhysicsWorld& w, Vec2 pos, Real hw, Real hh)
    {
        BodyDef def;
        def.type          = BodyType::Dynamic;
        def.position      = pos;
        def.shape         = MakeAabb(hw, hh);
        def.density       = Real(1);
        def.friction      = Real(0.4);
        def.fixedRotation = true;
        return w.AddBody(def);
    }
}

// ---------------------------------------------------------------------------
// Create-time: a fresh dynamic body is its own 1-body island; two separated
// dynamics have DISTINCT island roots; a static body never shares a dynamic's
// root (statics are not island members).
// ---------------------------------------------------------------------------
TEST_CASE("PhysicsPersistentIsland: fresh dynamic body is its own island; static has none",
          "[physics][island]")
{
    WorldDef wd;
    wd.gravityX = Real(0); // no gravity -> bodies stay put, never touch
    wd.gravityY = Real(0);
    PhysicsWorld w(wd);

    const BodyHandle floor = AddFloor(w, Vec2(Real(0), Real(50)), Real(5), Real(0.5));
    const BodyHandle b0    = AddBox(w, Vec2(Real(-20), Real(0)), Real(0.5), Real(0.5));
    const BodyHandle b1    = AddBox(w, Vec2(Real(20),  Real(0)), Real(0.5), Real(0.5));

    // Before any Step: each dynamic is its own island (assigned at AddBody).
    const std::uint32_t r0 = w.IslandRootOf(b0.index);
    const std::uint32_t r1 = w.IslandRootOf(b1.index);
    CHECK(r0 != r1); // two far-apart dynamics -> different islands

    // The static body is not a member. IslandRootOf returns a high-bit-tagged
    // slot for a non-member, which can NEVER equal a real (dense, small)
    // island id -> distinct from r0/r1.
    const std::uint32_t rf = w.IslandRootOf(floor.index);
    CHECK(rf != r0);
    CHECK(rf != r1);

    // Stepping with no contacts keeps them distinct (no merge happens).
    for (int k = 0; k < 5; ++k) { w.Step(kStep); }
    CHECK(w.IslandRootOf(b0.index) != w.IslandRootOf(b1.index));
}

// ---------------------------------------------------------------------------
// Merge: dynamic bodies in a touching chain coalesce into ONE island.
// ---------------------------------------------------------------------------
TEST_CASE("PhysicsPersistentIsland: touching chain merges to one island",
          "[physics][island]")
{
    WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
    PhysicsWorld w(wd);

    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(20), Real(0.5));
    const Real hw = Real(0.5), hh = Real(0.5);
    const Real gap = Real(0.05);
    const BodyHandle b0 = AddBox(w, Vec2(Real(0), -hh),                              hw, hh);
    const BodyHandle b1 = AddBox(w, Vec2(Real(0), -hh - (Real(2)*hh + gap)),         hw, hh);
    const BodyHandle b2 = AddBox(w, Vec2(Real(0), -hh - (Real(2)*hh + gap)*Real(2)), hw, hh);
    const BodyHandle far = AddBox(w, Vec2(Real(50), -hh), hw, hh);

    // Settle into contact while still awake (mirrors PhysicsIslandTest case 5).
    for (int k = 0; k < 30; ++k) { w.Step(kStep); }
    REQUIRE(w.IsAwake(b0));
    REQUIRE(w.IsAwake(b1));
    REQUIRE(w.IsAwake(b2));

    const std::uint32_t r0 = w.IslandRootOf(b0.index);
    CHECK(r0 == w.IslandRootOf(b1.index));
    CHECK(r0 == w.IslandRootOf(b2.index));
    CHECK(r0 != w.IslandRootOf(far.index)); // isolated body -> different island
}

// ---------------------------------------------------------------------------
// Split: separating a touching pair fractures the island (deferred); a body
// that loses all contacts becomes its own 1-body island.
// ---------------------------------------------------------------------------
TEST_CASE("PhysicsPersistentIsland: separating a chain splits the island (deferred)",
          "[physics][island]")
{
    WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
    PhysicsWorld w(wd);

    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(20), Real(0.5));
    const Real hw = Real(0.5), hh = Real(0.5);
    const Real gap = Real(0.05);
    const BodyHandle b0 = AddBox(w, Vec2(Real(0), -hh),                      hw, hh);
    const BodyHandle b1 = AddBox(w, Vec2(Real(0), -hh - (Real(2)*hh + gap)), hw, hh);

    for (int k = 0; k < 30; ++k) { w.Step(kStep); }
    REQUIRE(w.IsAwake(b0));
    REQUIRE(w.IsAwake(b1));
    REQUIRE(w.IslandRootOf(b0.index) == w.IslandRootOf(b1.index)); // merged

    // Fling the top box far up + sideways so its contact with b0 separates.
    w.SetVelocity(b1, Vec2(Real(4), Real(-20))); // m/s -- fast separation, far under the 400 cap
    // A few steps for the contact to drop + the deferred split to resolve
    // (quota = 1 per step, but only 1 candidate here, so it resolves quickly).
    for (int k = 0; k < 20; ++k) { w.Step(kStep); }

    CHECK(w.IslandRootOf(b0.index) != w.IslandRootOf(b1.index)); // split
}

// ---------------------------------------------------------------------------
// Wake fan-out: disturbing ONE member of a sleeping island wakes the WHOLE
// island immediately (Box2D-style island wake), not just the disturbed body.
// ---------------------------------------------------------------------------
TEST_CASE("PhysicsPersistentIsland: impulse on one member wakes the whole island",
          "[physics][island]")
{
    WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
    PhysicsWorld w(wd);

    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(20), Real(0.5));
    const Real hw = Real(0.4), hh = Real(0.4);
    const int N = 3;
    std::vector<BodyHandle> boxes;
    for (int i = 0; i < N; ++i)
    {
        const Real y = -(Real(2) * hh + Real(0.01)) * static_cast<Real>(i + 1);
        boxes.push_back(AddBox(w, Vec2(Real(0), y), hw, hh));
    }
    for (int k = 0; k < 700; ++k) { w.Step(kStep); }
    for (int i = 0; i < N; ++i) { REQUIRE_FALSE(w.IsAwake(boxes[i])); }

    // Impulse the MIDDLE box -> the whole island wakes at once. Authored as
    // mass x delta-v (rule 3): box mass = density * 4*hw*hh = 1*4*0.4*0.4 =
    // 0.64 kg; targetDv = -20 m/s (well above sleepThreshold 0.05, well under
    // the 400 m/s cap) is clearly enough to force a wake.
    const Real boxMass = Real(1) * Real(4) * hw * hh;
    const Real targetDv = Real(-20);
    w.ApplyImpulse(boxes[1], Vec2(Real(0), boxMass * targetDv));
    for (int i = 0; i < N; ++i)
    {
        CHECK(w.IsAwake(boxes[i])); // every member awake (island wake)
    }
}

// ---------------------------------------------------------------------------
// Determinism: a scene exercising merge + split + sleep is bit-identical
// across two runs (positions, awake state, island roots).
// ---------------------------------------------------------------------------
TEST_CASE("PhysicsPersistentIsland: merge+split+sleep is deterministic across two runs",
          "[physics][island]")
{
    auto run = [](std::vector<Vec2>& pos, std::vector<int>& awake,
                  std::vector<std::uint32_t>& roots)
    {
        WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
        PhysicsWorld w(wd);

        AddFloor(w, Vec2(Real(0), Real(0.5)), Real(20), Real(0.5));
        const Real hw = Real(0.4), hh = Real(0.4);
        std::vector<BodyHandle> boxes;
        for (int i = 0; i < 4; ++i)
        {
            const Real y = -(Real(2) * hh + Real(0.01)) * static_cast<Real>(i + 1);
            boxes.push_back(AddBox(w, Vec2(Real(0), y), hw, hh));
        }
        for (int k = 0; k < 200; ++k) { w.Step(kStep); }
        // Disturb the top -> a split candidate forms, then re-settles. Authored
        // as mass x delta-v (rule 3): box mass = 1*4*0.4*0.4 = 0.64 kg;
        // targetDv = (2, -25) m/s -- an up+sideways kick well under the 400 cap.
        const Real boxMass = Real(1) * Real(4) * hw * hh;
        const Vec2 targetDv = Vec2(Real(2), Real(-25));
        w.ApplyImpulse(boxes[3], boxMass * targetDv);
        for (int k = 0; k < 500; ++k) { w.Step(kStep); }

        pos.clear(); awake.clear(); roots.clear();
        for (const BodyHandle b : boxes)
        {
            pos.push_back(w.Position(b));
            awake.push_back(w.IsAwake(b) ? 1 : 0);
            roots.push_back(w.IslandRootOf(b.index));
        }
    };

    std::vector<Vec2> p1, p2; std::vector<int> a1, a2;
    std::vector<std::uint32_t> r1, r2;
    run(p1, a1, r1);
    run(p2, a2, r2);

    REQUIRE(p1.size() == p2.size());
    for (std::size_t i = 0; i < p1.size(); ++i)
    {
        REQUIRE(p1[i].x == p2[i].x);
        REQUIRE(p1[i].y == p2[i].y);
        REQUIRE(a1[i] == a2[i]);
        REQUIRE(r1[i] == r2[i]); // island ids reproduce run-to-run
    }
}

// Multi-component fracture: a connected chain that loses TWO internal links
// must split into THREE distinct islands, each internally connected. Guards
// the SplitIsland rewrite (per-body-adjacency walk) against the whole-pool scan
// it replaces -- byte-identical connected-component grouping.
TEST_CASE("PhysicsPersistentIsland: chain fractures into three islands",
          "[physics][island]")
{
    WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
    PhysicsWorld w(wd);

    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(40), Real(0.5));

    // Six boxes in a gravity-pressed vertical stack settle into ONE island, then
    // fracture into THREE components. Index assignment puts the must-stay-together
    // pair {0,1} at the BOTTOM (gravity presses them -> robustly touching through
    // the fling) and the two flung boxes {2,4} at the TOP, where nothing obstructs
    // them -- mirroring the proven "separate a chain by flinging the top box"
    // pattern. Stack levels (bottom->top): b0, b1, b3, b5, b2, b4.
    const Real hw = Real(0.5), hh = Real(0.5);
    const Real gap = Real(0.05);
    const Real pitch = Real(2) * hh + gap; // per-level rise; boxes drop `gap` to touch
    auto atLevel = [&](int level) { return Vec2(Real(0), -hh - pitch * Real(level)); };
    std::vector<BodyHandle> b(6);
    b[0] = AddBox(w, atLevel(0), hw, hh); // bottom of the {0,1} pair
    b[1] = AddBox(w, atLevel(1), hw, hh); // on b0 (gravity-pressed)
    b[3] = AddBox(w, atLevel(2), hw, hh);
    b[5] = AddBox(w, atLevel(3), hw, hh);
    b[2] = AddBox(w, atLevel(4), hw, hh); // near the top -> flung clear
    b[4] = AddBox(w, atLevel(5), hw, hh); // top -> nothing above it
    for (int k = 0; k < 80; ++k) { w.Step(kStep); }
    // All in one island once settled + merged (the whole stack is connected).
    for (int i = 1; i < 6; ++i)
    {
        REQUIRE(w.IslandRootOf(b[0].index) == w.IslandRootOf(b[i].index));
    }

    // Fling the two top boxes up + in OPPOSITE directions so they leave the stack
    // cleanly: the b5-2 and 2-4 links break, leaving the bottom pile {0,1,3,5}
    // plus the two flung boxes -- MULTIPLE disconnected components, forcing
    // repeated AllocIsland in split.
    w.SetVelocity(b[2], Vec2(Real(6), Real(-25)));
    w.SetVelocity(b[4], Vec2(Real(-6), Real(-25)));
    for (int k = 0; k < 60; ++k) { w.Step(kStep); } // quota=1/step -> let splits resolve

    // {0,1} stay together (gravity-pressed bottom of the pile) and are distinct
    // from the flung boxes. Distinct components must have distinct island roots;
    // same component shares a root.
    CHECK(w.IslandRootOf(b[0].index) == w.IslandRootOf(b[1].index));
    CHECK(w.IslandRootOf(b[0].index) != w.IslandRootOf(b[2].index));
    CHECK(w.IslandRootOf(b[2].index) != w.IslandRootOf(b[4].index));
    CHECK(w.IslandRootOf(b[0].index) != w.IslandRootOf(b[4].index));
}
