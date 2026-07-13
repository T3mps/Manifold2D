// Physics M6 P2.4: the island constraint-graph + sleep module -- BEHAVIORAL tests.
//
// PORT NOTE: P2.4 extracted the inline Lua island/sleep logic
// (Client/src/physics/PhysicsWorld.lua:28-31 + 403-452) into Physics/Island.cpp
// (folded into IslandManager by the 2026-07-09 decomp) and wired it at the
// Step island-sleep seam (after the solve, using this step's
// contacts). There is no Island.lua; the oracle is the harness "islands +
// sleeping (M3)" block (Client/src/tests/physics_harness/main.lua:756-777),
// reformulated Cartesian here (+Y is DOWN, gravity is +Y) plus the plan's
// island-as-a-unit invariants.
//
// INVARIANTS (behavioral, not bit-matched):
//   * a resting dynamic body sleeps after enough idle steps (sleepT > 0.5s);
//   * once asleep its position is FROZEN (no integrate, no solve, no drift);
//   * ApplyImpulse on a sleeper wakes it + it moves on the next step;
//   * an awake mover moving into a sleeper wakes it (wake-on-contact);
//   * an island settles + sleeps AS A UNIT (no half-sleep); disturbing one
//     member wakes it (and, via the contact graph, its neighbors on contact);
//   * determinism: run-twice identical.
//
// PRESENTATION-FREE + C++20-clean.

#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Island.hpp>

using namespace Manifold2D::Physics;
using Catch::Approx;

namespace
{
    constexpr Real kStep = Real(1) / Real(60);

    // A static floor (AABB) centered at `pos`; its TOP surface is pos.y - hh.
    BodyHandle AddFloor(PhysicsWorld& w, Vec2 pos, Real hw, Real hh)
    {
        BodyDef def;
        def.type     = BodyType::Static;
        def.position = pos;
        def.shape    = MakeAabb(hw, hh);
        def.friction = Real(0.6);
        return w.AddBody(def);
    }

    BodyHandle AddCircle(PhysicsWorld& w, Vec2 pos, Real r = Real(1.0),
                         Real friction = Real(0.4))
    {
        BodyDef def;
        def.type     = BodyType::Dynamic;
        def.position = pos;
        def.shape    = MakeCircle(r);
        def.density  = Real(1);
        def.friction = friction;
        return w.AddBody(def);
    }

    BodyHandle AddBox(PhysicsWorld& w, Vec2 pos, Real hw, Real hh, Real friction = Real(0.4))
    {
        BodyDef def;
        def.type          = BodyType::Dynamic;
        def.position      = pos;
        def.shape         = MakeAabb(hw, hh);
        def.density       = Real(1);
        def.friction      = friction;
        // fixedRotation: boxes stay axis-aligned (stack stability), isolates the
        // island-sleep behavior under test.
        def.fixedRotation = true;
        return w.AddBody(def);
    }
}

// ---------------------------------------------------------------------------
// A resting body sleeps; once asleep it does not integrate (position frozen).
// (Direct Cartesian port of harness main.lua:756-777.)
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsIsland: resting body sleeps then stays frozen", "[physics][island]")
{
    WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
    PhysicsWorld w(wd);

    // Floor top at y = 0.
    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(20), Real(0.5));
    BodyHandle ball = AddCircle(w, Vec2(Real(0), Real(-5)));

    // The ball drops, settles, and the island pass sleeps it once it has been
    // idle for > 0.5s. 400 steps is plenty (matches the harness loop count).
    REQUIRE(w.IsAwake(ball));
    for (int k = 0; k < 400; ++k)
    {
        w.Step(kStep);
    }
    REQUIRE_FALSE(w.IsAwake(ball)); // resting ball sleeps

    // FROZEN: position does not change across further steps (no integrate/solve).
    const Vec2 p0 = w.Position(ball);
    for (int k = 0; k < 120; ++k)
    {
        w.Step(kStep);
    }
    const Vec2 p1 = w.Position(ball);
    REQUIRE(p1.x == p0.x); // exact -- a sleeping body is not integrated at all
    REQUIRE(p1.y == p0.y);
    REQUIRE_FALSE(w.IsAwake(ball));
}

// ---------------------------------------------------------------------------
// ApplyImpulse wakes a sleeping body; it moves again on the next step.
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsIsland: impulse wakes a sleeping body", "[physics][island]")
{
    WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
    PhysicsWorld w(wd);

    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(20), Real(0.5));
    BodyHandle ball = AddCircle(w, Vec2(Real(0), Real(-5)));

    for (int k = 0; k < 400; ++k)
    {
        w.Step(kStep);
    }
    REQUIRE_FALSE(w.IsAwake(ball));

    const Real y0 = w.Position(ball).y;

    // Kick the ball upward (-Y). ApplyImpulse must wake it (P2.1 wake-on-force).
    // Authored as mass x delta-v (rule 3): circle mass = density*kPi*r*r =
    // 1*kPi*1.0*1.0 (density 1, r = 1.0 m); targetDv = -20 m/s (well above
    // sleepThreshold 0.05, well under the 400 m/s cap).
    const Real ballMass = Real(1) * kPi * Real(1.0) * Real(1.0);
    const Real targetDv = Real(-20);
    w.ApplyImpulse(ball, Vec2(Real(0), ballMass * targetDv));
    REQUIRE(w.IsAwake(ball)); // impulse wakes the body

    w.Step(kStep);
    REQUIRE(w.Position(ball).y < y0); // woken body moves again (upward = smaller y)
}

// ---------------------------------------------------------------------------
// A new contact wakes a sleeper: an awake mover driven into it wakes it.
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsIsland: new contact wakes a sleeping body", "[physics][island]")
{
    WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
    PhysicsWorld w(wd);

    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(40), Real(0.5));

    // A dynamic ball that will settle + sleep on the floor.
    const Real r = Real(1.0);
    BodyHandle sleeper = AddCircle(w, Vec2(Real(0), Real(-5)));

    // A kinematic mover (statics/kinematics never sleep) parked off to the side.
    BodyDef mover;
    mover.type     = BodyType::Kinematic;
    mover.position = Vec2(Real(-8), Real(-1));
    mover.shape    = MakeCircle(r);
    BodyHandle pusher = w.AddBody(mover);

    for (int k = 0; k < 400; ++k)
    {
        w.Step(kStep);
    }
    REQUIRE_FALSE(w.IsAwake(sleeper)); // settled + asleep

    // Drive the pusher rightward into the sleeping ball. The wake-on-contact
    // path in GenerateContacts must wake the sleeper once they AABB-overlap.
    // 6 m gap at 12 m/s = 0.5 s = 30 steps, inside the 120-step wake window.
    w.SetVelocity(pusher, Vec2(Real(12), Real(0)));
    bool woke = false;
    for (int k = 0; k < 120; ++k)
    {
        w.Step(kStep);
        if (w.IsAwake(sleeper))
        {
            woke = true;
            break;
        }
    }
    REQUIRE(woke); // an awake mover moving into the sleeper woke it
}

// ---------------------------------------------------------------------------
// An island settles + sleeps AS A UNIT; disturbing one member re-wakes it.
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsIsland: stack sleeps as a unit, disturbance wakes a member", "[physics][island]")
{
    WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
    PhysicsWorld w(wd);

    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(20), Real(0.5));

    // A small stack of boxes -> one island (each box contacts the one below).
    const Real hw = Real(0.4), hh = Real(0.4);
    const int N = 3;
    std::vector<BodyHandle> boxes;
    for (int i = 0; i < N; ++i)
    {
        // gap = 2*kLinearSlop (0.005) -- a small seating clearance, not a scale division.
        const Real y = -(Real(2) * hh + Real(0.01)) * static_cast<Real>(i + 1);
        boxes.push_back(AddBox(w, Vec2(Real(0), y), hw, hh));
    }

    for (int k = 0; k < 700; ++k)
    {
        w.Step(kStep);
    }

    // The island slept AS A UNIT -- every member is asleep (no half-sleep).
    for (int i = 0; i < N; ++i)
    {
        REQUIRE_FALSE(w.IsAwake(boxes[i]));
    }

    // Positions frozen while the whole island sleeps.
    std::vector<Vec2> frozen;
    for (int i = 0; i < N; ++i)
    {
        frozen.push_back(w.Position(boxes[i]));
    }
    for (int k = 0; k < 60; ++k)
    {
        w.Step(kStep);
    }
    for (int i = 0; i < N; ++i)
    {
        REQUIRE(w.Position(boxes[i]).x == frozen[i].x);
        REQUIRE(w.Position(boxes[i]).y == frozen[i].y);
    }

    // Disturb the TOP box. With persistent islands the disturbance wakes the
    // WHOLE island AT ONCE (Box2D-style island wake) -- every member, not just
    // the disturbed body, is awake IMMEDIATELY. (The old global-UF path woke
    // only the disturbed body and relied on next-step contact-graph propagation
    // to wake neighbors a few steps later; the persistent-island wake is
    // immediate and strictly stronger. After this the top box separates and the
    // undisturbed neighbors correctly re-settle + re-sleep as a sub-island --
    // so the wake is asserted at the moment of disturbance, not after a delay.)
    // Authored as mass x delta-v (rule 3): box mass = density*4*hw*hh =
    // 1*4*0.4*0.4 = 0.64 kg; targetDv = -20 m/s (well above sleepThreshold
    // 0.05, well under the 400 m/s cap) is clearly enough to force a wake.
    const Real boxMass = Real(1) * Real(4) * hw * hh;
    const Real targetDv = Real(-20);
    w.ApplyImpulse(boxes[N - 1], Vec2(Real(0), boxMass * targetDv));
    REQUIRE(w.IsAwake(boxes[N - 1]));
    for (int i = 0; i < N - 1; ++i)
    {
        REQUIRE(w.IsAwake(boxes[i])); // island wake: neighbors wake immediately
    }
}

// ---------------------------------------------------------------------------
// IslandRootOf: four stacked boxes in mutual contact (all awake) form ONE
// island; a body far away is in a DIFFERENT island.
//
// Phase A (persistent islands): IslandRootOf returns the body's persistent
// m_islandId (a non-member returns a high-bit-tagged slot). A stack coalesces
// into ONE island as its dynamic-dynamic contacts begin touching (touch-begin
// merges the per-body 1-body islands), so all four members share one root.
//
// WHY CHECKED WHILE AWAKE: the boxes only MERGE into one island once their
// contacts are touching, so we settle the stack into mutual contact first
// (still awake) and sample IslandRootOf on a step where all four are confirmed
// in contact. The persistent id is valid while asleep too, but checking while
// awake keeps the test independent of the sleep timeline.
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsIsland: IslandRootOf returns the same root for all members of one island",
          "[physics][island]")
{
    WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
    PhysicsWorld w(wd);

    // Floor: top surface at y = 0.
    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(20), Real(0.5));

    // A 4-box vertical stack. The boxes start slightly overlapping (small gap)
    // so dynamic-dynamic contacts form on the VERY FIRST step without having
    // to fall and settle first. This guarantees the UF has the 4-body chain
    // (b0<->b1, b1<->b2, b2<->b3 unioned) on each step while all are awake.
    // fixedRotation keeps them axis-aligned (same as AddBox helper).
    const Real hw = Real(0.5), hh = Real(0.5);
    // Place boxes 1 unit apart (so they just touch; diameter = 2*hh = 1).
    // top of box i = center_i - hh; bottom = center_i + hh.
    // Box0 bottom rests on floor (y=0), so box0 center = 0 - hh = -hh.
    // Box1 rests on box0: center1 = center0 - 2*hh - gap.
    // (With gravity +Y, "above" means smaller y. Lower y = higher in scene.)
    const Real gap = Real(0.02); // = kSkin -- small gap so they contact quickly
    const BodyHandle hBox0 = AddBox(w, Vec2(Real(0), -hh),                             hw, hh);
    const BodyHandle hBox1 = AddBox(w, Vec2(Real(0), -hh - (Real(2)*hh + gap)),        hw, hh);
    const BodyHandle hBox2 = AddBox(w, Vec2(Real(0), -hh - (Real(2)*hh + gap)*Real(2)), hw, hh);
    const BodyHandle hBox3 = AddBox(w, Vec2(Real(0), -hh - (Real(2)*hh + gap)*Real(3)), hw, hh);

    // One isolated dynamic body far away (its own island, never contacts the stack).
    const BodyHandle hFar = AddBox(w, Vec2(Real(50), -hh), hw, hh);

    // Step just enough for all four boxes to fall, contact each other, and form
    // an island -- but NOT so many steps that they all fall asleep (the bug is
    // only visible while the UF has the chain, which requires active contacts).
    // 30 steps: boxes are near the floor, awake, and in mutual contact.
    for (int k = 0; k < 30; ++k)
        w.Step(kStep);

    // Sanity: all four boxes must still be awake (if any slept the UF is stale).
    REQUIRE(w.IsAwake(hBox0));
    REQUIRE(w.IsAwake(hBox1));
    REQUIRE(w.IsAwake(hBox2));
    REQUIRE(w.IsAwake(hBox3));

    // All four stack members must share the SAME island root.
    // With the old one-hop `m_uf[i]` the chain b0->b1->b2->b3, after path-
    // halving makes b0 point to b2 (not root b3), IslandRootOf returns
    // b2 for b0/b1 and b3 for b2/b3 -- different roots for members of the
    // same island. The root-walk fix makes all four equal.
    const std::uint32_t root0 = w.IslandRootOf(hBox0.index);
    const std::uint32_t root1 = w.IslandRootOf(hBox1.index);
    const std::uint32_t root2 = w.IslandRootOf(hBox2.index);
    const std::uint32_t root3 = w.IslandRootOf(hBox3.index);
    CHECK(root0 == root1);
    CHECK(root0 == root2);
    CHECK(root0 == root3); // this fails with old one-hop impl (root0 != root3)

    // The isolated far body must be in a DIFFERENT island.
    const std::uint32_t rootFar = w.IslandRootOf(hFar.index);
    CHECK(rootFar != root0);
}

// ---------------------------------------------------------------------------
// Determinism: a settling-then-sleeping island is identical across two runs.
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsIsland: settle + sleep is deterministic across two runs", "[physics][island]")
{
    auto run = [](std::vector<Vec2>& pos, std::vector<int>& awake) {
        WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
        PhysicsWorld w(wd);

        AddFloor(w, Vec2(Real(0), Real(0.5)), Real(20), Real(0.5));
        const Real hw = Real(0.4), hh = Real(0.4);
        const int N = 3;
        std::vector<BodyHandle> boxes;
        for (int i = 0; i < N; ++i)
        {
            // gap = 2*kLinearSlop (0.005) -- a small seating clearance, not a scale division.
            const Real y = -(Real(2) * hh + Real(0.01)) * static_cast<Real>(i + 1);
            boxes.push_back(AddBox(w, Vec2(Real(0), y), hw, hh));
        }
        for (int k = 0; k < 700; ++k)
        {
            w.Step(kStep);
        }
        pos.clear();
        awake.clear();
        for (int i = 0; i < N; ++i)
        {
            pos.push_back(w.Position(boxes[i]));
            awake.push_back(w.IsAwake(boxes[i]) ? 1 : 0);
        }
    };

    std::vector<Vec2> p1, p2;
    std::vector<int>  a1, a2;
    run(p1, a1);
    run(p2, a2);

    REQUIRE(p1.size() == p2.size());
    for (std::size_t i = 0; i < p1.size(); ++i)
    {
        REQUIRE(p1[i].x == p2[i].x); // bit-identical (same scalar ops, same order)
        REQUIRE(p1[i].y == p2[i].y);
        REQUIRE(a1[i] == a2[i]);     // same sleep state
    }
}
