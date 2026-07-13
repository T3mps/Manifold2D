// Physics correctness batch, Task C: joints union islands + jointed constructs
// can sleep (the Box2D island-joint model).
//
// BEFORE this change jointed dynamics NEVER slept: the island sleep pass reset the
// sleep timer of every joint-attached dynamic every Step, so a settled rope /
// ragdoll / hinge kept solving forever. The fix makes a joint an ISLAND EDGE
// (AddJoint merges the two dynamics' islands; SplitIsland unions joint edges;
// RemoveJoint marks a split), drops the never-sleep pin, and skips asleep joints
// in the solver. A jointed construct now sleeps AS A UNIT via island membership.
//
// INVARIANTS (behavioral, not bit-matched):
//   (a) sleep-as-unit    : a jointed dynamic chain at rest sleeps (every member
//                          !IsAwake) -- the key new-behavior proof.
//   (b) shared-island    : jointed dynamics share one IslandRootOf (Step 1 merge).
//   (c) impulse-wakes-all: an impulse on ONE member wakes the WHOLE jointed island.
//   (d) remove-splits    : removing the joint lets the pieces settle/sleep
//                          independently (the island splits -> distinct roots).
//
// A zero-gravity resting construct is used deliberately: with the OLD code such a
// pair still would NOT sleep (the joint pin reset its timer each Step), so the
// sleep assertions prove the fix. Gravity-settling joint scenes remain covered by
// PhysicsJointsTest (whose awake-solve behavior is unchanged; they may now also
// sleep at their settled pose).
//
// PRESENTATION-FREE + C++23-clean.

#include <cstdint>

#include <catch2/catch_test_macros.hpp>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Joints/Joints.hpp>

using namespace Manifold2D::Physics;

namespace
{
    constexpr Real kStep = Real(1) / Real(60);

    // A zero-gravity world so a jointed pair truly rests (nothing accelerates
    // it): the only thing that could keep it awake is the removed never-sleep
    // pin. SoftStep is the default solver.
    PhysicsWorld MakeQuietWorld()
    {
        WorldDef wd;
        wd.gravityY = Real(0);
        wd.gravityX = Real(0);
        // sleepThreshold, restitutionThreshold, contactPushMaxVelocity, and
        // hashCellSize all inherit the MKS WorldDef defaults (0.05 / 1.0 / 3.0 / 1.0).
        return PhysicsWorld(wd);
    }

    // Default radius re-authored to the 0.1 m body-size floor (not /100's
    // 0.05 m -- nothing in this file depends on the exact radius).
    BodyHandle AddDynamicCircle(PhysicsWorld& w, Vec2 pos, Real r = Real(0.1))
    {
        BodyDef def;
        def.type          = BodyType::Dynamic;
        def.position      = pos;
        def.shape         = MakeCircle(r);
        def.density       = Real(1);
        def.linearDamping = Real(0);
        return w.AddBody(def);
    }

    // A distance joint between two dynamics at their creation separation
    // (length < 0 -> current |B - A|), so it applies ~zero impulse at rest.
    Joint* JoinDistance(PhysicsWorld& w, BodyHandle a, BodyHandle b)
    {
        JointDef jd;
        jd.kind   = JointKind::Distance;
        jd.a      = a;
        jd.b      = b;
        jd.length = Real(-1);
        return w.AddJoint(jd);
    }
} // namespace

// ---------------------------------------------------------------------------
// (a) sleep-as-unit + (b) shared-island: a jointed dynamic CHAIN (A-B-C, two
// distance joints) shares ONE island the instant it is created, and the whole
// chain sleeps as a unit once idle past the sleep time. Pre-fix this chain would
// never sleep (the joint pin reset each member's timer every Step).
// ---------------------------------------------------------------------------
TEST_CASE("PhysicsJointSleep: jointed chain shares one island and sleeps as a unit",
          "[physics][joint]")
{
    PhysicsWorld w = MakeQuietWorld();

    const BodyHandle a = AddDynamicCircle(w, Vec2(Real(0),   Real(0)));
    const BodyHandle b = AddDynamicCircle(w, Vec2(Real(0.3), Real(0)));
    const BodyHandle c = AddDynamicCircle(w, Vec2(Real(0.6), Real(0)));

    REQUIRE(JoinDistance(w, a, b) != nullptr);
    REQUIRE(JoinDistance(w, b, c) != nullptr);

    // (b) Step 1 merge: all three dynamics resolve to ONE island immediately
    // (before any Step), even though A and C share no touching contact.
    const std::uint32_t root = w.IslandRootOf(a.index);
    CHECK(root == w.IslandRootOf(b.index));
    CHECK(root == w.IslandRootOf(c.index));

    // They start awake (AddBody + AddJoint wake).
    REQUIRE(w.IsAwake(a));
    REQUIRE(w.IsAwake(b));
    REQUIRE(w.IsAwake(c));

    // (a) Step past the sleep time (0.5s): the whole chain sleeps as a unit.
    for (int k = 0; k < 180; ++k) { w.Step(kStep); }
    CHECK_FALSE(w.IsAwake(a));
    CHECK_FALSE(w.IsAwake(b));
    CHECK_FALSE(w.IsAwake(c));

    // Still one island after sleeping (sleep does not change membership).
    const std::uint32_t rootAsleep = w.IslandRootOf(a.index);
    CHECK(rootAsleep == w.IslandRootOf(b.index));
    CHECK(rootAsleep == w.IslandRootOf(c.index));
}

// ---------------------------------------------------------------------------
// (c) impulse-wakes-all: after a jointed chain sleeps, an impulse on ONE member
// wakes the WHOLE jointed island (island-fan-out wake through the joint edges).
// ---------------------------------------------------------------------------
TEST_CASE("PhysicsJointSleep: impulse on one member wakes the whole jointed island",
          "[physics][joint]")
{
    PhysicsWorld w = MakeQuietWorld();

    const BodyHandle a = AddDynamicCircle(w, Vec2(Real(0),   Real(0)));
    const BodyHandle b = AddDynamicCircle(w, Vec2(Real(0.3), Real(0)));
    const BodyHandle c = AddDynamicCircle(w, Vec2(Real(0.6), Real(0)));

    REQUIRE(JoinDistance(w, a, b) != nullptr);
    REQUIRE(JoinDistance(w, b, c) != nullptr);

    for (int k = 0; k < 180; ++k) { w.Step(kStep); }
    REQUIRE_FALSE(w.IsAwake(a));
    REQUIRE_FALSE(w.IsAwake(b));
    REQUIRE_FALSE(w.IsAwake(c));

    // Impulse an ENDPOINT (C): the whole island wakes at once. Authored as
    // mass x delta-v (rule 3): circle mass = density*kPi*r*r =
    // 1*kPi*0.1*0.1 (~0.0314 kg, density 1, r = 0.1 m); targetDv = -20 m/s
    // (well above sleepThreshold 0.05, well under the 400 m/s cap).
    const Real circleMass = Real(1) * kPi * Real(0.1) * Real(0.1);
    const Real targetDv = Real(-20);
    w.ApplyImpulse(c, Vec2(Real(0), circleMass * targetDv));
    CHECK(w.IsAwake(a));
    CHECK(w.IsAwake(b));
    CHECK(w.IsAwake(c));
}

// ---------------------------------------------------------------------------
// (d) remove-splits: a jointed pair shares one island; removing the joint marks
// the island a split candidate + wakes it, and after the deferred split the two
// bodies are DISTINCT islands that settle/sleep independently.
// ---------------------------------------------------------------------------
TEST_CASE("PhysicsJointSleep: removing a joint splits the island; pieces sleep independently",
          "[physics][joint]")
{
    PhysicsWorld w = MakeQuietWorld();

    const BodyHandle a = AddDynamicCircle(w, Vec2(Real(0),   Real(0)));
    const BodyHandle b = AddDynamicCircle(w, Vec2(Real(0.4), Real(0)));

    Joint* j = JoinDistance(w, a, b);
    REQUIRE(j != nullptr);

    // Shared island + settled asleep as a unit.
    REQUIRE(w.IslandRootOf(a.index) == w.IslandRootOf(b.index));
    for (int k = 0; k < 120; ++k) { w.Step(kStep); }
    REQUIRE_FALSE(w.IsAwake(a));
    REQUIRE_FALSE(w.IsAwake(b));

    // Drop the joint edge: RemoveJoint wakes the island + marks a split candidate.
    w.RemoveJoint(j);
    REQUIRE(w.JointCount() == 0u);

    // The deferred split resolves within a step or two; then the pieces re-idle
    // and sleep independently (zero gravity, no contact between them).
    for (int k = 0; k < 120; ++k) { w.Step(kStep); }

    // Island fractured: the two now-unjointed bodies have DISTINCT roots.
    CHECK(w.IslandRootOf(a.index) != w.IslandRootOf(b.index));
    // Each settled and slept on its own.
    CHECK_FALSE(w.IsAwake(a));
    CHECK_FALSE(w.IsAwake(b));
}
