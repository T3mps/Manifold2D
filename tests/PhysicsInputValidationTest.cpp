// Task B2: input-validation guards at the PhysicsWorld API boundary.
//
// Prior to this guard, a non-finite (NaN/Inf) BodyDef::position handed to
// AddBody -- or a non-finite value handed to SetPosition/SetVelocity, or a
// zero/non-finite FixtureDef::shape handed to AddFixture -- propagated
// unchecked into the broadphase. For AddBody/SetPosition (mover path) and
// AddFixture (non-finite-shape path) that reaches DynamicTree::Update's
// MOSAIC_ASSERT(isfinite(box)) and aborts the process in a Debug build; for
// SetVelocity it silently poisons the body's velocity (no crash, just wrong
// state). The guards added alongside these tests reject the bad input at the
// boundary (WARN + no-op / kInvalidBody / kInvalidFixture) instead of letting
// it propagate.
//
// NOTE ON THE NaN-SHAPE CASE: "AddFixture rejects non-finite shape" exercises
// the SAME pre-guard abort path (a NaN-radius circle produces a NaN AABB).
// It is safe to run inline HERE only because the guard now runs first; it is
// NOT safe to run against the pre-guard code in the same process as the rest
// of the suite (see the Task B2 report for the isolated RED-phase run that
// verified the abort before the guard existed).
//
// PRESENTATION-FREE + C++20-clean.

#include <cmath>
#include <limits>

#include <catch2/catch_test_macros.hpp>

#include <Manifold2D/Physics/Fixture.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Shapes.hpp>

using namespace Manifold2D::Physics;

namespace
{
    constexpr Real kNaN = std::numeric_limits<Real>::quiet_NaN();
    constexpr Real kInf = std::numeric_limits<Real>::infinity();
}

// ---------------------------------------------------------------------------
// (a) AddBody rejects a non-finite position.
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsWorld: AddBody rejects non-finite position (NaN)", "[physics][validation]")
{
    WorldDef wd;
    PhysicsWorld w(wd);
    const std::uint32_t countBefore = w.Count();

    BodyDef def;
    def.type     = BodyType::Dynamic;
    def.position = Vec2(kNaN, Real(0));
    def.shape    = MakeCircle(Real(0.5));

    BodyHandle h = w.AddBody(def);

    REQUIRE(h == kInvalidBody);
    REQUIRE_FALSE(w.IsValid(h));
    REQUIRE(w.Count() == countBefore); // no slot consumed
}

TEST_CASE("PhysicsWorld: AddBody rejects non-finite position (Inf)", "[physics][validation]")
{
    WorldDef wd;
    PhysicsWorld w(wd);
    const std::uint32_t countBefore = w.Count();

    BodyDef def;
    def.type     = BodyType::Static;
    def.position = Vec2(Real(0), kInf);
    def.shape    = MakeAabb(Real(1), Real(1));

    BodyHandle h = w.AddBody(def);

    REQUIRE(h == kInvalidBody);
    REQUIRE_FALSE(w.IsValid(h));
    REQUIRE(w.Count() == countBefore);
}

// A finite AddBody is unaffected by the guard (no false-positive rejection).
TEST_CASE("PhysicsWorld: AddBody accepts a finite position", "[physics][validation]")
{
    WorldDef wd;
    PhysicsWorld w(wd);

    BodyDef def;
    def.type     = BodyType::Dynamic;
    def.position = Vec2(Real(3), Real(4));
    def.shape    = MakeCircle(Real(0.5));

    BodyHandle h = w.AddBody(def);

    REQUIRE(w.IsValid(h));
    REQUIRE(w.Position(h).x == Real(3));
    REQUIRE(w.Position(h).y == Real(4));
}

// ---------------------------------------------------------------------------
// (b) SetPosition / SetVelocity with NaN/Inf is a no-op.
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsWorld: SetPosition rejects a non-finite value (no-op)", "[physics][validation]")
{
    WorldDef wd;
    PhysicsWorld w(wd);

    // Static: SetPosition on a Static body does not touch the broadphase
    // (only movers do), so this stays a clean before/after Position()
    // comparison regardless of the guard's presence -- it is the guard
    // under test, not an incidental broadphase side effect.
    BodyDef def;
    def.type     = BodyType::Static;
    def.position = Vec2(Real(1), Real(2));
    def.shape    = MakeAabb(Real(1), Real(1));
    BodyHandle h = w.AddBody(def);
    REQUIRE(w.IsValid(h));

    w.SetPosition(h, Vec2(kNaN, Real(5)));

    REQUIRE(w.Position(h).x == Real(1)); // unchanged
    REQUIRE(w.Position(h).y == Real(2)); // unchanged
    REQUIRE(std::isfinite(w.Position(h).x));
    REQUIRE(std::isfinite(w.Position(h).y));
}

TEST_CASE("PhysicsWorld: SetVelocity rejects a non-finite value (no-op)", "[physics][validation]")
{
    WorldDef wd;
    PhysicsWorld w(wd);

    BodyDef def;
    def.type     = BodyType::Dynamic;
    def.position = Vec2(Real(0), Real(0));
    def.shape    = MakeCircle(Real(0.5));
    BodyHandle h = w.AddBody(def);
    REQUIRE(w.IsValid(h));

    w.SetVelocity(h, Vec2(Real(2), Real(3)));
    REQUIRE(w.Velocity(h).x == Real(2));
    REQUIRE(w.Velocity(h).y == Real(3));

    w.SetVelocity(h, Vec2(kInf, Real(0)));

    REQUIRE(w.Velocity(h).x == Real(2)); // unchanged from the last GOOD set
    REQUIRE(w.Velocity(h).y == Real(3));
    REQUIRE(std::isfinite(w.Velocity(h).x));
    REQUIRE(std::isfinite(w.Velocity(h).y));
}

// A finite SetPosition/SetVelocity is unaffected by the guard.
TEST_CASE("PhysicsWorld: SetPosition/SetVelocity accept finite values", "[physics][validation]")
{
    WorldDef wd;
    PhysicsWorld w(wd);

    BodyDef def;
    def.type     = BodyType::Kinematic;
    def.position = Vec2(Real(0), Real(0));
    def.shape    = MakeCircle(Real(0.5));
    BodyHandle h = w.AddBody(def);

    w.SetPosition(h, Vec2(Real(7), Real(8)));
    w.SetVelocity(h, Vec2(Real(1), Real(-1)));

    REQUIRE(w.Position(h).x == Real(7));
    REQUIRE(w.Position(h).y == Real(8));
    REQUIRE(w.Velocity(h).x == Real(1));
    REQUIRE(w.Velocity(h).y == Real(-1));
}

// ---------------------------------------------------------------------------
// (c) AddFixture rejects a degenerate (zero / non-finite extent) shape.
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsWorld: AddFixture rejects a zero-extent circle", "[physics][validation]")
{
    WorldDef wd;
    PhysicsWorld w(wd);

    BodyDef def;
    def.type     = BodyType::Dynamic;
    def.position = Vec2(Real(0), Real(0));
    def.shape    = MakeCircle(Real(0.5)); // valid primary fixture
    BodyHandle h = w.AddBody(def);
    const std::uint32_t fxBefore = w.FixtureCount(h);

    FixtureDef bad;
    bad.shape = MakeCircle(Real(0)); // zero radius -- degenerate

    FixtureHandle fh = w.AddFixture(h, bad);

    REQUIRE(fh == kInvalidFixture);
    REQUIRE(w.FixtureCount(h) == fxBefore); // no fixture added
}

TEST_CASE("PhysicsWorld: AddFixture rejects a zero-extent aabb", "[physics][validation]")
{
    WorldDef wd;
    PhysicsWorld w(wd);

    BodyDef def;
    def.type        = BodyType::Dynamic;
    def.position    = Vec2(Real(0), Real(0));
    def.shape       = MakeCircle(Real(0.5));
    def.fixedRotation = true;
    BodyHandle h = w.AddBody(def);
    const std::uint32_t fxBefore = w.FixtureCount(h);

    FixtureDef bad;
    bad.shape = MakeAabb(Real(0), Real(1)); // zero half-width -- degenerate

    FixtureHandle fh = w.AddFixture(h, bad);

    REQUIRE(fh == kInvalidFixture);
    REQUIRE(w.FixtureCount(h) == fxBefore);
}

TEST_CASE("PhysicsWorld: AddFixture rejects a non-finite (NaN) shape", "[physics][validation]")
{
    WorldDef wd;
    PhysicsWorld w(wd);

    BodyDef def;
    def.type     = BodyType::Dynamic;
    def.position = Vec2(Real(0), Real(0));
    def.shape    = MakeCircle(Real(0.5));
    BodyHandle h = w.AddBody(def);
    const std::uint32_t fxBefore = w.FixtureCount(h);

    FixtureDef bad;
    bad.shape = MakeCircle(kNaN); // NaN radius -- degenerate; pre-guard this
                                   // reaches DynamicTree::Update's finite-AABB
                                   // assert (see file header note).

    FixtureHandle fh = w.AddFixture(h, bad);

    REQUIRE(fh == kInvalidFixture);
    REQUIRE(w.FixtureCount(h) == fxBefore);
}

// A finite/regular AddFixture is unaffected by the guard.
TEST_CASE("PhysicsWorld: AddFixture accepts a regular shape", "[physics][validation]")
{
    WorldDef wd;
    PhysicsWorld w(wd);

    BodyDef def;
    def.type     = BodyType::Dynamic;
    def.position = Vec2(Real(0), Real(0));
    def.shape    = MakeCircle(Real(0.5));
    BodyHandle h = w.AddBody(def);
    const std::uint32_t fxBefore = w.FixtureCount(h);

    FixtureDef good;
    good.shape    = MakeCircle(Real(0.2));
    good.localPos = Vec2(Real(1), Real(0));

    FixtureHandle fh = w.AddFixture(h, good);

    REQUIRE(fh != kInvalidFixture);
    REQUIRE(w.IsValid(fh));
    REQUIRE(w.FixtureCount(h) == fxBefore + 1u);
}
