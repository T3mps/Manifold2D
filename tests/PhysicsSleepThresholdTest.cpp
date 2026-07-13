// Sleep-threshold + maxExtent + combined-predicate tests (Box2D-faithful sleep).
// Companion to PhysicsAwakeSetTest.cpp / PhysicsIslandTest.cpp. The headline
// guard is "a slowly-rolling body sleeps": the never-settle bug was 1-2 circles
// rolling just over the old separate angular gate (0.05 rad/s) vetoing a whole
// island forever. The combined Box2D test ( |v| + |w|*maxExtent < sleepThreshold )
// counts a slow roll as small surface motion, so the pile sleeps.
// PRESENTATION-FREE + C++23-clean.
#include <cmath>
#include <cstdint>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
using namespace Manifold2D::Physics;
namespace { constexpr Real kStep = Real(1) / Real(60); }

TEST_CASE("PhysicsSleep: maxExtent equals circle radius for a centered circle", "[physics][sleep]")
{
    WorldDef wd;
    PhysicsWorld w(wd);
    BodyDef d; d.type = BodyType::Dynamic; d.position = Vec2(Real(0), Real(0));
    d.shape = MakeCircle(Real(0.1)); d.density = Real(1);
    const BodyHandle b = w.AddBody(d);
    CHECK_THAT(static_cast<double>(w.MaxExtentSlot(w.SlotOf(b))),
               Catch::Matchers::WithinAbs(0.1, 1e-4));
}

TEST_CASE("PhysicsSleep: maxExtent equals box half-diagonal for an AABB", "[physics][sleep]")
{
    WorldDef wd;
    PhysicsWorld w(wd);
    BodyDef d; d.type = BodyType::Dynamic; d.position = Vec2(Real(0), Real(0));
    // 3:4:5 kept at /10 scale; /100 would undershoot the 0.1 m body-size floor (MKS P3)
    d.shape = MakeAabb(Real(0.3), Real(0.4)); d.density = Real(1); d.fixedRotation = true;
    const BodyHandle b = w.AddBody(d);
    // corner distance from center = sqrt(0.3^2 + 0.4^2) = 0.5, radius 0.
    CHECK_THAT(static_cast<double>(w.MaxExtentSlot(w.SlotOf(b))),
               Catch::Matchers::WithinAbs(0.5, 1e-4));
}

TEST_CASE("PhysicsSleep: per-body sleepThreshold override beats the world default", "[physics][sleep]")
{
    WorldDef wd;
    wd.sleepThreshold = Real(0.02); // custom world default, deliberately != engine default 0.05 (MKS P3)
    PhysicsWorld w(wd);
    BodyDef d; d.type = BodyType::Dynamic; d.position = Vec2(Real(0), Real(0));
    d.shape = MakeCircle(Real(0.1)); d.density = Real(1);
    const BodyHandle inherit = w.AddBody(d);
    d.sleepThreshold = Real(0.08);  // per-body override, != world default and != engine default
    const BodyHandle overridden = w.AddBody(d);
    CHECK(w.SleepThresholdSlot(w.SlotOf(inherit))    == Real(0.02));
    CHECK(w.SleepThresholdSlot(w.SlotOf(overridden)) == Real(0.08));
}

// THE regression guard this bug lacked: a slowly-ROLLING body (the never-settle
// blocker class) must sleep. With the OLD separate gates (|w| < 0.05) a body at
// w=0.08 never idled -> never slept. The combined test ( |v| + |w|*maxExtent )
// counts 0.08*0.1 = 0.008 m/s < 0.05 -> idle -> sleeps.
TEST_CASE("PhysicsSleep: a slowly-rolling body sleeps (combined test)", "[physics][sleep]")
{
    WorldDef wd; wd.gravityY = Real(0);
    PhysicsWorld w(wd);
    BodyDef d; d.type = BodyType::Dynamic; d.position = Vec2(Real(0), Real(0));
    d.shape = MakeCircle(Real(0.1)); d.density = Real(1); // NOT fixedRotation -> can roll
    const BodyHandle b = w.AddBody(d);
    // combined = |w|*maxExtent = 0.08 * 0.1 = 0.008 m/s, 6.25x BELOW the 0.05
    // default (MKS P3) -- exactly the px-era margin (0.8 vs 5, same 6.25x ratio)
    w.SetAngularVelocity(b, Real(0.08));         // rolls at 0.08 rad/s; |v|=0
    for (int k = 0; k < 120; ++k) { w.Step(kStep); } // > kSleepTime (0.5s = 30 steps)
    REQUIRE_FALSE(w.IsAwake(b));                  // combined sleepVel 0.008 < 0.05 -> sleeps
}

// Counter-guard: a body genuinely spinning fast must NOT sleep.
TEST_CASE("PhysicsSleep: a fast-spinning body never sleeps", "[physics][sleep]")
{
    WorldDef wd; wd.gravityY = Real(0);
    PhysicsWorld w(wd);
    BodyDef d; d.type = BodyType::Dynamic; d.position = Vec2(Real(0), Real(0));
    d.shape = MakeCircle(Real(0.1)); d.density = Real(1);
    const BodyHandle b = w.AddBody(d);
    // combined = |w|*maxExtent = 2.0 * 0.1 = 0.2 m/s, 4x ABOVE the 0.05 default
    // (MKS P3) -- exactly the px-era margin (20 vs 5, same 4x ratio)
    w.SetAngularVelocity(b, Real(2.0));          // 2.0*0.1 = 0.2 m/s >> 0.05
    for (int k = 0; k < 120; ++k) { w.Step(kStep); REQUIRE(w.IsAwake(b)); }
}
