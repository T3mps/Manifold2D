// Physics M6 P2.5: the joint set (Distance/Revolute/Weld/Prismatic/Mouse +
// Wheel/Motor) -- BEHAVIORAL invariant tests.
//
// PORT NOTE: P2.5 ported Client/src/physics/Joints.lua into
// Arcane/Physics/Joints/ (Joint base + DistanceJoint/RevoluteJoint/WeldJoint/
// PrismaticJoint/MouseJoint) and ADDED two Box2D-derived joints (WheelJoint /
// MotorJoint). The oracle for the five ported joints is the harness "joints
// (M3)" block (Client/src/tests/physics_harness/main.lua:779-817): a revolute
// pendulum settles below its anchor at rod length; a distance joint holds its
// separation; a mouse joint drags a body to its target; removeJoint detaches.
// Reproduced here in the engine's Cartesian convention (+Y is DOWN, gravity is
// +Y) since the harness map used +Y down too. Wheel/Motor invariants are
// behavioral (suspension compresses + rebounds + stays on the axis; the motor
// drives the relative angular velocity to motorSpeed within its torque clamp).
//
// INVARIANTS (behavioral, not bit-matched):
//   * Distance : a hung body settles at the joint length from the hub.
//   * Revolute : a pendulum settles hanging straight below the anchor at rod
//                length (harness 791-797).
//   * Weld     : two welded bodies stay rigid (relative position + angle).
//   * Prismatic: a body slides only along its axis (no perp drift / rel rot).
//   * Mouse    : drags a body to a target (harness 809-814); removeJoint
//                detaches (joint count drops -- harness 816).
//   * Wheel    : suspension compresses under load + rebounds; body stays on the
//                axis line (no perp drift); the motor drives wheel rotation.
//   * Motor    : drives the relative angular velocity to motorSpeed (within the
//                maxMotorTorque clamp); reaches ~motorSpeed at steady state.
//   * determinism: a jointed scene run-twice is identical.
// Core joints run under SoftStep -- THE solver (the P2.3 Baumgarte A/B oracle
// was retired 2026-07-03, MKS P2).
//
// PRESENTATION-FREE + C++20-clean.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Joints/Joints.hpp>

using namespace Manifold2D::Physics;
using Catch::Approx;

namespace
{
    constexpr Real kStep = Real(1) / Real(60);

    BodyHandle AddStaticAnchor(PhysicsWorld& w, Vec2 pos, Real r = Real(0.2))
    {
        BodyDef def;
        def.type     = BodyType::Static;
        def.position = pos;
        def.shape    = MakeCircle(r);
        return w.AddBody(def);
    }

    BodyHandle AddDynamicCircle(PhysicsWorld& w, Vec2 pos, Real r = Real(0.5),
                                Real linDamp = Real(1.5))
    {
        BodyDef def;
        def.type          = BodyType::Dynamic;
        def.position      = pos;
        def.shape         = MakeCircle(r);
        def.density       = Real(1);
        def.linearDamping = linDamp;
        return w.AddBody(def);
    }

    WorldDef GravityWorld()
    {
        // Gravity inherits the MKS WorldDef default (+Y down, g=10); sleepThreshold,
        // restitutionThreshold, contactPushMaxVelocity, and hashCellSize inherit the
        // MKS defaults too (0.05 / 1.0 / 3.0 / 1.0).
        WorldDef wd;
        return wd;
    }
} // namespace

// ---------------------------------------------------------------------------
// Revolute pendulum: settles hanging straight below the anchor at rod length.
// (Direct Cartesian port of harness main.lua:790-797.)
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsJoints: revolute pendulum settles below anchor", "[physics][joints]")
{
    auto run = []() {
        PhysicsWorld w(GravityWorld());
        BodyHandle anchor = AddStaticAnchor(w, Vec2(Real(10), Real(5)));
        BodyHandle bob     = AddDynamicCircle(w, Vec2(Real(14), Real(5)));

        JointDef jd;
        jd.kind   = JointKind::Revolute;
        jd.a      = anchor;
        jd.b      = bob;
        jd.anchor = Vec2(Real(10), Real(5));
        REQUIRE(w.AddJoint(jd) != nullptr);

        for (int k = 0; k < 900; ++k)
        {
            w.Step(kStep);
        }
        const Vec2 p = w.Position(bob);
        // Rod length is the creation separation (14-10 = 4 m); the bob hangs
        // straight below the anchor (+Y down) -> x ~ 10, y ~ 9.
        CHECK(std::fabs(p.x - Real(10)) < Real(0.3));
        CHECK(std::fabs(p.y - Real(9)) < Real(0.3));
    };
    run();
}

// ---------------------------------------------------------------------------
// Distance joint holds its separation under gravity.
// (Direct Cartesian port of harness main.lua:799-806.)
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsJoints: distance joint holds separation", "[physics][joints]")
{
    auto run = []() {
        PhysicsWorld w(GravityWorld());
        BodyHandle hub = AddStaticAnchor(w, Vec2(Real(30), Real(5)));
        BodyHandle sat = AddDynamicCircle(w, Vec2(Real(30), Real(11)));

        JointDef jd;
        jd.kind   = JointKind::Distance;
        jd.a      = hub;
        jd.b      = sat;
        jd.length = Real(6);
        REQUIRE(w.AddJoint(jd) != nullptr);

        for (int k = 0; k < 600; ++k)
        {
            w.Step(kStep);
        }
        const Vec2 p = w.Position(sat);
        const Real d = std::sqrt((p.x - Real(30)) * (p.x - Real(30)) +
                                 (p.y - Real(5)) * (p.y - Real(5)));
        CHECK(std::fabs(d - Real(6)) < Real(0.2));
    };
    run();
}

// ---------------------------------------------------------------------------
// Mouse joint drags a body to its target; removeJoint detaches.
// (Direct Cartesian port of harness main.lua:808-816.)
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsJoints: mouse joint drags to target + removeJoint detaches",
          "[physics][joints]")
{
    WorldDef wd;
    wd.gravityY = Real(0); // the harness used gravityScale=0 for the mouse chip
    wd.gravityX = Real(0); // zero-g scene (deliberate, both axes); other WorldDef fields inherit MKS defaults
    PhysicsWorld w(wd);

    BodyHandle chip = AddDynamicCircle(w, Vec2(Real(50), Real(10)), Real(0.5), Real(0));

    JointDef jd;
    jd.kind     = JointKind::Mouse;
    jd.b        = chip;
    jd.target   = Vec2(Real(54), Real(6));
    // 8 N = the px-era 8000 /1000 (linear force: mass /100 x accel /10).
    jd.maxForce = Real(8);
    Joint* mj = w.AddJoint(jd);
    REQUIRE(mj != nullptr);
    REQUIRE(w.JointCount() == 1u);

    for (int k = 0; k < 240; ++k)
    {
        w.Step(kStep);
    }
    const Vec2 p = w.Position(chip);
    CHECK(std::fabs(p.x - Real(54)) < Real(0.4));
    CHECK(std::fabs(p.y - Real(6)) < Real(0.8));

    w.RemoveJoint(mj);
    CHECK(w.JointCount() == 0u);

    // After detach, stepping must not crash and the joint pull is gone (the key
    // invariant is the count drop, already checked -- harness 816).
    for (int k = 0; k < 60; ++k)
    {
        w.Step(kStep);
    }
}

// ---------------------------------------------------------------------------
// Weld joint: two welded bodies stay rigid (relative offset + angle locked).
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsJoints: weld keeps two bodies rigid", "[physics][joints]")
{
    PhysicsWorld w(GravityWorld());

    // A static anchor + a dynamic body welded to it: the dynamic body must not
    // fall (the weld is rigid against the static).
    BodyHandle anchor = AddStaticAnchor(w, Vec2(Real(0), Real(0)));
    BodyHandle welded  = AddDynamicCircle(w, Vec2(Real(2), Real(0)), Real(0.5), Real(0));

    const Vec2 start = w.Position(welded);
    const Real startAngle = w.GetAngle(welded);

    JointDef jd;
    jd.kind   = JointKind::Weld;
    jd.a      = anchor;
    jd.b      = welded;
    jd.anchor = Vec2(Real(2), Real(0)); // weld point at the body
    REQUIRE(w.AddJoint(jd) != nullptr);

    for (int k = 0; k < 600; ++k)
    {
        w.Step(kStep);
    }
    const Vec2 p = w.Position(welded);
    // Rigidly held to the static -> stays near its start (small Baumgarte give).
    CHECK(std::fabs(p.x - start.x) < Real(0.2));
    CHECK(std::fabs(p.y - start.y) < Real(0.2));
    // Angle stays locked (no spin under gravity).
    CHECK(std::fabs(w.GetAngle(welded) - startAngle) < Real(0.1));
}

// ---------------------------------------------------------------------------
// Prismatic joint: a body slides only along its axis (no perpendicular drift,
// no relative rotation). Axis is HORIZONTAL; gravity is +Y (perpendicular) so
// the constraint must hold the body's y while letting x move freely.
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsJoints: prismatic constrains to its axis", "[physics][joints]")
{
    PhysicsWorld w(GravityWorld());

    BodyHandle anchor = AddStaticAnchor(w, Vec2(Real(0), Real(0)));
    // Dynamic body offset along +X from the anchor; axis is +X (horizontal).
    BodyHandle slider = AddDynamicCircle(w, Vec2(Real(4), Real(0)), Real(0.5), Real(0));

    const Real startY = w.Position(slider).y;

    JointDef jd;
    jd.kind = JointKind::Prismatic;
    jd.a    = anchor;
    jd.b    = slider;
    jd.axis = Vec2(Real(1), Real(0)); // horizontal slide axis
    REQUIRE(w.AddJoint(jd) != nullptr);

    // Gravity pulls +Y (perpendicular to the axis): the perp constraint must
    // hold y. Push the slider along +X with an impulse so it slides. Rule-3:
    // impulse = mass x dv; sliderMass = density*pi*r^2 = 1*pi*0.5^2 (~0.785 kg,
    // r = 0.5 m), pushDv = 2.5 m/s (well under the 400 m/s cap; enough to carry
    // the slider past x > 4.5 m). ~1.96 N*s = the px-era 2000 impulse /1000
    // (mass /100 x velocity /10).
    const Real sliderMass = Real(1) * kPi * Real(0.5) * Real(0.5);
    const Real pushDv     = Real(2.5);
    w.ApplyImpulse(slider, Vec2(sliderMass * pushDv, Real(0)));

    for (int k = 0; k < 300; ++k)
    {
        w.Step(kStep);
    }
    const Vec2 p = w.Position(slider);
    // No perpendicular (y) drift despite gravity (the axis is horizontal).
    CHECK(std::fabs(p.y - startY) < Real(0.2));
    // It slid along +X (the axis) -- moved meaningfully from x=4.
    CHECK(p.x > Real(4.5));
    // No relative rotation locked in.
    CHECK(std::fabs(w.GetAngle(slider)) < Real(0.1));
}

// ---------------------------------------------------------------------------
// Motor joint: drives the relative angular velocity of B vs A toward
// motorSpeed within the maxMotorTorque clamp; reaches ~motorSpeed at steady
// state. A is a static anchor (angVelA = 0) so angVelB -> motorSpeed.
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsJoints: motor drives relative angular velocity", "[physics][joints]")
{
    WorldDef wd; // no gravity -- isolate the angular drive (deliberate zero-g, both axes)
    wd.gravityX = Real(0);
    wd.gravityY = Real(0);
    PhysicsWorld w(wd);

    BodyHandle anchor = AddStaticAnchor(w, Vec2(Real(0), Real(0)));
    // A dynamic disk free to spin (NOT fixedRotation).
    BodyDef def;
    def.type     = BodyType::Dynamic;
    def.position = Vec2(Real(0), Real(0));
    def.shape    = MakeCircle(Real(1));
    def.density  = Real(1);
    BodyHandle disk = w.AddBody(def);

    const Real targetSpeed = Real(5); // rad/s
    JointDef jd;
    jd.kind           = JointKind::Motor;
    jd.a              = anchor;
    jd.b              = disk;
    jd.motorSpeed     = targetSpeed;
    // 100 N*m = the px-era 1e6 /10000 (torque: inertia /10000 x angular-accel /1);
    // still "plenty" relative to the new-scale disk inertia.
    jd.maxMotorTorque = Real(100);
    REQUIRE(w.AddJoint(jd) != nullptr);

    REQUIRE(w.GetAngle(disk) == Approx(Real(0)));
    for (int k = 0; k < 120; ++k)
    {
        w.Step(kStep);
    }
    // angVel is not directly exposed on the public Body API; observe the angle
    // advancing at ~motorSpeed: over the last step, dAngle/dt ~ motorSpeed.
    const Real a0 = w.GetAngle(disk);
    w.Step(kStep);
    const Real a1 = w.GetAngle(disk);
    const Real measured = (a1 - a0) / kStep;
    CHECK(measured == Approx(targetSpeed).margin(Real(0.5)));
    // The disk has clearly spun up.
    CHECK(a1 > Real(0.1));
}

// ---------------------------------------------------------------------------
// Motor torque clamp: with a tiny maxMotorTorque the motor CANNOT reach the
// target speed in the budget -- it advances slowly (bounded by the clamp).
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsJoints: motor respects the torque clamp", "[physics][joints]")
{
    WorldDef wd; // deliberate zero-g, both axes
    wd.gravityX = Real(0);
    wd.gravityY = Real(0);
    PhysicsWorld w(wd);

    BodyHandle anchor = AddStaticAnchor(w, Vec2(Real(0), Real(0)));
    BodyDef def;
    def.type     = BodyType::Dynamic;
    def.position = Vec2(Real(0), Real(0));
    def.shape    = MakeCircle(Real(1));
    def.density  = Real(1);
    BodyHandle disk = w.AddBody(def);

    JointDef jd;
    jd.kind           = JointKind::Motor;
    jd.a              = anchor;
    jd.b              = disk;
    jd.motorSpeed     = Real(50);       // a high target
    // 0.0001 N*m = the px-era 1 /10000 (torque: inertia /10000 x angular-accel /1);
    // a tiny torque budget the motor cannot spend fast enough to reach 50 rad/s.
    jd.maxMotorTorque = Real(0.0001);
    REQUIRE(w.AddJoint(jd) != nullptr);

    const Real a0 = w.GetAngle(disk);
    w.Step(kStep);
    const Real a1 = w.GetAngle(disk);
    const Real rate = (a1 - a0) / kStep;
    // clamp keeps the one-step rate far below the 50 rad/s target; a removed clamp would blow past 2.
    CHECK(rate < Real(2));
    CHECK(rate >= Real(0));
}

// ---------------------------------------------------------------------------
// Wheel joint: suspension compresses under load + rebounds (spring); the body
// stays on the axis line (no perpendicular drift); the motor drives rotation.
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsJoints: wheel suspension holds the axis + springs", "[physics][joints]")
{
    PhysicsWorld w(GravityWorld());

    // A static chassis attach point + a dynamic wheel hanging below it. The
    // suspension anchor is at the WHEEL CENTER (the axle, the b2WheelJoint
    // convention -- the wheel rotates about its own anchor so spinning it does
    // not drive lateral drift). The axis is VERTICAL (+Y); gravity (+Y) loads
    // the spring along the axis. Lateral (perp, X) motion is held rigidly.
    BodyHandle chassis = AddStaticAnchor(w, Vec2(Real(0), Real(0)));
    BodyDef def;
    def.type     = BodyType::Dynamic;
    def.position = Vec2(Real(0), Real(3)); // 3 m below the chassis
    def.shape    = MakeCircle(Real(0.8));
    def.density  = Real(1);
    BodyHandle wheel = w.AddBody(def);

    JointDef jd;
    jd.kind          = JointKind::Wheel;
    jd.a             = chassis;
    jd.b             = wheel;
    jd.anchor        = Vec2(Real(0), Real(3)); // axle = wheel center
    jd.axis          = Vec2(Real(0), Real(1));  // vertical suspension axis
    jd.frequencyHz   = Real(4);
    jd.dampingRatio  = Real(0.7);
    jd.enableMotor   = true;
    jd.motorSpeed    = Real(8);  // spin the wheel
    // 100 N*m = the px-era 1e6 /10000 (torque: inertia /10000 x angular-accel /1).
    jd.maxMotorTorque = Real(100);
    REQUIRE(w.AddJoint(jd) != nullptr);

    const Real startAngle = w.GetAngle(wheel);

    // Let the suspension settle under gravity.
    Real minY = w.Position(wheel).y;
    Real maxY = w.Position(wheel).y;
    for (int k = 0; k < 600; ++k)
    {
        w.Step(kStep);
        const Vec2 p = w.Position(wheel);
        minY = std::min(minY, p.y);
        maxY = std::max(maxY, p.y);
        // PERPENDICULAR (x) drift must stay ~0 (the rigid perp constraint).
        CHECK(std::fabs(p.x) < Real(0.2));
    }
    const Vec2 settled = w.Position(wheel);
    // The wheel hangs near its suspension length below the chassis along +Y
    // (gravity stretches the spring a bit, but it is bounded).
    CHECK(settled.y > Real(2.8));        // still roughly at suspension length
    CHECK(settled.y < Real(12));         // spring did not let it fall away
    // The spring MOVED (compressed/extended) during settling, not a rigid lock.
    CHECK((maxY - minY) > Real(0.05));
    // The motor spun the wheel.
    CHECK(std::fabs(w.GetAngle(wheel) - startAngle) > Real(0.5));
}

// ---------------------------------------------------------------------------
// RemoveBody auto-removes joints referencing the destroyed body
// (PhysicsWorld.lua:281-286).
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsJoints: RemoveBody drops referencing joints", "[physics][joints]")
{
    PhysicsWorld w(GravityWorld());
    BodyHandle anchor = AddStaticAnchor(w, Vec2(Real(0), Real(0)));
    BodyHandle bob     = AddDynamicCircle(w, Vec2(Real(4), Real(0)));

    JointDef jd;
    jd.kind   = JointKind::Revolute;
    jd.a      = anchor;
    jd.b      = bob;
    jd.anchor = Vec2(Real(0), Real(0));
    REQUIRE(w.AddJoint(jd) != nullptr);
    REQUIRE(w.JointCount() == 1u);

    // A couple steps so the joint has Prepared (BodyA/BodyB resolved).
    w.Step(kStep);
    w.Step(kStep);

    w.RemoveBody(bob);
    CHECK(w.JointCount() == 0u); // the joint referencing bob is gone

    // Stepping the world after the removal must not crash (stale joint dropped).
    for (int k = 0; k < 10; ++k)
    {
        w.Step(kStep);
    }
    CHECK_FALSE(w.IsValid(bob));
}

// ---------------------------------------------------------------------------
// Determinism: a jointed scene run twice produces identical positions.
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsJoints: jointed scene is deterministic", "[physics][joints]")
{
    auto runOnce = []() -> Vec2 {
        PhysicsWorld w(GravityWorld());
        BodyHandle anchor = AddStaticAnchor(w, Vec2(Real(10), Real(5)));
        BodyHandle bob     = AddDynamicCircle(w, Vec2(Real(14), Real(5)));
        JointDef jd;
        jd.kind   = JointKind::Revolute;
        jd.a      = anchor;
        jd.b      = bob;
        jd.anchor = Vec2(Real(10), Real(5));
        w.AddJoint(jd);
        for (int k = 0; k < 300; ++k)
        {
            w.Step(kStep);
        }
        return w.Position(bob);
    };
    const Vec2 a = runOnce();
    const Vec2 b = runOnce();
    REQUIRE(a.x == b.x);
    REQUIRE(a.y == b.y);
}
