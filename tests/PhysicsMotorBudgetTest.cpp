// Motor torque budget + runtime motor setters (MotorJoint, WheelJoint).
//
// Box2D re-applies a motor's accumulated impulse every sub-step, so the
// requested maxMotorTorque acts in every sub-step. The engine used to reset the
// accumulator once per STEP while clamping it to maxMotorTorque * subDt, which
// delivered maxMotorTorque / substepCount. Each sub-step now gets its own
// budget (Joint::BeginSubstep), and the setters let a vehicle change speed or a
// winch stop without rebuilding the joint.
// PRESENTATION-FREE + C++23-clean.
#include <cmath>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Joints/Joint.hpp>
#include <Manifold2D/Physics/Joints/Joints.hpp>
using namespace Manifold2D::Physics;
using Catch::Matchers::WithinRel;

namespace
{
    constexpr Real kStep = Real(1) / Real(60);

    struct Rig
    {
        PhysicsWorld w;
        BodyHandle hub, wheel;
        Real inertia = Real(0);
        Rig() : w([] { WorldDef d; d.gravityX = Real(0); d.gravityY = Real(0); return d; }())
        {
            BodyDef s; s.type = BodyType::Static; s.shape = MakeCircle(Real(0.05)); s.position = Vec2(Real(0), Real(0));
            hub = w.AddBody(s);
            BodyDef d; d.type = BodyType::Dynamic; d.shape = MakeCircle(Real(0.5)); d.density = Real(1); d.position = Vec2(Real(0), Real(0));
            wheel = w.AddBody(d);
            const Real m = w.GetBodyMass(wheel);
            inertia = Real(0.5) * m * Real(0.25); // solid disc, r = 0.5
            JointDef pin; pin.kind = JointKind::Revolute; pin.a = hub; pin.b = wheel; pin.anchor = Vec2(Real(0), Real(0));
            w.AddJoint(pin);
        }
    };
} // namespace

TEST_CASE("MotorBudget: a motor joint delivers its full maxMotorTorque, not a quarter of it", "[physics][joints][motor]")
{
    Rig r;
    JointDef m; m.kind = JointKind::Motor; m.a = r.hub; m.b = r.wheel;
    m.motorSpeed = Real(1000); m.maxMotorTorque = Real(0.2); // far from the target: torque-limited
    r.w.AddJoint(m);
    r.w.Step(kStep);
    // torque-limited from rest: w = T * dt / I
    const double expected = 0.2 * (1.0 / 60.0) / double(r.inertia);
    CHECK_THAT(static_cast<double>(r.w.AngularVelocity(r.wheel)), WithinRel(expected, 0.02));
}

TEST_CASE("MotorBudget: a wheel joint's motor delivers its full maxMotorTorque", "[physics][joints][motor]")
{
    Rig r;
    // A wheel joint from the static hub along a vertical axis, rigid spring off (freq 0).
    JointDef wj; wj.kind = JointKind::Wheel; wj.a = r.hub; wj.b = r.wheel;
    wj.anchor = Vec2(Real(0), Real(0)); wj.axis = Vec2(Real(0), Real(1)); wj.frequencyHz = Real(0);
    wj.enableMotor = true; wj.motorSpeed = Real(1000); wj.maxMotorTorque = Real(0.2);
    r.w.AddJoint(wj);
    r.w.Step(kStep);
    const double expected = 0.2 * (1.0 / 60.0) / double(r.inertia);
    CHECK_THAT(static_cast<double>(r.w.AngularVelocity(r.wheel)), WithinRel(expected, 0.02));
}

TEST_CASE("MotorBudget: a motor still settles at its target speed", "[physics][joints][motor]")
{
    Rig r;
    JointDef m; m.kind = JointKind::Motor; m.a = r.hub; m.b = r.wheel;
    m.motorSpeed = Real(3); m.maxMotorTorque = Real(50);
    r.w.AddJoint(m);
    for (int i = 0; i < 60; ++i) { r.w.Step(kStep); }
    CHECK_THAT(static_cast<double>(r.w.AngularVelocity(r.wheel)), WithinRel(3.0, 0.01));
}

TEST_CASE("MotorBudget: SetMotorSpeed and SetMaxMotorTorque change a live motor joint", "[physics][joints][motor]")
{
    Rig r;
    JointDef m; m.kind = JointKind::Motor; m.a = r.hub; m.b = r.wheel;
    m.motorSpeed = Real(3); m.maxMotorTorque = Real(50);
    auto* joint = dynamic_cast<MotorJoint*>(r.w.AddJoint(m));
    REQUIRE(joint != nullptr);
    for (int i = 0; i < 60; ++i) { r.w.Step(kStep); }
    joint->SetMotorSpeed(Real(-2));
    for (int i = 0; i < 60; ++i) { r.w.Step(kStep); }
    CHECK_THAT(static_cast<double>(r.w.AngularVelocity(r.wheel)), WithinRel(-2.0, 0.01));
    CHECK(joint->MotorSpeed() == Real(-2));
    joint->SetMaxMotorTorque(Real(0)); // motor off: the wheel coasts
    const Real before = r.w.AngularVelocity(r.wheel);
    for (int i = 0; i < 30; ++i) { r.w.Step(kStep); }
    CHECK_THAT(static_cast<double>(r.w.AngularVelocity(r.wheel)), WithinRel(double(before), 1e-4));
}

TEST_CASE("MotorBudget: a wheel joint's motor can be retargeted, enabled and disabled live", "[physics][joints][motor]")
{
    Rig r;
    JointDef wj; wj.kind = JointKind::Wheel; wj.a = r.hub; wj.b = r.wheel;
    wj.anchor = Vec2(Real(0), Real(0)); wj.axis = Vec2(Real(0), Real(1)); wj.frequencyHz = Real(0);
    wj.enableMotor = false; wj.motorSpeed = Real(0); wj.maxMotorTorque = Real(50);
    auto* joint = dynamic_cast<WheelJoint*>(r.w.AddJoint(wj));
    REQUIRE(joint != nullptr);
    r.w.Step(kStep);
    CHECK(r.w.AngularVelocity(r.wheel) == Real(0)); // motor disabled at creation
    joint->EnableMotor(true);
    joint->SetMotorSpeed(Real(4));
    for (int i = 0; i < 60; ++i) { r.w.Step(kStep); }
    CHECK_THAT(static_cast<double>(r.w.AngularVelocity(r.wheel)), WithinRel(4.0, 0.01));
    CHECK(joint->IsMotorEnabled());
    CHECK(joint->MotorSpeed() == Real(4));
    CHECK(joint->MaxMotorTorque() == Real(50));
}
