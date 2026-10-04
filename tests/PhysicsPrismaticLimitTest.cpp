// PhysicsPrismaticLimitTest.cpp
// [physics][joints][prismatic][limit]: the prismatic joint's translation limit
// (Box2D v3 b2PrismaticJoint: prismatic_joint.c b2SolvePrismaticJoint,
// b2PrismaticJointDef enableLimit / lowerTranslation / upperTranslation).
//
//   * translation = dot(axis, (pB - pA) - origin): B's slide relative to A along
//     the axis, zero at creation (b2PrismaticJoint_GetTranslation).
//   * limit: lowerTranslation <= translation <= upperTranslation, two one-sided
//     constraints with their own accumulated (warm-started) impulses:
//     speculative while open (bias C / h), soft when breached (the joint
//     constraint softness: 60 Hz clamped to 0.25 / h, damping ratio 2), the
//     push-out only on the biased pass (useBias), each clamped >= 0.
//   * the motor is solved first: a motor driving into the limit stops there.
//   * reaction: the limit impulses are part of the force on B, as Box2D's
//     b2GetPrismaticJointForce includes lowerImpulse - upperImpulse.
//
// Engine convention: +Y is DOWN, gravity +Y (10 m/s^2 by default).
//
// PRESENTATION-FREE, ASCII comments, C++23.
#include <algorithm>
#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Joints/Joints.hpp>

using namespace Manifold2D::Physics;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace
{
    constexpr Real kStep = Real(1) / Real(60);
    constexpr double kG = 10.0;

    void Run(PhysicsWorld& w, int n)
    {
        for (int i = 0; i < n; ++i) { w.Step(kStep); }
    }

    // A static rail at the origin and a 1 x 1 m box (1 kg) on a prismatic joint
    // along `axis`; nothing collides (only the joint acts) and the box never
    // sleeps (every step is solved).
    struct Slide
    {
        PhysicsWorld w;
        BodyHandle rail, box;
        PrismaticJoint* joint = nullptr;
        Real mass = Real(0);
        Slide(bool gravity, Vec2 axis, JointDef j)
            : w([gravity] { WorldDef d; if (!gravity) { d.gravityX = Real(0); d.gravityY = Real(0); } return d; }())
        {
            BodyDef s; s.type = BodyType::Static; s.shape = MakeCircle(Real(0.05)); s.position = Vec2(Real(0), Real(0));
            s.maskBits = 0u;
            rail = w.AddBody(s);
            BodyDef d; d.type = BodyType::Dynamic; d.shape = MakeAabb(Real(0.5), Real(0.5)); d.fixedRotation = true; d.density = Real(1);
            d.position = Vec2(Real(0), Real(0)); d.maskBits = 0u; d.sleepThreshold = Real(0);
            box = w.AddBody(d);
            mass = w.GetBodyMass(box);
            j.kind = JointKind::Prismatic; j.a = rail; j.b = box; j.axis = axis;
            joint = dynamic_cast<PrismaticJoint*>(w.AddJoint(j));
            REQUIRE(joint != nullptr);
        }
    };

    JointDef Limited(Real lower, Real upper)
    {
        JointDef j;
        j.enableLimit = true;
        j.lowerTranslation = lower;
        j.upperTranslation = upper;
        return j;
    }
} // namespace

TEST_CASE("Prismatic limit: a box sliding down under gravity comes to rest on the limit",
          "[physics][joints][prismatic][limit]")
{
    // Vertical axis (+y down): gravity drives the translation up toward 0.5.
    Slide s(true, Vec2(Real(0), Real(1)), Limited(Real(-1), Real(0.5)));
    CHECK(s.joint->IsLimitEnabled());
    CHECK(s.joint->GetTranslation(s.w) == Real(0)); // zero at creation
    Real most = Real(-9);
    for (int i = 0; i < 180; ++i)
    {
        s.w.Step(kStep);
        most = std::max(most, s.w.Position(s.box).y);
    }
    const Vec2 p = s.w.Position(s.box);
    INFO("y " << p.y << " most " << most << " v " << s.w.Velocity(s.box).y);
    CHECK_THAT(static_cast<double>(p.y), WithinAbs(0.5, 0.003));       // rests on the limit (mm)
    CHECK(most < Real(0.505));                                           // and does not punch through it
    CHECK_THAT(static_cast<double>(s.w.Velocity(s.box).y), WithinAbs(0.0, 1e-3));
    CHECK_THAT(static_cast<double>(p.x), WithinAbs(0.0, 1e-4));         // still on the axis
    CHECK_THAT(static_cast<double>(s.joint->GetTranslation(s.w)), WithinAbs(double(p.y), 1e-6));
}

TEST_CASE("Prismatic limit: the reaction at rest on the limit carries m g along the axis",
          "[physics][joints][prismatic][limit][reaction]")
{
    Slide s(true, Vec2(Real(0), Real(1)), Limited(Real(-1), Real(0.5)));
    Run(s.w, 180);
    const double mg = double(s.mass) * kG;
    const Vec2 f = s.w.JointReactionForce(s.joint);
    INFO("f " << f.x << ", " << f.y << " mg " << mg);
    // The limit holds the box UP (-y) against its weight.
    CHECK_THAT(static_cast<double>(f.y), WithinRel(-mg, 0.02));
    CHECK_THAT(static_cast<double>(f.x), WithinAbs(0.0, 0.01 * mg));
}

TEST_CASE("Prismatic limit: with the limit off the box slides through where it would stop",
          "[physics][joints][prismatic][limit]")
{
    {
        Slide s(true, Vec2(Real(0), Real(1)), JointDef{});
        CHECK_FALSE(s.joint->IsLimitEnabled());
        Run(s.w, 60);
        CHECK(s.w.Position(s.box).y > Real(4)); // free fall: ~5 m in a second
    }
    {
        // Enabled, settled, then switched off live: it falls away.
        Slide s(true, Vec2(Real(0), Real(1)), Limited(Real(-1), Real(0.5)));
        Run(s.w, 60);
        REQUIRE_THAT(static_cast<double>(s.w.Position(s.box).y), WithinAbs(0.5, 0.003));
        s.joint->EnableLimit(false);
        Run(s.w, 60);
        CHECK(s.w.Position(s.box).y > Real(4));
    }
}

TEST_CASE("Prismatic limit: a strong motor driving into the upper limit stops there and stays",
          "[physics][joints][prismatic][limit][motor]")
{
    // Zero gravity, horizontal axis; the motor asks 3 m/s with 200 N on a 1 kg
    // box (20 g of push).
    JointDef j = Limited(Real(-0.5), Real(2));
    j.enableMotor = true;
    j.motorSpeed = Real(3);
    j.maxMotorForce = Real(200);
    Slide s(false, Vec2(Real(1), Real(0)), j);
    Run(s.w, 60); // reaches the limit within the first second
    Real lo = Real(9), hi = Real(-9), most = Real(-9);
    for (int i = 0; i < 300; ++i) // then five more seconds pushing into it
    {
        s.w.Step(kStep);
        const Real x = s.w.Position(s.box).x;
        lo = std::min(lo, x); hi = std::max(hi, x); most = std::max(most, x);
    }
    INFO("x in [" << lo << ", " << hi << "] v " << s.w.Velocity(s.box).x);
    CHECK_THAT(static_cast<double>(s.w.Position(s.box).x), WithinAbs(2.0, 0.002));
    CHECK(hi - lo < Real(1e-4));        // no drift while the motor keeps pushing
    CHECK(most < Real(2.002));
    CHECK_THAT(static_cast<double>(s.w.Velocity(s.box).x), WithinAbs(0.0, 1e-3));
    CHECK_THAT(static_cast<double>(s.w.Position(s.box).y), WithinAbs(0.0, 1e-4));

    // Reversed live, it drives back to the lower limit and stops there.
    s.joint->SetMotorSpeed(Real(-3));
    Run(s.w, 180);
    INFO("x " << s.w.Position(s.box).x);
    CHECK_THAT(static_cast<double>(s.w.Position(s.box).x), WithinAbs(-0.5, 0.002));
    CHECK_THAT(static_cast<double>(s.joint->GetTranslation(s.w)), WithinAbs(-0.5, 0.002));
}

TEST_CASE("Prismatic limit: a stalled motor sits in the soft limit by F / (m w^2), as in Box2D",
          "[physics][joints][prismatic][limit][motor]")
{
    // A motor stalled on the limit pushes with its full maxMotorForce F (it is
    // warm started into saturation, as Box2D's is); the breached limit holds it
    // at the joint constraint softness (60 Hz), so it rests inside the limit by
    // F / (m (2 pi 60)^2) -- 7 mm for 1000 N on 1 kg -- steady, not drifting.
    JointDef j = Limited(Real(-0.5), Real(2));
    j.enableMotor = true;
    j.motorSpeed = Real(3);
    j.maxMotorForce = Real(1000);
    Slide s(false, Vec2(Real(1), Real(0)), j);
    Run(s.w, 240);
    const Real x0 = s.w.Position(s.box).x;
    Run(s.w, 240);
    const double omega = 2.0 * 3.14159265358979 * 60.0;
    const double expected = 1000.0 / (double(s.mass) * omega * omega);
    INFO("x " << s.w.Position(s.box).x << " expected offset " << expected);
    CHECK_THAT(static_cast<double>(s.w.Position(s.box).x) - 2.0, WithinRel(expected, 0.05));
    CHECK(s.w.Position(s.box).x == x0); // steady
    // The motor delivers its full force into the limit, which cancels it.
    CHECK_THAT(static_cast<double>(s.w.JointReactionForce(s.joint).x), WithinAbs(0.0, 1.0));
}

TEST_CASE("Prismatic limit: a gentle motor rests on the limit with its force as the reaction",
          "[physics][joints][prismatic][limit][motor][reaction]")
{
    // A force-limited motor stalled against the limit delivers its full force
    // (as against a wall); the limit cancels it, so B feels the limit's push
    // and the motor's: the sum the reaction reports is ~0 along the axis.
    JointDef j = Limited(Real(-1), Real(1));
    j.enableMotor = true;
    j.motorSpeed = Real(2);
    j.maxMotorForce = Real(50);
    Slide s(false, Vec2(Real(1), Real(0)), j);
    Run(s.w, 240);
    CHECK_THAT(static_cast<double>(s.w.Position(s.box).x), WithinAbs(1.0, 0.003));
    const Vec2 f = s.w.JointReactionForce(s.joint);
    INFO("f " << f.x << ", " << f.y);
    CHECK_THAT(static_cast<double>(f.x), WithinAbs(0.0, 0.5));
}

TEST_CASE("Prismatic limit: GetTranslation follows the bodies' relative displacement along the axis",
          "[physics][joints][prismatic][limit]")
{
    // Two free bodies on a diagonal axis, A pushed one way and B the other.
    PhysicsWorld w{ [] { WorldDef d; d.gravityX = Real(0); d.gravityY = Real(0); return d; }() };
    BodyDef d; d.type = BodyType::Dynamic; d.shape = MakeAabb(Real(0.25), Real(0.25)); d.fixedRotation = true; d.density = Real(1);
    d.maskBits = 0u; d.sleepThreshold = Real(0);
    d.position = Vec2(Real(1), Real(2));
    const BodyHandle a = w.AddBody(d);
    d.position = Vec2(Real(3), Real(1));
    const BodyHandle b = w.AddBody(d);
    JointDef j; j.kind = JointKind::Prismatic; j.a = a; j.b = b; j.axis = Vec2(Real(3), Real(4)); // normalized to (0.6, 0.8)
    auto* pj = dynamic_cast<PrismaticJoint*>(w.AddJoint(j));
    REQUIRE(pj != nullptr);
    CHECK(pj->GetTranslation(w) == Real(0));
    w.SetVelocity(a, Vec2(Real(-0.6), Real(-0.8)));
    w.SetVelocity(b, Vec2(Real(1.2), Real(1.6)));
    const Vec2 a0 = w.Position(a), b0 = w.Position(b);
    Run(w, 30);
    const Vec2 a1 = w.Position(a), b1 = w.Position(b);
    const double moved = (double(b1.x - b0.x) - double(a1.x - a0.x)) * 0.6
                       + (double(b1.y - b0.y) - double(a1.y - a0.y)) * 0.8;
    INFO("moved " << moved << " translation " << pj->GetTranslation(w));
    CHECK(moved > 0.5); // the bodies really did slide apart
    CHECK_THAT(static_cast<double>(pj->GetTranslation(w)), WithinAbs(moved, 1e-4));
}

TEST_CASE("Prismatic limit: a box started past its limit is pushed back to it",
          "[physics][joints][prismatic][limit]")
{
    // Gravity pushes it further past the upper side; the soft push-out wins.
    Slide s(true, Vec2(Real(0), Real(1)), Limited(Real(-1), Real(-0.3)));
    Run(s.w, 60);
    INFO("y " << s.w.Position(s.box).y);
    CHECK_THAT(static_cast<double>(s.w.Position(s.box).y), WithinAbs(-0.3, 0.003));
}

TEST_CASE("Prismatic limit: the live setters sort the bounds and the def fields reach the joint",
          "[physics][joints][prismatic][limit]")
{
    Slide s(true, Vec2(Real(0), Real(1)), Limited(Real(0.5), Real(-1))); // given reversed
    CHECK(s.joint->GetLowerLimit() == Real(-1));
    CHECK(s.joint->GetUpperLimit() == Real(0.5));
    s.joint->SetLimits(Real(0.2), Real(-0.4));
    CHECK(s.joint->GetLowerLimit() == Real(-0.4));
    CHECK(s.joint->GetUpperLimit() == Real(0.2));
    Run(s.w, 120);
    CHECK_THAT(static_cast<double>(s.w.Position(s.box).y), WithinAbs(0.2, 0.003));
}
