// PhysicsRevoluteLimitTest.cpp
// [physics][joints][revolute]: the revolute joint's angle limits, rotational
// spring and motor (Box2D v3 b2RevoluteJoint: revolute_joint.c
// b2SolveRevoluteJoint, b2RevoluteJointDef).
//
//   * joint angle = angleB - angleA - referenceAngle (Box2D's relative angle of
//     the joint frames; referenceAngle plays the frames' rotation).
//   * limit: lowerAngle <= joint angle <= upperAngle. Each side is a one-sided
//     constraint: speculative while open (bias C / h), soft when breached
//     (constraintSoftness: 60 Hz clamped to 0.25 / h, damping ratio 2), the
//     soft push-out only on the biased pass (Box2D's useBias).
//   * spring: a soft constraint driving the joint angle to targetAngle at
//     (hertz, dampingRatio).
//   * motor: drives the relative angular velocity to motorSpeed, its impulse
//     clamped to h * maxMotorTorque per sub-step (a torque in N m).
//
// The joint tracks its own angle through the step's sub-steps (Box2D reads the
// bodies' deltaRotation; Manifold2D's joints see only velocities, so the joint
// integrates the relative angular velocity the solver integrated).
//
// Engine convention: +Y is DOWN, gravity +Y; a positive angle turns +x toward +y.
//
// PRESENTATION-FREE, ASCII comments, C++23.
#include <cmath>
#include <vector>

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

    Shape Box(Real hx, Real hy)
    {
        return MakePolygon(std::vector<Vec2>{ Vec2(-hx, -hy), Vec2(hx, -hy), Vec2(hx, hy), Vec2(-hx, hy) });
    }

    WorldDef World(bool gravity)
    {
        WorldDef wd;
        if (!gravity) { wd.gravityX = Real(0); wd.gravityY = Real(0); }
        return wd;
    }

    struct Hinge { BodyHandle base, bar; RevoluteJoint* joint; };

    // A 1 m bar (centre at x = 0.5) hinged at the origin to a static base,
    // lying along +x (joint angle 0). Gravity swings its free end down (+y),
    // i.e. toward a joint angle of +pi/2.
    Hinge MakeHinge(PhysicsWorld& w, JointDef jd)
    {
        BodyDef b;
        b.type = BodyType::Static;
        b.shape = MakeCircle(Real(0.05));
        b.maskBits = 0u; // the pin: it must not touch the bar's end (jointed bodies still collide here)
        const BodyHandle base = w.AddBody(b);
        BodyDef d;
        d.type = BodyType::Dynamic;
        d.position = Vec2(Real(0.5), Real(0));
        d.shape = Box(Real(0.5), Real(0.05));
        const BodyHandle bar = w.AddBody(d);
        jd.kind = JointKind::Revolute;
        jd.a = base;
        jd.b = bar;
        jd.anchor = Vec2(Real(0), Real(0));
        auto* j = dynamic_cast<RevoluteJoint*>(w.AddJoint(jd));
        REQUIRE(j != nullptr);
        return { base, bar, j };
    }
} // namespace

TEST_CASE("Revolute: an upper limit stops a falling bar at the limit angle",
          "[physics][joints][revolute]")
{
    PhysicsWorld w{ World(true) };
    JointDef jd;
    jd.enableLimit = true;
    jd.lowerAngle = Real(-0.5);
    jd.upperAngle = Real(0.5);
    const Hinge h = MakeHinge(w, jd);
    Real most = Real(-9);
    for (int i = 0; i < 180; ++i)
    {
        w.Step(kStep);
        most = std::max(most, w.GetAngle(h.bar));
    }
    INFO("final " << w.GetAngle(h.bar) << " most " << most);
    CHECK(std::abs(w.GetAngle(h.bar) - Real(0.5)) < Real(0.03)); // rests on the limit
    CHECK(most < Real(0.56));                                     // barely overshoots on the way
    CHECK(h.joint->JointAngle(w) == w.GetAngle(h.bar));           // (base static, unrotated, ref 0)
}

TEST_CASE("Revolute: without the limit the same bar swings through it",
          "[physics][joints][revolute]")
{
    PhysicsWorld w{ World(true) };
    const Hinge h = MakeHinge(w, JointDef{});
    Real most = Real(-9);
    for (int i = 0; i < 60; ++i) { w.Step(kStep); most = std::max(most, w.GetAngle(h.bar)); }
    CHECK(most > Real(1.4));
}

TEST_CASE("Revolute: a bar started past its limit is pushed back inside it",
          "[physics][joints][revolute]")
{
    PhysicsWorld w{ World(true) };
    JointDef jd;
    jd.enableLimit = true;
    jd.lowerAngle = Real(-1.2);
    jd.upperAngle = Real(-0.4); // gravity pushes it up against this side
    const Hinge h = MakeHinge(w, jd);  // starts at 0: 0.4 rad past the upper limit
    for (int i = 0; i < 60; ++i) { w.Step(kStep); }
    INFO("angle " << w.GetAngle(h.bar));
    CHECK(std::abs(w.GetAngle(h.bar) + Real(0.4)) < Real(0.03));
}

TEST_CASE("Revolute: referenceAngle makes the limits relative to a rotated rest pose",
          "[physics][joints][revolute]")
{
    PhysicsWorld w{ World(true) };
    JointDef jd;
    jd.enableLimit = true;
    jd.referenceAngle = Real(0.3);
    jd.lowerAngle = Real(-0.2);
    jd.upperAngle = Real(0.2); // body angle in [0.1, 0.5]
    const Hinge h = MakeHinge(w, jd);
    for (int i = 0; i < 180; ++i) { w.Step(kStep); }
    INFO("angle " << w.GetAngle(h.bar) << " joint " << h.joint->JointAngle(w));
    CHECK(std::abs(w.GetAngle(h.bar) - Real(0.5)) < Real(0.03));
    CHECK(std::abs(h.joint->JointAngle(w) - Real(0.2)) < Real(0.03));
}

TEST_CASE("Revolute: the spring drives the joint to its target angle and settles",
          "[physics][joints][revolute]")
{
    PhysicsWorld w{ World(false) };
    JointDef jd;
    jd.enableSpring = true;
    jd.frequencyHz = Real(2);
    jd.dampingRatio = Real(1);
    jd.targetAngle = Real(1);
    const Hinge h = MakeHinge(w, jd);
    for (int i = 0; i < 180; ++i) { w.Step(kStep); }
    INFO("angle " << w.GetAngle(h.bar));
    CHECK(std::abs(w.GetAngle(h.bar) - Real(1)) < Real(0.02));
    CHECK(std::abs(w.AngularVelocity(h.bar)) < Real(0.05));
}

TEST_CASE("Revolute: a soft spring holds a pose against gravity with a sag",
          "[physics][joints][revolute]")
{
    // The spring is a torque, not a lock: under gravity a 1 Hz spring holding the bar
    // level sags below the target, a 6 Hz one barely.
    const auto sag = [](Real hz)
    {
        PhysicsWorld w{ World(true) };
        JointDef jd;
        jd.enableSpring = true;
        jd.frequencyHz = hz;
        jd.dampingRatio = Real(1);
        jd.targetAngle = Real(0);
        const Hinge h = MakeHinge(w, jd);
        for (int i = 0; i < 240; ++i) { w.Step(kStep); }
        return w.GetAngle(h.bar);
    };
    const Real soft = sag(Real(1)), stiff = sag(Real(6));
    INFO("1 Hz " << soft << " 6 Hz " << stiff);
    CHECK(soft > stiff);
    CHECK(stiff > Real(0));
    CHECK(stiff < Real(0.15));
}

TEST_CASE("Revolute: the motor drives the relative angular velocity to motorSpeed",
          "[physics][joints][revolute]")
{
    PhysicsWorld w{ World(false) };
    JointDef jd;
    jd.enableMotor = true;
    jd.motorSpeed = Real(2);
    jd.maxMotorTorque = Real(100);
    const Hinge h = MakeHinge(w, jd);
    for (int i = 0; i < 60; ++i) { w.Step(kStep); }
    CHECK(std::abs(w.AngularVelocity(h.bar) - Real(2)) < Real(0.02));
}

TEST_CASE("Revolute: a weak motor stalls against gravity, a strong one lifts",
          "[physics][joints][revolute]")
{
    // The bar's weight about the hinge is m g L/2 = (1 * 0.1 kg/m^2 ... ) -- measured
    // through the motor budget: a torque below it lets the bar fall, above it lifts.
    const auto run = [](Real torque)
    {
        PhysicsWorld w{ World(true) };
        JointDef jd;
        jd.enableMotor = true;
        jd.motorSpeed = Real(-1); // up (toward -y)
        jd.maxMotorTorque = torque;
        const Hinge h = MakeHinge(w, jd);
        for (int i = 0; i < 60; ++i) { w.Step(kStep); }
        return w.GetAngle(h.bar);
    };
    // mass = 1.0 * 0.1 = 0.1 kg (density 1, 1 x 0.1 m); weight torque = 0.1 * 10 * 0.5 = 0.5 N m
    CHECK(run(Real(0.2)) > Real(0.5));  // falls
    CHECK(run(Real(2)) < Real(-0.5));   // lifts
}

TEST_CASE("Revolute: live control -- limits and spring switched on mid-run",
          "[physics][joints][revolute]")
{
    PhysicsWorld w{ World(true) };
    const Hinge h = MakeHinge(w, JointDef{});
    for (int i = 0; i < 30; ++i) { w.Step(kStep); } // falls freely
    h.joint->EnableSpring(true);
    h.joint->SetSpring(Real(4), Real(1));
    h.joint->SetTargetAngle(Real(-0.6));
    h.joint->EnableLimit(true);
    h.joint->SetLimits(Real(-0.5), Real(1.0));
    w.Wake(h.bar);
    for (int i = 0; i < 240; ++i) { w.Step(kStep); }
    // the spring pulls up toward -0.6, the lower limit stops it at -0.5
    INFO("angle " << w.GetAngle(h.bar));
    CHECK(std::abs(w.GetAngle(h.bar) + Real(0.5)) < Real(0.03));
}

TEST_CASE("Revolute: a limited chain (a ragdoll arm) falls, lands and settles finite and inside its limits",
          "[physics][joints][revolute]")
{
    const auto run = []()
    {
        PhysicsWorld w{ World(true) };
        BodyDef g;
        g.type = BodyType::Static;
        g.position = Vec2(Real(0), Real(1.5));
        g.shape = MakeAabb(Real(10), Real(0.5));
        w.AddBody(g);
        std::vector<BodyHandle> links;
        std::vector<RevoluteJoint*> joints;
        for (int i = 0; i < 6; ++i)
        {
            BodyDef d;
            d.type = BodyType::Dynamic;
            d.position = Vec2(Real(0.4) * Real(i), Real(-1));
            d.shape = Box(Real(0.2), Real(0.06));
            links.push_back(w.AddBody(d));
            if (i > 0)
            {
                JointDef jd;
                jd.kind = JointKind::Revolute;
                jd.a = links[i - 1];
                jd.b = links[i];
                jd.anchor = Vec2(Real(0.4) * Real(i) - Real(0.2), Real(-1));
                jd.enableLimit = true;
                jd.lowerAngle = Real(-0.6);
                jd.upperAngle = Real(0.6);
                joints.push_back(dynamic_cast<RevoluteJoint*>(w.AddJoint(jd)));
            }
        }
        w.SetAngularVelocity(links[0], Real(-6));
        for (int i = 0; i < 300; ++i) { w.Step(kStep); }
        std::vector<Real> out;
        for (std::size_t k = 0; k < joints.size(); ++k)
        {
            const Real a = joints[k]->JointAngle(w);
            CHECK(std::isfinite(a));
            CHECK(a > Real(-0.65));
            CHECK(a < Real(0.65));
        }
        for (const BodyHandle b : links) { out.push_back(w.Position(b).x); out.push_back(w.Position(b).y); }
        return out;
    };
    CHECK(run() == run());
}
