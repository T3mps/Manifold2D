// PhysicsJointReactionTest.cpp
// [physics][joints][reaction]: the joint reaction query (Box2D v3
// b2Joint_GetConstraintForce / b2Joint_GetConstraintTorque) --
// PhysicsWorld::JointReactionForce / JointReactionTorque.
//
// The reading is the force (N, world frame) and the pure torque (N m) the
// joint delivered to body B over the last step: the summed impulse / dt (see
// Joint::ReactionForce). The oracles are statics: a load at rest is in
// equilibrium, so whatever the joint carries must cancel gravity on B (and,
// for a torque-carrying joint, gravity's moment about the anchor); a motor
// stalled at its limit delivers exactly its limit.
//
// Gravity is (0, 10), +y DOWN: a load hanging from a joint is held by a force
// pointing UP (-y) on it.
//
// PRESENTATION-FREE, ASCII comments, C++23.
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
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace
{
    constexpr Real kStep = Real(1) / Real(60);
    constexpr double kG = 10.0; // WorldDef's default gravity magnitude

    // A static pin that touches nothing (only the joint holds the load).
    BodyHandle AddPin(PhysicsWorld& w, Vec2 pos)
    {
        BodyDef s;
        s.type = BodyType::Static;
        s.shape = MakeCircle(Real(0.05));
        s.position = pos;
        s.maskBits = 0u;
        return w.AddBody(s);
    }

    // A dynamic ball that touches nothing and never sleeps (the joint is
    // solved every step, so the reading is always this step's).
    BodyHandle AddBall(PhysicsWorld& w, Vec2 pos, Real r = Real(0.25))
    {
        BodyDef d;
        d.type = BodyType::Dynamic;
        d.shape = MakeCircle(r);
        d.density = Real(1);
        d.position = pos;
        d.maskBits = 0u;
        d.sleepThreshold = Real(0);
        return w.AddBody(d);
    }

    void Run(PhysicsWorld& w, int steps)
    {
        for (int i = 0; i < steps; ++i) { w.Step(kStep); }
    }

    double Len(Vec2 v) { return std::sqrt(double(v.x) * double(v.x) + double(v.y) * double(v.y)); }
} // namespace

TEST_CASE("JointReaction: a weld holding a ball straight below its anchor carries m g and no torque",
          "[physics][joints][reaction]")
{
    PhysicsWorld w{ WorldDef{} };
    const BodyHandle pin = AddPin(w, Vec2(Real(0), Real(0)));
    const BodyHandle ball = AddBall(w, Vec2(Real(0), Real(1)));
    JointDef j; j.kind = JointKind::Weld; j.a = pin; j.b = ball; j.anchor = Vec2(Real(0), Real(0));
    const Joint* weld = w.AddJoint(j);
    REQUIRE(weld != nullptr);
    Run(w, 120);
    const double mg = double(w.GetBodyMass(ball)) * kG;
    const Vec2 f = w.JointReactionForce(weld);
    CHECK_THAT(static_cast<double>(f.y), WithinRel(-mg, 0.02)); // up, on the ball
    CHECK_THAT(static_cast<double>(f.x), WithinAbs(0.0, 0.01 * mg));
    CHECK_THAT(static_cast<double>(w.JointReactionTorque(weld)), WithinAbs(0.0, 0.01 * mg));
    // the per-joint reading is the same number
    CHECK(weld->ReactionForce().y == f.y);
}

TEST_CASE("JointReaction: a weld holding a ball offset by d carries m g and the moment m g d",
          "[physics][joints][reaction]")
{
    PhysicsWorld w{ WorldDef{} };
    constexpr Real d = Real(0.75);
    const BodyHandle pin = AddPin(w, Vec2(Real(0), Real(0)));
    const BodyHandle ball = AddBall(w, Vec2(d, Real(0))); // a cantilever, level with the anchor
    JointDef j; j.kind = JointKind::Weld; j.a = pin; j.b = ball; j.anchor = Vec2(Real(0), Real(0));
    const Joint* weld = w.AddJoint(j);
    REQUIRE(weld != nullptr);
    Run(w, 180);
    const double mg = double(w.GetBodyMass(ball)) * kG;
    const Vec2 f = w.JointReactionForce(weld);
    CHECK_THAT(Len(f), WithinRel(mg, 0.03));
    CHECK_THAT(static_cast<double>(f.y), WithinRel(-mg, 0.03));
    // Equilibrium about the ball's centre: the anchor force's moment rB x F plus
    // the weld's pure torque is zero. rB = anchor - centre = (-d, 0), F = (0, -m g)
    // -> rB x F = d m g, so the weld's torque on the ball is -m g d.
    const double t = static_cast<double>(w.JointReactionTorque(weld));
    CHECK_THAT(t, WithinRel(-mg * double(d), 0.03));
}

TEST_CASE("JointReaction: a revolute pendulum at rest carries m g and no torque",
          "[physics][joints][reaction]")
{
    PhysicsWorld w{ WorldDef{} };
    const BodyHandle pin = AddPin(w, Vec2(Real(0), Real(0)));
    const BodyHandle bob = AddBall(w, Vec2(Real(0), Real(2)));
    JointDef j; j.kind = JointKind::Revolute; j.a = pin; j.b = bob; j.anchor = Vec2(Real(0), Real(0));
    const Joint* rev = w.AddJoint(j);
    REQUIRE(rev != nullptr);
    Run(w, 120);
    const double mg = double(w.GetBodyMass(bob)) * kG;
    const Vec2 f = w.JointReactionForce(rev);
    CHECK_THAT(static_cast<double>(f.y), WithinRel(-mg, 0.02));
    CHECK_THAT(static_cast<double>(f.x), WithinAbs(0.0, 0.01 * mg));
    CHECK(w.JointReactionTorque(rev) == Real(0)); // no spring, motor or limit: no angular impulse
}

TEST_CASE("JointReaction: a distance joint and a mouse joint each carry the weight they hold",
          "[physics][joints][reaction]")
{
    PhysicsWorld w{ WorldDef{} };
    const BodyHandle pin = AddPin(w, Vec2(Real(0), Real(0)));
    const BodyHandle hung = AddBall(w, Vec2(Real(0), Real(1.5)));
    JointDef dj; dj.kind = JointKind::Distance; dj.a = pin; dj.b = hung;
    const Joint* dist = w.AddJoint(dj);
    const BodyHandle held = AddBall(w, Vec2(Real(3), Real(0)));
    JointDef mj; mj.kind = JointKind::Mouse; mj.b = held; mj.target = Vec2(Real(3), Real(0)); mj.maxForce = Real(1000);
    const Joint* mouse = w.AddJoint(mj);
    REQUIRE(dist != nullptr);
    REQUIRE(mouse != nullptr);
    Run(w, 180);
    const double mgD = double(w.GetBodyMass(hung)) * kG;
    const double mgM = double(w.GetBodyMass(held)) * kG;
    CHECK_THAT(static_cast<double>(w.JointReactionForce(dist).y), WithinRel(-mgD, 0.02));
    CHECK_THAT(static_cast<double>(w.JointReactionForce(mouse).y), WithinRel(-mgM, 0.02));
    CHECK(w.JointReactionTorque(dist) == Real(0));
    CHECK(w.JointReactionTorque(mouse) == Real(0));
}

TEST_CASE("JointReaction: a prismatic motor stalled on a wall reports its maxMotorForce along the axis",
          "[physics][joints][reaction][prismatic]")
{
    PhysicsWorld w{ [] { WorldDef d; d.gravityX = Real(0); d.gravityY = Real(0); return d; }() };
    const BodyHandle frame = AddPin(w, Vec2(Real(-3), Real(0)));
    BodyDef wall; wall.type = BodyType::Static; wall.shape = MakeAabb(Real(0.5), Real(2)); wall.position = Vec2(Real(2), Real(0));
    w.AddBody(wall);
    BodyDef d; d.type = BodyType::Dynamic; d.shape = MakeAabb(Real(0.5), Real(0.5)); d.fixedRotation = true; d.density = Real(1);
    d.position = Vec2(Real(0), Real(0)); d.sleepThreshold = Real(0);
    const BodyHandle ram = w.AddBody(d);
    constexpr Real F = Real(40);
    JointDef j; j.kind = JointKind::Prismatic; j.a = frame; j.b = ram; j.axis = Vec2(Real(1), Real(0));
    j.enableMotor = true; j.motorSpeed = Real(4); j.maxMotorForce = F;
    const Joint* pj = w.AddJoint(j);
    REQUIRE(pj != nullptr);
    Run(w, 180);
    REQUIRE_THAT(static_cast<double>(w.Position(ram).x), WithinAbs(1.0, 0.05)); // stalled on the wall
    const Vec2 f = w.JointReactionForce(pj);
    CHECK_THAT(static_cast<double>(f.x), WithinRel(double(F), 0.02));
    CHECK_THAT(static_cast<double>(f.y), WithinAbs(0.0, 0.01 * double(F)));
    CHECK_THAT(static_cast<double>(w.JointReactionTorque(pj)), WithinAbs(0.0, 1e-6));
}

TEST_CASE("JointReaction: a wheel joint carries the weight of the chassis it supports",
          "[physics][joints][reaction][wheel]")
{
    PhysicsWorld w{ WorldDef{} };
    BodyDef g; g.type = BodyType::Static; g.shape = MakeAabb(Real(20), Real(0.5)); g.position = Vec2(Real(0), Real(0.5));
    w.AddBody(g); // top face at y = 0
    BodyDef wd; wd.type = BodyType::Dynamic; wd.shape = MakeCircle(Real(0.5)); wd.density = Real(1); wd.friction = Real(0.8);
    wd.position = Vec2(Real(0), Real(-0.5)); wd.sleepThreshold = Real(0);
    const BodyHandle wheel = w.AddBody(wd);
    // The chassis touches nothing (only the suspension holds it up) and keeps
    // its axis vertical (fixed rotation): the suspension carries its weight.
    BodyDef cd; cd.type = BodyType::Dynamic; cd.shape = MakeAabb(Real(0.6), Real(0.2)); cd.density = Real(2);
    cd.position = Vec2(Real(0), Real(-1.5)); cd.fixedRotation = true; cd.maskBits = 0u; cd.sleepThreshold = Real(0);
    const BodyHandle chassis = w.AddBody(cd);
    JointDef j; j.kind = JointKind::Wheel; j.a = chassis; j.b = wheel;
    j.anchor = Vec2(Real(0), Real(-0.5)); j.axis = Vec2(Real(0), Real(1));
    j.frequencyHz = Real(4); j.dampingRatio = Real(0.7);
    const Joint* wj = w.AddJoint(j);
    REQUIRE(wj != nullptr);
    Run(w, 240);
    const double Mg = double(w.GetBodyMass(chassis)) * kG;
    const Vec2 f = w.JointReactionForce(wj);
    // The joint presses the wheel DOWN (+y) with the chassis's weight.
    CHECK_THAT(static_cast<double>(f.y), WithinRel(Mg, 0.03));
    CHECK_THAT(static_cast<double>(f.x), WithinAbs(0.0, 0.01 * Mg));
    CHECK_THAT(static_cast<double>(w.JointReactionTorque(wj)), WithinAbs(0.0, 1e-6)); // no motor
}

TEST_CASE("JointReaction: a motor joint driving toward an unreachable speed reports its maxMotorTorque",
          "[physics][joints][reaction][motor]")
{
    PhysicsWorld w{ [] { WorldDef d; d.gravityX = Real(0); d.gravityY = Real(0); return d; }() };
    const BodyHandle hub = AddPin(w, Vec2(Real(0), Real(0)));
    const BodyHandle disc = AddBall(w, Vec2(Real(0), Real(0)), Real(0.5));
    JointDef pin; pin.kind = JointKind::Revolute; pin.a = hub; pin.b = disc; pin.anchor = Vec2(Real(0), Real(0));
    w.AddJoint(pin);
    JointDef m; m.kind = JointKind::Motor; m.a = hub; m.b = disc; m.motorSpeed = Real(1000); m.maxMotorTorque = Real(0.2);
    const Joint* motor = w.AddJoint(m);
    REQUIRE(motor != nullptr);
    Run(w, 10);
    CHECK_THAT(static_cast<double>(w.JointReactionTorque(motor)), WithinRel(0.2, 1e-4));
    CHECK(w.JointReactionForce(motor).x == Real(0));
    CHECK(w.JointReactionForce(motor).y == Real(0));
}

TEST_CASE("JointReaction: a joint the world does not own, or never solved, reads zero",
          "[physics][joints][reaction]")
{
    PhysicsWorld w{ WorldDef{} };
    CHECK(w.JointReactionForce(nullptr).x == Real(0));
    CHECK(w.JointReactionForce(nullptr).y == Real(0));
    CHECK(w.JointReactionTorque(nullptr) == Real(0));

    const BodyHandle pin = AddPin(w, Vec2(Real(0), Real(0)));
    const BodyHandle ball = AddBall(w, Vec2(Real(0), Real(1)));
    JointDef j; j.kind = JointKind::Weld; j.a = pin; j.b = ball; j.anchor = Vec2(Real(0), Real(0));
    Joint* weld = w.AddJoint(j);
    REQUIRE(weld != nullptr);
    CHECK(w.JointReactionForce(weld).y == Real(0)); // added, not yet stepped
    Run(w, 60);
    CHECK(w.JointReactionForce(weld).y < Real(0));  // carrying the ball

    // A second world's joint is not this world's.
    PhysicsWorld other{ WorldDef{} };
    CHECK(other.JointReactionForce(weld).y == Real(0));

    w.RemoveJoint(weld); // the pointer now matches nothing
    CHECK(w.JointCount() == 0u);
    CHECK(w.JointReactionForce(weld).x == Real(0));
    CHECK(w.JointReactionForce(weld).y == Real(0));
    CHECK(w.JointReactionTorque(weld) == Real(0));
    Run(w, 1); // and stepping without it is fine
}
