// PhysicsSetFilterPrismaticMotorTest.cpp
// [physics][filter][setfilter] and [physics][joints][prismatic][motor]:
//
// SetBodyFilter (b2Shape_SetFilter): a live body's collision filter changes
// mid-simulation. Contacts the new filter rejects end at once; pairs it admits
// are found on the next step -- including the static ground, whose contacts
// the reset also drops.
//
// The prismatic motor (b2PrismaticJoint's): drives the slide toward a speed
// with at most maxMotorForce, the full budget in every sub-step, and stalls
// against a load it cannot move instead of forcing through it.
//
// PRESENTATION-FREE, ASCII comments, C++23.
#include <cmath>
#include <cstdint>

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
    constexpr std::uint32_t kGround = 1u, kLaneA = 2u, kLaneB = 4u;

    BodyHandle AddGround(PhysicsWorld& w)
    {
        BodyDef g;
        g.type = BodyType::Static;
        g.shape = MakeAabb(Real(20), Real(0.5));
        g.position = Vec2(Real(0), Real(0.5)); // top face at y = 0 (+y down)
        g.categoryBits = kGround;
        return w.AddBody(g);
    }

    BodyHandle AddBox(PhysicsWorld& w, Vec2 pos, std::uint32_t cat, std::uint32_t mask)
    {
        BodyDef d;
        d.type = BodyType::Dynamic;
        d.shape = MakeAabb(Real(0.5), Real(0.5));
        d.fixedRotation = true; // a dynamic AABB shape is axis-aligned by definition
        d.density = Real(1);
        d.friction = Real(0.6);
        d.position = pos;
        d.categoryBits = cat;
        d.maskBits = mask;
        return w.AddBody(d);
    }
} // namespace

TEST_CASE("SetBodyFilter: a body stacked on another falls through once the filter rejects it",
          "[physics][filter][setfilter]")
{
    PhysicsWorld w{ WorldDef{} };
    AddGround(w);
    const BodyHandle low = AddBox(w, Vec2(Real(0), Real(-0.5)), kLaneA, kGround | kLaneA);
    const BodyHandle top = AddBox(w, Vec2(Real(0), Real(-1.5)), kLaneA, kGround | kLaneA);
    for (int i = 0; i < 90; ++i) { w.Step(kStep); }
    REQUIRE_THAT(static_cast<double>(w.Position(top).y), WithinAbs(-1.5, 0.05)); // stacked

    w.SetBodyFilter(top, kLaneB, kGround | kLaneB); // other lane: no longer meets `low`
    for (int i = 0; i < 90; ++i) { w.Step(kStep); }
    // It passed through `low` and now rests on the ground beside it in depth.
    CHECK_THAT(static_cast<double>(w.Position(top).y), WithinAbs(-0.5, 0.05));
    CHECK_THAT(static_cast<double>(w.Position(low).y), WithinAbs(-0.5, 0.05));
}

TEST_CASE("SetBodyFilter: two overlapping bodies separate once the filter admits the pair, and keep the ground",
          "[physics][filter][setfilter]")
{
    PhysicsWorld w{ WorldDef{} };
    AddGround(w);
    // Different lanes, resting on the ground, half-overlapping in x.
    const BodyHandle a = AddBox(w, Vec2(Real(0), Real(-0.5)), kLaneA, kGround | kLaneA);
    const BodyHandle b = AddBox(w, Vec2(Real(0.5), Real(-0.5)), kLaneB, kGround | kLaneB);
    for (int i = 0; i < 60; ++i) { w.Step(kStep); }
    REQUIRE_THAT(static_cast<double>(w.Position(b).x - w.Position(a).x), WithinAbs(0.5, 0.02)); // ghosts

    w.SetBodyFilter(b, kLaneA, kGround | kLaneA); // same lane now
    for (int i = 0; i < 120; ++i) { w.Step(kStep); }
    const Vec2 pa = w.Position(a), pb = w.Position(b);
    CHECK(std::abs(pb.x - pa.x) > Real(0.95));        // pushed apart to touching
    CHECK(std::isfinite(pa.x));
    CHECK(std::isfinite(pb.x));
    // Neither fell through the ground: its static contacts were re-found.
    CHECK_THAT(static_cast<double>(pa.y), WithinAbs(-0.5, 0.06));
    CHECK_THAT(static_cast<double>(pb.y), WithinAbs(-0.5, 0.06));
}

TEST_CASE("SetBodyFilter: a sleeping body is woken and an unchanged filter is a no-op",
          "[physics][filter][setfilter]")
{
    PhysicsWorld w{ WorldDef{} };
    AddGround(w);
    const BodyHandle a = AddBox(w, Vec2(Real(0), Real(-0.5)), kLaneA, kGround | kLaneA);
    for (int i = 0; i < 300; ++i) { w.Step(kStep); }
    REQUIRE_FALSE(w.IsAwake(a));
    w.SetBodyFilter(a, kLaneA, kGround | kLaneA); // same filter
    CHECK_FALSE(w.IsAwake(a));
    w.SetBodyFilter(a, kLaneB, kGround | kLaneB);
    CHECK(w.IsAwake(a));
    for (int i = 0; i < 60; ++i) { w.Step(kStep); }
    CHECK_THAT(static_cast<double>(w.Position(a).y), WithinAbs(-0.5, 0.05));
}

namespace
{
    // A static rail and a 1 kg-ish box on a prismatic joint along `axis`.
    struct Slide
    {
        PhysicsWorld w;
        BodyHandle rail, box;
        PrismaticJoint* joint = nullptr;
        Real mass = Real(0);
        Slide(Real gravityY, Vec2 axis, bool motor, Real speed, Real force)
            : w([gravityY] { WorldDef d; d.gravityX = Real(0); d.gravityY = gravityY; return d; }())
        {
            BodyDef s; s.type = BodyType::Static; s.shape = MakeCircle(Real(0.05)); s.position = Vec2(Real(0), Real(0));
            s.maskBits = 0u;
            rail = w.AddBody(s);
            BodyDef d; d.type = BodyType::Dynamic; d.shape = MakeAabb(Real(0.5), Real(0.5)); d.fixedRotation = true; d.density = Real(1);
            d.position = Vec2(Real(0), Real(0)); d.maskBits = 0u; d.sleepThreshold = Real(0);
            box = w.AddBody(d);
            mass = w.GetBodyMass(box);
            JointDef j; j.kind = JointKind::Prismatic; j.a = rail; j.b = box; j.axis = axis;
            j.enableMotor = motor; j.motorSpeed = speed; j.maxMotorForce = force;
            joint = dynamic_cast<PrismaticJoint*>(w.AddJoint(j));
        }
    };
} // namespace

TEST_CASE("Prismatic motor: delivers its full maxMotorForce from rest, every sub-step",
          "[physics][joints][prismatic][motor]")
{
    Slide s(Real(0), Vec2(Real(1), Real(0)), true, Real(1000), Real(3));
    REQUIRE(s.joint != nullptr);
    s.w.Step(kStep);
    const double expected = 3.0 * (1.0 / 60.0) / double(s.mass); // force-limited: v = F dt / m
    CHECK_THAT(static_cast<double>(s.w.Velocity(s.box).x), WithinRel(expected, 0.02));
    CHECK_THAT(static_cast<double>(s.w.Velocity(s.box).y), WithinAbs(0.0, 1e-6));
}

TEST_CASE("Prismatic motor: settles at its speed along the axis",
          "[physics][joints][prismatic][motor]")
{
    Slide s(Real(0), Vec2(Real(1), Real(1)), true, Real(2), Real(100));
    for (int i = 0; i < 60; ++i) { s.w.Step(kStep); }
    const Vec2 v = s.w.Velocity(s.box);
    const double r = std::sqrt(0.5);
    CHECK_THAT(static_cast<double>(v.x), WithinRel(2.0 * r, 0.01));
    CHECK_THAT(static_cast<double>(v.y), WithinRel(2.0 * r, 0.01));
}

TEST_CASE("Prismatic motor: a force below the load's weight loses, above it lifts",
          "[physics][joints][prismatic][motor]")
{
    // Vertical slide under gravity 10 (+y down); the motor asks for 1 m/s UP.
    {
        Slide weak(Real(10), Vec2(Real(0), Real(1)), true, Real(-1), Real(0));
        weak.joint->SetMaxMotorForce(weak.mass * Real(5)); // half the weight
        for (int i = 0; i < 60; ++i) { weak.w.Step(kStep); }
        CHECK(weak.w.Velocity(weak.box).y > Real(4)); // falls at about g/2
        CHECK(weak.w.Velocity(weak.box).y < Real(6));
    }
    {
        Slide strong(Real(10), Vec2(Real(0), Real(1)), true, Real(-1), Real(0));
        strong.joint->SetMaxMotorForce(strong.mass * Real(20));
        for (int i = 0; i < 60; ++i) { strong.w.Step(kStep); }
        CHECK_THAT(static_cast<double>(strong.w.Velocity(strong.box).y), WithinRel(-1.0, 0.02));
    }
}

TEST_CASE("Prismatic motor: a ram stalls against a wall instead of forcing through it",
          "[physics][joints][prismatic][motor]")
{
    PhysicsWorld w{ [] { WorldDef d; d.gravityX = Real(0); d.gravityY = Real(0); return d; }() };
    BodyDef s; s.type = BodyType::Static; s.shape = MakeCircle(Real(0.05)); s.position = Vec2(Real(-3), Real(0)); s.maskBits = 0u;
    const BodyHandle frame = w.AddBody(s);
    BodyDef wall; wall.type = BodyType::Static; wall.shape = MakeAabb(Real(0.5), Real(2)); wall.position = Vec2(Real(2), Real(0));
    w.AddBody(wall);
    BodyDef d; d.type = BodyType::Dynamic; d.shape = MakeAabb(Real(0.5), Real(0.5)); d.fixedRotation = true; d.density = Real(1);
    d.position = Vec2(Real(0), Real(0)); d.sleepThreshold = Real(0);
    const BodyHandle ram = w.AddBody(d);
    JointDef j; j.kind = JointKind::Prismatic; j.a = frame; j.b = ram; j.axis = Vec2(Real(1), Real(0));
    j.enableMotor = true; j.motorSpeed = Real(4); j.maxMotorForce = Real(40);
    auto* pj = dynamic_cast<PrismaticJoint*>(w.AddJoint(j));
    REQUIRE(pj != nullptr);
    for (int i = 0; i < 180; ++i) { w.Step(kStep); }
    // The wall's face is at x = 1.5, the ram's half-width 0.5: it stalls at x ~ 1.
    CHECK_THAT(static_cast<double>(w.Position(ram).x), WithinAbs(1.0, 0.05));
    CHECK_THAT(static_cast<double>(w.Velocity(ram).x), WithinAbs(0.0, 0.05));
    // Reversed live, it retracts at the commanded speed.
    pj->SetMotorSpeed(Real(-2));
    for (int i = 0; i < 30; ++i) { w.Step(kStep); }
    CHECK_THAT(static_cast<double>(w.Velocity(ram).x), WithinRel(-2.0, 0.02));
    CHECK(pj->MotorSpeed() == Real(-2));
    CHECK(pj->MaxMotorForce() == Real(40));
    CHECK(pj->IsMotorEnabled());
}
