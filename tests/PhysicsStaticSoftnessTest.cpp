// PhysicsStaticSoftnessTest.cpp
// [physics][solver][softness]: contacts against static geometry are stiffer.
//
// Box2D v3 builds two contact softnesses each step (solver.c):
//   contactSoftness = b2MakeSoft(contactHertz,        dampingRatio, h)
//   staticSoftness  = b2MakeSoft(2.0f * contactHertz, dampingRatio, h)
// and a contact uses staticSoftness when one side has no solver body -- a static
// body (contact_solver.c: "Stiffer for static contacts to avoid bodies getting
// pushed through the ground"). Kinematic bodies have solver bodies in Box2D (they
// are coloured like dynamic ones), so their contacts keep the regular softness.
//
// PRESENTATION-FREE, ASCII comments, C++23.
#include <algorithm>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>

using namespace Manifold2D::Physics;

namespace
{
    constexpr Real kStep = Real(1) / Real(60);

    Shape Box(Real hx, Real hy)
    {
        return MakePolygon(std::vector<Vec2>{ Vec2(-hx, -hy), Vec2(hx, -hy), Vec2(hx, hy), Vec2(-hx, hy) });
    }
} // namespace

// (A hard hit is NOT what this changes: the 0.1-0.15 m a fast box sinks into a wall
// on impact comes from the sequential two-point solve in the impact step, the same
// against static and kinematic walls; Box2D avoids it with a time-of-impact sweep
// for fast bodies against statics, not with the softness.)
TEST_CASE("Static softness: a box resting on static ground settles closer to its surface",
          "[physics][solver][softness]")
{
    // Under gravity, the resting overlap of the stiffer static contact is smaller
    // than that of the same box resting on a kinematic floor.
    const auto rest = [](BodyType floorType)
    {
        PhysicsWorld w{ WorldDef{} };
        BodyDef f;
        f.type = floorType;
        f.position = Vec2(Real(0), Real(0.5)); // top face at y = 0 (+y down)
        f.shape = Box(Real(10), Real(0.5));
        w.AddBody(f);
        BodyDef b;
        b.type = BodyType::Dynamic;
        b.position = Vec2(Real(0), Real(-2));
        b.shape = Box(Real(0.5), Real(0.5));
        b.density = Real(4);
        const BodyHandle h = w.AddBody(b);
        for (int i = 0; i < 180; ++i) { w.Step(kStep); }
        return w.Position(h).y + Real(0.5); // > 0: sunk into the floor
    };
    const Real st = rest(BodyType::Static), kin = rest(BodyType::Kinematic);
    INFO("static " << st << " kinematic " << kin);
    CHECK(st < kin * Real(0.5)); // measured ~0.09 mm vs ~0.34 mm
}


