// PhysicsCastRayTest.cpp
// [physics][queries][castray]: PhysicsWorld::CastRayClosest -- the nearest
// fixture a ray hits, filtered like Box2D v3's b2World_CastRayClosest.
//
//   * filter: a fixture is a candidate when (its categoryBits & filter.maskBits)
//     and (its maskBits & filter.categoryBits) are both nonzero (Box2D's
//     b2ShouldQueryShape); filter.exclude skips one body (Box2D does this in the
//     callback); sensors are never hit.
//   * a fixture that contains the ray's origin is not reported (Box2D v3's
//     shape ray casts miss from inside).
//   * the result: the body and fixture hit, the point, the surface normal
//     there (unit, facing back along the ray), and the fraction of the
//     translation travelled.
//   * every body type is a candidate: static, kinematic, dynamic; compound
//     fixtures are tested at their own rotated transforms.
//
// Engine convention: +Y is DOWN.
//
// PRESENTATION-FREE, ASCII comments, C++23.
#include <cmath>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Fixture.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>

using namespace Manifold2D::Physics;

namespace
{
    Shape Box(Real hx, Real hy)
    {
        return MakePolygon(std::vector<Vec2>{ Vec2(-hx, -hy), Vec2(hx, -hy), Vec2(hx, hy), Vec2(-hx, hy) });
    }

    WorldDef ZeroG()
    {
        WorldDef wd;
        wd.gravityX = Real(0);
        wd.gravityY = Real(0);
        return wd;
    }

    BodyHandle Add(PhysicsWorld& w, BodyType type, Vec2 at, Shape s, std::uint32_t cat = 1u, std::uint32_t mask = 0xFFFFFFFFu, bool sensor = false)
    {
        BodyDef d;
        d.type = type;
        d.position = at;
        d.shape = s;
        d.categoryBits = cat;
        d.maskBits = mask;
        d.isSensor = sensor;
        return w.AddBody(d);
    }

    bool Near(Real a, Real b, Real eps = Real(0.01)) { return std::abs(a - b) < eps; }
} // namespace

TEST_CASE("CastRayClosest: the nearest of several bodies, with point, normal and fraction",
          "[physics][queries][castray]")
{
    PhysicsWorld w{ ZeroG() };
    const BodyHandle far = Add(w, BodyType::Static, Vec2(Real(6), Real(0)), Box(Real(0.5), Real(0.5)));
    const BodyHandle near = Add(w, BodyType::Dynamic, Vec2(Real(3), Real(0)), Box(Real(0.5), Real(0.5)));
    const auto hit = w.CastRayClosest(Vec2(Real(0), Real(0)), Vec2(Real(10), Real(0)));
    REQUIRE(hit.has_value());
    INFO("point " << hit->point.x << "," << hit->point.y << " normal " << hit->normal.x << "," << hit->normal.y << " fraction " << hit->fraction);
    CHECK(hit->body == near);
    CHECK(hit->body != far);
    CHECK(Near(hit->point.x, Real(2.5)));
    CHECK(Near(hit->point.y, Real(0)));
    CHECK(Near(hit->normal.x, Real(-1)));
    CHECK(Near(hit->normal.y, Real(0)));
    CHECK(Near(hit->fraction, Real(0.25)));
}

TEST_CASE("CastRayClosest: a miss, and a ray too short to reach",
          "[physics][queries][castray]")
{
    PhysicsWorld w{ ZeroG() };
    Add(w, BodyType::Static, Vec2(Real(3), Real(0)), Box(Real(0.5), Real(0.5)));
    CHECK_FALSE(w.CastRayClosest(Vec2(Real(0), Real(2)), Vec2(Real(10), Real(0))).has_value());
    CHECK_FALSE(w.CastRayClosest(Vec2(Real(0), Real(0)), Vec2(Real(2), Real(0))).has_value());
    CHECK_FALSE(w.CastRayClosest(Vec2(Real(0), Real(0)), Vec2(Real(0), Real(0))).has_value());
}

TEST_CASE("CastRayClosest: circles and kinematic bodies, normals off-axis",
          "[physics][queries][castray]")
{
    PhysicsWorld w{ ZeroG() };
    const BodyHandle c = Add(w, BodyType::Kinematic, Vec2(Real(4), Real(0)), MakeCircle(Real(1)));
    const auto hit = w.CastRayClosest(Vec2(Real(0), Real(0.6)), Vec2(Real(8), Real(0)));
    REQUIRE(hit.has_value());
    CHECK(hit->body == c);
    // the circle's surface at y = 0.6: x = 4 - sqrt(1 - 0.36) = 3.2; normal (-0.8, 0.6)
    CHECK(Near(hit->point.x, Real(3.2)));
    CHECK(Near(hit->normal.x, Real(-0.8)));
    CHECK(Near(hit->normal.y, Real(0.6)));
}

TEST_CASE("CastRayClosest: the filter decides like b2ShouldQueryShape",
          "[physics][queries][castray][filter]")
{
    PhysicsWorld w{ ZeroG() };
    const BodyHandle a = Add(w, BodyType::Static, Vec2(Real(2), Real(0)), Box(Real(0.3), Real(0.3)), 2u, 0xFFFFFFFFu);
    const BodyHandle b = Add(w, BodyType::Static, Vec2(Real(4), Real(0)), Box(Real(0.3), Real(0.3)), 4u, 1u);
    const Vec2 o(Real(0), Real(0)), d(Real(10), Real(0));

    QueryFilter f; // default: category 1, mask all -> a (mask all has 1) and b (mask 1) both
    REQUIRE(w.CastRayClosest(o, d, f).has_value());
    CHECK(w.CastRayClosest(o, d, f)->body == a);

    f.maskBits = 4u; // only category 4 bodies: skips a, hits b (whose mask accepts category 1)
    REQUIRE(w.CastRayClosest(o, d, f).has_value());
    CHECK(w.CastRayClosest(o, d, f)->body == b);

    f.categoryBits = 8u; // b's mask (1) does not accept the query's category 8
    CHECK_FALSE(w.CastRayClosest(o, d, f).has_value());
}

TEST_CASE("CastRayClosest: exclude skips the caster's own body",
          "[physics][queries][castray]")
{
    PhysicsWorld w{ ZeroG() };
    const BodyHandle self = Add(w, BodyType::Dynamic, Vec2(Real(0), Real(0)), Box(Real(1), Real(0.4)));
    const BodyHandle wall = Add(w, BodyType::Static, Vec2(Real(5), Real(0)), Box(Real(0.2), Real(2)));
    // starting on the caster's surface, heading out through it: with exclude, the wall
    QueryFilter f;
    f.exclude = self;
    const auto hit = w.CastRayClosest(Vec2(Real(-2), Real(0)), Vec2(Real(10), Real(0)), f);
    REQUIRE(hit.has_value());
    CHECK(hit->body == wall);
    const auto plain = w.CastRayClosest(Vec2(Real(-2), Real(0)), Vec2(Real(10), Real(0)));
    REQUIRE(plain.has_value());
    CHECK(plain->body == self);
}

TEST_CASE("CastRayClosest: a fixture containing the origin is not reported",
          "[physics][queries][castray]")
{
    PhysicsWorld w{ ZeroG() };
    Add(w, BodyType::Static, Vec2(Real(0), Real(0)), Box(Real(1), Real(1)));
    const BodyHandle next = Add(w, BodyType::Static, Vec2(Real(4), Real(0)), Box(Real(0.5), Real(0.5)));
    const auto hit = w.CastRayClosest(Vec2(Real(0), Real(0)), Vec2(Real(10), Real(0)));
    REQUIRE(hit.has_value());
    CHECK(hit->body == next);
}

TEST_CASE("CastRayClosest: sensors are not hit",
          "[physics][queries][castray]")
{
    PhysicsWorld w{ ZeroG() };
    Add(w, BodyType::Static, Vec2(Real(2), Real(0)), Box(Real(0.5), Real(0.5)), 1u, 0xFFFFFFFFu, true);
    const BodyHandle solid = Add(w, BodyType::Static, Vec2(Real(5), Real(0)), Box(Real(0.5), Real(0.5)));
    const auto hit = w.CastRayClosest(Vec2(Real(0), Real(0)), Vec2(Real(10), Real(0)));
    REQUIRE(hit.has_value());
    CHECK(hit->body == solid);
}

TEST_CASE("CastRayClosest: compound fixtures at their rotated transforms; the fixture is reported",
          "[physics][queries][castray]")
{
    PhysicsWorld w{ ZeroG() };
    // a body at (5, 0) turned 90 degrees, with a second fixture offset 1 m along its local +x
    // -> that fixture sits at world (5, 1) (local +x maps to world +y)
    const BodyHandle h = Add(w, BodyType::Dynamic, Vec2(Real(5), Real(0)), Box(Real(0.2), Real(0.2)));
    FixtureDef fd;
    fd.shape = Box(Real(0.3), Real(0.3));
    fd.localPos = Vec2(Real(1), Real(0));
    const FixtureHandle arm = w.AddFixture(h, fd);
    w.SetAngle(h, Real(1.5707963));
    const auto hit = w.CastRayClosest(Vec2(Real(0), Real(1)), Vec2(Real(10), Real(0)));
    REQUIRE(hit.has_value());
    CHECK(hit->body == h);
    CHECK(hit->fixture == arm);
    CHECK(Near(hit->point.x, Real(4.7)));
}

TEST_CASE("CastRayClosest: candidates come from every broadphase the world can use",
          "[physics][queries][castray][broadphase]")
{
    for (const BroadphaseKind kind : { BroadphaseKind::Tree, BroadphaseKind::Hash, BroadphaseKind::Sap })
    {
        WorldDef wd = ZeroG();
        wd.broadphase = kind;
        PhysicsWorld w{ wd };
        const BodyHandle d = Add(w, BodyType::Dynamic, Vec2(Real(3), Real(0)), Box(Real(0.5), Real(0.5)));
        Add(w, BodyType::Static, Vec2(Real(6), Real(0)), Box(Real(0.5), Real(0.5)));
        w.Step(Real(1) / Real(60)); // proxies placed
        const auto hit = w.CastRayClosest(Vec2(Real(0), Real(0)), Vec2(Real(10), Real(0)));
        REQUIRE(hit.has_value());
        CHECK(hit->body == d);
    }
}
