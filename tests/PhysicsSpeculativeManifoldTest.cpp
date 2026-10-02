// PhysicsSpeculativeManifoldTest.cpp
// [physics][narrowphase][speculative]: polygons separated within the speculative
// margin get the reference-face clip manifold, as Box2D v3's b2CollidePolygons
// gives them -- not a single closest-point.
//
// A single speculative point at the GJK witness is arbitrary for parallel faces
// (here x = -0.98 under a box spanning -1.1..-0.9), so every impulse it carries
// torques the body: a box dropped flat landed spinning ~12 rad/s and could rock
// onto its end. b2CollidePolygons finds the reference face and clips the incident
// edge whether or not the shapes overlap, keeping each point whose separation is
// within the speculative distance; only a vertex-vertex approach (the closest
// features of the two edges are both vertices) keeps one point.
//
// PRESENTATION-FREE, ASCII comments, C++23.
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Narrowphase/Collide.hpp>

using namespace Manifold2D::Physics;

namespace
{
    Shape Box(Real hx, Real hy)
    {
        return MakePolygon(std::vector<Vec2>{ Vec2(-hx, -hy), Vec2(hx, -hy), Vec2(hx, hy), Vec2(-hx, hy) });
    }
} // namespace

TEST_CASE("Speculative manifold: a box just above a flat floor gets a point at each bottom corner",
          "[physics][narrowphase][speculative]")
{
    const Shape box = Box(Real(0.1), Real(0.05)), floor = Box(Real(5), Real(0.1));
    // box bottom at y = 0.87, floor top at 0.9 (+y down): a 3 cm gap, margin 5 cm
    const Manifold m = Collide(box, Transform{ Vec2(Real(-1), Real(0.82)), Real(0) },
                               floor, Transform{ Vec2(Real(0), Real(1.0)), Real(0) }, Real(0.05));
    REQUIRE(m.pointCount == 2);
    const Real x0 = std::min(m.points[0].point.x, m.points[1].point.x);
    const Real x1 = std::max(m.points[0].point.x, m.points[1].point.x);
    CHECK(std::abs(x0 - Real(-1.1)) < Real(0.01));
    CHECK(std::abs(x1 - Real(-0.9)) < Real(0.01));
    CHECK(std::abs(m.points[0].separation + Real(0.03)) < Real(0.005)); // a 3 cm gap: separation -0.03
    CHECK(std::abs(m.points[1].separation + Real(0.03)) < Real(0.005));
}

TEST_CASE("Speculative manifold: a corner approaching a corner keeps one point",
          "[physics][narrowphase][speculative]")
{
    // two boxes diagonal to each other, corners 2 cm apart (vertex-vertex)
    const Shape a = Box(Real(0.1), Real(0.1)), b = Box(Real(0.1), Real(0.1));
    const Real d = Real(0.2) + Real(0.02) / std::sqrt(Real(2));
    const Manifold m = Collide(a, Transform{ Vec2(Real(0), Real(0)), Real(0) },
                               b, Transform{ Vec2(d, d), Real(0) }, Real(0.05));
    CHECK(m.pointCount == 1);
}

TEST_CASE("Speculative manifold: a box dropped flat lands without spinning",
          "[physics][narrowphase][speculative]")
{
    PhysicsWorld w{ WorldDef{} };
    BodyDef g; g.type = BodyType::Static; g.position = Vec2(Real(0), Real(1.0)); g.shape = MakeAabb(Real(5), Real(0.1)); w.AddBody(g);
    BodyDef b; b.type = BodyType::Dynamic; b.position = Vec2(Real(-1), Real(0)); b.shape = Box(Real(0.1), Real(0.05));
    const BodyHandle p = w.AddBody(b);
    Real maxSpin = Real(0), maxTilt = Real(0);
    for (int i = 0; i < 120; ++i)
    {
        w.Step(Real(1) / Real(60));
        maxSpin = std::max(maxSpin, std::abs(w.AngularVelocity(p)));
        maxTilt = std::max(maxTilt, std::abs(w.GetAngle(p)));
    }
    INFO("max spin " << maxSpin << " max tilt " << maxTilt << " rest y " << w.Position(p).y);
    // The two corners are solved one after the other, so the landing step still ends
    // with a brief sign-flipping spin (-4.3, then +1.5 rad/s, settled within ~4 steps)
    // while the box turns < 1 degree. It was a one-point torque: 8-12 rad/s, rocking
    // to 0.1-0.75 rad and sometimes onto its end.
    CHECK(maxTilt < Real(0.02));
    CHECK(maxSpin < Real(6));
    CHECK(std::abs(w.AngularVelocity(p)) < Real(0.05)); // at rest
    CHECK(std::abs(w.Position(p).y - Real(0.85)) < Real(0.01));
}
