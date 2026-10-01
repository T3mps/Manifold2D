// PhysicsMoverCcdTest.cpp
// [physics][ccd][movers]: fast bodies against MOVING geometry, and bullets.
//
// Speculative contacts stop a fast body at a static wall because the static
// query is padded by the body's reach this step (|v| * dt). Pairs between two
// movers (dynamic vs kinematic, dynamic vs dynamic) were found only once their
// boxes overlapped, so a fast body was already inside before its contact
// existed: it sank in and was pushed out over several steps at the 3 m/s cap,
// or passed a thin moving wall entirely. A fast mover now registers the box it
// will sweep this step before pairs are found (Box2D v2.4 b2DynamicTree::
// MoveProxy's displacement prediction), so the speculative margin works the
// same against movers as against statics.
//
// Bullets (Box2D v3 b2SolveContinuous): a dynamic bullet also sweeps kinematic
// and non-bullet dynamic bodies, not only statics; the sweep honours collision
// filters and sensors, skips other bullets, and ignores a touch at the start
// (fraction 0) so a bullet resting on something is never frozen. SetBullet
// (b2Body_SetBullet) flips the flag on a live body.
//
// PRESENTATION-FREE, ASCII comments, C++23.
#include <algorithm>
#include <cmath>
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
    constexpr Real kWallHT = Real(0.07); // half-thickness: a hopper wall (14 cm)
    constexpr Real kBox = Real(0.15);    // projectile half-size

    // A rotatable box (a dynamic AABB shape must be fixedRotation).
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

    BodyHandle AddWall(PhysicsWorld& w, BodyType type, Real halfT = kWallHT, Real density = Real(1),
                       std::uint32_t cat = 1u, std::uint32_t mask = 0xFFFFFFFFu)
    {
        BodyDef d;
        d.type = type;
        d.position = Vec2(Real(0), Real(0));
        d.shape = Box(halfT, Real(2.5));
        d.density = density;
        d.categoryBits = cat;
        d.maskBits = mask;
        return w.AddBody(d);
    }

    BodyHandle AddProjectile(PhysicsWorld& w, Vec2 pos, bool bullet = false, Shape shape = Box(kBox, kBox),
                             std::uint32_t cat = 1u, std::uint32_t mask = 0xFFFFFFFFu)
    {
        BodyDef d;
        d.type = BodyType::Dynamic;
        d.position = pos;
        d.shape = shape;
        d.density = Real(1);
        d.restitution = Real(0.1);
        d.bullet = bullet;
        d.categoryBits = cat;
        d.maskBits = mask;
        return w.AddBody(d);
    }

    struct Run { Real maxPen = 0; bool through = false; int embedSteps = 0; };

    // Fire a box at +x into the wall (whose near face is at -halfT) and track how
    // far its leading face ever gets past that face, and whether it ends beyond.
    Run Fire(PhysicsWorld& w, BodyHandle wall, BodyHandle p, Real speed, Real halfT = kWallHT, int steps = 90)
    {
        w.SetVelocity(p, Vec2(speed, Real(0)));
        Run r;
        int run = 0;
        for (int i = 0; i < steps; ++i)
        {
            w.Step(kStep);
            const Real wx = w.Position(wall).x, px = w.Position(p).x;
            const Real pen = (px + kBox) - (wx - halfT);
            r.maxPen = std::max(r.maxPen, pen);
            run = pen > Real(0.02) ? run + 1 : 0;
            r.embedSteps = std::max(r.embedSteps, run);
            if (px - kBox > wx + halfT) { r.through = true; }
        }
        return r;
    }
} // namespace

// The soft contact (b2MakeSoft at contactHertz) gives a little on a hard hit once the
// speculative gap is closed -- about 0.1 m at 15 m/s, cleared within a few steps,
// the same as against a static wall. What this guards is the pairing: the contact
// exists before the box arrives, so it neither passes through nor sits half inside.
TEST_CASE("Mover CCD: a fast box does not sink into a kinematic wall",
          "[physics][ccd][movers]")
{
    for (const Real speed : { Real(15), Real(30), Real(60) })
    {
        PhysicsWorld w{ ZeroG() };
        const BodyHandle wall = AddWall(w, BodyType::Kinematic);
        const BodyHandle p = AddProjectile(w, Vec2(Real(-3), Real(0)));
        const Run r = Fire(w, wall, p, speed);
        INFO("speed " << speed << " max penetration " << r.maxPen << " embedded steps " << r.embedSteps);
        CHECK_FALSE(r.through);
        CHECK(r.maxPen < Real(0.12)); // was 0.22 m (most of the box), held up to 15 steps
        CHECK(r.embedSteps <= 6);
    }
}

TEST_CASE("Mover CCD: a fast box does not sink into a heavy dynamic wall",
          "[physics][ccd][movers]")
{
    for (const Real speed : { Real(15), Real(30) })
    {
        PhysicsWorld w{ ZeroG() };
        const BodyHandle wall = AddWall(w, BodyType::Dynamic, kWallHT, Real(200));
        const BodyHandle p = AddProjectile(w, Vec2(Real(-3), Real(0)));
        const Run r = Fire(w, wall, p, speed);
        INFO("speed " << speed << " max penetration " << r.maxPen);
        CHECK_FALSE(r.through); // went straight through before
        CHECK(r.maxPen < Real(0.12));
        CHECK(r.embedSteps <= 6);
    }
}

TEST_CASE("Mover CCD: a fast box does not pass a thin kinematic wall",
          "[physics][ccd][movers]")
{
    PhysicsWorld w{ ZeroG() };
    const BodyHandle wall = AddWall(w, BodyType::Kinematic);
    const BodyHandle p = AddProjectile(w, Vec2(Real(-3), Real(0)));
    const Run r = Fire(w, wall, p, Real(40)); // passed straight through before
    CHECK_FALSE(r.through);
}

TEST_CASE("Mover CCD: a body flung between steps is caught on its first step",
          "[physics][ccd][movers]")
{
    // At rest 0.4 m from a kinematic wall, then flung at 40 m/s (0.67 m a step):
    // last step's displacement says nothing, so the prediction must use the
    // velocity of the step about to be taken.
    PhysicsWorld w{ ZeroG() };
    const BodyHandle wall = AddWall(w, BodyType::Kinematic);
    const BodyHandle p = AddProjectile(w, Vec2(-kWallHT - kBox - Real(0.4), Real(0)));
    for (int i = 0; i < 5; ++i) { w.Step(kStep); }
    const Run r = Fire(w, wall, p, Real(40), kWallHT, 30);
    INFO("max penetration " << r.maxPen);
    CHECK_FALSE(r.through);
    CHECK(r.maxPen < Real(0.12));
    CHECK(r.embedSteps <= 6);
}

TEST_CASE("Mover CCD: a spinning thin bullet does not tunnel a thin kinematic wall",
          "[physics][ccd][movers][bullet]")
{
    // Rotation is outside the speculative margin (it scales by linear speed), so a
    // thin bar spinning fast can slip through; the bullet sweep catches it.
    PhysicsWorld w{ ZeroG() };
    const BodyHandle wall = AddWall(w, BodyType::Kinematic);
    const BodyHandle p = AddProjectile(w, Vec2(Real(-3), Real(0)), true, Box(Real(0.2), Real(0.05)));
    w.SetAngularVelocity(p, Real(25));
    const Run r = Fire(w, wall, p, Real(80));
    CHECK_FALSE(r.through);
}

TEST_CASE("Mover CCD: a bullet's sweep honours collision filters",
          "[physics][ccd][bullet][filter]")
{
    // Masked against the wall's category: no contact, so no clamp either -- for
    // a static wall (the sweep ignored filters before) and a kinematic one.
    for (const BodyType t : { BodyType::Static, BodyType::Kinematic })
    {
        PhysicsWorld w{ ZeroG() };
        const BodyHandle wall = AddWall(w, t, kWallHT, Real(1), 2u, 0xFFFFFFFFu);
        const BodyHandle p = AddProjectile(w, Vec2(Real(-3), Real(0)), true, Box(kBox, kBox), 1u, 1u);
        const Run r = Fire(w, wall, p, Real(20), kWallHT, 30);
        INFO("wall type " << static_cast<int>(t));
        CHECK(r.through);
    }
}

TEST_CASE("Mover CCD: a bullet already touching a surface is not frozen by it",
          "[physics][ccd][bullet]")
{
    // A bullet lying on a floor and sliding along it touches the floor at the
    // start of every sweep; that touch (fraction 0) must not clamp it.
    for (const BodyType t : { BodyType::Static, BodyType::Kinematic })
    {
        PhysicsWorld w{ ZeroG() };
        BodyDef f;
        f.type = t;
        f.position = Vec2(Real(0), Real(0.5)); // top face at y = 0 (+y down)
        f.shape = MakeAabb(Real(30), Real(0.5));
        w.AddBody(f);
        const BodyHandle p = AddProjectile(w, Vec2(Real(-10), -kBox), true);
        w.SetVelocity(p, Vec2(Real(10), Real(0)));
        for (int i = 0; i < 30; ++i) { w.Step(kStep); }
        INFO("floor type " << static_cast<int>(t) << " x " << w.Position(p).x);
        CHECK(w.Position(p).x > Real(-6)); // ~5 m in 0.5 s; frozen would stay at -10
    }
}

TEST_CASE("Mover CCD: SetBullet flips the flag on a live body",
          "[physics][ccd][bullet]")
{
    PhysicsWorld w{ ZeroG() };
    const BodyHandle wall = AddWall(w, BodyType::Static);
    const BodyHandle p = AddProjectile(w, Vec2(Real(-3), Real(0)), false, Box(Real(0.2), Real(0.05)));
    CHECK_FALSE(w.IsBullet(p));
    w.SetBullet(p, true);
    CHECK(w.IsBullet(p));
    w.SetAngularVelocity(p, Real(25));
    const Run r = Fire(w, wall, p, Real(80)); // tunnels when not a bullet
    CHECK_FALSE(r.through);
    w.SetBullet(p, false);
    CHECK_FALSE(w.IsBullet(p));
}


