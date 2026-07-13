// Physics v2 Task 7: rotation + fixture-aware migration of the remaining
// "old narrowphase" consumers (events, queries, CCD).
//
// T5 made the SOLVER contact path (GenerateContacts) rotation + fixture aware.
// T7 finishes the job for the THREE other narrowphase consumers that still ran
// the OLD path on the legacy single body shape at angle 0:
//   Part 0 -- the GJK conservative-advancement core builder (BuildCore) now
//             honors xf.rotation, making ShapeDistance / ShapeCast /
//             ShapePolyDistance rotation-aware automatically.
//   Part A -- ContactManager overlap (events / re-arm) is rotation + fixture
//             aware via PhysicsWorld::SlotsOverlap.
//   Part B -- Queries (RayVsBody, ShapeCast query, OverlapShape) iterate body
//             fixtures with the composed real angle.
//   Part C -- BulletSweep (CCD) sweeps the bullet's fixtures with real angle.
//
// All six gate cases are ANALYTIC (hand-derived expected values, NOT captured
// from the old engine):
//   (a) Part 0 -- ShapeDistance to a rotated capsule (and angle-0 unchanged).
//   (b) Part A -- a kinematic body that overlaps a static ONLY when rotated
//                 fires a Begin contact event (and not at angle 0).
//   (c) Part B -- a ray vs a 90-deg-rotated capsule hits the rotated extent
//                 (and angle-0 unchanged).
//   (d) Part B -- OverlapShape on a body that overlaps the query ONLY rotated.
//   (e) Part B -- OverlapShape on a COMPOUND body via fixture[1] (offset).
//   (f) Part C -- a fast bullet whose ROTATED fixture extent is clamped just
//                 outside a thin wall it would otherwise tunnel.
//
// PRESENTATION-FREE + C++20-clean.

#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Manifold2D/Physics/Fixture.hpp>
#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Narrowphase/Gjk.hpp>

using namespace Manifold2D::Physics;
using Catch::Approx;

namespace
{
    constexpr Real kStep   = Real(1) / Real(60);
    constexpr Real kHalfPi = Real(1.5707963267948966);

    // Box polygon (half-extents hw x hh), CCW: BL, BR, TR, TL. A polygon (not
    // an AABB) so a Dynamic/Kinematic body can carry it at a non-zero angle.
    Shape MakeBoxPolygon(Real hw, Real hh)
    {
        return MakePolygon({
            Vec2(-hw, -hh),
            Vec2( hw, -hh),
            Vec2( hw,  hh),
            Vec2(-hw,  hh),
        });
    }
}

// ===========================================================================
// (a) Part 0 -- BuildCore honors rotation: ShapeDistance to a rotated capsule.
//
// MakeCapsule(0.6, 0.2): core segment (-0.6,0)-(+0.6,0), radius 0.2.
//   At angle 0 it is HORIZONTAL: x in [-0.8,0.8], y in [-0.2,0.2].
//   Rotated +pi/2 it is VERTICAL: x in [-0.2,0.2], y in [-0.8,0.8].
//
// A zero-radius point (MakeCircle(0)) at (0,1.2):
//   vs the ROTATED (vertical) capsule: nearest segment point is (0,0.6),
//     gap 1.2-0.6 = 0.6, minus radius 0.2 = 0.4.
//   vs the angle-0 (horizontal) capsule: nearest segment point is (0,0),
//     gap 1.2, minus radius 0.2 = 1.0.
// Rotation-blind code would report 1.0 for BOTH (it would never rotate the
// segment), so the rotated case is the discriminator.
// ===========================================================================
TEST_CASE("physics-v2 T7 (a): ShapeDistance honors rotation (rotated capsule)",
          "[physics][PhysicsQueryRotation]")
{
    const Shape capsule = MakeCapsule(Real(0.6), Real(0.2));
    const Shape point   = MakeCircle(Real(0));

    // Rotated +pi/2 (vertical): surface distance from (0,1.2) = 1.2-0.6-0.2 = 0.4.
    {
        const Transform xfCap{ Vec2(Real(0), Real(0)), kHalfPi };
        const Transform xfPt { Vec2(Real(0), Real(1.2)), Real(0) };
        const ShapeDistanceResult d = ShapeDistance(point, xfPt, capsule, xfCap);
        CHECK(static_cast<double>(d.distance) == Approx(0.4).margin(1e-4));
    }

    // Angle 0 (horizontal): surface distance from (0,1.2) = 1.2-0-0.2 = 1.0.
    // (Proves the angle-0 path is unchanged by the rotation fix.)
    {
        const Transform xfCap{ Vec2(Real(0), Real(0)), Real(0) };
        const Transform xfPt { Vec2(Real(0), Real(1.2)), Real(0) };
        const ShapeDistanceResult d = ShapeDistance(point, xfPt, capsule, xfCap);
        CHECK(static_cast<double>(d.distance) == Approx(1.0).margin(1e-4));
    }
}

// ===========================================================================
// (b) Part A -- ContactManager events are rotation + fixture aware.
//
// A static CIRCLE radius 0.2 at (0,0.6) and a KINEMATIC box-polygon
// (half-extents 1 x 0.2) at the origin.
//   At angle 0 the box spans y in [-0.2,0.2]; the circle bottom is at y=0.4
//     -> a gap of 0.2 -> NO overlap -> no Begin event.
//   Rotated +pi/2 the box spans y in [-1,1]; it now overlaps the circle
//     (whose center y=0.6 is well inside the box) -> a Begin event fires.
// ===========================================================================
TEST_CASE("physics-v2 T7 (b): contact Begin fires only when the body is rotated",
          "[physics][PhysicsQueryRotation]")
{
    auto buildWorld = [](Real boxAngle) -> int
    {
        WorldDef wd;
        PhysicsWorld w(wd);

        BodyDef circ;
        circ.type     = BodyType::Static;
        circ.position = Vec2(Real(0), Real(0.6));
        circ.shape    = MakeCircle(Real(0.2));
        w.AddBody(circ);

        BodyDef box;
        box.type     = BodyType::Kinematic; // kinematic-vs-static emits events
        box.position = Vec2(Real(0), Real(0));
        box.shape    = MakeBoxPolygon(Real(1), Real(0.2));
        BodyHandle hb = w.AddBody(box);
        w.SetAngle(hb, boxAngle);

        int beginCount = 0;
        w.OnContact([&](const ContactEvent& ev)
        {
            if (ev.type == ContactEvent::Type::Begin)
            {
                ++beginCount;
            }
        });

        w.Step(kStep);
        return beginCount;
    };

    // Angle 0: no overlap -> no Begin.
    CHECK(buildWorld(Real(0)) == 0);
    // Rotated pi/2: overlap -> Begin fires.
    CHECK(buildWorld(kHalfPi) == 1);
}

// ===========================================================================
// (c) Part B -- Raycast / RayVsBody vs a rotated capsule hits the rotated
//               extent.
//
// A kinematic body with MakeCapsule(0.6,0.2) at the origin.
//   Rotated +pi/2 (vertical): top surface at y=0.8.  A ray from (0,2) down to
//     (0,-2) (delta y = -4) hits the top at y=0.8 -> t = (2-0.8)/4 = 0.30.
//   Angle 0 (horizontal): top surface at y=0.2 -> t = (2-0.2)/4 = 0.45.
// ===========================================================================
TEST_CASE("physics-v2 T7 (c): raycast vs a rotated capsule hits rotated extent",
          "[physics][PhysicsQueryRotation]")
{
    auto castT = [](Real capAngle) -> Real
    {
        WorldDef wd;
        PhysicsWorld w(wd);
        BodyDef bd;
        bd.type     = BodyType::Kinematic;
        bd.position = Vec2(Real(0), Real(0));
        bd.shape    = MakeCapsule(Real(0.6), Real(0.2));
        BodyHandle h = w.AddBody(bd);
        w.SetAngle(h, capAngle);

        const auto hit = w.Raycast(Vec2(Real(0), Real(2)), Vec2(Real(0), Real(-2)));
        REQUIRE(hit.has_value());
        REQUIRE_FALSE(hit->isCell);
        return hit->t;
    };

    // Rotated pi/2: top at y=0.8 -> t = 1.2/4 = 0.30.
    CHECK(static_cast<double>(castT(kHalfPi)) == Approx(0.30).margin(1e-3));
    // Angle 0: top at y=0.2 -> t = 1.8/4 = 0.45 (unchanged path).
    CHECK(static_cast<double>(castT(Real(0))) == Approx(0.45).margin(1e-3));
}

// ===========================================================================
// (d) Part B -- OverlapShape on a body that overlaps the query ONLY rotated.
//
// A static MakeAabb(1,0.2) body at the origin (a long thin box).
//   At angle 0 it spans y in [-0.2,0.2].
//   Rotated +pi/2 it spans y in [-1,1].
// A query MakeCircle(0.1) at (0,0.9):
//   unrotated: nearest body point y=0.2, gap 0.9-0.2=0.7 > radius 0.1 -> NO overlap.
//   rotated  : the query center y=0.9 is inside the body (y in [-1,1]),
//              x=0 in [-0.2,0.2] -> OVERLAP.
// ===========================================================================
TEST_CASE("physics-v2 T7 (d): OverlapShape sees a body only when rotated",
          "[physics][PhysicsQueryRotation]")
{
    auto overlaps = [](Real bodyAngle) -> int
    {
        WorldDef wd;
        PhysicsWorld w(wd);
        BodyDef bd;
        bd.type     = BodyType::Static;
        bd.position = Vec2(Real(0), Real(0));
        bd.shape    = MakeAabb(Real(1), Real(0.2));
        BodyHandle h = w.AddBody(bd);
        w.SetAngle(h, bodyAngle);

        const Shape query = MakeCircle(Real(0.1));
        const Transform xf{ Vec2(Real(0), Real(0.9)), Real(0) };
        std::vector<BodyHandle> hits;
        return w.OverlapShape(query, xf, hits);
    };

    CHECK(overlaps(Real(0)) == 0);     // unrotated: misses
    CHECK(overlaps(kHalfPi) == 1);     // rotated: overlaps
}

// ===========================================================================
// (e) Part B -- OverlapShape on a COMPOUND body via fixture[1].
//
// A body with TWO fixtures:
//   fixture0 = MakeAabb(0.2,0.2) at local (0,0) [the AddBody auto-fixture]
//   fixture1 = MakeCircle(0.2) at local (1.2,0)
// A query MakeCircle(0.1) at (1.2,0): the rotation-blind single-shape path
// would test only the primary box at the origin and MISS; the fixture-aware
// path finds the overlap via fixture[1] (the offset circle).
// ===========================================================================
TEST_CASE("physics-v2 T7 (e): OverlapShape sees a compound body via fixture[1]",
          "[physics][PhysicsQueryRotation]")
{
    WorldDef wd;
    PhysicsWorld w(wd);

    BodyDef bd;
    bd.type     = BodyType::Static;
    bd.position = Vec2(Real(0), Real(0));
    bd.shape    = MakeAabb(Real(0.2), Real(0.2)); // fixture0 at local (0,0)
    BodyHandle bh = w.AddBody(bd);

    FixtureDef fd;
    fd.shape    = MakeCircle(Real(0.2));
    fd.localPos = Vec2(Real(1.2), Real(0)); // fixture1 offset
    w.AddFixture(bh, fd);

    // Query at (1.2,0): overlaps fixture1 (a circle r=0.2 at (1.2,0)) but NOT
    // the primary box at the origin.
    const Shape query = MakeCircle(Real(0.1));
    const Transform xf{ Vec2(Real(1.2), Real(0)), Real(0) };
    std::vector<BodyHandle> hits;
    const int n = w.OverlapShape(query, xf, hits);
    REQUIRE(n == 1);
    CHECK(hits[0] == bh);

    // Control: the same query far from BOTH fixtures finds nothing.
    const Transform xfFar{ Vec2(Real(4), Real(0)), Real(0) };
    std::vector<BodyHandle> none;
    CHECK(w.OverlapShape(query, xfFar, none) == 0);
}

// ===========================================================================
// (f) Part C -- BulletSweep clamps a fast bullet whose ROTATED fixture extent
//               tunnels a thin wall.
//
// A thin horizontal static wall MakeAabb(5, 0.05) at (0,10): spans
//   y in [9.95, 10.05].
// A KINEMATIC bullet box MakeBoxPolygon(0.8, 0.1) rotated +pi/2 so its y
//   half-extent is 0.8 (instead of 0.1).  It starts at (0,0) and is given a
//   huge downward velocity so it would travel 20 units in one step
//   (prev=(0,0), curr=(0,20), delta=(0,20)).
//
// The rotated bullet's leading (bottom) edge is at center_y + 0.8.  It first
// touches the wall top (y=9.95) when center_y + 0.8 = 9.95 -> center_y = 9.15,
//   t = 9.15 / 20 = 0.4575.  Clamp = t - 0.001 -> clamped center_y ~= 9.13.
//
// Rotation-blind (y half-extent 0.1) would clamp at center_y ~= 9.85 instead,
// so the post-step y near 9.13 (NOT ~9.85) is the discriminator.  Either way
// the bullet must stop SHORT of the wall top (y=9.95).
// ===========================================================================
TEST_CASE("physics-v2 T7 (f): BulletSweep clamps a rotated bullet at its extent",
          "[physics][PhysicsQueryRotation]")
{
    WorldDef wd;
    PhysicsWorld w(wd);

    BodyDef wall;
    wall.type     = BodyType::Static;
    wall.position = Vec2(Real(0), Real(10));
    wall.shape    = MakeAabb(Real(5), Real(0.05)); // thin: y in [9.95,10.05]
    w.AddBody(wall);

    BodyDef bullet;
    bullet.type     = BodyType::Kinematic;
    bullet.position = Vec2(Real(0), Real(0));
    bullet.shape    = MakeBoxPolygon(Real(0.8), Real(0.1)); // long in x, thin in y
    bullet.bullet   = true;
    BodyHandle hb = w.AddBody(bullet);
    w.SetAngle(hb, kHalfPi); // rotate so the y half-extent becomes 0.8

    // Velocity that moves the bullet 20 units (+Y, down) in one 1/60s step.
    // KINEMATIC and deliberately faster than the 400 m/s cap: Arcane exempts
    // kinematics from the maxLinearVelocity clamp -- parity ledger B1,
    // ACCEPTED-AS-DIVERGENCE (Box2D v3 clamps its whole awake set incl.
    // kinematics; Arcane does not, because kinematic motion is authored and the
    // CCD bullet sweep handles fast kinematics -- see
    // docs/superpowers/audits/2026-07-08-physics-parity-ledger.md). Do NOT reduce
    // this below the tunneling speed -- a fast kinematic bullet is the point of
    // the rotated-extent CCD test.
    w.SetVelocity(hb, Vec2(Real(0), Real(20) / kStep));

    w.Step(kStep);

    const Vec2 finalPos = w.Position(hb);
    // The bullet's bottom edge (center_y + 0.8) must stop just OUTSIDE the wall
    // top (y=9.95): center_y < 9.15.
    CHECK(static_cast<double>(finalPos.y) < 9.15);
    // And it must be clamped at the rotated extent (~9.13), NOT the angle-0
    // extent (~9.85) -- so it is well below 9.5.
    CHECK(static_cast<double>(finalPos.y) > 8.8);
    CHECK(static_cast<double>(finalPos.y) < 9.5);
}
