// Physics v2 Task 4: Fixture data model + mass aggregation.
//
// All expected values are hand-derived from first principles.
// Tolerance 1e-2 for mass/inertia (f32 at these magnitudes).
// Tolerance 1e-4 for world-transform geometry.
//
// Gate cases:
//   (a) Compound mass / COM / inertia: body with fixture0 = MakeCircle(0.5) at
//       localPos (0,0) + fixture1 = MakeCircle(0.3) at localPos (1.2,0),
//       density=1 for both. Hand derivation (MKS P4: re-derived at /10 scale,
//       not a mechanical /100 of the old px-era numbers -- shown in full):
//
//         Circle0: r=0.5 -> area=pi*0.25 ~ 0.785398
//              mass0 ~ 0.785398
//              centroid0_body = (0,0) [shape centroid at origin + localPos (0,0)]
//              I0_centroid = 0.5*m0*r^2 = 0.5*0.785398*0.25 ~ 0.098175
//
//         Circle1: r=0.3 -> area=pi*0.09 ~ 0.282743
//              mass1 ~ 0.282743
//              centroid1_body = (1.2,0) [shape centroid at origin + localPos (1.2,0)]
//              I1_centroid = 0.5*m1*r^2 = 0.5*0.282743*0.09 ~ 0.012723
//
//         Total mass = 0.785398 + 0.282743 = 1.068142
//
//         COM_x = (0.785398*0 + 0.282743*1.2) / 1.068142 = 0.339292 / 1.068142 ~ 0.317647
//         COM_y = 0
//
//         Inertia about COM (parallel axis for each fixture):
//           d0 = |COM - (0,0)| = 0.317647
//           d1 = |1.2 - 0.317647| = 0.882353
//           I_total = (I0_c + m0*d0^2) + (I1_c + m1*d1^2)
//                   = (0.098175 + 0.785398*0.100900) + (0.012723 + 0.282743*0.778547)
//                   = (0.098175 + 0.079246)  +  (0.012723 + 0.220129)
//                   = 0.177421 + 0.232852 ~ 0.410274
//
//   (b) Fixture world transform: body at pos(0,0) angle=pi/2, fixture at
//       localPos (0.5,0) localAngle=0.  World shape center:
//         R(pi/2)*(0.5,0) = (0.5*cos(pi/2) - 0*sin(pi/2), 0.5*sin(pi/2) + 0*cos(pi/2))
//                       = (0, 0.5)
//       So worldPos ~ (0, 0.5) within 1e-4.
//
//   (c) Back-compat: AddBody(BodyDef{ MakeCircle(0.2), density=1 }) creates a
//       body with exactly 1 fixture; mass ~ pi*0.04 ~ 0.125664; localCenter ~ (0,0).
//
//   (d) Stale handle: DropFixture then IsValid(h)==false; new fixture from
//       the same slot has a bumped generation (old handle stays invalid).
//
// NOTE: MakeAabb on a Dynamic body requires fixedRotation=true (engine invariant),
// which forces invInertia=0 and makes the inertia test trivial.  To exercise the
// full mass/COM/inertia aggregation path we use two MakeCircle fixtures, which are
// valid as Dynamic without fixedRotation.
//
// PRESENTATION-FREE + C++20-clean.

#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <Manifold2D/Physics/Fixture.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Shapes.hpp>

using Catch::Approx;
using namespace Manifold2D::Physics;

namespace
{
    constexpr double kPiD    = 3.14159265358979323846;
    constexpr double kMassTol = 1e-2;  // tolerance for mass/inertia (f32 magnitude)
    constexpr double kGeomTol = 1e-4;  // tolerance for geometry
}

// ============================================================================
// (a) Compound mass / COM / inertia: two circle fixtures.
//   fixture0 = MakeCircle(0.5) at localPos (0,0)
//   fixture1 = MakeCircle(0.3) at localPos (1.2,0)
//   density = 1 for both.
// ============================================================================
TEST_CASE("physics: Fixture compound mass/COM/inertia (two offset circles)", "[physics]")
{
    WorldDef wd;
    PhysicsWorld w(wd);

    BodyDef bd;
    bd.type        = BodyType::Dynamic;
    bd.position    = Vec2(Real(0), Real(0));
    // fixture0: MakeCircle(0.5) via AddBody back-compat.
    bd.shape       = MakeCircle(Real(0.5));
    bd.density     = Real(1);
    bd.restitution = Real(0);
    bd.friction    = Real(0.4);
    const BodyHandle bh = w.AddBody(bd);
    REQUIRE(w.IsValid(bh));

    // 1 auto-fixture after AddBody.
    REQUIRE(w.FixtureCount(bh) == 1u);

    // Add fixture1: MakeCircle(0.3) at localPos (1.2,0).
    FixtureDef fd;
    fd.shape      = MakeCircle(Real(0.3));
    fd.localPos   = Vec2(Real(1.2), Real(0));
    fd.localAngle = Real(0);
    fd.density    = Real(1);
    const FixtureHandle fh2 = w.AddFixture(bh, fd);
    REQUIRE(w.IsValid(fh2));

    REQUIRE(w.FixtureCount(bh) == 2u);

    // ---- check aggregated mass ------------------------------------------
    //   mass0 = pi*0.25 ~ 0.785398
    //   mass1 = pi*0.09 ~ 0.282743
    //   total ~ 1.068142
    const double mass0     = kPiD * 0.25;
    const double mass1     = kPiD * 0.09;
    const double massTotal = mass0 + mass1;

    const Real bodyMass = w.GetBodyMass(bh);
    REQUIRE(static_cast<double>(bodyMass) == Approx(massTotal).epsilon(kMassTol));

    // ---- check COM ---------------------------------------------------------
    //   COM_x = (mass0*0 + mass1*1.2) / massTotal
    const double comX = (mass0 * 0.0 + mass1 * 1.2) / massTotal;
    const double comY = 0.0;

    const Vec2 lc = w.GetLocalCenter(bh);
    REQUIRE(static_cast<double>(lc.x) == Approx(comX).epsilon(kMassTol));
    REQUIRE(static_cast<double>(lc.y) == Approx(0.0).margin(kGeomTol));

    // ---- check inertia about COM -------------------------------------------
    //   I0_c = 0.5*mass0*0.25 (I about centroid of circle0, r=0.5)
    //   I1_c = 0.5*mass1*0.09 (I about centroid of circle1, r=0.3)
    //   d0   = comX - 0   = comX
    //   d1   = 1.2 - comX
    //   I_total = (I0_c + mass0*d0^2) + (I1_c + mass1*d1^2)
    const double iCirc0C = 0.5 * mass0 * 0.25;
    const double iCirc1C = 0.5 * mass1 * 0.09;
    const double d0      = comX;
    const double d1      = 1.2 - comX;
    const double iTotal  = (iCirc0C + mass0 * d0 * d0)
                         + (iCirc1C + mass1 * d1 * d1);

    const Real bodyInertia = w.GetBodyInertia(bh);
    REQUIRE(static_cast<double>(bodyInertia) == Approx(iTotal).epsilon(kMassTol * 10.0));
}

// ============================================================================
// (b) Fixture world transform: R(pi/2)*(0.5,0) = (0,0.5).
// ============================================================================
TEST_CASE("physics: Fixture world transform (body angle pi/2, localPos (0.5,0))", "[physics]")
{
    WorldDef wd;
    PhysicsWorld w(wd);

    BodyDef bd;
    bd.type     = BodyType::Static; // use Static so no dynamics interference
    bd.position = Vec2(Real(0), Real(0));
    bd.shape    = MakeCircle(Real(0.1)); // minimal body shape
    bd.density  = Real(0);
    const BodyHandle bh = w.AddBody(bd);
    REQUIRE(w.IsValid(bh));

    // Set body angle to pi/2.
    w.SetAngle(bh, Real(kPiD / 2.0));

    // Add a fixture at localPos (0.5,0).
    FixtureDef fd;
    fd.shape      = MakeCircle(Real(0.1));
    fd.localPos   = Vec2(Real(0.5), Real(0));
    fd.localAngle = Real(0);
    fd.density    = Real(0);
    const FixtureHandle fh = w.AddFixture(bh, fd);
    REQUIRE(w.IsValid(fh));

    // Query the fixture world transform.
    // worldPos = bodyPos + R(bodyAngle) * localPos
    //          = (0,0)  + R(pi/2) * (0.5,0)
    //          = (0.5*cos(pi/2) - 0*sin(pi/2),  0.5*sin(pi/2) + 0*cos(pi/2))
    //          = (0, 0.5)
    const Vec2 worldPos = w.GetFixtureWorldPos(fh);
    REQUIRE(static_cast<double>(worldPos.x) == Approx(0.0).margin(kGeomTol));
    REQUIRE(static_cast<double>(worldPos.y) == Approx(0.5).epsilon(kGeomTol));

    // worldAngle = bodyAngle + localAngle = pi/2 + 0 = pi/2
    const Real worldAngle = w.GetFixtureWorldAngle(fh);
    REQUIRE(static_cast<double>(worldAngle) == Approx(kPiD / 2.0).epsilon(kGeomTol));
}

// ============================================================================
// (c) Back-compat: AddBody creates exactly 1 fixture; mass matches single-shape.
// ============================================================================
TEST_CASE("physics: Fixture back-compat AddBody creates exactly 1 fixture", "[physics]")
{
    WorldDef wd;
    PhysicsWorld w(wd);

    BodyDef bd;
    bd.type    = BodyType::Dynamic;
    bd.shape   = MakeCircle(Real(0.2));
    bd.density = Real(1);
    const BodyHandle bh = w.AddBody(bd);
    REQUIRE(w.IsValid(bh));

    // Exactly 1 fixture.
    REQUIRE(w.FixtureCount(bh) == 1u);

    // Mass ~ pi * r^2 = pi * 0.04 ~ 0.125664.
    const double expectedMass = kPiD * 0.04;
    const Real bodyMass = w.GetBodyMass(bh);
    REQUIRE(static_cast<double>(bodyMass) == Approx(expectedMass).epsilon(kMassTol));

    // invMass ~ 1/0.125664.
    const Real expectedInvMass = Real(1) / Real(expectedMass);
    // We read it via GetBodyMass/InvMass indirectly; verify it by inverting.
    REQUIRE(static_cast<double>(Real(1) / bodyMass) ==
            Approx(1.0 / expectedMass).epsilon(kMassTol));

    // localCenter ~ (0,0) for a centred circle.
    const Vec2 lc = w.GetLocalCenter(bh);
    REQUIRE(static_cast<double>(lc.x) == Approx(0.0).margin(kGeomTol));
    REQUIRE(static_cast<double>(lc.y) == Approx(0.0).margin(kGeomTol));
}

// ============================================================================
// (d) Stale handle invariant.
// ============================================================================
TEST_CASE("physics: Fixture stale handle after DropFixture", "[physics]")
{
    WorldDef wd;
    PhysicsWorld w(wd);

    // Create a body (back-compat creates 1 fixture for us, but we want to test
    // explicit AddFixture / DropFixture, so let's add an extra one).
    BodyDef bd;
    bd.type    = BodyType::Static;
    bd.shape   = MakeCircle(Real(0.1));
    bd.density = Real(0);
    const BodyHandle bh = w.AddBody(bd);

    FixtureDef fd;
    fd.shape    = MakeCircle(Real(0.1));
    fd.density  = Real(0);
    const FixtureHandle fh = w.AddFixture(bh, fd);
    REQUIRE(w.IsValid(fh));

    const std::uint32_t oldIndex = fh.index;
    const std::uint32_t oldGen   = fh.generation;

    // Drop the fixture -- the slot is recycled with a bumped generation.
    w.DropFixture(fh);
    REQUIRE(!w.IsValid(fh)); // stale: generation mismatch

    // Add a new fixture -- it should reuse the same slot (free-list).
    const FixtureHandle fh2 = w.AddFixture(bh, fd);
    REQUIRE(w.IsValid(fh2));

    // Same slot index (free-list reuse), but newer generation.
    REQUIRE(fh2.index == oldIndex);
    REQUIRE(fh2.generation > oldGen);

    // Original stale handle still invalid.
    REQUIRE(!w.IsValid(fh));
}
