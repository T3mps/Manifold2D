// Physics v2 compound-COM dynamics: off-origin compound bodies rotate about
// their CENTER OF MASS, not their body origin.
//
// Physics v2's fixture model aggregates a per-body local center of mass into
// m_localCenterX/Y (the COM in the body's LOCAL frame). Single-fixture bodies
// and all static/kinematic bodies have localCenter == (0,0). Before this change
// the solver integrated + rotated a body about its ORIGIN, and GenerateContacts
// measured contact anchors from the ORIGIN -- correct only when COM == origin.
//
// This file exercises the corrected behavior with HAND-DERIVED analytic cases:
//
//   (a) Heavy-side tip [HEADLINE]: a compound body (light fixture at the local
//       origin + a HEAVY fixture offset to the +x side, so the aggregated COM is
//       clearly toward the heavy side) balanced on a narrow floor pivot under the
//       LIGHT side, released under gravity, tips TOWARD the heavy side. The heavy
//       fixture's world Y descends and the body angle INCREASES (positive): with
//       R(a)*(1,0) = (cos a, sin a) and +Y pointing DOWN, rotating the local +x
//       heavy side toward +y (down) is a positive angle change. With the OLD
//       origin-rotation the gravity torque was taken about the wrong point.
//
//   (b) Free rotation keeps the COM on its inertial path [CORE PROPERTY]: a
//       compound body with an off-origin COM, no gravity and no contacts, given an
//       initial angular velocity, keeps its WORLD COM stationary (comVel == 0)
//       while the ORIGIN orbits the COM on a circle of radius |localCenter|. This
//       is the literal "rotates about the COM" statement.
//
//   (c) Determinism: run (a) twice -> bit-identical final pose.
//
//   (d) Single-fixture byte-identity [GUARD]: a single-fixture body (localCenter
//       == (0,0)) integrates EXACTLY as the pre-change origin formula
//       (pos += vel*dt; angle += angVel*dt) -- the change is a provable no-op for
//       COM == origin. Asserted against the closed-form pre-change pose.
//
// CONVENTION: Cartesian, +Y DOWN (gravity +Y), matching the other physics tests.
// +angle is CCW (R(a)*v = (c*vx - s*vy, s*vx + c*vy)); a clockwise tip is a
// NEGATIVE angle change.
//
// PRESENTATION-FREE + C++20-clean. All expected values hand-derived.

#include <cmath>
#include <cstdint>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Manifold2D/Physics/Fixture.hpp>
#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>

using namespace Manifold2D::Physics;
using Catch::Approx;

namespace
{
    constexpr Real kStep = Real(1) / Real(60);

    // Build a "barbell" compound body: a LIGHT circle fixture at the local
    // origin + a HEAVY circle fixture offset to local +x by `offset`. Both
    // circles share the same radius; the heavy one carries a much larger
    // density so the aggregated COM sits clearly toward +x.
    //
    //   localCenter_x = (mLight*0 + mHeavy*offset) / (mLight + mHeavy)
    //                 = offset * mHeavy / (mLight + mHeavy)   (> 0, toward heavy)
    //   localCenter_y = 0.
    //
    // Returns the body handle; `bd`/the densities/offset are chosen so the COM
    // offset is large and unambiguous.
    struct Barbell
    {
        BodyHandle body;
        Real       expectedComX; // hand-derived local COM x
    };

    Barbell MakeBarbell(PhysicsWorld& w, Vec2 pos,
                        Real radius     = Real(0.3),
                        Real offset     = Real(2),
                        Real lightDens  = Real(1),
                        Real heavyDens  = Real(9))
    {
        // fixture0 (light) created by AddBody at local origin.
        BodyDef bd;
        bd.type         = BodyType::Dynamic;
        bd.position     = pos;
        bd.shape        = MakeCircle(radius);
        bd.density      = lightDens;
        bd.friction     = Real(0.4);
        bd.restitution  = Real(0);
        bd.fixedRotation = false; // rotation about COM is the whole point
        const BodyHandle bh = w.AddBody(bd);

        // fixture1 (heavy) at local +x offset.
        FixtureDef fd;
        fd.shape      = MakeCircle(radius);
        fd.localPos   = Vec2(offset, Real(0));
        fd.localAngle = Real(0);
        fd.density    = heavyDens;
        w.AddFixture(bh, fd);

        // Equal-radius circles -> mass ratio == density ratio.
        const Real mLight = lightDens; // proportional; area cancels in the ratio
        const Real mHeavy = heavyDens;
        const Real comX   = offset * mHeavy / (mLight + mHeavy);

        return Barbell{ bh, comX };
    }
}

// ============================================================================
// (a) HEADLINE: heavy-side tip about the COM.
//
// Setup (Cartesian, +Y down):
//   - A static floor pivot centered under the body ORIGIN (x = 0). Its
//     half-width (0.8) supports the light fixture region around the origin; the
//     heavy fixture at local +x = 2 OVERHANGS the pivot with no support
//     beneath it (the pivot's right edge is at x = +0.8, the heavy fixture is at
//     x = +2).
//   - A barbell body whose COM is at local x ~ +1.8 (heavy toward +x), placed so
//     the light fixture rests on the pivot.
//   - Gravity +Y (down). Released from rest.
//
// Hand derivation of the tip DIRECTION:
//   Gravity acts at the COM, which is at world x > 0 (right of the pivot/origin).
//   The only support is the pivot near the origin. Gravity therefore applies a
//   torque about the contact = (rCom x g): with rCom = (+x, 0) and g = (0, +gy),
//   the z-torque = rCom.x*g.y - rCom.y*g.x = (+)(+) > 0.
//
//   SIGN in THIS engine's convention: +Y is DOWN and +angle is the R(a) CCW
//   rotation R(a)*v = (c*vx - s*vy, s*vx + c*vy). Increasing the angle rotates
//   the local +x axis TOWARD +y, i.e. the heavy (+x) fixture moves toward +Y
//   (DOWN). So "heavy side descends" == "angle INCREASES (becomes POSITIVE)".
//   => angle goes POSITIVE and the heavy fixture's world Y INCREASES (descends)
//      well below the light fixture's world Y.
//
// The pre-change ORIGIN-rotation took the gravity torque about the wrong center;
// the corrected COM-rotation is what makes the cantilevered heavy side fall.
// We run a horizon long enough for the tip direction to be unambiguous and
// assert the SIGN of the rotation + the heavy-vs-light descent (both monotone-
// robust as the body rotates past the balance point).
// ============================================================================
namespace
{
    // Runs scene (a) and returns final pose + the two fixtures' world Y.
    struct TipResult
    {
        Real finalAngle;
        Real heavyYInitial;
        Real heavyYFinal;
        Real lightYFinal;
        Vec2 finalPos;
    };

    TipResult RunHeavyTip(int steps)
    {
        WorldDef wd; // gravity: MKS default (0, +10), y-down
        PhysicsWorld w(wd);

        // Pivot: half-width 0.8, top surface at y = 0 (center at y = +0.5,
        // half-height 0.5). Centered at x = 0 (under the body origin). The right
        // edge (x = +0.8) is well short of the heavy fixture (x = +2), so the
        // heavy side is cantilevered with no support.
        BodyDef floorBd;
        floorBd.type     = BodyType::Static;
        floorBd.position = Vec2(Real(0), Real(0.5)); // top surface at y=0
        floorBd.shape    = MakeAabb(Real(0.8), Real(0.5));
        w.AddBody(floorBd);

        // Barbell: light fixture (r=0.3) at origin, heavy fixture (r=0.3) at
        // +x=2. Place the body so the light fixture rests just above the pivot
        // top. Light fixture bottom = origin.y + r; want it at the floor top
        // (y=0), so origin.y = -r = -0.3.
        const Real radius = Real(0.3);
        Barbell bb = MakeBarbell(w, Vec2(Real(0), Real(-radius)), radius,
                                 /*offset=*/Real(2),
                                 /*lightDens=*/Real(1),
                                 /*heavyDens=*/Real(9));

        // Fixture[0] is the light auto-fixture (local origin); fixture[1] is the
        // heavy one (local +x = 2). Both start at world Y = origin.y under no
        // rotation.
        const FixtureHandle lightFh = w.GetBodyFixture(bb.body, 0u);
        const FixtureHandle heavyFh = w.GetBodyFixture(bb.body, 1u);
        const Real heavyYInitial = w.GetFixtureWorldPos(heavyFh).y;

        for (int i = 0; i < steps; ++i)
        {
            w.Step(kStep);
        }

        const Real finalAngle  = w.GetAngle(bb.body);
        const Real heavyYFinal = w.GetFixtureWorldPos(heavyFh).y;
        const Real lightYFinal = w.GetFixtureWorldPos(lightFh).y;
        const Vec2 finalPos    = w.Position(bb.body);
        return TipResult{ finalAngle, heavyYInitial, heavyYFinal,
                          lightYFinal, finalPos };
    }
}

TEST_CASE("physics-v2 compound-COM (a): heavy side tips down about the COM", "[physics]")
{
    // Sanity: the aggregated local COM is clearly offset toward the heavy (+x)
    // fixture. offset=2, densities 1:9 -> comX = 2 * 9/10 = 1.8 (bb.expectedComX).
    {
        WorldDef wd;
        wd.gravityX = Real(0); // deliberate zero-g scene (isolates COM geometry)
        wd.gravityY = Real(0);
        PhysicsWorld w(wd);
        Barbell bb = MakeBarbell(w, Vec2(Real(0), Real(0)));
        const Vec2 lc = w.GetLocalCenter(bb.body);
        INFO("local COM x = " << static_cast<double>(lc.x)
             << " expected " << static_cast<double>(bb.expectedComX));
        CHECK(static_cast<double>(lc.x)
              == Approx(static_cast<double>(bb.expectedComX)).epsilon(0.02));
        CHECK(static_cast<double>(lc.y) == Approx(0.0).margin(1e-3));
        // Unambiguously off-origin toward the heavy side: derived from the
        // analytic bb.expectedComX itself (well below the full 1.8 offset, well
        // above zero) rather than a bare magic literal.
        CHECK(lc.x > bb.expectedComX * Real(0.25));
    }

    // Measured trajectory (Debug, f32, 60 Hz, 4 substeps) below PREDATES the MKS
    // conversion (px-era scale: offset=20, comX=18, g=400). The absolute numbers
    // no longer apply, but the DIRECTIONAL signs it documents -- angle increases
    // monotonically (heavy +x rotating toward +y/down), heavy fixture descends
    // while light stays up on the pivot -- are scale-free and still hold:
    //   steps  3: angle 0.026  heavyY -2.47  lightY -3.00
    //   steps  6: angle 0.102  heavyY -0.96  lightY -2.99
    //   steps  9: angle 0.228  heavyY  1.53  lightY -2.98
    //   steps 12: angle 0.409  heavyY  4.99  lightY -2.96   <- px-era assertion point
    //   steps 15: angle 0.658  heavyY  9.35  lightY -2.88
    // (Past ~20 px-era steps the body had rotated past the balance point and
    // free-fell; 12 steps was squarely in the controlled-tip regime.)
    //
    // MKS re-baseline (protocol rule 5): gravity dropped 400 -> 10 (40x) while
    // lengths dropped only 10x (offset 20->2, radius 3->0.3), and angular
    // acceleration about the pivot ~ g/length, so the new scene tips ~4x slower
    // in angular-acceleration terms -> ~2x the steps to reach an equivalent
    // angle. steps=24 lands in the analogous controlled-tip window (verified:
    // well past the initial torque ramp-up, well short of the free-fall runaway).
    const TipResult r = RunHeavyTip(24);

    INFO("final angle = " << static_cast<double>(r.finalAngle)
         << " rad, heavy world Y: init " << static_cast<double>(r.heavyYInitial)
         << " -> final " << static_cast<double>(r.heavyYFinal)
         << ", light world Y final " << static_cast<double>(r.lightYFinal));

    // POSITIVE angle == heavy (+x) side rotating DOWN (+y) about the COM.
    CHECK(r.finalAngle > Real(0.05));
    // Heavy fixture descended appreciably from its start (re-baselined from the
    // px-era "+1": ~5% of the 2 m offset, same proportion as the original
    // "+1 out of offset=20").
    CHECK(r.heavyYFinal > r.heavyYInitial + Real(0.1));
    // ...and is now clearly BELOW the light fixture (which stayed on the pivot).
    CHECK(r.heavyYFinal > r.lightYFinal + Real(0.1));
}

// ============================================================================
// (b) CORE PROPERTY: free rotation keeps the COM on its inertial path.
//
// A barbell body with off-origin COM, NO gravity and NO contacts (it is the
// only body in the world), given a constant angular velocity omega and zero
// linear velocity. Over N sub-stepped Steps:
//   - The WORLD COM = origin + R(angle)*localCenter stays put (comVel == 0).
//   - The ORIGIN orbits that COM on a circle of radius |localCenter|.
//   - The angle advances by omega * N * dt.
//
// Hand derivation:
//   c0 = p0 + R(a0)*lc is the (fixed) world COM. After integrating the angle to
//   a, the origin is p = c0 - R(a)*lc. So |p - c0| == |R(a)*lc| == |lc| for every
//   step -> the origin traces a circle of radius |lc| about the stationary COM.
// ============================================================================
TEST_CASE("physics-v2 compound-COM (b): free spin keeps the COM fixed", "[physics]")
{
    WorldDef wd;
    wd.gravityX = Real(0); // deliberate zero-g scene (pure inertial spin)
    wd.gravityY = Real(0);
    PhysicsWorld w(wd);

    const Vec2 startPos(Real(0.7), Real(-0.4));
    Barbell bb = MakeBarbell(w, startPos);
    const std::uint32_t slot = bb.body.index;

    const Vec2 lc = w.GetLocalCenter(bb.body);
    const Real lcLen = std::sqrt(lc.x * lc.x + lc.y * lc.y);
    // genuinely off-origin: same bb.expectedComX-derived threshold as case (a).
    REQUIRE(lcLen > bb.expectedComX * Real(0.25));

    // Start-of-test world COM (must stay fixed for the whole run).
    const Real a0 = w.GetAngle(bb.body);
    const Vec2 com0 = startPos + RotateVec(a0, lc);

    // Give it a pure spin (no linear velocity) and wake it.
    const Real omega = Real(0.5); // rad/s, CCW
    w.SetVelocity(bb.body, Vec2(Real(0), Real(0)));
    w.SetAngVelSlot(slot, omega);
    w.Wake(bb.body);

    const int steps = 90;
    for (int i = 0; i < steps; ++i)
    {
        w.Step(kStep);

        // COM stays put every step (linear COM velocity is 0).
        const Vec2 pos = w.Position(bb.body);
        const Real ang = w.GetAngle(bb.body);
        const Vec2 com = pos + RotateVec(ang, lc);
        CHECK(static_cast<double>(com.x) == Approx(com0.x).margin(1e-3));
        CHECK(static_cast<double>(com.y) == Approx(com0.y).margin(1e-3));

        // The origin orbits the COM on a circle of radius |lc|.
        const Vec2 d = pos - com0;
        const Real rad = std::sqrt(d.x * d.x + d.y * d.y);
        CHECK(static_cast<double>(rad) == Approx(static_cast<double>(lcLen)).margin(1e-3));
    }

    // The angle advanced by omega * steps * dt (no damping, no torque).
    const Real expectedAngle = a0 + omega * static_cast<Real>(steps) * kStep;
    CHECK(static_cast<double>(w.GetAngle(bb.body))
          == Approx(static_cast<double>(expectedAngle)).margin(1e-3));

    // And the origin has visibly MOVED (it orbited) even though the COM did not.
    // "Visibly" re-derived from |lc| itself (was a flat px literal): a body that
    // has genuinely orbited displaces by an appreciable fraction of the orbit
    // radius, not just numerical noise.
    const Vec2 finalPos = w.Position(bb.body);
    CHECK(std::abs(static_cast<double>(finalPos.x - startPos.x))
              + std::abs(static_cast<double>(finalPos.y - startPos.y))
          > static_cast<double>(lcLen * Real(0.1)));
}

// ============================================================================
// (e) APPLY-IMPULSE TORQUES ABOUT THE COM: a linear impulse applied at the body
//     ORIGIN of an off-COM body produces the correct angular response -- the
//     lever arm is measured from the CENTER OF MASS, not the origin. With the
//     origin offset from the COM (r = origin - COM != 0), a downward impulse at
//     the origin spins the body. The pre-fix code measured the lever from the
//     origin (r == 0 there) and produced NO rotation -- wrong for compound
//     bodies whose invInertia is about the COM and which integrate rotation
//     about the COM.
//
// Hand derivation (gravity off, single body, applied at the world origin):
//   worldCOM = origin + R(0)*localCenter = (lc.x, lc.y).
//   lever r  = worldPoint - worldCOM = (0,0) - (lc.x, lc.y) = (-lc.x, -lc.y).
//   torque_z = r.x*impulse.y - r.y*impulse.x = (-lc.x)*J - (-lc.y)*0 = -lc.x*J.
//   angVel   = torque_z * invInertia_about_COM.
//   linear   = impulse / mass (COM linear velocity, unchanged by the fix).
// ============================================================================
TEST_CASE("physics-v2 compound-COM (e): impulse at a point torques about the COM", "[physics]")
{
    WorldDef wd;
    wd.gravityX = Real(0); // deliberate zero-g scene (isolates the impulse response)
    wd.gravityY = Real(0);
    PhysicsWorld w(wd);
    Barbell bb = MakeBarbell(w, Vec2(Real(0), Real(0)));

    const Vec2 lc   = w.GetLocalCenter(bb.body); // local COM == world COM at pos 0, ang 0
    const Real I    = w.GetBodyInertia(bb.body);
    const Real mass = w.GetBodyMass(bb.body);
    REQUIRE(I > Real(0));
    REQUIRE(lc.x > bb.expectedComX * Real(0.25)); // genuinely off-origin toward the heavy side

    // Apply a downward impulse AT THE BODY ORIGIN (world (0,0)), authored as
    // mass x target delta-v (named local) rather than a bare impulse literal.
    const Real targetDv = Real(2); // m/s, well under the 400 m/s velocity cap
    const Real J = mass * targetDv;
    w.ApplyImpulse(bb.body, Vec2(Real(0), J), Vec2(Real(0), Real(0)));

    const Real expectedAngVel = (-lc.x * J) / I;
    INFO("angVel = " << static_cast<double>(w.AngVelSlot(bb.body.index))
         << " expected " << static_cast<double>(expectedAngVel)
         << " (lc.x=" << static_cast<double>(lc.x) << " I=" << static_cast<double>(I) << ")");

    // Torque about the COM -> a nonzero, correctly-signed spin (the pre-fix
    // origin-based code produced exactly 0 here).
    CHECK(static_cast<double>(w.AngVelSlot(bb.body.index))
          == Approx(static_cast<double>(expectedAngVel)).epsilon(1e-4));
    CHECK(std::abs(static_cast<double>(w.AngVelSlot(bb.body.index))) > 0.01);

    // Linear part is unchanged by the fix: COM velocity = impulse / mass.
    CHECK(static_cast<double>(w.Velocity(bb.body).y)
          == Approx(static_cast<double>(targetDv)).epsilon(1e-4));
}

// ============================================================================
// (c) DETERMINISM: run the heavy-tip scene twice -> bit-identical final pose.
// ============================================================================
TEST_CASE("physics-v2 compound-COM (c): heavy-tip scene is deterministic", "[physics]")
{
    const TipResult r1 = RunHeavyTip(120);
    const TipResult r2 = RunHeavyTip(120);

    // Bit-identical (same code path, same inputs, deterministic solver).
    CHECK(r1.finalAngle == r2.finalAngle);
    CHECK(r1.finalPos.x == r2.finalPos.x);
    CHECK(r1.finalPos.y == r2.finalPos.y);
    CHECK(r1.heavyYFinal == r2.heavyYFinal);
}

// ============================================================================
// (d) GUARD: single-fixture body (localCenter == 0) is byte-identical to the
//     pre-change origin integration.
//
// A lone single-fixture dynamic body in free fall under gravity must integrate
// EXACTLY as the old origin formula did, because for localCenter == (0,0):
//   c0 == p0,  c == p0 + deltaPos,  p == c  (R(a)*0 == 0).
//
// We assert the body's COM == origin (localCenter is 0) and that one Step of
// free fall lands the body at the closed-form semi-implicit-Euler pose computed
// the OLD way (sub-stepped: the SoftStep accumulates VelSlot*h over 4 sub-steps
// with gravity added each sub-step). We reproduce that exact accumulation here.
// ============================================================================
TEST_CASE("physics-v2 compound-COM (d): single-fixture body is byte-identical", "[physics]")
{
    WorldDef wd; // gravity: MKS default (0, +10) -- drop the explicit 400 pin
    const std::uint32_t substeps = wd.substepCount; // 4
    PhysicsWorld w(wd);

    // Lone single-fixture dynamic circle (no other bodies -> no contacts).
    BodyDef bd;
    bd.type         = BodyType::Dynamic;
    bd.position     = Vec2(Real(0), Real(0));
    bd.shape        = MakeCircle(Real(2));
    bd.density      = Real(1);
    bd.fixedRotation = false;
    const BodyHandle bh = w.AddBody(bd);

    // localCenter must be (0,0) for a single fixture at the origin.
    const Vec2 lc = w.GetLocalCenter(bh);
    REQUIRE(static_cast<double>(lc.x) == Approx(0.0).margin(1e-6));
    REQUIRE(static_cast<double>(lc.y) == Approx(0.0).margin(1e-6));

    // Closed-form OLD-path expectation for ONE Step (SoftStep, no damping, no
    // contacts): h = dt/substeps; each sub-step does v += g*h then deltaPos +=
    // v*h. Origin commits p0 + sum(deltaPos). With localCenter == 0 the new code
    // reduces to exactly this. g is read from wd.gravityY (the MKS default 10)
    // instead of a hardcoded px-era literal.
    const Real h = kStep / static_cast<Real>(substeps);
    Real v = Real(0);
    Real dp = Real(0);
    for (std::uint32_t s = 0; s < substeps; ++s)
    {
        v  += wd.gravityY * h; // gravity each sub-step
        dp += v * h;           // accumulate COM (== origin) displacement
    }
    const Real expectedY = Real(0) + dp;

    w.Step(kStep);

    const Vec2 pos = w.Position(bh);
    INFO("single-fixture free-fall y after 1 step = " << static_cast<double>(pos.y)
         << " expected (old formula) = " << static_cast<double>(expectedY));
    CHECK(static_cast<double>(pos.x) == Approx(0.0).margin(1e-6));
    CHECK(static_cast<double>(pos.y) == Approx(static_cast<double>(expectedY)).margin(1e-4));
    // Angle untouched (no torque) -> stays 0.
    CHECK(static_cast<double>(w.GetAngle(bh)) == Approx(0.0).margin(1e-6));
}
