// SoftStep solver resting-penetration + stability BUDGET tests.
//
// SoftStep (Solver/SoftStep.cpp) is THE constraint solver -- the Box2D-v3 TGS
// Soft modernization centerpiece. This file runs the SAME behavioral STABILITY
// invariants as PhysicsSolverTest.cpp against it: a ball rests on a floor, a
// stack settles with bounded penetration, a high mass-ratio pair stays stable, a
// settling scene's energy stays bounded, a bouncy ball rebounds, friction stops
// a slide, a kinematic body pushes a dynamic one, and the solver is
// deterministic across two runs -- each with an explicit resting-PENETRATION
// BUDGET on the contact-heavy scenes. The point is that SoftStep is STABLE +
// bounded, so the thresholds are SoftStep-appropriate.
//
// HISTORY: this file was the P2.3 A/B cross-check -- it ran these same scenes
// against BOTH SoftStep and the Baumgarte PGS oracle (parameterized by
// WorldDef::solverKind). The oracle was retired 2026-07-03 (MKS P2): nothing
// selected it in production and SoftStep's own coverage exceeds it. The scenes +
// SoftStep budgets survive here as the single-solver budget gate.
//
// MKS CONTENT (MKS P2): bodies 0.1-10 m, gravity is the engine default (+10
// m/s^2, y-down) unless a case is a deliberate zero-g scene, velocities in
// m/s. Scene geometry converts from the retired px-era original at a uniform
// /10 (round-meter, not mechanical) mapping; densities (1 and 100) are
// UNCHANGED -- the mass-ratio scenario is the whole point of that case.
//
// PENETRATION BUDGETS (SoftStep): Box2D-v3 soft contacts drive overlap toward
// kLinearSlop (0.005). MEASURED (f32, 60 Hz) at MKS content: ball pen
// ~0.00028 (assert < 0.001), stack ~0.00137 (< 0.003), massratio ~0.01415
// (< 0.025). All FINITE + non-exploding; each bound carries ~1.5-2.2x
// headroom over the measured value (see Budget() below for the per-case
// justification against kSkin (0.02) / kLinearSlop (0.005)). The single/few-
// body scenes here settle noticeably tighter than the kSkin-scale (~0.02)
// ballpark the mass-ratio case actually lands in.
//
// CARTESIAN, +Y DOWN (gravity +Y), matching PhysicsSolverTest.cpp.
//
// PRESENTATION-FREE + C++20-clean.

#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>

using namespace Manifold2D::Physics;
using Catch::Approx;

namespace
{
    constexpr Real kStep = Real(1) / Real(60);

    // A WorldDef with +Y-down gravity (SoftStep is the only solver). Everything
    // besides gravityY inherits the MKS engine defaults (sleepThreshold 0.05,
    // restitutionThreshold 1.0, contactPushMaxVelocity 3.0, hashCellSize 1.0).
    WorldDef MakeWorldDef(Real gravityY)
    {
        WorldDef wd;
        wd.gravityY = gravityY;
        return wd;
    }

    BodyHandle AddBox(PhysicsWorld& w, Vec2 pos, Real hw, Real hh,
                      Real density = Real(1), Real friction = Real(0.4),
                      Real restitution = Real(0))
    {
        BodyDef def;
        def.type          = BodyType::Dynamic;
        def.position      = pos;
        def.shape         = MakeAabb(hw, hh);
        def.density       = density;
        def.friction      = friction;
        def.restitution   = restitution;
        def.fixedRotation = true;
        return w.AddBody(def);
    }

    BodyHandle AddCircle(PhysicsWorld& w, Vec2 pos, Real r,
                         Real density = Real(1), Real friction = Real(0.4),
                         Real restitution = Real(0))
    {
        BodyDef def;
        def.type        = BodyType::Dynamic;
        def.position    = pos;
        def.shape       = MakeCircle(r);
        def.density     = density;
        def.friction    = friction;
        def.restitution = restitution;
        return w.AddBody(def);
    }

    BodyHandle AddFloor(PhysicsWorld& w, Vec2 pos, Real hw, Real hh,
                        Real friction = Real(0.6))
    {
        BodyDef def;
        def.type     = BodyType::Static;
        def.position = pos;
        def.shape    = MakeAabb(hw, hh);
        def.friction = friction;
        return w.AddBody(def);
    }

    Real Speed(const PhysicsWorld& w, BodyHandle h)
    {
        const Vec2 v = w.Velocity(h);
        return std::sqrt(v.x * v.x + v.y * v.y);
    }

    // SoftStep resting-penetration budget (see the header block above).
    struct PenBudget
    {
        Real ball;
        Real stack;
        Real massRatio;
    };
    PenBudget Budget()
    {
        // SoftStep: crisp soft contacts, re-baselined for MKS content (MKS P2).
        // Measured (f32, 60 Hz): ball ~0.00028, stack ~0.00137, massratio
        // ~0.01415. Bounds are ~1.5-3.6x measured (rounded to a clean value):
        //   ball      0.001  -- well under kLinearSlop (0.005); ~3.6x measured
        //   stack     0.003  -- well under kSkin (0.02); ~2.2x measured
        //   massRatio 0.025  -- ~1.25x kSkin (0.02); ~1.8x measured, the
        //                       100:1 load's real overlap approaches kSkin scale
        return PenBudget{ Real(0.001), Real(0.003), Real(0.025) };
    }
}

// ---------------------------------------------------------------------------
// Ball rests on a floor (SoftStep)
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsSolverBudget: ball rests on floor", "[physics][solver][budget]")
{
    PhysicsWorld w(MakeWorldDef(Real(10)));
    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(5), Real(0.5));

    const Real r = Real(0.2);
    BodyHandle ball = AddCircle(w, Vec2(Real(0), Real(-2)), r);

    // Drop height ~1.8 m under g=10 -> t = sqrt(2*1.8/10) ~= 0.6 s ~= 36 steps
    // to first contact; 480 steps (8 s) gives ample settle headroom.
    for (int k = 0; k < 480; ++k)
    {
        w.Step(kStep);
    }

    const Vec2 p = w.Position(ball);
    const Real floorTop = Real(0);
    const PenBudget budget = Budget();

    // Rests ON TOP of the floor (within the penetration budget).
    REQUIRE(p.y == Approx(floorTop - r).margin(budget.ball + Real(0.01)));
    const Real penetration = (p.y + r) - floorTop;
    REQUIRE(penetration < budget.ball);
    REQUIRE(penetration > Real(-0.05)); // not floating well above the floor
    REQUIRE(Speed(w, ball) < Real(1.0));
    REQUIRE(p.x == Approx(Real(0)).margin(Real(0.05)));
}

// ---------------------------------------------------------------------------
// Stack of N boxes settles (SoftStep)
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsSolverBudget: box stack settles", "[physics][solver][budget]")
{
    PhysicsWorld w(MakeWorldDef(Real(10)));
    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(5), Real(0.5));

    const Real hw = Real(0.2), hh = Real(0.2);
    const int N = 5;
    std::vector<BodyHandle> boxes;
    for (int i = 0; i < N; ++i)
    {
        const Real y = -(Real(2) * hh + Real(0.01)) * static_cast<Real>(i + 1);
        boxes.push_back(AddBox(w, Vec2(Real(0), y), hw, hh));
    }

    // Cascading small-gap settle (analogous px case used 600 steps for a
    // ~2-2.2 unit fall); MKS fall/settle timescale is ~2x px at equal round-
    // meter mapping (g dropped 40x while lengths dropped 10x), so 1200 steps.
    for (int k = 0; k < 1200; ++k)
    {
        w.Step(kStep);
    }

    Real maxPen = Real(0);
    Real maxSpeed = Real(0);
    for (int i = 0; i < N; ++i)
    {
        maxSpeed = std::max(maxSpeed, Speed(w, boxes[i]));
    }
    const Real floorTop = Real(0);
    const Real bottomPen = (w.Position(boxes[0]).y + hh) - floorTop;
    maxPen = std::max(maxPen, bottomPen);
    for (int i = 1; i < N; ++i)
    {
        const Real below = w.Position(boxes[i - 1]).y;
        const Real cur   = w.Position(boxes[i]).y;
        const Real centerGap = below - cur;
        const Real pen = (Real(2) * hh) - centerGap;
        maxPen = std::max(maxPen, pen);
    }

    const PenBudget budget = Budget();
    REQUIRE(maxSpeed < Real(2.0));      // settled, no jitter/explosion
    REQUIRE(maxPen   < budget.stack);   // bounded overlap (SoftStep-appropriate)
    for (int i = 0; i < N; ++i)
    {
        REQUIRE(w.Position(boxes[i]).x == Approx(Real(0)).margin(Real(0.05)));
        REQUIRE(std::isfinite(w.Position(boxes[i]).y));
    }
}

// ---------------------------------------------------------------------------
// High mass-ratio pair is stable (SoftStep)
//
// The 100:1 mass ratio is a SoftStep showcase -- its soft contacts + the
// contactPushMaxVelocity clamp tame it at the default 4 substeps.
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsSolverBudget: high mass-ratio stable", "[physics][solver][budget]")
{
    PhysicsWorld w(MakeWorldDef(Real(10)));
    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(5), Real(0.5));

    const Real hw = Real(0.2), hh = Real(0.2);
    BodyHandle light = AddBox(w, Vec2(Real(0), Real(-0.21)), hw, hh, /*density=*/Real(1));
    BodyHandle heavy = AddBox(w, Vec2(Real(0), Real(-0.63)), hw, hh, /*density=*/Real(100));

    for (int k = 0; k < 1200; ++k)
    {
        w.Step(kStep);
    }

    REQUIRE(Speed(w, light) < Real(5.0));
    REQUIRE(Speed(w, heavy) < Real(5.0));
    const Real floorTop = Real(0);
    const Real lightPen = (w.Position(light).y + hh) - floorTop;
    const PenBudget budget = Budget();
    REQUIRE(lightPen < budget.massRatio); // bounded under the 100:1 load
    REQUIRE(std::isfinite(w.Position(heavy).y));
    REQUIRE(std::isfinite(w.Position(light).y));
}

// ---------------------------------------------------------------------------
// Energy bounded (SoftStep)
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsSolverBudget: energy bounded", "[physics][solver][budget]")
{
    PhysicsWorld w(MakeWorldDef(Real(10)));
    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(5), Real(0.5));

    const Real hw = Real(0.2), hh = Real(0.2);
    std::vector<BodyHandle> boxes;
    for (int i = 0; i < 4; ++i)
    {
        boxes.push_back(AddBox(w, Vec2(Real(0), -(Real(2) * hh + Real(0.01)) * static_cast<Real>(i + 1)),
                               hw, hh));
    }

    // MKS P2 review fix (A7): track peakKE from step 0 so the window covers
    // the real energy-injection risk -- the falling/settling transient --
    // instead of the post-sleep steady state. Under MKS's tighter
    // sleepThreshold (0.05 m/s) the stack fully settles and SLEEPS well
    // before step 600, so the old pre-settle-then-measure split measured a
    // window that was already asleep (peakKE == 0.0 exactly, a vacuous
    // bound). isfinite coverage stays over the FULL 1800-step run -- not
    // weakened; it now covers MORE of the run than before, since the old
    // 600-step pre-loop had no isfinite check at all.
    Real peakKE = Real(0);
    for (int k = 0; k < 1800; ++k)
    {
        w.Step(kStep);
        Real ke = Real(0);
        for (auto h : boxes)
        {
            const Vec2 v = w.Velocity(h);
            ke += Real(0.5) * (v.x * v.x + v.y * v.y);
        }
        peakKE = std::max(peakKE, ke);
        REQUIRE(std::isfinite(ke));
    }
    // Re-derived (MKS P2 review, A7): measured peakKE ~= 8.0 at step 11
    // (0.18 s in), NOT during any impact/settling instability -- this is
    // this test's per-body sum of 0.5*|v|^2 (note: the formula above never
    // multiplies by mass, a pre-existing quirk of this test left as-is; the
    // quantity tracked is proportional to squared speed, not literal
    // Joules). All 4 boxes share the same 0.01 m gap and so free-fall
    // TOGETHER at identical velocity until box0 (the bottom box) makes its
    // first floor contact: box0 falls ~0.206 m under g=10 -> v ~= 2.0 m/s
    // at contact -> 4 boxes * 0.5 * 2.0^2 = 8.0. Nothing later in the
    // 1800-step run exceeds this pre-contact free-fall peak (the stack
    // settles and sleeps well after). Bound set at ~1.5x measured: 12.0.
    REQUIRE(peakKE < Real(12.0));
}

// ---------------------------------------------------------------------------
// Restitution rebounds (SoftStep)
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsSolverBudget: restitution rebounds", "[physics][solver][budget]")
{
    PhysicsWorld w(MakeWorldDef(Real(10)));
    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(5), Real(0.5));

    const Real r = Real(0.2);
    const Real startY = Real(-4);
    BodyHandle ball = AddCircle(w, Vec2(Real(0), startY), r,
                                /*density=*/Real(1), /*friction=*/Real(0.1),
                                /*restitution=*/Real(0.8));

    const Real restY = -r;
    const Real dropHeight = std::abs(startY - restY);

    // MKS P2 review fix (A7): apexY must start at the authored REST y, not
    // startY. The update below is gated on hitFloor, so pre-contact steps
    // never touch apexY; seeding it with startY (the old code) made
    // "y < apexY" unsatisfiable forever (y never falls below its own drop
    // origin after a lossy bounce), so reboundHeight == dropHeight
    // identically and both bounds below were vacuously true. Seeding with
    // restY lets the post-contact minimum (highest point of the rebound,
    // since +Y is down) actually get tracked.
    Real apexY = restY;
    bool hitFloor = false;
    // Drop height ~3.8 m under g=10 -> t ~= sqrt(2*3.8/10) ~= 0.87 s ~= 53
    // steps to first contact; impact speed ~= sqrt(2*10*3.8) ~= 8.72 m/s,
    // comfortably above the MKS default restitutionThreshold (1.0 m/s), so
    // restitution actually applies. First-bounce rebound apex (e=0.8 ->
    // rebound speed ~= 0.8*8.72 = 6.97 m/s, time-to-apex ~= 0.70 s ~= 42
    // steps) arrives well within 480 steps (8 s); the loop keeps running
    // afterward, but later (smaller) bounces cannot lower the tracked
    // minimum further, so the extra steps are harmless headroom.
    for (int k = 0; k < 480; ++k)
    {
        w.Step(kStep);
        const Real y = w.Position(ball).y;
        if ((y + r) > Real(-0.05))
        {
            hitFloor = true;
        }
        if (hitFloor && y < apexY)
        {
            apexY = y;
        }
    }

    REQUIRE(hitFloor);
    const Real reboundHeight = std::abs(apexY - restY);
    // Re-derived (MKS P2 review, A7): measured reboundHeight ~= 2.389 m at
    // e=0.8, dropHeight=3.8 m (ratio ~= 0.63). Analytic single-bounce rebound
    // is e^2 * dropHeight = 0.64 * 3.8 = 2.432 m; SoftStep's soft contact
    // lands ~0.98 of analytic here (a prior e=0.6 probe measured ~0.94 of
    // analytic, same "a bit lossier than rigid rebound" pattern). Bounds
    // bracket the measured 2.389 m with real headroom on both sides:
    //   lower 1.5 m (measured is ~1.6x above it) proves a REAL rebound,
    //                far from the vacuous "> 0" the old tautology allowed
    //   upper 3.0 m (measured is ~0.8x of it; 3.0 is ~21% under
    //                dropHeight=3.8) proves energy was LOST
    REQUIRE(reboundHeight > Real(1.5)); // real rebound, not floor noise
    REQUIRE(reboundHeight < Real(3.0)); // energy was lost (well under dropHeight)
}

// ---------------------------------------------------------------------------
// Friction stops a sliding box (SoftStep)
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsSolverBudget: friction stops a sliding box", "[physics][solver][budget]")
{
    PhysicsWorld w(MakeWorldDef(Real(10)));
    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(20), Real(0.5), /*friction=*/Real(0.8));

    const Real hw = Real(0.2), hh = Real(0.2);
    BodyHandle box = AddBox(w, Vec2(Real(0), Real(-0.2)), hw, hh,
                            /*density=*/Real(1), /*friction=*/Real(0.8));

    for (int k = 0; k < 60; ++k)
    {
        w.Step(kStep);
    }
    w.SetVelocity(box, Vec2(Real(12), Real(0)));

    // Friction-decel stop time scales ~4x vs px (v dropped 10x, g dropped 40x,
    // so v/g dropped by 1/4 -> stop time x4): ~90 steps to stop; 2400 (40 s)
    // preserves the px case's ~27x headroom ratio.
    Real maxX = w.Position(box).x;
    for (int k = 0; k < 2400; ++k)
    {
        w.Step(kStep);
        maxX = std::max(maxX, w.Position(box).x);
    }

    const PenBudget budget = Budget();
    REQUIRE(std::abs(w.Velocity(box).x) < Real(2.0)); // friction stopped it
    REQUIRE(maxX > hw);                                // it DID slide (past its own half-width)
    REQUIRE((w.Position(box).y + hh) - Real(0) < budget.ball); // did not sink
}

// ---------------------------------------------------------------------------
// Kinematic pushes dynamic (SoftStep)
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsSolverBudget: kinematic pushes dynamic", "[physics][solver][budget]")
{
    WorldDef wd; // zero-g scene: isolate the push from gravity (deliberate)
    wd.gravityX = Real(0);
    wd.gravityY = Real(0);
    PhysicsWorld w(wd);

    const Real hw = Real(0.2), hh = Real(0.2);
    BodyDef kdef;
    kdef.type     = BodyType::Kinematic;
    kdef.position = Vec2(Real(-1), Real(0));
    kdef.shape    = MakeAabb(hw, hh);
    BodyHandle plate = w.AddBody(kdef);
    w.SetVelocity(plate, Vec2(Real(6), Real(0)));

    BodyHandle box = AddBox(w, Vec2(Real(0), Real(0)), hw, hh);

    const Real boxX0 = w.Position(box).x;
    // Zero-g, purely kinematic: distance/velocity ratio is scale-invariant
    // under the uniform /10 mapping, so the px step count carries unchanged.
    for (int k = 0; k < 120; ++k)
    {
        w.Step(kStep);
    }

    const Real t = kStep * Real(120);
    // Kinematic kept its own trajectory (undeflected).
    REQUIRE(w.Position(plate).x == Approx(Real(-1) + Real(6) * t).margin(Real(0.01)));
    REQUIRE(w.Position(plate).y == Approx(Real(0)).margin(Real(1e-3)));
    // Dynamic box was pushed forward (+x) and out of the way.
    REQUIRE(w.Position(box).x > boxX0 + hw);
    REQUIRE(w.Position(box).x > w.Position(plate).x);
}

// ---------------------------------------------------------------------------
// Determinism: run twice -> identical state trace (SoftStep)
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsSolverBudget: deterministic across two runs", "[physics][solver][budget]")
{
    auto run = [](std::vector<Real>& trace)
    {
        PhysicsWorld w(MakeWorldDef(Real(10)));
        AddFloor(w, Vec2(Real(0), Real(0.5)), Real(5), Real(0.5));
        std::vector<BodyHandle> boxes;
        const Real hw = Real(0.2), hh = Real(0.2);
        for (int i = 0; i < 4; ++i)
        {
            boxes.push_back(AddBox(w, Vec2(Real(0.0), -(Real(2) * hh + Real(0.01)) * static_cast<Real>(i + 1)),
                                   hw, hh));
        }
        // Impulse authored as mass * dv (protocol rule 3): box mass = density *
        // 4*hw*hh = 1 * 4*0.2*0.2 = 0.16; target nudge dv = 2.5 m/s sideways.
        const Real mass = Real(1) * Real(4) * hw * hh;
        w.ApplyImpulse(boxes[2], mass * Vec2(Real(2.5), Real(0)));

        trace.clear();
        for (int k = 0; k < 200; ++k)
        {
            w.Step(kStep);
            for (auto h : boxes)
            {
                const Vec2 p = w.Position(h);
                const Vec2 v = w.Velocity(h);
                trace.push_back(p.x);
                trace.push_back(p.y);
                trace.push_back(v.x);
                trace.push_back(v.y);
            }
        }
    };

    std::vector<Real> a, b;
    run(a);
    run(b);

    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        REQUIRE(a[i] == b[i]); // bit-identical: deterministic, index-ordered
    }
}
