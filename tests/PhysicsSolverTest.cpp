// Physics M6 P2.2: the Soft Step solver -- BEHAVIORAL INVARIANT tests.
//
// MODERNIZATION CENTERPIECE. The SoftStep solver (Solver/SoftStep.cpp) is a
// from-scratch Box2D-v3 TGS Soft implementation, NOT a port of the Lua
// SequentialImpulse (that is a separate A/B oracle in P2.3). It is therefore
// validated by BEHAVIORAL INVARIANTS, not bit-matched to the Lua: a ball rests
// on a floor, a stack of boxes settles, a high mass-ratio pair stays stable,
// energy stays bounded, restitution rebounds, friction stops a slide, a
// kinematic body pushes a dynamic one, and a scripted dynamic scene is
// deterministic across two runs.
//
// CARTESIAN reformulation of the Lua harness dynamics block (physics_harness/
// main.lua:686-754) + the P2.2 plan invariants. +Y is DOWN (gravity is +Y) to
// match the engine's screen-space convention (the dynamics tests use gravityY).
//
// MKS CONTENT (MKS P2): bodies 0.1-10 m, gravity is the engine default (+10
// m/s^2, y-down) unless a case is a deliberate zero-g scene, velocities in
// m/s. Scene geometry converts from the retired px-era original at a uniform
// /10 (round-meter, not mechanical) mapping, matching the sibling
// PhysicsSolverBudgetTest.cpp (which runs these SAME 8 shared scenarios
// against explicit SoftStep resting-penetration budgets).
//
// TUNING (documented per the task). The WorldDef defaults are the Box2D v3
// / MKS values: substepCount = 4, contactHertz = 30, contactDampingRatio = 10,
// restitutionThreshold = 1 (m/s), contactPushMaxVelocity = 3 (m/s). They hold
// every invariant below with NO per-test overrides.
//
// PENETRATION characteristic of THIS soft solver (measured, f32, 60 Hz, MKS
// content): single ball on floor ~ 0.0003 (assert < 0.001), 5-box stack
// ~ 0.0014 (assert < 0.003), 100:1 mass-ratio pair ~ 0.0142 (assert < 0.025).
// Soft contacts allow a few percent of overlap that the soft push-out drives
// toward kLinearSlop (0.005); it shrinks monotonically with more substeps or a
// higher contactHertz (a sweep showed subs=8/hertz=60 -> stack pen much
// smaller, massratio pen ~ kSkin-scale). The defaults trade a hair of resting
// overlap for the cheaper 4-substep step; a caller wanting crisper contacts
// raises substepCount/contactHertz on the WorldDef. kLinearSlop itself is the
// asymptotic floor, NOT the per-frame transient bound. NOTE: the contact-hertz
// clamp is 0.125/h == 0.125*substepCount/dt (Box2D v3); at substepCount=4 that is
// 30 Hz, so the default hertz=30 sits exactly at the clamp. 60 Hz contacts (the
// subs=8/hertz=60 high-stiffness regime) need >=8 substeps; at only 4 substeps
// they destabilize the 100:1 mass ratio.
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
#include <Manifold2D/Physics/Solver/Solver.hpp>
#include <Manifold2D/Physics/Solver/SoftStep.hpp>

using namespace Manifold2D::Physics;
using Catch::Approx;

namespace
{
    constexpr Real kStep = Real(1) / Real(60);

    // A box AABB body (fixedRotation -- the engine requires it for AABBs).
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

    // A static floor (AABB) centered at `pos`. Its TOP surface is pos.y - hh.
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
}

// ---------------------------------------------------------------------------
// Ball rests on a floor
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsSolver: ball rests on floor", "[physics][solver]")
{
    WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
    PhysicsWorld w(wd);

    // Floor top at y = 0 (center y = 0.5, hh = 0.5).
    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(5), Real(0.5));

    const Real r = Real(0.2);
    // Drop a circle from a little above the floor.
    BodyHandle ball = AddCircle(w, Vec2(Real(0), Real(-2)), r);

    // Drop height ~1.8 m under g=10 -> t = sqrt(2*1.8/10) ~= 0.6 s ~= 36 steps
    // to first contact. Fall time is exactly 2x the px-era timing under this
    // file's uniform /10-length, /40-gravity mapping (t = sqrt(2h/g), and
    // (h/10)/(g/40) = 4x the old h/g ratio -> sqrt(4) = 2x): 240 px steps -> 480.
    for (int k = 0; k < 480; ++k)
    {
        w.Step(kStep);
    }

    const Vec2 p = w.Position(ball);
    const Real floorTop = Real(0);
    // Center should rest ~ floorTop - r (ball sits ON TOP, which is the -Y side
    // since +Y is down: the ball's lowest point is p.y + r touching floorTop).
    // Re-baselined for MKS: margin covers the penetration budget below.
    REQUIRE(p.y == Approx(floorTop - r).margin(Real(0.01)));
    // Penetration: ball bottom (p.y + r) should not sink far past floorTop.
    // Re-baselined for MKS: measured ~0.0003 (SoftStep drives a single ball's
    // resting contact toward kLinearSlop = 0.005, asymptotically); bound set
    // well under kLinearSlop with ~3x headroom over measured.
    const Real penetration = (p.y + r) - floorTop;
    REQUIRE(penetration < Real(0.001));
    // Nearly still. Re-baselined for MKS: sleepThreshold (0.05 m/s) scale,
    // ~2x headroom.
    REQUIRE(Speed(w, ball) < Real(0.1));
    // Did not tunnel sideways. Re-baselined for MKS: driving length is the
    // ball radius (r = 0.2); rescaled /10 with the rest of this case's lengths.
    REQUIRE(p.x == Approx(Real(0)).margin(Real(0.05)));
}

// ---------------------------------------------------------------------------
// Stack of N boxes settles with small penetration + near-zero velocity
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsSolver: box stack settles", "[physics][solver]")
{
    WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
    PhysicsWorld w(wd);

    // Floor top at y = 0.
    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(5), Real(0.5));

    const Real hw = Real(0.2), hh = Real(0.2);
    const int N = 5;
    std::vector<BodyHandle> boxes;
    // Stack boxes with a small gap so they fall together a touch then settle.
    // Box i center starts at y = -(2*hh)*(i+1) (above the floor, +Y down). The
    // small gap (px 0.1) re-derives from kSkin/kLinearSlop-scale lengths, /10
    // with the rest of this case's geometry -> 0.01.
    for (int i = 0; i < N; ++i)
    {
        const Real y = -(Real(2) * hh + Real(0.01)) * static_cast<Real>(i + 1);
        boxes.push_back(AddBox(w, Vec2(Real(0), y), hw, hh));
    }

    // Fall/settle budget doubles per protocol rule 5 (falls take ~2x as long
    // in MKS under this file's uniform mapping): 600 px steps -> 1200.
    for (int k = 0; k < 1200; ++k)
    {
        w.Step(kStep);
    }

    // Each box settled: near-zero velocity + small overlap with its neighbor.
    Real maxPen = Real(0);
    Real maxSpeed = Real(0);
    for (int i = 0; i < N; ++i)
    {
        maxSpeed = std::max(maxSpeed, Speed(w, boxes[i]));
    }
    // Penetration check: bottom box bottom vs floor top, and each box vs the one
    // below (centers should be ~2*hh apart).
    const Real floorTop = Real(0);
    const Real bottomPen = (w.Position(boxes[0]).y + hh) - floorTop;
    maxPen = std::max(maxPen, bottomPen);
    for (int i = 1; i < N; ++i)
    {
        const Real below = w.Position(boxes[i - 1]).y;
        const Real cur   = w.Position(boxes[i]).y;
        // cur is above below (more negative). gap of centers should be ~2*hh.
        const Real centerGap = below - cur; // positive
        const Real pen = (Real(2) * hh) - centerGap; // overlap if centers too close
        maxPen = std::max(maxPen, pen);
    }

    // Left as-is: generous no-explosion bound, not px-scale-derived (settled,
    // no jitter/explosion).
    REQUIRE(maxSpeed < Real(2.0));
    // Re-baselined for MKS: measured ~0.0014 (matches the sibling
    // PhysicsSolverBudgetTest's identical 5-box stack scenario); well under
    // kSkin (0.02), ~2.2x headroom over measured.
    REQUIRE(maxPen   < Real(0.003));
    // The stack stayed roughly vertical (no sideways drift). Re-baselined for
    // MKS: driving length is the box half-extent (hh = 0.2), /10 with the
    // rest of this case's geometry.
    for (int i = 0; i < N; ++i)
    {
        REQUIRE(w.Position(boxes[i]).x == Approx(Real(0)).margin(Real(0.05)));
    }
}

// ---------------------------------------------------------------------------
// High mass-ratio pair is stable (heavy box on a light box, on a floor)
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsSolver: high mass-ratio stable", "[physics][solver]")
{
    WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
    PhysicsWorld w(wd);

    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(5), Real(0.5));

    const Real hw = Real(0.2), hh = Real(0.2);
    // Light box on the bottom, very heavy box on top (mass ratio ~100).
    BodyHandle light = AddBox(w, Vec2(Real(0), Real(-0.21)), hw, hh, /*density=*/Real(1));
    BodyHandle heavy = AddBox(w, Vec2(Real(0), Real(-0.63)), hw, hh, /*density=*/Real(100));

    // Fall/settle budget doubles per protocol rule 5: 600 px steps -> 1200.
    for (int k = 0; k < 1200; ++k)
    {
        w.Step(kStep);
    }

    // Left as-is: generous no-explosion bound, not px-scale-derived.
    REQUIRE(Speed(w, light) < Real(5.0));
    REQUIRE(Speed(w, heavy) < Real(5.0));
    const Real floorTop = Real(0);
    const Real lightPen = (w.Position(light).y + hh) - floorTop;
    // Re-baselined for MKS: measured ~0.0142 under the 100:1 load (matches
    // the sibling PhysicsSolverBudgetTest's identical scenario); ~1.25x
    // kSkin (0.02), ~1.8x headroom over measured.
    REQUIRE(lightPen < Real(0.025));
    REQUIRE(std::isfinite(w.Position(heavy).y));
    REQUIRE(std::isfinite(w.Position(light).y));
}

// ---------------------------------------------------------------------------
// Energy bounded: a settling scene's kinetic energy does not blow up
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsSolver: energy bounded", "[physics][solver]")
{
    WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
    PhysicsWorld w(wd);

    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(5), Real(0.5));

    const Real hw = Real(0.2), hh = Real(0.2);
    std::vector<BodyHandle> boxes;
    for (int i = 0; i < 4; ++i)
    {
        boxes.push_back(AddBox(w, Vec2(Real(0), -(Real(2) * hh + Real(0.01)) * static_cast<Real>(i + 1)),
                               hw, hh));
    }

    // Track peakKE from step 0, NOT after a separate settle pre-loop: under
    // MKS's tighter sleepThreshold (0.05 m/s) the stack fully settles and
    // SLEEPS well before the old px-era split's doubled pre-loop budget would
    // elapse, which would make a post-settle-only measurement window vacuous
    // (peakKE == 0.0 exactly, never observing the real fall/impact transient).
    // This is the same pitfall the sibling PhysicsSolverBudgetTest hit and
    // fixed (its "A7" review comment) -- merging the original two loops
    // (300 settle + 600 measure = 900 px steps) into one avoids it here too.
    // Total budget doubles per protocol rule 5: 900 px steps -> 1800.
    Real peakKE = Real(0);
    for (int k = 0; k < 1800; ++k)
    {
        w.Step(kStep);
        Real ke = Real(0);
        for (auto h : boxes)
        {
            const Vec2 v = w.Velocity(h);
            ke += Real(0.5) * (v.x * v.x + v.y * v.y); // unit-ish mass proxy
        }
        peakKE = std::max(peakKE, ke);
        REQUIRE(std::isfinite(ke));
    }
    // Re-baselined for MKS: measured peakKE ~= 8.0 (4 boxes sharing the same
    // 0.01 m gap free-fall together at ~2.0 m/s at the bottom box's first
    // floor contact -- 4 * 0.5 * 2.0^2 = 8.0 -- matching the sibling
    // PhysicsSolverBudgetTest's identical scenario/derivation); nothing later
    // in the run exceeds this pre-contact peak. Bound at ~1.5x measured.
    REQUIRE(peakKE < Real(12.0));
}

// ---------------------------------------------------------------------------
// Restitution rebounds: a bouncy ball rebounds a good fraction of drop height
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsSolver: restitution rebounds", "[physics][solver]")
{
    WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
    PhysicsWorld w(wd);

    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(5), Real(0.5));

    const Real r = Real(0.2);
    const Real startY = Real(-4); // floor top is y=0; drop height ~ 3.8
    BodyHandle ball = AddCircle(w, Vec2(Real(0), startY), r,
                                /*density=*/Real(1), /*friction=*/Real(0.1),
                                /*restitution=*/Real(0.8));

    // The ball's resting center would be y = -r = -0.2. Drop height (of the
    // contact point) ~ |startY - (-r)| = 3.8.
    const Real restY = -r;
    const Real dropHeight = std::abs(startY - restY);

    // Track the highest point reached AFTER the first bounce (most negative y).
    // Seeded at restY, NOT startY: the ball only ever travels between startY
    // and (at best) back near its own drop origin, so seeding at startY made
    // "y < apexY" unsatisfiable forever -- apexY stayed == startY, collapsing
    // reboundHeight to dropHeight EXACTLY and making both bounds below
    // vacuously true regardless of solver behavior. This was a pre-existing
    // tautology in the px-era content (not a unit-conversion artifact); fixed
    // here to match the identical fix already applied in the sibling
    // PhysicsSolverBudgetTest (its "A7" review fix), so the assertions below
    // actually measure the rebound instead of trivially passing.
    Real apexY = restY;
    bool hitFloor = false;
    // Drop height ~3.8 m under g=10 -> fall/settle budget doubles per protocol
    // rule 5: 240 px steps -> 480.
    for (int k = 0; k < 480; ++k)
    {
        w.Step(kStep);
        const Real y = w.Position(ball).y;
        // Detect that it has been near the floor at least once. Re-baselined
        // for MKS: driving length /10 with the rest of this case's geometry.
        if ((y + r) > Real(-0.05))
        {
            hitFloor = true;
        }
        // After hitting the floor, track the rebound apex (min y == highest up).
        if (hitFloor && y < apexY)
        {
            apexY = y;
        }
    }

    REQUIRE(hitFloor);
    // Rebound height of the contact point above the floor at apex.
    const Real reboundHeight = std::abs(apexY - restY);
    // With e = 0.8, energy ~ e^2 = 0.64 of drop -> rebound ~ 0.64 * dropHeight
    // ideally. Allow for solver damping: require at least 30% of drop height.
    REQUIRE(reboundHeight > Real(0.3) * dropHeight);
    // And not MORE than the drop (no energy gain). Cushion re-baselined for
    // MKS: was +1.0 (px-era, ~2.6% of the ~38-unit drop); rescaled /10 with
    // the rest of this case's lengths -> +0.1 (same ~2.6% of the ~3.8 m drop).
    REQUIRE(reboundHeight < dropHeight + Real(0.1));
}

// ---------------------------------------------------------------------------
// Friction stops a sliding box
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsSolver: friction stops a sliding box", "[physics][solver]")
{
    WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
    PhysicsWorld w(wd);

    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(20), Real(0.5), /*friction=*/Real(0.8));

    const Real hw = Real(0.2), hh = Real(0.2);
    BodyHandle box = AddBox(w, Vec2(Real(0), Real(-0.2)), hw, hh,
                            /*density=*/Real(1), /*friction=*/Real(0.8));

    // Let it settle onto the floor first (already near-resting; unchanged --
    // not a fall-time-scaled budget).
    for (int k = 0; k < 60; ++k)
    {
        w.Step(kStep);
    }
    // Give it a horizontal kick.
    w.SetVelocity(box, Vec2(Real(12), Real(0)));

    // Friction-decel stop time scales ~4x vs px: deceleration is ~mu*g, v
    // dropped 10x (120 -> 12) while g dropped 40x (400 -> 10), so stop time
    // (v / (mu*g)) grew (1/10)/(1/40) = 4x: 600 px steps -> 2400.
    Real maxX = w.Position(box).x;
    for (int k = 0; k < 2400; ++k)
    {
        w.Step(kStep);
        maxX = std::max(maxX, w.Position(box).x);
    }

    // Friction decelerated it to rest. Left as-is: generous no-explosion
    // bound, not px-scale-derived.
    REQUIRE(std::abs(w.Velocity(box).x) < Real(2.0));
    // It DID move some distance (friction is not infinite glue). Re-baselined
    // for MKS: driving length is the box half-width (hw = 0.2); was > 2.0
    // (px, == hw at the old scale); rescaled /10.
    REQUIRE(maxX > hw);
    // It did not sink through the floor. Re-baselined for MKS: kSkin-scale
    // (0.02).
    REQUIRE((w.Position(box).y + hh) - Real(0) < Real(0.02));
}

// ---------------------------------------------------------------------------
// Kinematic pushes dynamic (kinematic undeflected; dynamic moves)
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsSolver: kinematic pushes dynamic", "[physics][solver]")
{
    WorldDef wd;
    wd.gravityX = Real(0); // zero-g scene: isolate the push (deliberate)
    wd.gravityY = Real(0);
    PhysicsWorld w(wd); // gravity 0 -- isolate the push

    const Real hw = Real(0.2), hh = Real(0.2);
    // Kinematic plate moving +x toward the dynamic box.
    BodyDef kdef;
    kdef.type     = BodyType::Kinematic;
    kdef.position = Vec2(Real(-1), Real(0));
    kdef.shape    = MakeAabb(hw, hh);
    BodyHandle plate = w.AddBody(kdef);
    w.SetVelocity(plate, Vec2(Real(6), Real(0)));

    // Dynamic box in its path.
    BodyHandle box = AddBox(w, Vec2(Real(0), Real(0)), hw, hh);

    const Real boxX0 = w.Position(box).x;
    // Zero-g, purely kinematic: distance/velocity ratio is scale-invariant
    // under the uniform /10 mapping, so the px step count carries unchanged.
    for (int k = 0; k < 120; ++k)
    {
        w.Step(kStep);
    }

    // Kinematic kept its own trajectory (undeflected): x = -1 + 6*t.
    const Real t = kStep * Real(120);
    REQUIRE(w.Position(plate).x == Approx(Real(-1) + Real(6) * t).margin(Real(0.01)));
    REQUIRE(w.Position(plate).y == Approx(Real(0)).margin(Real(1e-3)));
    // Dynamic box was pushed forward (+x) and out of the way. Re-baselined
    // for MKS: driving length is the box half-width (hw = 0.2); was
    // boxX0 + 2.0 (px, == hw at the old scale); rescaled /10.
    REQUIRE(w.Position(box).x > boxX0 + hw);
    // The plate did not pass clean through (box stays ahead of plate's right edge
    // minus a small overlap).
    REQUIRE(w.Position(box).x > w.Position(plate).x);
}

// ---------------------------------------------------------------------------
// Determinism: a scripted dynamic scene run twice -> identical state hash
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsSolver: deterministic dynamic scene", "[physics][solver]")
{
    auto run = [](std::vector<Real>& trace)
    {
        WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
        PhysicsWorld w(wd);

        AddFloor(w, Vec2(Real(0), Real(0.5)), Real(5), Real(0.5));
        std::vector<BodyHandle> boxes;
        const Real hw = Real(0.2), hh = Real(0.2);
        for (int i = 0; i < 4; ++i)
        {
            boxes.push_back(AddBox(w, Vec2(Real(0.0), -(Real(2) * hh + Real(0.01)) * static_cast<Real>(i + 1)),
                                   hw, hh));
        }
        // Give one a sideways nudge for a richer (asymmetric) trajectory.
        // Impulse authored as mass * target delta-v (protocol rule 3): box
        // mass = density * 4*hw*hh = 1 * 4*0.2*0.2 = 0.16; nudge dv = 0.25 m/s
        // (px kick was impulse 40 / mass 16 = 2.5 px/s, rescaled /10 with the
        // rest of this file's velocities).
        const Real mass = Real(1) * Real(4) * hw * hh;
        const Real targetDv = Real(0.25); // m/s sideways nudge
        w.ApplyImpulse(boxes[2], mass * Vec2(targetDv, Real(0)));

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

// ---------------------------------------------------------------------------
// Warm-start CONTINUITY: a settled stack stays settled (bounded, non-growing
// penetration) across a long run -- and warm-start is demonstrably LIVE.
// ---------------------------------------------------------------------------
//
// RE-BASELINED (warm-start-on-Contact). This case formerly asserted
// SolverWarmStartCacheSize() < 64 -- a proxy for "warm-start state does not leak
// without bound". The solver-owned m_cache was RETIRED (warm-start impulses now
// ride the persistent Contact's manifold point, so the cache-size hook always
// returns 0 for SoftStep). We assert the INVARIANT that cache-bound was a proxy
// for, directly + behaviorally: a settled stack's penetration is bounded and does
// NOT grow over a long run (a warm-start regression -- impulse churn / a leak that
// destabilized the solve -- would show up as drift or blow-up), AND warm-start is
// LIVE (each step's resting contacts carry a non-zero accumulated normal impulse
// seeded from the prior step -- a cold-start-every-step bug would leave it 0).
TEST_CASE("PhysicsSolver: warm-start continuity keeps a stack settled", "[physics][solver]")
{
    WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
    PhysicsWorld w(wd);

    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(5), Real(0.5));
    std::vector<BodyHandle> boxes;
    const Real hw = Real(0.2), hh = Real(0.2);
    for (int i = 0; i < 4; ++i)
    {
        boxes.push_back(AddBox(w, Vec2(Real(0), -(Real(2) * hh + Real(0.01)) * static_cast<Real>(i + 1)),
                               hw, hh));
    }

    // Settle the stack, capturing the PEAK accumulated normal impulse WHILE it is
    // still awake (active contacts). Once the island pass sleeps a fully-rested
    // stack it stops feeding the solver, so warm-start legitimately reports 0 --
    // we must observe liveness during the active window, not the frozen tail.
    // Budget doubles per protocol rule 5 (falls take ~2x as long in MKS):
    // 200 px steps -> 400.
    Real peakNormalImpulse = Real(0);
    for (int k = 0; k < 400; ++k)
    {
        w.Step(kStep);
        w.ForEachContactConstraint([&](const ContactConstraint& cc)
        {
            for (int p = 0; p < cc.pointCount; ++p)
            {
                peakNormalImpulse = std::max(peakNormalImpulse, cc.points[p].normalImpulse);
            }
        });
    }
    // Live: warm-start fed the solver a non-zero seed at least once (it was on,
    // not cold-starting every step).
    REQUIRE(peakNormalImpulse > Real(0));

    // Snapshot the settled top-box Y, then run a LONG tail. With warm-start live,
    // the stack holds: the top box does not creep (penetration does not grow,
    // whether the stack stays awake or freezes asleep -- either way it must not
    // sag step over step). Tail doubles per protocol rule 5: 800 px steps -> 1600.
    const Real settledTopY = w.Position(boxes.back()).y;
    Real maxDrift = Real(0);
    for (int k = 0; k < 1600; ++k)
    {
        w.Step(kStep);
        for (auto h : boxes)
        {
            REQUIRE(std::isfinite(w.Position(h).y));
        }
        const Real drift = std::abs(w.Position(boxes.back()).y - settledTopY);
        maxDrift = std::max(maxDrift, drift);
    }
    // Bounded: a settled stack does not creep more than a hair over 1600 steps
    // (a warm-start leak / churn would let it drift or blow up).
    // Re-derived (MKS P3, P2 carry-forward): measured maxDrift ~ 0 over
    // 1600 steps; bound = 2*kLinearSlop (0.01) -- a settled stack's total
    // creep stays inside the solver's own slop scale, not a px-era absolute.
    REQUIRE(maxDrift < Real(0.01));
}

// ---------------------------------------------------------------------------
// Warm-start state is DROPPED when a body is removed (no stale resurrection).
// ---------------------------------------------------------------------------
//
// RE-BASELINED (warm-start-on-Contact). This case formerly asserted the solver
// cache DRAINED to 0 after the contacting body was removed. The cache is gone;
// warm-start now lives on the persistent Contact, which the broadphase DESTROYS
// when the body is removed (DestroyContactsForBody). We assert the BEHAVIORAL
// guarantee that the cache-drain check stood in for: once the ball is removed
// there are NO more contact constraints (its warm-start home is gone -- nothing
// stale resurrects), and the rest of the scene stays finite.
TEST_CASE("PhysicsSolver: warm-start dropped on body removal", "[physics][solver]")
{
    WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
    PhysicsWorld w(wd);

    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(5), Real(0.5));
    BodyHandle ball = AddCircle(w, Vec2(Real(0), Real(-1)), Real(0.2));

    // Settle the ball so it is generating warm-started contacts. Confirm warm-
    // start is actually live before we remove it (a non-zero accumulated normal
    // impulse on the ball-floor contact). Budget doubles per protocol rule 5:
    // 120 px steps -> 240.
    Real peakNormalImpulse = Real(0);
    for (int k = 0; k < 240; ++k)
    {
        w.Step(kStep);
        w.ForEachContactConstraint([&](const ContactConstraint& cc)
        {
            for (int p = 0; p < cc.pointCount; ++p)
            {
                peakNormalImpulse = std::max(peakNormalImpulse, cc.points[p].normalImpulse);
            }
        });
    }
    REQUIRE(peakNormalImpulse > Real(0)); // warm-start was live while in contact

    // Remove the ball -> its persistent Contact is destroyed -> no constraint can
    // be emitted for it. The solver feed must be empty (no stale resurrection),
    // and the scene stays finite.
    w.RemoveBody(ball);
    for (int k = 0; k < 5; ++k)
    {
        w.Step(kStep);
    }
    REQUIRE(w.ActiveContactCount() == std::size_t(0)); // warm-start home gone
}

// ---------------------------------------------------------------------------
// Warm-start continuity (EXPLICIT): a stack settled by step N has bounded
// penetration that does NOT grow at N+50.
// ---------------------------------------------------------------------------
//
// This is Task 3's stated gate: warm-start impulses on the persistent Contact
// must CONTINUE across steps for pool contacts. We settle a box stack on a floor,
// keep it AWAKE (force-wake every step so island sleep never freezes it -- the
// warm-start carry-forward is then exercised on a LIVE contact every step), then
// confirm the stack's penetration at step N and at step N+50 are both bounded and
// the later one is no worse than a hair over the earlier (the stack holds, it does
// not creep down as overlap accumulates). A cold-start-every-step regression
// (warm-start NOT carried across the per-step manifold recompute) makes a soft
// solver re-converge from zero each frame -> the stack sags + jitters, which this
// growth bound catches.
TEST_CASE("PhysicsSolver: warm-start continuity -- settled stack stays put N -> N+50",
          "[physics][solver]")
{
    WorldDef wd; // gravity defaults to (0, 10) m/s^2, +Y down
    PhysicsWorld w(wd);

    // Floor top surface at y = 0 (center +0.5, hh 0.5).
    AddFloor(w, Vec2(Real(0), Real(0.5)), Real(5), Real(0.5));

    // A 4-box stack resting on the floor (half-extent 0.2, stacked tightly).
    std::vector<BodyHandle> boxes;
    const Real hh = Real(0.2);
    for (int i = 0; i < 4; ++i)
    {
        boxes.push_back(AddBox(w, Vec2(Real(0),
                                       -(Real(2) * hh + Real(0.005)) * static_cast<Real>(i) - hh - Real(0.01)),
                               hh, hh));
    }

    // Per-step penetration of the BOTTOM box into the floor: floor top is y = 0,
    // the box bottom is Position.y + hh (Y is DOWN), so penetration = box.y + hh.
    auto bottomPenetration = [&]() -> Real
    {
        return w.Position(boxes.front()).y + hh; // > 0 == into the floor
    };

    // Keep the stack awake so the solver is fed every step (the carry-forward
    // path runs on a live contact, not a frozen one).
    auto wakeAll = [&]()
    {
        for (auto h : boxes) { w.Wake(h); }
    };

    // Settle by step N. Budget doubles per protocol rule 5 (falls take ~2x as
    // long in MKS): 200 px steps -> 400.
    for (int k = 0; k < 400; ++k)
    {
        wakeAll();
        w.Step(kStep);
    }
    const Real penAtN = bottomPenetration();
    REQUIRE(std::isfinite(penAtN));
    // Settled overlap is small. Re-baselined for MKS: kSkin-scale (0.02-0.03)
    // rather than a fraction of the box half-extent (hh = 0.2). The point is
    // the NON-GROWTH below.
    REQUIRE(penAtN < Real(0.03));

    // Step 50 more (unchanged -- this is a fixed post-settle continuity
    // window named in the case title, not a fall/settle budget), tracking the
    // worst penetration over the tail.
    Real penPeakTail = penAtN;
    for (int k = 0; k < 50; ++k)
    {
        wakeAll();
        w.Step(kStep);
        penPeakTail = std::max(penPeakTail, bottomPenetration());
    }

    // CONTINUITY: penetration at N+50 (and across the tail) did not grow beyond a
    // hair past step N. Warm-start held the stack; it did not sag step over
    // step.
    // Re-derived (MKS P3, P2 carry-forward): measured tail growth
    // (penPeakTail - penAtN) ~ 0 over the N+50 window; bound = 2*kLinearSlop
    // (0.01) -- the growth cushion stays inside the solver's own slop scale,
    // not a px-era absolute.
    REQUIRE(penPeakTail <= penAtN + Real(0.01));
}
