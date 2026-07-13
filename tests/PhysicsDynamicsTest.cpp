// Physics M6 P2.1: dynamic-body velocity integration (NO solver yet).
//
// PORT NOTE: P2.1 OPENED Phase 2 (dynamics). It extended PhysicsWorld with the
// dynamics SoA (mass/invMass/invInertia, angle/angVel, restitution/friction/
// damping/sleep) + AddBody dynamics params + the Step velocity/position
// integration + the Body dynamics accessors. These tests are PURE integration:
// a dynamic body free-falls under constant gravity with NO collision response.
//
// UPDATED FOR P2.2 (Box2D v3 TGS Soft): the solver now OWNS dynamic velocity +
// position integration, folded INTO its sub-step loop (gravity + position per
// sub-step). For a body with NO contacts this is still semi-implicit Euler, but
// REGROUPED over substepCount sub-steps of length h = dt/N. The analytic
// reference below is updated to the sub-stepped closed form (the old once-per-
// step form was the P2.1 integration timing, which the v3 restructure replaced).
//
// ANALYTIC reference (constant gravity g, start at rest v0=0, pos0, N sub-steps):
//   per sub-step: v += g*h ; deltaPos += v*h    (h = dt/N)
//   VELOCITY is unchanged from once-per-step: v_k = g*dt*k (N applications of
//     g*h == g*dt per full step).
//   POSITION regroups: pos_k = pos0 + g*dt^2 * [ k(k-1)/2 + k*(N+1)/(2N) ].
//   (At N -> infinity this tends to the continuous g*dt^2*k^2/2; at N == 1 it
//    is exactly the P2.1 once-per-step g*dt^2*k(k+1)/2.)
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

// The solver scaffolding header must compile + the interface must exist (it is
// NOT called in P2.1 -- this is a presence + design-anchor check).
#include <Manifold2D/Physics/Solver/Solver.hpp>

using namespace Manifold2D::Physics;
using Catch::Approx;

namespace
{
    constexpr Real kStep = Real(1) / Real(60);
    // kPi comes from Manifold2D::Physics::kPi (PhysicsTypes.hpp) via the using-
    // namespace directive above -- do not redeclare it here (ambiguous symbol).

    // Build a world with gravity and one free-falling dynamic circle at origin.
    BodyHandle AddFallingBody(PhysicsWorld& w, Vec2 pos = Vec2(Real(0), Real(0)))
    {
        BodyDef def;
        def.type     = BodyType::Dynamic;
        def.position = pos;
        def.shape    = MakeCircle(Real(0.5));
        def.density  = Real(1);
        return w.AddBody(def);
    }
}

// ---------------------------------------------------------------------------
// Free-fall under constant gravity matches the semi-implicit Euler closed form
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsDynamics: free-fall matches sub-stepped semi-implicit Euler", "[physics][dynamics]")
{
    WorldDef wd; // gravity defaults to (0, 10) MKS -- Box2D v3 default, y-down
    PhysicsWorld w(wd);

    const Vec2 pos0(Real(1), Real(-0.5));
    BodyHandle h = AddFallingBody(w, pos0);

    const Real g  = Real(10);
    const Real dt = kStep;
    const Real N  = static_cast<Real>(wd.substepCount); // 4 by default

    // Body starts at rest, awake, at pos0.
    REQUIRE(w.GetType(h) == BodyType::Dynamic);
    REQUIRE(w.IsAwake(h));
    REQUIRE(w.Velocity(h).x == Approx(Real(0)));
    REQUIRE(w.Velocity(h).y == Approx(Real(0)));

    const int kN = 30;
    for (int k = 1; k <= kN; ++k)
    {
        w.Step(dt);

        // v_k = g*dt*k (sub-stepping does not change the per-step velocity gain)
        // pos_k = pos0 + g*dt^2 * [ k(k-1)/2 + k*(N+1)/(2N) ]
        const Real kf     = static_cast<Real>(k);
        const Real expV   = g * dt * kf;
        const Real expPos = pos0.y + g * dt * dt
                          * (kf * (kf - Real(1)) * Real(0.5)
                             + kf * (N + Real(1)) / (Real(2) * N));

        const Vec2 v = w.Velocity(h);
        const Vec2 p = w.Position(h);

        // No horizontal force -> x stays put; gravity acts on y only.
        REQUIRE(v.x == Approx(Real(0)).margin(Real(1e-4)));
        REQUIRE(p.x == Approx(pos0.x).margin(Real(1e-3)));

        REQUIRE(v.y == Approx(expV).margin(Real(1e-3) + std::abs(expV) * Real(1e-4)));
        // generous margin: semi-implicit Euler position error accumulates ~O(k*dt^2)
        REQUIRE(p.y == Approx(expPos).margin(Real(1e-2) + std::abs(expPos) * Real(1e-4)));
    }
}

// ---------------------------------------------------------------------------
// prev/curr snapshot still drives DrawPosition for a falling body
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsDynamics: DrawPosition lerps the fall", "[physics][dynamics]")
{
    WorldDef wd; // gravity defaults to (0, 10) MKS
    PhysicsWorld w(wd);

    BodyHandle h = AddFallingBody(w);
    w.Step(kStep); // one step: prev = start (0), curr = first integrated pos

    const Vec2 prev = w.DrawPosition(h, Real(0)); // alpha 0 -> prev snapshot
    const Vec2 curr = w.DrawPosition(h, Real(1)); // alpha 1 -> current pos
    const Vec2 mid  = w.DrawPosition(h, Real(0.5));

    REQUIRE(prev.y == Approx(Real(0)).margin(Real(1e-5)));
    REQUIRE(curr.y == Approx(w.Position(h).y));
    REQUIRE(mid.y  == Approx((prev.y + curr.y) * Real(0.5)));
    REQUIRE(curr.y > prev.y); // fell downward (positive gravity)
}

// ---------------------------------------------------------------------------
// Linear damping reduces velocity vs the undamped fall (exact recurrence ref)
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsDynamics: linear damping decays velocity", "[physics][dynamics]")
{
    const Real g  = Real(10);
    const Real dt = kStep;
    const Real damp = Real(2);

    WorldDef wd; // gravity defaults to (0, 10) MKS, matching g above

    // Damped body.
    PhysicsWorld wDamp(wd);
    BodyDef ddef;
    ddef.type          = BodyType::Dynamic;
    ddef.shape         = MakeCircle(Real(0.5));
    ddef.linearDamping = damp;
    BodyHandle hd = wDamp.AddBody(ddef);

    // Undamped body (same gravity, no damping) -- the comparison baseline.
    PhysicsWorld wPlain(wd);
    BodyHandle hp = AddFallingBody(wPlain);

    // Analytic damped recurrence (gravity + damping run PER SUB-STEP). Box2D v3
    // order (solver.c:102-106, B2 parity fix): damp the OLD velocity, THEN add the
    // undamped gravity delta -- v = v/(1 + damp*h) + g*h, with h = dt/N, N sub-steps
    // per full Step. (Before B2 the gravity was damped too, v = (v + g*h)/(1+damp*h);
    // the two agree only in the d == 0 limit.)
    const std::uint32_t N = WorldDef{}.substepCount; // 4
    const Real h  = dt / static_cast<Real>(N);
    const Real fh = Real(1) / (Real(1) + damp * h);
    Real refV = Real(0);

    // Track the recurrence for the first 60 steps (still far from terminal).
    const int kN = 60;
    for (int k = 1; k <= kN; ++k)
    {
        wDamp.Step(dt);
        wPlain.Step(dt);
        for (std::uint32_t s = 0; s < N; ++s)
        {
            refV = refV * fh + g * h;
        }

        const Real vDamp  = wDamp.Velocity(hd).y;
        const Real vPlain = wPlain.Velocity(hp).y;

        // Damped velocity tracks the closed-form recurrence.
        REQUIRE(vDamp == Approx(refV).margin(Real(1e-3) + std::abs(refV) * Real(1e-4)));
        // Damping always leaves the body slower than the undamped baseline.
        REQUIRE(vDamp < vPlain);
    }

    // The discrete fixed point of v = v*f + g*h (gravity undamped, added AFTER
    // damping -- Box2D v3 order) is g/damp + g*h: v(1-f) = g*h, and
    // 1-f = d*h/(1+d*h) = d*h*f, so v = g*h/(d*h*f) = g/(d*f) = g/d + g*h. (The
    // old damp-the-gravity form sat at exactly g/d, the continuous terminal; the
    // B2 parity fix shifts it up by one sub-step of undamped gravity, g*h.)
    // Convergence is geometric in f, so step long enough for the transient to die.
    for (int k = 0; k < 400; ++k)
    {
        wDamp.Step(dt);
    }
    const Real terminal = g / damp + g * h;
    // Margin 0.2% (~0.010) is comfortably below the old-vs-new fixed-point gap g*h
    // (~0.042) yet ~100x above the converged f32 error, so this line INDEPENDENTLY
    // catches a revert to the old damp-the-gravity terminal (g/damp), not only the
    // recurrence loop above.
    REQUIRE(wDamp.Velocity(hd).y == Approx(terminal).margin(terminal * Real(0.002)));
}

// ---------------------------------------------------------------------------
// ApplyImpulse changes velocity by impulse * invMass and wakes the body
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsDynamics: ApplyImpulse adds impulse/mass and wakes", "[physics][dynamics]")
{
    WorldDef wd;
    wd.gravityX = Real(0); // zero-g scene: isolate impulse response from gravity
    wd.gravityY = Real(0);
    PhysicsWorld w(wd);

    BodyDef def;
    def.type    = BodyType::Dynamic;
    def.shape   = MakeCircle(Real(0.5));
    def.density = Real(1);
    BodyHandle h = w.AddBody(def);

    // mass = density * pi * r^2 = pi * 0.25 (r = 0.5); invMass = 1/mass.
    const Real r       = Real(0.5);
    const Real mass    = Real(1) * kPi * r * r;
    const Real invMass = Real(1) / mass;

    REQUIRE(w.Velocity(h).x == Approx(Real(0)));

    // Author the impulse as mass * target delta-v (2, -1 m/s) rather than a
    // magic impulse literal.
    const Vec2 targetDv(Real(2), Real(-1));
    const Vec2 impulse = mass * targetDv;
    w.ApplyImpulse(h, impulse);

    REQUIRE(w.Velocity(h).x == Approx(impulse.x * invMass).margin(Real(1e-4)));
    REQUIRE(w.Velocity(h).y == Approx(impulse.y * invMass).margin(Real(1e-4)));
    REQUIRE(w.IsAwake(h));

    // Off-center impulse imparts angular velocity: angVel += cross(r, i)*invI.
    const Real angBefore = w.GetAngle(h);
    // Body at origin; apply at world point (0, 1) with a +x impulse -> negative
    // z torque (cross((0,1),(j,0)) = 0*0 - 1*j = -j). With invInertia > 0 this
    // changes the body's angle after a Step.
    const Vec2 targetDv2(Real(1), Real(0)); // second impulse's target delta-v
    w.ApplyImpulse(h, mass * targetDv2, Vec2(Real(0), Real(1)));
    w.Step(kStep);
    REQUIRE(w.GetAngle(h) != Approx(angBefore)); // rotated (invInertia > 0)
}

// ---------------------------------------------------------------------------
// Static / kinematic bodies are UNAFFECTED by gravity (no dynamics regression)
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsDynamics: static + kinematic ignore gravity", "[physics][dynamics]")
{
    WorldDef wd; // gravity defaults to (0, 10) MKS
    PhysicsWorld w(wd);

    BodyDef sdef;
    sdef.type     = BodyType::Static;
    sdef.position = Vec2(Real(0), Real(0));
    sdef.shape    = MakeAabb(Real(2), Real(2));
    BodyHandle hs = w.AddBody(sdef);

    BodyDef kdef;
    kdef.type     = BodyType::Kinematic;
    kdef.position = Vec2(Real(5), Real(0));
    kdef.shape    = MakeCircle(Real(0.5));
    BodyHandle hk = w.AddBody(kdef);
    w.SetVelocity(hk, Vec2(Real(3), Real(0))); // pure horizontal kinematic motion

    for (int k = 0; k < 20; ++k)
    {
        w.Step(kStep);
    }

    // Static never moves; gravity does not touch it.
    REQUIRE(w.Position(hs).x == Approx(Real(0)));
    REQUIRE(w.Position(hs).y == Approx(Real(0)));

    // Kinematic integrates ONLY its own velocity -- no gravity on y.
    REQUIRE(w.Position(hk).y == Approx(Real(0)).margin(Real(1e-4)));
    REQUIRE(w.Position(hk).x == Approx(Real(5) + Real(3) * kStep * Real(20)).margin(Real(1e-3)));
    REQUIRE(w.Velocity(hk).y == Approx(Real(0)));
}

// ---------------------------------------------------------------------------
// mass override scales mass + inertia (ports addBody lines 241-243)
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsDynamics: mass override sets invMass", "[physics][dynamics]")
{
    WorldDef wd;
    wd.gravityX = Real(0); // zero-g scene: isolate mass-override response from gravity
    wd.gravityY = Real(0);
    PhysicsWorld w(wd);

    BodyDef def;
    def.type    = BodyType::Dynamic;
    def.shape   = MakeCircle(Real(0.5));
    def.density = Real(1);
    def.mass    = Real(2); // override: invMass should be 1/2 regardless of density
    BodyHandle h = w.AddBody(def);

    // impulse = mass * target delta-v (1 m/s along x) -> velocity exactly 1.
    const Real mass = def.mass;
    const Vec2 targetDv(Real(1), Real(0));
    w.ApplyImpulse(h, mass * targetDv);
    REQUIRE(w.Velocity(h).x == Approx(Real(1)).margin(Real(1e-5)));
}

// ---------------------------------------------------------------------------
// Run-twice determinism: identical inputs -> bit-identical trajectory
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsDynamics: run-twice determinism", "[physics][dynamics]")
{
    auto run = [](std::vector<Real>& trace)
    {
        WorldDef wd;
        wd.gravityX = Real(2.5); // custom asymmetric gravity: both axes nonzero
        wd.gravityY = Real(10);  // is the case's point (exercises the 2D integrate)
        PhysicsWorld w(wd);

        BodyDef def;
        def.type          = BodyType::Dynamic;
        def.position      = Vec2(Real(1), Real(2));
        def.shape         = MakeCircle(Real(0.5));
        def.density       = Real(1.5);
        def.linearDamping = Real(0.3);
        BodyHandle h = w.AddBody(def);

        // impulse = mass * target delta-v (3, -1 m/s); mass = density*pi*r^2.
        const Real r    = Real(0.5);
        const Real mass = def.density * kPi * r * r;
        w.ApplyImpulse(h, mass * Vec2(Real(3), Real(-1)));

        trace.clear();
        for (int k = 0; k < 40; ++k)
        {
            w.Step(kStep);
            const Vec2 p = w.Position(h);
            const Vec2 v = w.Velocity(h);
            trace.push_back(p.x);
            trace.push_back(p.y);
            trace.push_back(v.x);
            trace.push_back(v.y);
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
