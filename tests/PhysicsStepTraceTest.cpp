// Physics: PhysicsWorld::StepTraced -- the per-solver-stage snapshot seam.
//
// StepTraced runs the SAME implementation as Step (PhysicsWorld::StepImpl) with
// a snapshot hook installed on the solver's stage walk, so these cases pin two
// different things: the SHAPE of the trace (the D1 stage order, 5 sub-step
// stages x substepCount then Restitution + StoreImpulses) and the FIDELITY of
// its contents (the last snapshot is the state the step leaves behind; the
// world Step leaves is bit-identical either way; a Relax pass never has more
// normal-direction closing velocity than the biased Solve it follows).
//
// +Y is DOWN (gravity is +10 m/s^2 in y), MKS content, WorldDef defaults
// (substepCount 4) unless a case says otherwise. PRESENTATION-FREE.

#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/StepTrace.hpp>
#include <Manifold2D/Physics/Solver/Solver.hpp>
#include <Manifold2D/Physics/Solver/SolverStages.hpp>   // StageType enumerators

using namespace Manifold2D::Physics;

namespace
{
    constexpr Real kStep = Real(1) / Real(60);

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

    // The shared scene: a static floor with `count` fixed-rotation boxes stacked
    // above it (2*hh + 1 cm gaps, so they fall a few cm and land in order).
    // sleepThreshold 0 on the WorldDef means `sleepVel < threshold` is never true
    // (sleepVel is a magnitude, IslandManager.cpp:252-257), so no island ever
    // sleeps -- a slept island ZEROES its members' velocities after the solve,
    // which would make every comparison below trivially or wrongly satisfied.
    std::vector<BodyHandle> BuildStack(PhysicsWorld& w, int count)
    {
        AddFloor(w, Vec2(Real(0), Real(0.5)), Real(5), Real(0.5));
        std::vector<BodyHandle> boxes;
        const Real hw = Real(0.2), hh = Real(0.2);
        for (int i = 0; i < count; ++i)
        {
            boxes.push_back(AddBox(w, Vec2(Real(0),
                -(Real(2) * hh + Real(0.01)) * static_cast<Real>(i + 1)), hw, hh));
        }
        return boxes;
    }
}

// ---------------------------------------------------------------------------
// (1) The stage sequence
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsStepTrace: 22 snapshots in the D1 substep order", "[physics][trace]")
{
    WorldDef wd;                       // substepCount defaults to 4
    wd.sleepThreshold = Real(0);
    PhysicsWorld w(wd);
    BuildStack(w, 2);
    for (int k = 0; k < 30; ++k) { w.Step(kStep); }   // land and settle into contact

    StepTrace trace;
    w.StepTraced(kStep, trace);

    REQUIRE(trace.snapshots.size() == 22u);

    const StageType subStages[5] = { StageType::IntegrateVelocities,
                                     StageType::WarmStart,
                                     StageType::Solve,
                                     StageType::IntegratePositions,
                                     StageType::Relax };
    for (int i = 0; i < 20; ++i)
    {
        const StepTraceSnapshot& s = trace.snapshots[static_cast<std::size_t>(i)];
        CHECK(s.stage   == subStages[i % 5]);
        CHECK(s.substep == static_cast<std::uint8_t>(i / 5));
    }
    CHECK(trace.snapshots[20].stage   == StageType::Restitution);
    CHECK(trace.snapshots[20].substep == 4u);
    CHECK(trace.snapshots[21].stage   == StageType::StoreImpulses);
    CHECK(trace.snapshots[21].substep == 4u);

    // Every snapshot carries every solver row: the awake dynamics followed by
    // the kinematics (none here), so 2 boxes -> 2 bodies per snapshot.
    for (const StepTraceSnapshot& s : trace.snapshots)
    {
        CHECK(s.bodies.size() == 2u);
    }

    // StepTraced APPENDS; it does not clear the caller's trace.
    w.StepTraced(kStep, trace);
    CHECK(trace.snapshots.size() == 44u);
}

// ---------------------------------------------------------------------------
// (3) The last snapshot is the state the step leaves behind
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsStepTrace: the last snapshot equals the world's post-step state",
          "[physics][trace]")
{
    WorldDef wd;
    wd.sleepThreshold = Real(0);
    PhysicsWorld w(wd);
    const std::vector<BodyHandle> boxes = BuildStack(w, 3);
    for (int k = 0; k < 30; ++k) { w.Step(kStep); }
    for (const BodyHandle h : boxes) { REQUIRE(w.IsAwake(h)); }

    StepTrace trace;
    w.StepTraced(kStep, trace);
    for (const BodyHandle h : boxes) { REQUIRE(w.IsAwake(h)); }

    const StepTraceSnapshot& last = trace.snapshots.back();
    REQUIRE(last.stage == StageType::StoreImpulses);

    for (const BodyHandle h : boxes)
    {
        bool found = false;
        for (const StepTraceBody& tb : last.bodies)
        {
            if (tb.body != h.index) { continue; }
            found = true;
            const Vec2 p = w.Position(h);
            const Vec2 v = w.Velocity(h);
            // Bit-exact: the snapshot composes the pose with the SAME
            // compound-COM expression SoftStep::FinalizePositionsSoA commits,
            // and reads velocity from the same BodyState row SyncOutCompacted
            // pushes to the world. Nothing between StoreImpulses and the end of
            // Step touches an awake body's pose or velocity in this scene (no
            // bullets, no joints, no sleeping).
            CHECK(tb.position.x      == p.x);
            CHECK(tb.position.y      == p.y);
            CHECK(tb.angle           == w.GetAngle(h));
            CHECK(tb.velocity.x      == v.x);
            CHECK(tb.velocity.y      == v.y);
            CHECK(tb.angularVelocity == w.AngularVelocity(h));
        }
        CHECK(found);
    }
}

// ---------------------------------------------------------------------------
// (2) StepTraced is behaviourally identical to Step
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsStepTrace: StepTraced leaves the world bit-identical to Step",
          "[physics][trace]")
{
    // The snapshot path adds ONE write to simulation state: it runs
    // SimdSolve::StoreImpulses for every colored batch before reading, to pull
    // the lane-resident accumulated impulses onto ctx.contacts. This case is the
    // guard that the extra copy-out changes nothing -- the deterministic dynamic
    // scene from PhysicsSolverTest.cpp:473-522, 240 steps, one world on Step and
    // one on StepTraced, compared with ==.
    WorldDef wd;
    wd.sleepThreshold = Real(0);   // never sleep: all 240 steps really solve
    PhysicsWorld plain(wd);
    PhysicsWorld traced(wd);

    const std::vector<BodyHandle> pb = BuildStack(plain, 4);
    const std::vector<BodyHandle> tb = BuildStack(traced, 4);
    REQUIRE(pb.size() == tb.size());

    // The same sideways nudge the determinism case applies (box index 2).
    const Real hw = Real(0.2), hh = Real(0.2);
    const Real mass     = Real(1) * Real(4) * hw * hh;
    const Real targetDv = Real(0.25);   // m/s
    plain.ApplyImpulse(pb[2],  mass * Vec2(targetDv, Real(0)));
    traced.ApplyImpulse(tb[2], mass * Vec2(targetDv, Real(0)));

    for (int k = 0; k < 240; ++k)
    {
        plain.Step(kStep);
        StepTrace throwaway;
        traced.StepTraced(kStep, throwaway);
        REQUIRE(throwaway.snapshots.size() == 22u);
    }

    for (std::size_t i = 0; i < pb.size(); ++i)
    {
        const Vec2 pp = plain.Position(pb[i]);
        const Vec2 tp = traced.Position(tb[i]);
        const Vec2 pv = plain.Velocity(pb[i]);
        const Vec2 tv = traced.Velocity(tb[i]);
        REQUIRE(tp.x == pp.x);
        REQUIRE(tp.y == pp.y);
        REQUIRE(tv.x == pv.x);
        REQUIRE(tv.y == pv.y);
        REQUIRE(traced.GetAngle(tb[i])         == plain.GetAngle(pb[i]));
        REQUIRE(traced.AngularVelocity(tb[i])  == plain.AngularVelocity(pb[i]));
    }
}

// ---------------------------------------------------------------------------
// (4) Relax removes the bias-injected energy
// ---------------------------------------------------------------------------

namespace
{
    // Max |normal-direction relative velocity| over a snapshot's contact points,
    // computed from that snapshot's OWN bodies: the relative velocity at the
    // contact point is dv = (vA + wA x rA) - (vB + wB x rB) with
    // CrossWR(w, r) == (-w*r.y, w*r.x) and r = point - body position. A
    // static/span B (bodyB == kInvalidSlot, or a slot with no row in this
    // snapshot) contributes nothing, matching the solver's zero identity.
    Real MaxNormalVel(const StepTraceSnapshot& s)
    {
        Real worst = Real(0);
        for (const StepTraceContact& c : s.contacts)
        {
            const StepTraceBody* a = nullptr;
            const StepTraceBody* b = nullptr;
            for (const StepTraceBody& body : s.bodies)
            {
                if (body.body == c.bodyA) { a = &body; }
                if (c.bodyB != kInvalidSlot && body.body == c.bodyB) { b = &body; }
            }
            if (a == nullptr) { continue; }   // A is always an awake dynamic row
            const Vec2 rA(c.point.x - a->position.x, c.point.y - a->position.y);
            Vec2 dv(a->velocity.x - a->angularVelocity * rA.y,
                    a->velocity.y + a->angularVelocity * rA.x);
            if (b != nullptr)
            {
                const Vec2 rB(c.point.x - b->position.x, c.point.y - b->position.y);
                dv.x -= (b->velocity.x - b->angularVelocity * rB.y);
                dv.y -= (b->velocity.y + b->angularVelocity * rB.x);
            }
            const Real vn = std::fabs(dv.x * c.normal.x + dv.y * c.normal.y);
            if (vn > worst) { worst = vn; }
        }
        return worst;
    }
}

TEST_CASE("PhysicsStepTrace: a resting stack's Relax never exceeds its Solve",
          "[physics][trace]")
{
    WorldDef wd;
    wd.sleepThreshold = Real(0);   // the stack must stay awake through 120 steps
    PhysicsWorld w(wd);
    const std::vector<BodyHandle> boxes = BuildStack(w, 4);
    for (int k = 0; k < 120; ++k) { w.Step(kStep); }   // resting
    for (const BodyHandle h : boxes) { REQUIRE(w.IsAwake(h)); }

    StepTrace trace;
    w.StepTraced(kStep, trace);
    REQUIRE(trace.snapshots.size() == 22u);

    // A resting 4-box stack on a floor has contacts in EVERY stage -- if a
    // snapshot came back empty, the contact capture (not the physics) is broken.
    for (const StepTraceSnapshot& s : trace.snapshots)
    {
        CHECK(s.contacts.size() > 0u);
    }

    for (int sub = 0; sub < 4; ++sub)
    {
        const StepTraceSnapshot& solve = trace.snapshots[static_cast<std::size_t>(5 * sub + 2)];
        const StepTraceSnapshot& relax = trace.snapshots[static_cast<std::size_t>(5 * sub + 4)];
        REQUIRE(solve.stage == StageType::Solve);
        REQUIRE(relax.stage == StageType::Relax);
        REQUIRE(solve.substep == static_cast<std::uint8_t>(sub));
        REQUIRE(relax.substep == static_cast<std::uint8_t>(sub));
        // The bias-free relax pass removes the energy the biased solve injected,
        // so it never leaves MORE closing/separating normal velocity behind.
        CHECK(MaxNormalVel(relax) <= MaxNormalVel(solve));
    }
}

// ---------------------------------------------------------------------------
// (5) The write-back is load-bearing: a contact's normalImpulse is the
// ACCUMULATED value as of each stop, not the previous step's warm-start seed
// frozen in place for the whole traced step.
// ---------------------------------------------------------------------------

TEST_CASE("PhysicsStepTrace: contacts carry the accumulated impulse, not the prepare-time warm-start seed",
          "[physics][trace]")
{
    // A STILL-SETTLING stack, not the fully-converged rest state case (4)
    // traces: a few steps in, the previous step's warm-start impulse has not
    // yet converged onto the exact value gravity needs, so THIS step's Solve
    // keeps correcting it substep to substep -- the delta this case looks for
    // is guaranteed real. A perfectly warm-started equilibrium is the wrong
    // scene to pick: once WarmStart alone reproduces the exact resting
    // impulse, Solve's bias correction can land on exactly zero every
    // substep, which would make this detector flaky (or vacuously pass on a
    // frozen value, since "no change from a warm start" and "no change
    // because the copy-out is missing" would look identical).
    WorldDef wd;
    wd.sleepThreshold = Real(0);
    PhysicsWorld w(wd);
    const std::vector<BodyHandle> boxes = BuildStack(w, 4);
    for (int k = 0; k < 30; ++k) { w.Step(kStep); }   // landed, still settling
    for (const BodyHandle h : boxes) { REQUIRE(w.IsAwake(h)); }

    StepTrace trace;
    w.StepTraced(kStep, trace);
    REQUIRE(trace.snapshots.size() == 22u);

    const StepTraceSnapshot& solve0 = trace.snapshots[2];    // substep 0 Solve
    const StepTraceSnapshot& solve3 = trace.snapshots[17];   // substep 3 Solve
    REQUIRE(solve0.stage   == StageType::Solve);
    REQUIRE(solve3.stage   == StageType::Solve);
    REQUIRE(solve0.substep == 0u);
    REQUIRE(solve3.substep == 3u);
    // m_contactConstraints is built once per step (stage 2) and never
    // touched again until the next step, so contacts[i] names the SAME
    // constraint (same bodies, same fixtures, same manifold point) in both
    // snapshots of this one traced step -- no matching by id needed.
    REQUIRE(solve0.contacts.size() == solve3.contacts.size());
    REQUIRE(solve0.contacts.size() > 0u);

    // (a) the write-back produced a real non-zero impulse somewhere -- a
    // sentinel against a copy-out that silently zeroes everything instead of
    // forwarding the lane values.
    bool anyNonZero = false;
    for (const StepTraceContact& c : solve3.contacts)
    {
        if (c.normalImpulse > Real(0)) { anyNonZero = true; break; }
    }
    CHECK(anyNonZero);

    // (b) at least one contact's normalImpulse CHANGED between the two Solve
    // snapshots of this SAME step. Without the lane->constraint copy-out this
    // task adds, every colored constraint's normalImpulse stays pinned at the
    // Prepare-time warm-start seed (ConstraintGraph.cpp:1178) for the whole
    // traced step, so every pair below would compare equal and this CHECK
    // would fail -- exactly the regression this case exists to catch.
    bool anyChanged = false;
    for (std::size_t i = 0; i < solve0.contacts.size(); ++i)
    {
        if (solve0.contacts[i].normalImpulse != solve3.contacts[i].normalImpulse)
        {
            anyChanged = true;
            break;
        }
    }
    CHECK(anyChanged);
}
