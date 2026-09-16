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
