#include <catch2/catch_test_macros.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include "Support/TestWorkScheduler.hpp"
#include <thread>
#include <array>
#include <cstdint>
#include <Manifold2D/Physics/Solver/BodyState.hpp>
#include <Manifold2D/Physics/Solver/SolverStages.hpp>

using namespace Manifold2D::Physics;

TEST_CASE("PhysicsWorld accepts an executor and steps with it (serial default)", "[physics][solvermt]")
{
    WorldDef wd;
    wd.gravityX = Real(0); // zero-g: API-mechanics test, no physics content
    wd.gravityY = Real(0);
    PhysicsWorld w(wd);
    Manifold2D::SerialWorkScheduler serial;
    w.SetExecutor(&serial);                 // explicit serial
    REQUIRE(w.Executor() == &serial);       // always-non-null invariant: returns the injected executor
    w.Step(1.0f / 60.0f);                   // must not crash; serial path unchanged
    SUCCEED("stepped with an injected executor");

    WorldDef wd2;
    wd2.gravityX = Real(0); // zero-g: API-mechanics test, no physics content
    wd2.gravityY = Real(0);
    PhysicsWorld w2(wd2);
    w2.SetExecutor(nullptr);                // null -> falls back to the world's serial default
    REQUIRE(w2.Executor() != nullptr);      // always-non-null invariant: serial fallback, never null
    w2.Step(1.0f / 60.0f);
    SUCCEED("stepped with null executor (serial fallback)");
}

namespace
{
    // Deterministic pile: a static floor + 500 dynamic boxes in a 10-wide column.
    // 500 bodies + the dense contact pile size the per-step SolverStage list (Gap 1.2)
    // into many blocks per stage: BuildStages targets ~4*WorkerCount() blocks (body
    // stages min 32 bodies/block -> ceil(500/32)=16 blocks; colored contact stages
    // min 4 batches/block), so at WorkerCount() > 1 the persistent solver region runs
    // GENUINE within-color (ring-CAS block-stealing) and within-body MT -- not a single
    // inline block. (The old kSolverBodyGrain/kSolverColorGrain per-color ParallelFor
    // dispatch is gone; the block partition replaced it.) The whole substep loop is
    // dispatched ONCE per step via ParallelFor(workerCount, 1, ...): begin==0 is the
    // main, begin>0 are thieves. Because every block is body-disjoint, the block count
    // -- hence the worker count -- cannot change any float, so the MT result is
    // byte-identical to serial. Gravity pulls in +Y; floor at y=5 (half-extent 0.5),
    // boxes spawn at negative Y (above the floor) -- same convention as
    // PhysicsAwakeSetTest. Returns (x, y, angle, vx, vy) per body after `steps`:
    // serial == enki(1) == enki(N), byte-identical across executors/thread counts.
    std::vector<float> RunPile(Manifold2D::IWorkScheduler* exec, int steps)
    {
        WorldDef wd; // gravityY inherits the MKS default (+10)
        PhysicsWorld w(wd);
        w.SetExecutor(exec);

        // Static floor: half-extents 25 x 0.5 at origin (wide enough for 10 columns).
        {
            BodyDef fd;
            fd.type     = BodyType::Static;
            fd.position = Vec2(Real(0), Real(5));
            fd.shape    = MakeAabb(Real(25), Real(0.5));
            w.AddBody(fd);
        }

        // 500 dynamic boxes in a 10-column grid, spawned above the floor.
        // Column index c = i % 10  -> x in [-4.5 .. 4.5] step 1.0
        // Row index r    = i / 10  -> y = -1 - r*1.2 (above the floor)
        // 500 bodies / 10 columns = 50 rows -> column height ~60 units above floor.
        std::vector<BodyHandle> bodies;
        bodies.reserve(500);
        for (int i = 0; i < 500; ++i)
        {
            const int c = i % 10;
            const int r = i / 10;
            BodyDef bd;
            bd.type     = BodyType::Dynamic;
            bd.position = Vec2(static_cast<Real>(c - 5) * Real(1.0),
                               Real(-1) - static_cast<Real>(r) * Real(1.2));
            bd.shape    = MakeAabb(Real(0.4), Real(0.4));
            bd.density       = Real(1);
            bd.friction      = Real(0.3);
            bd.fixedRotation = true; // AABB shapes require fixedRotation (axis-aligned by definition)
            bodies.push_back(w.AddBody(bd));
        }

        for (int s = 0; s < steps; ++s)
        {
            w.Step(Real(1) / Real(60));
        }

        std::vector<float> out;
        out.reserve(bodies.size() * 5u);
        for (auto h : bodies)
        {
            const Vec2 p = w.Position(h);
            const Vec2 v = w.Velocity(h);
            out.push_back(static_cast<float>(p.x));
            out.push_back(static_cast<float>(p.y));
            out.push_back(static_cast<float>(w.GetAngle(h)));
            out.push_back(static_cast<float>(v.x));
            out.push_back(static_cast<float>(v.y));
        }
        return out;
    }
} // namespace

TEST_CASE("solver thread-count invariance: serial == enki(1) == enki(N)", "[physics][determinism][solvermt]")
{
    Manifold2D::SerialWorkScheduler serial;
    // Drive Manifold2D through a std::thread pool via the IWorkScheduler seam
    // (test-only; the library creates no threads). serial == 1 worker == N workers.
    const std::uint32_t hw = std::thread::hardware_concurrency();
    Manifold2D::Testing::TestWorkScheduler oneSched(1);
    Manifold2D::Testing::TestWorkScheduler manySched(hw > 1u ? hw : 2u);

    const auto a = RunPile(&serial,    120);
    const auto b = RunPile(&oneSched,  120);
    const auto c = RunPile(&manySched, 120);

    INFO("workers=" << manySched.WorkerCount());
    REQUIRE(manySched.WorkerCount() >= 1);
    if (manySched.WorkerCount() <= 1u)
    {
        WARN("single worker: MT thief path not exercised this run");
    }
    REQUIRE(a.size() == b.size());
    REQUIRE(a == b);
    REQUIRE(a == c);
}

TEST_CASE("BodyState is a 32-byte 32-aligned AoS row", "[physics][solvermt]")
{
    STATIC_REQUIRE(sizeof(Manifold2D::Physics::BodyState) == 32);
    STATIC_REQUIRE(alignof(Manifold2D::Physics::BodyState) == 32);
    Manifold2D::Physics::BodyStateStore s; s.Resize(33);
    REQUIRE(reinterpret_cast<std::uintptr_t>(s.data()) % 32u == 0u);  // aligned storage
}

// Stage-coverage unit test for the ring-CAS claim loop (Gap 1.1). Exercises the
// block-claim protocol in isolation via the test-only ExecuteStageForTest shim:
// a single (serial main) caller must claim + run EVERY block exactly once and
// drive the stage's completionCount up to blockCount.
TEST_CASE("ExecuteStage visits every block exactly once (serial main)", "[physics][solvermt]")
{
    using namespace Manifold2D::Physics;
    std::array<SolverBlock, 5> blk{};            // 5 blocks
    for (int i = 0; i < 5; ++i) { blk[i].begin = i; blk[i].end = i + 1; blk[i].syncIndex.store(0); }
    SolverStage st{}; st.type = StageType::IntegrateVelocities; st.blocks = blk.data(); st.blockCount = 5;
    std::array<int, 5> hits{};
    ExecuteStageForTest(st, /*prevSync*/0, /*curSync*/1, [&](int b) { hits[b]++; });
    for (int h : hits) REQUIRE(h == 1);          // each block run once
    REQUIRE(st.completionCount.load() == 5);
}
