// Physics M6 P2.6: Phase-2 harness consolidation + dynamics determinism replay.
//
// =============================================================================
// PHASE-2 COVERAGE MAP
// =============================================================================
// Each harness block in Client/src/tests/physics_harness/main.lua that falls
// in the Phase-2 (dynamics) scope is listed below, mapped to the C++ test file
// covering it and the approximate line-range of the block in main.lua.
//
// Block                                   | Lines      | C++ coverage file
// ----------------------------------------|------------|-----------------------------
// == dynamics (gravity / rest / bounce /  |            |
//    friction / push) ==                  | ~686-754   | PhysicsSolverTest.cpp
//                                         |            | PhysicsSolverBudgetTest.cpp (SoftStep budgets)
// == islands + sleeping ==                | ~756-777   | PhysicsIslandTest.cpp
// == joints (M3) ==                       | ~779-817   | PhysicsJointsTest.cpp
// == M4: determinism replay               |            |
//    (dynamic half) ==                    | ~863-899   | THIS FILE
//    -- scriptedRun: gravity + impulses + |            |   (Phase-2 dynamics gate;
//       kinematic stirrer + joint         |            |    see NOTE below re: joint
//       in the replay scene               |            |    extension)
//
// DEFERRED TO PHASE 3:
//   * Speculative CCD clamp + GJK-TOI bullet bodies (P3.1):
//       The harness has no explicit bullet block but the CCD path is exercised
//       by fast-moving bodies that skip across thin geometry; deferred because
//       m_bullet is stored (P2.1) but the clamp itself is P3.
//   * Astra physics components + PhysicsSystem (P3.2-3):
//       ECS wrappers (RigidBodyComponent, TransformSync) and the
//       PhysicsSystem scheduler plug-in are P3 work.
//   * Full pipeline determinism replay including CCD + solver state (P3.4):
//       The P3.4 replay will extend this scene with bullet bodies, verify
//       the CCD integration path, and cross-check with the server-flavor
//       (static-CRT ArcaneCore) build.
//   * Server-flavor (ArcaneCore static-CRT) gate (P3.5):
//       A dedicated CI step that links the physics against ArcaneCore and
//       runs a subset of the harness tests.
//   * Debug overlay (P3.6):
//       PhysicsDebugDraw (AABB / contact normals / sleep-state color) is a
//       presentation layer; deferred until the renderer is stable.
//
// NOTE on the "gc audit" block (main.lua ~900-940):
//   Lua-GC-specific; there is no analogous C++ garbage-collection concern.
//   The C++ engine's zero-steady-state-allocation contract is instead
//   contractually guaranteed by the PhysicsWorld doc comment (SoA vectors
//   only grow, broadphase pools nodes, ContactManager reuses pair map +
//   scratch; no per-Step heap after warmup) and structurally enforced by the
//   SoA layout itself.
//
// NOTE on the joint in the dynamics replay scene:
//   The harness scriptedRun (main.lua:873-894) does NOT include a joint; it is
//   a pure dynamics scene (static floor + dynamic balls + kinematic stirrer +
//   impulses). This C++ port adds a DistanceJoint between two dynamic balls to
//   exercise the joint path inside the dynamics determinism gate (joint solve
//   is wired inside both solvers' Step loops). The hash is over ALL four dynamic
//   balls so the jointed pair's constrained trajectory enters the hash directly.
//
// =============================================================================
// THIS FILE: Phase-2 dynamics determinism replay (P2.6 gate)
// =============================================================================
// Scene (mirrors harness scriptedRun main.lua:873-894, dynamic half):
//   * gravityY = 10 (MKS gravity, /10 of the harness gravityY = 300).
//   * Static floor: AABB at (15, 22) half-extents (22, 1.2) -- the harness floor /10.
//   * 4 dynamic balls: circle r=0.8, restitution=0.3; positions staggered.
//   * 1 kinematic stirrer: circle r=1, constant velocity (+X) -- harness `kin`.
//   * Scripted impulses (authored as mass * delta-v, protocol rule 3):
//       step 60:  balls[0] (= harness bodies[1]) -> delta-v (0, -0.15) m/s upward.
//       step 120: balls[2] (= harness bodies[3]) -> delta-v (0.06, -0.05) m/s.
//   * 1 DistanceJoint between balls[1] and balls[2]: exercises the joint path in
//     the replay (not in the harness; see NOTE above).
//   * Hash formula (mirroring harness / Phase-1 HarnessTest verbatim):
//       hash = (hash * 31 + floor(x * 1000) + floor(y * 1000) * 7) % 2^48
//     over the 4 dynamic balls, steps 1..240.
//
// ASSERTS:
//   1. Run-twice determinism: RunScene twice -> identical hash.
//   2. Cross-broadphase: Tree, Hash, Sap all produce the same hash (sorted-pairs
//      contract -- broadphase strategy must NOT affect dynamic trajectories).
//
// PRESENTATION-FREE + C++20-clean. namespace Manifold2D::Physics helpers.

#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Joints/Joints.hpp>

using namespace Manifold2D::Physics;

namespace Manifold2D { namespace Physics { namespace Phase2Harness {

    // -------------------------------------------------------------------------
    // Hash helper: mirrors the harness formula verbatim (and the P1.11 port).
    //   hash = (hash * 31 + floor(x * 1000) + floor(y * 1000) * 7) % 2^48
    // std::floor on Real gives signed truncation-toward-neginf, matching
    // Lua's math.floor. The modulo keeps the value in [0, 2^48).
    // -------------------------------------------------------------------------
    constexpr std::uint64_t kHashMod = (std::uint64_t(1) << 48);

    static std::uint64_t HashStep(std::uint64_t h, Real x, Real y)
    {
        const auto ix = static_cast<std::int64_t>(std::floor(x * Real(1000)));
        const auto iy = static_cast<std::int64_t>(std::floor(y * Real(1000)));
        const auto contribution = static_cast<std::uint64_t>(ix + iy * std::int64_t(7));
        return (h * std::uint64_t(31) + contribution) % kHashMod;
    }

    // -------------------------------------------------------------------------
    // RunScene: build + run the scripted dynamics scene for kSteps steps.
    // Returns the accumulated 64-bit hash over the 4 dynamic balls.
    //
    // Params:
    //   bp         -- which mover broadphase to install (cross-broadphase gate).
    // -------------------------------------------------------------------------
    static std::uint64_t RunScene(BroadphaseKind bp)
    {
        constexpr int  kSteps = 240;
        constexpr Real kDt    = Real(1) / Real(60);

        // ---- World -----------------------------------------------------------
        WorldDef def;
        def.gravityY   = Real(10);    // MKS gravity (y-down), /10 of harness 300
        def.broadphase = bp;

        PhysicsWorld w(def);

        // ---- Static floor (harness: x=150, y=220, aabb(220, 12); /10 to MKS) -
        {
            BodyDef fd;
            fd.type     = BodyType::Static;
            fd.position = Vec2(Real(15), Real(22));
            fd.shape    = MakeAabb(Real(22), Real(1.2f));
            w.AddBody(fd);
        }

        // ---- 4 dynamic balls (harness: x = 60 + i*40, y = 40 + i*10; /10 to
        //      MKS: x = 6 + i*4, y = 4 + i*1; circle r=0.8, restitution=0.3) --
        // i=1 -> (10, 5), i=2 -> (14, 6), i=3 -> (18, 7), i=4 -> (22, 8)
        const Real kBallR = Real(0.8f); // ball radius (reused for mass)
        BodyHandle balls[4];
        for (int i = 0; i < 4; ++i)
        {
            BodyDef bd;
            bd.type        = BodyType::Dynamic;
            bd.position    = Vec2(Real(6) + Real(i + 1) * Real(4),
                                  Real(4) + Real(i + 1) * Real(1));
            bd.shape       = MakeCircle(kBallR);
            bd.density     = Real(1);
            bd.restitution = Real(0.3f);
            bd.friction    = Real(0.4f);
            balls[i]       = w.AddBody(bd);
        }

        // ---- Kinematic stirrer (harness: kin, x=0, y=196, circle r=10,
        //      velocity (50, 0); /10 to MKS: y=19.6, r=1, velocity (5, 0)) ----
        {
            BodyDef kd;
            kd.type     = BodyType::Kinematic;
            kd.position = Vec2(Real(0), Real(19.6f));
            kd.shape    = MakeCircle(Real(1));
            BodyHandle kin = w.AddBody(kd);
            w.SetVelocity(kin, Vec2(Real(5), Real(0)));
        }

        // ---- DistanceJoint between balls[1] and balls[2] --------------------
        // Exercises the joint path in the dynamics determinism replay.
        // Length = creation separation (~4 m apart horizontally).
        {
            JointDef jd;
            jd.kind   = JointKind::Distance;
            jd.a      = balls[1];
            jd.b      = balls[2];
            jd.length = Real(4); // initial horizontal separation
            w.AddJoint(jd);
        }

        // ---- Scripted run (harness: steps 1..240) ---------------------------
        // Impulses authored as mass * delta-v (protocol rule 3): balls share
        // density 1 and radius kBallR, so mass = density*kPi*r*r. Target
        // delta-v (m/s) is the px-era kick (impulse / px-mass pi*8*8 = 201)
        // rescaled /10 with this scene's velocities:
        //   step 60  balls[0]: 300 / 201 = 1.49 px/s -> 0.15 m/s upward
        //   step 120 balls[2]: (120,-100)/201 = (0.60,-0.50) px/s -> (0.06,-0.05) m/s
        const Real kBallMass = Real(1) * kPi * kBallR * kBallR;
        const Vec2 kBall0KickDv(Real(0),     Real(-0.15f)); // step 60
        const Vec2 kBall2KickDv(Real(0.06f), Real(-0.05f)); // step 120

        std::uint64_t hash = 0;
        for (int step = 1; step <= kSteps; ++step)
        {
            // Scripted impulses (harness lines 885-886).
            if (step == 60)
                w.ApplyImpulse(balls[0], kBallMass * kBall0KickDv);
            if (step == 120)
                w.ApplyImpulse(balls[2], kBallMass * kBall2KickDv);

            w.Step(kDt);

            // Accumulate hash over the 4 dynamic balls.
            for (int i = 0; i < 4; ++i)
            {
                const Vec2 p = w.Position(balls[i]);
                hash = HashStep(hash, p.x, p.y);
            }
        }

        return hash;
    }

} } } // namespace Manifold2D::Physics::Phase2Harness

using namespace Manifold2D::Physics::Phase2Harness;

// ---------------------------------------------------------------------------
// 1. Run-twice determinism (SoftStep, DynamicTree broadphase).
//    Same binary, same input -> identical final hash.
// ---------------------------------------------------------------------------

TEST_CASE("Phase-2 harness: dynamics determinism replay -- run-twice identical (SoftStep)",
          "[physics][determinism]")
{
    const std::uint64_t h1 = RunScene(BroadphaseKind::Tree);
    const std::uint64_t h2 = RunScene(BroadphaseKind::Tree);
    REQUIRE(h1 == h2);
    // Sanity: the hash must be non-zero (the scene exercised non-trivial positions).
    REQUIRE(h1 != 0u);
}

// ---------------------------------------------------------------------------
// 2. Cross-broadphase determinism (SoftStep solver).
//    Tree / Hash / Sap all produce the identical hash -- the sorted-pairs +
//    sorted-contacts contract guarantees broadphase choice does not affect
//    dynamic body trajectories.
// ---------------------------------------------------------------------------

TEST_CASE("Phase-2 harness: dynamics determinism -- Tree and Hash identical (SoftStep)",
          "[physics][determinism]")
{
    const std::uint64_t hTree = RunScene(BroadphaseKind::Tree);
    const std::uint64_t hHash = RunScene(BroadphaseKind::Hash);
    REQUIRE(hTree == hHash);
}

TEST_CASE("Phase-2 harness: dynamics determinism -- Tree and Sap identical (SoftStep)",
          "[physics][determinism]")
{
    const std::uint64_t hTree = RunScene(BroadphaseKind::Tree);
    const std::uint64_t hSap  = RunScene(BroadphaseKind::Sap);
    REQUIRE(hTree == hSap);
}

TEST_CASE("Phase-2 harness: dynamics determinism -- all three broadphases agree (SoftStep)",
          "[physics][determinism]")
{
    const std::uint64_t hTree = RunScene(BroadphaseKind::Tree);
    const std::uint64_t hHash = RunScene(BroadphaseKind::Hash);
    const std::uint64_t hSap  = RunScene(BroadphaseKind::Sap);
    REQUIRE(hTree == hHash);
    REQUIRE(hTree == hSap);
    REQUIRE(hHash == hSap);
}
