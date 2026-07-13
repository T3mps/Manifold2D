// Physics M6 P1.11: Phase-1 harness consolidation + kinematic determinism replay.
//
// =============================================================================
// PHASE-1 COVERAGE MAP
// =============================================================================
// Each harness block in Client/src/tests/physics_harness/main.lua is listed
// below, mapped to the C++ test file that covers it and the line-range of the
// block in main.lua (approximate, for traceability). Deferred items are
// explicitly annotated.
//
// Block                                   | Lines    | C++ coverage file
// ----------------------------------------|----------|-----------------------------
// == shapes ==                            | ~1-50    | PhysicsShapesTest.cpp
// == Geometry kernel + SAT ==             | ~51-160  | PhysicsManifoldTest.cpp
// == Circle/Capsule narrow ==             | ~161-220 | PhysicsSpecializedTest.cpp
// == GJK / conservative advance ==        | ~221-280 | PhysicsGjkTest.cpp
// == PhysicsWorld core ==                 | ~281-337 | PhysicsWorldTest.cpp
// == ContactManager events ==             | ~339-379 | PhysicsWorldTest.cpp
// == event gating ==                      | ~381-418 | PhysicsWorldTest.cpp
// == Raycast / queryAABB / overlapShape   |
//    / shapeCast / lineOfSight ==         | ~420-556 | PhysicsQueriesTest.cpp
// == CharacterController slide ==         | ~558-650 | PhysicsCharacterTest.cpp
// == TileGrid merge ==                    | ~651-760 | PhysicsTileGridTest.cpp
// == SpatialHash ==                       | ~761-810 | PhysicsBroadphaseTest.cpp
// == broadphase equivalence ==            | ~811-862 | PhysicsBroadphaseTest.cpp
// == M4: determinism replay +
//    cross-broadphase ==                  | ~863-899 | THIS FILE (kinematic only;
//                                         |          |   see DEFERRED below)
// == gc audit ==                          | ~900-940 | NOT ported (no GC in C++);
//                                         |          |   steady-state alloc is
//                                         |          |   tested by design in World
// == IsoProjection ==                     | n/a      | DROPPED -- iso removed from
//                                         |          |   the engine. Not a gap.
//
// DEFERRED TO PHASE 2 / 3:
//   * DYNAMIC bodies: gravity / linear damping / rest / bounce / friction / push.
//     The harness passes gravityY=300 and applyImpulse to dynamic bodies; these
//     are solver features gated to Phase 2 (P2.1 extends PhysicsWorld for
//     dynamics). Porting the dynamic half of the determinism-replay block requires
//     a working solver; it is P3.4 ("full pipeline determinism replay incl.
//     dynamics + solver state").
//   * Island sleeping and island wake-up (Phase 2).
//   * Joints (Phase 2 / 3).
//   * Bullet / speculative CCD clamp (Phase 3).
//   * The "gc audit" block is Lua-GC-specific; it has no C++ analogue. Steady-
//     state allocation is instead contractually guaranteed by the PhysicsWorld
//     doc comment (SoA vectors only grow, broadphase pools nodes, ContactManager
//     reuses pair map + scratch; no per-Step heap after warmup).
//
// =============================================================================
// THIS FILE: kinematic determinism replay (P1.11 gate)
// =============================================================================
// PORT NOTE: the harness's M4 block (~863-899) runs a MIXED scene (statics +
// dynamics + one kinematic mover) with gravityY=300 and applies impulses to
// dynamic bodies, then hashes only the DYNAMIC bodies' positions. The dynamic
// half is DEFERRED (P3.4). This port extracts only the KINEMATIC dimension:
//
//   * Multiple kinematic movers with scripted per-step velocities.
//   * A TileGrid (GridPassability, all-solid floor row) as tile statics.
//   * Static body obstacles.
//   * ContactManager hooked (events generated deterministically).
//   * Per-step state accumulation into a 64-bit hash:
//       hash = (hash * 31 + trunc(x * 1000) + trunc(y * 1000) * 7) % 2^48
//     mirroring the harness formula verbatim.
//   * Run-twice assertion: same binary, same input -> identical final hash.
//   * Cross-broadphase assertion: Tree, Hash, and Sap strategies all produce
//     the IDENTICAL hash (sorted-pairs contract isolates broadphase strategy
//     from body trajectories -> positions are independent of broadphase choice).
//
// PRESENTATION-FREE + C++20-clean.

#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Broadphase/Passability.hpp>

using namespace Manifold2D::Physics;

namespace
{
    // Mirroring the harness formula:
    //   hash = (hash * 31 + floor(x * 1000) + floor(y * 1000) * 7) % 2^48
    //
    // std::floor on Real gives signed truncation-toward-neginf, matching Lua's
    // math.floor. The modulo keeps the value in [0, 2^48) so it matches the
    // Lua numeric domain exactly for these test magnitudes (positions stay well
    // under 2^38 after scaling by 1000).
    constexpr std::uint64_t kHashMod = (std::uint64_t(1) << 48);

    std::uint64_t HashStep(std::uint64_t h, Real x, Real y)
    {
        const auto ix = static_cast<std::int64_t>(std::floor(x * Real(1000)));
        const auto iy = static_cast<std::int64_t>(std::floor(y * Real(1000)));
        const auto contribution = static_cast<std::uint64_t>(ix + iy * 7);
        return (h * 31u + contribution) % kHashMod;
    }

    // Build a minimal GridPassability: a 16x16 grid with a solid bottom row and
    // a solid left-column barrier, all other cells open. The TileGrid derived
    // from it produces tile spans that kinematic movers bump into when driven
    // hard against the bottom or left edge. The presence of tile statics raises
    // the complexity of the determinism proof (static candidate queries fire per
    // Step for ContactManager).
    GridPassability MakeGrid()
    {
        constexpr int W = 16;
        constexpr int H = 16;
        GridPassability grid(W, H);
        // Bottom row solid (y=0).
        for (int x = 0; x < W; ++x) grid.SetSolid(x, 0, true);
        // Left column solid (x=0).
        for (int y = 0; y < H; ++y) grid.SetSolid(0, y, true);
        return grid;
    }

    // Build a WorldDef from the grid for the given broadphase strategy.
    // tileCellSize 1 m, origin (0,0) -- matching the grid layout above.
    WorldDef MakeDef(const IPassabilitySource& src, BroadphaseKind bp)
    {
        WorldDef def;
        def.broadphase   = bp;
        def.hashCellSize = Real(2);  // SpatialHash cell = 2 * tileCellSize (2-cell relation)
        def.passability  = &src;
        def.tileCellSize = Real(1);  // MKS: 1 m tiles
        def.tileOrigin   = Vec2(Real(0), Real(0));
        // Zero-g scene: the kinematic movers are driven purely by scripted
        // velocities, so gravity is deliberately held at zero (an intentional
        // scene statement, not a leftover px default).
        def.gravityX = Real(0);
        def.gravityY = Real(0);
        return def;
    }

    // Run the scripted kinematic scene under the given broadphase strategy.
    // Returns the accumulated state hash after kSteps steps.
    //
    // Scene (mirrors the M4 harness kinematic dimension):
    //   * 4 kinematic movers at staggered starting positions, scripted velocities
    //     that change every 60 steps (emulating the harness's hand-cranked loop).
    //   * One static obstacle body mid-scene (a tile alternative path is the
    //     TileGrid floor above; this adds body-vs-body contact).
    //   * ContactManager hooked to count events (unused for the hash, but exercises
    //     the event path deterministically).
    //   * Hash accumulates positions of KINEMATIC movers only, steps 1..240.
    std::uint64_t RunScene(const GridPassability& grid, BroadphaseKind bp)
    {
        constexpr int   kSteps = 240;
        constexpr Real  kDt    = Real(1) / Real(60);
        constexpr Real  kCell  = Real(1); // MKS: 1 m tiles (matches tileCellSize)

        PhysicsWorld w(MakeDef(grid, bp));

        // Hook events (determinism proof: the listener must not affect positions).
        int eventCount = 0;
        w.OnContact([&eventCount](const ContactEvent&) { ++eventCount; });

        // A single static obstacle body at cell (8,8) center = (8.5, 8.5).
        {
            BodyDef sd;
            sd.type     = BodyType::Static;
            sd.position = Vec2(Real(8) * kCell + kCell * Real(0.5),
                               Real(8) * kCell + kCell * Real(0.5));
            sd.shape    = MakeAabb(Real(0.75f), Real(0.75f));
            w.AddBody(sd);
        }

        // 4 kinematic movers, staggered.
        // Starting positions: cell (i+3, 6) center for i in [0..3], kCell = 1 m.
        // i=0 -> cell (3,6) -> (3*1+0.5, 6*1+0.5) = (3.5, 6.5)
        // i=1 -> cell (4,6) -> (4.5, 6.5)
        // i=2 -> cell (5,6) -> (5.5, 6.5)
        // i=3 -> cell (6,6) -> (6.5, 6.5)
        BodyHandle movers[4];
        for (int i = 0; i < 4; ++i)
        {
            BodyDef kd;
            kd.type     = BodyType::Kinematic;
            kd.position = Vec2(static_cast<Real>(i + 3) * kCell + kCell * Real(0.5),
                               Real(6) * kCell + kCell * Real(0.5));
            kd.shape    = MakeCircle(Real(0.4f));
            movers[i]   = w.AddBody(kd);
        }

        // Scripted per-step velocity table: every 60 steps the movers change
        // direction/speed. Mirrors the harness hand-cranked loop pattern.
        // Phase 0 (steps  1-60):  mover 0 right, 1 up, 2 left, 3 down
        // Phase 1 (steps 61-120): mover 0 up,    1 right, 2 down, 3 left
        // Phase 2 (steps 121-180): mover 0 left,  1 down, 2 right, 3 up
        // Phase 3 (steps 181-240): mover 0 down,  1 left, 2 up,    3 right
        // Scripted speed: 2.5 m/s = 40/16 cells-per-second preserved (px 40 over a
        // 16 px cell -> MKS 2.5 m over a 1 m cell).
        const Vec2 kPhases[4][4] = {
            // Phase 0
            { Vec2(Real(2.5f), Real(0)),  Vec2(Real(0), Real(-2.5f)),
              Vec2(Real(-2.5f), Real(0)), Vec2(Real(0), Real(2.5f))  },
            // Phase 1
            { Vec2(Real(0), Real(-2.5f)), Vec2(Real(2.5f), Real(0)),
              Vec2(Real(0), Real(2.5f)),  Vec2(Real(-2.5f), Real(0)) },
            // Phase 2
            { Vec2(Real(-2.5f), Real(0)), Vec2(Real(0), Real(2.5f)),
              Vec2(Real(2.5f), Real(0)),  Vec2(Real(0), Real(-2.5f)) },
            // Phase 3
            { Vec2(Real(0), Real(2.5f)),  Vec2(Real(-2.5f), Real(0)),
              Vec2(Real(0), Real(-2.5f)), Vec2(Real(2.5f), Real(0))  },
        };

        std::uint64_t hash = 0;
        for (int step = 1; step <= kSteps; ++step)
        {
            const int phase = (step - 1) / 60;

            // Apply scripted velocities for this phase.
            for (int i = 0; i < 4; ++i)
                w.SetVelocity(movers[i], kPhases[phase][i]);

            w.Step(kDt);

            // Accumulate hash over kinematic body positions.
            for (int i = 0; i < 4; ++i)
            {
                const Vec2 p = w.Position(movers[i]);
                hash = HashStep(hash, p.x, p.y);
            }
        }

        // eventCount is computed as a side effect; ignoring its value is
        // intentional (the hash is the determinism surface; events just need
        // to fire without crashing).
        (void)eventCount;

        return hash;
    }

} // namespace

// ---------------------------------------------------------------------------
// Run-twice determinism: same binary + same input -> identical final hash.
// ---------------------------------------------------------------------------

TEST_CASE("Phase-1 harness: kinematic determinism replay -- run-twice identical hash",
          "[physics][determinism]")
{
    const GridPassability grid = MakeGrid();
    const std::uint64_t h1 = RunScene(grid, BroadphaseKind::Tree);
    const std::uint64_t h2 = RunScene(grid, BroadphaseKind::Tree);
    // Identical binary, identical input: must produce the SAME hash.
    REQUIRE(h1 == h2);
    // Sanity: hash is not trivially zero (the scene must have exercised
    // non-zero positions).
    REQUIRE(h1 != 0u);
}

// ---------------------------------------------------------------------------
// Cross-broadphase determinism: Tree / Hash / Sap all produce the same hash.
// Validates the sorted-pairs contract: broadphase strategy must NOT affect
// body trajectories.
// ---------------------------------------------------------------------------

TEST_CASE("Phase-1 harness: kinematic determinism -- Tree and Hash produce identical hash",
          "[physics][determinism]")
{
    const GridPassability grid = MakeGrid();
    const std::uint64_t hTree = RunScene(grid, BroadphaseKind::Tree);
    const std::uint64_t hHash = RunScene(grid, BroadphaseKind::Hash);
    REQUIRE(hTree == hHash);
}

TEST_CASE("Phase-1 harness: kinematic determinism -- Tree and Sap produce identical hash",
          "[physics][determinism]")
{
    const GridPassability grid = MakeGrid();
    const std::uint64_t hTree = RunScene(grid, BroadphaseKind::Tree);
    const std::uint64_t hSap  = RunScene(grid, BroadphaseKind::Sap);
    REQUIRE(hTree == hSap);
}

TEST_CASE("Phase-1 harness: kinematic determinism -- all three broadphase strategies agree",
          "[physics][determinism]")
{
    const GridPassability grid  = MakeGrid();
    const std::uint64_t hTree  = RunScene(grid, BroadphaseKind::Tree);
    const std::uint64_t hHash  = RunScene(grid, BroadphaseKind::Hash);
    const std::uint64_t hSap   = RunScene(grid, BroadphaseKind::Sap);
    // All three must agree.
    REQUIRE(hTree == hHash);
    REQUIRE(hTree == hSap);
    REQUIRE(hHash == hSap);
}
