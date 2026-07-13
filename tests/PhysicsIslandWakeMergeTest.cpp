// Physics Task B: restore the "island is uniformly awake" invariant.
//
// A body that settles + sleeps resting PURELY on tile spans (no static anchor
// BODY) becomes a singleton SLEEPING island. When a still-awake neighbour later
// drifts into speculative contact, the pending-merge tail grafts the two islands
// together. Historically MergeIslands preserved awake flags verbatim, so an
// already-sleeping singleton was grafted into an AWAKE island WITHOUT being
// woken (the WakeMoverPair `moverIsMoving` gate declined to wake a near-idle
// incoming body). That left a MIXED island -- an awake dynamic and a sleeping
// dynamic sharing one island -- and the very next EmitContactConstraints emitted
// a constraint for the awake side while the sleeping side of the same contact
// tripped the "no emitted constraint references a SLEEPING dynamic" assert
// (ConstraintGraph.cpp, EmitContactConstraints). In a Debug build (asserts live)
// that ABORTS the process.
//
// The fix wakes the sleeping island at merge time so the merged island is
// uniformly awake. This test drives the exact stressor the narrowphase-MT span
// suite deliberately sidestepped (it disabled sleep specifically to avoid this
// invariant): a dense pile raining into a solid tile BOWL with sleep ENABLED.
// It (1) never trips the assert / aborts, and (2) asserts the post-step
// invariant every step -- no island ever simultaneously holds an awake dynamic
// and a sleeping dynamic (the precondition that fed the bad constraint).
//
// PRESENTATION-FREE + C++23-clean.

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Broadphase/Passability.hpp> // GridPassability (tile-span bowl)

using namespace Manifold2D::Physics;

namespace
{
    constexpr Real kStep = Real(1) / Real(60);

    // Tile-span bowl geometry -- mirrors PhysicsNarrowphaseMtTest's RunCaptureSpans:
    // a 64x64 grid of 0.2 m cells shaped into a solid bowl (thick full-width floor
    // rows 40..43 -> a merged span with top face y=8.0, plus solid side walls so
    // the pile cannot escape sideways off the floor edges). Every resting body
    // sits on the merged tile SPAN -- there is NO static anchor BODY, so a settled
    // body sleeps as a SINGLETON island (the exact precondition for the bug).
    //
    // Cell size is /100 (20 -> 0.2). NarrowphaseMt's own RunCaptureSpans now uses
    // the SAME /100 divisor and the SAME constants (MKS P4) -- the mirror between
    // the two files is numeric again, not just structural: both grids are 64x64
    // cells of 0.2 m, both floor top faces sit at y=8.0. This is a deliberate
    // divergence from MKS P2-T8's tile-span precedent (which re-authored its cell
    // to the 1 m residency scale): THIS scene is a 64x64-cell bowl whose whole
    // geometry (grid extent, floor/wall rows, floor top) hangs off the cell size,
    // so /100 preserves the proven 140-body scene shape exactly instead of
    // re-deriving a new bowl. Grid row/col indices below are counts, not
    // distances -- unchanged.
    constexpr int  kSpanGridW       = 64;
    constexpr int  kSpanGridH       = 64;
    constexpr Real kSpanCellSize    = Real(0.2);
    constexpr int  kSpanFloorRow    = 40;
    constexpr int  kSpanFloorRowEnd = 43;
    constexpr int  kSpanWallRowTop  = 24;
    constexpr Real kSpanFloorTop    = Real(kSpanFloorRow) * kSpanCellSize; // 8.0

    void BuildBowl(GridPassability& grid)
    {
        for (int cy = kSpanFloorRow; cy <= kSpanFloorRowEnd; ++cy)
        {
            for (int cx = 0; cx < kSpanGridW; ++cx)
            {
                grid.SetSolid(cx, cy, true);
            }
        }
        for (int cy = kSpanWallRowTop; cy <= kSpanFloorRowEnd; ++cy)
        {
            grid.SetSolid(0, cy, true);
            grid.SetSolid(1, cy, true);
            grid.SetSolid(kSpanGridW - 2, cy, true);
            grid.SetSolid(kSpanGridW - 1, cy, true);
        }
    }

    // Rain N mixed-shape dynamic bodies INSIDE the bowl, above the floor. Spawn x
    // in [1.2,11.6] keeps every body clear of the side walls; spawn y in [5.2,7.2]
    // is well ABOVE the floor top (8.0) so they fall onto the merged tile span.
    std::vector<BodyHandle> BuildRain(PhysicsWorld& w, int n)
    {
        std::uint32_t seed = 0xC0FFEEu;
        auto rnd = [&](Real a, Real b) -> Real
        {
            seed = seed * 1664525u + 1013904223u;
            return a + (b - a) * Real((seed >> 8) & 0xFFFF) / Real(65535);
        };

        std::vector<BodyHandle> handles;
        handles.reserve(static_cast<std::size_t>(n));

        for (int j = 0; j < n; ++j)
        {
            BodyDef d;
            d.type     = BodyType::Dynamic;
            d.density  = Real(1);
            d.friction = Real(0.4);
            d.position = Vec2(rnd(Real(1.2), Real(11.6)),
                              rnd(Real(5.2), Real(7.2)));

            // Shape dims (0.05-0.11 m) land below the 0.1 m body-size floor --
            // ACCEPTED EXCEPTION for this file: every dim is still 2.5-5.5x
            // kSkin (0.02), not degenerate, and preserving the body:tile ratio
            // (vs. the 0.2 m cell) against the proven 140-body scene is what
            // the wake-merge scenario needs. LCG seed/algorithm untouched.
            if (j % 3 == 0)
            {
                d.shape = MakeCircle(rnd(Real(0.06), Real(0.11)));
            }
            else if (j % 3 == 1)
            {
                d.shape         = MakeAabb(Real(0.08), Real(0.08));
                d.fixedRotation = true;
            }
            else
            {
                // MakeCapsule(halfLen, r) -- Shapes.hpp:155 order.
                d.shape = MakeCapsule(Real(0.10), Real(0.05));
            }

            handles.push_back(w.AddBody(d));
        }

        return handles;
    }

    // The invariant under test: no island simultaneously contains an awake
    // dynamic and a sleeping dynamic. Group every dynamic body by its island root
    // and flag any root that reports BOTH an awake and a sleeping member. Returns
    // true when the invariant holds (uniformly-awake islands).
    bool IslandsUniformlyAwake(const PhysicsWorld& w,
                               const std::vector<BodyHandle>& bodies)
    {
        // root -> (sawAwake, sawAsleep).
        std::unordered_map<std::uint32_t, std::pair<bool, bool>> byRoot;
        byRoot.reserve(bodies.size());
        for (const BodyHandle h : bodies)
        {
            const std::uint32_t root = w.IslandRootOf(h.index);
            auto& seen = byRoot[root];
            if (w.IsAwake(h)) { seen.first  = true; }
            else              { seen.second = true; }
        }
        for (const auto& [root, seen] : byRoot)
        {
            (void)root;
            if (seen.first && seen.second)
            {
                return false; // mixed awake/asleep island -> invariant violated
            }
        }
        return true;
    }
}

// ---------------------------------------------------------------------------
// REALISTIC-SCENE sanity check: a dense pile raining into a solid tile BOWL with
// sleep ENABLED. Bodies settle and sleep resting purely on the merged tile span
// (singleton sleeping islands) while later arrivals drift into speculative
// contact and fire island merges. It asserts the uniform-awake invariant after
// EVERY step (no island holds both an awake and a sleeping dynamic) + that bodies
// really slept (sawAnySleep) + containment.
//
// NOTE: after the MKS scene conversion (sleepThreshold 8 px/s -> 0.05 m/s, /100
// geometry, g -> 10) this emergent pile no longer reliably hits the exact
// "lone span-sleeper + near-idle awake toucher" window that trips the guard, so
// it is NO LONGER the load-bearing regression: with the merge-time wake removed
// it still passes. The DETERMINISTIC tripwire below ("cross-awake merge wakes a
// span-sleeping singleton") is the load-bearing guard -- it fails/aborts if the
// wake is ever removed. This case stays as a broad realistic-pile smoke check.
// ---------------------------------------------------------------------------
TEST_CASE("PhysicsIslandWakeMerge: tile-span pile with sleep keeps islands uniformly awake",
          "[physics][island]")
{
    GridPassability grid(kSpanGridW, kSpanGridH);
    BuildBowl(grid);

    WorldDef wd;
    wd.gravityY     = Real(10);
    wd.passability  = &grid;
    wd.tileCellSize = kSpanCellSize;
    wd.tileOrigin   = Vec2(Real(0), Real(0));
    // Sleep ENABLED (default threshold 0.05 m/s). This is the whole point: bodies
    // resting purely on the tile span DO sleep and become singleton sleeping
    // islands -- the precondition the narrowphase-MT span suite disabled sleep to
    // avoid. Do NOT set sleepThreshold to 0 -- this precondition is now satisfied
    // by the inherited MKS default, so no pin is needed. gravityX,
    // restitutionThreshold, contactPushMaxVelocity, and hashCellSize likewise all
    // inherit the MKS WorldDef defaults (0 / 1.0 / 3.0 / 1.0).
    PhysicsWorld w(wd);

    const int kBodies = 140;
    const std::vector<BodyHandle> handles = BuildRain(w, kBodies);

    // Step long enough for the pile to fall, settle, sleep, and re-merge as late
    // arrivals drift into the sleepers. The invariant must hold after EVERY step
    // (a mixed island would already have tripped the Debug assert INSIDE Step()).
    bool sawAnySleep = false;
    for (int k = 0; k < 600; ++k)
    {
        w.Step(kStep);
        REQUIRE(IslandsUniformlyAwake(w, handles));

        for (const BodyHandle h : handles)
        {
            if (!w.IsAwake(h)) { sawAnySleep = true; break; }
        }
    }

    // The scenario is only meaningful if bodies actually slept on the span (the
    // bug's precondition). If nothing ever slept, the invariant check above is
    // vacuous and this guard fails loud rather than passing silently.
    REQUIRE(sawAnySleep);

    // Containment sanity (mirrors RunCaptureSpans): the bowl held the pile on the
    // merged span rather than tunnelling through it (smaller y is higher, +y is
    // down). Proves spans were generated + effective, so the sleepers really did
    // rest on tile spans (no static anchor body).
    bool sawBoundedY = false;
    for (const BodyHandle h : handles)
    {
        REQUIRE(w.Position(h).y < kSpanFloorTop + Real(0.8)); // 0.8 = 4 floor rows x 0.2 m cell (same structural relation as the px-era 80 = 4x20)
        sawBoundedY = true;
    }
    REQUIRE(sawBoundedY);
}

// ---------------------------------------------------------------------------
// DETERMINISTIC guard tripwire -- the LOAD-BEARING regression for the fix.
//
// The 140-body emergent pile above is a realistic-scene invariant SANITY check,
// but it does not by itself prove the merge-time wake is load-bearing: after the
// MKS scene conversion its settling dynamics no longer reliably produce the exact
// "lone span-sleeper + near-idle awake toucher" window (verified: with the wake
// guard commented out, the pile test still passes). This case reproduces that
// window DETERMINISTICALLY, so it aborts in Debug if the guard is ever removed:
//
//   * D_waker is added FIRST -> LOWER body slot. TryCreateContact orients the
//     lower dynamic slot as contact bodyA (ConstraintGraph::TryCreateContact), and
//     EmitContactConstraints only emits (and only asserts B) when A is AWAKE --
//     so the AWAKE body must be A (lower slot) and the SLEEPER must be B (higher).
//   * Both bodies settle + sleep as SINGLETON islands on the merged tile span
//     (no static anchor body). Then D_waker is Wake()d (awake, sleepTimer 0,
//     velocity still 0) and teleported to just overlap the still-sleeping
//     D_sleeper. WakeMoverPair runs in stage 2 BEFORE the solver integrates
//     gravity, so D_waker reads |v| = 0 < sleepThreshold -> its moverIsMoving
//     gate declines to wake D_sleeper, yet the new contact touch-begins.
//   * The touch-begin queues a cross-awake island merge (A awake, B asleep).
//     Guard OFF: MergeIslands grafts the sleeping singleton into the awake island
//     -> mixed island -> EmitContactConstraints trips the no-sleeping-dynamic
//     assert and aborts INSIDE Step(). Guard ON: the merge wakes D_sleeper's
//     island first, so the merged island is uniformly awake and the step completes.
// ---------------------------------------------------------------------------
TEST_CASE("PhysicsIslandWakeMerge: cross-awake merge wakes a span-sleeping singleton",
          "[physics][island]")
{
    GridPassability grid(kSpanGridW, kSpanGridH);
    BuildBowl(grid);

    WorldDef wd;
    wd.gravityY     = Real(10);
    wd.passability  = &grid;
    wd.tileCellSize = kSpanCellSize;
    wd.tileOrigin   = Vec2(Real(0), Real(0));
    PhysicsWorld w(wd);

    // Two flat boxes (half-extent 0.1) dropped a little above the floor span
    // (top face y = 8.0), FAR apart so they settle as SEPARATE singleton islands.
    // D_waker is added first (lower slot -> contact bodyA); D_sleeper second.
    auto makeBox = [&](Real x) {
        BodyDef d;
        d.type          = BodyType::Dynamic;
        d.density       = Real(1);
        d.friction      = Real(0.4);
        d.fixedRotation = true;                        // stays flat -> settles fast
        d.shape         = MakeAabb(Real(0.1), Real(0.1));
        d.position      = Vec2(x, Real(7.5));
        return w.AddBody(d);
    };
    const BodyHandle waker   = makeBox(Real(4));       // slot lo -> bodyA (must be awake)
    const BodyHandle sleeper = makeBox(Real(8));       // slot hi -> bodyB (the sleeper)

    // Settle + sleep BOTH (kSleepTime = 0.5 s of idle = 30 steps after they stop).
    int settledStep = -1;
    for (int k = 0; k < 400; ++k)
    {
        w.Step(kStep);
        if (!w.IsAwake(waker) && !w.IsAwake(sleeper)) { settledStep = k; break; }
    }
    REQUIRE(settledStep >= 0);                          // both actually slept
    REQUIRE_FALSE(w.IsAwake(waker));
    REQUIRE_FALSE(w.IsAwake(sleeper));
    // Separate singleton islands (rest purely on the span -- the bug precondition).
    REQUIRE(w.IslandRootOf(waker.index) != w.IslandRootOf(sleeper.index));

    // Wake ONLY the waker (awake, sleepTimer 0, velocity still 0) and teleport it
    // to just overlap the still-sleeping sleeper's left face (0.01 m overlap:
    // centres 0.19 < 0.2 = sum of half-extents apart).
    const Vec2 ps = w.Position(sleeper);
    w.Wake(waker);
    w.SetPosition(waker, Vec2(ps.x - Real(0.19), ps.y));
    REQUIRE(w.IsAwake(waker));
    REQUIRE_FALSE(w.IsAwake(sleeper));                  // sleeper still asleep -> mixed pair on merge

    // One step: the new waker(A, awake)-sleeper(B, asleep) contact touch-begins and
    // queues a cross-awake merge. Reaching the line AFTER this Step (no abort)
    // proves the guard woke the sleeper before the merge.
    w.Step(kStep);

    REQUIRE(w.IsAwake(sleeper));                        // woken by the merge-time guard
    REQUIRE(w.IsAwake(waker));
    REQUIRE(w.IslandRootOf(waker.index) == w.IslandRootOf(sleeper.index)); // one merged island
    REQUIRE(IslandsUniformlyAwake(w, { waker, sleeper }));
}
