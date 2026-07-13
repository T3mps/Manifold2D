// Physics: broadphase strategy equivalence (oracle-free, analytic scene).
//
// HISTORY: this file once pinned the broadphase pair-set against a captured Lua
// oracle fixture (the retired broadphase JSON). Physics v2 Task T8 retired the
// oracle gate. Broadphase is rotation-AGNOSTIC and was unchanged by Phase A, so
// its behavior is still valid -- but the CONTRACT is self-consistent and needs
// no external oracle: the three strategies (DynamicTree, SpatialHash,
// SweepAndPrune) must produce the IDENTICAL sorted pair-set + identical
// QueryAABB id-sets for the same scene. We build a SELF-DEFINED analytic scene
// (a deterministic grid of boxes with hand-known overlaps) and assert that
// invariance directly. The cross-strategy-invariance case is ALSO folded into
// PhysicsInvariantsTest.cpp as the canonical invariant gate; this file keeps the
// focused mutation/query coverage of the individual strategies.
//
// The broadphase CONTRACT is the narrowed pair-set: candidate pairs filtered to
// true AABB overlap, emitted SORTED by (a,b). Pair ids are integers -> exact
// comparisons throughout.

#include <cstdint>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <Manifold2D/Physics/Broadphase/Broadphase.hpp>
#include <Manifold2D/Physics/Broadphase/DynamicTree.hpp>
#include <Manifold2D/Physics/Broadphase/SpatialHash.hpp>
#include <Manifold2D/Physics/Broadphase/SweepAndPrune.hpp>

using namespace Manifold2D::Physics;

namespace
{
    // A scene box: id + tight AABB.
    struct SceneBox
    {
        std::uint32_t id = 0;
        Aabb2         box{};
    };

    SceneBox MakeBox(std::uint32_t id, Real x0, Real y0, Real x1, Real y1)
    {
        SceneBox sb;
        sb.id      = id;
        sb.box.min = Vec2(x0, y0);
        sb.box.max = Vec2(x1, y1);
        return sb;
    }

    // Build the three strategies for a scene's cellSize. DynamicTree is the
    // default mover broadphase; SpatialHash + SweepAndPrune are the alternates.
    std::vector<std::unique_ptr<IBroadphase>> MakeAll(Real cellSize)
    {
        std::vector<std::unique_ptr<IBroadphase>> v;
        v.emplace_back(std::make_unique<DynamicTree>());
        v.emplace_back(std::make_unique<SpatialHash>(cellSize));
        v.emplace_back(std::make_unique<SweepAndPrune>());
        return v;
    }

    std::vector<BroadphasePair> PairsOf(const IBroadphase& bp)
    {
        std::vector<BroadphasePair> out;
        bp.Pairs(out);
        return out;
    }

    std::vector<std::uint32_t> QueryOf(const IBroadphase& bp, const Aabb2& box)
    {
        std::vector<std::uint32_t> out;
        bp.QueryAABB(box, out);
        return out;
    }

    // ----------------------------------------------------------------
    // The analytic scene: a 3x3 grid of 40x40 boxes on a 50-px pitch, so each
    // box's right/top edge is at +40 while its neighbour starts at +50 -> a
    // 10-px GAP, NO overlaps in the base grid. Then three OVERLAP pairs are
    // injected by adding boxes that straddle two grid cells. The overlaps are
    // hand-known, so the test asserts the EXACT narrowed pair-set without an
    // oracle.
    // ----------------------------------------------------------------
    std::vector<SceneBox> BuildScene()
    {
        std::vector<SceneBox> boxes;
        // 3x3 non-overlapping grid: ids 1..9, pitch 50, size 40.
        std::uint32_t id = 1;
        for (int gy = 0; gy < 3; ++gy)
        {
            for (int gx = 0; gx < 3; ++gx)
            {
                const Real x0 = Real(gx * 50);
                const Real y0 = Real(gy * 50);
                boxes.push_back(MakeBox(id++, x0, y0, x0 + Real(40), y0 + Real(40)));
            }
        }
        // Injected straddlers (ids 10..12), each overlapping a known set of grid
        // boxes by construction. They are placed in DISJOINT regions so they do
        // NOT overlap each other (verified by hand; AabbOverlap is INCLUSIVE):
        //   id 10 spans x[30..70] y[5..35] -> overlaps box 1 ([0..40]x[0..40])
        //         and box 2 ([50..90]x[0..40]) on the bottom row. (y[5..35] is
        //         strictly below the y=50 second row.)
        boxes.push_back(MakeBox(10, Real(30), Real(5), Real(70), Real(35)));
        //   id 11 spans x[105..135] y[30..70] -> overlaps box 3
        //         ([100..140]x[0..40]) and box 6 ([100..140]x[50..90]) in the
        //         right column. (x>=105 is clear of straddler 10's right edge 70.)
        boxes.push_back(MakeBox(11, Real(105), Real(30), Real(135), Real(70)));
        //   id 12 spans x[105..135] y[105..135] -> overlaps box 9
        //         ([100..140]x[100..140]) only (single overlap; y>=105 is clear
        //         of straddler 11's top edge 70).
        boxes.push_back(MakeBox(12, Real(105), Real(105), Real(135), Real(135)));
        return boxes;
    }
} // namespace

// ====================================================================
// 1) The analytic scene: all three strategies emit the IDENTICAL sorted pair
//    set == the hand-known narrowed pairs.
// ====================================================================
TEST_CASE("physics: broadphase analytic scene == known narrowed pairs", "[physics]")
{
    const Real            cs    = Real(64);
    std::vector<SceneBox> boxes = BuildScene();

    auto impls = MakeAll(cs);
    for (auto& bp : impls)
    {
        for (const SceneBox& b : boxes)
        {
            bp->Update(b.id, b.box);
        }
    }

    // Hand-known overlaps (sorted by (a,b); pairs are emitted with a<b):
    //   straddler 10 overlaps 1 and 2 -> (1,10),(2,10)
    //   straddler 11 overlaps 3 and 6 -> (3,11),(6,11)
    //   straddler 12 overlaps 9       -> (9,12)
    const std::vector<BroadphasePair> expected = {
        BroadphasePair{ 1u, 10u },
        BroadphasePair{ 2u, 10u },
        BroadphasePair{ 3u, 11u },
        BroadphasePair{ 6u, 11u },
        BroadphasePair{ 9u, 12u },
    };

    const std::vector<BroadphasePair> treePairs = PairsOf(*impls[0]);
    const std::vector<BroadphasePair> hashPairs = PairsOf(*impls[1]);
    const std::vector<BroadphasePair> sapPairs  = PairsOf(*impls[2]);

    CHECK(treePairs == expected);
    CHECK(hashPairs == expected);
    CHECK(sapPairs == expected);
    // Explicit cross-strategy identity (the equivalence invariant).
    CHECK(treePairs == hashPairs);
    CHECK(hashPairs == sapPairs);

    // Determinism: calling Pairs() again yields the identical vector.
    CHECK(PairsOf(*impls[0]) == treePairs);
}

// ====================================================================
// 2) Update (move) + Remove: equivalence holds after mutation. Exercises the
//    DynamicTree fat-box reinsert, the SpatialHash re-bucketing, and SAP's
//    insertion-sort, then re-asserts all three agree.
// ====================================================================
TEST_CASE("physics: broadphase equivalence under Update + Remove", "[physics]")
{
    const Real            cs    = Real(64);
    std::vector<SceneBox> boxes = BuildScene();

    auto impls = MakeAll(cs);
    for (auto& bp : impls)
    {
        for (const SceneBox& b : boxes)
        {
            bp->Update(b.id, b.box);
        }
    }

    // Move a couple of bodies by a fixed offset (same offset to every strategy
    // so they stay in lockstep). Translation preserves the box extents.
    const Vec2 shift(37.0f, -23.0f);
    for (std::uint32_t moveId : { 3u, 6u, 12u })
    {
        for (SceneBox& b : boxes)
        {
            if (b.id == moveId)
            {
                Aabb2 moved;
                moved.min = b.box.min + shift;
                moved.max = b.box.max + shift;
                for (auto& bp : impls)
                {
                    bp->Update(b.id, moved);
                }
            }
        }
    }

    // Remove a handful of bodies from every strategy.
    const std::uint32_t toRemove[] = { 5u, 8u };
    for (const std::uint32_t id : toRemove)
    {
        for (auto& bp : impls)
        {
            bp->Remove(id);
        }
    }
    // Removing an absent id is a no-op on every strategy.
    for (auto& bp : impls)
    {
        bp->Remove(9999u);
    }

    // Cross-strategy equivalence is the invariant after arbitrary mutation.
    const std::vector<BroadphasePair> treePairs = PairsOf(*impls[0]);
    const std::vector<BroadphasePair> hashPairs = PairsOf(*impls[1]);
    const std::vector<BroadphasePair> sapPairs  = PairsOf(*impls[2]);

    CHECK(treePairs == hashPairs);
    CHECK(hashPairs == sapPairs);

    // The removed ids must not appear in any emitted pair.
    for (const BroadphasePair& p : treePairs)
    {
        for (const std::uint32_t id : toRemove)
        {
            CHECK(p.a != id);
            CHECK(p.b != id);
        }
    }
}

// ====================================================================
// 3) QueryAABB equivalence: a few query boxes return identical id sets across
//    all three strategies (each narrows against the TIGHT box, sorted asc).
// ====================================================================
TEST_CASE("physics: broadphase QueryAABB identical across strategies", "[physics]")
{
    const Real            cs    = Real(64);
    std::vector<SceneBox> boxes = BuildScene();

    auto impls = MakeAll(cs);
    for (auto& bp : impls)
    {
        for (const SceneBox& b : boxes)
        {
            bp->Update(b.id, b.box);
        }
    }

    const Aabb2 queries[] = {
        { Vec2(0.0f, 0.0f), Vec2(100.0f, 100.0f) },       // bottom-left cluster
        { Vec2(90.0f, 90.0f), Vec2(150.0f, 150.0f) },     // top-right cluster
        { Vec2(30.0f, 30.0f), Vec2(40.0f, 40.0f) },       // tight window
        { Vec2(-1000.0f, -1000.0f), Vec2(-900.0f, -900.0f) }, // empty (offscreen)
        { Vec2(-50.0f, -50.0f), Vec2(200.0f, 200.0f) },   // whole scene
    };

    for (const Aabb2& qb : queries)
    {
        const std::vector<std::uint32_t> tree = QueryOf(*impls[0], qb);
        const std::vector<std::uint32_t> hash = QueryOf(*impls[1], qb);
        const std::vector<std::uint32_t> sap  = QueryOf(*impls[2], qb);
        CHECK(tree == hash);
        CHECK(hash == sap);
    }

    // The whole-scene query returns every present id (12 boxes, sorted asc).
    const std::vector<std::uint32_t> all =
        QueryOf(*impls[0], Aabb2{ Vec2(-50.0f, -50.0f), Vec2(200.0f, 200.0f) });
    CHECK(all.size() == 12u);
    CHECK(all.front() == 1u);
    CHECK(all.back() == 12u);
}
