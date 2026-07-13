// SpatialGrid: a per-shape fixed-tile spatial hash. Contract: after the caller's
// tight AABB filter, QueryAABB returns EXACTLY the brute-force overlap set,
// sorted + unique. The grid only narrows; correctness is gated by the oracle.
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <Manifold2D/Physics/Broadphase/SpatialGrid.hpp>
#include <Manifold2D/Physics/Broadphase/Broadphase.hpp> // AabbOverlap

using namespace Manifold2D::Physics;

namespace
{
    Aabb2 Box(Real x0, Real y0, Real x1, Real y1)
    { Aabb2 a; a.min = Vec2(x0,y0); a.max = Vec2(x1,y1); return a; }
}

TEST_CASE("SpatialGrid query == brute-force after tight filter", "[physics][grid]")
{
    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<float> pos(-50.0f, 50.0f);
    std::uniform_real_distribution<float> ext(0.2f, 6.0f);

    SpatialGrid grid(/*tileSize=*/3.2f);
    std::vector<Aabb2> boxes;
    for (std::uint32_t i = 0; i < 400; ++i)
    {
        const float x = pos(rng), y = pos(rng), w = ext(rng), h = ext(rng);
        Aabb2 b = Box(x, y, x + w, y + h);
        boxes.push_back(b);
        grid.Insert(i, b);
    }

    auto tightFilter = [&](const Aabb2& q, std::vector<std::uint32_t> cand) {
        std::vector<std::uint32_t> out;
        for (std::uint32_t id : cand) if (AabbOverlap(boxes[id], q)) out.push_back(id);
        std::sort(out.begin(), out.end()); out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    };
    auto brute = [&](const Aabb2& q) {
        std::vector<std::uint32_t> out;
        for (std::uint32_t i = 0; i < boxes.size(); ++i) if (AabbOverlap(boxes[i], q)) out.push_back(i);
        return out; // already sorted ascending
    };

    for (int t = 0; t < 200; ++t)
    {
        const float x = pos(rng), y = pos(rng), w = ext(rng), h = ext(rng);
        Aabb2 q = Box(x, y, x + w, y + h);
        std::vector<std::uint32_t> cand;
        grid.QueryAABB(q, cand);
        REQUIRE(tightFilter(q, cand) == brute(q));
    }

    SECTION("Move + Remove keep the invariant")
    {
        grid.Move(0, Box(90.0f, 90.0f, 92.0f, 92.0f)); boxes[0] = Box(90,90,92,92);
        grid.Remove(1); boxes[1] = Box(1e9f, 1e9f, 1e9f, 1e9f); // unreachable
        Aabb2 q = Box(89.0f, 89.0f, 93.0f, 93.0f);
        std::vector<std::uint32_t> cand; grid.QueryAABB(q, cand);
        std::vector<std::uint32_t> got = tightFilter(q, cand);
        REQUIRE(std::find(got.begin(), got.end(), 0u) != got.end());
        REQUIRE(std::find(got.begin(), got.end(), 1u) == got.end());
    }

    SECTION("Empty-grid query returns 0 and empty out")
    {
        SpatialGrid empty(3.2f);
        std::vector<std::uint32_t> out;
        int n = empty.QueryAABB(Box(-10.0f, -10.0f, 10.0f, 10.0f), out);
        REQUIRE(n == 0);
        REQUIRE(out.empty());
    }

    SECTION("Double-insert self-heals: id moves from A to B without explicit Remove")
    {
        SpatialGrid g(3.2f);
        // Insert id 7 at box A (well within query range of origin)
        Aabb2 boxA = Box(0.0f, 0.0f, 1.0f, 1.0f);
        // Box B is far away so no AABB overlap between A and B
        Aabb2 boxB = Box(500.0f, 500.0f, 501.0f, 501.0f);
        g.Insert(7u, boxA);
        // Re-insert at B without an explicit Remove -- self-healing fix must handle this
        g.Insert(7u, boxB);

        std::vector<std::uint32_t> cand;
        // Query near A: id 7 must NOT be there (it moved to B)
        g.QueryAABB(Box(-0.5f, -0.5f, 1.5f, 1.5f), cand);
        REQUIRE(std::find(cand.begin(), cand.end(), 7u) == cand.end());

        // Query near B: id 7 MUST be there
        cand.clear();
        g.QueryAABB(Box(499.5f, 499.5f, 501.5f, 501.5f), cand);
        REQUIRE(std::find(cand.begin(), cand.end(), 7u) != cand.end());
    }
}

#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Shapes.hpp>

TEST_CASE("StaticCandidates static-body set unchanged by grid reroute", "[physics][grid]")
{
    WorldDef wd;
    PhysicsWorld w(wd);
    auto addStatic = [&](Real cx, Real cy, Real hw, Real hh) {
        BodyDef d; d.type = BodyType::Static; d.position = Vec2(cx, cy);
        d.shape = MakeAabb(hw, hh); return w.AddBody(d);
    };
    addStatic(0,  0,  5, 1);
    addStatic(12, 0,  5, 1);
    addStatic(0,  20, 1, 8);

    // Reference: brute-force overlap of the query box against every static slot.
    Aabb2 q; q.min = Vec2(-6, -2); q.max = Vec2(6, 2);
    std::vector<Aabb2> spans; std::vector<std::uint32_t> statics;
    std::vector<std::uint32_t> gridScratch;
    w.StaticCandidates(q, spans, statics, gridScratch);
    // Only the first static (centered at 0,0, half 5x1) overlaps q.
    REQUIRE(statics.size() == 1);
}

// Behavior-preservation oracle for the static-index swap (SpatialGrid ->
// DynamicTree). Pins that StaticCandidates returns EXACTLY the brute-force
// static-overlap set. This passes today against the grid and must STILL pass
// after the tree swap -- the invariant is the same either way (a refactor
// guarded by an oracle, not a red->green feature).
TEST_CASE("StaticCandidates set is stable across many statics (tree-backed)", "[physics][grid]")
{
    WorldDef wd;
    PhysicsWorld w(wd);
    std::mt19937 rng(0xABCDEF);
    std::uniform_real_distribution<float> pos(-40.0f, 40.0f);
    std::vector<BodyHandle> handles;
    for (int i = 0; i < 200; ++i) {
        BodyDef d; d.type = BodyType::Static;
        d.position = Vec2(Real(pos(rng)), Real(pos(rng)));
        d.shape = MakeAabb(Real(0.6), Real(0.6));
        handles.push_back(w.AddBody(d));
    }
    // Brute-force oracle: statics whose slot-AABB overlaps the query.
    auto brute = [&](const Aabb2& q) {
        std::vector<std::uint32_t> out;
        for (std::uint32_t i = 0; i < w.Count(); ++i)
            if (w.Alive(i) && w.TypeSlot(i) == BodyType::Static && AabbOverlap(w.SlotAabb(i), q))
                out.push_back(i);
        std::sort(out.begin(), out.end());
        return out;
    };
    std::uniform_real_distribution<float> qc(-40.0f, 40.0f);
    for (int t = 0; t < 100; ++t) {
        const float cx = qc(rng), cy = qc(rng);
        Aabb2 q; q.min = Vec2(Real(cx-2), Real(cy-2)); q.max = Vec2(Real(cx+2), Real(cy+2));
        std::vector<Aabb2> spans; std::vector<std::uint32_t> statics, scratch;
        w.StaticCandidates(q, spans, statics, scratch);
        std::sort(statics.begin(), statics.end());
        REQUIRE(statics == brute(q));   // exact set, before AND after the tree swap
    }
}

TEST_CASE("Residents(region) returns bodies in a tile region", "[physics][grid][residency]")
{
    WorldDef wd;
    PhysicsWorld w(wd);
    BodyDef d; d.type = BodyType::Dynamic; d.shape = MakeCircle(Real(0.8));
    d.position = Vec2(1, 1);   BodyHandle a = w.AddBody(d);
    d.position = Vec2(30, 30); BodyHandle b = w.AddBody(d);
    w.Step(Real(1)/Real(60));    // commit positions -> residency updated

    // Region covering ~(0,0)..(6.4,6.4) in world space should contain `a`, not `b`.
    Aabb2 region; region.min = Vec2(0, 0); region.max = Vec2(6.4, 6.4);
    std::vector<std::uint32_t> residents;
    w.Residents(region, residents);
    REQUIRE(std::find(residents.begin(), residents.end(), a.index) != residents.end());
    REQUIRE(std::find(residents.begin(), residents.end(), b.index) == residents.end());
}

TEST_CASE("Residency tracks a body across a per-step position commit", "[physics][grid][residency]")
{
    WorldDef wd;
    PhysicsWorld w(wd);
    BodyDef d; d.type = BodyType::Kinematic; d.shape = MakeCircle(Real(0.8));
    d.position = Vec2(1, 1);
    BodyHandle a = w.AddBody(d);
    w.SetVelocity(a, Vec2(100, 0));   // moves +x deterministically

    w.Step(Real(1));                   // kinematic integrate: a -> ~(101, 1)

    // It must have LEFT the origin region (proves Move-on-commit ran -- an
    // Insert-only residency would wrongly still report it here).
    Aabb2 origin; origin.min = Vec2(0, 0);   origin.max = Vec2(6.4, 6.4);
    std::vector<std::uint32_t> residents;
    w.Residents(origin, residents);
    REQUIRE(std::find(residents.begin(), residents.end(), a.index) == residents.end());

    // ...and ENTERED the destination region.
    Aabb2 dest; dest.min = Vec2(95, 0); dest.max = Vec2(110, 6.4);
    w.Residents(dest, residents);
    REQUIRE(std::find(residents.begin(), residents.end(), a.index) != residents.end());
}

// G1: RemoveBody must evict a static from the grid (StaticCandidates stops
// returning it). Guards the m_staticTree.Remove lockstep in RemoveBody.
TEST_CASE("StaticCandidates drops a static after RemoveBody", "[physics][grid]")
{
    WorldDef wd;
    PhysicsWorld w(wd);
    BodyDef d; d.type = BodyType::Static; d.position = Vec2(0, 0);
    d.shape = MakeAabb(Real(5), Real(1));
    BodyHandle s = w.AddBody(d);

    Aabb2 q; q.min = Vec2(-6, -2); q.max = Vec2(6, 2);
    std::vector<Aabb2> spans; std::vector<std::uint32_t> statics;
    std::vector<std::uint32_t> gridScratch;
    w.StaticCandidates(q, spans, statics, gridScratch);
    REQUIRE(statics.size() == 1);          // present before removal

    w.RemoveBody(s);
    statics.clear(); spans.clear();
    w.StaticCandidates(q, spans, statics, gridScratch);
    REQUIRE(statics.empty());              // gone after removal
}

// G2: AddFixture to a static grows its grid AABB so it becomes findable at the
// new (wider) region. Guards the AddFixture static-grid Move.
TEST_CASE("AddFixture grows a static's grid AABB", "[physics][grid]")
{
    WorldDef wd;
    PhysicsWorld w(wd);
    BodyDef d; d.type = BodyType::Static; d.position = Vec2(0, 0);
    d.shape = MakeAabb(Real(1), Real(1));   // small primary fixture near origin
    BodyHandle s = w.AddBody(d);

    // A region far to the +x side, NOT covered by the small primary fixture.
    Aabb2 farRegion; farRegion.min = Vec2(18, -1); farRegion.max = Vec2(22, 1);
    std::vector<Aabb2> spans; std::vector<std::uint32_t> statics;
    std::vector<std::uint32_t> gridScratch;
    w.StaticCandidates(farRegion, spans, statics, gridScratch);
    REQUIRE(statics.empty());              // not reachable yet

    // Add a second fixture offset far to +x so the body's union AABB now covers farRegion.
    FixtureDef fd; fd.shape = MakeAabb(Real(1), Real(1));
    fd.localPos = Vec2(20, 0);
    w.AddFixture(s, fd);

    statics.clear(); spans.clear();
    w.StaticCandidates(farRegion, spans, statics, gridScratch);
    REQUIRE(statics.size() == 1);          // now findable at the grown AABB
}

// G4: a DYNAMIC body's residency is refreshed via the solver commit path
// (CommitSlotPosition), not just the kinematic integrate. Drive it with velocity.
// WorldDef defaults are the MKS defaults: gravity=(0,+10) (y-only), so it does
// not perturb the x-axis displacement this test checks. With v=60 m/s and 30
// steps of dt=1/60: x displacement ~ 60*(30/60) = 30 m. Starting at x=1, final
// x ~ 31, well outside the (0..6.4) origin tile. v=60 m/s is fast-for-scale
// (well under the 400 m/s cap) -- deliberately brisk so 0.5s of travel clears
// the tile with margin.
TEST_CASE("Residency tracks a dynamic body via the solver commit", "[physics][grid][residency]")
{
    WorldDef wd;
    PhysicsWorld w(wd);
    BodyDef d; d.type = BodyType::Dynamic; d.shape = MakeCircle(Real(0.8));
    d.fixedRotation = true; d.position = Vec2(1, 1);
    BodyHandle a = w.AddBody(d);
    w.SetVelocity(a, Vec2(60, 0));        // dynamics integrate + commit via the solver; fast-for-scale (see comment above)

    for (int i = 0; i < 30; ++i) w.Step(Real(1)/Real(60)); // ~ +30 m x over 0.5s (gravity is y-only, does not affect x)

    Aabb2 origin; origin.min = Vec2(0, 0); origin.max = Vec2(6.4, 6.4);
    std::vector<std::uint32_t> residents;
    w.Residents(origin, residents);
    // It should have left the origin tile region (residency followed the solver commit).
    REQUIRE(std::find(residents.begin(), residents.end(), a.index) == residents.end());
}

// G5: Residents on an unoccupied region returns empty.
TEST_CASE("Residents empty region returns no bodies", "[physics][grid][residency]")
{
    WorldDef wd;
    PhysicsWorld w(wd);
    BodyDef d; d.type = BodyType::Dynamic; d.shape = MakeCircle(Real(0.8));
    d.position = Vec2(1, 1); w.AddBody(d);

    Aabb2 empty; empty.min = Vec2(500, 500); empty.max = Vec2(510, 510);
    std::vector<std::uint32_t> residents;
    const int n = w.Residents(empty, residents);
    REQUIRE(n == 0);
    REQUIRE(residents.empty());
}

TEST_CASE("SpatialGrid survives a non-finite AABB", "[physics][grid]")
{
    SpatialGrid g(32.0f);
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Aabb2 bad; bad.min = Vec2(Real(nan), Real(0)); bad.max = Vec2(Real(inf), Real(10));
    g.Insert(1u, bad);                 // must not hang / OOM / crash
    std::vector<std::uint32_t> out;
    const int n = g.QueryAABB(bad, out);   // must not hang / OOM / crash
    REQUIRE(n == 0);
    REQUIRE(out.empty());
    // A valid id nearby still works (grid is not corrupted).
    Aabb2 ok; ok.min = Vec2(0,0); ok.max = Vec2(10,10);
    g.Insert(2u, ok);
    g.QueryAABB(ok, out);
    REQUIRE(std::find(out.begin(), out.end(), 2u) != out.end());
}

TEST_CASE("SpatialGrid survives an absurdly large AABB", "[physics][grid]")
{
    SpatialGrid g(32.0f);
    Aabb2 huge; huge.min = Vec2(Real(-1e30), Real(-1e30)); huge.max = Vec2(Real(1e30), Real(1e30));
    g.Insert(1u, huge);                // must not attempt ~1e56 cells
    std::vector<std::uint32_t> out;
    const int n = g.QueryAABB(huge, out);
    REQUIRE(n == 0);                   // treated as empty (out-of-budget)
}

// The per-axis budget (kMaxCellsPerAxis == 1<<16) does not bound the TOTAL cell
// count: a box spanning ~30000 cells on EACH axis is well under the per-axis
// span budget AND under the raw-magnitude bound (kMaxCellsPerAxis * tileSize ==
// 65536*32 ~= 2.1M, vs 480000 here) -- but (30000+1)^2 ~= 9e8 total cells, which
// must still be rejected by a total-cell budget.
TEST_CASE("SpatialGrid rejects a box whose TOTAL cell count blows the budget", "[physics][grid]")
{
    SpatialGrid g(32.0f);
    Aabb2 huge; huge.min = Vec2(Real(-480000), Real(-480000)); huge.max = Vec2(Real(480000), Real(480000));
    g.Insert(1u, huge);                 // must not attempt ~9e8 cells
    std::vector<std::uint32_t> out;
    const int n = g.QueryAABB(huge, out);
    REQUIRE(n == 0);                    // treated as empty (out-of-budget)
    REQUIRE(out.empty());

    // A small valid box still works afterwards (grid is not corrupted).
    Aabb2 ok; ok.min = Vec2(0,0); ok.max = Vec2(10,10);
    g.Insert(2u, ok);
    g.QueryAABB(ok, out);
    REQUIRE(std::find(out.begin(), out.end(), 2u) != out.end());
}
