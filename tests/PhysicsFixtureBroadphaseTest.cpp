// Per-fixture broadphase oracle: the fixture-pair candidate set the world's
// per-fixture broadphase produces must EQUAL the brute-force O(n^2) set over all
// live MOVER fixture world-AABBs (a<b by fixture slot). Compound bodies (the whisk
// case) are the point: each fixture is its own proxy.
#include <algorithm>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/Broadphase/Broadphase.hpp> // BroadphasePair, AabbOverlap

using namespace Manifold2D::Physics;

namespace
{
    std::vector<BroadphasePair> BruteFixturePairs(const PhysicsWorld& w)
    {
        std::vector<std::uint32_t> fx; std::vector<Aabb2> box;
        w.LiveFixtureAabbs(fx, box);
        std::vector<BroadphasePair> out;
        for (std::size_t i = 0; i < fx.size(); ++i)
            for (std::size_t j = i + 1; j < fx.size(); ++j)
                if (AabbOverlap(box[i], box[j]))
                {
                    std::uint32_t a = fx[i], b = fx[j];
                    out.push_back(a < b ? BroadphasePair{a, b} : BroadphasePair{b, a});
                }
        std::sort(out.begin(), out.end());
        return out;
    }

    // Returns sorted Pairs() from the fixture broadphase.
    std::vector<BroadphasePair> FxBpPairs(const PhysicsWorld& w)
    {
        std::vector<BroadphasePair> got;
        w.FixtureBroadphase().Pairs(got);
        std::sort(got.begin(), got.end());
        return got;
    }

    // Returns the current live mover-fixture count.
    std::size_t LiveFxCount(const PhysicsWorld& w)
    {
        std::vector<std::uint32_t> fx; std::vector<Aabb2> box;
        w.LiveFixtureAabbs(fx, box);
        return fx.size();
    }
}

TEST_CASE("Per-fixture broadphase pairs == brute-force (compound scene)", "[physics][fxbroadphase]")
{
    WorldDef wd;
    PhysicsWorld w(wd);
    auto addBox = [&](Real x, Real y, Real hw, Real hh, BodyType t) {
        BodyDef d; d.type = t; d.position = Vec2(x, y); d.fixedRotation = true;
        d.shape = MakeAabb(hw, hh); return w.AddBody(d);
    };
    BodyHandle c = addBox(0, 0, Real(1), Real(1), BodyType::Dynamic);
    FixtureDef f; f.shape = MakeAabb(Real(1), Real(1));
    f.localPos = Vec2(Real(4), Real(0));  w.AddFixture(c, f);
    f.localPos = Vec2(Real(8), Real(0));  w.AddFixture(c, f);
    addBox(Real(3.5),  0, Real(1.2), Real(1.2), BodyType::Dynamic);
    addBox(Real(8.2),  0, Real(1.2), Real(1.2), BodyType::Dynamic);
    addBox(Real(40),   0, Real(1.2), Real(1.2), BodyType::Dynamic);
    addBox(Real(4), Real(-0.3), Real(1.4), Real(1.4), BodyType::Static);   // static fixtures excluded from the tree

    w.Step(Real(1) / Real(60));

    std::vector<BroadphasePair> got;
    w.FixtureBroadphase().Pairs(got);
    std::sort(got.begin(), got.end());
    REQUIRE(got == BruteFixturePairs(w));
}

// ----------------------------------------------------------------
// Lifecycle case 1: DropFixture removes the proxy
// ----------------------------------------------------------------
// Build a 3-fixture compound body c (fixtures at x=0, x=4, x=8)
// with a neighbor that overlaps the 2nd and 3rd fixtures.  After an
// initial Step + oracle check, drop the 2nd fixture and verify:
//   (a) LiveFixtureAabbs count drops by exactly 1 (immediately, before Step), and
//   (b) Pairs()==brute (the dropped fixture's pairs are GONE) after the Step.
TEST_CASE("DropFixture removes proxy from fixture broadphase", "[physics][fxbroadphase]")
{
    WorldDef wd;
    PhysicsWorld w(wd);
    auto addBox = [&](Real x, Real y, Real hw, Real hh, BodyType t) {
        BodyDef d; d.type = t; d.position = Vec2(x, y); d.fixedRotation = true;
        d.shape = MakeAabb(hw, hh); return w.AddBody(d);
    };

    // Compound body: primary fixture at origin, plus two extra fixtures.
    BodyHandle c = addBox(0, 0, Real(1), Real(1), BodyType::Dynamic);
    FixtureDef f; f.shape = MakeAabb(Real(1), Real(1));
    f.localPos = Vec2(Real(4), Real(0)); FixtureHandle fx2 = w.AddFixture(c, f); // 2nd fixture
    f.localPos = Vec2(Real(8), Real(0));              w.AddFixture(c, f);       // 3rd fixture

    // Neighbor overlapping the 2nd fixture (at x=4) and the 3rd (at x=8).
    addBox(Real(4.5), 0, Real(1.2), Real(1.2), BodyType::Dynamic);

    w.Step(Real(1) / Real(60));

    // Initial oracle check: broadphase must match brute.
    REQUIRE(FxBpPairs(w) == BruteFixturePairs(w));
    // Pairs must be non-empty (the neighbor overlaps at least the 2nd fixture).
    REQUIRE(!FxBpPairs(w).empty());

    // Record compound body's fixture count and live mover-fixture count before drop.
    std::uint32_t cFxBefore = w.FixtureCount(c);
    std::size_t liveBefore = LiveFxCount(w);
    REQUIRE(cFxBefore == 3u); // primary + fx2 + 3rd

    // Drop the 2nd fixture (fx2) from the compound body.
    w.DropFixture(fx2);
    REQUIRE_FALSE(w.IsValid(fx2)); // handle must be stale

    // Both FixtureCount(c) and LiveFixtureAabbs reflect the drop immediately:
    // DropFixture unlinks the slot from m_bodyFixtures and removes its proxy.
    REQUIRE(w.FixtureCount(c) == cFxBefore - 1u);
    REQUIRE(LiveFxCount(w) == liveBefore - 1u);

    w.Step(Real(1) / Real(60));

    // After drop + Step: broadphase must still match brute.
    // The dropped fixture's proxy was removed by DropFixture, and the fixed
    // LiveFixtureAabbs now enumerates via m_bodyFixtures (the authoritative
    // live set), so both sides of the oracle agree exactly.
    REQUIRE(FxBpPairs(w) == BruteFixturePairs(w));
}

// ----------------------------------------------------------------
// Lifecycle case 2: RemoveBody removes all its proxies
// ----------------------------------------------------------------
// Build 2 overlapping dynamic bodies (A and B), each single-fixture.
// After initial oracle check, remove body A and verify:
//   (a) Pairs()==brute (no pairs reference A's fixture), and
//   (b) live mover-fixture count dropped by A's fixture count (1).
TEST_CASE("RemoveBody removes all fixture proxies from broadphase", "[physics][fxbroadphase]")
{
    WorldDef wd;
    PhysicsWorld w(wd);
    auto addBox = [&](Real x, Real y, Real hw, Real hh, BodyType t) {
        BodyDef d; d.type = t; d.position = Vec2(x, y); d.fixedRotation = true;
        d.shape = MakeAabb(hw, hh); return w.AddBody(d);
    };

    // Two overlapping dynamic bodies.
    BodyHandle a = addBox(Real(0), 0, Real(1.5), Real(1.5), BodyType::Dynamic);
    BodyHandle b = addBox(Real(2), 0, Real(1.5), Real(1.5), BodyType::Dynamic);
    // A third body far away so the pair set stays non-trivial.
    addBox(Real(20), 0, Real(1.5), Real(1.5), BodyType::Dynamic);

    w.Step(Real(1) / Real(60));

    // Initial oracle check.
    REQUIRE(FxBpPairs(w) == BruteFixturePairs(w));
    // a and b overlap, so at least one pair must exist.
    REQUIRE(!FxBpPairs(w).empty());

    std::uint32_t aFxCount = w.FixtureCount(a); // should be 1 for a plain AddBody
    std::size_t countBefore = LiveFxCount(w);

    w.RemoveBody(a);
    REQUIRE_FALSE(w.IsValid(a)); // handle must be stale

    w.Step(Real(1) / Real(60));

    // After removal: broadphase must still match brute.
    REQUIRE(FxBpPairs(w) == BruteFixturePairs(w));

    // Live count must have dropped by exactly aFxCount.
    REQUIRE(LiveFxCount(w) == countBefore - aFxCount);
}

// ----------------------------------------------------------------
// Lifecycle case 3: Move across steps -- separated pair is removed
// ----------------------------------------------------------------
// Two dynamic bodies start overlapping.  Body B is then teleported far
// away (SetPosition).  After a Step the pair must be gone and
// Pairs()==brute must still hold.
//
// WorldDef gravityY now defaults to MKS 10 (free fall, no floor); the two
// steps here only accumulate ~0.5*10*(1/60)^2 ~ 0.0014 m of drift, dwarfed
// by the box separations below, so both assertions (oracle equality and the
// overlap/empty relation) hold regardless of gravity.  SetPosition is the
// deterministic teleport that also snaps prev, avoiding lerp smear.
TEST_CASE("Fixture broadphase pair removed after bodies separate (move)", "[physics][fxbroadphase]")
{
    WorldDef wd;
    PhysicsWorld w(wd);
    auto addBox = [&](Real x, Real y, Real hw, Real hh, BodyType t) {
        BodyDef d; d.type = t; d.position = Vec2(x, y); d.fixedRotation = true;
        d.shape = MakeAabb(hw, hh); return w.AddBody(d);
    };

    // Two overlapping bodies.
    BodyHandle a = addBox(Real(0), 0, Real(2), Real(2), BodyType::Dynamic);
    BodyHandle b = addBox(Real(1), 0, Real(2), Real(2), BodyType::Dynamic);

    w.Step(Real(1) / Real(60));

    // Initial oracle check: must overlap, pair must be present.
    REQUIRE(FxBpPairs(w) == BruteFixturePairs(w));
    REQUIRE(!FxBpPairs(w).empty());

    // Teleport body B far away so its fixture no longer overlaps A's.
    w.SetPosition(b, Vec2(Real(100), Real(0)));

    w.Step(Real(1) / Real(60));

    // After separation: oracle still holds AND the pair is gone.
    REQUIRE(FxBpPairs(w) == BruteFixturePairs(w));
    REQUIRE(FxBpPairs(w).empty());
}

#include <random>
#include <Manifold2D/Physics/Broadphase/DynamicTree.hpp>

// The incrementally-maintained pair set (move buffer) must equal a full Pairs()
// recompute AND the brute-force O(n^2) set, after EVERY mutation, across a long
// randomized sequence of inserts / moves / removes. Includes SMALL (within-fat-box)
// moves -- the riskiest case for tight-membership maintenance.
TEST_CASE("DynamicTree incremental pairs == full == brute-force", "[physics][fxbroadphase][movebuffer]")
{
    std::mt19937 rng(0xBADC0DE);
    std::uniform_real_distribution<float> pos(-30.f, 30.f), ext(0.4f, 4.0f);
    // re-baselined: kMargin 8->0.05 (MKS P1.iii) -- nudges must stay within the
    // fat margin to keep exercising the no-reinsert tight-membership path (the
    // declared riskiest case); +/-3 would now always force a reinsert.
    // NOTE (MKS P4): nudge is intentionally NOT re-divided alongside pos/ext
    // above -- it was already re-tuned in P1.iii against the engine's fixed
    // DynamicTree::kMargin=0.05 and must stay at that scale. Its ratio to the
    // now-smaller ext range (0.02 against a 0.4-4.0 box, vs. the old 0.02
    // against a 4-40 box) is now proportionally LARGER -- a more rigorous
    // small-move perturbation, not a weaker one.
    std::uniform_real_distribution<float> nudge(-0.02f, 0.02f); // within fat margin (kMargin=0.05)
    DynamicTree tree;
    std::vector<std::pair<std::uint32_t, Aabb2>> live;

    auto boxAt = [&](float x, float y, float w, float h) {
        Aabb2 a; a.min = Vec2(x, y); a.max = Vec2(x + w, y + h); return a; };
    auto brute = [&]() {
        std::vector<BroadphasePair> out;
        for (std::size_t i = 0; i < live.size(); ++i)
            for (std::size_t j = i + 1; j < live.size(); ++j)
                if (AabbOverlap(live[i].second, live[j].second)) {
                    std::uint32_t a = live[i].first, b = live[j].first;
                    out.push_back(a < b ? BroadphasePair{a,b} : BroadphasePair{b,a}); }
        std::sort(out.begin(), out.end()); return out; };

    std::uint32_t nextId = 0;
    for (int iter = 0; iter < 600; ++iter)
    {
        const int op = rng() % 4;
        if (op == 0 || live.size() < 4) {                 // insert
            Aabb2 b = boxAt(pos(rng), pos(rng), ext(rng), ext(rng));
            std::uint32_t id = nextId++; tree.Update(id, b); live.push_back({id, b});
        } else if (op == 1) {                              // big move
            auto& e = live[rng() % live.size()];
            e.second = boxAt(pos(rng), pos(rng), ext(rng), ext(rng)); tree.Update(e.first, e.second);
        } else if (op == 2) {                              // SMALL move (within fat box)
            auto& e = live[rng() % live.size()];
            const float dx = nudge(rng), dy = nudge(rng);
            e.second.min = Vec2(e.second.min.x + dx, e.second.min.y + dy);
            e.second.max = Vec2(e.second.max.x + dx, e.second.max.y + dy);
            tree.Update(e.first, e.second);
        } else {                                           // remove
            std::size_t k = rng() % live.size();
            tree.Remove(live[k].first); live.erase(live.begin() + static_cast<std::ptrdiff_t>(k));
        }
        std::vector<BroadphasePair> incr; tree.UpdatePairs(incr);
        std::vector<BroadphasePair> full; tree.Pairs(full);
        std::sort(incr.begin(), incr.end()); std::sort(full.begin(), full.end());
        const std::vector<BroadphasePair> bf = brute();
        REQUIRE(incr == bf);
        REQUIRE(full == bf);
    }
}

// ----------------------------------------------------------------
// Fix 1 regression: SetAngle immediately refreshes mover proxies
// ----------------------------------------------------------------
// A KINEMATIC body carries a long thin box (half-extents 4 x 0.4) at the
// origin.  At angle 0 it spans x in [-4,4], y in [-0.4,0.4].  A second
// kinematic neighbor box sits at (0, 3) with half-extents 0.8x0.8
// (spans y in [2.2,3.8]) -- clearly disjoint from the thin box at angle 0.
//
// After one Step both proxies are registered.  We assert NO pair exists
// at angle 0, then call SetAngle(thin, pi/2) WITHOUT a Step.  Rotated
// 90 deg, the thin box's AABB becomes x in [-0.4,0.4], y in [-4,4]
// (swapped extents), which DOES overlap the neighbor at y=3.  Fix 1
// ensures SetAngle refreshes the proxy immediately; the pair must appear
// in FixtureBroadphase().Pairs() before any Step.
TEST_CASE("SetAngle immediately refreshes mover fixture proxies (no Step)", "[physics][fxbroadphase]")
{
    WorldDef wd; // both bodies are Kinematic, so gravity is moot regardless
    PhysicsWorld w(wd);

    // Thin long box: half-extents 4 x 0.4, kinematic so it can rotate.
    BodyDef defThin;
    defThin.type     = BodyType::Kinematic;
    defThin.position = Vec2(Real(0), Real(0));
    defThin.shape    = MakeAabb(Real(4), Real(0.4)); // half-extents 4 x 0.4
    BodyHandle thin = w.AddBody(defThin);

    // Neighbor box at (0, 3): half-extents 0.8 x 0.8, kinematic (must be a
    // mover for its fixture to live in the per-fixture broadphase tree).
    BodyDef defNeigh;
    defNeigh.type     = BodyType::Kinematic;
    defNeigh.position = Vec2(Real(0), Real(3));
    defNeigh.shape    = MakeAabb(Real(0.8), Real(0.8)); // spans y in [2.2, 3.8]
    w.AddBody(defNeigh);

    // One Step registers both proxies in the fixture broadphase.
    w.Step(Real(1) / Real(60));

    // At angle 0: thin spans y in [-0.4,0.4]; neighbor spans y in [2.2,3.8].
    // They must NOT overlap -- no pair.
    const auto pairsBefore = FxBpPairs(w);
    REQUIRE(pairsBefore.empty());

    // Rotate the thin box 90 degrees WITHOUT stepping.
    // Its AABB becomes y in [-4, 4], which covers the neighbor's y in [2.2,3.8].
    w.SetAngle(thin, kPi / Real(2));

    // The fixture broadphase must immediately reflect the rotated AABB.
    const auto pairsAfter = FxBpPairs(w);
    REQUIRE_FALSE(pairsAfter.empty()); // at least one pair must now exist

    // Also verify the brute-force oracle agrees (broadphase == brute, post-rotation).
    REQUIRE(pairsAfter == BruteFixturePairs(w));
}
