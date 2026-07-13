// Read-only debug accessors: enumerate exactly the live structures, and tag each
// contact with the narrowphase that produced it. Determinism/behavior unchanged.
//
// These accessors back the Slice A physics debug-visualization overlay. They are
// pure read paths over the broadphase trees + the static/residency grids + the
// solver's ContactConstraint pool. Nothing here feeds back into the Step path:
// iteration order is for display only.
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/Broadphase/Broadphase.hpp>
#include <Manifold2D/Physics/Broadphase/DynamicTree.hpp>
#include <Manifold2D/Physics/Broadphase/SpatialGrid.hpp>
#include <Manifold2D/Physics/Narrowphase/NarrowphaseTrace.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Solver/Solver.hpp>

using namespace Manifold2D::Physics;

TEST_CASE("Debug accessors enumerate broadphase + contacts", "[physics][debugviz]")
{
    // WorldDef gravityY now defaults to MKS 10; this scene keeps the default
    // (no override needed) -- one Step's worth of drift (~0.0014 m) is far
    // below the 0.1 m gap/overlap scale below, and every assertion here is
    // a structural/relational check (leaf counts, fat-encloses-tight,
    // contact-count agreement), not a position target.
    WorldDef wd;
    PhysicsWorld w(wd);
    auto addBox = [&](Real x, Real y, BodyType t)
    {
        BodyDef d;
        d.type          = t;
        d.position      = Vec2(x, y);
        d.fixedRotation = true;
        d.shape         = MakeAabb(Real(1), Real(1));
        return w.AddBody(d);
    };
    addBox(0, Real(10),  BodyType::Static);  // a static -> static grid
    addBox(0, Real(7.9), BodyType::Dynamic); // overlaps the static -> a contact
    w.Step(Real(1) / Real(60));

    // (a) ForEachLeaf yields live mover fixtures, tight inside fat.
    // FixtureBroadphaseTree() is a POINTER (the broadphase is selectable; it is
    // non-null only for the default DynamicTree mover broadphase, which this
    // default-constructed world uses).
    const DynamicTree* tree = w.FixtureBroadphaseTree();
    REQUIRE(tree != nullptr);
    std::size_t leaves = 0;
    tree->ForEachLeaf(
        [&](std::uint32_t, const Aabb2& tight, const Aabb2& fat)
        {
            ++leaves;
            REQUIRE(fat.min.x <= tight.min.x);
            REQUIRE(fat.max.x >= tight.max.x);
            REQUIRE(fat.min.y <= tight.min.y);
            REQUIRE(fat.max.y >= tight.max.y);
        });
    REQUIRE(leaves >= 1);

    // (b) static tree yields one leaf per static body (one static added above),
    // each leaf's fat box enclosing its tight box (mirrors the mover-tree walk).
    std::size_t staticLeaves = 0;
    w.StaticTree().ForEachLeaf(
        [&](std::uint32_t, const Aabb2& tight, const Aabb2& fat)
        {
            ++staticLeaves;
            REQUIRE(fat.min.x <= tight.min.x);
            REQUIRE(fat.max.x >= tight.max.x);
            REQUIRE(fat.min.y <= tight.min.y);
            REQUIRE(fat.max.y >= tight.max.y);
        });
    REQUIRE(staticLeaves == 1);   // exactly the one static body added above

    // (c) ForEachContactConstraint count == ActiveContactCount, and kind is set.
    std::size_t n = 0;
    bool kindSet  = false;
    w.ForEachContactConstraint(
        [&](const ContactConstraint& cc)
        {
            ++n;
            if (cc.kind != NarrowphaseKind::Separated) kindSet = true;
        });
    REQUIRE(n == w.ActiveContactCount());
    if (n > 0) REQUIRE(kindSet);
}

// Slice B Core (Task 3): PhysicsWorld::DebugCollide re-runs the REAL
// narrowphase on two fixtures and records a NarrowphaseTrace. The Step path
// passes no trace to Collide (byte-identical), so the recorder is observable
// only here. Two overlapping AABBs route through CollidePoly -> SAT, so the
// trace tags SatPolygon and records >= 1 candidate axis.
TEST_CASE("DebugCollide reproduces the manifold + records a trace", "[physics][debugviz]")
{
    // WorldDef gravityY now defaults to MKS 10 (kept, no override needed):
    // gravity moves both dynamics down together and never touches the
    // x-axis, so the x-overlap this test depends on is unaffected.
    WorldDef wd;
    PhysicsWorld w(wd);
    BodyDef d; d.type=BodyType::Dynamic; d.fixedRotation=true; d.shape=MakeAabb(Real(1),Real(1));
    d.position=Vec2(0,0);        BodyHandle a = w.AddBody(d);
    d.position=Vec2(Real(1.5),0); BodyHandle b = w.AddBody(d); // overlapping boxes -> SAT/EPA
    w.Step(Real(1)/Real(60));

    // The primary (back-compat) fixture slot of each body (fixture[0]).
    FixtureHandle fa = w.GetBodyFixture(a, 0);
    FixtureHandle fb = w.GetBodyFixture(b, 0);
    REQUIRE(w.IsValid(fa));
    REQUIRE(w.IsValid(fb));

    NarrowphaseTrace trace;
    Manifold m = w.DebugCollide(fa, fb, trace);
    REQUIRE(m.pointCount >= 1);                 // they overlap
    REQUIRE(trace.kind == m.kind);              // trace tags the algorithm
    REQUIRE(trace.kind != NarrowphaseKind::Separated);
    // The trace also copied the final manifold + the two world shapes.
    REQUIRE(trace.manifold.pointCount == m.pointCount);
    // For a poly-poly overlap, SAT recorded >=1 candidate axis OR EPA >=1
    // polytope snapshot.
    REQUIRE((trace.satAxes.size() >= 1 || trace.epaSnapshots.size() >= 1));
    // Exactly one SAT candidate axis is the chosen min-penetration reference.
    if (!trace.satAxes.empty())
    {
        std::size_t chosen = 0;
        for (const auto& ax : trace.satAxes) if (ax.chosen) ++chosen;
        REQUIRE(chosen == 1);
    }
}
