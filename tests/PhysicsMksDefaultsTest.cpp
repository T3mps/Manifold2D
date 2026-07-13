// Box2D v3 default parity for WorldDef (b2DefaultWorldDef) -- every value cites
// the vendored source at ThirdParty/box2d-3.1.1. Spec:
// docs/superpowers/specs/2026-07-02-arcane-physics-mks-units-design.md #3.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Broadphase/DynamicTree.hpp>
#include <Manifold2D/Physics/Broadphase/SpatialGrid.hpp>

using namespace Manifold2D::Physics;
using Catch::Approx;

TEST_CASE("WorldDef defaults are Box2D v3 MKS", "[physics][mks]")
{
    WorldDef def;
    CHECK(def.gravityX == Real(0));                        // types.c:12
    CHECK(def.gravityY == Real(10));                       // types.c:13 ((0,-10) y-up -> +10 y-down)
    CHECK(def.sleepThreshold == Approx(Real(0.05)));       // types.c:34 (BodyDef-level in b2)
    CHECK(def.restitutionThreshold == Approx(Real(1)));    // types.c:15
    CHECK(def.contactPushMaxVelocity == Approx(Real(3)));  // types.c:16 maxContactPushSpeed
    CHECK(def.maxLinearVelocity == Approx(Real(400)));     // types.c:21 (unchanged)
    CHECK(def.contactHertz == Approx(Real(30)));           // types.c:17 (unchanged)
    CHECK(def.contactDampingRatio == Approx(Real(10)));    // types.c:18 (unchanged)
    CHECK(def.hashCellSize == Approx(Real(1)));            // Arcane grid tuning (MKS)
}

TEST_CASE("BodyDef.sleepThreshold defaults to inherit", "[physics][mks]")
{
    BodyDef bd;
    CHECK(bd.sleepThreshold == Real(-1)); // inherit WorldDef (Arcane architecture; value carries)
}

TEST_CASE("Engine length constants are Box2D v3 MKS", "[physics][mks]")
{
    CHECK(kLinearSlop == Approx(Real(0.005)));          // constants.h:23 (unchanged)
    CHECK(kSkin == Approx(Real(0.02)));                 // constants.h:38 B2_SPECULATIVE_DISTANCE = 4*slop
    CHECK(kSkin == Approx(Real(4) * kLinearSlop));      // the RELATION, not just the value
    CHECK(DynamicTree::kMargin == Approx(Real(0.05))); // constants.h:44 B2_AABB_MARGIN (v3.1.1)
    CHECK(kMaxRotation == Approx(Real(0.25) * kPi));    // constants.h:33 (unchanged)
}

// ---------------------------------------------------------------------------
// Blind-spot closure (MKS P4): "WorldDef.hashCellSize == 1" (asserted above)
// does NOT mean the engine's actual default broadphase, or the separate
// residency index, use a 1 m cell just because the default happens to also
// be 1. Three independent facts, documented/verified here so a reader of
// this file cannot mistake hashCellSize==1 for "the" live cell size:
//
//   (a) hashCellSize is read ONLY inside MakeBroadphase's
//       BroadphaseKind::Hash case (PhysicsWorld.cpp:93-107). Every other
//       kind -- including Tree, the default every test in this suite (and
//       every survey'd P4 file) uses -- ignores it entirely. Verified below:
//       an absurd hashCellSize under the default WorldDef still yields a
//       DynamicTree-backed world (FixtureBroadphaseTree() non-null).
//
//   (b) PhysicsWorld::m_residencyGrid (PhysicsWorld.hpp:1402) is a SEPARATE,
//       hardcoded SpatialGrid{Real(1)} -- tagged
//       TODO(map-integration): wire to the map's real tile size -- with NO
//       wiring to WorldDef.hashCellSize at all. Verified below: it stays
//       1 m even when hashCellSize is set far away from 1.
//
//   (c) SpatialHash's own argless-default cellSize is ALSO Real(1)
//       (SpatialHash.hpp:51, `explicit SpatialHash(Real cellSize = Real(1))`)
//       -- a third independent "1 m" that happens to agree with (a)/(b) by
//       authoring convention, not by any shared code path (no live caller or
//       test ever default-constructs a SpatialHash). SpatialHash exposes no
//       public accessor for its cell size (m_cellSize is private, unread by
//       any test or live caller), so this fact is documented here rather
//       than asserted -- adding an engine getter solely to assert a private
//       constant is out of scope for this tests-only pass.
TEST_CASE("hashCellSize blind spot: Tree broadphase and the residency grid both ignore it", "[physics][mks]")
{
    WorldDef wd;
    wd.hashCellSize = Real(999); // deliberately absurd; must have zero effect under the default (Tree) broadphase
    PhysicsWorld w(wd);

    // (a) Still Tree-backed -- hashCellSize did not reroute or reconfigure
    // the default broadphase in any way.
    CHECK(w.FixtureBroadphaseTree() != nullptr);

    // (b) The residency grid's tile size is untouched by hashCellSize.
    CHECK(w.ResidencyGrid().TileSize() == Approx(Real(1)));
}
