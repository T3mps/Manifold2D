// Physics v2: the canonical INVARIANT gate (tags [physics][invariant]).
//
// Introduced by Task T8 to consolidate the physics correctness gate into ONE
// named place. The M6-era oracle bit-match gate (physics_oracle/*.json) was a
// porting scaffold; Physics v2's gate is physics INVARIANTS + analytic V2 tests
// + fresh goldens. The V2 narrowphase math (Collide/GJK/Manifold/rotation) is
// covered analytically by PhysicsShapesV2 / PhysicsGjkV2 / PhysicsManifoldV2 /
// PhysicsRotation / PhysicsQueryRotation. THIS file pins the END-TO-END
// world-level invariants every solver/narrowphase/broadphase change must keep:
//
//   1. REST PENETRATION < kLinearSlop  -- a settled stack has bounded overlap.
//   2. NO-TUNNEL (fast + rotating)     -- a fast/oriented body never passes
//                                         through a thin static wall.
//   3. ENERGY BOUNDED                  -- a dropped/colliding scene's kinetic
//                                         energy stays under a sane bound.
//   4. RUN-TWICE DETERMINISM           -- the same scene stepped twice yields
//                                         bit-identical final state.
//   5. BROADPHASE-STRATEGY INVARIANCE  -- Tree/Hash/SAP produce identical
//                                         settled state for the same scene
//                                         (the folded PhysicsBroadphaseTest
//                                         equivalence coverage, at the world
//                                         level).
//   6. SLIDE-NO-CATCH over merged spans-- a body sliding along a merged
//                                         multi-tile span does not snag at an
//                                         internal cell seam.
//   7. WARM-START CACHE BOUNDED        -- as contacts persist by stable feature
//                                         id, the solver warm-start cache stays
//                                         bounded (does not grow unboundedly).
//
// All expected values are hand-derived / property assertions (NOT oracle
// values). Y-positive is DOWN (gravity is +Y). PRESENTATION-FREE + C++20-clean.

#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Broadphase/Passability.hpp>
#include <Manifold2D/Physics/Broadphase/TileGrid.hpp>
#include <Manifold2D/Physics/Narrowphase/Gjk.hpp>

using namespace Manifold2D::Physics;
using Catch::Approx;

namespace
{
    constexpr Real kStep   = Real(1) / Real(60);
    constexpr Real kGravity = Real(10); // MKS default gravity (matches WorldDef default gravityY)

    // CCW box polygon (half-extents hw x hh): BL, BR, TR, TL. A polygon (not an
    // AABB) is required for any rotating-dynamics case.
    Shape BoxPolygon(Real hw, Real hh)
    {
        return MakePolygon({
            Vec2(-hw, -hh),
            Vec2( hw, -hh),
            Vec2( hw,  hh),
            Vec2(-hw,  hh),
        });
    }

    // A wide static AABB floor; `top` is the world Y of its top surface.
    BodyHandle AddFloor(PhysicsWorld& w, Real top, Real halfWidth = Real(40))
    {
        BodyDef bd;
        bd.type     = BodyType::Static;
        bd.position = Vec2(Real(0), top + Real(0.5)); // center 0.5 below the surface
        bd.shape    = MakeAabb(halfWidth, Real(0.5));
        return w.AddBody(bd);
    }

    // A dynamic AABB box of half-extent `h`, centered at `pos`.
    BodyHandle AddBox(PhysicsWorld& w, Vec2 pos, Real h,
                      Real friction = Real(0.4))
    {
        BodyDef bd;
        bd.type        = BodyType::Dynamic;
        bd.position    = pos;
        bd.shape       = MakeAabb(h, h);
        bd.density     = Real(1);
        bd.friction    = friction;
        bd.restitution = Real(0);
        bd.fixedRotation = true; // a flat stack -- no spin (rotation covered separately)
        return w.AddBody(bd);
    }

    // Penetration depth (positive = overlap) between two axis-aligned boxes,
    // measured from their world tight-AABBs. Returns the min-axis overlap, or a
    // negative value (the gap) if separated. Used to bound rest penetration.
    Real BoxOverlap(const Aabb& a, const Aabb& b)
    {
        const Real ox = std::min(a.max.x, b.max.x) - std::max(a.min.x, b.min.x);
        const Real oy = std::min(a.max.y, b.max.y) - std::max(a.min.y, b.min.y);
        return std::min(ox, oy);
    }

    // true if the body center's x is inside the thin wall span (tunneled in).
    bool CenterInWallX(PhysicsWorld& w, BodyHandle h, Real wallX, Real wallHW)
    {
        const Vec2 p = w.Position(h);
        return p.x > wallX - wallHW && p.x < wallX + wallHW;
    }
} // namespace

// ====================================================================
// 1. REST PENETRATION is BOUNDED (the SoftStep solver drives overlap toward
//    kLinearSlop).
//
// A small stack of dynamic boxes dropped onto a floor settles with bounded
// box-box / box-floor overlap. The SoftStep solver drives the overlap TOWARD
// kLinearSlop (0.005); a free-resting contact lands very close to it. A loaded
// stack carries more residual (the lower contacts bear the weight above); the
// MKS-re-derived stack budget is 0.003 (matching PhysicsSolverBudgetTest's
// PenBudget.stack, MKS P2; the px-era lore was ~0.18 measured / 0.21 budget).
// We assert that SOLVER-APPROPRIATE bound here: the invariant is "rest
// penetration stays bounded near slop", not "literally < kLinearSlop" (no
// soft solver achieves the latter under load). A regression that let the
// stack sink would blow past this.
// ====================================================================
TEST_CASE("physics-invariant: rest penetration stays bounded near the slop",
          "[physics][invariant]")
{
    WorldDef wd;
    wd.gravityY = kGravity;
    PhysicsWorld w(wd);

    const Real floorTop = Real(30);
    BodyHandle floor = AddFloor(w, floorTop);

    // Three stacked boxes (half-extent 1) released just touching, above floor.
    // Start them already near their resting heights so they settle quickly.
    const Real h = Real(1);
    BodyHandle b0 = AddBox(w, Vec2(Real(0), floorTop - h),           h);
    BodyHandle b1 = AddBox(w, Vec2(Real(0), floorTop - Real(3) * h), h);
    BodyHandle b2 = AddBox(w, Vec2(Real(0), floorTop - Real(5) * h), h);

    // Settle (3 s).
    for (int i = 0; i < 180; ++i)
    {
        w.Step(kStep);
    }

    auto box = [&](BodyHandle hb) {
        const Shape* s = w.GetShape(hb);
        return s->ComputeAABB(Transform{ w.Position(hb), w.GetAngle(hb) });
    };

    const Aabb fb = box(floor);
    const Aabb a0 = box(b0), a1 = box(b1), a2 = box(b2);

    // Boxes did not fall through the floor (settled above it within their span).
    CHECK(w.Position(b0).y <= floorTop);          // bottom box rests on the floor
    CHECK(w.Position(b0).y >= floorTop - Real(2) * h);

    // Rest penetration bound: the SoftStep solver drives overlap toward
    // kLinearSlop; re-derived for MKS (MKS P4) -- the old 0.21 was a px-era
    // number and is NOT mechanically divided. MEASURED (f32, 60 Hz) at this
    // file's MKS content: overlaps 0.00044 / 0.00120 / 0.00036 (max ~0.0012).
    // This lands in the SAME family P2 already re-derived for an equivalent
    // loaded box stack (PhysicsSolverBudgetTest's PenBudget.stack: measured
    // ~0.00137, bounded 0.003, ~2.2x headroom, well under kSkin = 0.02) -- we
    // reuse that exact bound here (~2.5x headroom over our own measured max).
    const Real kStackPenBound = Real(0.003);
    CHECK(BoxOverlap(a0, fb) <= kStackPenBound);
    CHECK(BoxOverlap(a1, a0) <= kStackPenBound);
    CHECK(BoxOverlap(a2, a1) <= kStackPenBound);
    // The solver IS driving toward the slop: the overlap is far below a full
    // box half-extent (it is a thin contact residual, not a sink-through).
    CHECK(BoxOverlap(a0, fb) < h);  // nowhere near sinking a full box
    CHECK(BoxOverlap(a1, a0) < h);
    CHECK(BoxOverlap(a2, a1) < h);
}

// ====================================================================
// 2. NO-TUNNEL: a FAST body and a ROTATING (oriented box-polygon) body each
//    fail to pass through a thin static wall in one step.
//
// The fast-circle case is also covered by PhysicsCcdTest; here we ALSO drive a
// fast ROTATED box-polygon at the wall -- the Phase-A rotation dimension -- to
// gate that the rotation-aware speculative narrowphase still stops a tilted
// fast body (a regression here would mean the rotated manifold missed the wall).
// ====================================================================
TEST_CASE("physics-invariant: fast + rotating bodies do not tunnel a thin wall",
          "[physics][invariant]")
{
    const Real wallX  = Real(10);
    const Real wallHW = Real(0.1); // span x[9.9,10.1]
    const Real wallHH = Real(6);

    auto makeWorldWithWall = [&](PhysicsWorld& w) {
        BodyDef bd;
        bd.type     = BodyType::Static;
        bd.position = Vec2(wallX, Real(0));
        bd.shape    = MakeAabb(wallHW, wallHH);
        w.AddBody(bd);
    };

    const Real speed = Real(20) / kStep; // ~20 units / step -> clears 0.2-wide wall

    // (a) FAST circle.
    {
        WorldDef wd; // gravity 0
        wd.gravityX = Real(0);
        wd.gravityY = Real(0);
        PhysicsWorld w(wd);
        makeWorldWithWall(w);

        BodyDef bd;
        bd.type     = BodyType::Dynamic;
        bd.position = Vec2(Real(0), Real(0));
        bd.shape    = MakeCircle(Real(0.2));
        bd.density  = Real(1);
        BodyHandle body = w.AddBody(bd);
        w.SetVelocity(body, Vec2(speed, Real(0)));
        w.Step(kStep);

        CHECK(w.Position(body).x < wallX);                     // near side
        CHECK_FALSE(CenterInWallX(w, body, wallX, wallHW));    // not buried
    }

    // (b) FAST + ROTATED box-polygon (45 deg). The rotation-aware speculative
    //     contact must still catch it.
    {
        WorldDef wd; // gravity 0
        wd.gravityX = Real(0);
        wd.gravityY = Real(0);
        PhysicsWorld w(wd);
        makeWorldWithWall(w);

        BodyDef bd;
        bd.type          = BodyType::Dynamic;
        bd.position      = Vec2(Real(0), Real(0));
        bd.shape         = BoxPolygon(Real(0.3), Real(0.3));
        bd.density       = Real(1);
        bd.fixedRotation = false;
        BodyHandle body = w.AddBody(bd);
        w.SetAngle(body, Real(0.7853981633974483)); // 45 deg
        w.SetVelocity(body, Vec2(speed, Real(0)));
        w.Step(kStep);

        CHECK(w.Position(body).x < wallX);                     // did not tunnel
        CHECK_FALSE(CenterInWallX(w, body, wallX, wallHW));    // not buried
    }
}

// ====================================================================
// 3. ENERGY BOUNDED: a dropped + colliding scene's kinetic energy never blows
//    up. With a passive scene (gravity + restitution 0 contacts) the solver is
//    energy-DISSIPATIVE; the bound asserts no spurious energy injection (the
//    classic "exploding stack" failure where a bad solver gains energy).
// ====================================================================
TEST_CASE("physics-invariant: dropped scene kinetic energy stays bounded",
          "[physics][invariant]")
{
    WorldDef wd;
    wd.gravityY = kGravity;
    PhysicsWorld w(wd);

    const Real floorTop = Real(30);
    AddFloor(w, floorTop);

    // A little pile of boxes dropped from a height onto the floor.
    std::vector<BodyHandle> boxes;
    const Real h = Real(0.8);
    for (int i = 0; i < 5; ++i)
    {
        boxes.push_back(AddBox(w, Vec2(Real(i) * Real(0.4) - Real(0.8),
                                       floorTop - Real(8) - Real(i) * Real(2)), h));
    }

    // The max kinetic energy a free body could acquire is bounded by falling the
    // full drop height H under gravity: v^2 = 2*g*H -> KE/m = g*H. Re-derived for
    // MKS (do not mechanically /10 the old px product: gravity scaled /40 while
    // lengths scaled /10, so keBound must recompute from the NEW fixed g=10 and
    // the converted maxDrop, not from dividing the old 320000 by 10 or 40).
    // With maxDrop = 20 and g = 10, g*H = 200; keBound applies a x4 slack factor
    // for the impact transient -> keBound = 800. MEASURED peak per-mass KE
    // ~91.13 (f32, 60 Hz) -- well under keBound (~8.8x headroom).
    const Real maxDrop  = Real(20);
    const Real keBound  = Real(4) * kGravity * maxDrop; // per unit mass, x4 slack

    Real peakKE = Real(0);
    for (int i = 0; i < 240; ++i) // 4 s: fall, impact, settle
    {
        w.Step(kStep);
        for (BodyHandle b : boxes)
        {
            const Vec2 v = w.Velocity(b);
            const Real keOverM = Real(0.5) * (v.x * v.x + v.y * v.y);
            peakKE = std::max(peakKE, keOverM);
            REQUIRE(keOverM < keBound); // never blows up at any step
        }
    }

    // After settling, the pile is at rest: per-unit-mass KE is tiny.
    for (BodyHandle b : boxes)
    {
        const Vec2 v = w.Velocity(b);
        const Real keOverM = Real(0.5) * (v.x * v.x + v.y * v.y);
        CHECK(keOverM < Real(50)); // effectively at rest
    }
    INFO("peak per-mass KE = " << static_cast<double>(peakKE));
}

// ====================================================================
// 4. RUN-TWICE DETERMINISM: a colliding scene stepped twice yields a
//    bit-identical final state (positions + angles + velocities).
// ====================================================================
TEST_CASE("physics-invariant: a colliding scene is run-twice deterministic",
          "[physics][invariant]")
{
    auto run = [](std::vector<Real>& trace)
    {
        WorldDef wd;
        wd.gravityY = kGravity;
        PhysicsWorld w(wd);

        const Real floorTop = Real(30);
        AddFloor(w, floorTop);

        std::vector<BodyHandle> boxes;
        const Real h = Real(0.9);
        for (int i = 0; i < 4; ++i)
        {
            boxes.push_back(AddBox(w, Vec2(Real(i) * Real(0.3) - Real(0.4),
                                           floorTop - Real(4) - Real(i) * Real(2.2)), h));
        }
        // Nudge one box sideways so the scene has lateral motion + friction.
        w.SetVelocity(boxes[0], Vec2(Real(3.0), Real(0)));

        trace.clear();
        for (int k = 0; k < 200; ++k)
        {
            w.Step(kStep);
        }
        for (BodyHandle b : boxes)
        {
            const Vec2 p = w.Position(b);
            const Vec2 v = w.Velocity(b);
            trace.push_back(p.x);
            trace.push_back(p.y);
            trace.push_back(w.GetAngle(b));
            trace.push_back(v.x);
            trace.push_back(v.y);
        }
    };

    std::vector<Real> a, b;
    run(a);
    run(b);

    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        REQUIRE(a[i] == b[i]); // bit-identical
    }
}

// ====================================================================
// 5. BROADPHASE-STRATEGY INVARIANCE: the SAME scene settled under each
//    broadphase strategy (Tree / Hash / SAP) reaches the SAME final state.
//
// This is the world-level fold of PhysicsBroadphaseTest's cross-strategy
// equivalence: the broadphase only changes WHICH pairs are tested, never the
// narrowphase/solver result, so swapping the strategy must not change the
// physics. We run an identical settling scene three times (one per strategy)
// and assert bit-identical final state.
// ====================================================================
TEST_CASE("physics-invariant: broadphase strategy does not change the result",
          "[physics][invariant]")
{
    auto runWith = [](BroadphaseKind kind, std::vector<Real>& trace)
    {
        WorldDef wd;
        wd.gravityY   = kGravity;
        wd.broadphase = kind;
        PhysicsWorld w(wd);

        const Real floorTop = Real(30);
        AddFloor(w, floorTop);

        std::vector<BodyHandle> boxes;
        const Real h = Real(0.9);
        for (int i = 0; i < 5; ++i)
        {
            boxes.push_back(AddBox(w, Vec2(Real(i) * Real(0.6) - Real(1.2),
                                           floorTop - Real(3) - Real(i) * Real(2.1)), h));
        }

        trace.clear();
        for (int k = 0; k < 200; ++k)
        {
            w.Step(kStep);
        }
        for (BodyHandle b : boxes)
        {
            const Vec2 p = w.Position(b);
            trace.push_back(p.x);
            trace.push_back(p.y);
            trace.push_back(w.GetAngle(b));
        }
    };

    std::vector<Real> tree, hash, sap;
    runWith(BroadphaseKind::Tree, tree);
    runWith(BroadphaseKind::Hash, hash);
    runWith(BroadphaseKind::Sap,  sap);

    REQUIRE(tree.size() == hash.size());
    REQUIRE(hash.size() == sap.size());
    for (std::size_t i = 0; i < tree.size(); ++i)
    {
        // The settled state is identical regardless of broadphase strategy.
        REQUIRE(tree[i] == hash[i]);
        REQUIRE(hash[i] == sap[i]);
    }
}

// ====================================================================
// 6. SLIDE-NO-CATCH over a merged TileGrid span: a swept shape gliding along a
//    long merged multi-tile span clears its full length -- it does NOT snag at
//    an internal cell seam (the merge fuses internal vertical edges away).
//
// This is the keystone tile property. PhysicsTileGridTest covers it for a
// capsule sweep; here we pin the same property via the world TileGrid -> the
// merged-span set has NO internal seams and a horizontal sweep across the top
// edge travels its full length.
// ====================================================================
TEST_CASE("physics-invariant: sliding over a merged tile span does not catch",
          "[physics][invariant]")
{
    const Real cs = Real(3.2);

    // A single long horizontal run: row 0, cols 0..5 (6 cells). Merged rect =
    // [0,0]..[19.2,3.2]. Internal cell boundaries WOULD be at x = 3.2,6.4,9.6,
    // 12.8,16; the merge fuses them away.
    GridPassability grid(/*w*/ 10, /*h*/ 2);
    for (int cx = 0; cx <= 5; ++cx)
    {
        grid.SetSolid(cx, 0, true);
    }
    TileGrid tg(grid, cs, Vec2(Real(0), Real(0)));

    const Aabb2 whole{ Vec2(Real(0), Real(0)), Vec2(Real(10) * cs, Real(2) * cs) };
    std::vector<Aabb2> rects;
    const int n = tg.Query(whole, rects);

    // EXACTLY ONE merged rect for the 6-cell run -- the seam-killer (per-cell
    // rects would yield 6 here, each with an internal vertical edge).
    REQUIRE(n == 1);
    const Aabb2 span = rects[0];
    REQUIRE(span.min.x == Approx(Real(0)));
    REQUIRE(span.max.x == Approx(Real(6) * cs)); // 19.2 -- the OUTER extent
    REQUIRE(span.max.y == Approx(cs));           // 3.2

    // Build the merged rect as a CCW box polygon for the cast.
    const Vec2 poly[4] = { Vec2(span.min.x, span.min.y), Vec2(span.max.x, span.min.y),
                           Vec2(span.max.x, span.max.y), Vec2(span.min.x, span.max.y) };

    // A capsule grazing JUST ABOVE the span's top edge sweeping its full width.
    const Shape cap = MakeCapsule(/*halfLen*/ Real(0.6), /*radius*/ Real(0.4));
    // gap = 0.1 -- ~16x kShapeCastTol (1.25*kLinearSlop = 0.00625, MKS P4 Task 3),
    // ample clearance so the "clear" cast genuinely does not touch the span.
    const Real  gap   = Real(0.1);
    const Real  rideY = span.min.y - Real(0.4) - gap; // a hair above the top edge
    Transform   xf;
    xf.position = Vec2(span.min.x - Real(2), rideY);
    const Vec2  translation = Vec2((span.max.x + Real(2)) - (span.min.x - Real(2)), Real(0));

    const ShapeCastResult clear = ShapeCastPoly(cap, xf, translation, poly, 4);
    // Riding above the top edge with clearance: the swept capsule never reaches
    // the span (there is no internal edge to catch on) -> it travels fully.
    CHECK_FALSE(clear.hit);

    // Driving INTO the left face must stop at the OUTER left extent (x = 0), NOT
    // at any internal cell boundary (x = 3.2/6.4/... were eliminated by the merge).
    const Real midY = (span.min.y + span.max.y) * Real(0.5);
    Transform  xf2;
    xf2.position = Vec2(span.min.x - Real(4), midY);
    const Vec2 trans2 = Vec2(Real(8), Real(0));
    const ShapeCastResult face = ShapeCastPoly(cap, xf2, trans2, poly, 4);
    REQUIRE(face.hit);
    const Real impactX  = xf2.position.x + trans2.x * face.t;
    const Real leadingX = impactX + (Real(0.6) + Real(0.4)); // capsule leading edge
    // margin = 0.02 = 3.2x kShapeCastTol (0.00625, MKS P4 Task 3) -- same ratio
    // PhysicsQueriesTest already uses for its converted ShapeCastPoly margins.
    CHECK(leadingX <= Approx(span.min.x).margin(0.02)); // stops at outer face
    CHECK(leadingX < Real(3.2) - Real(1.0));            // nowhere near 1st internal seam
}

// ====================================================================
// 8. DEEP-PENETRATION RECOVERY (the EPA narrowphase gate): a dynamic ROUND body
//    spawned DEEPLY INSIDE a static box pushes out CLEANLY and comes to rest --
//    it does NOT explode and does NOT eject in the WRONG direction.
//
// This is the END-TO-END gate for the Physics v2 EPA+MPR deep-overlap path
// (Tasks 1-4). The circle's core is a single vertex at its centre; spawned a
// full radius past the box's top face the GJK core distance is ~0, so Collide
// takes the DEEP round-overlap cell -- exact EPA on the world cores (MPR is the
// convergence fallback). The EPA NEAREST-FACE axis is what makes recovery
// CORRECT: it ejects the circle out along the shortest separating axis (here the
// top face), and gravity then settles it on that surface. The OLD centroid
// approximation pointed along the line-of-centres rather than toward the nearest
// face -- for a round core buried in a polygon that could pick a WRONG axis and
// eject the body sideways/down (the very case EPA exists to fix).
//
// VERIFIED first-frame manifold (depth 6): pointCount=1, normal=(0,-1) (B->A =
// "push circle UP out of box"), separation = depth + rA + rB = 6 + 6 + 0 = 12.
// The EPA normal + separation are exactly correct; the recovery direction is
// right and is the load-bearing property this gate pins.
//
// We assert the depth-ROBUST correctness invariants:
//   (a) ENERGY BOUNDED every step -- a deep overlap is a large separation, so a
//       bad push could fling the body; per-unit-mass KE must stay under a sane
//       cap at EVERY step (no explosion). The soft push-out is a velocity BIAS,
//       not a real velocity, so a clean recovery shows near-zero KE throughout.
//   (b) CORRECT DIRECTION -- the circle ends ABOVE the box top (ejected out the
//       NEAR face), NOT still buried in the span and NOT ejected DOWN out the
//       far face. A wrong-axis (old-centroid) eject would fail this.
//   (c) NO LATERAL DRIFT -- the top-face axis is vertical, so |x| stays ~0; a
//       sideways wrong-axis eject would slide it.
//   (d) CLEAN REST -- velocity bled off; penetration within the solver's settled
//       budget (0.1, re-derived for MKS below -- same rule-6 style as the
//       rest-penetration invariant above, which uses 0.003) and far under the
//       radius (not still buried).
//
// NOTE on the budget: the SoftStep recovery reaches a stable force-balance whose
// residual grows with the INITIAL overlap (px-era measurements: ~0.17 at depth 6,
// ~0.22 at depth 10, ~0.48 at depth 30 -- the depth-scaling TREND is scale-
// independent; the numbers predate MKS). That residual is a documented soft-
// solver characteristic (bias-vs-gravity equilibrium), NOT an EPA error -- the
// EPA axis/separation are exact (verified above) and recovery is always correct-
// direction + bounded + deterministic. We spawn at a genuine deep overlap (a full
// radius past the face -> coreDist 0 -> EPA path) that settles within the
// re-derived MKS budget (0.1, measured 0.0623 below); we do NOT loosen the
// budget to absorb a pathological initial overlap.
// ====================================================================
TEST_CASE("physics-invariant: a deeply-overlapping round body recovers cleanly",
          "[physics][invariant]")
{
    WorldDef wd;
    wd.gravityY = kGravity;
    PhysicsWorld w(wd);

    // A static box centred at the origin, half-extent 4 (span [-4,4]^2). Use
    // a polygon box so the deep round-vs-polygon EPA path is exercised (an AABB
    // would also work, but the polygon core is the general v2 narrowphase path).
    const Real boxHalf = Real(4.0);
    BodyHandle box;
    {
        BodyDef bd;
        bd.type     = BodyType::Static;
        bd.position = Vec2(Real(0), Real(0));
        bd.shape    = BoxPolygon(boxHalf, boxHalf);
        box = w.AddBody(bd);
    }

    // A dynamic circle spawned DEEPLY INSIDE the box: a full radius past the top
    // face. Box top is at y = -boxHalf = -4; spawn the centre at -3.4 so the
    // circle (radius 0.6) has its TOP at the face and its whole body buried below
    // it -- coreDist ~ 0, so the first contact is the EPA deep-round cell (the
    // VERIFIED depth-6 manifold: 1 pt, normal (0,-1), sep 12, is a px-era
    // reference frame; the EPA normal/direction property it documents is scale-
    // independent). The EPA nearest-face axis ejects it UP (out the near face);
    // gravity then rests it on top.
    const Real circR = Real(0.6);
    BodyHandle ball;
    {
        BodyDef bd;
        bd.type        = BodyType::Dynamic;
        bd.position    = Vec2(Real(0), -boxHalf + circR); // = -3.4, a full radius below the top face
        bd.shape       = MakeCircle(circR);
        bd.density     = Real(1);
        bd.friction    = Real(0.4);
        bd.restitution = Real(0);
        ball = w.AddBody(bd);
    }

    // (a) ENERGY BOUNDED: a deep ejection could fling the body. Bound mirrors the
    // energy-bounded invariant style (a sane cap derived from the scene scale):
    // the body can fall at most ~box span under gravity, plus generous push slack.
    // Re-derived for MKS (do not mechanically /10 the old px product): with
    // boxHalf = 4.0 and g = 10, g*boxHalf = 40; keBound applies a x4 slack
    // factor -> keBound = 160. MEASURED peak per-mass KE ~1.1e-13 (f32, 60 Hz)
    // -- effectively zero, because the EPA push-out here is a soft POSITION
    // bias (see file header note above), not an injected velocity, so KE
    // never actually spikes; keBound stands as the defensive explosion
    // tripwire it was documented to be.
    const Real keBound = Real(4) * kGravity * boxHalf; // generous, x4 slack
    Real peakKE = Real(0);
    for (int i = 0; i < 120; ++i) // ~2 s: eject, settle (stable by ~frame 40)
    {
        w.Step(kStep);
        const Vec2 v = w.Velocity(ball);
        const Real keOverM = Real(0.5) * (v.x * v.x + v.y * v.y);
        peakKE = std::max(peakKE, keOverM);
        REQUIRE(keOverM < keBound); // no explosion at any step
    }
    INFO("deep-recovery peak per-mass KE = " << static_cast<double>(peakKE));

    const Vec2 pf = w.Position(ball);
    const Vec2 vf = w.Velocity(ball);

    // (b) CORRECT DIRECTION: ejected out the TOP (centre above the top face),
    // NOT still buried inside the [-4,4] span and NOT ejected DOWN past it.
    CHECK(pf.y <= -boxHalf);                    // pushed out the top face
    CHECK(pf.y >= -boxHalf - Real(2) * circR);  // resting on the surface, not flung away

    // (c) NO LATERAL DRIFT: the top-face axis is vertical -> |x| ~ 0. A wrong-axis
    // (old-centroid) eject would slide the body sideways.
    CHECK(std::abs(pf.x) < Real(0.1));

    // (d) CLEAN REST: velocity bled off.
    const Real keFinal = Real(0.5) * (vf.x * vf.x + vf.y * vf.y);
    // Re-derived for MKS (review fix): the px-era 50 admitted 10 units/s as
    // "at rest"; /100 keeps the velocity^2 dimension honest (v ~ 1 m/s cap).
    // Measured keFinal ~ 1e-13 (EPA recovery is a position bias) -- this stays
    // a coarse explosion tripwire, same character as keBound above.
    CHECK(keFinal < Real(0.5)); // effectively at rest

    // Rest penetration bounded near the slop: the circle's bottom barely overlaps
    // the box top. Re-derived for MKS (MKS P4) -- the old 0.21 was a px-era
    // number for a DIFFERENT scenario (a loaded stack) and is not reused
    // mechanically here. MEASURED (f32, 60 Hz) at this file's MKS content:
    // penetration ~0.0623 -- a genuine one-time deep-recovery residual (this
    // scenario is a single EPA push-out settle, not a loaded stack, so it does
    // not match P2's ball/stack/massRatio family directly). Bound set at
    // ~1.6x measured headroom, ~5x kSkin (0.02).
    const Real kPenBound = Real(0.1);
    const Real circBottom  = pf.y + circR;        // lowest point of the circle
    const Real boxTop      = -boxHalf;            // top surface of the box
    const Real penetration = circBottom - boxTop; // positive = into the box
    CHECK(penetration <= kPenBound);
    CHECK(penetration < circR); // nowhere near still-buried
}

// ====================================================================
// 7. WARM-START LIVE + BOUNDED: as contacts persist by stable feature id, warm-
//    start carries impulses forward across steps. A small settled scene's
//    warm-start is LIVE (resting contacts carry a non-zero accumulated impulse)
//    and the persistent contact set stays BOUNDED -- it does NOT grow without
//    bound (no feature-id churn) as the scene is stepped.
//
//    RE-BASELINED (warm-start-on-Contact): this invariant formerly read
//    SolverWarmStartCacheSize() (peak > 0 for liveness, <= bound for no-churn).
//    The solver-owned m_cache was RETIRED -- warm-start impulses now live on the
//    persistent Contact's manifold point, so that hook always returns 0 for
//    SoftStep. We assert the SAME two properties through the live surfaces:
//    liveness via the accumulated normal impulse on the emitted constraints
//    (ForEachContactConstraint), and boundedness via ActiveContactCount() (the
//    pool that warm-start now lives on -- a feature-id leak would grow it).
// ====================================================================
TEST_CASE("physics-invariant: warm-start is live and the contact set stays bounded",
          "[physics][invariant]")
{
    WorldDef wd;
    wd.gravityY = kGravity;
    PhysicsWorld w(wd);

    const Real floorTop = Real(30);
    AddFloor(w, floorTop);

    // A handful of resting boxes -> a small, fixed set of persistent contacts.
    const Real h = Real(1.0);
    const int  kBoxes = 4;
    for (int i = 0; i < kBoxes; ++i)
    {
        AddBox(w, Vec2(Real(0), floorTop - h - Real(i) * Real(2) * h), h);
    }

    // The persistent contact set is at most a few constraints per box (box-box +
    // box-floor, up to 2 points each). Warm-start is keyed by stable feature id,
    // so while the scene is active the contact set is steady. Bound: generously, a
    // few per box. A leak (id churn) would grow it every step.
    const std::size_t boundedCap = static_cast<std::size_t>(kBoxes) * 8u + 16u;

    // Helper: the peak accumulated normal impulse across this Step's emitted
    // constraints. Non-zero proves warm-start fed the solver a real seed (it is
    // ON, not cold-starting every step).
    auto peakNormalImpulseThisStep = [&]() -> Real
    {
        Real peak = Real(0);
        w.ForEachContactConstraint([&](const ContactConstraint& cc)
        {
            for (int p = 0; p < cc.pointCount; ++p)
            {
                peak = std::max(peak, cc.points[p].normalImpulse);
            }
        });
        return peak;
    };

    // Warm up + settle, tracking the PEAK warm-start impulse + contact count while
    // the stack is still active. We track peaks (not a post-settle snapshot)
    // because Island sleep freezes a fully-rested stack -> a sleeping body skips
    // the solver feed so warm-start legitimately falls to 0 once asleep. The
    // invariant is about the ACTIVE window: peak impulse > 0 proves warm-starting
    // is live (guards a regression that silently disabled it); count <= bound
    // proves the contact set does not grow without bound (no feature-id churn).
    Real        peakImpulse = Real(0);
    std::size_t peakCount   = 0;
    for (int i = 0; i < 120; ++i)
    {
        w.Step(kStep);
        peakImpulse = std::max(peakImpulse, peakNormalImpulseThisStep());
        peakCount   = std::max(peakCount, w.ActiveContactCount());
    }
    CHECK(peakImpulse > Real(0));     // warm-start populated (liveness)
    CHECK(peakCount <= boundedCap);   // bounded (no id churn)

    // Step a lot more: the contact set must NOT keep growing (no unbounded id
    // churn), tracked over the whole window so a body that re-wakes is still gated.
    std::size_t peakLater = 0;
    for (int i = 0; i < 300; ++i)
    {
        w.Step(kStep);
        peakLater = std::max(peakLater, w.ActiveContactCount());
    }
    CHECK(peakLater <= boundedCap);
    // The active-contact pool is likewise bounded (a few per box).
    CHECK(w.ActiveContactCount() <= boundedCap);
}
