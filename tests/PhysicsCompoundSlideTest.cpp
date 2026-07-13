// Physics v2 compound-body behavior guards (Task: compound-COM / multi-fixture
// bug investigation).
//
// REPORTED (interactive sandbox "Compound bodies" / "Mixed shapes"): off-COM
// 2-fixture bodies (a light core circle + a heavy offset circle) looked buggy --
// accelerating while sliding, over-penetrating. Investigation found the Core
// SoftStep solver is ROBUST for compound bodies (it settles them in every
// configuration), and pinned two real defects in the compound/multi-fixture path
// that these tests + PhysicsCompoundComTest (e) guard:
//
//   1. WARM-START ID COLLISION across fixture pairs -- two fixtures of one body
//      that hit the same surface produced the SAME warm-start key and aliased
//      (one fixture's impulse seeded the other's at the wrong lever arm). PROBE
//      below pins that each contact point now owns a DISTINCT key and is
//      individually warm-started (non-zero accumulated impulse). [Originally
//      asserted via the solver cache size; re-baselined after warm-start moved
//      onto the persistent Contact -- the solver cache was retired.]
//   2. ApplyImpulse-at-point torqued about the ORIGIN, not the COM (off-COM
//      bodies span wrong) -- guarded analytically by PhysicsCompoundComTest (e).
//
// The behavioral guards (A/B) pin that compound bodies -- the EXACT sandbox
// "lopsided" body, dropped and free to rotate, including dynamic-on-dynamic
// piles -- DISSIPATE to rest and never inject energy or tunnel the floor.
//
// CONVENTION: Cartesian, +Y DOWN (gravity +Y). PRESENTATION-FREE + C++20-clean.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Manifold2D/Physics/Fixture.hpp>
#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>

using namespace Manifold2D::Physics;
using Catch::Approx;

namespace
{
    constexpr Real kStep    = Real(1) / Real(60);
    constexpr Real kGravity = Real(10); // MKS default gravity magnitude

    // Wide static AABB floor centered at x=0; `top` is the world Y of its top
    // surface. Half-width 5.6 -> spans x[-5.6,5.6], so bodies near x=0 cannot
    // roll off the edge.
    BodyHandle AddFloor(PhysicsWorld& w, Real top, Real halfWidth = Real(5.6))
    {
        BodyDef bd;
        bd.type     = BodyType::Static;
        bd.position = Vec2(Real(0), top + Real(0.2));
        bd.shape    = MakeAabb(halfWidth, Real(0.2));
        bd.friction = Real(0.5);
        return w.AddBody(bd);
    }

    void AddWall(PhysicsWorld& w, Real cx, Real cy, Real hw, Real hh)
    {
        BodyDef bd;
        bd.type     = BodyType::Static;
        bd.position = Vec2(cx, cy);
        bd.shape    = MakeAabb(hw, hh);
        bd.friction = Real(0.5);
        w.AddBody(bd);
    }

    Real KineticEnergy(const PhysicsWorld& w, BodyHandle b)
    {
        const Vec2 v  = w.Velocity(b);
        const Real wv = w.AngVelSlot(b.index);
        return Real(0.5) * w.GetBodyMass(b) * (v.x * v.x + v.y * v.y)
             + Real(0.5) * w.GetBodyInertia(b) * (wv * wv);
    }

    // EXACT sandbox "Compound bodies" body (Scenes.cpp makeLopsided), at MKS
    // scale (/100 from the sandbox's ~100 px/m authoring): a light core circle
    // (r0.18, density 0.5) at the body origin + a heavy circle (r0.22, density
    // 4.0) at local (heavySign*0.4, 0). Dynamic, free to rotate, rest 0,
    // friction 0.5. The aggregate COM sits at local x ~ +0.369*heavySign.
    BodyHandle MakeLopsided(PhysicsWorld& w, Vec2 pos, Real heavySign)
    {
        BodyDef bd;
        bd.type          = BodyType::Dynamic;
        bd.position      = pos;
        bd.shape         = MakeCircle(Real(0.18));
        bd.density       = Real(0.5);
        bd.friction      = Real(0.5);
        bd.restitution   = Real(0);
        bd.fixedRotation = false;
        const BodyHandle bh = w.AddBody(bd);

        FixtureDef fd;
        fd.shape       = MakeCircle(Real(0.22));
        fd.localPos    = Vec2(heavySign * Real(0.4), Real(0));
        fd.density     = Real(4.0);
        fd.friction    = Real(0.5);
        fd.restitution = Real(0);
        w.AddFixture(bh, fd);
        return bh;
    }
} // namespace

// ============================================================================
// PROBE [FIX 1 GUARD]: two-fixture contacts own DISTINCT warm-start ids.
//
// A 2-fixture body whose two fixtures both rest on the floor produces two
// contact points. Warm-start is keyed by the contact point id; before the fix
// both points shared one id (the geometric feature id carried no fixture
// identity) and ALIASED -- one fixture's accumulated impulse would seed the
// other's contact at the wrong lever arm. After the fix each fixture pair's
// contact owns a distinct key (MixContactId folds the fixture slots in).
//
// RE-BASELINED (warm-start-on-Contact): this PROBE used SolverWarmStartCacheSize
// >= 2 as a proxy for "two distinct warm-start ids exist" -- but the solver-owned
// m_cache was RETIRED (warm-start impulses now live on the persistent Contact's
// manifold point, seeded into each ContactConstraintPoint at emit), so that hook
// always returns 0 for SoftStep. We assert the property the cache-size check was
// a proxy for, DIRECTLY and more strictly: there are >= 2 emitted contact points,
// their warm-start ids are DISTINCT (no aliasing), AND each carries a non-zero
// accumulated normal impulse (both fixtures are individually warm-started -- an
// aliasing regression would collapse them to one warm-started point + one cold).
// ============================================================================
TEST_CASE("compound-slide PROBE: two-fixture contacts have distinct warm-start ids",
          "[physics]")
{
    WorldDef wd;
    wd.gravityY = kGravity;
    PhysicsWorld w(wd);

    AddFloor(w, Real(6.2));

    // A flat 2-circle body (both r=0.12) straddling so BOTH circles rest on the
    // floor: fixture0 at origin, fixture1 at +0.4. Centers at y=6.2-0.12=6.08.
    BodyDef bd;
    bd.type          = BodyType::Dynamic;
    bd.position      = Vec2(Real(-0.2), Real(6.08));
    bd.shape         = MakeCircle(Real(0.12));
    bd.density       = Real(1);
    bd.friction      = Real(0.5);
    bd.fixedRotation = true;   // keep it flat so both circles stay on the floor
    BodyHandle body  = w.AddBody(bd);
    {
        FixtureDef fd;
        fd.shape    = MakeCircle(Real(0.12));
        fd.localPos = Vec2(Real(0.4), Real(0));
        fd.density  = Real(1);
        fd.friction = Real(0.5);
        w.AddFixture(body, fd);
    }

    for (int i = 0; i < 30; ++i) w.Step(kStep);

    // Collect each emitted contact point's (id, accumulated normal impulse). The
    // impulse on the LAST Step's constraints is the warm-start seed carried in
    // from the prior step's converged solve -- non-zero for a contact bearing the
    // body's weight, zero for a fresh/aliased-away point.
    std::vector<std::uint32_t> ids;
    std::vector<Real>          normalImpulses;
    w.ForEachContactConstraint([&](const ContactConstraint& cc)
    {
        for (int p = 0; p < cc.pointCount; ++p)
        {
            ids.push_back(cc.points[p].id);
            normalImpulses.push_back(cc.points[p].normalImpulse);
        }
    });

    const std::size_t activeContacts = w.ActiveContactCount();
    INFO("active contact constraints = " << activeContacts
         << ", emitted contact points = " << ids.size());

    CHECK(activeContacts >= 2u);     // both circles rest on the floor
    REQUIRE(ids.size() >= 2u);       // at least the two floor-contact points

    // No aliasing: every contact point id is DISTINCT (the FIX-1 invariant).
    std::sort(ids.begin(), ids.end());
    CHECK(std::adjacent_find(ids.begin(), ids.end()) == ids.end());

    // Warm-start is LIVE on each point: at least two points carry a non-zero
    // accumulated normal impulse (an aliasing regression would warm-start one and
    // leave the other cold -- this is the behavioral teeth the cache-size check
    // lacked).
    int warmStarted = 0;
    for (Real ni : normalImpulses)
    {
        if (ni > Real(0)) { ++warmStarted; }
    }
    CHECK(warmStarted >= 2);
}

// ============================================================================
// (A) DROPPED lopsided bodies SETTLE (no energy injection, no tunnelling). Two
//     of the EXACT sandbox compound bodies (one tipping each way) are dropped on
//     a floor, free to rotate. A passive (restitution-0) scene is energy-
//     DISSIPATIVE: after the impact transient each body comes to rest. A solver
//     that injected energy (the reported "accelerate") would keep them moving.
// ============================================================================
TEST_CASE("compound-slide (A): dropped lopsided bodies settle to rest", "[physics]")
{
    WorldDef wd;
    wd.gravityY = kGravity;
    PhysicsWorld w(wd);

    AddFloor(w, Real(6.2));
    // Spaced so the heavy lobes (centers -0.5/+0.5, r0.22) do not overlap, and
    // well inside the floor span so a rolling body cannot fall off the edge.
    BodyHandle b0 = MakeLopsided(w, Vec2(-Real(0.9), Real(3.6)),  Real(1)); // tips right
    BodyHandle b1 = MakeLopsided(w, Vec2( Real(0.9), Real(3.6)), -Real(1)); // tips left

    Real impactPeak = Real(0);
    for (int i = 0; i < 90; ++i)   // fall + impact + initial tip
    {
        w.Step(kStep);
        impactPeak = std::max(impactPeak,
                              KineticEnergy(w, b0) + KineticEnergy(w, b1));
    }
    REQUIRE(impactPeak > Real(0)); // the scene actually moved (guards a no-op test)

    Real lateMax = Real(0);
    for (int i = 90; i < 540; ++i) // 7.5 s to dissipate
    {
        w.Step(kStep);
        lateMax = std::max(lateMax, KineticEnergy(w, b0) + KineticEnergy(w, b1));
    }
    const Real finalKE = KineticEnergy(w, b0) + KineticEnergy(w, b1);
    INFO("impact-peak KE = " << static_cast<double>(impactPeak)
         << ", late-max KE = " << static_cast<double>(lateMax)
         << ", final KE = " << static_cast<double>(finalKE));

    // Energy never climbs back above the impact transient (no injection) --
    // scale-free ratio, unchanged.
    CHECK(lateMax <= impactPeak * Real(1.05));
    // ...and the pair dissipates essentially to rest -- scale-free ratio, unchanged.
    CHECK(finalKE < impactPeak * Real(0.02));
    // Neither body tunnelled the floor or flew away (stayed in the play area).
    // Re-baselined from the px-era "< 700" (floor top 620, 80 px margin) to the
    // same proportion at MKS scale: floor top 6.2, 0.8 m margin -> < 7.
    CHECK(w.Position(b0).y < Real(7));
    CHECK(w.Position(b1).y < Real(7));
}

// ============================================================================
// (B) CONFINED HEAP of free-rotating compound bodies dissipates. Six lopsided
//     bodies packed between two walls cannot separate -- they pile and slide on
//     each other in sustained multi-fixture contact (the pure-Core analog of
//     "compound bodies sliding on top of each other"). A passive heap MUST reach
//     rest; energy that kept growing here is the reported instability.
// ============================================================================
TEST_CASE("compound-slide (B): confined heap of compound bodies dissipates",
          "[physics]")
{
    WorldDef wd;
    wd.gravityY = kGravity;
    PhysicsWorld w(wd);

    AddFloor(w, Real(6.2));
    AddWall(w, -Real(0.9), Real(5.0), Real(0.2), Real(1.4)); // left wall
    AddWall(w,  Real(0.9), Real(5.0), Real(0.2), Real(1.4)); // right wall

    std::vector<BodyHandle> bodies;
    for (int i = 0; i < 6; ++i)
    {
        const Real sign = (i % 2 == 0) ? Real(1) : -Real(1);
        const Real x    = Real(-0.2 + (i % 3) * 0.2);
        const Real y    = Real(4.2 - 0.55 * i);
        bodies.push_back(MakeLopsided(w, Vec2(x, y), sign));
    }
    auto totalKE = [&] {
        Real s = Real(0);
        for (BodyHandle b : bodies) s += KineticEnergy(w, b);
        return s;
    };

    // Settle (5 s), then assert it stays at rest over the next 5 s -- a solver
    // that injects energy in confined multi-fixture contact would re-energize.
    for (int i = 0; i < 300; ++i) w.Step(kStep);
    Real lateMax = Real(0);
    for (int i = 0; i < 300; ++i)
    {
        w.Step(kStep);
        lateMax = std::max(lateMax, totalKE());
    }
    INFO("confined-heap late-max KE = " << static_cast<double>(lateMax));

    // The whole heap is at rest: measured lateMax == 0 (every body reached the
    // MKS sleepThreshold default of 0.05 m/s and went fully to sleep -- the
    // solver zeroes velocity on sleep). Re-baselined empirically for MKS
    // (protocol rule 6) from the px-era "< 50": bound derived analytically as
    // the worst-case linear residual KE for 6 lopsided bodies (mass ~0.66 each,
    // light r0.18/d0.5 + heavy r0.22/d4.0) sitting AT the 0.05 m/s sleep gate --
    // 6 * 0.5*0.66*0.05^2 ~ 0.0049 -- doubled for headroom (angular residual +
    // margin over the analytic worst case; measured value is 0, well inside).
    CHECK(lateMax < Real(0.01));
    // No body escaped the bowl / tunnelled the floor (same re-baseline as (A)).
    for (BodyHandle b : bodies)
        CHECK(w.Position(b).y < Real(7));
}
