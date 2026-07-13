// Physics SIMD solver: solver-local body-state store (Task 1) -- ROUND-TRIP test.
//
// The SIMD constraint-solver initiative mirrors the world's per-component Vec2
// fields into dense AoS rows (BodyState: 32-byte 32-aligned struct) stored by
// solver index (not world slot), so the lane-wide solve can gather/scatter body
// velocities by solverIndex lanes. This TU validates the FIRST de-risking piece:
// the SyncIn/SyncOut world<->solver bridge.
//
// SCOPE (Task 1): only the SoA struct + its two sync helpers exist yet -- the
// solver does NOT consume BodyStateStore (that is a later task). These tests pin
// the sync contract:
//   * SyncIn copies awake-dynamic world velocities into the packed arrays and
//     zeroes the TGS position deltas.
//   * mutating the packed velocities then SyncOut writes them BACK to the world.
//   * non-matching slots (a Static body) are LEFT UNTOUCHED by SyncOut.
//
// The awake-dynamic predicate MUST mirror SoftStep::FinalizePositionsSoA:
//   Alive(i) && TypeSlot(i) == BodyType::Dynamic && AwakeSlot(i).
//
// PRESENTATION-FREE + C++20-clean.

#include <cstdint>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Solver/BodyState.hpp>
#include <Manifold2D/Physics/Solver/ContactColoring.hpp>
#include <Manifold2D/Physics/Solver/ContactConstraintSimd.hpp>
#include <Manifold2D/Physics/Solver/Solver.hpp>

using namespace Manifold2D::Physics;
using Catch::Approx;

namespace
{
    // A dynamic circle (non-fixedRotation -> a meaningful angular velocity).
    BodyHandle AddDynamicCircle(PhysicsWorld& w, Vec2 pos, Real r)
    {
        BodyDef def;
        def.type     = BodyType::Dynamic;
        def.position = pos;
        def.shape    = MakeCircle(r);
        def.density  = Real(1);
        return w.AddBody(def);
    }

    // A static box (fails the Dynamic predicate -> never synced).
    BodyHandle AddStaticBox(PhysicsWorld& w, Vec2 pos, Real hw, Real hh)
    {
        BodyDef def;
        def.type     = BodyType::Static;
        def.position = pos;
        def.shape    = MakeAabb(hw, hh);
        return w.AddBody(def);
    }
} // namespace

TEST_CASE("PhysicsSimd: BodyStateStore SyncIn/SyncOut round-trips world velocities",
          "[physics]")
{
    // MKS content (MKS P2): this case never calls w.Step(), so gravity is
    // inert here -- use the WorldDef default (0, 10) m/s^2 rather than an
    // explicit zero-g override (no scene reason to declare zero-g).
    WorldDef wd;
    PhysicsWorld w(wd);

    // Two awake dynamic bodies with known velocities + one static body.
    // r = 1 m is already in the MKS happy range (0.1-10 m) -- kept as authored.
    const BodyHandle a = AddDynamicCircle(w, Vec2(Real(0), Real(0)), Real(1));
    const BodyHandle b = AddDynamicCircle(w, Vec2(Real(10), Real(0)), Real(1));
    // Static box half-extents (20,1) m are already sane at MKS scale -- kept as authored.
    const BodyHandle s = AddStaticBox(w, Vec2(Real(0), Real(50)), Real(20), Real(1));

    // Seed known linear + angular velocities. SetVelocity wakes the dynamics
    // (clears the sleep timer), which the awake-dynamic predicate requires.
    const Vec2 vaIn(Real(3), Real(-4));
    const Vec2 vbIn(Real(-7), Real(2));
    const Real waIn = Real(1.5);
    const Real wbIn = Real(-0.25);
    w.SetVelocity(a, vaIn);
    w.SetVelocity(b, vbIn);
    // Angular velocity has no handle-based setter; the public solver-seam slot
    // accessor is the canonical write path (BodyHandle.index IS the SoA slot).
    w.SetAngVelSlot(a.index, waIn);
    w.SetAngVelSlot(b.index, wbIn);

    // Give the static body a velocity too (Static ignores SetVelocity, so this
    // stays 0); we will assert SyncOut never disturbs the static slot.
    REQUIRE(w.VelSlot(s.index) == Vec2(Real(0), Real(0)));

    // ---- SyncIn: world velocities -> packed SoA; dp/dq zeroed --------------
    BodyStateStore soa;
    soa.Resize(w.Count());            // caller sizes to world slot count
    // Pre-poison the position deltas so we can prove SyncIn zeroes them.
    soa[a.index].dpx = 99.f;
    soa[a.index].dpy = 99.f;
    soa[a.index].dq  = 99.f;
    soa.SyncIn(w);

    CHECK(soa[a.index].vx == Approx(vaIn.x));
    CHECK(soa[a.index].vy == Approx(vaIn.y));
    CHECK(soa[a.index].w  == Approx(waIn));
    CHECK(soa[b.index].vx == Approx(vbIn.x));
    CHECK(soa[b.index].vy == Approx(vbIn.y));
    CHECK(soa[b.index].w  == Approx(wbIn));

    // dp/dq for synced slots are zeroed by SyncIn (TGS delta accumulator).
    CHECK(soa[a.index].dpx == Approx(0.0f));
    CHECK(soa[a.index].dpy == Approx(0.0f));
    CHECK(soa[a.index].dq  == Approx(0.0f));

    // ---- mutate the packed velocities, then SyncOut -----------------------
    const Vec2 vaOut(Real(100), Real(-200));
    const Vec2 vbOut(Real(-1), Real(0.5));
    const Real waOut = Real(-9);
    const Real wbOut = Real(42);
    soa[a.index].vx = static_cast<float>(vaOut.x); soa[a.index].vy = static_cast<float>(vaOut.y); soa[a.index].w = static_cast<float>(waOut);
    soa[b.index].vx = static_cast<float>(vbOut.x); soa[b.index].vy = static_cast<float>(vbOut.y); soa[b.index].w = static_cast<float>(wbOut);

    soa.SyncOut(w);

    // The world's velocities now reflect the mutated packed values.
    CHECK(w.VelSlot(a.index).x == Approx(vaOut.x));
    CHECK(w.VelSlot(a.index).y == Approx(vaOut.y));
    CHECK(w.AngVelSlot(a.index) == Approx(waOut));
    CHECK(w.VelSlot(b.index).x == Approx(vbOut.x));
    CHECK(w.VelSlot(b.index).y == Approx(vbOut.y));
    CHECK(w.AngVelSlot(b.index) == Approx(wbOut));

    // ---- invariant: the non-synced static slot is untouched ---------------
    CHECK(w.VelSlot(s.index) == Vec2(Real(0), Real(0)));
    CHECK(w.AngVelSlot(s.index) == Approx(0.0f));
}

// ===========================================================================
// Graph coloring (Task 2) -- the SIMD solve processes constraints 8-wide; two
// constraints in one lane-batch that share a DYNAMIC body would race on the
// read-modify-write of that body's velocity. ColorConstraints partitions edges
// into colors such that no two edges in a color share a dynamic body, so each
// color is safe to solve lane-wide. Static/kinematic endpoints (invMass==0) are
// read-only in the solve, so they may be shared freely within a color.
//
// SCOPE (Task 2): only ColorConstraints + these property tests exist; nothing
// consumes the coloring yet (Task 5 wires it into the solve). These cases pin:
//   (a) within-color dynamic-disjointness,
//   (b) a static body may fan out across one color,
//   (c) determinism (same input -> identical output),
//   (d) star overflow when one dynamic body needs > kColorCount colors,
//   plus the no-drop / no-duplicate ref invariant.
// ===========================================================================

namespace
{
    // Collect every ref emitted across all colors + overflow, so a test can
    // assert nothing was dropped or duplicated.
    std::vector<std::uint32_t> AllRefs(const Coloring& c)
    {
        std::vector<std::uint32_t> out;
        for (const auto& color : c.colors)
        {
            out.insert(out.end(), color.begin(), color.end());
        }
        out.insert(out.end(), c.overflow.begin(), c.overflow.end());
        std::sort(out.begin(), out.end());
        return out;
    }
} // namespace

TEST_CASE("PhysicsSimd: ColorConstraints keeps dynamic bodies disjoint per color",
          "[physics]")
{
    // A chain of dynamic edges 0-1, 1-2, 2-3, 3-4: adjacent edges share a
    // dynamic body, so each must land in a different color than its neighbour.
    std::vector<ColorEdge> edges = {
        { 0u, 1u, true, true, 0u },
        { 1u, 2u, true, true, 1u },
        { 2u, 3u, true, true, 2u },
        { 3u, 4u, true, true, 3u },
    };

    const Coloring c = ColorConstraints(edges, /*bodyCount=*/5u);

    // (a) Within each color, no two edges share a dynamic body. Walk each color
    // and assert its dynamic endpoints are all distinct.
    for (const auto& color : c.colors)
    {
        std::vector<std::uint32_t> dynBodies;
        for (std::uint32_t ref : color)
        {
            const ColorEdge& e = edges[ref];
            if (e.aDyn) dynBodies.push_back(e.a);
            if (e.bDyn) dynBodies.push_back(e.b);
        }
        std::sort(dynBodies.begin(), dynBodies.end());
        const bool hasDup =
            std::adjacent_find(dynBodies.begin(), dynBodies.end()) != dynBodies.end();
        CHECK_FALSE(hasDup);
    }

    // No overflow expected -- a 5-vertex chain needs only 2 colors.
    CHECK(c.overflow.empty());

    // No-drop / no-duplicate: every input ref appears exactly once.
    const std::vector<std::uint32_t> refs = AllRefs(c);
    REQUIRE(refs.size() == edges.size());
    CHECK(refs == std::vector<std::uint32_t>({ 0u, 1u, 2u, 3u }));
}

TEST_CASE("PhysicsSimd: ColorConstraints shares a static body across one color",
          "[physics]")
{
    // (b) A fan of edges that all share ONE static body (body 0, aDyn=false),
    // each with a distinct dynamic partner. The static endpoint does not
    // constrain coloring, so -- since each edge's dynamic body is unique --
    // every edge should land in color 0.
    std::vector<ColorEdge> edges = {
        { 0u, 1u, false, true, 0u },
        { 0u, 2u, false, true, 1u },
        { 0u, 3u, false, true, 2u },
        { 0u, 4u, false, true, 3u },
        { 0u, 5u, false, true, 4u },
    };

    const Coloring c = ColorConstraints(edges, /*bodyCount=*/6u);

    // All five edges land in color 0; later colors stay empty.
    CHECK(c.colors[0] == std::vector<std::uint32_t>({ 0u, 1u, 2u, 3u, 4u }));
    for (std::size_t k = 1; k < c.colors.size(); ++k)
    {
        CHECK(c.colors[k].empty());
    }
    CHECK(c.overflow.empty());

    const std::vector<std::uint32_t> refs = AllRefs(c);
    REQUIRE(refs.size() == edges.size());
}

TEST_CASE("PhysicsSimd: ColorConstraints is deterministic", "[physics]")
{
    // (c) Same input vector -> identical Coloring output. Use a non-trivial mix
    // of dynamic/static endpoints so a non-deterministic implementation would
    // be likely to diverge.
    std::vector<ColorEdge> edges = {
        { 0u, 1u, true,  true,  0u },
        { 1u, 2u, true,  true,  1u },
        { 0u, 2u, true,  true,  2u },   // closes a triangle -> needs color 2
        { 3u, 0u, false, true,  3u },   // static 3 fans onto dynamic 0
        { 4u, 1u, true,  true,  4u },
        { 2u, 4u, true,  true,  5u },
    };

    const Coloring first  = ColorConstraints(edges, /*bodyCount=*/5u);
    const Coloring second = ColorConstraints(edges, /*bodyCount=*/5u);

    REQUIRE(first.colors.size() == second.colors.size());
    for (std::size_t k = 0; k < first.colors.size(); ++k)
    {
        CHECK(first.colors[k] == second.colors[k]);
    }
    CHECK(first.overflow == second.overflow);
}

TEST_CASE("PhysicsSimd: ColorConstraints overflows a star past kColorCount",
          "[physics]")
{
    // (d) A star: one central dynamic body (0) shared by N edges with
    // N > kColorCount. Each edge touches body 0, so each needs its OWN color;
    // only kColorCount colors exist, so the excess edges spill to overflow.
    const std::uint32_t n = static_cast<std::uint32_t>(kColorCount) + 3u;
    std::vector<ColorEdge> edges;
    edges.reserve(n);
    for (std::uint32_t i = 0; i < n; ++i)
    {
        // hub = body 0 (dynamic); each spoke = a distinct dynamic body i+1.
        edges.push_back(ColorEdge{ 0u, i + 1u, true, true, i });
    }

    const Coloring c = ColorConstraints(edges, /*bodyCount=*/n + 1u);

    // The first kColorCount edges each take a distinct color (one per color).
    for (int k = 0; k < kColorCount; ++k)
    {
        REQUIRE(c.colors[k].size() == 1u);
        CHECK(c.colors[k][0] == static_cast<std::uint32_t>(k));
    }

    // The remaining edges (refs kColorCount .. n-1) found no free color.
    REQUIRE(c.overflow.size() == n - static_cast<std::uint32_t>(kColorCount));
    for (std::uint32_t i = 0; i < c.overflow.size(); ++i)
    {
        CHECK(c.overflow[i] == static_cast<std::uint32_t>(kColorCount) + i);
    }

    // No-drop / no-duplicate across the full union.
    const std::vector<std::uint32_t> refs = AllRefs(c);
    REQUIRE(refs.size() == n);
    for (std::uint32_t i = 0; i < n; ++i)
    {
        CHECK(refs[i] == i);
    }
}

// ===========================================================================
// SoA contact-constraint batch (Task 4) -- the lane-wide solve (Task 5) wants
// the per-color ContactConstraints packed into width-lane Structure-of-Arrays
// batches (one CONTACT per lane, up to 2 points per contact). ContactConstraint-
// Simd::Build is the pure packer that fills those batches from a color's refs.
//
// SCOPE (Task 4): only the SoA struct + Build exist; nothing solves it yet
// (Task 5 loads these lane arrays into f32w/i32w/b32w and runs the TGS-Soft
// math). These cases pin the packer contract:
//   (a) every per-lane SoA field mirrors the source ContactConstraint/...Point;
//   (b) a 1-point contact has point-2's pointValid false + its masses zeroed;
//   (c) a partial final batch masks its padding lanes to a lane-wide no-op
//       (invMassA==invMassB==0, pointValid false, impulses 0, bodyIndex 0).
// ===========================================================================

namespace
{
    // Build a fully-populated ContactConstraint with distinct known values keyed
    // off `seed`, so every packed lane field can be checked against a unique
    // number (a transposition bug -> a mismatch). `pts` chooses 1 or 2 points.
    ContactConstraint MakeCC(float seed, int pts, std::uint32_t bA,
                             std::uint32_t bB, bool bIsBody, float invMB)
    {
        ContactConstraint cc{};
        cc.bodyA       = bA;
        cc.bodyB       = bB;
        cc.bodyBIsBody = bIsBody;
        cc.invMassA    = seed + 0.10f;
        cc.invInertiaA = seed + 0.20f;
        cc.invMassB    = invMB;                 // caller controls (dynB depends on it)
        cc.invInertiaB = seed + 0.40f;
        cc.normal      = Vec2(seed + 0.50f, seed + 0.60f);
        cc.friction    = seed + 0.70f;
        cc.restitution = seed + 0.80f;
        cc.biasRate    = seed + 0.90f;
        cc.massScale   = seed + 1.10f;
        cc.impulseScale= seed + 1.20f;
        cc.pointCount  = pts;
        for (int p = 0; p < pts; ++p)
        {
            const float ps = seed + 10.0f * static_cast<float>(p + 1);
            ContactConstraintPoint& cp = cc.points[p];
            cp.anchorA         = Vec2(ps + 0.01f, ps + 0.02f);
            cp.anchorB         = Vec2(ps + 0.03f, ps + 0.04f);
            cp.baseSeparation  = ps + 0.05f;
            cp.normalMass      = ps + 0.06f;
            cp.tangentMass     = ps + 0.07f;
            cp.normalImpulse   = ps + 0.08f;
            cp.tangentImpulse  = ps + 0.09f;
            cp.relativeVelocity= ps + 0.11f;
            cp.id              = static_cast<std::uint32_t>(seed) * 100u
                                 + static_cast<std::uint32_t>(p);
        }
        return cc;
    }

    // Assert lane L of batch `b` mirrors source ContactConstraint `cc` exactly.
    void CheckLaneMatches(const ContactConstraintSimd& b, int L,
                          const ContactConstraint& cc)
    {
        CHECK(b.normalX[L]     == Approx(cc.normal.x));
        CHECK(b.normalY[L]     == Approx(cc.normal.y));
        CHECK(b.friction[L]    == Approx(cc.friction));
        CHECK(b.restitution[L] == Approx(cc.restitution));
        CHECK(b.biasRate[L]    == Approx(cc.biasRate));
        CHECK(b.massScale[L]   == Approx(cc.massScale));
        CHECK(b.impulseScale[L]== Approx(cc.impulseScale));
        CHECK(b.invMassA[L]    == Approx(cc.invMassA));
        CHECK(b.invInertiaA[L] == Approx(cc.invInertiaA));
        CHECK(b.invMassB[L]    == Approx(cc.invMassB));
        CHECK(b.invInertiaB[L] == Approx(cc.invInertiaB));
        CHECK(b.bodyIndexA[L]  == static_cast<std::int32_t>(cc.bodyA));

        // dynB float mask: 1.0f iff B is a dynamic body (bodyBIsBody && invMassB>0).
        const bool dyn = cc.bodyBIsBody && cc.invMassB > 0.0f;
        CHECK(b.dynB[L] == Approx(dyn ? 1.0f : 0.0f));

        // bodyIndexB mirrors cc.bodyB for ANY real body B (bodyBIsBody) -- dynamic
        // OR read-only (kinematic/static): a read-only B's velocity must still be
        // gatherable (a kinematic plate's authored velocity feeds the push). Only a
        // tile span (bodyBIsBody==false, cc.bodyB==kInvalidSlot) packs kNullBodyIndex
        // (-1): the gather injects a zero identity row for it, the scatter skips it.
        CHECK(b.bodyIndexB[L]  == (cc.bodyBIsBody
                                       ? static_cast<std::int32_t>(cc.bodyB)
                                       : kNullBodyIndex));

        for (int p = 0; p < 2; ++p)
        {
            const ContactConstraintSimd::Point& pt = b.points[p];
            if (p < cc.pointCount)
            {
                const ContactConstraintPoint& cp = cc.points[p];
                CHECK(pt.anchorAx[L]      == Approx(cp.anchorA.x));
                CHECK(pt.anchorAy[L]      == Approx(cp.anchorA.y));
                CHECK(pt.anchorBx[L]      == Approx(cp.anchorB.x));
                CHECK(pt.anchorBy[L]      == Approx(cp.anchorB.y));
                CHECK(pt.baseSep[L]       == Approx(cp.baseSeparation));
                CHECK(pt.normalMass[L]    == Approx(cp.normalMass));
                CHECK(pt.tangentMass[L]   == Approx(cp.tangentMass));
                CHECK(pt.normalImpulse[L] == Approx(cp.normalImpulse));
                CHECK(pt.tangentImpulse[L]== Approx(cp.tangentImpulse));
                CHECK(pt.relVel[L]        == Approx(cp.relativeVelocity));
                CHECK(pt.pointValid[L]    == Approx(1.0f));
            }
            else
            {
                // (b) point >= pointCount: invalid lane, masses + impulses zeroed.
                CHECK(pt.pointValid[L]    == Approx(0.0f));
                CHECK(pt.normalMass[L]    == Approx(0.0f));
                CHECK(pt.tangentMass[L]   == Approx(0.0f));
                CHECK(pt.normalImpulse[L] == Approx(0.0f));
                CHECK(pt.tangentImpulse[L]== Approx(0.0f));
            }
        }
    }
} // namespace

TEST_CASE("PhysicsSimd: ContactConstraintSimd::Build packs lane fields verbatim",
          "[physics]")
{
    constexpr int W = ContactConstraintSimd::kWidth;

    // One full batch worth of distinct 2-point contacts, all dynamic-vs-dynamic
    // so dynB == 1 on every lane.
    std::vector<ContactConstraint> ccs;
    std::vector<std::uint32_t>     refs;
    for (int i = 0; i < W; ++i)
    {
        ccs.push_back(MakeCC(/*seed=*/static_cast<float>(i + 1), /*pts=*/2,
                             /*bA=*/static_cast<std::uint32_t>(i * 2 + 0),
                             /*bB=*/static_cast<std::uint32_t>(i * 2 + 1),
                             /*bIsBody=*/true, /*invMB=*/0.5f + static_cast<float>(i)));
        refs.push_back(static_cast<std::uint32_t>(i));
    }

    const std::vector<ContactConstraintSimd> batches =
        ContactConstraintSimd::Build(ccs.data(), refs.data(),
                                     static_cast<int>(refs.size()));

    REQUIRE(batches.size() == 1u);
    CHECK(batches[0].count == W);

    // (a) every lane mirrors its source ContactConstraint verbatim.
    for (int L = 0; L < W; ++L)
    {
        CheckLaneMatches(batches[0], L, ccs[refs[L]]);
    }
}

TEST_CASE("PhysicsSimd: ContactConstraintSimd::Build handles 1-point + static-B "
          "contacts", "[physics]")
{
    // Lane 0: a 1-point contact against a STATIC fixture (bodyBIsBody=false ->
    // dynB must be 0 even though we never set invMassB). Lane 1: a 2-point
    // contact against a body with invMassB==0 (kinematic -> dynB 0 too).
    std::vector<ContactConstraint> ccs;
    ccs.push_back(MakeCC(/*seed=*/3.0f, /*pts=*/1, /*bA=*/5u, /*bB=*/9u,
                         /*bIsBody=*/false, /*invMB=*/0.0f));
    ccs.push_back(MakeCC(/*seed=*/7.0f, /*pts=*/2, /*bA=*/2u, /*bB=*/4u,
                         /*bIsBody=*/true,  /*invMB=*/0.0f));   // kinematic B
    std::vector<std::uint32_t> refs = { 0u, 1u };

    const std::vector<ContactConstraintSimd> batches =
        ContactConstraintSimd::Build(ccs.data(), refs.data(), 2);

    REQUIRE(batches.size() == 1u);
    CHECK(batches[0].count == 2);

    // Lane 0: 1-point contact against a tile SPAN (bodyBIsBody=false, bodyB=9 but
    // NOT a real body). B is read-only (dynB 0) AND not a real body, so Build packs
    // the null index (-1: zero-identity gather, no scatter). 2nd point absent.
    CheckLaneMatches(batches[0], 0, ccs[0]);
    CHECK(batches[0].dynB[0] == Approx(0.0f));               // span fixture
    CHECK(batches[0].bodyIndexB[0] == kNullBodyIndex);       // span -> null index
    CHECK(batches[0].points[1].pointValid[0] == Approx(0.0f));  // 2nd point absent

    // Lane 1: 2-point contact, B a REAL kinematic body (bodyBIsBody=true,
    // invMassB==0) -> dynB 0 (not mutated) but the index is its REAL slot (4) so
    // the solver can gather the kinematic's velocity (the push driver).
    CheckLaneMatches(batches[0], 1, ccs[1]);
    CHECK(batches[0].dynB[1] == Approx(0.0f));
    CHECK(batches[0].bodyIndexB[1] == 4);                // real kinematic slot, gatherable
}

TEST_CASE("PhysicsSimd: ContactConstraintSimd::Build packs a tile-span "
          "(kInvalidSlot) bodyIndexB as the null index", "[physics]")
{
    // A tile-span virtual fixture contact: B is NOT a body (bodyBIsBody=false)
    // and cc.bodyB == kInvalidSlot == 0xFFFFFFFF. Build MUST pack the packed
    // bodyIndexB as kNullBodyIndex (-1): the lane-wide gather then injects a shared
    // zero IDENTITY row for the -1 lane (no real-body memory touch -- NOT a
    // states[-1] under-read), and the scatter skips it. dynB==0 (read-only). This
    // pins the contract the finite-slot static-B case above does not exercise.
    ContactConstraint span = MakeCC(/*seed=*/5.0f, /*pts=*/1, /*bA=*/2u,
                                    /*bB=*/kInvalidSlot, /*bIsBody=*/false,
                                    /*invMB=*/0.0f);
    span.invInertiaB = 0.0f;   // a span has no rotational inertia either

    std::vector<ContactConstraint> ccs = { span };
    std::vector<std::uint32_t>     refs = { 0u };

    const std::vector<ContactConstraintSimd> batches =
        ContactConstraintSimd::Build(ccs.data(), refs.data(), 1);

    REQUIRE(batches.size() == 1u);
    CHECK(batches[0].count == 1);

    // The crux: B is read-only -> dynB 0 AND bodyIndexB == kNullBodyIndex (-1).
    CHECK(batches[0].dynB[0]       == Approx(0.0f));
    CHECK(batches[0].bodyIndexB[0] == kNullBodyIndex);

    // Sanity: bodyIndexA is the real (dynamic) A slot, never the null index.
    CHECK(batches[0].bodyIndexA[0] == 2);

    // The rest of the lane still mirrors the source (CheckLaneMatches now expects
    // the null index for any span B).
    CheckLaneMatches(batches[0], 0, span);
}

TEST_CASE("PhysicsSimd: ContactConstraintSimd::Build masks partial-batch padding "
          "lanes", "[physics]")
{
    constexpr int W = ContactConstraintSimd::kWidth;

    // count = W + 1 -> two batches; the SECOND batch has exactly one live lane
    // and (W - 1) padding lanes (>=0; if W==1 there is no partial tail, so the
    // test still passes trivially with the count==1 live lane).
    const int count = W + 1;
    std::vector<ContactConstraint> ccs;
    std::vector<std::uint32_t>     refs;
    for (int i = 0; i < count; ++i)
    {
        ccs.push_back(MakeCC(static_cast<float>(i + 1), /*pts=*/2,
                             static_cast<std::uint32_t>(i * 2 + 0),
                             static_cast<std::uint32_t>(i * 2 + 1),
                             /*bIsBody=*/true, /*invMB=*/1.0f));
        refs.push_back(static_cast<std::uint32_t>(i));
    }

    const std::vector<ContactConstraintSimd> batches =
        ContactConstraintSimd::Build(ccs.data(), refs.data(), count);

    const std::size_t expectedBatches =
        static_cast<std::size_t>((count + W - 1) / W);
    REQUIRE(batches.size() == expectedBatches);

    const ContactConstraintSimd& tail = batches.back();
    const int live = count - static_cast<int>((batches.size() - 1) * W);
    CHECK(tail.count == live);

    // Live lanes of the tail batch mirror their source.
    for (int L = 0; L < live; ++L)
    {
        const std::uint32_t ref = refs[(batches.size() - 1) * W + L];
        CheckLaneMatches(tail, L, ccs[ref]);
    }

    // (c) padding lanes [live, W) are a lane-wide no-op: zero inv-mass on BOTH
    // bodies, both points invalid + impulses zero, both body indices the null
    // index (-1: zero-identity gather, scatter skipped).
    for (int L = live; L < W; ++L)
    {
        CHECK(tail.invMassA[L]    == Approx(0.0f));
        CHECK(tail.invInertiaA[L] == Approx(0.0f));
        CHECK(tail.invMassB[L]    == Approx(0.0f));
        CHECK(tail.invInertiaB[L] == Approx(0.0f));
        CHECK(tail.bodyIndexA[L]  == kNullBodyIndex);
        CHECK(tail.bodyIndexB[L]  == kNullBodyIndex);
        CHECK(tail.dynB[L]        == Approx(0.0f));
        for (int p = 0; p < 2; ++p)
        {
            CHECK(tail.points[p].pointValid[L]     == Approx(0.0f));
            CHECK(tail.points[p].normalImpulse[L]  == Approx(0.0f));
            CHECK(tail.points[p].tangentImpulse[L] == Approx(0.0f));
            CHECK(tail.points[p].normalMass[L]     == Approx(0.0f));
            CHECK(tail.points[p].tangentMass[L]    == Approx(0.0f));
        }
    }
}

// ===========================================================================
// Lane-wide contact solve (Task 5) -- the THREE solve-correctness gates:
//
//   * LANE-WIDTH INVARIANCE: within a color all contacts touch DISJOINT dynamic
//     bodies, so the solve is independent per lane -> the result must NOT depend
//     on the SIMD packing width. We solve the SAME independent contact set two
//     ways through the SAME (active-backend) SimdSolve passes -- packed wide (one
//     batch of W lanes) vs packed NARROW (W batches of 1 live lane + padding) --
//     and assert BIT-IDENTICAL body velocities. If a lane leaked into another, or
//     a padding lane were not a true no-op, the two packings would diverge.
//
//   * SCALAR-ORACLE MATCH: the lane-wide AVX2 result must match a width-1 SCALAR
//     reference (plain `float` math transcribed straight from SoftStep.cpp) within
//     1e-5. This is the SIMD-vs-scalar oracle: the scalar reference IS the ported
//     math, computed without the Simd wrapper, so a transcription error in the
//     lane-wide passes shows up here.
//
//   * SCATTER-CORRUPTION GUARD: a color-batch where body slot 0 is a real DYNAMIC
//     body sharing the batch with PADDING lanes (and a read-only-B lane) must NOT
//     let those lanes clobber body 0's solved velocity. Padding lanes pack the
//     null index (-1) and are SKIPPED by the scatter; a read-only B (dynB=0) is
//     never scattered. We assert body 0's velocity equals the scalar-oracle value
//     (a corruption would zero/stale it).
//
// All three drive the header-only SimdSolve passes directly (no PhysicsWorld), so
// they pin the ported math in isolation. Tag [physics][simd].
// ===========================================================================

namespace
{
    // 2D cross helpers (mirror SoftStep.cpp's anonymous-namespace helpers).
    inline Vec2 XCrossWR(Real wv, const Vec2& r) { return Vec2(-wv * r.y, wv * r.x); }
    inline Real XCrossRP(const Vec2& r, const Vec2& p) { return r.x * p.y - r.y * p.x; }
    inline Real XDot(const Vec2& a, const Vec2& b) { return a.x * b.x + a.y * b.y; }

    // A self-contained WIDTH-1 SCALAR reference for ONE contact, reading/writing a
    // BodyStateStore -- a verbatim transcription of SoftStep.cpp's WarmStart /
    // SolveContacts / ApplyRestitution math (plain float, no Simd wrapper). This is
    // the oracle the lane-wide passes are compared against. A is always dynamic;
    // dynB gates the B-side write. All contacts here use 1 point (pointCount==1).
    struct ScalarRef
    {
        BodyStateStore& bs;
        Real h, invH, maxBiasVel, threshold;

        void WarmStart(ContactConstraint& cc)
        {
            const Vec2 n = cc.normal;
            const Vec2 tangent(-n.y, n.x);
            const bool dynB = cc.bodyBIsBody && cc.invMassB > Real(0);
            Vec2 vA(bs[cc.bodyA].vx, bs[cc.bodyA].vy); Real wA = bs[cc.bodyA].w;
            Vec2 vB(bs[cc.bodyB].vx, bs[cc.bodyB].vy); Real wB = bs[cc.bodyB].w;
            for (int p = 0; p < cc.pointCount; ++p)
            {
                const ContactConstraintPoint& cp = cc.points[p];
                const Vec2 P = n * cp.normalImpulse + tangent * cp.tangentImpulse;
                vA += P * cc.invMassA; wA += cc.invInertiaA * XCrossRP(cp.anchorA, P);
                if (dynB) { vB -= P * cc.invMassB; wB -= cc.invInertiaB * XCrossRP(cp.anchorB, P); }
            }
            bs[cc.bodyA].vx = static_cast<float>(vA.x); bs[cc.bodyA].vy = static_cast<float>(vA.y); bs[cc.bodyA].w = static_cast<float>(wA);
            if (dynB) { bs[cc.bodyB].vx = static_cast<float>(vB.x); bs[cc.bodyB].vy = static_cast<float>(vB.y); bs[cc.bodyB].w = static_cast<float>(wB); }
        }

        void Solve(ContactConstraint& cc, bool useBias)
        {
            const Vec2 n = cc.normal;
            const Vec2 tangent(-n.y, n.x);
            const bool dynB = cc.bodyBIsBody && cc.invMassB > Real(0);
            const Real iMa = cc.invMassA, iIa = cc.invInertiaA, iMb = cc.invMassB, iIb = cc.invInertiaB;
            Vec2 vA(bs[cc.bodyA].vx, bs[cc.bodyA].vy); Real wA = bs[cc.bodyA].w;
            Vec2 vB(bs[cc.bodyB].vx, bs[cc.bodyB].vy); Real wB = bs[cc.bodyB].w;
            const Vec2 dpA(bs[cc.bodyA].dpx, bs[cc.bodyA].dpy); const Real drA = bs[cc.bodyA].dq;
            const Vec2 dpB(bs[cc.bodyB].dpx, bs[cc.bodyB].dpy); const Real drB = bs[cc.bodyB].dq;
            for (int p = 0; p < cc.pointCount; ++p)
            {
                ContactConstraintPoint& cp = cc.points[p];
                const Vec2 rA = cp.anchorA, rB = cp.anchorB;
                const Vec2 prA = dpA + XCrossWR(drA, rA), prB = dpB + XCrossWR(drB, rB);
                const Real s = cp.baseSeparation + XDot(prA - prB, n);
                Real bias = Real(0), massScale = Real(1), impulseScale = Real(0);
                if (s > Real(0)) { bias = s * invH; }
                else if (useBias) { bias = std::max(cc.biasRate * s, -maxBiasVel); massScale = cc.massScale; impulseScale = cc.impulseScale; }
                const Vec2 dv = (vA + XCrossWR(wA, rA)) - (vB + XCrossWR(wB, rB));
                const Real vn = XDot(dv, n);
                Real impulse = -cp.normalMass * massScale * (vn + bias) - impulseScale * cp.normalImpulse;
                const Real newI = std::max(cp.normalImpulse + impulse, Real(0));
                impulse = newI - cp.normalImpulse; cp.normalImpulse = newI;
                const Vec2 P = n * impulse;
                vA += P * iMa; wA += iIa * XCrossRP(rA, P);
                if (dynB) { vB -= P * iMb; wB -= iIb * XCrossRP(rB, P); }
            }
            for (int p = 0; p < cc.pointCount; ++p)
            {
                ContactConstraintPoint& cp = cc.points[p];
                const Vec2 rA = cp.anchorA, rB = cp.anchorB;
                const Vec2 dv = (vA + XCrossWR(wA, rA)) - (vB + XCrossWR(wB, rB));
                const Real vt = XDot(dv, tangent);
                Real impulse = -cp.tangentMass * vt;
                const Real maxF = cc.friction * cp.normalImpulse;
                const Real newI = std::clamp(cp.tangentImpulse + impulse, -maxF, maxF);
                impulse = newI - cp.tangentImpulse; cp.tangentImpulse = newI;
                const Vec2 P = tangent * impulse;
                vA += P * iMa; wA += iIa * XCrossRP(rA, P);
                if (dynB) { vB -= P * iMb; wB -= iIb * XCrossRP(rB, P); }
            }
            bs[cc.bodyA].vx = static_cast<float>(vA.x); bs[cc.bodyA].vy = static_cast<float>(vA.y); bs[cc.bodyA].w = static_cast<float>(wA);
            if (dynB) { bs[cc.bodyB].vx = static_cast<float>(vB.x); bs[cc.bodyB].vy = static_cast<float>(vB.y); bs[cc.bodyB].w = static_cast<float>(wB); }
        }

        void Restitution(ContactConstraint& cc)
        {
            if (cc.restitution <= Real(0)) { return; }
            const Vec2 n = cc.normal;
            const bool dynB = cc.bodyBIsBody && cc.invMassB > Real(0);
            Vec2 vA(bs[cc.bodyA].vx, bs[cc.bodyA].vy); Real wA = bs[cc.bodyA].w;
            Vec2 vB(bs[cc.bodyB].vx, bs[cc.bodyB].vy); Real wB = bs[cc.bodyB].w;
            for (int p = 0; p < cc.pointCount; ++p)
            {
                ContactConstraintPoint& cp = cc.points[p];
                if (cp.relativeVelocity > -threshold || cp.normalImpulse <= Real(0)) { continue; }
                const Vec2 rA = cp.anchorA, rB = cp.anchorB;
                const Vec2 dv = (vA + XCrossWR(wA, rA)) - (vB + XCrossWR(wB, rB));
                const Real vn = XDot(dv, n);
                Real impulse = -cp.normalMass * (vn + cc.restitution * cp.relativeVelocity);
                const Real newI = std::max(cp.normalImpulse + impulse, Real(0));
                impulse = newI - cp.normalImpulse; cp.normalImpulse = newI;
                const Vec2 P = n * impulse;
                vA += P * cc.invMassA; wA += cc.invInertiaA * XCrossRP(rA, P);
                if (dynB) { vB -= P * cc.invMassB; wB -= cc.invInertiaB * XCrossRP(rB, P); }
            }
            bs[cc.bodyA].vx = static_cast<float>(vA.x); bs[cc.bodyA].vy = static_cast<float>(vA.y); bs[cc.bodyA].w = static_cast<float>(wA);
            if (dynB) { bs[cc.bodyB].vx = static_cast<float>(vB.x); bs[cc.bodyB].vy = static_cast<float>(vB.y); bs[cc.bodyB].w = static_cast<float>(wB); }
        }
    };

    // Build a PREPARED 1-point contact between dynamic A (slot bA) and dynamic
    // B (slot bB), with effective masses already filled (as PrepareContacts would).
    // Geometry is a simple head-on normal contact with a slight penetration.
    ContactConstraint MakePreparedContact(std::uint32_t bA, std::uint32_t bB, float seed)
    {
        ContactConstraint cc{};
        cc.bodyA = bA; cc.bodyB = bB; cc.bodyBIsBody = true;
        cc.invMassA = Real(0.5); cc.invInertiaA = Real(0.2);
        cc.invMassB = Real(0.5); cc.invInertiaB = Real(0.2);
        cc.normal = Vec2(Real(0), Real(-1));            // B->A push (A above B)
        cc.friction = Real(0.4); cc.restitution = Real(0.2);
        cc.biasRate = Real(-2.0); cc.massScale = Real(0.7); cc.impulseScale = Real(0.1);
        cc.pointCount = 1;
        ContactConstraintPoint& cp = cc.points[0];
        cp.anchorA = Vec2(Real(0.1) * seed, Real(-1));
        cp.anchorB = Vec2(Real(0.1) * seed, Real(1));
        cp.baseSeparation = Real(-0.05);                // small penetration
        cp.normalMass = Real(1.0); cp.tangentMass = Real(1.0);
        cp.normalImpulse = Real(0.3) * seed;            // a warm-start seed
        cp.tangentImpulse = Real(0.05) * seed;
        cp.relativeVelocity = Real(-30) - seed;         // approaching (restitution-eligible)
        cp.id = static_cast<std::uint32_t>(seed) + 1u;
        return cc;
    }

    // Run the full substep solve on `batches` over `bs` (lane-wide path).
    void RunLaneWide(std::vector<ContactConstraintSimd>& batches, BodyStateStore& bs,
                     int substeps, float h, float maxBiasVel, float threshold,
                     const std::vector<std::uint32_t>& dynSlots)
    {
        // No integrate-velocities here -- we isolate the SOLVE math (gravity is
        // covered by the [physics] world-level invariants). dp/dq accumulate
        // between the bias + relax passes (the TGS separation re-eval).
        for (int s = 0; s < substeps; ++s)
        {
            SimdSolve::WarmStart(batches, bs.data());
            SimdSolve::SolveNormalAndFriction(batches, bs.data(), h, /*useBias=*/true, maxBiasVel);
            for (std::uint32_t i : dynSlots) { bs[i].dpx += bs[i].vx * h; bs[i].dpy += bs[i].vy * h; bs[i].dq += bs[i].w * h; }
            SimdSolve::SolveNormalAndFriction(batches, bs.data(), h, /*useBias=*/false, maxBiasVel);
        }
        SimdSolve::ApplyRestitution(batches, bs.data(), threshold);
    }
} // namespace

TEST_CASE("PhysicsSimd: lane-wide solve is packing-width invariant + matches scalar",
          "[physics][simd]")
{
    constexpr int W = ContactConstraintSimd::kWidth;
    // N independent contacts: each between a unique pair of dynamic bodies
    // (2*N dynamic slots, all disjoint) so coloring puts them ALL in one color.
    const int N = W; // exactly one full wide batch
    const std::uint32_t bodyCount = static_cast<std::uint32_t>(2 * N);

    std::vector<ContactConstraint> ccs;
    std::vector<std::uint32_t> refs, dynSlots;
    for (int i = 0; i < N; ++i)
    {
        const std::uint32_t bA = static_cast<std::uint32_t>(2 * i);
        const std::uint32_t bB = static_cast<std::uint32_t>(2 * i + 1);
        ccs.push_back(MakePreparedContact(bA, bB, static_cast<float>(i + 1)));
        refs.push_back(static_cast<std::uint32_t>(i));
        dynSlots.push_back(bA); dynSlots.push_back(bB);
    }

    // Seed a BodyStateStore (no dummy slot -- null-index handles read-only/padding).
    auto seedSoA = [&](BodyStateStore& bs) {
        bs.Resize(bodyCount);
        for (std::uint32_t b = 0; b < bodyCount; ++b)
        {
            bs[b].vx = 0.5f * static_cast<float>(b) - 1.0f;
            bs[b].vy = (b % 2 == 0) ? 40.0f : -10.0f;   // A's fall onto B's
            bs[b].w  = 0.01f * static_cast<float>(b);
        }
    };

    const int substeps = 4;
    const float h = (1.0f / 60.0f) / static_cast<float>(substeps);
    const float maxBiasVel = 4.0f;
    const float threshold = 1.0f;

    // ---- Path A: lane-wide, ONE wide batch (W lanes) -----------------------
    BodyStateStore bsWide; seedSoA(bsWide);
    {
        std::vector<ContactConstraint> wccs = ccs;   // local copy (impulses mutate)
        std::vector<ContactConstraintSimd> batches =
            ContactConstraintSimd::Build(wccs.data(), refs.data(), N);
        REQUIRE(batches.size() == 1u);
        RunLaneWide(batches, bsWide, substeps, h, maxBiasVel, threshold, dynSlots);
    }

    // ---- Path B: lane-wide, NARROW -- N batches of 1 live lane + padding ----
    // Same SimdSolve passes, different packing width per batch. Lane-width
    // invariance => bit-identical to Path A.
    BodyStateStore bsNarrow; seedSoA(bsNarrow);
    {
        std::vector<ContactConstraint> nccs = ccs;
        std::vector<ContactConstraintSimd> batches;
        for (int i = 0; i < N; ++i)
        {
            std::uint32_t one = static_cast<std::uint32_t>(i);
            std::vector<ContactConstraintSimd> b1 =
                ContactConstraintSimd::Build(nccs.data(), &one, 1);
            REQUIRE(b1.size() == 1u);
            batches.push_back(b1[0]);
        }
        RunLaneWide(batches, bsNarrow, substeps, h, maxBiasVel, threshold, dynSlots);
    }

    // BIT-IDENTICAL: packing width does not change the result (the core invariant).
    for (std::uint32_t b = 0; b < bodyCount; ++b)
    {
        CHECK(bsWide[b].vx == bsNarrow[b].vx);
        CHECK(bsWide[b].vy == bsNarrow[b].vy);
        CHECK(bsWide[b].w  == bsNarrow[b].w);
    }

    // ---- Path C: width-1 SCALAR oracle (plain float, no Simd wrapper) -------
    BodyStateStore bsScalar; seedSoA(bsScalar);
    {
        std::vector<ContactConstraint> sccs = ccs;
        ScalarRef ref{ bsScalar, Real(h), Real(1.0 / h), Real(maxBiasVel), Real(threshold) };
        for (int s = 0; s < substeps; ++s)
        {
            for (auto& cc : sccs) { ref.WarmStart(cc); }
            for (auto& cc : sccs) { ref.Solve(cc, /*useBias=*/true); }
            for (std::uint32_t i : dynSlots) { bsScalar[i].dpx += bsScalar[i].vx * h; bsScalar[i].dpy += bsScalar[i].vy * h; bsScalar[i].dq += bsScalar[i].w * h; }
            for (auto& cc : sccs) { ref.Solve(cc, /*useBias=*/false); }
        }
        for (auto& cc : sccs) { ref.Restitution(cc); }
    }

    // SCALAR-ORACLE MATCH within 1e-5: the lane-wide AVX2 math equals the plain-
    // float reference (a transcription error would show here). Per-op float
    // rounding differs only at the ~1e-6 ulp scale for these magnitudes.
    for (std::uint32_t b = 0; b < bodyCount; ++b)
    {
        CHECK(bsWide[b].vx == Approx(bsScalar[b].vx).margin(1e-5));
        CHECK(bsWide[b].vy == Approx(bsScalar[b].vy).margin(1e-5));
        CHECK(bsWide[b].w  == Approx(bsScalar[b].w).margin(1e-5));
    }
}

TEST_CASE("PhysicsSimd: padding + read-only-B lanes cannot corrupt body slot 0",
          "[physics][simd]")
{
    // The SCATTER-CORRUPTION guard. Build ONE batch where:
    //   lane 0 = a real DYNAMIC contact whose A is BODY SLOT 0 (the slot a naive
    //            "clamp read-only/padding index to 0" would clobber),
    //   lane 1 = a read-only-B contact (B kinematic) whose A is a different dynamic,
    //   lanes 2..W-1 = PADDING.
    // Padding lanes pack the null index (-1) and are skipped by the scatter; the
    // read-only B (dynB=0) is never scattered. So neither can touch slot 0. We
    // solve the batch and assert body 0's velocity equals the scalar-oracle value
    // (a corruption would zero/stale it).
    constexpr int W = ContactConstraintSimd::kWidth;
    if (W < 2) { SUCCEED("scalar backend: single lane, no padding to corrupt"); return; }

    // Bodies: slot 0 (dyn A of lane0), slot 1 (dyn B of lane0), slot 2 (dyn A of
    // lane1), slot 3 (kinematic B of lane1, invMassB=0). No dummy slot anymore.
    const std::uint32_t bodyCount = 4u;

    std::vector<ContactConstraint> ccs;
    ccs.push_back(MakePreparedContact(0u, 1u, 1.0f));            // lane 0: dyn-dyn, A = slot 0
    ContactConstraint kin = MakePreparedContact(2u, 3u, 2.0f);  // lane 1: A=slot2, B=slot3
    kin.invMassB = Real(0); kin.invInertiaB = Real(0);          // make B kinematic (read-only)
    ccs.push_back(kin);
    std::vector<std::uint32_t> refs = { 0u, 1u };
    std::vector<std::uint32_t> dynSlots = { 0u, 1u, 2u };       // slot 3 is kinematic (read-only)

    auto seedSoA = [&](BodyStateStore& bs) {
        bs.Resize(bodyCount);
        bs[0].vx = 3.0f;  bs[0].vy = 50.0f; bs[0].w = 0.1f;     // body 0: a distinctive velocity
        bs[1].vx = -1.0f; bs[1].vy = -5.0f; bs[1].w = 0.0f;
        bs[2].vx = 2.0f;  bs[2].vy = 30.0f; bs[2].w = 0.05f;
        bs[3].vx = 7.0f;  bs[3].vy = 0.0f;  bs[3].w = 0.0f;     // kinematic plate moving +x
    };

    const int substeps = 4;
    const float h = (1.0f / 60.0f) / static_cast<float>(substeps);
    const float maxBiasVel = 4.0f, threshold = 1.0f;

    // Lane-wide path (the batch has W lanes: 2 live + W-2 padding).
    BodyStateStore bsWide; seedSoA(bsWide);
    {
        std::vector<ContactConstraint> wccs = ccs;
        std::vector<ContactConstraintSimd> batches =
            ContactConstraintSimd::Build(wccs.data(), refs.data(), 2);
        REQUIRE(batches.size() == 1u);
        REQUIRE(batches[0].count == 2);     // 2 live lanes, the rest padding
        RunLaneWide(batches, bsWide, substeps, h, maxBiasVel, threshold, dynSlots);
    }

    // Scalar oracle (no lanes, no padding -> cannot corrupt by construction).
    BodyStateStore bsScalar; seedSoA(bsScalar);
    {
        std::vector<ContactConstraint> sccs = ccs;
        ScalarRef ref{ bsScalar, Real(h), Real(1.0 / h), Real(maxBiasVel), Real(threshold) };
        for (int s = 0; s < substeps; ++s)
        {
            for (auto& cc : sccs) { ref.WarmStart(cc); }
            for (auto& cc : sccs) { ref.Solve(cc, true); }
            for (std::uint32_t i : dynSlots) { bsScalar[i].dpx += bsScalar[i].vx * h; bsScalar[i].dpy += bsScalar[i].vy * h; bsScalar[i].dq += bsScalar[i].w * h; }
            for (auto& cc : sccs) { ref.Solve(cc, false); }
        }
        for (auto& cc : sccs) { ref.Restitution(cc); }
    }

    // Body 0 (slot 0) is NOT corrupted: its solved velocity matches the oracle.
    // A padding/read-only lane clobbering slot 0 would make these diverge wildly.
    CHECK(bsWide[0].vx == Approx(bsScalar[0].vx).margin(1e-5));
    CHECK(bsWide[0].vy == Approx(bsScalar[0].vy).margin(1e-5));
    CHECK(bsWide[0].w  == Approx(bsScalar[0].w).margin(1e-5));
    // And it genuinely moved (the contact solve changed it -> a real, not trivial,
    // velocity that corruption could destroy).
    CHECK(bsWide[0].vy != Approx(50.0f));

    // The kinematic plate (slot 3, read-only B) was NOT mutated by the solve
    // (its velocity is preserved through the read-only-B write-back).
    CHECK(bsWide[3].vx == Approx(7.0f));
    CHECK(bsWide[3].vy == Approx(0.0f));
    CHECK(bsWide[3].w  == Approx(0.0f));
}

// ===========================================================================
// Null-index branch (Task 3, Gap 2.2) -- a read-only B packed as kNullBodyIndex
// (-1) injects a shared ZERO identity row in the gather (no real-body memory
// touch) instead of reading a dummy slot. This must be EQUIVALENT to a contact
// against a real, read-only (invMassB==0), zero-velocity body: both feed vB==0
// into A's solve and never scatter B, so body A's post-solve state must be
// BIT-IDENTICAL across the two. This is the contract that lets the solver drop
// the scatter-safe dummy tail (Resize(solverCount), no +1) without changing any
// float. Tag [physics][simd].
// ===========================================================================
TEST_CASE("PhysicsSimd: null-index B equals a zero-velocity read-only body",
          "[physics][simd]")
{
    const int substeps = 4;
    const float h = (1.0f / 60.0f) / static_cast<float>(substeps);
    const float maxBiasVel = 4.0f, threshold = 1.0f;

    // Solve ONE contact (dynamic A = slot 0 vs a read-only B) and return A's row.
    //   nullB == true : B is a tile span (bodyBIsBody=false) -> packed bodyIndexB
    //                   == kNullBodyIndex (-1) -> gathered as the zero identity.
    //   nullB == false: B is a REAL body (slot 1), invMassB==0 (read-only),
    //                   seeded at rest -> packed bodyIndexB == 1, gathered zero.
    auto runA = [&](bool nullB) -> BodyState
    {
        ContactConstraint cc = MakePreparedContact(0u, 1u, 1.0f);
        cc.invMassB = Real(0); cc.invInertiaB = Real(0);   // B is read-only in both
        if (nullB) { cc.bodyBIsBody = false; cc.bodyB = kInvalidSlot; }

        std::vector<ContactConstraint> ccs = { cc };
        std::uint32_t ref = 0u;
        std::vector<ContactConstraintSimd> batches =
            ContactConstraintSimd::Build(ccs.data(), &ref, 1);
        REQUIRE(batches.size() == 1u);

        // Pin the sentinel wiring: a span packs -1; a real read-only body keeps
        // its real slot (its zero row is gathered, never scattered).
        if (nullB) { CHECK(batches[0].bodyIndexB[0] == kNullBodyIndex); }
        else       { CHECK(batches[0].bodyIndexB[0] == 1); }
        CHECK(batches[0].dynB[0] == Approx(0.0f));         // read-only either way

        BodyStateStore bs; bs.Resize(2u);                  // NO +1 dummy tail
        bs[0].vx = 3.0f; bs[0].vy = 40.0f; bs[0].w = 0.1f; // body A start
        bs[1].vx = 0.0f; bs[1].vy = 0.0f;  bs[1].w = 0.0f; // body B at rest (REAL path)
        std::vector<std::uint32_t> dynSlots = { 0u };      // only A integrates dp/dq
        RunLaneWide(batches, bs, substeps, h, maxBiasVel, threshold, dynSlots);
        return bs[0];
    };

    const BodyState nb = runA(true);
    const BodyState rb = runA(false);

    // BIT-IDENTICAL: the -1 identity injection equals a real zero-velocity B.
    CHECK(nb.vx == rb.vx);
    CHECK(nb.vy == rb.vy);
    CHECK(nb.w  == rb.w);
    // And A genuinely solved (it moved off its initial fall velocity), so the
    // match is a real result, not a trivial no-op.
    CHECK(nb.vy != Approx(40.0f));
}

// ===========================================================================
// Overflow (un-colorable) contacts settle + stay bounded.
//
// Coloring spills a constraint when one DYNAMIC endpoint already occupies all
// kColorCount colors (a hub dynamic body sharing a dynamic endpoint with >
// kColorCount contacts). Those overflow refs are solved SEQUENTIALLY, scalar,
// over the same BodyStateStore -- they must NOT be dropped (or the hub would sink
// /explode). We build a central dynamic disk surrounded by a dense ring of many
// dynamic disks all overlapping it (so the hub is a dynamic endpoint in > 12
// dynamic-dynamic contacts -> guaranteed overflow), drop it onto a floor under
// gravity, and assert the whole cluster settles to a bounded rest (energy bled
// off, nothing flung). Tag [physics][simd].
// ===========================================================================
TEST_CASE("PhysicsSimd: overflow (un-colorable hub) contacts settle + bounded",
          "[physics][simd]")
{
    // MKS content (MKS P2): gravity uses the WorldDef default (10 m/s^2, +Y down).
    WorldDef wd;
    PhysicsWorld w(wd);

    // Static floor.
    const Real floorTop = Real(3);
    {
        BodyDef bd;
        bd.type     = BodyType::Static;
        bd.position = Vec2(Real(0), floorTop + Real(0.5));
        bd.shape    = MakeAabb(Real(20), Real(0.5));
        w.AddBody(bd);
    }

    // Central HUB dynamic disk resting just above the floor.
    BodyHandle hub;
    const Real hubR = Real(1.2);
    {
        BodyDef bd;
        bd.type        = BodyType::Dynamic;
        bd.position    = Vec2(Real(0), floorTop - hubR);
        bd.shape       = MakeCircle(hubR);
        bd.density     = Real(1);
        bd.friction    = Real(0.4f);
        bd.restitution = Real(0);
        hub = w.AddBody(bd);
    }

    // A dense ring of small dynamic disks all OVERLAPPING the hub (so each forms
    // a dynamic-dynamic contact with it). 20 > kColorCount (12) -> the hub is an
    // endpoint in > 12 dynamic-dynamic contacts -> coloring overflows for the
    // excess, exercising the scalar overflow path.
    const int kRing = 20;
    std::vector<BodyHandle> ring;
    for (int i = 0; i < kRing; ++i)
    {
        const Real ang = (Real(2) * Real(3.14159265358979323846)) * Real(i) / Real(kRing);
        // Place each small disk so it overlaps the hub's rim (centre ~ hub R).
        const Real rr = hubR;
        BodyDef bd;
        bd.type        = BodyType::Dynamic;
        bd.position    = Vec2(std::cos(ang) * rr, floorTop - hubR + std::sin(ang) * rr);
        bd.shape       = MakeCircle(Real(0.3));
        bd.density     = Real(1);
        bd.friction    = Real(0.4f);
        bd.restitution = Real(0);
        ring.push_back(w.AddBody(bd));
    }

    REQUIRE(w.Count() >= static_cast<std::uint32_t>(kRing + 2));

    // Settle (4 s). Track peak per-mass KE of every dynamic -> a dropped/over-
    // packed overflow would explode (huge KE) or sink (hub through the floor).
    // ALSO track the peak solver overflow count: this test is only meaningful if
    // the scene actually SPILLS past kColorCount into the scalar overflow path.
    // We assert that directly (REQUIRE(peakOverflow > 0) below) via the
    // SolverOverflowCount inspection hook instead of trusting the dense fan to
    // overflow "by construction" -- a future coloring change that stops it
    // spilling would then fail loud rather than silently skip the overflow code.
    const Real kStep = Real(1) / Real(60);
    Real peakKE = Real(0);
    std::size_t peakOverflow = 0;
    // Re-baselined for MKS (measured f32 60 Hz on this exact content, peak
    // specific KE ~26.8): the spawn config is a severe overlap cascade, not a
    // free fall -- each ring disk overlaps the hub's rim by ~0.3 m (rr == hubR
    // < hubR + ringR) AND overlaps its neighbors by ~0.225 m (chord ~0.375 m
    // vs 2*ringR == 0.6 m), so the first few steps eject significant energy as
    // the soft solver pushes the pile apart. Bound = ~1.5x measured headroom,
    // still far under the WorldDef velocity-cap ceiling (0.5 * 400^2).
    const Real keBound = Real(40);
    for (int s = 0; s < 240; ++s)
    {
        w.Step(kStep);
        peakOverflow = std::max(peakOverflow, w.SolverOverflowCount());
        const Vec2 vh = w.Velocity(hub);
        peakKE = std::max(peakKE, Real(0.5) * (vh.x * vh.x + vh.y * vh.y));
        for (BodyHandle b : ring)
        {
            const Vec2 v = w.Velocity(b);
            const Real ke = Real(0.5) * (v.x * v.x + v.y * v.y);
            peakKE = std::max(peakKE, ke);
            REQUIRE(ke < keBound);               // never blows up (overflow not dropped)
        }
    }

    // The overflow path was genuinely exercised: the hub overran kColorCount
    // dynamic-dynamic contacts and the excess spilled to the scalar tail. This is
    // the whole point of the scene -- if it ever stops overflowing the test is
    // void, so prove it with the direct accessor (not "by construction").
    REQUIRE(peakOverflow > 0);
    INFO("overflow-hub peak overflow constraint count = " << peakOverflow);

    // The hub did not sink through the floor (overflow contacts held it up).
    // Rewritten from the authored radius/floorTop (was the magic-literal form
    // `+ 12 <= floorTop + 1`): margin re-baselined to 0.1 m (measured hub rest
    // 1.800 m + hubR 1.2 m = 3.000 m vs floorTop + 0.1 = 3.1 m -> ~0.1 m clear).
    CHECK(w.Position(hub).y + hubR <= floorTop + Real(0.1));
    // The hub came to rest (the overflow solve dissipated its energy).
    // Re-baselined for MKS (measured specific KE ~0.100 at s=240): the pile does
    // not fully reach sleepThreshold-scale (0.05 m/s -> specific KE 0.00125)
    // within this exact 240-step window (a congested 21-body cascade settles
    // slower than a single body), but energy has dropped ~268x from the
    // keBound-scale peak (~26.8) -- bound at ~2x measured confirms substantial,
    // bounded dissipation without requiring full sleep in this window.
    const Vec2 vhf = w.Velocity(hub);
    CHECK(Real(0.5) * (vhf.x * vhf.x + vhf.y * vhf.y) < Real(0.2));
    INFO("overflow-hub peak per-mass KE = " << static_cast<double>(peakKE));
}
