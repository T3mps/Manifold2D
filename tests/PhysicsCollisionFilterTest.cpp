// PhysicsCollisionFilterTest.cpp
// [physics][filter]: per-fixture collision-filter enforcement (Box2D category/mask).
//
// Three cases:
//   (a) Two dynamics whose primary fixture category/mask sets are disjoint fall
//       THROUGH each other: no contact is created in the pool, no begin event
//       fires (the gap this Task D closes -- case (a) FAILS on unpatched code
//       because TryCreateContact had no filter gate).
//   (b) Two dynamics with compatible (but non-default) filters DO collide and
//       accumulate a persistent contact (regression guard for the gate).
//   (c) Two dynamics with default filters (cat=1/mask=0xFFFFFFFF) collide
//       (byte-identical regression for all existing scenes).
//
// PRESENTATION-FREE, ASCII comments, C++23, /MD.

#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/ContactManager.hpp>

using namespace Manifold2D::Physics;

namespace
{
    constexpr Real kStep = Real(1) / Real(60);

    // Helper: add a Dynamic circle at `pos` with explicit filter bits.
    // The BodyDef.categoryBits / maskBits flow through to the auto-fixture.
    BodyHandle AddFiltered(PhysicsWorld& w, Vec2 pos, Real r,
                           std::uint32_t catBits, std::uint32_t maskBits)
    {
        BodyDef d;
        d.type         = BodyType::Dynamic;
        d.position     = pos;
        d.shape        = MakeCircle(r);
        d.density      = Real(1);
        d.categoryBits = catBits;
        d.maskBits     = maskBits;
        return w.AddBody(d);
    }
} // namespace

// ---------------------------------------------------------------------------
// (a) Filtered-apart pair: must fall through -- no contact, no begin event.
//     catA=0x0002, maskA=0x0002, catB=0x0004, maskB=0x0004
//     => (catA & maskB) = (0x0002 & 0x0004) = 0  -> filtered
// ---------------------------------------------------------------------------
TEST_CASE("Collision filter: disjoint category/mask pair never creates a contact",
          "[physics][filter]")
{
    // Gravity pulls downward so the bodies are kinetically moving (covers the
    // broadphase path). WorldDef default gravityY is now MKS 10; this scene
    // keeps the default (no override needed).
    WorldDef wd;
    PhysicsWorld w(wd);

    // Place A above B, overlapping (radius=1 each, centers 0.8 apart -> overlap).
    // Without filtering they would push apart; with filtering they pass through.
    BodyHandle bA = AddFiltered(w, Vec2(Real(0), Real(0)),  Real(1), 0x0002u, 0x0002u);
    BodyHandle bB = AddFiltered(w, Vec2(Real(0), Real(0.8)),  Real(1), 0x0004u, 0x0004u);

    int beginCount = 0;
    w.OnContact([&](const ContactEvent& ev)
    {
        if (ev.type == ContactEvent::Type::Begin)
        {
            ++beginCount;
        }
    });

    // Step enough frames for any real collision to manifest.
    for (int i = 0; i < 10; ++i)
    {
        w.Step(kStep);
    }

    // No contact should have been created (filter gate rejects at TryCreateContact).
    CHECK(w.DebugContactCount() == 0);
    CHECK_FALSE(w.DebugHasContact(bA, bB));
    CHECK(beginCount == 0); // no begin event fired
}

// ---------------------------------------------------------------------------
// (b) Compatible (non-default) filters: pair collides and builds a contact.
//     catA=0x0002, maskA=0x0006, catB=0x0004, maskB=0x0003
//     => (catA & maskB) = (0x0002 & 0x0003) = 0x0002 != 0
//     => (catB & maskA) = (0x0004 & 0x0006) = 0x0004 != 0  -> allowed
// ---------------------------------------------------------------------------
TEST_CASE("Collision filter: compatible non-default filters allow contact creation",
          "[physics][filter]")
{
    WorldDef wd;
    wd.gravityX = Real(0);
    wd.gravityY = Real(0); // zero-g: overlap is purely geometric
    PhysicsWorld w(wd);

    // Overlapping circles (centers 0.8 apart, radius 1 each -> overlap).
    BodyHandle bA = AddFiltered(w, Vec2(Real(0), Real(0)), Real(1), 0x0002u, 0x0006u);
    BodyHandle bB = AddFiltered(w, Vec2(Real(0), Real(0.8)), Real(1), 0x0004u, 0x0003u);

    w.Step(kStep);

    // A contact MUST have been created for the overlapping compatible pair.
    CHECK(w.DebugContactCount() >= 1);
    CHECK(w.DebugHasContact(bA, bB));
}

// ---------------------------------------------------------------------------
// (c) Default filters (cat=1, mask=0xFFFFFFFF): always collide -- no regression.
//     This is the existing scene default; none of the live test cases should
//     change behavior.
// ---------------------------------------------------------------------------
TEST_CASE("Collision filter: default category/mask (cat=1 mask=all) collide",
          "[physics][filter]")
{
    WorldDef wd;
    wd.gravityX = Real(0);
    wd.gravityY = Real(0); // zero-g: overlap is purely geometric
    PhysicsWorld w(wd);

    // Two overlapping dynamics with default filters (BodyDef defaults cat=1 mask=all).
    BodyDef d;
    d.type     = BodyType::Dynamic;
    d.density  = Real(1);
    d.shape    = MakeCircle(Real(1));

    d.position = Vec2(Real(0), Real(0));
    BodyHandle bA = w.AddBody(d);

    d.position = Vec2(Real(0), Real(0.8));
    BodyHandle bB = w.AddBody(d);

    w.Step(kStep);

    CHECK(w.DebugContactCount() >= 1);
    CHECK(w.DebugHasContact(bA, bB));
}
