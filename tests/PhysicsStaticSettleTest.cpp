// PhysicsStaticSettleTest.cpp -- regression for the "dynamic body never settles
// on / tunnels through a STATIC body" bug (Physics v2 debug).
//
// SYMPTOMS reproduced here as FAILING CPU tests (red before the fix):
//   (a) SETTLE  -- a horizontal capsule (radius 0.14 m, halfLen 0.24 m, the
//       Sandbox "Mixed shapes" green capsule) dropped onto a wide static floor
//       must come to REST: |velocity| ~ 0, |angVel| ~ 0, penetration within the
//       documented budget. Before the fix the capsule keeps a persistent
//       into-floor velocity (a single-point round manifold lets it rock) and
//       never sleeps.
//   (b) NO PULL-THROUGH -- a dynamic body resting on a static floor, driven with
//       a strong sustained DOWNWARD push every frame (mimicking the mouse drag),
//       must NOT sink deep into / pass through the floor. Tested for a capsule
//       AND a box. Before the fix the forced body penetrates well past the budget.
//
// The floor is CENTERED under the bodies (x in [-5.6,5.6] m) and bodies spawn
// near x=0 so they cannot roll off the floor edge and give a false "fell through".
//
// PRESENTATION-FREE + C++20-clean.

#include <cmath>
#include <cstdint>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>

using namespace Manifold2D::Physics;
using Catch::Approx;

namespace
{
    // Mimic the Sandbox mouse-spring drag (Interaction.cpp): drive the body's
    // grab anchor toward a target world point with a CAPPED impulse applied at
    // the anchor, then Step. This is the path the user uses to drag a body into
    // a static body. The caps below are copied VERBATIM from Interaction.hpp
    // (do not derive them) -- MKS P6 retuned production's drag caps from
    // px-era values (4000 / 40000 / 8) to these m/s-scale values, so this
    // local port must be re-synced whenever Interaction.hpp's caps change.
    void DragStep(PhysicsWorld& w, BodyHandle h, Vec2 grabLocalAnchor,
                  Vec2 cursorWorld, Real dt)
    {
        // Verbatim from Arcane/Sandbox/src/Interaction.hpp:109 (kDragMaxSpeed),
        // :121 (kDragMaxAccel), :129 (kDragMaxAngVel) -- MKS P6.
        constexpr Real kDragMaxSpeed  = Real(40.0);
        constexpr Real kDragMaxAccel  = Real(400.0);
        constexpr Real kDragMaxAngVel = Real(8.0);

        const std::uint32_t si = h.index;
        const Vec2 o = w.Position(h);
        const Real a = w.GetAngle(h);
        const Vec2 wa = o + RotateVec(a, grabLocalAnchor);
        const Vec2 com = o + RotateVec(a, w.GetLocalCenter(h));
        const Vec2 r(wa.x - com.x, wa.y - com.y);

        Vec2 desiredVel = (dt > Real(0))
            ? Vec2((cursorWorld.x - wa.x) / dt, (cursorWorld.y - wa.y) / dt)
            : Vec2(Real(0), Real(0));
        const Real speed = std::sqrt(desiredVel.x * desiredVel.x +
                                     desiredVel.y * desiredVel.y);
        if (speed > kDragMaxSpeed && speed > Real(0))
        {
            desiredVel.x *= kDragMaxSpeed / speed;
            desiredVel.y *= kDragMaxSpeed / speed;
        }

        const Vec2 vc = w.Velocity(h);
        const Real omega = w.AngVelSlot(si);
        const Vec2 anchorVel(vc.x - omega * r.y, vc.y + omega * r.x);

        const Real invM = w.InvMassSlot(si);
        const Real invI = w.InvInertiaSlot(si);
        const Vec2 cdv(desiredVel.x - anchorVel.x, desiredVel.y - anchorVel.y);
        const Real k11 = invM + invI * r.y * r.y;
        const Real k12 = -invI * r.x * r.y;
        const Real k22 = invM + invI * r.x * r.x;
        const Real det = k11 * k22 - k12 * k12;
        Vec2 impulse(Real(0), Real(0));
        if (det != Real(0))
        {
            const Real invDet = Real(1) / det;
            impulse.x = invDet * (k22 * cdv.x - k12 * cdv.y);
            impulse.y = invDet * (-k12 * cdv.x + k11 * cdv.y);
        }

        const Real mass = w.GetBodyMass(h);
        const Real maxImpulse = mass * kDragMaxAccel * dt;
        Real impLen = std::sqrt(impulse.x * impulse.x + impulse.y * impulse.y);
        if (impLen > maxImpulse && impLen > Real(0))
        {
            impulse.x *= maxImpulse / impLen;
            impulse.y *= maxImpulse / impLen;
        }
        const Real dOmega = invI * (r.x * impulse.y - r.y * impulse.x);
        const Real adO = std::abs(dOmega);
        if (adO > kDragMaxAngVel && adO > Real(0))
        {
            impulse.x *= kDragMaxAngVel / adO;
            impulse.y *= kDragMaxAngVel / adO;
        }

        w.Wake(h);
        w.ApplyImpulse(h, impulse, wa);
        w.Step(dt);
    }
}

namespace
{
    constexpr Real kStep = Real(1) / Real(60);

    // Sandbox "Mixed shapes" parameters (MKS: meters, g = 10 -- MKS P3).
    constexpr Real kGravityY  = Real(10);
    constexpr Real kCapRadius = Real(0.14);
    constexpr Real kCapHalf   = Real(0.24);

    // Floor: a wide static box centered at x=0. Top surface at y=0; the box
    // spans y in [0, 0.4] (half-height 0.2), x in [-5.6, 5.6] (half-width 5.6).
    constexpr Real kFloorTopY  = Real(0);
    constexpr Real kFloorHalfW = Real(5.6);
    constexpr Real kFloorHalfH = Real(0.2);

    BodyHandle AddFloor(PhysicsWorld& w)
    {
        BodyDef def;
        def.type     = BodyType::Static;
        def.position = Vec2(Real(0), kFloorTopY + kFloorHalfH); // center
        def.shape    = MakeAabb(kFloorHalfW, kFloorHalfH);
        def.friction = Real(0.4);
        return w.AddBody(def);
    }

    BodyHandle AddCapsule(PhysicsWorld& w, Vec2 pos)
    {
        BodyDef def;
        def.type        = BodyType::Dynamic;
        def.position    = pos;
        def.shape       = MakeCapsule(kCapHalf, kCapRadius);
        def.density     = Real(1);
        def.friction    = Real(0.4);
        def.restitution = Real(0.1);
        return w.AddBody(def);
    }

    BodyHandle AddBox(PhysicsWorld& w, Vec2 pos, Real hw, Real hh)
    {
        BodyDef def;
        def.type          = BodyType::Dynamic;
        def.position      = pos;
        def.shape         = MakeAabb(hw, hh);
        def.density       = Real(1);
        def.friction      = Real(0.4);
        def.restitution   = Real(0.1);
        def.fixedRotation = true; // axis-aligned box must be fixedRotation
        return w.AddBody(def);
    }
}

// ---------------------------------------------------------------------------
// (a) SETTLE: a horizontal capsule dropped onto a static floor comes to rest.
// ---------------------------------------------------------------------------
TEST_CASE("physics: capsule settles at rest on a static floor", "[physics]")
{
    WorldDef wd;
    wd.gravityY = kGravityY;
    PhysicsWorld w(wd);

    AddFloor(w);
    // Spawn the capsule a little above the floor so it drops a short distance
    // and lands flat (resting y ~ floorTop - radius == -0.14).
    BodyHandle cap = AddCapsule(w, Vec2(Real(0), Real(-0.6)));

    // Simulate ~3 seconds.
    for (int k = 0; k < 180; ++k)
    {
        w.Step(kStep);
    }

    const Vec2 v    = w.Velocity(cap);
    const Real spd  = std::sqrt(v.x * v.x + v.y * v.y);
    const Real ang  = std::abs(w.GetAngle(cap));
    const Real avel = std::abs(w.AngVelSlot(cap.index));
    const Vec2 p    = w.Position(cap);

    INFO("pos=(" << p.x << "," << p.y << ") vel=(" << v.x << "," << v.y
         << ") angVel=" << avel << " angle=" << ang << " awake=" << w.IsAwake(cap));

    // The capsule must have come to rest: near-zero linear + angular velocity.
    // Re-baselined for MKS (protocol rule 6): measured spd=0, avel=0 (the body
    // is already asleep by assert time -- see CHECK_FALSE(IsAwake) below -- and
    // island sleep zeroes both velocities, IslandManager::UpdateSleep). The spd
    // bound IS the sleep gate's ceiling: that pass requires
    // |v| + |w|*maxExtent < sleepThreshold (default 0.05 m/s) before sleeping,
    // so |v| < 0.05 at sleep onset. The avel bound rests on sleep-zeroing
    // alone: the gate by itself only bounds |w| by sleepThreshold/maxExtent
    // (~0.13 rad/s at this capsule's maxExtent = kCapHalf + kCapRadius =
    // 0.38 m), so 0.05 rad/s is measured-0 with headroom, not a gate-derived
    // ceiling.
    CHECK(spd  < Real(0.05));
    CHECK(avel < Real(0.05));

    // Penetration budget: the capsule bottom (p.y + radius) must not sink more
    // than the documented budget below the floor top (y=0). Re-baselined for
    // MKS (protocol rule 6): measured ~0.0003 m (SoftStep drives a single
    // resting contact toward kLinearSlop = 0.005 asymptotically, matching the
    // PhysicsSolverTest single-ball precedent); bound set at kLinearSlop/5 =
    // 0.001, ~3x headroom over measured.
    const Real penetration = (p.y + kCapRadius) - kFloorTopY;
    CHECK(penetration < Real(0.001));

    // And it must actually be resting on the floor (not floating far above it).
    // Re-baselined for MKS (protocol rule 6): a sanity gate, not a tight
    // residual bound -- measured penetration ~0.0003 m (essentially zero, not
    // floating). Bound at -kSkin (-0.02 m): a settled body should never be
    // found resting more than one skin-width above the floor surface.
    CHECK(penetration > Real(-0.02));

    // Eventually it should sleep (the island pass sleeps a settled body).
    CHECK_FALSE(w.IsAwake(cap));
}

// ---------------------------------------------------------------------------
// (a') SETTLE FROM A TILT: a capsule that lands at a small angle (as it does in
//     the Sandbox "Mixed shapes" scene after bumping a neighbor / an off-center
//     drag) must STILL settle. This is the precise repro of the never-settling
//     rocking bug: the round narrowphase emitted a SINGLE contact point, so the
//     tilted capsule rocked end-to-end in a never-damped limit cycle (constant
//     into-floor velocity + constant angVel, never sleeps).
// ---------------------------------------------------------------------------
TEST_CASE("physics: tilted capsule settles (no rocking limit cycle)", "[physics]")
{
    WorldDef wd;
    wd.gravityY = kGravityY;
    PhysicsWorld w(wd);

    AddFloor(w);
    BodyHandle cap = AddCapsule(w, Vec2(Real(0), Real(-0.6)));
    w.SetAngle(cap, Real(0.25)); // land tilted ~14 degrees

    for (int k = 0; k < 240; ++k)
    {
        w.Step(kStep);
    }

    const Vec2 v    = w.Velocity(cap);
    const Real spd  = std::sqrt(v.x * v.x + v.y * v.y);
    const Real avel = std::abs(w.AngVelSlot(cap.index));
    const Vec2 p    = w.Position(cap);
    const Real penetration = (p.y + kCapRadius) - kFloorTopY;

    INFO("pos=(" << p.x << "," << p.y << ") vel=(" << v.x << "," << v.y
         << ") angVel=" << avel << " penetration=" << penetration
         << " awake=" << w.IsAwake(cap));

    // Must come to rest -- the rocking bug left a persistent ~7 u/s into-floor
    // speed and ~0.29 rad/s angVel that never decayed. Re-baselined for MKS
    // (protocol rule 6): measured spd=0, avel=0, penetration ~0.0003 m (same
    // shape as the untilted settle case above -- see that case's comments for
    // the sleepThreshold / kLinearSlop reasoning behind these bounds).
    CHECK(spd  < Real(0.05));
    CHECK(avel < Real(0.05));
    CHECK(penetration < Real(0.001));
    CHECK_FALSE(w.IsAwake(cap)); // a settled capsule sleeps
}

// ---------------------------------------------------------------------------
// (b) NO PULL-THROUGH (capsule): a forced downward push every frame must not
//     drive the capsule deep into / through the static floor.
// ---------------------------------------------------------------------------
TEST_CASE("physics: forced capsule does not tunnel a static floor", "[physics]")
{
    WorldDef wd;
    wd.gravityY = kGravityY;
    PhysicsWorld w(wd);

    AddFloor(w);
    BodyHandle cap = AddCapsule(w, Vec2(Real(0), Real(-kCapRadius))); // resting

    // Let it settle a couple frames first.
    for (int k = 0; k < 10; ++k)
    {
        w.Step(kStep);
    }

    // Drive a strong sustained downward velocity every frame (mimics the drag
    // forcing the body into the floor). 6.0 m/s downward, far faster than the
    // body would ever move under gravity in one step.
    for (int k = 0; k < 120; ++k)
    {
        Vec2 v = w.Velocity(cap);
        w.SetVelocity(cap, Vec2(v.x, Real(6.0)));
        w.Step(kStep);

        const Vec2 p = w.Position(cap);
        const Real penetration = (p.y + kCapRadius) - kFloorTopY;
        INFO("frame " << k << " pos.y=" << p.y << " penetration=" << penetration);
        // The capsule must stay essentially on top of the floor: bounded
        // penetration, never sinking past the floor's far side (y=+0.4 bottom).
        // Re-baselined for MKS (protocol rule 6): measured max ~0.0097 m over
        // the loop. Bound set at 0.02 m -- 10% of kFloorHalfH (0.2 m), matching
        // the original file's 2.0/20 == 10% ratio, which also lands exactly at
        // kSkin (0.02 m); ~2.1x headroom over measured.
        CHECK(penetration < Real(0.02));
        CHECK(p.y < kFloorTopY + kFloorHalfH * Real(2)); // never below the floor bottom
    }
}

// ---------------------------------------------------------------------------
// (b'') NO DRAG PULL-THROUGH (faithful mouse-spring): grab the capsule off
//     center and drag the cursor straight DOWN to a point far below the floor's
//     far side. The capped-impulse mouse-spring (exactly Interaction.cpp) must
//     NOT pull the capsule through the static floor. This is the user's reported
//     "drag the capsule straight through the floor" path.
// ---------------------------------------------------------------------------
TEST_CASE("physics: dragged capsule does not pull through a static floor", "[physics]")
{
    WorldDef wd;
    wd.gravityY = kGravityY;
    PhysicsWorld w(wd);

    AddFloor(w);
    BodyHandle cap = AddCapsule(w, Vec2(Real(0), Real(-kCapRadius)));

    for (int k = 0; k < 10; ++k) { w.Step(kStep); }

    // Grab near the right end of the capsule (off-center, so the drag torques it
    // like a real off-center mouse grab) and aim the cursor far BELOW the floor.
    const Vec2 grabLocalAnchor(Real(0.2), Real(0));
    const Vec2 cursorBelow(Real(0), Real(2.0)); // 2.0 m below the floor top

    for (int k = 0; k < 200; ++k)
    {
        DragStep(w, cap, grabLocalAnchor, cursorBelow, kStep);

        const Vec2 p = w.Position(cap);
        const Real penetration = (p.y + kCapRadius) - kFloorTopY;
        INFO("frame " << k << " pos=(" << p.x << "," << p.y << ") penetration="
             << penetration << " angle=" << w.GetAngle(cap));
        // The capsule COM must never cross to the floor's far side: its origin
        // must stay above the floor bottom (y < +0.4). A pull-through would send
        // p.y past +0.4 toward the cursor at +2.0.
        CHECK(p.y < kFloorTopY + kFloorHalfH * Real(2));
        // And bounded penetration of the top surface (the solver resists it).
        // Re-baselined for MKS (protocol rule 6): measured max ~0.00357 m over
        // the loop -- LESS than the forced-push case's 0.0097 m, because the
        // capped-impulse mouse-spring (kDragMaxAccel, bounded FORCE) lets the
        // solver resist it, unlike the forced-push case's raw velocity
        // override. Bound set at 2*kLinearSlop = 0.01 m, ~2.8x headroom.
        CHECK(penetration < Real(0.01));
    }

    // After all that dragging, the capsule is still on top of the floor.
    const Vec2 pEnd = w.Position(cap);
    CHECK(pEnd.y < kFloorTopY);
}

// ---------------------------------------------------------------------------
// (b') NO PULL-THROUGH (box): same forced push for a polygon body.
// ---------------------------------------------------------------------------
TEST_CASE("physics: forced box does not tunnel a static floor", "[physics]")
{
    WorldDef wd;
    wd.gravityY = kGravityY;
    PhysicsWorld w(wd);

    AddFloor(w);
    const Real hw = Real(0.2), hh = Real(0.2);
    BodyHandle box = AddBox(w, Vec2(Real(0), -hh), hw, hh); // resting on floor

    for (int k = 0; k < 10; ++k)
    {
        w.Step(kStep);
    }

    for (int k = 0; k < 120; ++k)
    {
        Vec2 v = w.Velocity(box);
        w.SetVelocity(box, Vec2(v.x, Real(6.0)));
        w.Step(kStep);

        const Vec2 p = w.Position(box);
        const Real penetration = (p.y + hh) - kFloorTopY;
        INFO("frame " << k << " pos.y=" << p.y << " penetration=" << penetration);
        // Re-baselined for MKS (protocol rule 6): measured max ~0.00079 m over
        // the loop (a fixedRotation box settles flatter than the capsule).
        // Same bound as the forced-capsule case above (0.02 m, 10% of
        // kFloorHalfH / kSkin-scale) for a shared, well-clear tunneling gate;
        // ~25x headroom over measured here.
        CHECK(penetration < Real(0.02));
        CHECK(p.y < kFloorTopY + kFloorHalfH * Real(2));
    }
}
