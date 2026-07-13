// Physics M6 P3.4: full-pipeline determinism replay (CCD + multi-joint + angle hash).
//
// =============================================================================
// COVERAGE MAP — P3.4 FULL-PIPELINE REPLAY
// =============================================================================
//
// This file is the P3.4 determinism gate explicitly deferred by P2.6.
// See the DEFERRED TO PHASE 3 note in PhysicsPhase2HarnessTest.cpp (lines 23-37
// and 50-56) for what this file must add over the P2.6 dynamics replay:
//
//   What P2.6 covered (PhysicsPhase2HarnessTest.cpp):
//     * Gravity + impulses + kinematic stirrer + 1 DistanceJoint, 240 steps.
//     * Run-twice identical hash (SoftStep self-check).
//     * Cross-broadphase (Tree/Hash/Sap) identical hashes.
//     * Hash formula: floor(x*1000) + floor(y*1000)*7 over 4 dynamic balls.
//
//   What P3.4 ADDS (this file):
//     * Bullet/CCD bodies: one bullet-kinematic + one bullet-dynamic aimed at a
//       thin static wall (exercises the P3.1 BulletSweep clamp).
//     * Multiple joint types: 3 distinct joints — Revolute, Weld, Prismatic —
//       between dynamic bodies. Their constrained trajectories enter the hash.
//     * Angle in the hash: floor(angle*1000)*13 per body per step. Dynamics +
//       joints + CCD all rotate bodies; angle is new state worth gating.
//     * Freely-rotating polygon box (v2, T9): one Dynamic MakePolygon box
//       (fixedRotation=false) released at a NONZERO angle above the floor.
//       Gravity + the v2 rotation-aware narrowphase + compound-COM contact
//       solver rotate it as it settles flat (the T5 "box settles flat" path).
//       Its position + angle enter the hash, so the determinism gate now also
//       covers the rotation-aware contact generation, not just joints/CCD.
//     * Anti-tunnel sub-assert: proves the CCD path is actually exercised (the
//       bullet body does NOT pass through the thin wall).
//     * Determinism: SoftStep run-twice identical.
//
// =============================================================================
// SCENE (single rich scene exercising the full Step pipeline)
// =============================================================================
//
// * gravityY = 10 (MKS gravity, matching the P2.6 harness convention).
// * Static floor: AABB at (20, 25) half-extents (30, 1.2).
// * Thin static wall: AABB at (35, 10) half-extents (0.1, 8). "Thin" (0.2 m
//   wide) -- a bullet body fired right at it would tunnel without CCD.
// * 3 dynamic bodies (d0, d1, d2): circle r=0.8, restitution=0.3, staggered.
//   Scripted impulses at step 60 and step 120.
// * 1 kinematic mover: constant authored velocity (+X), hashed for position.
// * 3 joints between dynamic bodies:
//     Revolute (static anchor above d0 ↔ d0): pendulum — angle evolves.
//     Weld    (d1 ↔ d2): rigid pair — relative angle locked.
//     Prismatic (static rail anchor ↔ d2): slide along +X only.
// * Bullet kinematic mover: starts left of the wall, fires right at it.
//   Flagged bullet=true → BulletSweep clamp catches it at the wall face.
// * Bullet dynamic body: also aimed at the wall, bullet=true (GJK-TOI backup).
//
// HASH FORMULA (extends P2.6):
//   h = (h * 31 + floor(x*1000) + floor(y*1000)*7 + floor(angle*1000)*13) % 2^48
// Hashed bodies: d0, d1, d2, kinematic mover, bullet-kinematic, bullet-dynamic,
//   rotating polygon box (appended last so the index order stays deterministic).
// Hash range: steps 1..240.
//
// ASSERTS:
//   1. Anti-tunnel: bullet bodies did NOT pass through the thin wall (CCD gate).
//   2. Run-twice identical (SoftStep): same hash both runs; hash != 0.
//
// PRESENTATION-FREE + C++20-clean. namespace Manifold2D::Physics helpers.
// Mirror structure + doc-comment style of PhysicsPhase2HarnessTest.cpp.

#include <cmath>
#include <cstdint>

#include <catch2/catch_test_macros.hpp>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Body.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Joints/Joints.hpp>

using namespace Manifold2D::Physics;

namespace Manifold2D { namespace Physics { namespace P34Determinism {

    // -------------------------------------------------------------------------
    // Scene constants.
    // -------------------------------------------------------------------------
    constexpr Real kDt    = Real(1) / Real(60);
    constexpr int  kSteps = 240;

    // Thin static wall geometry. Wall span: x[kWallX - kWallHW, kWallX + kWallHW].
    constexpr Real kWallX  = Real(35);
    constexpr Real kWallHW = Real(0.1f); // half-width -> wall 0.2 m wide
    constexpr Real kWallHH = Real(8);    // half-height
    constexpr Real kWallCY = Real(10);   // wall centre y

    // Bullet bodies start here (left of the wall), aimed right at it.
    constexpr Real kBulletStartX = Real(10);
    // Bullet speed: flat 150 m/s (probe-validated 2026-07-03), well under the
    // 400 m/s velocity cap. One-step travel = kBulletSpeed * kDt = 2.5 m, far
    // more than the 0.2 m wall thickness -- so the body genuinely tunnels
    // without CCD, which is exactly what the anti-tunnel gate proves is caught.
    constexpr Real kBulletSpeed = Real(150);

    // Probe radius for bullet bodies.
    constexpr Real kProbeR = Real(0.2f);

    // Rotating polygon box (v2, T9). A Dynamic AABB shape asserts fixedRotation
    // (PhysicsWorld.cpp:690), so a freely-rotating box MUST be a MakePolygon.
    // CCW corners (BL, BR, TR, TL) of a half-extent hw x hh box.
    static Shape BoxPolygon(Real hw, Real hh)
    {
        return MakePolygon({
            Vec2(-hw, -hh),
            Vec2( hw, -hh),
            Vec2( hw,  hh),
            Vec2(-hw,  hh),
        });
    }

    // Initial drop angle for the rotating box (radians). NONZERO so gravity +
    // the rotation-aware contact solver rotate it toward flat as it settles.
    constexpr Real kBoxInitAngle = Real(0.6);

    // -------------------------------------------------------------------------
    // Hash helper (P3.4 extension of the P2.6 formula).
    // Extended: adds the angle term (floor(angle*1000)*13).
    //
    //   h = (h*31 + floor(x*1000) + floor(y*1000)*7 + floor(angle*1000)*13) % 2^48
    //
    // std::floor on Real gives signed truncation-toward-neginf, matching
    // Lua's math.floor. The modulo keeps the value in [0, 2^48).
    // -------------------------------------------------------------------------
    constexpr std::uint64_t kHashMod = (std::uint64_t(1) << 48);

    static std::uint64_t HashStep(std::uint64_t h, Real x, Real y, Real angle)
    {
        const auto ix     = static_cast<std::int64_t>(std::floor(x     * Real(1000)));
        const auto iy     = static_cast<std::int64_t>(std::floor(y     * Real(1000)));
        const auto ia     = static_cast<std::int64_t>(std::floor(angle * Real(1000)));
        const auto contrib = static_cast<std::uint64_t>(
            ix + iy * std::int64_t(7) + ia * std::int64_t(13));
        return (h * std::uint64_t(31) + contrib) % kHashMod;
    }

    // -------------------------------------------------------------------------
    // Output struct: bullet positions after the full run (for the anti-tunnel
    // assert, which is evaluated once outside the hash loop).
    // -------------------------------------------------------------------------
    struct BulletResult
    {
        Vec2 bulletKinPos{ Real(0), Real(0) };
        Vec2 bulletDynPos{ Real(0), Real(0) };
    };

    // -------------------------------------------------------------------------
    // RunScene: build + step the full-pipeline scene for kSteps steps.
    //
    // Returns the accumulated 64-bit hash over all hashed bodies (d0/d1/d2,
    // kinematic mover, bullet-kinematic, bullet-dynamic) across steps 1..kSteps.
    //
    // If outBullets is non-null, final positions of the two bullet bodies are
    // written to it so the caller can assert the anti-tunnel invariant.
    //
    // Params:
    //   outBullets  -- optional: receives final bullet body positions.
    // -------------------------------------------------------------------------
    static std::uint64_t RunScene(BulletResult* outBullets = nullptr)
    {
        // ---- World -----------------------------------------------------------
        WorldDef wdef;
        wdef.gravityY   = Real(10);              // MKS gravity (y-down)
        wdef.broadphase = BroadphaseKind::Tree;  // pinned to Tree (deliberate scene choice)

        PhysicsWorld w(wdef);

        // ---- Static floor (wide, centred at (20, 25)) ----------------------
        {
            BodyDef fd;
            fd.type     = BodyType::Static;
            fd.position = Vec2(Real(20), Real(25));
            fd.shape    = MakeAabb(Real(30), Real(1.2f));
            w.AddBody(fd);
        }

        // ---- Thin static wall (bullet target; CCD path) ---------------------
        {
            BodyDef wd;
            wd.type     = BodyType::Static;
            wd.position = Vec2(kWallX, kWallCY);
            wd.shape    = MakeAabb(kWallHW, kWallHH);
            w.AddBody(wd);
        }

        // ---- Static anchor for Revolute joint (above d0) --------------------
        BodyHandle revAnchor;
        {
            BodyDef ad;
            ad.type     = BodyType::Static;
            ad.position = Vec2(Real(8), Real(5));
            ad.shape    = MakeCircle(Real(0.3f));
            revAnchor   = w.AddBody(ad);
        }

        // ---- Static rail anchor for Prismatic joint (beside d2) -------------
        BodyHandle railAnchor;
        {
            BodyDef ad;
            ad.type     = BodyType::Static;
            ad.position = Vec2(Real(20), Real(8));
            ad.shape    = MakeCircle(Real(0.3f));
            railAnchor  = w.AddBody(ad);
        }

        // ---- 3 dynamic bodies (d0, d1, d2): circle r=0.8, restitution=0.3 ---
        // d0 hangs below revAnchor for the pendulum.
        // d1 and d2 are welded together; d2 also attached to the prismatic rail.
        const Real kDynR = Real(0.8f); // dynamic-circle radius (reused for mass)
        BodyHandle dyn[3];
        {
            const Vec2 positions[3] = {
                Vec2(Real(8),  Real(10)), // d0: below revAnchor
                Vec2(Real(15), Real(6)),  // d1: weld partner A
                Vec2(Real(20), Real(8)),  // d2: weld partner B + prismatic slider
            };
            for (int i = 0; i < 3; ++i)
            {
                BodyDef bd;
                bd.type        = BodyType::Dynamic;
                bd.position    = positions[i];
                bd.shape       = MakeCircle(kDynR);
                bd.density     = Real(1);
                bd.restitution = Real(0.3f);
                bd.friction    = Real(0.4f);
                dyn[i]         = w.AddBody(bd);
            }
        }

        // ---- Kinematic stirrer sweeping right --------------------------------
        BodyHandle kin;
        {
            BodyDef kd;
            kd.type     = BodyType::Kinematic;
            kd.position = Vec2(Real(0), Real(22));
            kd.shape    = MakeCircle(Real(1));
            kin = w.AddBody(kd);
            w.SetVelocity(kin, Vec2(Real(6), Real(0)));
        }

        // ---- Bullet kinematic: fires at the thin wall -----------------------
        // bullet=true → the BulletSweep stage in Step clamps it to TOI.
        BodyHandle bulletKin;
        {
            BodyDef bd;
            bd.type     = BodyType::Kinematic;
            bd.position = Vec2(kBulletStartX, kWallCY);
            bd.shape    = MakeCircle(kProbeR);
            bd.bullet   = true;
            bulletKin   = w.AddBody(bd);
            w.SetVelocity(bulletKin, Vec2(kBulletSpeed, Real(0)));
        }

        // ---- Bullet dynamic: also fires at the thin wall --------------------
        // bullet=true → speculative contacts + GJK-TOI backup both apply.
        BodyHandle bulletDyn;
        {
            BodyDef bd;
            bd.type     = BodyType::Dynamic;
            bd.position = Vec2(kBulletStartX, kWallCY + Real(0.6f));
            bd.shape    = MakeCircle(kProbeR);
            bd.density  = Real(1);
            bd.bullet   = true;
            bulletDyn   = w.AddBody(bd);
            w.SetVelocity(bulletDyn, Vec2(kBulletSpeed, Real(0)));
        }

        // ---- Rotating polygon box (v2, T9): angled drop onto the floor ------
        // A Dynamic MakePolygon box (NOT MakeAabb -- a dynamic AABB asserts
        // fixedRotation) released at kBoxInitAngle above the static floor
        // (floor centre y=25, half-height 1.2 -> top surface ~23.8). Gravity
        // pulls it down; the v2 rotation-aware narrowphase + compound-COM
        // contact solver rotate it toward flat as it settles (T5 behavior).
        // fixedRotation=false so its angle is free to evolve and is hashed.
        BodyHandle box;
        {
            BodyDef bd;
            bd.type          = BodyType::Dynamic;
            bd.position      = Vec2(Real(20), Real(18)); // above the floor
            bd.shape         = BoxPolygon(Real(1), Real(1));
            bd.density       = Real(1);
            bd.restitution   = Real(0.1f);
            bd.friction      = Real(0.4f);
            bd.fixedRotation = false; // free to spin/settle
            box = w.AddBody(bd);
            // Release at a nonzero tilt so settling rotates it (SetAngle writes
            // the live m_angle the narrowphase transform reads).
            w.SetAngle(box, kBoxInitAngle);
        }

        // ---- Deep-overlap round body (v2 EPA/MPR): spawned OVERLAPPING a
        // static block so its first contact is a DEEP overlap, exercising
        // Collide's EPA deep-round cell (MPR fallback) in the full pipeline.
        // The f64->f32 EPA/MPR paths are deterministic; this body's position +
        // angle enter the hash LAST (stable index order), so the determinism
        // gate now also covers the deep-overlap recovery path. (This adds a body
        // to the scene -> it DELIBERATELY changes the hash value; the gate is
        // run-twice IDENTITY + != 0, not a pinned constant.)
        BodyHandle deepBlock; // static block the round body is buried in
        {
            BodyDef bd;
            bd.type     = BodyType::Static;
            bd.position = Vec2(Real(6), Real(18)); // off to the side, clear of the floor stack
            bd.shape    = MakeAabb(Real(2), Real(2));
            deepBlock   = w.AddBody(bd);
        }
        (void)deepBlock;
        BodyHandle deepBall;
        {
            BodyDef bd;
            bd.type        = BodyType::Dynamic;
            bd.position    = Vec2(Real(6), Real(17.4f)); // 0.6 above the block centre, DEEP inside
            bd.shape       = MakeCircle(Real(0.5f));
            bd.density     = Real(1);
            bd.restitution = Real(0.1f);
            bd.friction    = Real(0.4f);
            deepBall       = w.AddBody(bd);
        }

        // ---- Joint 1: Revolute — d0 as pendulum from revAnchor --------------
        // d0 starts offset from the anchor; gravity swings it down.
        // The body's angle evolves continuously → angle term is non-trivial.
        {
            JointDef jd;
            jd.kind   = JointKind::Revolute;
            jd.a      = revAnchor;
            jd.b      = dyn[0];
            jd.anchor = Vec2(Real(8), Real(5)); // pivot at static anchor centre
            w.AddJoint(jd);
        }

        // ---- Joint 2: Weld — d1 and d2 locked rigidly together --------------
        // The weld holds relative position + angle; the locked pair falls as a
        // unit. Their shared angle enters the hash once per body per step.
        {
            JointDef jd;
            jd.kind   = JointKind::Weld;
            jd.a      = dyn[1];
            jd.b      = dyn[2];
            jd.anchor = w.Position(dyn[1]); // weld point at d1's creation pos
            w.AddJoint(jd);
        }

        // ---- Joint 3: Prismatic — d2 slides along +X from railAnchor --------
        // Gravity is +Y (perpendicular to the +X axis); the perp constraint
        // must hold y. An impulse at step 60 pushes d2 along +X.
        {
            JointDef jd;
            jd.kind = JointKind::Prismatic;
            jd.a    = railAnchor;
            jd.b    = dyn[2];
            jd.axis = Vec2(Real(1), Real(0)); // horizontal slide axis
            w.AddJoint(jd);
        }

        // ---- Scripted run (steps 1..kSteps) ---------------------------------
        // Impulses authored as mass * delta-v (protocol rule 3): the dynamic
        // circles share density 1 and radius kDynR, so mass = density*kPi*r*r.
        // Target delta-v values (m/s) are the px-era kicks (impulse / px-mass
        // pi*8*8 = 201) rescaled /10 with this scene's velocities:
        //   d0 up  : 300  / 201 = 1.49 px/s -> 0.15 m/s
        //   d2 +X  : 2000 / 201 = 9.95 px/s -> 1.0  m/s
        //   d1 pair: (80,-120)/201 = (0.40,-0.60) px/s -> (0.04,-0.06) m/s
        const Real kDynMass = Real(1) * kPi * kDynR * kDynR;
        const Vec2 kD0KickDv(Real(0),     Real(-0.15f)); // step 60: kick d0 upward
        const Vec2 kD2KickDv(Real(1),     Real(0));      // step 60: slide d2 along +X
        const Vec2 kD1KickDv(Real(0.04f), Real(-0.06f)); // step 120: nudge weld pair

        std::uint64_t hash = 0;
        for (int step = 1; step <= kSteps; ++step)
        {
            // Scripted impulses (mirror P2.6 step-60/step-120 pattern).
            if (step == 60)
            {
                // Kick d0 upward (pendulum gets energy).
                w.ApplyImpulse(dyn[0], kDynMass * kD0KickDv);
                // Kick d2 along the prismatic axis (+X) so it slides.
                w.ApplyImpulse(dyn[2], kDynMass * kD2KickDv);
            }
            if (step == 120)
            {
                // Kick d1 to inject more motion into the weld pair.
                w.ApplyImpulse(dyn[1], kDynMass * kD1KickDv);
            }

            w.Step(kDt);

            // -- Accumulate hash over all hashed bodies (stable index order) --

            // Dynamic bodies d0, d1, d2: position + angle.
            for (int i = 0; i < 3; ++i)
            {
                const Vec2 p = w.Position(dyn[i]);
                const Real a = w.GetAngle(dyn[i]);
                hash = HashStep(hash, p.x, p.y, a);
            }

            // Kinematic mover: position + angle (returns 0; uniform with other bodies).
            {
                const Vec2 p = w.Position(kin);
                const Real a = w.GetAngle(kin);
                hash = HashStep(hash, p.x, p.y, a);
            }

            // Bullet-kinematic: position + angle (post-CCD clamp on step 1).
            {
                const Vec2 p = w.Position(bulletKin);
                const Real a = w.GetAngle(bulletKin);
                hash = HashStep(hash, p.x, p.y, a);
            }

            // Bullet-dynamic: position + angle (speculative contacts keep it
            // on the near side; angle may change from contact torques).
            {
                const Vec2 p = w.Position(bulletDyn);
                const Real a = w.GetAngle(bulletDyn);
                hash = HashStep(hash, p.x, p.y, a);
            }

            // Rotating polygon box: position + angle (appended LAST so the
            // index order stays deterministic). Its angle evolves from the
            // angled drop + rotation-aware contact solve toward flat -- this
            // is the v2 narrowphase/compound-COM coverage this body adds.
            {
                const Vec2 p = w.Position(box);
                const Real a = w.GetAngle(box);
                hash = HashStep(hash, p.x, p.y, a);
            }

            // Deep-overlap round body: position + angle (appended AFTER the box
            // so the index order stays deterministic). Spawned buried in a
            // static block, its first contact takes the EPA deep-round cell; the
            // f64->f32 EPA/MPR recovery path is deterministic, so its evolving
            // position + angle gate that path's determinism. (Adding this body
            // intentionally re-baselines the hash VALUE; run-twice identity + !=0
            // is the gate, not a pinned constant.)
            {
                const Vec2 p = w.Position(deepBall);
                const Real a = w.GetAngle(deepBall);
                hash = HashStep(hash, p.x, p.y, a);
            }
        }

        // Write bullet final positions for the anti-tunnel assert.
        if (outBullets)
        {
            outBullets->bulletKinPos = w.Position(bulletKin);
            outBullets->bulletDynPos = w.Position(bulletDyn);
        }

        return hash;
    }

} } } // namespace Manifold2D::Physics::P34Determinism

using namespace Manifold2D::Physics::P34Determinism;

// ---------------------------------------------------------------------------
// 1. Anti-tunnel sub-assert: prove the CCD path is actually exercised.
//    Both bullet bodies (kinematic + dynamic) must NOT tunnel through the thin
//    static wall. This is a focused sub-assert outside the hash loop so the
//    test reader can see clearly what is being validated.
//
//    Wall near face: kWallX - kWallHW = 34.9.
//    Without CCD, one-step displacement = kBulletSpeed * kDt = 2.5 m, landing
//    the bullet well past the 0.2 m wall in a single step.
//    With CCD (TOI clamp for kinematic; speculative margin for dynamic), the
//    bullet centre stops at or before the near face. Because the bullet shape
//    has radius kProbeR = 0.2, the analytic stopped centre is about
//    kWallNearX - kProbeR (= 34.7); the <= kWallNearX bound has generous margin
//    while still rejecting any half-tunnel (a centre past the near face x = 34.9
//    is rejected here).
// ---------------------------------------------------------------------------

// Wall near face constant (near side of the 0.2 m wall).
constexpr Real kWallNearX = kWallX - kWallHW; // = 34.9

TEST_CASE("P3.4 full-pipeline determinism: bullet bodies do not tunnel (CCD gate)",
          "[physics][determinism]")
{
    BulletResult bullets{};
    RunScene(&bullets);

    // Bullet-kinematic: BulletSweep clamps to TOI -> centre at or before near face.
    // A half-tunnel or full tunnel (centre past the near face) fails the gate.
    // Empirical stopped centre: x ~ 34.70 (~ kWallNearX - kProbeR, CCD skin margin).
    REQUIRE(bullets.bulletKinPos.x <= kWallNearX);

    // Bullet-dynamic: speculative contacts + GJK-TOI -> same invariant.
    // Empirical stopped centre: x ~ 34.70 (clamped at the near face, ~ kWallNearX - kProbeR).
    REQUIRE(bullets.bulletDynPos.x <= kWallNearX);
}

// ---------------------------------------------------------------------------
// 2. Run-twice identical (SoftStep, DynamicTree broadphase).
//    Full pipeline: dynamics + CCD bullets + Revolute/Weld/Prismatic joints +
//    angle in hash. Same binary, same input → identical final hash.
// ---------------------------------------------------------------------------

TEST_CASE("P3.4 full-pipeline determinism: run-twice identical (SoftStep)",
          "[physics][determinism]")
{
    const std::uint64_t h1 = RunScene();
    const std::uint64_t h2 = RunScene();
    REQUIRE(h1 == h2);
    // Sanity: hash must be non-zero (the scene exercised non-trivial state).
    REQUIRE(h1 != 0u);
}
