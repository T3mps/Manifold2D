#pragma once

// CharacterController: game-facing character motion (M6, Task P1.10).
//
// PORT NOTE: a faithful port of Client/src/physics/CharacterController.lua --
// the LIVE GAMEPLAY layer (the strongest parity requirement of the milestone).
// The Lua module is the behavioral oracle; the algorithm below is its direct
// translation. Two motion modes:
//
//   SlideMove(dx, dy) -> actual displacement: WASD. SUBSTEPPED capsule
//     DEPENETRATION with multi-pass slide vs TileGrid spans + static bodies.
//     The Lua comment is explicit: "Discrete substeps (<= 8px) stand in for the
//     true swept test until GJK/CCD (M2)". So this is NOT a true swept/ShapeCast
//     test -- it marches the body forward in <= MAX_SUBSTEP-px steps and
//     depenetrates after each. Combined, the two P1 invariants hold: never
//     buried in a solid footprint, never tunneling on a dt spike. The milestone
//     plan's "swept-capsule / two-pass slide" wording maps onto exactly this
//     substep + MAX_PASSES depenetration -- we port the REAL algorithm the Lua
//     decides, not a ShapeCast sweep (that would change behavior + break the
//     oracle invariants).
//
//   FollowVelocity(vx, vy): click-to-move. Plain kinematic velocity; the world
//     Step integrates it without deflection (paths are cell-safe by
//     construction; the follower is never deflected -- it only reports overlaps).
//
// INTEGRATION CONTRACT (ORDER MATTERS): the caller runs world.Step(dt) FIRST,
// then SlideMove / FollowVelocity, every fixedUpdate. Step snapshots prev at its
// top, so with the slide applied AFTER it the render boundary's prev->pos lerp
// spans EXACTLY this tick's slide. In WASD mode the body's velocity stays 0
// (Step never double-moves the slid body -- the CC writes position directly via
// PhysicsWorld::MovePosition, which leaves prev to Step; velocity is untouched
// at 0). In click mode the velocity set THIS tick is integrated by the NEXT
// Step.
//
// Why MovePosition and not SetPosition: SetPosition is a TELEPORT (it snaps prev
// too, killing the lerp across the jump). The slide must keep prev managed by
// Step, so it writes through PhysicsWorld::MovePosition (pos + mover-broadphase
// AABB, prev untouched) -- the exact analog of the Lua's direct
// w.posX[i]=x / w.posY[i]=y / moverHash:update write-back.
//
// PLAYER SHAPE: a capsule (the segment (x-halfLen,y)..(x+halfLen,y) inflated by
// r). A circle is handled as a ZERO-LENGTH capsule (halfLen treated as 0), so a
// circle body drives the CC through the identical code path.
//
// DETERMINISM: substep order is fixed (march remaining in MAX_SUBSTEP chunks);
// the deepest push-out is selected with a stable strict-greater tiebreak (first
// candidate of equal depth wins, in StaticCandidates' index order: spans before
// statics, both index-ordered); no wall-clock; no fast-math. ZERO steady-state
// allocation: the span/static candidate buffers and the poly scratch are member
// state reused across calls (clear()+push_back preserves capacity), mirroring
// the Lua controller's _spans / _statics / _polyScratch.
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.
// No SDL3/NVRHI/Batcher2D/ImGui, no iso/Map/world coupling. Compiles both /MD
// (Arcane.dll) and pure-compute (Arcane-only, /MD).
// namespace Manifold2D::Physics, Core style.

#include <array>
#include <cstdint>
#include <vector>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        // ----------------------------------------------------------------
        // CharacterController: WASD slide + click-to-move over a kinematic body.
        // ----------------------------------------------------------------
        class CharacterController
        {
        public:
            // The capsule march step. Discrete substeps <= this length stand in
            // for the true swept test (Lua MAX_SUBSTEP).
            static constexpr Real kMaxSubstep = Real(0.1); // m -- spec sec 3 (MKS P4); kMaxPasses stays the iteration count
            // Max depenetration passes per substep. Corner regions resolve
            // against 2+ span normals; alternating push-out needs headroom to
            // converge (Lua MAX_PASSES).
            static constexpr int kMaxPasses = 8;
            // Resolve to a hair outside the surface (Lua SKIN).
            static constexpr Real kDepenetrationSkin = Real(0.02); // = kSkin value (spec sec 3, MKS P4); still distinct from Manifold2D::Physics::kSkin in purpose

            // `world` and `body` must outlive the controller. The body is the
            // player (a kinematic capsule or circle). The controller never owns
            // either -- it is a thin motion driver over them (ports
            // CharacterController.new).
            CharacterController(PhysicsWorld& world, BodyHandle body) noexcept
                : m_world(&world), m_body(body)
            {
            }

            [[nodiscard]] BodyHandle Body() const noexcept { return m_body; }

            // WASD: move by (dx,dy) with substepped depenetration slide.
            // Returns the ACTUAL displacement (resolved end pos - start pos).
            // Writes the body position through PhysicsWorld::MovePosition (prev
            // untouched). |d| < 1e-9 -> no-op, returns (0,0). Ports slideMove.
            [[nodiscard]] Vec2 SlideMove(Real dx, Real dy);

            // Click-to-move: hand the velocity to the kinematic body; the world
            // Step integrates it without deflection. Ports followVelocity.
            void FollowVelocity(Real vx, Real vy) noexcept
            {
                m_world->SetVelocity(m_body, Vec2(vx, vy));
            }

            void FollowVelocity(Vec2 v) noexcept { FollowVelocity(v.x, v.y); }

        private:
            // Push the body at (x,y) out of all nearby statics (tile spans +
            // static bodies). Returns the resolved (x,y). Round shapes only:
            // capsule, or circle as a zero-length capsule. Ports _depenetrate.
            Vec2 Depenetrate(Vec2 p);

            PhysicsWorld* m_world = nullptr;
            BodyHandle    m_body{};

            // Reused candidate scratch (zero steady-state alloc) -- the Lua
            // controller's _spans / _statics / _polyScratch. NOT re-entrant
            // (matches the world query-scratch contract: single-threaded, never
            // called from inside a contact callback).
            std::vector<Aabb2>         m_spans;
            std::vector<std::uint32_t> m_statics;
            // Scratch for DynamicTree::QueryAABB (against m_staticTree) inside
            // StaticCandidates (caller-supplied since the narrowphase-MT refactor;
            // see task 1).
            std::vector<std::uint32_t> m_staticGridScratch;
            // 4 corners of a span / static-AABB expanded to a poly for
            // CapsulePoly (the Lua aabbPoly write into _polyScratch).
            std::array<Vec2, 4>        m_polyScratch{};
            // World-space verts of a static POLYGON (local Shape verts offset by
            // the body pos). Reused (clear()+push_back preserves capacity).
            std::vector<Vec2>          m_polyWorld;
        };

    } // namespace Physics
} // namespace Manifold2D
