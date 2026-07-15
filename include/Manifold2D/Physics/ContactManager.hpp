#pragma once

// ContactManager: persistent contact pairs + begin/stay/end + sensor events,
// with two-granularity event gating (M6, Task P1.8).
//
// PORT NOTE: a behavior-preserving port of
// Client/src/physics/ContactManager.lua (begin/stay/end events queued during
// Step and delivered AFTER it -- spec: no user code runs mid-step), with the
// collision-rebuild Phase-4 events-as-byproduct change: this manager NO LONGER
// runs its own broadphase + narrowphase candidate pass. PhysicsWorld derives
// the TOUCHED event body-pairs from the persistent ContactPool -- mover-mover
// (dynamic/kinematic, sensors included) and kinematic-vs-staticBody, EXCLUDING
// dynamic-vs-staticBody -- and hands them to Step() already canonical (min,max),
// sorted, and deduped. Mover-vs-TILE contacts deliberately do NOT flow here
// (cells are not bodies; the character controller owns its statics response).
//
// EVENT GATING (ported VERBATIM -- spec resolved decision #3): a per-body
// eventsEnabled flag PLUS a world-level gate. Gated events are DROPPED, not
// queued, and the pair's `begun` flag resets so re-enabling re-arms it; on
// re-enable every currently-overlapping pair emits a fresh begin
// (level-triggered re-arm).
//
// MODERNIZE (over the Lua, all behavior-preserving):
//   * PairKey is 64-bit: (uint64(min) << 32) | max. The Lua used a 16-bit
//     a*0x10000+b key (capped at 65535 bodies); widening to 64-bit lifts that
//     cap with no behavioral change.
//   * Overlap reuses the narrowphase via PhysicsWorld::SlotsOverlap (T7):
//     rotation + FIXTURE aware (iterates every fixture pair at the composed
//     real-angle world transform through the unified Collide, returning true on
//     the first contact point; margin=0). The fixture SoA is private to the
//     world, so the manager delegates rather than reaching in. (Before T7 the
//     manager called the rotation-blind single-shape CollideShapes(...,0).) The
//     Lua had a dedicated boolean shapesOverlap; delegating is DRY and keeps the
//     event overlap test consistent with the rotation-aware GenerateContacts.
//     ONE boundary subtlety carries over: Collide emits a point only for STRICT
//     penetration (depth > 0); an EXACT edge-touch (zero penetration) is NOT an
//     overlap here, whereas the Lua shapesOverlap used an INCLUSIVE aabbOverlap
//     (<=) for AABB/poly pairs. The tested scenes are non-degenerate (clear
//     overlap / clear gap), so this never bites; port shapesOverlap if a later
//     test needs inclusive edge-touch semantics.
//   * DETERMINISM: begin/stay events come from the caller-supplied TOUCHED
//     body-pairs -- ConstraintGraph::CollectTouchedEventPairs walks the pool ascending-id and
//     SORTS the touched pairs by (a,b) before calling Step (already
//     deterministic). The END events iterate the pairs map (unordered_map ->
//     nondeterministic order), so we COLLECT the to-end pairs, SORT them by
//     (a,b), then emit -- so End order is deterministic too.
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.
// No SDL3/NVRHI/Batcher2D/ImGui. Compiles both /MD (Arcane.dll) and
// pure-compute (Arcane-only, /MD). namespace
// Manifold2D::Physics, Core style.

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Broadphase/Broadphase.hpp>
#include <Mosaic/FunctionRef.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        class PhysicsWorld; // the manager reaches into the world SoA (port seam).

        // ----------------------------------------------------------------
        // ContactEvent: one delivered begin/stay/end (ports the Lua _queue row).
        // ----------------------------------------------------------------
        //
        // a/b are BodyHandles (the Lua delivered world.handles[idx]); a is the
        // lower slot, b the higher (pair canonical order). sensor is true iff
        // EITHER participant is a sensor (the Lua ev.sensor).
        struct ContactEvent
        {
            enum class Type : std::uint8_t
            {
                Begin = 0,
                Stay  = 1,
                End   = 2,
            };

            Type       type = Type::Begin;
            BodyHandle a{};
            BodyHandle b{};
            bool       sensor = false;
        };

        // ----------------------------------------------------------------
        // ContactManager: the persistent-pair + event machine.
        // ----------------------------------------------------------------
        class ContactManager
        {
        public:
            using Listener = std::function<void(const ContactEvent&)>;

            ContactManager() = default;

            // Install / replace the listener (ports onContact). Events fire only
            // when a listener is set.
            void SetListener(Listener fn) { m_listener = std::move(fn); }

            // Derive one step's events from the pre-built TOUCHED event body-pairs.
            // Phase 4 (events-as-byproduct): the candidate set is no longer an OWN
            // second broadphase + kinematic-static overlap pass -- the persistent
            // ContactPool already computed each pair's `touching` once this Step, so
            // PhysicsWorld hands us the deduped + sorted (min,max) body-pairs that
            // are event-relevant AND touching. For each: get/create the pair, stamp,
            // gate, queue begin/stay (NO overlap re-test -- the pair IS touching);
            // detect separations -> queue end (sorted); flush deferred to the
            // listener. `touchedPairs` MUST be ascending-sorted + deduped (Begin/Stay
            // emit order is exactly that order -- the determinism contract). Ports
            // the Lua step() with the overlap pass deleted.
            void Step(PhysicsWorld& w,
                      const std::vector<BroadphasePair>& touchedPairs);

            // Reset `begun` on pairs involving idx (idx omitted = all). No
            // synthetic end (gated events are dropped, not queued). Ports disarm.
            void Disarm();
            void Disarm(std::uint32_t idx);

            // Level-triggered re-arm: for currently-OVERLAPPING pairs involving
            // idx (idx omitted = all) that are !begun AND not gated, emit a fresh
            // begin IMMEDIATELY (re-arm happens outside Step, so synchronous
            // delivery is safe). Ports rearm.
            void Rearm(PhysicsWorld& w);
            void Rearm(PhysicsWorld& w, std::uint32_t idx);

            // Erase all pairs involving idx (ports dropBody). Called by
            // World::RemoveBody.
            void DropBody(std::uint32_t idx);

        private:
            // Shared Rearm body (filterIdx selects the idx-scoped variant).
            void RearmImpl(PhysicsWorld& w, bool filterIdx, std::uint32_t idx);

            // ------------------------------------------------------------
            // Pair: one persistent contact (ports the Lua pair table). a is the
            // lower slot, b the higher. begun gates begin-vs-stay; stamp marks
            // the step it was last touched (untouched -> separated).
            // ------------------------------------------------------------
            struct Pair
            {
                std::uint32_t a = 0; // min slot
                std::uint32_t b = 0; // max slot
                bool          begun = false;
                std::uint32_t stamp = 0;
            };

            // 64-bit pair key: (min << 32) | max (ports pairKey, widened).
            [[nodiscard]] static std::uint64_t PairKey(std::uint32_t a,
                                                       std::uint32_t b) noexcept
            {
                if (a > b)
                {
                    std::uint32_t t = a;
                    a = b;
                    b = t;
                }
                return (static_cast<std::uint64_t>(a) << 32) |
                       static_cast<std::uint64_t>(b);
            }

            // True iff slots a,b overlap NOW. Delegates to
            // PhysicsWorld::SlotsOverlap (T7: rotation + fixture aware,
            // pointCount>0). Sensor fixtures are NOT skipped (event gating must
            // see sensor overlaps). Phase 4: still used by RearmImpl (an out-of-step
            // op -- it must re-test overlap because no per-Step touch-state is
            // available); the hot Step path no longer calls it (the pair handed in
            // is ALREADY known touching).
            [[nodiscard]] static bool ShapesOverlap(const PhysicsWorld& w,
                                                    std::uint32_t a,
                                                    std::uint32_t b);

            // The Lua touch(a,b) for an ALREADY-TOUCHING body-pair (Phase 4): get/
            // create the pair, stamp it, gate, and queue begin/stay (or drop on
            // gate). The overlap re-test is GONE -- Step's caller (PhysicsWorld)
            // already filtered to touching event-relevant pairs from the pool, so
            // re-testing would be the redundant second narrowphase Phase 4 deletes.
            // (The alive guard is also redundant -- a pooled touching pair is alive
            // -- but kept defensive/cheap.)
            void Touch(PhysicsWorld& w, std::uint32_t a, std::uint32_t b,
                       std::uint32_t stamp);

            // Queue an event row (the Lua _emit -- deferred; not delivered here).
            void Emit(PhysicsWorld& w, ContactEvent::Type type,
                      std::uint32_t a, std::uint32_t b);

        public:
            // Visit each pair currently in the begun state (begun == true),
            // passing the two body SLOT indices (a <= b by canonical order).
            // Read-only; for debug draw / inspection.  Iteration order is
            // unordered (unordered_map traversal).
            void ForEachBegunPair(
                Mosaic::FunctionRef<void(std::uint32_t a, std::uint32_t b)> fn) const;

        private:
            // pair key -> pair. Reused across steps (no per-step alloc).
            std::unordered_map<std::uint64_t, Pair> m_pairs;
            Listener                                m_listener;

            // Deferred event queue (the Lua _queue/_qn). Reused; cleared by
            // resize-to-zero (capacity preserved).
            std::vector<ContactEvent> m_queue;
            std::uint32_t             m_stamp = 0;

            // Neutral scratch buffer: used by Step to collect-then-sort End
            // events, and by RearmImpl to collect-then-sort rearm Begin events.
            // Named neutrally because both code paths reuse the same member to
            // avoid per-call allocation (no dedicated per-purpose scratch).
            std::vector<Pair> m_workPairs;
            // PHASE 4: the m_pairScratch / m_bodyPairScratch broadphase scratch are
            // GONE -- the candidate set is now the pre-built touched body-pairs the
            // PhysicsWorld derives from the persistent pool (no second broadphase
            // walk + body-pair dedup here).
        };

    } // namespace Physics
} // namespace Manifold2D
