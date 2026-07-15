#pragma once

// Contact.hpp -- persistent fixture-pair contact + a deduped, generation-safe
// pool of them (Manifold2D 2D physics, collision rebuild Phase 3, Task 1).
//
// One Contact is the durable record for a single solver-relevant overlapping
// fixture-pair. It SURVIVES across steps (so warm-starting + per-pair manifold
// caching have a stable home); the broadphase creates/destroys the pool entries
// and the narrowphase recomputes the cached Manifold at most once per step.
//
// ContactPool is the storage: a free-list-backed vector of Contacts indexed by
// a stable contact id, deduped by the UNORDERED fixture-slot pair, and
// generation-safe -- a recycled fixture slot (same index, bumped generation)
// is a DISTINCT key, so a stale contact never aliases a new fixture that lands
// in the same slot.
//
// PHASE 3 NOTE: this is a pure data structure -- defined + unit-tested but NOT
// yet consumed by the Step path, so it introduces no behavior change. Events
// stay in ContactManager; the Contact carries only what the solver feed needs.
//
// PRESENTATION-FREE + C++20-clean: std + sibling Physics headers only. No
// SDL3/NVRHI/Batcher2D/ImGui. Compiles both /MD (Arcane.dll) and static-CRT
// (Arcane-only, /MD).

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <Mosaic/FunctionRef.hpp>

#include <Manifold2D/Physics/Fixture.hpp>        // pulls in PhysicsTypes.hpp (kInvalidSlot)
#include <Manifold2D/Physics/Narrowphase/Manifold.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        // PERSISTENT INCREMENTAL COLORING (collision-rebuild Phase C, Stage 2,
        // Task 4). A solver-relevant body-body contact is assigned a graph color
        // ONCE at create and releases it at destroy (no per-step greedy recolor).
        // kInvalidColor marks "uncolored" -- the default for a fresh/recycled slot
        // and the value carried by sensors, non-solver, span, and OVERFLOW contacts
        // (a body-body contact that found no free color among the kColorCount). The
        // value is < kColorCount sentinel territory and < 32 (the per-body mask is a
        // 32-bit bitmask), so 0xFF can never collide with a real color index.
        inline constexpr std::uint8_t kInvalidColor = 0xFFu;

        // Transient per-step narrowphase state-change flags (Box2D simFlags
        // equivalent). Set in the MT collide pass, consumed+cleared in the serial
        // tail. NOT serialized (transient). 0 = no change this step.
        inline constexpr std::uint8_t kNpDestroy = 1u; // stale handle OR fat-box separated
        inline constexpr std::uint8_t kNpStarted = 2u; // false->true touching
        inline constexpr std::uint8_t kNpStopped = 4u; // true->false touching

        // ----------------------------------------------------------------
        // Contact: one persistent contact per solver-relevant overlapping
        // fixture-pair. Survives across steps; the manifold is recomputed at
        // most once per step (a future UpdateContacts). Phase 3 carries only
        // what the solver feed needs (events stay in ContactManager).
        // ----------------------------------------------------------------
        struct Contact
        {
            FixtureHandle a{};                      // canonical: A's body is dynamic (oriented at create)
            FixtureHandle b{};
            std::uint32_t bodyA = kInvalidSlot;     // cached body SLOT for A (kInvalidSlot until the caller fills it)
            std::uint32_t bodyB = kInvalidSlot;     // cached body SLOT for B (kInvalidSlot for a tile span / until filled)
            bool          bIsBody = true; // false => B is a tile-span virtual fixture
            Manifold      manifold{};     // last computed (pointCount 0 => not touching)
            bool          touching = false;
            // Transient narrowphase MT state-change flags (kNp*); reset each step.
            std::uint8_t  npState = 0;
            // SOLVER vs EVENT relevance (collision-rebuild Phase 4, Task 1). The
            // pool now holds the EVENT union -- sensors + kinematic-kinematic +
            // kinematic-vs-static-body in addition to the solver-relevant pairs.
            // This flag is the OLD create filter: true iff (da||db) && !sensorA &&
            // !sensorB, i.e. dynamic-dynamic / dynamic-kinematic / dynamic-static,
            // non-sensor. EmitContactConstraints emits a ContactConstraint ONLY for
            // solverRelevant contacts, so the solver feed stays byte-identical to
            // Phase 3 even though the pool is now a superset. LIFETIME-INVARIANT:
            // set once at create time, NEVER updated on a HIT/recompute (the
            // body-type/sensor classification of a contact's pair is fixed for its
            // lifetime).
            bool          solverRelevant = false;
            // EVENT relevance (collision-rebuild Phase 4, Task 2). The body-pair
            // event machine (ContactManager) derives Begin/Stay/End ONLY from
            // event-relevant touching contacts. True iff the contact's body-pair is
            // NOT dynamic-vs-static-body: dynamic-vs-static is a SOLVER concern, not
            // a gameplay trigger (the design's explicit exclusion). The only pooled
            // pairs are mover-mover (dynamic/kinematic, sensors included),
            // dynamic-static, and kinematic-static; tile spans live in the
            // transient m_spanContacts scratch, never in m_contactPool, so they
            // never reach event derivation. So eventRelevant is true for
            // mover-mover + kinematic-static, false ONLY for dynamic-static.
            // LIFETIME-INVARIANT like solverRelevant: set once at create, never
            // updated on a HIT.
            bool          eventRelevant = false;
            // PERSISTENT COLOR (collision-rebuild Phase C, Stage 2, Tasks 4-5).
            // The graph color assigned ONCE when this solver-relevant body-body
            // contact is created (ConstraintGraph::AssignContactColor), released back
            // at destroy (ReleaseContactColor). kInvalidColor means uncolored:
            // a sensor / non-solver / span contact, an OVERFLOW contact (no free
            // color), or a fresh/recycled pool slot. CONSUMED by the solver
            // (Task 5): EmitContactConstraints copies this onto the emitted
            // ContactConstraint::color, and SoftStep buckets the emitted constraints
            // by it (the per-step greedy recolor is gone). The persistent coloring is
            // a DIFFERENT but equally-valid color partition than the old per-step
            // greedy one, so the colored Gauss-Seidel solve is an INTENTIONAL
            // re-baseline vs pre-Phase-C main (different floats), NOT bit-identical.
            // The contract that holds is run-twice DETERMINISM + the behavioral
            // [physics] suite (no exact goldens) -- per the engine's
            // re-baseline-numerics-on-purpose rule. Reset to kInvalidColor on every
            // pool-slot recycle (ContactPool::EnsurePair MISS path) so a recycled
            // slot never carries a stale color.
            std::uint8_t  color = kInvalidColor;
        };

        // ----------------------------------------------------------------
        // ContactPool: pooled, deduped by the unordered fixture-slot pair,
        // generation-safe.
        // ----------------------------------------------------------------
        class ContactPool
        {
        public:
            static constexpr std::uint32_t kNone = 0xFFFFFFFFu;

            // Result of EnsurePair: the contact id plus an EXPLICIT created/existing
            // signal. `created == true` means a fresh slot was minted (its body
            // slots are kInvalidSlot -- the caller MUST fill them via Get(id));
            // `created == false` means the existing pair was returned untouched.
            // This is reliable under churn, where the old "Count grew" heuristic
            // (a destroy+create in the same step leaves Count unchanged) is not.
            struct EnsureResult
            {
                std::uint32_t id      = kNone;
                bool          created = false;
            };

            // Create-if-absent. On a MISS returns { newId, true } with the slot
            // freshly reset (body slots kInvalidSlot); the caller fills body
            // slots/orientation via Get(id). On a HIT returns { existingId, false }
            // WITHOUT touching the existing contact. Order-independent:
            // EnsurePair(a,b) and EnsurePair(b,a) resolve to the same id.
            EnsureResult EnsurePair(FixtureHandle a, FixtureHandle b);

            // kNone if absent.
            std::uint32_t Find(FixtureHandle a, FixtureHandle b) const;

            void Destroy(std::uint32_t id);

            Contact& Get(std::uint32_t id)
            {
                assert(id < m_pool.size() && m_alive[id] && "ContactPool::Get on a dead/invalid id");
                return m_pool[id];
            }
            const Contact& Get(std::uint32_t id) const
            {
                assert(id < m_pool.size() && m_alive[id] && "ContactPool::Get on a dead/invalid id");
                return m_pool[id];
            }
            std::size_t Count() const { return m_live; }
            // True iff id is a live (non-recycled) contact slot. Used by the
            // per-body adjacency validator (DebugValidateBodyContacts) to guard
            // against stale ids in m_bodyContacts without asserting.
            [[nodiscard]] bool Alive(std::uint32_t id) const noexcept
            { return id < m_alive.size() && m_alive[id] != 0; }
            // Id-slot capacity (max valid id + 1). Bit sets keyed on contact id are
            // sized to this (Box2D's contactIdCapacity). NOT the live count.
            std::size_t Capacity() const noexcept { return m_pool.size(); }

            // Visit live contacts in ascending id order (deterministic). A const
            // overload (Phase 3 Task 3 walks the pool read-only).
            void ForEach(Mosaic::FunctionRef<void(std::uint32_t id, Contact&)> fn);
            void ForEach(Mosaic::FunctionRef<void(std::uint32_t id, const Contact&)> fn) const;
            void Clear();

        private:
            // Mix each handle's index + generation into a 32-bit half, then
            // pack (min<<32)|max so a recycled slot (new gen) never aliases and
            // the pair is order-independent. The per-handle mix is a NON-cryptographic
            // index ^ (generation * prime) scramble, so a full 64-bit key collision
            // is theoretically possible but negligible for realistic slot/generation
            // ranges (revisit if either field approaches 2^32).
            static std::uint64_t Key(FixtureHandle a, FixtureHandle b);

            std::vector<Contact>       m_pool;   // indexed by id; holes reused via m_free
            std::vector<std::uint8_t>  m_alive;  // 1 if id is a live contact
            std::vector<std::uint32_t> m_free;   // recycled ids
            std::unordered_map<std::uint64_t, std::uint32_t> m_index; // Key -> id
            std::size_t m_live = 0;
        };

    } // namespace Physics
} // namespace Manifold2D
