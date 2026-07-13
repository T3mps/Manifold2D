#pragma once

// ConstraintGraph: owns the CONTACT SUBSYSTEM -- the persistent fixture-pair
// contact pool, the incremental solver contact-coloring (per-body color masks +
// per-color contact-id lists), and the per-Step narrowphase/create/emit driving
// (UpdateContacts / EmitContactConstraints + their MT scratch). Extracted from
// PhysicsWorld as a contained collaborator (PhysicsWorld decomposition step 2,
// 2026-07-09), mirroring Box2D v3's contact subsystem boundary: contact.c +
// constraint_graph.c + the b2Collide driver in world.c form one cohesive unit
// whose only outward seams are island link/unlink, event push, and wake. Distinct
// from:
//   * IslandManager -- island TOPOLOGY + the sleep pass (decomp step 1); the graph
//     reaches it only through PhysicsWorld's existing island forwarders + the
//     public IslandManager API (Attach/DetachContactAdjacency), never directly.
//   * ContactManager -- the begin/stay/end EVENT machine; a downstream consumer of
//     the graph's derived touched-pair list, never sees fixtures/manifolds/colors.
//   * The solver -- consumes the EMITTED ContactConstraint array only (bucketed by
//     the constraint's own `color` field); it never touches the pool or the
//     coloring state and never learns the graph exists (ISolver seam preserved).
//
// STAYS WORLD-OWNED (hot hand-off buffers + shared hot SoA -- resist wrapping):
//   * m_contactConstraints -- the emitted solver feed; SolverContext captures its
//     raw .data() pointer ONCE per Step (the contact-side twin of the
//     AwakeIndexData() handoff). The graph fills a caller-provided vector.
//   * m_touchedEventPairs -- the stage-output buffer handed to ContactManager
//     (CollectTouchedEventPairs fills it).
//   * The body/fixture SoA, awake/sleep state + awake-set mechanism, broadphase,
//     and executor -- reached through PhysicsWorld& w under friendship.
//
// DELEGATION: like IslandManager, ConstraintGraph owns its state outright and
// takes PhysicsWorld& per method; PhysicsWorld befriends it (the SAME trust
// boundary this code had when it lived inside the world -- UpdateContacts /
// TryCreateContact / EmitContactConstraints need intimate world SoA). Friendship
// rays stay world -> collaborator only: IslandManager reads the pool through the
// public const Pool() accessor, not friendship.
//
// DETERMINISM (frozen contracts -- see pwdecomp2 surveys 2/3): the create order ->
// EnsurePair LIFO id recycling -> AssignContactColor lowest-free search ->
// ascending-color Gauss-Seidel chain; the MixContactId warm-start keying; the
// wake-before-create order (WakeMoverPair); the merge drain at the END of
// UpdateContacts. All moved verbatim; byte-identity gated.
//
// PRESENTATION-FREE + C++23-clean: std + sibling Physics headers only.
// namespace Manifold2D::Physics, Core style.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <Manifold2D/Physics/PhysicsTypes.hpp>          // Real / Vec2 / BodyHandle / Aabb2
#include <Manifold2D/Physics/Contact.hpp>               // Contact + ContactPool + FixtureHandle
#include <Manifold2D/Physics/Broadphase/Broadphase.hpp> // BroadphasePair
#include <Manifold2D/Physics/Solver/Solver.hpp>         // ContactConstraint
#include <Manifold2D/Core/BitSet.hpp>                   // per-worker npState BitSets

namespace Manifold2D
{
    namespace Physics
    {
        class PhysicsWorld; // ConstraintGraph reaches the world SoA (befriended).

        class ConstraintGraph
        {
        public:
            // Sizes the per-color contact-id lists once to kColorCount (overflow
            // contacts are not listed -- they carry color == kInvalidColor and
            // never enter m_colorContacts). The per-body mask grows via Grow.
            ConstraintGraph();

            // ---- the per-Step drivers (PhysicsWorld::Step stages 2/3b/6) -----
            //
            // UpdateContacts(w, dt): the ONE-PASS persistent-contact update
            // (Step stage 2). The SOLE narrowphase for the solver feed. Three
            // phases: (1) CREATE a Contact for every fixture-pair in the EVENT
            // union (mover<->mover from the parallel broadphase pair-find;
            // mover<->static-BODY + tile spans from the parallel per-awake-body
            // detect; kinematic<->static serial), with WakeMoverPair's
            // wake-before-create rule; (2) UPDATE + DESTROY: gather -> parallel
            // UpdateOneContact into per-worker BitSets -> serial ascending-id
            // apply (destroy / queue merge / mark split); (3) drain the queued
            // island merges in canonical order (WakeIsland-if-mixed +
            // MergeIslands via w). Reaches the broadphase, executor, island
            // forwarders, and body/fixture SoA through w.
            void UpdateContacts(PhysicsWorld& w, Real dt);

            // EmitContactConstraints(w, out): the persistent solver feed. Walk
            // the pool (ascending id, solverRelevant only) AND the transient
            // tile-span scratch, emit a ContactConstraint per touching+awake
            // contact (mirroring the retired GenerateContacts emit field-for-
            // field: materials sqrt/max, COM anchors, baseSeparation, id =
            // MixContactId, warm-start seed), then canonical-sort by
            // (bodyA, bodyB, fixA, fixB). Writes ONLY `out` + the mutable emit
            // scratch (stays const for the DebugEmitPoolConstraints oracle).
            void EmitContactConstraints(const PhysicsWorld& w,
                                        std::vector<ContactConstraint>& out) const;

            // WritebackImpulses(ccs): Step stage 3b -- write the solver's
            // converged per-point impulses back onto each source Contact's
            // manifold (warm-start-on-Contact). Spans (kNoContact) skipped.
            void WritebackImpulses(const std::vector<ContactConstraint>& ccs);

            // CollectTouchedEventPairs(out): Step stage 6 -- derive the deduped,
            // sorted event-relevant EXACTLY-OVERLAPPING body-pairs from the pool
            // (events-as-byproduct). The world hands `out` to ContactManager.
            void CollectTouchedEventPairs(std::vector<BroadphasePair>& out) const;

            // ---- lifecycle-seam contact destruction (world drives these) -----
            // Destroy every pooled contact referencing the removed fixture/body
            // the moment it is removed (DropFixture / RemoveBody seams); marks
            // split candidates + wakes the affected islands via w.
            void DestroyContactsForFixture(PhysicsWorld& w, std::uint32_t fixtureSlot);
            void DestroyContactsForBody(PhysicsWorld& w, std::uint32_t bodySlot);

            // The canonical pooled-contact teardown: detach island adjacency
            // (via w.m_islandMgr) + release the persistent color (while c still
            // holds it) + pool.Destroy. Order is FROZEN.
            void ReleaseAndDestroyContact(PhysicsWorld& w, std::uint32_t id,
                                          const Contact& c) noexcept;

            // ---- persistent incremental contact coloring (Phase C, Task 4;
            //      moved here in decomp step 2 Task 2) -------------------------
            //
            // A solver-relevant body-body contact is assigned a graph color ONCE
            // at create (AssignContactColor, from TryCreateContact) and releases
            // it at destroy (ReleaseContactColor, from ReleaseAndDestroyContact)
            // -- the assign-at-create / release-at-destroy replacement for the
            // per-step greedy recolor. The invariant: no two same-color contacts
            // share a DYNAMIC body (a static/kinematic endpoint never constrains
            // coloring, mirroring ColorConstraints' aDyn/bDyn rule). The solver
            // consumes the coloring only via the emitted constraint's own `color`
            // field (bucketed into SoftStep's m_colorRefs); it never reads this
            // state -- m_colorContacts is lifecycle/validation bookkeeping.

            // Assign the lowest free color to a NEW solver-relevant body-body
            // contact `id` between body slots `a`/`b`. `aDyn`/`bDyn` mark which
            // endpoints are dynamic (only dynamic endpoints constrain coloring +
            // occupy a per-body color bit). If no color in [0, kColorCount) is
            // free for both dynamic endpoints, the contact spills to overflow
            // (color == kInvalidColor). Caller passes the ORIENTED slots/dyn
            // flags computed in TryCreateContact.
            void AssignContactColor(std::uint32_t id,
                                    std::uint32_t a, std::uint32_t b,
                                    bool aDyn, bool bDyn);

            // Release contact `id`'s color back to its dynamic endpoints. No-op
            // for an uncolored (kInvalidColor) contact. Recomputes dyn-ness from
            // the contact's cached body slots via w (a body's type is fixed for
            // its life), so this is exact: the coloring invariant guarantees a
            // body has at most one contact per color, so clearing the bit on
            // destroy frees it cleanly.
            void ReleaseContactColor(PhysicsWorld& w, std::uint32_t id);

            // The persistent color of contact `id` (kInvalidColor if uncolored).
            // Read-only probe; not used by the Step path.
            [[nodiscard]] std::uint8_t ContactColorOf(std::uint32_t id) const;

            // Oracle: walk the live coloring and prove the invariant -- no
            // DYNAMIC body appears twice in one color, every listed contact is
            // alive and tagged with its color, and the per-body mask matches the
            // lists bit-for-bit. Used by the [phasec] coloring-validity test.
            [[nodiscard]] bool ValidatePersistentColoring(const PhysicsWorld& w) const;

            // Read-only probe: the total number of COLORED contacts (the sum of
            // m_colorContacts[k].size() over all colors).
            [[nodiscard]] std::size_t ColoredContactCount() const noexcept;

            // ---- read seams -------------------------------------------------
            // Read-only pool access for IslandManager (SplitIsland's edge walk +
            // DebugValidateBodyContacts) and any const inspection. BY REFERENCE
            // (a by-value slip would be a perf cliff in SplitIsland). The
            // non-const pool stays private to the graph.
            [[nodiscard]] const ContactPool& Pool() const noexcept { return m_contactPool; }

            // Live pooled-contact count (test/inspection hook backing).
            [[nodiscard]] std::size_t DebugContactCount() const noexcept
            {
                return m_contactPool.Count();
            }
            // True iff the pool holds a body-body contact matching the unordered
            // handle pair (test hook backing; stale/dead handles return false).
            [[nodiscard]] bool DebugHasContact(const PhysicsWorld& w,
                                               BodyHandle a, BodyHandle b) const;

            // ---- world-lifecycle seams (PhysicsWorld drives these) -----------
            // Grow the per-body color-mask column to `next` (EnsureCapacity seam).
            // A fresh/recycled slot starts with NO colors occupied.
            void Grow(std::uint32_t next);
            // Debug probe for the RemoveBody leak-detector assert: true iff the
            // slot's color mask is 0 (every contact referencing it released its
            // bits). Out-of-range counts as clear (mirrors the original guard).
            [[nodiscard]] bool DebugBodyMaskClear(std::uint32_t slot) const noexcept
            {
                return slot >= m_bodyColorMask.size() || m_bodyColorMask[slot] == 0u;
            }

        private:
            // CREATE-pass helper: ensure a Contact for the fixture-pair (fa, fb),
            // applying the EXACT filters + orientation rule (A's body dynamic;
            // both-dynamic -> lower body slot is A; bodies + fixtures swap together).
            // Fills the body slots ONLY on a fresh create; assigns the persistent
            // color + attaches the island edge (via w.m_islandMgr).
            void TryCreateContact(PhysicsWorld& w, std::uint32_t fa, std::uint32_t fb);

            // CREATE-pass helper (Task 4): wake-on-contact for a mover<->mover
            // fixture-pair (the retired GenerateContacts wake rule). Mutates
            // w.m_awake / w.m_sleepTimer + the awake-set/island wake mechanisms
            // through w. MUST run before the manifold update pass.
            void WakeMoverPair(PhysicsWorld& w, std::uint32_t fa, std::uint32_t fb);

            // Recompute one contact's manifold + classify its state change for the
            // MT serial tail. Reads stable body transforms + the contact; writes ONLY
            // c.manifold / c.touching / c.npState (no pool/color/island mutation).
            void UpdateOneContact(const PhysicsWorld& w, std::uint32_t id, Contact& c,
                                  Real moveDt, Real threshSq) noexcept;

            // True iff both bodies are asleep (a static/kinematic body counts as
            // asleep -- it never wakes a recompute on its own).
            [[nodiscard]] bool BothAsleep(const PhysicsWorld& w,
                                          const Contact& c) const noexcept;

            // True iff the two fixtures' FAT broadphase boxes (optionally grown by
            // an extra speculative margin) still overlap. The contact owns its
            // destruction when this is false.
            [[nodiscard]] bool FatBoxesOverlap(const PhysicsWorld& w, const Contact& c,
                                               Real extraMargin = Real(0)) const noexcept;

            // ---- persistent contact pool ------------------------------------
            // The durable fixture-pair contact pool, POPULATED each Step by
            // UpdateContacts and CONSUMED by EmitContactConstraints (the solver
            // feed). Survives across steps; the create pass dedups, the update
            // pass recomputes manifolds + destroys dead pairs. m_cpPairs is the
            // move-buffer scratch for the mover<->mover create pass.
            ContactPool                 m_contactPool;
            std::vector<BroadphasePair> m_cpPairs;

            // Step scratch (zero steady-state alloc -- clear() keeps capacity).
            // m_pendingMerges: dynamic-dynamic touch-BEGIN body-pairs collected in
            // the UpdateContacts pass, sorted canonically then applied (determinism).
            // Filled AND drained entirely inside UpdateContacts (rides with it).
            std::vector<BroadphasePair> m_pendingMerges;

            // Narrowphase-MT scratch (gather -> parallel collide+flag -> serial apply).
            // m_npContacts: the gathered stable live-contact id list (Box2D's
            // contactSims gather). m_npStateBits: one BitSet per worker, each Resize'd
            // to the pool id capacity per step (Box2D's per-worker contactStateBitSet).
            std::vector<std::uint32_t>  m_npContacts;
            std::vector<Manifold2D::BitSet> m_npStateBits;

            // ---- broadphase per-worker scratch (Phase D2, Task 3) ----------
            // Reused across steps (resize-to-WorkerCount on growth; clear per step).
            //   m_bpMovedScratch  : snapshot of moved proxy ids from EvictTouchedAndCollectMoved.
            //   m_bpFindScratch[w]: per-worker canonical key accumulator for QueryProxyPairs.
            //   m_bpStackScratch[w]: per-worker descent stack for QueryProxyPairs.
            // Their ONLY driver is UpdateContacts' stage-1a parallel pair-find.
            static constexpr std::size_t kBroadphaseGrain = 64;
            // Grain for the create-phase parallel detect (stage-2b ParallelFor).
            // Below kCreateGrain awake bodies the ParallelFor degrades to serial on
            // worker 0 -> byte-identical to the pre-MT path with zero overhead.
            static constexpr std::size_t kCreateGrain      = 16;
            std::vector<std::uint32_t>              m_bpMovedScratch;
            std::vector<std::vector<std::uint64_t>> m_bpFindScratch;   // per-worker key buffers
            std::vector<std::vector<std::uint32_t>> m_bpStackScratch;  // per-worker descent stacks

            // Transient TILE-SPAN contacts: virtual/transient fixtures (a merged
            // run of solid cells, no fixture slot), NOT pooled; UpdateContacts
            // refills this per-Step scratch. The span's geometric center is
            // stashed in a PARALLEL vector (m_spanCenters) at the same index
            // (Contact has no span-center field).
            std::vector<Contact> m_spanContacts;
            std::vector<Vec2>    m_spanCenters; // parallel to m_spanContacts

            // ---- Create-phase MT: deferred new-pair records (narrowphase-MT G2) --
            // The detect pass emits one record per fixture-pair candidate; the
            // serial tail replays TryCreateContact in awakeIndex-ascending order
            // -- reproducing the serial ForEachAwake create order exactly, so the
            // order-dependent AssignContactColor + the EnsurePair id allocation
            // are byte-identical. awakeIndex = the index into m_awakeBodies (NOT
            // the body slot), which is the ForEachAwake visit order.
            struct NewPairRecord { std::uint32_t awakeIndex; std::uint32_t fiA; std::uint32_t fiB; };
            std::vector<NewPairRecord> m_newPairs;

            // Create-phase MT per-worker scratch (sized to WorkerCount() each step,
            // grow-only). Each worker uses ONLY its own [w] entry -> contention-free.
            // SpanEntry tags each transient span contact with the awakeIndex of the
            // body that produced it, so the serial tail can stable_sort back to
            // ForEachAwake order.
            struct SpanEntry { std::uint32_t awakeIndex; Contact c; Vec2 center; };
            std::vector<std::vector<Aabb2>>         m_genSpansW;    // per-worker StaticCandidates spans out
            std::vector<std::vector<std::uint32_t>> m_genStaticsW;  // per-worker statics out
            std::vector<std::vector<std::uint32_t>> m_gridScratchW; // per-worker QueryAABB scratch
            std::vector<std::vector<SpanEntry>>     m_spanEntriesW; // per-worker span contacts (awakeIndex-tagged)
            std::vector<std::vector<NewPairRecord>> m_newPairsW;    // per-worker new-pair records
            // Serial span-merge buffer: the per-worker m_spanEntriesW are concatenated
            // here + stable_sorted by awakeIndex before being appended to m_spanContacts.
            std::vector<SpanEntry>                  m_allSpans;

            // ---- EmitContactConstraints sort scratch --------------------------
            // The canonical-sort scratch consumed by EmitContactConstraints every
            // Step (parallel sort key + index permutation + permuted staging).
            // `mutable` because EmitContactConstraints stays `const` (the oracle
            // DebugEmitPoolConstraints hook is const) -- mutable cache scratch
            // behind a const read-only API is idiomatic. clear() keeps capacity.
            struct EmitSortKey
            {
                std::uint32_t bodyA, bodyB, fixA, fixB;
            };
            mutable std::vector<EmitSortKey>       m_emitKeys;
            mutable std::vector<std::size_t>       m_emitOrder;
            mutable std::vector<ContactConstraint> m_emitSorted;

            // m_bodyColorMask[slot] is a 12-bit (one bit per color, kColorCount<32)
            // occupancy mask over the colors that body slot currently has a
            // contact in. Keyed by WORLD body slot; a removed/recycled slot resets
            // to 0 (Grow defaults new slots to 0; the RemoveBody leak-detector
            // asserts a removed body left mask 0). m_colorContacts[k] is the list
            // of contact ids assigned to color k (sized kColorCount in the ctor).
            // Maintained at create/destroy; the mask GATES AssignContactColor's
            // lowest-free search. The persistent coloring is a DIFFERENT but
            // equally-valid color partition than the old per-step greedy one, so
            // the colored Gauss-Seidel solve is an INTENTIONAL re-baseline vs
            // pre-Phase-C main (different floats), NOT bit-identical. The contract
            // is run-twice DETERMINISM + the behavioral [physics] suite (no exact
            // goldens) -- per the engine's re-baseline-numerics-on-purpose rule.
            std::vector<std::uint32_t>              m_bodyColorMask;
            std::vector<std::vector<std::uint32_t>> m_colorContacts;
        };
    } // namespace Physics
} // namespace Manifold2D
