#pragma once

// IslandManager: owns island TOPOLOGY -- the persistent island registry (per-body
// island id + the id-indexed record pool + free list), merge/split, and the
// SplitIsland connected-component rebuild. Extracted from PhysicsWorld as a
// contained collaborator (PhysicsWorld decomposition step 1, 2026-07-09),
// mirroring Box2D v3's b2Island (connectivity topology + the sleep trigger) as
// distinct from:
//   * b2ConstraintGraph -- the coloring / solve partition, owned (with the
//     contact pool) by the ConstraintGraph collaborator (decomp step 2), and
//   * b2SolverSet -- the hot awake/kinematic sets, which STAY world-owned as flat
//     SoA: m_awakeIndex[slot] is read per-body every substep (and handed as a raw
//     pointer into the SIMD packer), and the awake-set element ORDER feeds
//     persistent contact coloring -- a determinism tripwire. "Behavior over
//     shared hot SoA -> keep flat" (the data-oriented design constraint).
//
// DELEGATION: like ContactManager, IslandManager owns its state outright and takes
// PhysicsWorld& per method to reach the flat body/fixture/contact SoA + the
// awake-set MECHANISM (AddToAwakeSet) it does not own. Unlike ContactManager's
// narrow public "port seam", SplitIsland/WakeIsland need intimate world state (the
// contact pool + joint edges + the awake flag/timer), so PhysicsWorld befriends
// IslandManager -- the SAME trust boundary this code had when it lived inside the
// world. This keeps the extraction a verbatim, byte-identity-preserving move
// rather than a new public accessor surface.
//
// Split-linkage adjacency (m_bodyContacts -- the per-body dyn-dyn contact edge
// lists SplitIsland walks) is OWNED here as of decomp step 1 Task 3; the world
// drives it through the attach/detach/clear/grow seams below. The per-Step sleep
// pass (UpdateSleep, formerly the Island:: free function) is OWNED here as of
// decomp step 1 Task 4 -- it reaches body-slot state + the awake-set removal
// MECHANISM (RemoveFromAwakeSet) through PhysicsWorld&. The awake/kinematic sets
// + the m_awake flag stay world-owned (hot SoA, read by the solver).
//
// PRESENTATION-FREE + C++20-clean: std + sibling Physics headers only.
// namespace Manifold2D::Physics, Core style.

#include <cstdint>
#include <vector>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Island.hpp> // Island registry struct + constants (Phase A)
#include <Manifold2D/Core/FunctionRef.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        class PhysicsWorld; // IslandManager reaches the world SoA (befriended).
        struct Contact;     // detach/validate seams take a Contact by ref.

        class IslandManager
        {
        public:
            // ---- registry query (inline; all cold / seam-time, NONE in the substep
            //      solve -- see pwdecomp survey 2, Part B5) --------------------------
            // m_islandId[slot] (or kInvalidIsland). Inline -> zero call cost.
            [[nodiscard]] std::uint32_t IslandOf(std::uint32_t slot) const noexcept
            {
                return slot < m_islandId.size() ? m_islandId[slot] : Island::kInvalidIsland;
            }
            // Per-body island-id column size (== body-slot capacity). Bounds guard.
            [[nodiscard]] std::size_t IslandIdCount() const noexcept { return m_islandId.size(); }
            // Island ROOT of slot i (debug/inspection, Phase A): the persistent id IS
            // the root; a non-member returns a high-bit-tagged slot (never collides).
            [[nodiscard]] std::uint32_t IslandRootOf(std::uint32_t i) const noexcept;

            // Visit each LIVE island's member-slot list, ascending island-id
            // (deterministic). A live island has a non-empty member list; freed ids
            // (empty) are skipped. Const callback (the sleep pass mutates bodies
            // through the world's slot accessors, not the island record).
            void ForEachIsland(
                FunctionRef<void(const std::vector<std::uint32_t>&)> fn) const
            {
                for (const Island::Island& isl : m_islands)
                {
                    if (!isl.bodies.empty())
                    {
                        fn(isl.bodies);
                    }
                }
            }

            // ---- registry lifecycle (Phase A) -------------------------------
            // Mint or reuse an island id (empty members, not a split candidate).
            std::uint32_t AllocIsland();
            // Return an island id to the free list (clears members + flag).
            void          FreeIsland(std::uint32_t id) noexcept;
            // Weighted union: relabel the smaller island's members into the larger,
            // free the smaller, return the survivor id. Pass two DISTINCT live ids.
            std::uint32_t MergeIslands(std::uint32_t idA, std::uint32_t idB);
            // Flag an island for the deferred split pass (no-op for kInvalidIsland).
            void          MarkSplitCandidate(std::uint32_t islandId) noexcept;
            // Rebuild one candidate island into 1+ connected components (fresh local
            // UF over its members' current touching pool contacts + joint edges);
            // clears the flag. Walks its own m_bodyContacts adjacency + the graph's
            // pool (read-only, w.m_graph.Pool()) + w.m_joints.
            void          SplitIsland(PhysicsWorld& w, std::uint32_t islandId);
            // Wake every member of the body's island (set awake flag + reset timer +
            // re-add to the world's awake-set).
            void          WakeIsland(PhysicsWorld& w, std::uint32_t slot) noexcept;

            // ---- per-Step sleep pass (decomp step 1 Task 4) -----------------
            // The per-Step island sleep pass (PhysicsWorld::Step stage 5). Iterates
            // the PERSISTENT island registry (members already known -- no per-step
            // union-find rebuild, no O(n^2) global scan): advances each awake dynamic
            // body's idle timer by `dt` using Box2D v3's combined, extent-weighted
            // test (a body is idle when |v| + |w|*maxExtent < its per-body
            // sleepThreshold), then for each island sleeps it AS A UNIT iff every
            // awake-dynamic member is past Island::kSleepTime (clearing awake +
            // zeroing linear & angular velocity + migrating the member OUT of the
            // world's awake-set). Reads body-slot state + the awake-set removal
            // MECHANISM (RemoveFromAwakeSet, which STAYS world-owned hot SoA) through
            // PhysicsWorld& w; iterates the island records it owns via ForEachIsland.
            // Jointed dynamics are NOT pinned awake here (a joint is an island edge,
            // so a jointed construct sleeps as a unit via island membership). No-op
            // when the world has no dynamics. Thresholds + whole-island-unit +
            // exact-freeze are UNCHANGED from the global-UF version this replaces.
            void UpdateSleep(PhysicsWorld& w, Real dt);

            // ---- body-lifecycle seams (PhysicsWorld drives these) -----------
            // Grow the per-body columns to `next` (EnsureCapacity seam).
            void Grow(std::uint32_t next);
            // A new dynamic body is its own 1-body island (AddBody seam).
            void CreateSingletonIsland(std::uint32_t slot);
            // Mark a slot as belonging to no island (static/kinematic; AddBody seam).
            void ClearIsland(std::uint32_t slot) noexcept;
            // Release a destroyed body from its island: swap-erase membership, then
            // free the island if empty or mark it a split candidate (RemoveBody seam).
            void RemoveBodyFromIsland(std::uint32_t slot) noexcept;
            // Collect (clear + fill) the split-candidate island ids, ascending id
            // (Step stage 5 seam; determinism).
            void CollectSplitCandidates(std::vector<std::uint32_t>& out) const;

            // ---- split-linkage adjacency (m_bodyContacts; decomp step 1 Task 3) --
            // The per-body dyn-dyn contact edge lists SplitIsland walks. Maintained
            // ONLY at contact create/destroy + body add/remove; merge/split/sleep/
            // wake never touch it (body slots are stable -- only m_islandId changes).
            // The world drives these seams (it owns the create/destroy/grow points).
            //
            // Grow the per-body edge-list column to `next`. Kept a DISTINCT grow from
            // Grow() because the world sizes this column with its body-aux group
            // (EnsureBodyAuxCapacity), not the main body SoA (EnsureCapacity) -- a
            // 1:1 move of the original resize seam, not a relocation.
            void GrowBodyContacts(std::uint32_t next);
            // Drop a slot's edge list (AddBody recycle + RemoveBody defensive clear).
            void ClearBodyContacts(std::uint32_t slot) noexcept;
            // Record a NEW dyn-dyn solver body contact `id` on BOTH endpoints. The
            // world gates dyn-dyn + solverRelevant at the create site (a sensor
            // dyn-dyn pair fires events but must NOT become an island edge), then
            // calls this to attach the island edge. Mirrors DetachContactAdjacency.
            void AttachContactAdjacency(std::uint32_t a, std::uint32_t b, std::uint32_t id);
            // Remove contact `id` from both endpoints' lists (dyn-dyn only; no-op for
            // non-dyn-dyn or already-absent). Reads c BEFORE any pool Destroy frees it.
            void DetachContactAdjacency(PhysicsWorld& w, std::uint32_t id, const Contact& c) noexcept;
            // Test invariant: true iff m_bodyContacts exactly mirrors the live dyn-dyn
            // body contacts (every such contact's id appears once in BOTH endpoints'
            // lists; every id in every list is a live dyn-dyn body contact incident to
            // that slot, no duplicates).
            [[nodiscard]] bool DebugValidateBodyContacts(const PhysicsWorld& w) const;

        private:
            // SplitIsland scratch sentinel: member body slot -> local DSU index.
            static constexpr std::uint32_t kSplitLocalNone = 0xFFFFFFFFu;

            // m_islandId[slot] is the body's persistent island id (id == root);
            // Island::kInvalidIsland for Static/Kinematic or an un-assigned slot.
            // m_islands is the id-indexed record pool; freed ids recycle via
            // m_islandFree. Maintained incrementally at the lifecycle seams above.
            std::vector<std::uint32_t>  m_islandId;        // per-body island id
            std::vector<Island::Island> m_islands;         // id-indexed record pool
            std::vector<std::uint32_t>  m_islandFree;      // recycled island ids
            // SplitIsland scratch: member body slot -> local DSU index, O(1).
            // All-sentinel; SplitIsland writes only its members then resets them.
            std::vector<std::uint32_t>  m_splitLocalIndex;

            // Box2D b2ContactEdge analogue used by SplitIsland to walk only an
            // island's own contacts. Holds the POOL IDS of each dynamic body's
            // dyn-dyn body contacts (both endpoints carry the id). Maintained ONLY
            // at contact create + destroy; merge/split/sleep/wake never touch it
            // (body slots are stable -- only m_islandId changes). Mirrors the
            // world's m_bodyFixtures vector-of-vectors lifecycle.
            std::vector<std::vector<std::uint32_t>> m_bodyContacts;
        };
    } // namespace Physics
} // namespace Manifold2D
