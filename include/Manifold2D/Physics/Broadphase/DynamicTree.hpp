#pragma once

// DynamicTree: dynamic AABB tree broadphase (M6, Task P1.6).
//
// PORT + MODERNIZE of Client/src/physics/AABBTree.lua. This is the DEFAULT
// mover broadphase (MODERNIZE: the Lua engine defaulted to SpatialHash).
//
// Faithful ports of the Lua algorithm:
//   * Fat-box leaves: each leaf stores a TIGHT box (the body's real AABB) and
//     a FAT box (tight grown by MARGIN). Insertion / queries descend on the
//     FAT box; narrowing (queryAABB / pairs) tests the TIGHT box so results
//     match SpatialHash / SweepAndPrune exactly.
//   * _insertLeaf descends by least combined-perimeter growth (the Lua's
//     "perim" -- a 2D SAH analogue), splitting the chosen leaf with a new
//     internal node.
//   * refit walks parents up, recomputing the union box.
//   * _removeLeaf promotes the sibling.
//   * update fast-path: if the new tight box still fits inside the stored FAT
//     box, only the tight box is updated (no reinsert) -- the Lua's coherent-
//     motion trick. Otherwise remove + reinsert with a fresh fat box.
//   * NO tree rotations (the Lua skips them; adequate at our scales).
//   * MARGIN ported as the named kMargin constant (value now Box2D v3
//     B2_AABB_MARGIN = 0.05, MKS; the Lua's px-era value was 8).
//
// MODERNIZE: nodes are POOLED in an index-based free-list (the Lua used GC
// tables). Node references are pool indices, not pointers, so there is zero
// per-insert heap traffic in steady state (the pool only `new`s to grow) and
// the structure is determinism-friendly (indices are stable + comparable).
// Query scratch (an explicit descent stack) and the pairs/query dedup buffers
// are pooled members reused across calls.
//
// Phase 2 Task 4 — INCREMENTAL PAIR SET (move-buffer):
//   UpdatePairs() maintains a persistent tight-overlap pair set (m_pairSet)
//   incrementally.  Every Update() call marks the proxy in m_moved (including
//   within-fat-box moves, where only the tight box changes).  Every Remove()
//   call moves the id to m_removed.  UpdatePairs() then:
//     1. Drops all cached pairs that touch any moved/removed proxy.
//     2. Re-queries the tree for each live moved proxy (FAT-box descent,
//        tight-box filter) and inserts fresh pairs into m_pairSet.
//     3. Emits the full m_pairSet sorted.
//   The incremental output is byte-identical to Pairs() (oracle-gated).
//   NOT yet consumed by the solver -- the existing Pairs() call site is
//   unchanged.  This is a pure add.
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.

#include <cstdint>
#include <unordered_set>
#include <vector>

#include <Mosaic/FunctionRef.hpp>

#include <Manifold2D/Physics/Broadphase/Broadphase.hpp>
#include <Manifold2D/Physics/PhysicsTypes.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        class DynamicTree final : public IBroadphase
        {
        public:
            // Box2D v3 B2_AABB_MARGIN (constants.h:44): 0.05 m fat-AABB margin.
            // (v3.1.1 value -- older Box2D used 0.1.)
            static constexpr Real kMargin = Real(0.05);

            DynamicTree() = default;

            void Update(std::uint32_t id, const Aabb2& box) override;
            void Remove(std::uint32_t id) override;
            int  QueryAABB(const Aabb2& box,
                           std::vector<std::uint32_t>& out) const override;
            int  Pairs(std::vector<BroadphasePair>& out) const override;

            // Incremental pair maintenance: move-buffer path (Phase 2, Task 4).
            // Maintains m_pairSet in sync with each Update/Remove, then emits
            // sorted. Oracle-verified == Pairs() after every mutation.
            // Rewritten in Phase D2 Task 2 as a serial wrapper over the three seams.
            int  UpdatePairs(std::vector<BroadphasePair>& out) override;

            // ---- Phase D2 Task 2 seam overrides --------------------------------
            // See IBroadphase for the contracts. DynamicTree provides the incremental
            // implementations; the serial wrapper UpdatePairs calls them in order.
            void EvictTouchedAndCollectMoved(std::vector<std::uint32_t>& movedOut) override;
            void QueryProxyPairs(std::uint32_t id,
                                 std::vector<std::uint32_t>& stack,
                                 std::vector<std::uint64_t>& out) const override;
            int  MergeAndEmit(std::span<const std::vector<std::uint64_t>> perWorker,
                              std::vector<BroadphasePair>& out) override;

            // ---- read-only debug-visualization accessor (Slice A) ------------
            //
            // Visit each LIVE leaf, passing (body/fixture id, tight box, fat box).
            // Iterates m_leafOfId (the id->leaf map), skipping kNull entries, so
            // only ACTUALLY-INDEXED proxies are yielded (not stale pool slots).
            // Read-only; iteration order is for DISPLAY only -- no sim path may
            // depend on it. The fat box always contains the tight box (the
            // MARGIN-grown invariant) -- the overlay draws both.
            void ForEachLeaf(
                Mosaic::FunctionRef<void(std::uint32_t id,
                                 const Aabb2& tight,
                                 const Aabb2& fat)> fn) const;

            // O(1) fat-AABB lookup by proxy/fixture id. Returns false if id has
            // no live leaf. Mirrors ForEachLeaf's node.fat, but resolves the
            // leaf in O(1) via m_leafOfId instead of scanning every proxy.
            [[nodiscard]] bool TryGetFatBox(std::uint32_t id, Aabb2& out) const;

        private:
            // Sentinel for "no node" (the Lua used nil parents/children).
            static constexpr std::uint32_t kNull = 0xFFFFFFFFu;

            // A pooled tree node. Internal nodes have two children; leaves have
            // kNull children, a body id, and a tight box. The `fat` box is the
            // node's bounds: for a leaf it is the tight box grown by MARGIN;
            // for an internal node it is the union of its children (the Lua's
            // x0..y1). Slots on the free-list chain via `parent` (== next).
            struct Node
            {
                Aabb2         fat{};                 // descent / query bounds
                Aabb2         tight{};               // leaf only: real body box
                std::uint32_t parent = kNull;        // also free-list "next"
                std::uint32_t left   = kNull;        // kNull => leaf
                std::uint32_t right  = kNull;
                std::uint32_t id     = 0;            // leaf only: body id

                [[nodiscard]] bool IsLeaf() const noexcept
                {
                    return left == kNull;
                }
            };

            // Allocate a node slot from the free-list (grows the pool only when
            // empty). Returns the slot index. Caller fills fields.
            std::uint32_t AllocNode();
            // Return a node slot to the free-list.
            void          FreeNode(std::uint32_t idx);

            // Lua _insertLeaf / _removeLeaf (operate on a leaf slot index).
            void InsertLeaf(std::uint32_t leaf);
            void RemoveLeaf(std::uint32_t leaf);
            // Lua refit: recompute union boxes from n up to the root.
            void Refit(std::uint32_t n);

            std::vector<Node>          m_nodes;     // node pool
            std::uint32_t              m_root      = kNull;
            std::uint32_t              m_freeList  = kNull;
            // id -> leaf slot index (the Lua's `leaves` table). Sized lazily;
            // entry kNull means "no such body".
            std::vector<std::uint32_t> m_leafOfId;

            // Reused scratch (no per-call heap).
            // m_stack: descent scratch for the SERIAL const Pairs() and the
            // serial UpdatePairs wrapper. QueryAABB does NOT use it -- QueryAABB
            // uses a thread_local stack instead, so it stays re-entrant when the
            // parallel create-phase queries the static tree from many workers.
            mutable std::vector<std::uint32_t> m_stack;

            // Move-buffer state (Phase 2, Task 4 -- incremental pair set).
            //
            // m_moved:   ids that were Update()'d since the last UpdatePairs().
            //            Marked on EVERY Update call (including within-fat-box
            //            moves where only tight box changes -- those still affect
            //            tight-overlap membership).
            // m_removed: ids that were Remove()'d since the last UpdatePairs().
            //            Pairs touching removed ids must be evicted.
            // m_pairSet: canonical persistent tight-overlap pair set.
            //            Key = (lo << 32) | hi, lo = min(a,b), hi = max(a,b).
            //            UpdatePairs() keeps this in sync and emits it sorted.
            std::unordered_set<std::uint32_t> m_moved;
            std::unordered_set<std::uint32_t> m_removed;
            std::unordered_set<std::uint64_t> m_pairSet;
            std::vector<std::uint64_t>        m_toErase; // UpdatePairs step-1 evict scratch (pooled; capacity preserved)

            // Reuse buffers for the serial UpdatePairs wrapper (Phase D2 Task 2).
            // m_movedSerial: snapshot of m_moved produced by EvictTouchedAndCollectMoved.
            // m_findSerial:  single-worker key accumulator fed to QueryProxyPairs.
            std::vector<std::uint32_t>        m_movedSerial;
            std::vector<std::uint64_t>        m_findSerial;

            // Map an id to its leaf slot (kNull if absent).
            [[nodiscard]] std::uint32_t LeafOf(std::uint32_t id) const noexcept;
            void SetLeafOf(std::uint32_t id, std::uint32_t leaf);
        };

    } // namespace Physics
} // namespace Manifold2D
