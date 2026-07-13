#pragma once

// SpatialHash: uniform-grid broadphase (M6, Task P1.6).
//
// PORT + MODERNIZE of Client/src/physics/SpatialHash.lua. An alternate mover
// broadphase behind IBroadphase (the DEFAULT is DynamicTree).
//
// Faithful ports of the Lua algorithm:
//   * Uniform grid of cellSize (default 1.0 m, MKS). Buckets keyed kx*0x40000 + ky.
//   * Per-id range tracking {kx0,ky0,kx1,ky1}; update skips re-bucketing when
//     the cell range is unchanged (the Lua's update-skip-if-same-range).
//   * remove clears the id from its bucket range exactly.
//   * queryAABB dedups candidate ids via a generation-stamped "seen" scratch.
//
// MODERNIZE (uniform contract -- the key change over the Lua):
//   * The Lua's pairs() returned broad CANDIDATES (shared-bucket pairs); the
//     harness narrowed them afterward. Here pairs() narrows to TRUE TIGHT-BOX
//     OVERLAP and dedups, so it returns the IDENTICAL set as DynamicTree /
//     SweepAndPrune. queryAABB likewise narrows against the stored tight box
//     (the Lua returned all bucket members). Output is SORTED (ids ascending /
//     pairs by (a,b)) for determinism.
//
// MODERNIZE (zero steady-state alloc): buckets, the id->range map, and the
// dedup scratch are reused std containers (clear() preserves capacity). The
// only allocation churn after warmup is bucket growth on novel cells. We store
// each id's tight box so pairs()/queryAABB can narrow without a back-reference
// to the world.
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <Manifold2D/Physics/Broadphase/Broadphase.hpp>
#include <Manifold2D/Physics/PhysicsTypes.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        class SpatialHash final : public IBroadphase
        {
        public:
            // cellSize default 1.0 m (MKS). Mirrors WorldDef::hashCellSize (the
            // value the world actually passes via MakeBroadphase); this argless
            // default is dormant on every live/test path but is aligned so a
            // future argless construction cannot silently mean a 64 m cell. Was
            // 64 px, ported from SpatialHash.new(cellSize or 64); flipped for MKS
            // units (spec 2026-07-02 s3, matching WorldDef::hashCellSize -> 1.0).
            explicit SpatialHash(Real cellSize = Real(1));

            void Update(std::uint32_t id, const Aabb2& box) override;
            void Remove(std::uint32_t id) override;
            int  QueryAABB(const Aabb2& box,
                           std::vector<std::uint32_t>& out) const override;
            int  Pairs(std::vector<BroadphasePair>& out) const override;

        private:
            // Integer cell range an id occupies (the Lua ranges[id] tuple).
            struct CellRange
            {
                std::int32_t kx0 = 0, ky0 = 0, kx1 = 0, ky1 = 0;
            };

            // Per-id record: its tight box (for narrowing) + cell range.
            struct Entry
            {
                Aabb2     box{};
                CellRange range{};
            };

            // Bucket key (the Lua bkey: kx*0x40000 + ky).
            // Collision-free while |ky| < 0x20000 (131072 cells on Y; at the
            // 1.0 m default cell that is +/-131072 m ~= +/-131 km).
            [[nodiscard]] static std::int64_t BucketKey(std::int32_t kx,
                                                        std::int32_t ky) noexcept
            {
                return static_cast<std::int64_t>(kx) * 0x40000 +
                       static_cast<std::int64_t>(ky);
            }

            // floor(v / cs) as an int cell coordinate.
            [[nodiscard]] std::int32_t CellOf(Real v) const noexcept;

            // Reject NaN / finite-but-huge boxes before CellOf's int32 cast, and
            // any box whose cell span would blow the bucket loop. Mirrors
            // SpatialGrid::SaneBox (the default broadphase's escape guard); this
            // class has NO origin member, so the magnitude bound is |coord|.
            [[nodiscard]] bool SaneBox(const Aabb2& b) const noexcept;

            void RemoveFromBuckets(std::uint32_t id, const CellRange& r);

            Real m_cellSize = Real(1);  // MKS default; see the ctor comment.

            // bucket key -> ids in that cell. unordered_map iteration order is
            // NOT used for output (pairs() iterates ids ascending via m_entries
            // and sorts), so determinism is preserved.
            std::unordered_map<std::int64_t, std::vector<std::uint32_t>> m_buckets;

            // id -> entry. Keyed by id; we iterate it sorted for deterministic
            // Pairs() (collect ids, sort, then walk).
            std::unordered_map<std::uint32_t, Entry> m_entries;

            // Generation-stamped dedup scratch (the Lua _seen / _gen). Reused
            // across queries; mutable because QueryAABB/Pairs are const.
            // m_seen: id -> generation stamp (body id key, uint64 gen value).
            mutable std::unordered_map<std::uint32_t, std::uint64_t> m_seen;
            mutable std::uint64_t                                    m_gen = 0;
            // Reused id list for sorted Pairs() iteration.
            mutable std::vector<std::uint32_t> m_idScratch;
        };

    } // namespace Physics
} // namespace Manifold2D
