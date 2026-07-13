#pragma once

// SweepAndPrune: sweep-and-prune broadphase (M6, Task P1.6).
//
// PORT + MODERNIZE of Client/src/physics/SAP.lua. An alternate mover
// broadphase behind IBroadphase (the DEFAULT is DynamicTree).
//
// Faithful ports of the Lua algorithm:
//   * Entries kept x0-sorted via an insertion-sort pass on Update (near O(n)
//     under temporal coherence).
//   * queryAABB sweeps the sorted list, breaking once an entry's x0 > the query
//     x1, with the y-overlap + x1 >= query.x0 test.
//   * pairs() sweeps j forward from each i until b.x0 > a.x1, emitting on
//     y-overlap, a < b.
//
// MODERNIZE (uniform contract): pairs() already narrows to true AABB overlap
// (SAP's sweep IS the narrow phase). We sort the final output by (a,b) and
// queryAABB sorts ids ascending, so the emitted sets match DynamicTree /
// SpatialHash exactly and are deterministic.
//
// MODERNIZE (zero steady-state alloc): the sorted entry list + id->slot map are
// reused std containers; Update sorts in place. Output fills the caller's
// vector via clear()+push_back (capacity preserved).
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
        class SweepAndPrune final : public IBroadphase
        {
        public:
            SweepAndPrune() = default;

            void Update(std::uint32_t id, const Aabb2& box) override;
            void Remove(std::uint32_t id) override;
            int  QueryAABB(const Aabb2& box,
                           std::vector<std::uint32_t>& out) const override;
            int  Pairs(std::vector<BroadphasePair>& out) const override;

        private:
            // One x0-sorted entry (the Lua list element). Stores the full tight
            // box; id is the body index.
            struct Entry
            {
                std::uint32_t id = 0;
                Aabb2         box{};
            };

            // x0-sorted list (the Lua `list`).
            std::vector<Entry> m_list;
            // id -> index into m_list (the Lua `byId`, but storing the slot so
            // Remove is exact). Rebuilt after the insertion-sort pass.
            std::unordered_map<std::uint32_t, std::size_t> m_indexOf;

            // Re-sort m_list by x0 (insertion sort, like the Lua) and refresh
            // m_indexOf.
            void SortByX0();
        };

    } // namespace Physics
} // namespace Manifold2D
