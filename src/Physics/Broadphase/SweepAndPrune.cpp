// SweepAndPrune.cpp -- sweep-and-prune broadphase.
//
// See SweepAndPrune.hpp for the contract. Faithful port of
// Client/src/physics/SAP.lua: x0-sorted entry list (insertion sort on update),
// sweep-based queryAABB / pairs. Output is sorted (ids ascending / pairs by
// (a,b)) for the uniform deterministic contract.
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.

#include <Manifold2D/Physics/Broadphase/SweepAndPrune.hpp>

#include <algorithm>

namespace Manifold2D
{
    namespace Physics
    {
        void SweepAndPrune::SortByX0()
        {
            // Insertion sort on x0 (the Lua's pass; cheap under coherence and a
            // stable choice). For an arbitrary rebuild this is O(n^2) worst
            // case, matching the Lua exactly; fine at our body counts.
            const std::size_t n = m_list.size();
            for (std::size_t i = 1; i < n; ++i)
            {
                Entry v   = m_list[i];
                std::size_t j = i;
                while (j > 0 && m_list[j - 1].box.min.x > v.box.min.x)
                {
                    m_list[j] = m_list[j - 1];
                    --j;
                }
                m_list[j] = v;
            }

            m_indexOf.clear();
            for (std::size_t i = 0; i < n; ++i)
            {
                m_indexOf[m_list[i].id] = i;
            }
        }

        void SweepAndPrune::Update(std::uint32_t id, const Aabb2& box)
        {
            auto it = m_indexOf.find(id);
            if (it == m_indexOf.end())
            {
                m_list.push_back(Entry{ id, box });
            }
            else
            {
                m_list[it->second].box = box;
            }
            SortByX0();
        }

        void SweepAndPrune::Remove(std::uint32_t id)
        {
            auto it = m_indexOf.find(id);
            if (it == m_indexOf.end())
            {
                return; // Lua guard: not present
            }
            const std::size_t idx = it->second;
            m_list.erase(m_list.begin() +
                         static_cast<std::ptrdiff_t>(idx));
            // Rebuild the index map (positions shifted). Keep x0 order intact;
            // erase preserves it, so a full re-sort is unnecessary -- just
            // refresh indices.
            m_indexOf.clear();
            for (std::size_t i = 0; i < m_list.size(); ++i)
            {
                m_indexOf[m_list[i].id] = i;
            }
        }

        int SweepAndPrune::QueryAABB(const Aabb2& box,
                                     std::vector<std::uint32_t>& out) const
        {
            out.clear();
            for (const Entry& e : m_list)
            {
                if (e.box.min.x > box.max.x)
                {
                    break; // x0 past the query's x1: nothing further overlaps
                }
                // inline overlap (not AabbOverlap) so the x-sorted early break works
                // Lua: e.x1 >= x0 and e.y0 <= y1 and e.y1 >= y0
                if (e.box.max.x >= box.min.x && e.box.min.y <= box.max.y &&
                    e.box.max.y >= box.min.y)
                {
                    out.push_back(e.id);
                }
            }
            std::sort(out.begin(), out.end());
            return static_cast<int>(out.size());
        }

        int SweepAndPrune::Pairs(std::vector<BroadphasePair>& out) const
        {
            out.clear();
            const std::size_t n = m_list.size();
            for (std::size_t i = 0; i < n; ++i)
            {
                const Entry& a = m_list[i];
                for (std::size_t j = i + 1; j < n; ++j)
                {
                    const Entry& b = m_list[j];
                    if (b.box.min.x > a.box.max.x)
                    {
                        break; // x-spans no longer can overlap; sweep done
                    }
                    // inline overlap (not AabbOverlap) so the x-sorted early break works
                    // y-overlap test (Lua: a.y0 <= b.y1 and b.y0 <= a.y1).
                    if (a.box.min.y <= b.box.max.y && b.box.min.y <= a.box.max.y)
                    {
                        const std::uint32_t lo = std::min(a.id, b.id);
                        const std::uint32_t hi = std::max(a.id, b.id);
                        out.push_back(BroadphasePair{ lo, hi });
                    }
                }
            }
            std::sort(out.begin(), out.end());
            return static_cast<int>(out.size());
        }

    } // namespace Physics
} // namespace Manifold2D
