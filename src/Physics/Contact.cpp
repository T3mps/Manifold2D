// Contact.cpp -- ContactPool implementation (Manifold2D 2D physics, Phase 3 Task 1).
//
// See Contact.hpp for the contract. PRESENTATION-FREE + C++20-clean.

#include <Manifold2D/Physics/Contact.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        // Mix a single handle's {index, generation} into a 32-bit half so that
        // a recycled slot (same index, bumped generation) hashes DIFFERENTLY
        // from the stale handle that previously occupied the slot. The
        // multiply-by-golden-ratio-prime scrambles the generation across the
        // whole 32-bit range before xor-ing it into the index, so two distinct
        // (index, generation) pairs do not collide in practice (and the test's
        // {7,1} vs {7,2} land on different halves -> different keys).
        //
        // This is a NON-cryptographic mix; each handle collapses to one 32-bit
        // half, so a full-key (64-bit pair) collision is theoretically possible
        // but negligible for realistic slot/generation ranges -- revisit if
        // either field approaches 2^32.
        static std::uint32_t MixHandle(FixtureHandle h)
        {
            return h.index ^ (h.generation * 0x9E3779B9u);
        }

        std::uint64_t ContactPool::Key(FixtureHandle a, FixtureHandle b)
        {
            const std::uint32_t ha = MixHandle(a);
            const std::uint32_t hb = MixHandle(b);
            // Order the two halves canonically (min in the high word) so the
            // pair is unordered: Key(a,b) == Key(b,a).
            const std::uint32_t lo = ha < hb ? ha : hb;
            const std::uint32_t hi = ha < hb ? hb : ha;
            return (static_cast<std::uint64_t>(lo) << 32) | static_cast<std::uint64_t>(hi);
        }

        ContactPool::EnsureResult ContactPool::EnsurePair(FixtureHandle a, FixtureHandle b)
        {
            const std::uint64_t key = Key(a, b);

            // HIT: return the existing pair untouched, signalling not-created.
            const auto it = m_index.find(key);
            if (it != m_index.end())
                return { it->second, false };

            // MISS: recycle a freed id or grow the pool.
            std::uint32_t id;
            if (!m_free.empty())
            {
                id = m_free.back();
                m_free.pop_back();
                // Reset the recycled slot to a fresh default Contact: body slots ->
                // kInvalidSlot AND color -> kInvalidColor (Phase C, Task 4 -- the
                // recycled slot must NOT carry a stale color from the contact that
                // previously occupied this id; the caller re-colors via
                // AssignContactColor on a fresh create).
                m_pool[id]  = Contact{};
                m_alive[id] = 1u;
            }
            else
            {
                id = static_cast<std::uint32_t>(m_pool.size());
                m_pool.emplace_back();   // default Contact (body slots kInvalidSlot, color kInvalidColor)
                m_alive.push_back(1u);
            }

            // Set the handles; caller fills body slots/orientation on first
            // creation via Get(id) (created == true tells it to).
            m_pool[id].a = a;
            m_pool[id].b = b;

            m_index.emplace(key, id);
            ++m_live;
            return { id, true };
        }

        std::uint32_t ContactPool::Find(FixtureHandle a, FixtureHandle b) const
        {
            const auto it = m_index.find(Key(a, b));
            return it == m_index.end() ? kNone : it->second;
        }

        void ContactPool::Destroy(std::uint32_t id)
        {
            if (id >= m_alive.size() || m_alive[id] == 0u)
                return;

            m_index.erase(Key(m_pool[id].a, m_pool[id].b));
            m_alive[id] = 0u;
            m_free.push_back(id);
            --m_live;
        }

        void ContactPool::ForEach(FunctionRef<void(std::uint32_t id, Contact&)> fn)
        {
            // Ascending id order is the determinism seam -- iterate the dense
            // pool, NOT the unordered_map.
            for (std::uint32_t id = 0; id < m_pool.size(); ++id)
            {
                if (m_alive[id] != 0u)
                    fn(id, m_pool[id]);
            }
        }

        void ContactPool::ForEach(FunctionRef<void(std::uint32_t id, const Contact&)> fn) const
        {
            // Same ascending-id walk, read-only (Phase 3 Task 3).
            for (std::uint32_t id = 0; id < m_pool.size(); ++id)
            {
                if (m_alive[id] != 0u)
                    fn(id, m_pool[id]);
            }
        }

        void ContactPool::Clear()
        {
            m_pool.clear();
            m_alive.clear();
            m_free.clear();
            m_index.clear();
            m_live = 0;
        }

    } // namespace Physics
} // namespace Manifold2D
