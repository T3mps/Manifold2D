// DynamicTree.cpp -- pooled dynamic AABB tree broadphase.
//
// See DynamicTree.hpp for the contract. This is a faithful port of
// Client/src/physics/AABBTree.lua with the GC-table nodes replaced by an
// index-based pool (zero steady-state heap) and the uniform deterministic
// output contract (sorted, true-overlap pairs / sorted query ids).
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.

#include <Manifold2D/Physics/Broadphase/DynamicTree.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

namespace Manifold2D
{
    namespace Physics
    {
        namespace
        {
            // perim(x0,y0,x1,y1) = 2 * ((x1-x0) + (y1-y0)) -- the Lua's 2D SAH
            // analogue (box "cost" used to pick the cheaper child to grow).
            Real Perim(const Aabb2& b) noexcept
            {
                return Real(2) * ((b.max.x - b.min.x) + (b.max.y - b.min.y));
            }

            // Union of two boxes (the Lua nodeUnion / refit min-max).
            Aabb2 Union(const Aabb2& a, const Aabb2& b) noexcept
            {
                Aabb2 u;
                u.min.x = std::min(a.min.x, b.min.x);
                u.min.y = std::min(a.min.y, b.min.y);
                u.max.x = std::max(a.max.x, b.max.x);
                u.max.y = std::max(a.max.y, b.max.y);
                return u;
            }

            // Fat-box overlap test (descent predicate): the Lua query's
            // "n.x0 <= x1 and x0 <= n.x1 and n.y0 <= y1 and y0 <= n.y1".
            bool FatOverlap(const Aabb2& n, const Aabb2& q) noexcept
            {
                return n.min.x <= q.max.x && q.min.x <= n.max.x &&
                       n.min.y <= q.max.y && q.min.y <= n.max.y;
            }
        } // namespace

        // --------------------------------------------------------------------
        // Node pool (index-based free-list).
        // --------------------------------------------------------------------
        std::uint32_t DynamicTree::AllocNode()
        {
            if (m_freeList != kNull)
            {
                const std::uint32_t idx = m_freeList;
                m_freeList              = m_nodes[idx].parent; // next in chain
                m_nodes[idx]           = Node{};               // reset
                return idx;
            }
            const std::uint32_t idx =
                static_cast<std::uint32_t>(m_nodes.size());
            m_nodes.emplace_back();
            return idx;
        }

        void DynamicTree::FreeNode(std::uint32_t idx)
        {
            m_nodes[idx]        = Node{};
            m_nodes[idx].parent = m_freeList; // push onto free-list
            m_freeList          = idx;
        }

        // --------------------------------------------------------------------
        // id -> leaf slot map (the Lua `leaves` table). A simple dense vector
        // keyed by id; entry kNull means absent. Avoids unordered_map so the
        // Pairs() iteration order is deterministic (we iterate ids ascending).
        // --------------------------------------------------------------------
        std::uint32_t DynamicTree::LeafOf(std::uint32_t id) const noexcept
        {
            return id < m_leafOfId.size() ? m_leafOfId[id] : kNull;
        }

        void DynamicTree::SetLeafOf(std::uint32_t id, std::uint32_t leaf)
        {
            if (id >= m_leafOfId.size())
            {
                m_leafOfId.resize(static_cast<std::size_t>(id) + 1u, kNull);
            }
            m_leafOfId[id] = leaf;
        }

        // --------------------------------------------------------------------
        // refit: walk parents up, recomputing the union box (Lua refit).
        // --------------------------------------------------------------------
        void DynamicTree::Refit(std::uint32_t n)
        {
            while (n != kNull)
            {
                Node& node = m_nodes[n];
                if (!node.IsLeaf())
                {
                    node.fat = Union(m_nodes[node.left].fat,
                                     m_nodes[node.right].fat);
                }
                n = node.parent;
            }
        }

        // --------------------------------------------------------------------
        // _insertLeaf: descend by least combined-perimeter growth, split the
        // chosen leaf with a new internal node (faithful port).
        // --------------------------------------------------------------------
        void DynamicTree::InsertLeaf(std::uint32_t leaf)
        {
            if (m_root == kNull)
            {
                m_root                 = leaf;
                m_nodes[leaf].parent   = kNull;
                return;
            }

            const Aabb2 leafBox = m_nodes[leaf].fat;

            std::uint32_t n = m_root;
            while (!m_nodes[n].IsLeaf())
            {
                const std::uint32_t l = m_nodes[n].left;
                const std::uint32_t r = m_nodes[n].right;

                const Real growL =
                    Perim(Union(m_nodes[l].fat, leafBox)) - Perim(m_nodes[l].fat);
                const Real growR =
                    Perim(Union(m_nodes[r].fat, leafBox)) - Perim(m_nodes[r].fat);

                n = (growL <= growR) ? l : r; // Lua: growL <= growR ? l : r
            }

            // Split leaf n with a new internal node (Lua _insertLeaf tail).
            const std::uint32_t oldParent = m_nodes[n].parent;
            const std::uint32_t internal  = AllocNode();
            m_nodes[internal].parent = oldParent;
            m_nodes[internal].left   = n;
            m_nodes[internal].right  = leaf;

            if (oldParent != kNull)
            {
                if (m_nodes[oldParent].left == n)
                    m_nodes[oldParent].left = internal;
                else
                    m_nodes[oldParent].right = internal;
            }
            else
            {
                m_root = internal;
            }

            m_nodes[n].parent    = internal;
            m_nodes[leaf].parent = internal;
            Refit(internal);
        }

        // --------------------------------------------------------------------
        // _removeLeaf: promote the sibling into the parent's slot (Lua port).
        // The freed internal parent goes back on the pool free-list.
        // --------------------------------------------------------------------
        void DynamicTree::RemoveLeaf(std::uint32_t leaf)
        {
            const std::uint32_t p = m_nodes[leaf].parent;
            if (p == kNull)
            {
                m_root = kNull; // leaf was the root
                return;
            }

            const std::uint32_t sib =
                (m_nodes[p].left == leaf) ? m_nodes[p].right : m_nodes[p].left;
            const std::uint32_t gp = m_nodes[p].parent;

            m_nodes[sib].parent = gp;
            if (gp != kNull)
            {
                if (m_nodes[gp].left == p)
                    m_nodes[gp].left = sib;
                else
                    m_nodes[gp].right = sib;
                Refit(gp);
            }
            else
            {
                m_root = sib;
            }
            FreeNode(p); // recycle the now-orphaned internal node
        }

        // --------------------------------------------------------------------
        // update (upsert): fat-box-contains fast path, else remove + reinsert
        // with a fresh fat box (Lua AABBTree:update).
        //
        // Move-buffer: mark id in m_moved on EVERY Update, even the within-fat-
        // box fast path.  Within-fat-box moves still change tight-overlap pairs
        // (a body that nudges slightly may gain or lose tight overlap with its
        // neighbours), so "moved" must mean "any Update call", not just reinserts.
        // --------------------------------------------------------------------
        void DynamicTree::Update(std::uint32_t id, const Aabb2& box)
        {
            // A non-finite (NaN/Inf) AABB poisons ancestor fat boxes via Refit's
            // unions (NaN comparisons are false, so FatOverlap silently fails and
            // OTHER proxies -- not just this one -- become unqueryable). Box2D
            // asserts box validity for the same reason (b2AABB_IsValid). Loud in
            // Debug is the point: dropping the insert instead would desync the
            // tree from the body's SlotAabb, which is worse than a hard stop.
            assert(std::isfinite(box.min.x) && std::isfinite(box.min.y) &&
                   std::isfinite(box.max.x) && std::isfinite(box.max.y) &&
                   "DynamicTree::Update: AABB must be finite");
            std::uint32_t leaf = LeafOf(id);
            if (leaf != kNull)
            {
                // Still inside the fat box: only refresh the tight box.
                const Aabb2& fat = m_nodes[leaf].fat;
                if (box.min.x >= fat.min.x && box.min.y >= fat.min.y &&
                    box.max.x <= fat.max.x && box.max.y <= fat.max.y)
                {
                    m_nodes[leaf].tight = box;
                    // ALWAYS mark moved (tight box changed -- may affect overlap).
                    m_moved.insert(id);
                    m_removed.erase(id);
                    return;
                }
                RemoveLeaf(leaf);
            }
            else
            {
                leaf = AllocNode();
                m_nodes[leaf].left  = kNull; // explicit leaf
                m_nodes[leaf].right = kNull;
                m_nodes[leaf].id    = id;
                SetLeafOf(id, leaf);
            }

            m_nodes[leaf].tight   = box;
            m_nodes[leaf].fat.min = Vec2(box.min.x - kMargin, box.min.y - kMargin);
            m_nodes[leaf].fat.max = Vec2(box.max.x + kMargin, box.max.y + kMargin);
            InsertLeaf(leaf);

            // Mark moved (reinsert or fresh insert -- pairs must be recomputed).
            m_moved.insert(id);
            m_removed.erase(id);
        }

        void DynamicTree::Remove(std::uint32_t id)
        {
            const std::uint32_t leaf = LeafOf(id);
            if (leaf == kNull)
            {
                return; // Lua guard: not present
            }
            RemoveLeaf(leaf);
            FreeNode(leaf);
            SetLeafOf(id, kNull);

            // Move-buffer: track removed id so UpdatePairs evicts its pairs.
            m_removed.insert(id);
            m_moved.erase(id);
        }

        // --------------------------------------------------------------------
        // queryAABB: stack-based descent on the FAT boxes, narrow to the TIGHT
        // box (Lua query + queryAABB). Output is collected then SORTED ascending
        // for the deterministic contract.
        // --------------------------------------------------------------------
        int DynamicTree::QueryAABB(const Aabb2& box,
                                   std::vector<std::uint32_t>& out) const
        {
            out.clear();
            if (m_root == kNull)
            {
                return 0;
            }

            // Descent scratch: a THREAD-LOCAL stack, NOT the shared mutable
            // m_stack member, so QueryAABB is re-entrant across worker threads.
            // PhysicsWorld::StaticCandidates queries the static tree from every
            // worker in the parallel create-phase (Executor()->ParallelFor); a
            // shared m_stack would race and corrupt the descent (SpatialGrid's
            // QueryAABB was likewise re-entrant, touching only caller scratch).
            // thread_local is reused per-thread, so there is no per-call heap in
            // steady state, and the output is unchanged (sorted true-overlap set).
            thread_local std::vector<std::uint32_t> stack;
            stack.clear();
            stack.push_back(m_root);
            while (!stack.empty())
            {
                const std::uint32_t ni = stack.back();
                stack.pop_back();
                const Node& n = m_nodes[ni];
                if (!FatOverlap(n.fat, box))
                {
                    continue;
                }
                if (!n.IsLeaf())
                {
                    stack.push_back(n.left);
                    stack.push_back(n.right);
                }
                else if (AabbOverlap(n.tight, box)) // narrow against tight box
                {
                    out.push_back(n.id);
                }
            }

            std::sort(out.begin(), out.end());
            return static_cast<int>(out.size());
        }

        // --------------------------------------------------------------------
        // pairs: for each leaf, descend querying its TIGHT box; emit (id,other)
        // when other.id > id (the Lua's dup/self guard) AND the tight boxes
        // overlap. Iterate leaves by id ASCENDING (deterministic) then sort the
        // final output by (a,b). The result is the canonical true-overlap set.
        // --------------------------------------------------------------------
        int DynamicTree::Pairs(std::vector<BroadphasePair>& out) const
        {
            out.clear();
            if (m_root == kNull)
            {
                return 0;
            }

            // Iterate ids ascending so the (smaller-id) leaf drives each pair;
            // m_leafOfId is keyed by id, giving a deterministic order without
            // any hash-map iteration.
            for (std::uint32_t id = 0;
                 id < static_cast<std::uint32_t>(m_leafOfId.size()); ++id)
            {
                const std::uint32_t leaf = m_leafOfId[id];
                if (leaf == kNull)
                {
                    continue;
                }
                const Aabb2 tight = m_nodes[leaf].tight;

                m_stack.clear();
                m_stack.push_back(m_root);
                while (!m_stack.empty())
                {
                    const std::uint32_t ni = m_stack.back();
                    m_stack.pop_back();
                    const Node& n = m_nodes[ni];
                    if (!FatOverlap(n.fat, tight))
                    {
                        continue;
                    }
                    if (!n.IsLeaf())
                    {
                        m_stack.push_back(n.left);
                        m_stack.push_back(n.right);
                    }
                    else if (n.id > id && AabbOverlap(n.tight, tight))
                    {
                        out.push_back(BroadphasePair{ id, n.id });
                    }
                }
            }

            std::sort(out.begin(), out.end());
            return static_cast<int>(out.size());
        }

        // --------------------------------------------------------------------
        // UpdatePairs seam: EvictTouchedAndCollectMoved (Phase D2, Task 2).
        //
        // STEP 1 of the three-seam decomposition. Evicts all cached pairs in
        // m_pairSet that touch any moved or removed proxy, then snapshots the
        // set of moved ids into movedOut and clears m_moved / m_removed.
        //
        // The snapshot-then-clear ordering is safe: the caller (UpdatePairs
        // serial wrapper, or PhysicsWorld in Task 3) iterates movedOut for the
        // STEP 2 re-query, which does not touch m_moved or m_removed.
        // --------------------------------------------------------------------
        void DynamicTree::EvictTouchedAndCollectMoved(
            std::vector<std::uint32_t>& movedOut)
        {
            // Cannot erase while iterating unordered_set -- collect keys first.
            if (!m_moved.empty() || !m_removed.empty())
            {
                m_toErase.clear();
                for (const std::uint64_t key : m_pairSet)
                {
                    const std::uint32_t lo = static_cast<std::uint32_t>(key >> 32);
                    const std::uint32_t hi = static_cast<std::uint32_t>(key & 0xFFFFFFFFu);
                    if (m_moved.count(lo) || m_moved.count(hi) ||
                        m_removed.count(lo) || m_removed.count(hi))
                    {
                        m_toErase.push_back(key);
                    }
                }
                for (const std::uint64_t key : m_toErase)
                {
                    m_pairSet.erase(key);
                }
            }

            // Snapshot m_moved into movedOut, then clear both move-buffer sets.
            movedOut.clear();
            movedOut.reserve(m_moved.size());
            for (auto id : m_moved)
            {
                movedOut.push_back(id);
            }
            m_moved.clear();
            m_removed.clear();
        }

        // --------------------------------------------------------------------
        // UpdatePairs seam: QueryProxyPairs (Phase D2, Task 2).
        //
        // STEP 2 for ONE proxy id. Descends the tree on the proxy's FAT box
        // (broad candidate set), then filters by tight-box overlap and appends
        // canonical (lo<<32|hi) keys to out.
        //
        // const: read-only view of the tree. Uses the caller-supplied stack
        // (NOT m_stack) so the serial wrapper can pass m_stack and Task 3 can
        // pass per-worker scratch without contention.
        //
        // If the proxy's leaf is absent (LeafOf returns kNull), returns
        // immediately with no output appended -- matches the original STEP 2
        // guard for ids removed between mark and flush.
        // --------------------------------------------------------------------
        void DynamicTree::QueryProxyPairs(std::uint32_t id,
                                          std::vector<std::uint32_t>& stack,
                                          std::vector<std::uint64_t>& out) const
        {
            const std::uint32_t leafA = LeafOf(id);
            if (leafA == kNull)
            {
                return; // was removed between mark and flush -- append nothing.
            }

            const Aabb2 fatA   = m_nodes[leafA].fat;
            const Aabb2 tightA = m_nodes[leafA].tight;

            // defensive: a live leafA (guarded above) implies a non-null root, so this cannot trigger; the early return is the per-proxy equivalent of the old STEP-2 loop's continue.
            if (m_root == kNull)
            {
                return;
            }

            stack.clear();
            stack.push_back(m_root);
            while (!stack.empty())
            {
                const std::uint32_t ni = stack.back();
                stack.pop_back();
                const Node& n = m_nodes[ni];

                if (!FatOverlap(n.fat, fatA))
                {
                    continue;
                }
                if (!n.IsLeaf())
                {
                    stack.push_back(n.left);
                    stack.push_back(n.right);
                }
                else if (n.id != id && AabbOverlap(n.tight, tightA))
                {
                    const std::uint32_t lo = (id < n.id) ? id : n.id;
                    const std::uint32_t hi = (id < n.id) ? n.id : id;
                    const std::uint64_t key =
                        (static_cast<std::uint64_t>(lo) << 32) |
                        static_cast<std::uint64_t>(hi);
                    out.push_back(key);
                }
            }
        }

        // --------------------------------------------------------------------
        // UpdatePairs seam: MergeAndEmit (Phase D2, Task 2).
        //
        // STEP 3 of the three-seam decomposition. Inserts all keys from the
        // per-worker buffers into m_pairSet (Task 3 supplies one buffer per
        // worker; the serial wrapper supplies a single-element span), then
        // emits the full m_pairSet as a sorted BroadphasePair vector.
        // --------------------------------------------------------------------
        int DynamicTree::MergeAndEmit(
            std::span<const std::vector<std::uint64_t>> perWorker,
            std::vector<BroadphasePair>& out)
        {
            for (const auto& buf : perWorker)
            {
                for (const std::uint64_t key : buf)
                {
                    m_pairSet.insert(key);
                }
            }

            // Emit sorted.
            out.clear();
            out.reserve(m_pairSet.size());
            for (const std::uint64_t key : m_pairSet)
            {
                const std::uint32_t lo = static_cast<std::uint32_t>(key >> 32);
                const std::uint32_t hi = static_cast<std::uint32_t>(key & 0xFFFFFFFFu);
                out.push_back(BroadphasePair{ lo, hi });
            }
            std::sort(out.begin(), out.end());
            return static_cast<int>(out.size());
        }

        // --------------------------------------------------------------------
        // UpdatePairs: incremental pair-set maintenance (Phase 2, Task 4).
        // Rewritten as a serial wrapper over the three seams (Phase D2, Task 2).
        //
        // Correctness invariant (oracle-gated):
        //   UpdatePairs() == Pairs() == brute-force after every mutation.
        // --------------------------------------------------------------------
        int DynamicTree::UpdatePairs(std::vector<BroadphasePair>& out)
        {
            EvictTouchedAndCollectMoved(m_movedSerial);
            m_findSerial.clear();
            for (const std::uint32_t id : m_movedSerial)
            {
                QueryProxyPairs(id, m_stack, m_findSerial);
            }
            return MergeAndEmit(
                std::span<const std::vector<std::uint64_t>>(&m_findSerial, 1),
                out);
        }

        // ----------------------------------------------------------------
        // ForEachLeaf: read-only debug-viz enumeration of the live proxies.
        // Walks m_leafOfId (id -> leaf slot), skipping kNull holes, and yields
        // each leaf node's (id, tight, fat). The map index is the proxy id, but
        // we read node.id so the callback gets the value the tree actually
        // stored. Pure read path -- no mutation, display-order only.
        // ----------------------------------------------------------------
        void DynamicTree::ForEachLeaf(
            Mosaic::FunctionRef<void(std::uint32_t,
                             const Aabb2&,
                             const Aabb2&)> fn) const
        {
            for (std::uint32_t id = 0;
                 id < static_cast<std::uint32_t>(m_leafOfId.size()); ++id)
            {
                const std::uint32_t leaf = m_leafOfId[id];
                if (leaf == kNull)
                {
                    continue; // no proxy for this id
                }
                const Node& node = m_nodes[leaf];
                fn(node.id, node.tight, node.fat);
            }
        }

        // TryGetFatBox: O(1) fat-AABB lookup by proxy id. LeafOf(id) returns the
        // leaf slot or kNull (absent); on a live leaf the fat box is node.fat --
        // the same value ForEachLeaf yields, just resolved without scanning.
        bool DynamicTree::TryGetFatBox(std::uint32_t id, Aabb2& out) const
        {
            const std::uint32_t leaf = LeafOf(id);
            if (leaf == kNull)
            {
                return false; // no live proxy for this id
            }
            out = m_nodes[leaf].fat;
            return true;
        }

    } // namespace Physics
} // namespace Manifold2D
