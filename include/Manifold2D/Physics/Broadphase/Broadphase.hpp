#pragma once

// Broadphase interface for the Manifold2D 2D physics engine (M6, Task P1.6).
//
// PORT + MODERNIZE: the three Lua broadphases
// (Client/src/physics/AABBTree.lua, SpatialHash.lua, SAP.lua) share one
// surface -- Insert/Update/Remove/queryAABB/pairs. This header is the C++
// port of that surface as the abstract IBroadphase. The DynamicTree port
// (AABBTree.lua) is the DEFAULT mover broadphase (MODERNIZE: the Lua defaulted
// to SpatialHash); SpatialHash and SweepAndPrune are alternates behind the
// same interface. The world in P1.8 holds an IBroadphase*.
//
// DETERMINISM + UNIFORM CONTRACT (MODERNIZE -- the key behavioral change over
// the Lua):
//   * Pairs(out) returns TRUE-AABB-OVERLAP pairs ONLY. Every strategy performs
//     the final tight-box overlap test before emitting. The Lua SpatialHash
//     returned broad bucket CANDIDATES (the harness narrowed them afterward);
//     here SpatialHash narrows + dedups internally so ALL THREE strategies
//     emit the IDENTICAL set. This is the broadphase equivalence invariant
//     gated by PhysicsBroadphaseTest + the broadphase-strategy-invariance case
//     in PhysicsInvariantsTest (analytic; the physics_oracle bit-match gate was
//     retired in Physics v2 Phase A).
//   * Pairs are emitted with a < b and the output is SORTED lexicographically
//     by (a, b). Strategy-independent + deterministic -- the Phase-2 solver
//     needs a fixed contact order. We do NOT leak std::unordered_map iteration
//     order into the output.
//   * QueryAABB returns ids SORTED ascending, narrowed against the TIGHT box
//     (the Lua AABBTree narrows tight; all strategies match it).
//
// ZERO STEADY-STATE ALLOCATION (MODERNIZE): the Lua used GC tables for tree
// nodes / buckets / scratch. The C++ implementations pool their internal
// storage (index-based node pool, reused scratch vectors) so there is no
// per-insert / per-Pairs heap traffic after warmup. Pairs/QueryAABB fill the
// caller's std::vector with clear()+push_back, which preserves capacity.
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.
// No SDL3/NVRHI/Batcher2D/ImGui. Compiles both /MD (Arcane.dll) and
// pure-compute (Arcane-only, /MD). namespace
// Manifold2D::Physics, Core style.

#include <cstdint>
#include <span>
#include <vector>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        // ----------------------------------------------------------------
        // Aabb2: the broadphase box. The Lua used a (x0,y0,x1,y1) tuple; we
        // reuse the engine's Shapes::Aabb { Vec2 min, max } so the world can
        // feed Shape::ComputeAABB output straight in without a conversion.
        // ----------------------------------------------------------------
        using Aabb2 = Aabb;

        // True iff two tight boxes overlap on both axes (inclusive, matching
        // the Lua "A.x0 <= B.x1 and B.x0 <= A.x1 and ..." test exactly). This
        // is the single narrowing predicate every strategy uses before
        // emitting a pair, so the emitted sets are identical.
        [[nodiscard]] inline bool AabbOverlap(const Aabb2& a,
                                              const Aabb2& b) noexcept
        {
            return a.min.x <= b.max.x && b.min.x <= a.max.x &&
                   a.min.y <= b.max.y && b.min.y <= a.max.y;
        }

        // ----------------------------------------------------------------
        // BroadphasePair: an overlapping id pair, always a < b.
        // ----------------------------------------------------------------
        struct BroadphasePair
        {
            std::uint32_t a = 0;
            std::uint32_t b = 0;

            friend constexpr bool operator==(const BroadphasePair& p,
                                             const BroadphasePair& q) noexcept
            {
                return p.a == q.a && p.b == q.b;
            }
            friend constexpr bool operator!=(const BroadphasePair& p,
                                             const BroadphasePair& q) noexcept
            {
                return !(p == q);
            }
            // Lexicographic order on (a, b) -- the deterministic emit order.
            friend constexpr bool operator<(const BroadphasePair& p,
                                            const BroadphasePair& q) noexcept
            {
                return p.a != q.a ? p.a < q.a : p.b < q.b;
            }
        };

        // ----------------------------------------------------------------
        // IBroadphase: the shared surface of all three strategies.
        // ----------------------------------------------------------------
        //
        // Ids are body indices (the Lua used integer ids). Update is an UPSERT
        // (insert-or-update), matching the Lua update-as-insert; there is no
        // separate Insert -- the first Update for an id inserts it.
        class IBroadphase
        {
        public:
            virtual ~IBroadphase() = default;

            // Upsert id with the given TIGHT box (insert if new, else move).
            virtual void Update(std::uint32_t id, const Aabb2& box) = 0;

            // Remove id. No-op if id is not present (matches the Lua guard).
            virtual void Remove(std::uint32_t id) = 0;

            // Ids whose TIGHT box overlaps box. out is cleared then filled,
            // SORTED ascending (deterministic). Returns out.size().
            virtual int QueryAABB(const Aabb2& box,
                                  std::vector<std::uint32_t>& out) const = 0;

            // All true-AABB-overlap pairs (a < b), SORTED lexicographically by
            // (a, b). out is cleared then filled. Returns out.size(). All three
            // strategies return the IDENTICAL set for the same scene.
            virtual int Pairs(std::vector<BroadphasePair>& out) const = 0;

            // Incremental pair maintenance. Default: full recompute (== Pairs). DynamicTree
            // overrides with a move-buffer + persistent pair set.
            virtual int UpdatePairs(std::vector<BroadphasePair>& out) { return Pairs(out); }

            // ---- Seams for parallel broadphase pair maintenance (Phase D2, Task 2) ----
            //
            // DynamicTree overrides all three. Implementations that do NOT support
            // incremental pair sets get serial-fallback defaults that are correct but
            // perform a full recompute (EvictTouchedAndCollectMoved yields nothing;
            // QueryProxyPairs is a no-op; MergeAndEmit ignores perWorker and calls Pairs).

            // STEP 1 seam: evict stale pairs + snapshot the moved-proxy ids.
            // Clears m_moved / m_removed after the snapshot.
            // Base default: movedOut is empty -- no incremental state, nothing to do.
            virtual void EvictTouchedAndCollectMoved(std::vector<std::uint32_t>& movedOut)
            {
                movedOut.clear();
            }

            // STEP 2 seam: fat-descent + tight-filter for ONE proxy id.
            // Appends canonical (lo<<32|hi) keys to out. Read-only (const).
            // Uses the caller-supplied stack (NOT any shared member) so callers
            // can provide per-worker scratch in Task 3.
            // Base default: no-op (no tree to query).
            virtual void QueryProxyPairs(std::uint32_t id,
                                         std::vector<std::uint32_t>& stack,
                                         std::vector<std::uint64_t>& out) const
            {
                (void)id; (void)stack; (void)out;
            }

            // STEP 3 seam: merge per-worker key buffers into the persistent pair set,
            // then emit sorted BroadphasePair output.
            // Base default: ignore perWorker, fall back to full Pairs() recompute.
            virtual int MergeAndEmit(std::span<const std::vector<std::uint64_t>> perWorker,
                                     std::vector<BroadphasePair>& out)
            {
                (void)perWorker; return Pairs(out);
            }
        };

    } // namespace Physics
} // namespace Manifold2D
