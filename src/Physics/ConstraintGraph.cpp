#include <Manifold2D/Physics/ConstraintGraph.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>

#include <Manifold2D/Physics/Contact.hpp>                    // Contact + ContactPool + kInvalidColor
#include <Manifold2D/Physics/PhysicsWorld.hpp>               // the world SoA + island/awake seams (befriended)
#include <Manifold2D/Physics/Broadphase/DynamicTree.hpp>     // DynamicTree::kMargin + TryGetFatBox (fat-box gates)
#include <Manifold2D/Physics/Narrowphase/GeometryKernel.hpp> // AabbOverlap
#include <Manifold2D/Physics/Narrowphase/Collide.hpp>        // Collide (fixture-pair narrowphase)
#include <Manifold2D/Physics/Shapes.hpp>                     // MakeAabb (tile-span virtual fixtures)
#include <Manifold2D/Physics/Island.hpp>                     // Island::kInvalidIsland (merge drain)
#include <Manifold2D/Physics/Solver/ContactColoring.hpp>     // kColorCount

namespace Manifold2D
{
    namespace Physics
    {
        ConstraintGraph::ConstraintGraph()
        {
            // Phase C, Task 4: one contact-id list per graph color. Sized once here
            // to kColorCount (overflow contacts are not listed -- they carry
            // color == kInvalidColor and never enter m_colorContacts). m_bodyColorMask
            // grows lazily with the body SoA via Grow (default 0).
            m_colorContacts.resize(static_cast<std::size_t>(kColorCount));
        }

        // ----------------------------------------------------------------
        // Persistent incremental contact coloring (Phase C, Stage 2, Tasks 4-5)
        // ----------------------------------------------------------------
        //
        // A solver-relevant body-body contact is colored ONCE at create and frees
        // its color at destroy -- the incremental replacement for the per-step
        // greedy recolor. Task 5 makes the solver CONSUME this coloring
        // (EmitContactConstraints copies Contact::color onto the emitted constraint;
        // SoftStep buckets by it instead of recoloring), so the color is now
        // load-bearing -- the mask GATES this lowest-free search. This persistent
        // coloring is a DIFFERENT but equally-valid color partition than the old
        // per-step greedy one; because the colored solve is Gauss-Seidel (color k's
        // velocity updates feed color k+1), a different valid partition is an
        // INTENTIONAL re-baseline vs pre-Phase-C main (different floats), NOT
        // bit-identical. The contract that holds is run-twice DETERMINISM + the
        // behavioral [physics] suite (no exact goldens) -- per the engine's
        // re-baseline-numerics-on-purpose rule. ValidatePersistentColoring
        // cross-checks the mask against the lists.

        void ConstraintGraph::AssignContactColor(std::uint32_t id,
                                                 std::uint32_t a, std::uint32_t b,
                                                 bool aDyn, bool bDyn)
        {
            // Lowest free color: one whose bit is unset in BOTH dynamic endpoints'
            // masks. A static/kinematic endpoint never blocks (it is read-only in
            // the solve, so sharing it across a color is harmless -- mirrors
            // ColorConstraints' aDyn/bDyn rule).
            int chosen = -1;
            for (int k = 0; k < kColorCount; ++k)
            {
                const std::uint32_t bit = 1u << k;
                const bool aFree = !aDyn || !(m_bodyColorMask[a] & bit);
                const bool bFree = !bDyn || !(m_bodyColorMask[b] & bit);
                if (aFree && bFree) { chosen = k; break; }
            }
            Contact& c = m_contactPool.Get(id);
            if (chosen < 0)
            {
                // OVERFLOW: no free color. The contact stays uncolored
                // (kInvalidColor) and is NOT listed in m_colorContacts; the solver
                // (Task 5) will solve it in the scalar tail.
                c.color = kInvalidColor;
                return;
            }
            c.color = static_cast<std::uint8_t>(chosen);
            const std::uint32_t bit = 1u << chosen;
            // Occupy the color bit on each DYNAMIC endpoint only.
            if (aDyn) m_bodyColorMask[a] |= bit;
            if (bDyn) m_bodyColorMask[b] |= bit;
            m_colorContacts[chosen].push_back(id);
        }

        void ConstraintGraph::ReleaseContactColor(PhysicsWorld& w, std::uint32_t id)
        {
            Contact& c = m_contactPool.Get(id);
            const std::uint8_t col = c.color;
            if (col == kInvalidColor)
            {
                // Sensor / non-solver / span / overflow contact -- never colored,
                // never in m_colorContacts. Nothing to release.
                return;
            }

            // RATIONALE: the coloring invariant (no two same-color contacts share a
            // dynamic body) means each body has AT MOST ONE contact per color, so
            // clearing the body's bit for this color is exact -- no OTHER live
            // contact of this body occupies the same color, so we never strip a bit
            // a sibling contact still needs.
            //
            // Recompute dyn-ness from the cached body slots (a body's type is fixed
            // for its life -- m_btype is set only in AddBody -- so this matches the
            // aDyn/bDyn passed at AssignContactColor). A SPAN (c.bIsBody == false)
            // has no real B body, so only A can be a dynamic endpoint there.
            const std::uint32_t bit = 1u << col;
            const std::uint32_t bA  = c.bodyA;
            const std::uint32_t bB  = c.bodyB;
            const bool aDyn = (bA != kInvalidSlot) &&
                              (static_cast<BodyType>(w.m_btype[bA]) == BodyType::Dynamic);
            const bool bDyn = c.bIsBody && (bB != kInvalidSlot) &&
                              (static_cast<BodyType>(w.m_btype[bB]) == BodyType::Dynamic);
            if (aDyn) m_bodyColorMask[bA] &= ~bit;
            if (bDyn) m_bodyColorMask[bB] &= ~bit;

            // Swap-remove id from this color's contact list (order-independent --
            // the list is a set, never walked in a determinism-sensitive order).
            // TODO(perf): O(1) swap-remove by a stored per-contact index if destroy
            // frequency shows on a profile.
            std::vector<std::uint32_t>& list = m_colorContacts[col];
            for (std::size_t i = 0; i < list.size(); ++i)
            {
                if (list[i] == id)
                {
                    list[i] = list.back();
                    list.pop_back();
                    break;
                }
            }
            c.color = kInvalidColor;
        }

        std::uint8_t ConstraintGraph::ContactColorOf(std::uint32_t id) const
        {
            // Read-only probe (not used by the Step path). Get() asserts liveness
            // in Debug; the caller is expected to pass a live id. A dead/recycled
            // slot carries kInvalidColor (the EnsurePair recycle reset), so the
            // value is meaningful even on the recycled path.
            return m_contactPool.Get(id).color;
        }

        bool ConstraintGraph::ValidatePersistentColoring(const PhysicsWorld& w) const
        {
            // For each color, no DYNAMIC body slot may appear in two contacts, and
            // every listed contact must be alive AND tagged with this color. Task 5
            // also cross-checks the per-body color MASK against the lists: now that
            // the mask is load-bearing (it GATES AssignContactColor's lowest-free
            // search), a mask/list divergence would silently corrupt the coloring, so
            // reconstruct the mask from the lists and require it to match m_bodyColorMask
            // bit-for-bit -- bit k is set IFF the slot has exactly one contact in
            // m_colorContacts[k] (uniqueness is enforced by the per-color seen check
            // below; "no stray bits" is enforced by the final equality).
            std::vector<std::uint8_t>  seen;      // per-body-slot "claimed this color"
            std::vector<std::uint32_t> rebuilt(m_bodyColorMask.size(), 0u); // mask from the lists
            for (int k = 0; k < kColorCount; ++k)
            {
                seen.assign(m_bodyColorMask.size(), 0u);
                const std::uint32_t bit = 1u << k;
                const std::vector<std::uint32_t>& list = m_colorContacts[static_cast<std::size_t>(k)];
                for (const std::uint32_t id : list)
                {
                    const Contact& c = m_contactPool.Get(id); // asserts alive in Debug
                    if (c.color != static_cast<std::uint8_t>(k))
                    {
                        return false; // listed under the wrong color
                    }
                    const std::uint32_t bA = c.bodyA;
                    const std::uint32_t bB = c.bodyB;
                    const bool aDyn = (bA != kInvalidSlot) &&
                                      (static_cast<BodyType>(w.m_btype[bA]) == BodyType::Dynamic);
                    const bool bDyn = c.bIsBody && (bB != kInvalidSlot) &&
                                      (static_cast<BodyType>(w.m_btype[bB]) == BodyType::Dynamic);
                    if (aDyn)
                    {
                        if (seen[bA] != 0u) return false; // dynamic body twice in one color
                        seen[bA] = 1u;
                        rebuilt[bA] |= bit;
                    }
                    if (bDyn)
                    {
                        if (seen[bB] != 0u) return false;
                        seen[bB] = 1u;
                        rebuilt[bB] |= bit;
                    }
                }
            }
            // The mask must equal what the lists imply -- catches a set bit with no
            // backing contact (a missed ReleaseContactColor) or a contact in a list
            // whose mask bit was never set (a missed AssignContactColor mask write).
            for (std::size_t s = 0; s < m_bodyColorMask.size(); ++s)
            {
                if (m_bodyColorMask[s] != rebuilt[s]) { return false; }
            }
            return true;
        }

        std::size_t ConstraintGraph::ColoredContactCount() const noexcept
        {
            // Sum the per-color lists. Read-only probe (not on the Step path): the
            // [phasec] coloring-validity test asserts this > 0 so the oracle cannot
            // trivially pass on an EMPTY coloring.
            std::size_t n = 0;
            for (const std::vector<std::uint32_t>& list : m_colorContacts)
            {
                n += list.size();
            }
            return n;
        }

        // ---- world-lifecycle seams (PhysicsWorld drives these) ---------------

        void ConstraintGraph::Grow(std::uint32_t next)
        {
            // Phase C, Task 4: per-body color-occupancy bitmask. A fresh/recycled
            // slot starts with NO colors occupied (the RemoveBody leak-detector
            // asserts a removed body left mask 0, so a recycled slot is always 0).
            m_bodyColorMask.resize(next, 0u);
        }

        namespace
        {
            // Combine the geometric feature id (from Collide) with the FIXTURE-PAIR
            // identity so two DIFFERENT fixture pairs on the SAME body pair get
            // DISTINCT, STABLE warm-start keys. The solver's warm-start cache is
            // keyed by this id; without the fixture mix two fixtures of a compound
            // body that hit the same feature of the other body produce the SAME id
            // and ALIAS in the cache (one fixture-pair's accumulated impulse would
            // seed the other's contact point at a different lever arm -> degraded
            // warm start / penetration resistance for every compound body). The
            // mix is a deterministic integer hash (no fp; stable per (base, fixA,
            // fixB) across steps -- the warm-start invariant only needs the same
            // physical contact to map to the SAME id each step, never a particular
            // value, so this is physics-neutral for any non-aliasing contact).
            //
            // Legacy single-shape bodies (no fixtures -> both indices kInvalidSlot)
            // pass the base id through UNCHANGED: there is exactly one geometric
            // pair per body pair so the base is already unique, and byte-identity
            // is preserved for the pre-fixture-era paths.
            [[nodiscard]] inline std::uint32_t MixContactId(std::uint32_t base,
                                                            std::uint32_t fixA,
                                                            std::uint32_t fixB) noexcept
            {
                if (fixA == kInvalidSlot && fixB == kInvalidSlot)
                {
                    return base;
                }
                // 64-bit MurmurHash3-style finalizer over (base, fixA, fixB).
                std::uint64_t h = static_cast<std::uint64_t>(base);
                h = (h ^ (static_cast<std::uint64_t>(fixA) + 0x9E3779B97F4A7C15ull))
                    * 0xFF51AFD7ED558CCDull;
                h = (h ^ (static_cast<std::uint64_t>(fixB) + 0xC2B2AE3D27D4EB4Full))
                    * 0xFF51AFD7ED558CCDull;
                h ^= h >> 33;
                return static_cast<std::uint32_t>(h);
            }
        } // namespace

        bool ConstraintGraph::BothAsleep(const PhysicsWorld& w, const Contact& c) const noexcept
        {
            // A static/kinematic body never wakes the recompute on its own, so it
            // counts as "asleep" here. w.m_awake is 1 for static/kinematic (they are
            // never integrated/slept), so guard on body TYPE: only a DYNAMIC body
            // being awake forces a recompute. Mirrors GenerateContacts' awake gate
            // (it only ever generates for an awake DYNAMIC A; a static/kinematic B
            // does not drive the narrowphase by itself).
            auto bodyAwake = [&](std::uint32_t b) -> bool
            {
                if (b >= w.m_count || w.m_alive[b] == 0)
                {
                    return false;
                }
                return static_cast<BodyType>(w.m_btype[b]) == BodyType::Dynamic &&
                       w.m_awake[b] != 0;
            };
            const bool aAwake = bodyAwake(c.bodyA);
            const bool bAwake = c.bIsBody && bodyAwake(c.bodyB);
            return !aAwake && !bAwake;
        }

        bool ConstraintGraph::FatBoxesOverlap(const PhysicsWorld& w, const Contact& c,
                                           Real extraMargin) const noexcept
        {
            // Fat box of a fixture slot: a MOVER fixture's fat box lives in the
            // DynamicTree (margin-grown, the broadphase invariant); a STATIC
            // fixture is not in the mover tree, so reconstruct its fat box as the
            // tight fixture AABB grown by the SAME tree margin (DynamicTree::kMargin)
            // -- the consistent fat-box definition the broadphase would use.
            // w.FixtureBroadphaseTree() is non-null ONLY for a DynamicTree mover
            // broadphase; it returns nullptr for SpatialHash / SweepAndPrune. The
            // tight + DynamicTree::kMargin reconstruction below assumes the tree's
            // margin invariant, so for a non-DynamicTree WorldDef::broadphase even
            // mover fixtures fall to that conservative fallback (a fixture's contact
            // may persist slightly longer than its true fat box). The default /
            // production broadphase is DynamicTree, so this is latent.
            const DynamicTree* tree = w.FixtureBroadphaseTree();
            auto fatOf = [&](std::uint32_t fxSlot, std::uint32_t bodySlot) -> Aabb2
            {
                // Mover fixture (Dynamic/Kinematic): its proxy is in the tree.
                // O(1) id->fat lookup (vs. the old O(leaves) ForEachLeaf scan).
                if (tree != nullptr && bodySlot < w.m_count && w.m_alive[bodySlot] != 0 &&
                    static_cast<BodyType>(w.m_btype[bodySlot]) != BodyType::Static)
                {
                    Aabb2 fat{};
                    if (tree->TryGetFatBox(fxSlot, fat))
                    {
                        return fat;
                    }
                }
                // Static fixture (or no tree): tight AABB grown by the tree margin.
                const Aabb2 tight = w.FixtureAabb(fxSlot);
                const Real  m     = DynamicTree::kMargin;
                return Aabb2{ Vec2(tight.min.x - m, tight.min.y - m),
                              Vec2(tight.max.x + m, tight.max.y + m) };
            };
            Aabb2 fatA = fatOf(c.a.index, c.bodyA);
            Aabb2 fatB = fatOf(c.b.index, c.bodyB);
            // Speculative widening: grow BOTH boxes by half the extra margin so the
            // overlap test admits a pair whose CLOSING distance this Step is up to
            // `extraMargin` (a fast mover approaching a wall). Zero by default keeps
            // the plain fat-box gate (the resting / settled persistence behavior).
            if (extraMargin > Real(0))
            {
                const Real h = extraMargin * Real(0.5);
                fatA.min.x -= h; fatA.min.y -= h; fatA.max.x += h; fatA.max.y += h;
                fatB.min.x -= h; fatB.min.y -= h; fatB.max.x += h; fatB.max.y += h;
            }
            return AabbOverlap(fatA, fatB);
        }

        void ConstraintGraph::WakeMoverPair(PhysicsWorld& w, std::uint32_t fa, std::uint32_t fb)
        {
            // Wake-on-contact (Task 4): ports the rule the retired GenerateContacts
            // ran in its mover-mover loop (PhysicsWorld.lua:369-382). A sleeping
            // dynamic touched by an awake mover wakes so the island re-forms; the
            // [physics][island] "new contact wakes a sleeping body" test gates this.
            //
            // Apply the gates the old GenerateContacts mover-mover loop ran BEFORE
            // its wake block (same body, alive, BODY sensor, da/db) on the
            // PRE-ORIENTATION (a, b). The fixture-sensor gate is INTENTIONALLY NOT
            // applied here: in the old path the wake ran first and the fixture-sensor
            // `continue` came AFTER it (skipping only the CONTACT/constraint, not the
            // wake). TryCreateContact still applies the fixture-sensor gate, so a
            // pair overlapping only via a sensor fixture wakes but creates no contact
            // -- byte-identical to the old ordering.
            const std::uint32_t a = w.m_fxBody[fa];
            const std::uint32_t b = w.m_fxBody[fb];
            if (a == b)
            {
                return;
            }
            if (a >= w.m_count || b >= w.m_count || w.m_alive[a] == 0 || w.m_alive[b] == 0)
            {
                return;
            }
            if (w.m_sensor[a] != 0 || w.m_sensor[b] != 0)
            {
                return;
            }
            const bool da = static_cast<BodyType>(w.m_btype[a]) == BodyType::Dynamic;
            const bool db = static_cast<BodyType>(w.m_btype[b]) == BodyType::Dynamic;
            if (!da && !db)
            {
                return; // kinematic-kinematic: no dynamic response, nothing to wake
            }
            // Only a NON-IDLE (moving) mover wakes a sleeping neighbour. A body idle
            // enough to be a sleep candidate itself (same predicate as
            // IslandManager::UpdateSleep) must NOT wake its sleeping neighbours -- otherwise
            // two near-resting bodies in different islands (a sub-millimeter gap; NOT
            // touching) ping-pong each other awake forever (each wakes the other the
            // step it sleeps). A real mover (thrown body, moving/spinning kinematic)
            // is non-idle and still wakes. (Static wakers never reach here -- this
            // loop is the mover-mover broadphase.)
            auto moverIsMoving = [&](std::uint32_t s) -> bool {
                // Same combined test as IslandManager::UpdateSleep: a mover "is moving"
                // (and thus wakes a sleeping neighbour) iff it is NOT idle, i.e.
                // |v| + |w|*maxExtent >= its sleepThreshold. Keeps the wake + sleep
                // predicates consistent at the threshold margin.
                const Real lin = std::sqrt(w.m_velX[s] * w.m_velX[s] + w.m_velY[s] * w.m_velY[s]);
                return (lin + std::fabs(w.m_angVel[s]) * w.m_maxExtent[s]) >= w.m_sleepThreshold[s];
            };
            // Wake a sleeping dynamic touched by an awake mover (a static/kinematic
            // counterpart reports awake; a dynamic counterpart must itself be awake).
            // The wake GATING here matches the old GenerateContacts pre-orientation
            // wake byte-for-byte -- including fixture-sensor pairs, which the old
            // path woke before reaching its fixture-sensor `continue` (that skip
            // gated the constraint, not the wake).
            if (da && w.m_awake[a] == 0 && (!db || w.m_awake[b] != 0) && moverIsMoving(b))
            {
                w.m_awake[a]      = 1;
                w.m_sleepTimer[a] = Real(0);
                w.AddToAwakeSet(a); // Phase B: kInvalidIsland safety net
                w.WakeIsland(a); // wake the sleeper's whole island (Box2D contact wake)
            }
            if (db && w.m_awake[b] == 0 && (!da || w.m_awake[a] != 0) && moverIsMoving(a))
            {
                w.m_awake[b]      = 1;
                w.m_sleepTimer[b] = Real(0);
                w.AddToAwakeSet(b); // Phase B: kInvalidIsland safety net
                w.WakeIsland(b); // wake the sleeper's whole island (Box2D contact wake)
            }
        }

        void ConstraintGraph::TryCreateContact(PhysicsWorld& w, std::uint32_t fa, std::uint32_t fb)
        {
            // Resolve owning body slots (mover-mover orientation rule).
            std::uint32_t a = w.m_fxBody[fa];
            std::uint32_t b = w.m_fxBody[fb];

            // Same body -> two fixtures of one body never collide.
            if (a == b)
            {
                return;
            }
            if (a >= w.m_count || b >= w.m_count || w.m_alive[a] == 0 || w.m_alive[b] == 0)
            {
                return;
            }
            // Collision filter (Box2D rule): collide iff each side's category is in the
            // other's mask. A filtered-out pair never enters the pool -> no solve, no event.
            if (((w.m_fxFilterCat[fa] & w.m_fxFilterMask[fb]) == 0u) ||
                ((w.m_fxFilterCat[fb] & w.m_fxFilterMask[fa]) == 0u))
            {
                return;
            }
            // PHASE 4, Task 1: the sensor skip + the `!da && !db` skip are REMOVED
            // from creation -- a mover<->mover overlapping fixture-pair now ALWAYS
            // creates a contact (sensors + kinematic-kinematic INCLUDED), so the
            // pool covers the EVENT union, not just the solver-relevant pairs. The
            // SOLVER feed stays byte-identical because each contact is tagged
            // solverRelevant below and EmitContactConstraints emits only those.

            const bool da = static_cast<BodyType>(w.m_btype[a]) == BodyType::Dynamic;
            const bool db = static_cast<BodyType>(w.m_btype[b]) == BodyType::Dynamic;

            // ORIENT: prefer A dynamic (so the solver-side A is the dynamic body,
            // matching GenerateContacts' (fixA, fixB) order -- Task 3's oracle +
            // MixContactId rely on this). If neither is dynamic (kinematic-kinematic,
            // now poolable), fall back to the LOWER BODY SLOT as A so the orientation
            // is deterministic. Swap the BODY index + its FIXTURE index together.
            std::uint32_t ia = a,  ib = b;
            std::uint32_t fia = fa, fib = fb;
            const bool swap = da ? (db && ib < ia)   // both dynamic: lower slot is A
                                 : (db || ib < ia);  // B dynamic, or neither -> lower slot is A
            if (swap)
            {
                std::swap(ia, ib);
                std::swap(fia, fib);
            }

            // SOLVER RELEVANCE (the OLD create filter): true iff a dynamic body is
            // present AND neither side is a sensor (body-level OR per-fixture). The
            // sensor check uses the ORIENTED fixtures so each side's body+fixture is
            // tested consistently. dynamic-dynamic / dynamic-kinematic / dynamic-static
            // non-sensor -> true; sensors + kinematic-kinematic -> false.
            const bool sensorA = (w.m_sensor[ia] != 0) || (w.m_fxSensor[fia] != 0u);
            const bool sensorB = (w.m_sensor[ib] != 0) || (w.m_fxSensor[fib] != 0u);
            const bool solverRelevant = (da || db) && !sensorA && !sensorB;

            // EVENT RELEVANCE (Phase 4, Task 2): events fire for every pooled
            // body-pair EXCEPT dynamic-vs-static-body (the design's explicit
            // exclusion -- the solver owns dynamic-vs-static response; events are
            // gameplay triggers). A static body is `TypeSlot == Static`; the only
            // created pairs are mover-mover, dynamic-static, kinematic-static (tiles
            // never reach the pool), so this filter leaves mover-mover (sensors +
            // kinematic-kinematic included) + kinematic-static event-relevant and
            // excludes ONLY dynamic-static. Uses the ORIENTED (ia, ib) types so it
            // reads symmetrically; a/b vs ia/ib is identical (orientation only swaps
            // the two slots, not their type set).
            const bool aStatic =
                static_cast<BodyType>(w.m_btype[ia]) == BodyType::Static;
            const bool bStatic =
                static_cast<BodyType>(w.m_btype[ib]) == BodyType::Static;
            const bool aDyn =
                static_cast<BodyType>(w.m_btype[ia]) == BodyType::Dynamic;
            const bool bDyn =
                static_cast<BodyType>(w.m_btype[ib]) == BodyType::Dynamic;
            const bool eventRelevant = !((aDyn && bStatic) || (aStatic && bDyn));

            const FixtureHandle hA{ fia, w.m_fxGen[fia] };
            const FixtureHandle hB{ fib, w.m_fxGen[fib] };
            const ContactPool::EnsureResult r = m_contactPool.EnsurePair(hA, hB);
            if (r.created)
            {
                // EnsurePair already stored c.a = hA / c.b = hB on the fresh slot
                // (and Destroy re-keys from them, so we must NOT overwrite them).
                // We only fill the body slots + bIsBody + solver/event relevance,
                // which the pool defaults until the caller sets them (created == true).
                Contact& c = m_contactPool.Get(r.id);
                c.bodyA          = ia;
                c.bodyB          = ib;
                c.bIsBody        = true;
                c.solverRelevant = solverRelevant;
                c.eventRelevant  = eventRelevant;

                // Phase C, Task 4: assign a persistent graph color to a NEW
                // solver-relevant body-body contact (assign-at-create). Sensors,
                // non-solver, and span contacts stay uncolored (kInvalidColor).
                // Every created contact here is body-body (bIsBody == true), but
                // keep the explicit gate for intent.
                //
                // Pass the ORIENTED slots ia/ib together with the ORIENTED dyn
                // flags aDyn/bDyn (computed above from w.m_btype[ia]/w.m_btype[ib]),
                // NOT the pre-orientation da/db: the swap moves the slots but not
                // the da/db labels, so under a swap da/db would describe the wrong
                // endpoint. ReleaseContactColor + ValidatePersistentColoring both
                // recompute dyn-ness from w.m_btype[bodyA]/w.m_btype[bodyB] (== ia/ib),
                // so assign MUST key off the same oriented flags to stay consistent.
                if (solverRelevant && c.bIsBody)
                {
                    AssignContactColor(r.id, ia, ib, aDyn, bDyn);
                }
                // Per-body contact adjacency (G1 island-split linkage): a dyn-dyn
                // body contact is an island edge -> record it on BOTH endpoints so
                // SplitIsland can walk only this island's contacts. aDyn/bDyn are
                // the ORIENTED dyn flags (w.m_btype[ia]/w.m_btype[ib]); bodyA is
                // canonical-dynamic, so this fires exactly for dyn-dyn pairs.
                // Gate on solverRelevant: a sensor dyn-dyn pair must NOT become
                // an island edge (sensors fire events but must not merge islands).
                // The GATE stays here (it reads world SoA); the attach is delegated
                // to IslandManager, which owns the adjacency (decomp step 1 Task 3).
                if (aDyn && bDyn && solverRelevant)
                {
                    w.m_islandMgr.AttachContactAdjacency(ia, ib, r.id);
                }
            }
            // On a non-created HIT we leave the existing contact untouched (its
            // body slots + manifold persist) -- mirrors EnsurePair's contract.
        }

        void ConstraintGraph::UpdateOneContact(const PhysicsWorld& w, std::uint32_t id, Contact& c,
                                            Real moveDt, Real threshSq) noexcept
        {
            (void)id;
            c.npState = 0;

            // Stale-handle: a removed/recycled fixture or dead body -> flag destroy.
            if (!w.FixtureSlotLive(c.a) || (c.bIsBody && !w.FixtureSlotLive(c.b)))
            {
                c.npState |= kNpDestroy;
                return;
            }

            // Velocity-scaled speculative margin (CCD) over the two bodies.
            const Real speedSqA = w.m_velX[c.bodyA] * w.m_velX[c.bodyA] +
                                  w.m_velY[c.bodyA] * w.m_velY[c.bodyA];
            const Real speedSqB = w.m_velX[c.bodyB] * w.m_velX[c.bodyB] +
                                  w.m_velY[c.bodyB] * w.m_velY[c.bodyB];
            const Real maxSpeedSq = std::max(speedSqA, speedSqB);
            const Real margin = (maxSpeedSq > threshSq)
                                    ? std::sqrt(maxSpeedSq) * moveDt
                                    : kSkin;

            // Fat-box separation (widened by the speculative margin) -> flag destroy.
            const Real extra = std::max(Real(0), margin - DynamicTree::kMargin);
            if (!FatBoxesOverlap(w, c, extra))
            {
                c.npState |= kNpDestroy;
                return;
            }

            // Both asleep (and not an event-only pair) -> keep cached manifold, no work.
            const bool eventOnly = c.eventRelevant && !c.solverRelevant;
            if (BothAsleep(w, c) && !eventOnly)
            {
                return;
            }

            // Warm-start carry-forward snapshot, then recompute the manifold.
            const Manifold oldManifold = c.manifold;
            const Transform xfA = PhysicsWorld::ComposeFixtureXf(
                Vec2(w.m_posX[c.bodyA], w.m_posY[c.bodyA]), w.m_angle[c.bodyA],
                Vec2(w.m_fxLocalPosX[c.a.index], w.m_fxLocalPosY[c.a.index]),
                w.m_fxLocalAngle[c.a.index]);
            const Transform xfB = PhysicsWorld::ComposeFixtureXf(
                Vec2(w.m_posX[c.bodyB], w.m_posY[c.bodyB]), w.m_angle[c.bodyB],
                Vec2(w.m_fxLocalPosX[c.b.index], w.m_fxLocalPosY[c.b.index]),
                w.m_fxLocalAngle[c.b.index]);
            c.manifold = Collide(w.m_fxShape[c.a.index], xfA,
                                 w.m_fxShape[c.b.index], xfB, margin);

            const bool wasTouching = c.touching;
            c.touching = (c.manifold.pointCount > 0);

            // Classify the dyn-dyn touch transition (island edge) -> flag for the tail.
            // Gate on c.solverRelevant: sensor dyn-dyn pairs must never trigger a
            // merge (kNpStarted) or split (kNpStopped) -- they fire events but must
            // not couple rigid islands.
            if (c.solverRelevant && c.bIsBody && c.bodyB != kInvalidSlot &&
                w.TypeSlot(c.bodyA) == BodyType::Dynamic &&
                w.TypeSlot(c.bodyB) == BodyType::Dynamic)
            {
                if (!wasTouching && c.touching)      { c.npState |= kNpStarted; }
                else if (wasTouching && !c.touching) { c.npState |= kNpStopped; }
            }

            // Warm-start: copy impulses forward by feature id (<=2x2 fixed loop).
            for (int np = 0; np < c.manifold.pointCount; ++np)
            {
                ManifoldPoint& nm = c.manifold.points[np];
                for (int op = 0; op < oldManifold.pointCount; ++op)
                {
                    const ManifoldPoint& om = oldManifold.points[op];
                    if (om.id == nm.id)
                    {
                        nm.normalImpulse  = om.normalImpulse;
                        nm.tangentImpulse = om.tangentImpulse;
                        break;
                    }
                }
            }
        }

        void ConstraintGraph::UpdateContacts(PhysicsWorld& w, Real dt)
        {
            // Phase A per-step island merge scratch: reset here so the apply pass
            // below only sees pairs collected THIS step. clear() keeps capacity
            // -> zero steady-state allocation after the first few steps.
            m_pendingMerges.clear();

            // ---- 1. CREATE: a contact for every fixture-pair in the EVENT UNION --
            // (solver-relevant pairs + the event-only tail: sensors,
            //  kinematic-kinematic, kinematic-vs-static-body; tiles stay out).
            //
            // (a) mover<->mover: the Phase-2 incremental fixture-pair set (sorted
            //     fa < fb). Phase D2 Task 3: parallel broadphase pair-finding.
            //     Serial seams: EvictTouchedAndCollectMoved (snapshot moved ids,
            //     evict stale pairs) and MergeAndEmit (union per-worker key sets
            //     into the persistent pair set, emit sorted pairs) bracket a
            //     per-proxy QueryProxyPairs ParallelFor. Each worker uses its OWN
            //     stack and key buffer (disjoint write) so the tree descent is
            //     read-only under parallelism. MergeAndEmit + the sorted output
            //     are order-independent -> byte-identical at any worker count.
            //
            //     The serial UpdatePairs wrapper (tests/oracle) is unchanged.
            {
                auto* bp   = w.m_fixtureBroadphase.get();
                auto* exec = w.Executor();   // always non-null (serial fallback)

                bp->EvictTouchedAndCollectMoved(m_bpMovedScratch);

                const auto W = static_cast<std::size_t>(exec->WorkerCount());
                if (m_bpFindScratch.size()  < W) m_bpFindScratch.resize(W);
                if (m_bpStackScratch.size() < W) m_bpStackScratch.resize(W);
                for (auto& s : m_bpFindScratch) s.clear(); // per-step clear; capacity retained

                exec->ParallelFor(m_bpMovedScratch.size(), kBroadphaseGrain,
                    [&](std::size_t b, std::size_t e, std::uint32_t w) {
                        for (std::size_t k = b; k < e; ++k)
                            bp->QueryProxyPairs(m_bpMovedScratch[k],
                                                m_bpStackScratch[w],
                                                m_bpFindScratch[w]);
                    });

                bp->MergeAndEmit(
                    std::span<const std::vector<std::uint64_t>>(m_bpFindScratch.data(), W),
                    m_cpPairs);
            }
            for (const BroadphasePair& p : m_cpPairs)
            {
                WakeMoverPair(w, p.a, p.b);
                TryCreateContact(w, p.a, p.b);
            }

            // (b) mover<->static-BODY + tile SPANS: per awake non-sensor DYNAMIC
            //     body, the StaticCandidates lists. MIRRORS the legacy
            //     GenerateContacts static path (the query pad + the genStatics loop)
            //     AND its tile-SPAN path (Task 4). A static body's fixture is a real
            //     fixture slot, so we pair each (dynamic fixture, static fixture)
            //     into the PERSISTENT pool the same way TryCreateContact does. Tile
            //     spans are virtual fixtures (no slot), so they go into the TRANSIENT
            //     m_spanContacts scratch (cleared here, refilled each Step).
            //
            // Rerouted to ForEachAwake (Phase B, Task 4): the awake-set is the
            // compact list of awake dynamic slots; sleeping dynamics are skipped
            // entirely (they cannot move, so their static candidates are stable).
            // The !Dynamic / !awake guards inside the old loop body are DROPPED
            // (the set guarantees awake-dynamic); the sensor skip is KEPT.
            // The kinematic<->static-body sub-loop (c) stays on 0..w.m_count
            // (kinematics are not in the awake-set -- YAGNI a kinematic list).
            //
            // Clear the transient span scratch -- it is rebuilt from scratch this
            // Step (spans are not pooled; they are virtual/transient fixtures).
            m_spanContacts.clear();
            m_spanCenters.clear();
            // (m_newPairs is rebuilt by the serial merge below, which clears it
            //  right before the per-worker concat -- no top-of-stage clear needed.)
            const Real moveDt = dt > Real(0) ? dt : Real(0);
            const Real threshSq = (moveDt > Real(0))
                                      ? (kSkin / moveDt) * (kSkin / moveDt)
                                      : Real(0);
            // ---- DETECT (parallel; writes ONLY per-worker scratch) ----------------
            // Each worker handles a disjoint range of m_awakeBodies[begin..end).
            // The parallel section is STRUCTURALLY MUTATION-FREE: no TryCreateContact,
            // no m_spanContacts/m_spanCenters writes, no m_newPairs writes, no pool or
            // color mutation.  Every write targets only m_spanEntriesW[worker] or
            // m_newPairsW[worker] -- the caller's own [worker] entry.
            //
            // The serial tail (after ParallelFor) concatenates the per-worker buffers,
            // stable_sorts by awakeIndex (reproducing ForEachAwake / k-ascending order;
            // within-k push order is preserved because each k is processed by exactly
            // one worker and ranges are disjoint), then appends spans and calls
            // TryCreateContact per record.  AssignContactColor runs here, serially,
            // so the persistent coloring sequence is byte-identical to the serial path.
            const std::uint32_t awakeCount = w.AwakeCount();
            const std::uint32_t cWorkers   = w.Executor()->WorkerCount();
            if (m_genSpansW.size()    < cWorkers) { m_genSpansW.resize(cWorkers); }
            if (m_genStaticsW.size()  < cWorkers) { m_genStaticsW.resize(cWorkers); }
            if (m_gridScratchW.size() < cWorkers) { m_gridScratchW.resize(cWorkers); }
            if (m_spanEntriesW.size() < cWorkers) { m_spanEntriesW.resize(cWorkers); }
            if (m_newPairsW.size()    < cWorkers) { m_newPairsW.resize(cWorkers); }
            for (std::uint32_t w = 0; w < cWorkers; ++w)
            {
                m_spanEntriesW[w].clear();
                m_newPairsW[w].clear();
            }
            w.Executor()->ParallelFor(awakeCount, /*minRange=*/kCreateGrain,
                [&](std::size_t begin, std::size_t end, std::uint32_t worker)
                {
                    std::vector<Aabb2>&         spans   = m_genSpansW[worker];
                    std::vector<std::uint32_t>& statics = m_genStaticsW[worker];
                    std::vector<std::uint32_t>& grid    = m_gridScratchW[worker];
                    std::vector<SpanEntry>&     spanOut = m_spanEntriesW[worker];
                    std::vector<NewPairRecord>& pairOut = m_newPairsW[worker];
                    for (std::size_t kk = begin; kk < end; ++kk)
                    {
                        const std::uint32_t k = static_cast<std::uint32_t>(kk);
                        const std::uint32_t i = w.AwakeBodies()[k];
                        if (w.m_sensor[i] != 0) { continue; }
                        // Query pad: max(fat-AABB margin, velocity-scaled speculative
                        // margin). The floor is DynamicTree::kMargin -- the same role
                        // Box2D's B2_AABB_MARGIN plays in pair discovery (constants.h:44).
                        const Real speedSqA   = w.m_velX[i] * w.m_velX[i] + w.m_velY[i] * w.m_velY[i];
                        const Real specMargin = (speedSqA > threshSq)
                                                    ? std::sqrt(speedSqA) * moveDt : kSkin;
                        const Aabb box = w.SlotAabb(i);
                        const Real pad = std::max(DynamicTree::kMargin, specMargin);
                        Aabb2 query;
                        query.min = Vec2(box.min.x - pad, box.min.y - pad);
                        query.max = Vec2(box.max.x + pad, box.max.y + pad);
                        // Fills BOTH spans (tile spans, processed transiently below) and
                        // statics (static bodies, pooled). Uses per-worker static-tree
                        // query scratch.
                        w.StaticCandidates(query, spans, statics, grid);

                        const std::vector<std::uint32_t>* fxListA = nullptr;
                        if (i < w.m_bodyFixtures.size() && !w.m_bodyFixtures[i].empty())
                        {
                            fxListA = &w.m_bodyFixtures[i];
                        }

                        // ---- tile spans: push SpanEntry{k, c, spanCenter} into spanOut
                        // (no m_spanContacts/m_spanCenters writes here -- per-worker only)
                        for (std::size_t s = 0; s < spans.size(); ++s)
                        {
                            const Aabb2& span = spans[s];
                            const Vec2 spanCenter = (span.min + span.max) * Real(0.5);
                            const Vec2 he = (span.max - span.min) * Real(0.5);
                            const Shape spanShape = MakeAabb(he.x, he.y);
                            const Transform xfB{ spanCenter, Real(0) };

                            if (fxListA != nullptr)
                            {
                                for (const std::uint32_t fi : *fxListA)
                                {
                                    if (fi >= w.m_fxCount || w.m_fxGen[fi] == 0u)
                                    {
                                        continue;
                                    }
                                    if (w.m_fxSensor[fi] != 0u)
                                    {
                                        continue; // sensor fixture: no constraint
                                    }
                                    const Transform xfA = PhysicsWorld::ComposeFixtureXf(
                                        Vec2(w.m_posX[i], w.m_posY[i]), w.m_angle[i],
                                        Vec2(w.m_fxLocalPosX[fi], w.m_fxLocalPosY[fi]),
                                        w.m_fxLocalAngle[fi]);
                                    const Manifold mfld = Collide(w.m_fxShape[fi], xfA,
                                                                  spanShape, xfB, specMargin);
                                    if (mfld.pointCount <= 0)
                                    {
                                        continue; // not touching -> no transient contact
                                    }
                                    Contact c;
                                    c.a        = FixtureHandle{ fi, w.m_fxGen[fi] };
                                    c.b        = FixtureHandle{}; // span has no fixture slot
                                    c.bodyA    = i;
                                    c.bodyB    = kInvalidSlot;
                                    c.bIsBody  = false;
                                    c.manifold = mfld;
                                    c.touching = true;
                                    spanOut.push_back(SpanEntry{ k, c, spanCenter });
                                }
                            }
                            else
                            {
                                // Legacy single-shape fallback (a dynamic body with no live
                                // fixtures -- AddBody always makes one, so this is defensive).
                                const Transform xfA{ Vec2(w.m_posX[i], w.m_posY[i]), w.m_angle[i] };
                                const Manifold mfld = Collide(w.m_shape[i], xfA,
                                                              spanShape, xfB, specMargin);
                                if (mfld.pointCount <= 0)
                                {
                                    continue;
                                }
                                Contact c;
                                c.a        = FixtureHandle{}; // no fixture slot
                                c.b        = FixtureHandle{};
                                c.bodyA    = i;
                                c.bodyB    = kInvalidSlot;
                                c.bIsBody  = false;
                                c.manifold = mfld;
                                c.touching = true;
                                spanOut.push_back(SpanEntry{ k, c, spanCenter });
                            }
                        }

                        if (fxListA == nullptr) { continue; }

                        // ---- static fixture pairs: EMIT records into pairOut --------
                        // (no m_newPairs.push_back / TryCreateContact here -- per-worker only)
                        for (std::size_t s = 0; s < statics.size(); ++s)
                        {
                            const std::uint32_t idx = statics[s];
                            if (idx >= w.m_count || w.m_alive[idx] == 0 || w.m_sensor[idx] != 0) { continue; }
                            const std::vector<std::uint32_t>* fxListB = nullptr;
                            if (idx < w.m_bodyFixtures.size() && !w.m_bodyFixtures[idx].empty())
                            {
                                fxListB = &w.m_bodyFixtures[idx];
                            }
                            if (fxListB == nullptr) { continue; }
                            for (const std::uint32_t fiA : *fxListA)
                            {
                                if (fiA >= w.m_fxCount || w.m_fxGen[fiA] == 0u || w.m_fxSensor[fiA] != 0u) { continue; }
                                for (const std::uint32_t fiB : *fxListB)
                                {
                                    if (fiB >= w.m_fxCount || w.m_fxGen[fiB] == 0u || w.m_fxSensor[fiB] != 0u) { continue; }
                                    pairOut.push_back(NewPairRecord{ k, fiA, fiB });
                                }
                            }
                        }
                    }
                });
            // ---- SERIAL APPLY: order by awakeIndex, reproduce ForEachAwake order ---
            // Spans: concatenate all per-worker entries + stable_sort by awakeIndex ->
            // append to m_spanContacts/m_spanCenters in the same order the serial loop
            // produced them (k ascending, within-k span-s ascending).
            {
                m_allSpans.clear(); // reuse the member buffer (zero steady-state alloc)
                for (std::uint32_t w = 0; w < cWorkers; ++w)
                    m_allSpans.insert(m_allSpans.end(), m_spanEntriesW[w].begin(), m_spanEntriesW[w].end());
                std::stable_sort(m_allSpans.begin(), m_allSpans.end(),
                    [](const SpanEntry& a, const SpanEntry& b) { return a.awakeIndex < b.awakeIndex; });
                for (const SpanEntry& e : m_allSpans)
                {
                    m_spanContacts.push_back(e.c);
                    m_spanCenters.push_back(e.center);
                }
            }
            // New pairs: same merge + sort -> TryCreateContact each (AssignContactColor
            // runs here, serially, preserving the order-dependent color assignment).
            m_newPairs.clear();
            for (std::uint32_t w = 0; w < cWorkers; ++w)
                m_newPairs.insert(m_newPairs.end(), m_newPairsW[w].begin(), m_newPairsW[w].end());
            std::stable_sort(m_newPairs.begin(), m_newPairs.end(),
                [](const NewPairRecord& a, const NewPairRecord& b) { return a.awakeIndex < b.awakeIndex; });
            for (const NewPairRecord& rec : m_newPairs)
            {
                TryCreateContact(w, rec.fiA, rec.fiB);
            }

            // (c) KINEMATIC<->static-BODY (Phase 4, Task 1): event-relevant but NOT
            //     solver-relevant. Static bodies are NOT in the mover broadphase and
            //     the dynamic-driven static-candidate loop above only covers DYNAMIC
            //     bodies, so kinematic-vs-static pairs are created here by iterating
            //     StaticList() per alive Kinematic body -- MIRRORING the old
            //     ContactManager::Step kinematic-static loop (StaticList, AABB-reject)
            //     so the create order is index-deterministic. TryCreateContact tags
            //     these solverRelevant == false (no dynamic body), so the solver feed
            //     is unchanged; the touch-state still drives the contact's manifold +
            //     `touching` in the update pass below for the event derivation (Task 2).
            //     dynamic-vs-static is ALREADY created in (b) -- this path is additive.
            {
                const std::vector<std::uint32_t>& statics = w.m_staticList;
                for (std::uint32_t i = 0; i < w.m_count; ++i)
                {
                    if (w.m_alive[i] == 0 ||
                        static_cast<BodyType>(w.m_btype[i]) != BodyType::Kinematic)
                    {
                        continue;
                    }
                    const std::vector<std::uint32_t>* fxListK = nullptr;
                    if (i < w.m_bodyFixtures.size() && !w.m_bodyFixtures[i].empty())
                    {
                        fxListK = &w.m_bodyFixtures[i];
                    }
                    if (fxListK == nullptr)
                    {
                        continue; // kinematic body with no live fixtures
                    }
                    const Aabb2 kinBox = w.SlotAabb(i);
                    for (std::size_t s = 0; s < statics.size(); ++s)
                    {
                        const std::uint32_t idx = statics[s];
                        if (idx >= w.m_count || w.m_alive[idx] == 0)
                        {
                            continue;
                        }
                        // Cheap body-union AABB reject before the per-fixture pairing
                        // (mirrors the old ContactManager AABB pre-filter).
                        if (!AabbOverlap(kinBox, w.SlotAabb(idx)))
                        {
                            continue;
                        }
                        const std::vector<std::uint32_t>* fxListB = nullptr;
                        if (idx < w.m_bodyFixtures.size() && !w.m_bodyFixtures[idx].empty())
                        {
                            fxListB = &w.m_bodyFixtures[idx];
                        }
                        if (fxListB == nullptr)
                        {
                            continue; // static body with no real fixture slot
                        }
                        for (const std::uint32_t fiK : *fxListK)
                        {
                            if (fiK >= w.m_fxCount || w.m_fxGen[fiK] == 0u)
                            {
                                continue; // dead slot (defensive); sensors INCLUDED
                            }
                            for (const std::uint32_t fiB : *fxListB)
                            {
                                if (fiB >= w.m_fxCount || w.m_fxGen[fiB] == 0u)
                                {
                                    continue; // dead slot; sensors INCLUDED (events)
                                }
                                // Orientation: A = the kinematic mover (neither is
                                // dynamic, so TryCreateContact's lower-slot tiebreak
                                // orders deterministically); for events the body-pair
                                // is canonicalized later anyway.
                                TryCreateContact(w, fiK, fiB);
                            }
                        }
                    }
                }
            }

            // ---- 2. UPDATE + DESTROY: Box2D b2Collide -- gather, parallel collide
            //         (flag only), serial apply. ------------------------------------
            // Seam 0: gather the stable live-contact id list (Box2D contactSims).
            m_npContacts.clear();
            m_contactPool.ForEach([&](std::uint32_t id, Contact&) {
                m_npContacts.push_back(id);
            });

            // Seam 1: parallel collide. Each worker recomputes its range's manifolds
            // and sets a bit (keyed on pool id) in its OWN BitSet -- no structural
            // mutation. minRange=64 (Box2D's grain); below it, runs serial on worker 0.
            const std::uint32_t workers = w.Executor()->WorkerCount();
            if (m_npStateBits.size() < workers) { m_npStateBits.resize(workers); }
            const std::size_t idCap = m_contactPool.Capacity();
            for (std::uint32_t w = 0; w < workers; ++w) {
                m_npStateBits[w].Resize(idCap);
                m_npStateBits[w].ClearAll();
            }
            w.Executor()->ParallelFor(m_npContacts.size(), /*minRange=*/64,
                [&](std::size_t begin, std::size_t end, std::uint32_t worker)
                {
                    Mosaic::BitSet& bits = m_npStateBits[worker];
                    for (std::size_t k = begin; k < end; ++k)
                    {
                        const std::uint32_t id = m_npContacts[k];
                        Contact& c = m_contactPool.Get(id);
                        UpdateOneContact(w, id, c, moveDt, threshSq);
                        if (c.npState != 0) { bits.Set(id); }
                    }
                });

            // Seam 2: serial apply. OR-reduce into bits[0], walk ascending (CTZ),
            // apply destroy / merge-edge / split per npState (ascending id == serial).
            if (workers > 0)
            {
                for (std::uint32_t w = 1; w < workers; ++w) {
                    m_npStateBits[0].InPlaceUnion(m_npStateBits[w]);
                }
                m_npStateBits[0].ForEachSetBit([&](std::uint32_t id)
                {
                    Contact& c = m_contactPool.Get(id);
                    if (c.npState & kNpDestroy)
                    {
                        // Defensive: only schedule a split for solver-relevant
                        // contacts (island edges). A sensor dyn-dyn contact was
                        // never an island edge, so destroying it cannot fracture
                        // an island -- skip the MarkSplitCandidate call.
                        if (c.solverRelevant && c.bIsBody && c.touching &&
                            c.bodyA != kInvalidSlot && c.bodyB != kInvalidSlot &&
                            c.bodyA < w.m_islandMgr.IslandIdCount() &&
                            w.TypeSlot(c.bodyA) == BodyType::Dynamic &&
                            c.bodyB < w.m_islandMgr.IslandIdCount() &&
                            w.TypeSlot(c.bodyB) == BodyType::Dynamic)
                        {
                            w.MarkSplitCandidate(w.IslandOf(c.bodyA));
                        }
                        ReleaseAndDestroyContact(w, id, c);
                    }
                    else if (c.npState & kNpStarted)
                    {
                        const std::uint32_t lo = c.bodyA < c.bodyB ? c.bodyA : c.bodyB;
                        const std::uint32_t hi = c.bodyA < c.bodyB ? c.bodyB : c.bodyA;
                        m_pendingMerges.push_back(BroadphasePair{ lo, hi });
                    }
                    else if (c.npState & kNpStopped)
                    {
                        w.MarkSplitCandidate(w.IslandOf(c.bodyA));
                    }
                });
            }

            // ---- apply queued island merges in a canonical order ----------------
            // Sort by (min,max) body slot (mirrors the m_touchedEventPairs sort) so
            // the merge sequence is run-twice-identical regardless of pool emission
            // order. Each pair re-resolves its bodies' CURRENT islands (an earlier
            // merge this step may have already united them -> MergeIslands is a
            // no-op when both resolve to the same id). No dedup needed: duplicate
            // pairs become same-island no-ops after the first merge.
            std::sort(m_pendingMerges.begin(), m_pendingMerges.end());
            for (const BroadphasePair& pr : m_pendingMerges)
            {
                const std::uint32_t ia = w.IslandOf(pr.a);
                const std::uint32_t ib = w.IslandOf(pr.b);
                if (ia != Island::kInvalidIsland &&
                    ib != Island::kInvalidIsland &&
                    ia != ib)
                {
                    // Uniform-awake invariant (Box2D "island is uniformly awake"):
                    // a begin-touch always involves at least one moving/awake body,
                    // so the merged island MUST end awake. If the two sides differ in
                    // awake state, wake the SLEEPING side's island BEFORE the merge.
                    // Otherwise MergeIslands would graft an already-sleeping singleton
                    // (a body that slept early resting purely on tile spans, which the
                    // WakeMoverPair moverIsMoving gate declined to wake) into the awake
                    // island, leaving a mixed awake/asleep island whose touching
                    // contact trips the no-sleeping-dynamic assert in
                    // EmitContactConstraints. Per-body awake flags mean this is robust
                    // to earlier merges this step (WakeIsland resolves the CURRENT
                    // island of the sleeping slot).
                    if (w.m_awake[pr.a] != w.m_awake[pr.b])
                    {
                        w.WakeIsland(w.m_awake[pr.a] == 0 ? pr.a : pr.b);
                    }
                    w.MergeIslands(ia, ib);
                }
            }
        }

        void ConstraintGraph::EmitContactConstraints(
            const PhysicsWorld& w, std::vector<ContactConstraint>& out) const
        {
            // The persistent solver feed (Phase 3, Task 4). Walk BOTH the persistent
            // m_contactPool (fixture<->fixture, ascending id) AND the transient
            // m_spanContacts (dynamic-fixture<->tile-span), emit a ContactConstraint
            // per touching+awake contact (mirroring the retired GenerateContacts
            // `emit` lambda field-for-field), then sort into the canonical
            // (bodyA, bodyB, fixtureA, fixtureB) order so the live feed is
            // run-twice-identical regardless of pool/broadphase emission order.
            // READ-ONLY w.r.t. SIM STATE: writes ONLY `out` + the persistent emit
            // scratch members (m_emitKeys/m_emitOrder/m_emitSorted -- `mutable`, so
            // this method stays `const`). The span scratch was filled by
            // UpdateContacts; this method mutates no body/contact sim state.
            out.clear();

            // Parallel sort key per emitted constraint: (bodyA, bodyB, fixA, fixB).
            // Carried from each source Contact during emit so the canonical sort
            // has the fixture slots available (ContactConstraint does not store
            // fixture slots -- only body slots). Sorted together with `out`.
            // Persistent scratch: clear() keeps capacity (no realloc after warmup).
            std::vector<EmitSortKey>& keys = m_emitKeys;
            keys.clear();

            // Emit one ContactConstraint for a single Contact. Returns true if a
            // constraint was emitted (touching + awake A). `spanCenter` is the
            // span's geometric center (used as comB for a tile span); unused when
            // c.bIsBody. Mirrors the retired GenerateContacts `emit` lambda.
            auto emitContact = [&](const Contact& c, const Vec2& spanCenter) -> bool
            {
                // Not touching -> the legacy emit early-returns on pointCount <= 0.
                if (!c.touching || c.manifold.pointCount <= 0)
                {
                    return false;
                }
                // Awake-gate: GenerateContacts only ever emitted for an AWAKE dynamic
                // A. The pool keeps contacts for sleeping mover-pairs too, so without
                // this gate an asleep pair would (wrongly) feed the solver.
                const std::uint32_t aIdx = c.bodyA;
                if (aIdx >= w.m_count || w.m_alive[aIdx] == 0 || w.m_awake[aIdx] == 0)
                {
                    return false;
                }

                const bool          bIsBody = c.bIsBody;
                const std::uint32_t bIdx    = c.bodyB;
                // fixA: the dynamic fixture slot for a fixture-path contact; for the
                // legacy single-shape span fallback c.a is an invalid handle (gen 0)
                // and fixA is kInvalidSlot (matching GenerateContacts' fallback id).
                const bool          aHasFix = (c.a.generation != 0u);
                const std::uint32_t fixA    = aHasFix ? c.a.index : kInvalidSlot;
                const std::uint32_t fixB    = bIsBody ? c.b.index : kInvalidSlot;

                const Manifold& m = c.manifold;

                ContactConstraint cc;
                cc.bodyA       = aIdx;
                cc.bodyB       = bIsBody ? bIdx : kInvalidSlot;
                cc.bodyBIsBody = bIsBody;
                cc.invMassA    = w.m_invMass[aIdx];
                cc.invInertiaA = w.m_invInertia[aIdx];
                cc.invMassB    = bIsBody ? w.m_invMass[bIdx] : Real(0);
                cc.invInertiaB = bIsBody ? w.m_invInertia[bIdx] : Real(0);
                cc.normal      = m.normal;
                cc.kind        = m.kind;

                // Combined material: friction = sqrt(fA*fB), restitution = max(rA,rB).
                //   * fixture-path (aHasFix): per-fixture material (both sides for a
                //     fixture<->fixture; for a span, body A's fixture material as both
                //     sides + restB = 0 -- exactly GenerateContacts' span emit args).
                //   * single-shape fallback (!aHasFix): body-level w.m_fric/w.m_rest as
                //     both sides + restB = 0 (the legacy fallback span emit).
                Real fricA, fricB, restA, restB;
                if (aHasFix)
                {
                    fricA = w.m_fxFriction[fixA];
                    fricB = bIsBody ? w.m_fxFriction[fixB] : w.m_fxFriction[fixA];
                    restA = w.m_fxRestitution[fixA];
                    restB = bIsBody ? w.m_fxRestitution[fixB] : Real(0);
                }
                else
                {
                    // Single-shape body A (no fixture) vs span: body-level material.
                    fricA = w.m_fric[aIdx];
                    fricB = w.m_fric[aIdx];
                    restA = w.m_rest[aIdx];
                    restB = Real(0);
                }
                cc.friction    = std::sqrt(fricA * fricB);
                cc.restitution = std::max(restA, restB);

                // Compound-COM anchors: from each body's world CENTER OF MASS
                // (WorldCom == origin for localCenter==0). A is always dynamic; B is
                // either a real body's world COM or the span's geometric center.
                const Vec2 cA = WorldCom(Vec2(w.m_posX[aIdx], w.m_posY[aIdx]),
                                         w.m_angle[aIdx],
                                         Vec2(w.m_localCenterX[aIdx],
                                              w.m_localCenterY[aIdx]));
                const Vec2 comB = bIsBody
                    ? WorldCom(Vec2(w.m_posX[bIdx], w.m_posY[bIdx]),
                               w.m_angle[bIdx],
                               Vec2(w.m_localCenterX[bIdx], w.m_localCenterY[bIdx]))
                    : spanCenter; // tile span: invInertiaB==0 zeros the lever arm

                cc.pointCount = m.pointCount;
                for (int p = 0; p < m.pointCount; ++p)
                {
                    const ManifoldPoint&    mp = m.points[p];
                    ContactConstraintPoint& cp = cc.points[p];
                    cp.anchorA        = mp.point - cA;
                    cp.anchorB        = mp.point - comB;
                    cp.baseSeparation = -mp.separation;
                    cp.id             = MixContactId(mp.id, fixA, fixB);
                    // WARM-START SEED (read path): carry the persistent Contact's
                    // accumulated impulses INTO the emitted constraint point. For a
                    // pool contact these were written back after last step's Solve
                    // (+ carried across the manifold recompute by UpdateContacts);
                    // for a transient tile span mp.normalImpulse is 0, so spans
                    // cold-start (acceptable -- they have no persistent home). The
                    // solver's Prepare leaves these untouched.
                    cp.normalImpulse  = mp.normalImpulse;
                    cp.tangentImpulse = mp.tangentImpulse;
                }
                // Phase B invariant: no emitted constraint references a SLEEPING
                // dynamic. Awake-A gate above + island-as-a-unit sleep (a touching
                // dynamic-dynamic pair shares one island, so awake-A => awake-B)
                // guarantees this. This assertion proves SyncIn can safely skip
                // sleeping dynamics (they are NEVER gathered by a live constraint).
                assert(!(static_cast<BodyType>(w.m_btype[aIdx]) == BodyType::Dynamic &&
                         w.m_awake[aIdx] == 0));
                assert(!(bIsBody &&
                         static_cast<BodyType>(w.m_btype[bIdx]) == BodyType::Dynamic &&
                         w.m_awake[bIdx] == 0));

                out.push_back(cc);
                keys.push_back(EmitSortKey{ aIdx, cc.bodyB, fixA, fixB });
                return true;
            };

            // (a) fixture<->fixture: the persistent pool (ascending id).
            // SOLVER-RELEVANCE FILTER (Phase 4, Task 1): the pool now holds the
            // EVENT union (sensors + kinematic-kinematic + kinematic-vs-static-body)
            // in addition to the solver pairs. Emit a ContactConstraint ONLY for a
            // solverRelevant contact, so the solver feed stays byte-identical to
            // Phase 3 even though the pool is a superset. (Spans below are always
            // solver-relevant -- a dynamic fixture vs a tile span -- and are NOT
            // gated here; they carry the default solverRelevant==false, so the gate
            // must NOT be inside the shared emitContact lambda.)
            m_contactPool.ForEach([&](std::uint32_t id, const Contact& c)
            {
                if (!c.solverRelevant)
                {
                    return; // event-only contact (sensor / kinematic): no constraint
                }
                if (emitContact(c, Vec2(Real(0), Real(0))))
                {
                    // Tag the just-emitted constraint with its persistent pool id so
                    // the post-Solve write-back lands the converged impulses on THIS
                    // Contact. Spans (below) leave the default kNoContact. The field
                    // travels with the constraint through the canonical sort.
                    out.back().sourceContactId = id;
                    // Phase C, Task 5: also carry the persistent contact COLOR so the
                    // solver buckets by it (deleting the per-step greedy recolor). An
                    // overflow contact carries kInvalidColor here -> the scalar tail;
                    // spans (below) keep the default kInvalidColor (no pool home).
                    out.back().color = c.color;
                }
            });

            // (b) tile spans: the transient scratch UpdateContacts filled this Step.
            for (std::size_t s = 0; s < m_spanContacts.size(); ++s)
            {
                emitContact(m_spanContacts[s], m_spanCenters[s]);
            }

            // ---- canonical sort (design Sec 7): (bodyA, bodyB, fixtureA, fixtureB).
            // Sort `out` and `keys` together via an index permutation so the live
            // solver feed is deterministic / run-twice-identical. The key is unique
            // per emitted constraint (a fixture-pair contributes exactly one
            // constraint per body-pair; two fixture-pairs differ in fixA/fixB).
            const std::size_t n = out.size();
            std::vector<std::size_t>& order = m_emitOrder; // persistent scratch
            order.clear();
            order.resize(n);
            for (std::size_t k = 0; k < n; ++k)
            {
                order[k] = k;
            }
            std::sort(order.begin(), order.end(),
                      [&](std::size_t lhs, std::size_t rhs)
            {
                const EmitSortKey& x = keys[lhs];
                const EmitSortKey& y = keys[rhs];
                if (x.bodyA != y.bodyA) { return x.bodyA < y.bodyA; }
                if (x.bodyB != y.bodyB) { return x.bodyB < y.bodyB; }
                if (x.fixA  != y.fixA)  { return x.fixA  < y.fixA;  }
                return x.fixB < y.fixB;
            });
            // Apply the permutation into the persistent staging vector (n is small
            // per Step). clear() keeps capacity; swap hands the sorted buffer to
            // `out` and parks `out`'s old buffer in m_emitSorted for next Step.
            std::vector<ContactConstraint>& sorted = m_emitSorted;
            sorted.clear();
            sorted.reserve(n);
            for (std::size_t k = 0; k < n; ++k)
            {
                sorted.push_back(out[order[k]]);
            }
            out.swap(sorted);
        }

        // ---- Step-stage lifts (decomp step 2 Task 3) --------------------------

        void ConstraintGraph::WritebackImpulses(const std::vector<ContactConstraint>& ccs)
        {
            // Warm-start write-back (Step stage 3b). The solver no longer owns a
            // warm-start cache: the converged per-point impulses live on the
            // persistent Contact. After Solve() (POST-restitution, so we capture
            // the FINAL accumulated impulses -- exactly what the retired SoftStep
            // Store loop stored), walk the emitted constraints and write each
            // point's (normal, tangent) impulse back onto its source Contact's
            // manifold point. Pool contacts carry their pool id in sourceContactId;
            // transient tile spans carry kNoContact and are skipped (they have no
            // persistent home -> cold-start next step). The Contact then carries
            // these into next step's emit (UpdateContacts copies them across the
            // manifold recompute by feature id). Decoupled: the solver never
            // touches the pool; only the graph does the round-trip.
            for (const ContactConstraint& cc : ccs)
            {
                if (cc.sourceContactId == ContactConstraint::kNoContact)
                {
                    continue; // transient span: no persistent Contact to update
                }
                Contact& src = m_contactPool.Get(cc.sourceContactId);
                // Defensive bound: within one Step the source Contact's manifold
                // cannot change after emit, so cc.pointCount == manifold.pointCount;
                // clamp anyway so a future reordering can never index out of range.
                const int n = std::min(cc.pointCount, src.manifold.pointCount);
                for (int p = 0; p < n; ++p)
                {
                    src.manifold.points[p].normalImpulse  = cc.points[p].normalImpulse;
                    src.manifold.points[p].tangentImpulse = cc.points[p].tangentImpulse;
                }
            }
        }

        void ConstraintGraph::CollectTouchedEventPairs(std::vector<BroadphasePair>& out) const
        {
            // Events-as-byproduct derivation (Step stage 6). Walk the pool
            // ascending-id (deterministic), collect {min,max} body-pairs for every
            // event-relevant EXACTLY-OVERLAPPING contact, then sort + unique so a
            // compound body's N^2 fixture-pairs collapse to ONE body-pair and the
            // Begin/Stay order matches the old sorted-body-pair emission order.
            // clear() keeps capacity.
            //
            // EXACT-OVERLAP, NOT speculative `touching`: the old ContactManager
            // tested overlap via SlotsOverlap with margin 0, which reports a contact
            // ONLY on STRICT penetration (depth > 0); a speculative gap (the manifold
            // point a velocity-scaled margin emits at NEGATIVE separation) is NOT an
            // event overlap. The pool's c.touching is pointCount>0 INCLUDING those
            // speculative gaps (correct for the SOLVER feed), so event derivation
            // must instead require a manifold point with separation > 0 -- byte-
            // identical to the old margin-0 SlotsOverlap (a genuinely penetrating
            // point reports the SAME positive separation regardless of the margin
            // used to compute the manifold, and an exact edge-touch at separation==0
            // is excluded by both, matching the old semantics).
            auto exactlyOverlapping = [](const Contact& c) noexcept -> bool
            {
                for (int p = 0; p < c.manifold.pointCount; ++p)
                {
                    if (c.manifold.points[p].separation > Real(0))
                    {
                        return true;
                    }
                }
                return false;
            };
            out.clear();
            m_contactPool.ForEach(
                [&](std::uint32_t /*id*/, const Contact& c)
                {
                    if (!c.eventRelevant || !exactlyOverlapping(c))
                    {
                        return;
                    }
                    const std::uint32_t a = c.bodyA < c.bodyB ? c.bodyA : c.bodyB;
                    const std::uint32_t b = c.bodyA < c.bodyB ? c.bodyB : c.bodyA;
                    out.push_back(BroadphasePair{ a, b });
                });
            std::sort(out.begin(), out.end());
            out.erase(
                std::unique(out.begin(), out.end()),
                out.end());
        }

        bool ConstraintGraph::DebugHasContact(const PhysicsWorld& w,
                                              BodyHandle a, BodyHandle b) const
        {
            if (!w.IsValid(a) || !w.IsValid(b))
            {
                return false;
            }
            const std::uint32_t sa = a.index;
            const std::uint32_t sb = b.index;
            bool found = false;
            m_contactPool.ForEach(
                [&](std::uint32_t /*id*/, const Contact& c)
            {
                if (!c.bIsBody)
                {
                    return; // tile span (never a body-pair)
                }
                if ((c.bodyA == sa && c.bodyB == sb) ||
                    (c.bodyA == sb && c.bodyB == sa))
                {
                    found = true;
                }
            });
            return found;
        }

        // ---- immediate lifecycle-seam contact destruction (Task 5) -------------
        //
        // Walk the persistent pool by STORED slot (ascending id, deterministic) and
        // destroy any contact that touches the removed fixture / body. The match is
        // by slot INDEX (not the live handle) because the caller runs these AFTER
        // the slot has already been recycled (generation bumped). Destroying the
        // CURRENT id mid-ForEach is safe: ForEach iterates ascending ids checking
        // w.m_alive[id], and Destroy only flips the alive flag + frees the id without
        // resizing the pool. Additive over the update-pass stale-handle guard.
        // NOTE: each helper is a full ContactPool::ForEach scan (O(contacts)), so
        // mass world teardown should prefer ContactPool::Clear() over a RemoveBody
        // loop (which would be O(bodies x contacts)).
        // CAVEAT (Phase C, Task 5): ContactPool::Clear() bypasses ReleaseContactColor,
        // so a future Clear()-based teardown path MUST also reset m_bodyColorMask (to
        // 0) and m_colorContacts (clear each list) -- otherwise the persistent coloring
        // bookkeeping leaks stale bits/ids against recycled slots.
        void ConstraintGraph::DestroyContactsForFixture(PhysicsWorld& w, std::uint32_t fixtureSlot)
        {
            m_contactPool.ForEach([&](std::uint32_t id, Contact& c)
            {
                if (c.a.index == fixtureSlot ||
                    (c.bIsBody && c.b.index == fixtureSlot))
                {
                    // A removed fixture's touching dynamic-dynamic contact may
                    // fracture its island. Mark both bodies' islands and wake them
                    // so the removed body's pile re-settles.
                    if (c.bIsBody && c.touching &&
                        c.bodyA != kInvalidSlot && c.bodyB != kInvalidSlot &&
                        c.bodyA < w.m_islandMgr.IslandIdCount() && c.bodyB < w.m_islandMgr.IslandIdCount() &&
                        w.TypeSlot(c.bodyA) == BodyType::Dynamic &&
                        w.TypeSlot(c.bodyB) == BodyType::Dynamic)
                    {
                        w.MarkSplitCandidate(w.IslandOf(c.bodyA));
                        w.MarkSplitCandidate(w.IslandOf(c.bodyB));
                        w.WakeIsland(c.bodyA);
                        w.WakeIsland(c.bodyB);
                    }
                    ReleaseAndDestroyContact(w, id, c);
                }
            });
        }

        void ConstraintGraph::DestroyContactsForBody(PhysicsWorld& w, std::uint32_t bodySlot)
        {
            m_contactPool.ForEach([&](std::uint32_t id, Contact& c)
            {
                if (c.bodyA == bodySlot ||
                    (c.bIsBody && c.bodyB == bodySlot))
                {
                    // A removed body's touching dynamic-dynamic contact may fracture
                    // its island. Mark both bodies' islands and wake them so the
                    // remaining pile re-settles. The removed body's slot still has a
                    // valid island id (IslandOf) here -- RemoveBody clears it AFTER this call.
                    if (c.bIsBody && c.touching &&
                        c.bodyA != kInvalidSlot && c.bodyB != kInvalidSlot &&
                        c.bodyA < w.m_islandMgr.IslandIdCount() && c.bodyB < w.m_islandMgr.IslandIdCount() &&
                        w.TypeSlot(c.bodyA) == BodyType::Dynamic &&
                        w.TypeSlot(c.bodyB) == BodyType::Dynamic)
                    {
                        w.MarkSplitCandidate(w.IslandOf(c.bodyA));
                        w.MarkSplitCandidate(w.IslandOf(c.bodyB));
                        w.WakeIsland(c.bodyA);
                        w.WakeIsland(c.bodyB);
                    }
                    ReleaseAndDestroyContact(w, id, c);
                }
            });
        }

        // ---- pooled-contact teardown (G1 island-split linkage) ------------------
        //
        // The per-body dyn-dyn contact adjacency (SwapRemoveId + DetachContactAdjacency
        // + DebugValidateBodyContacts) lives in IslandManager (decomp step 1 Task 3):
        // it is the split-linkage the island topology owns. ReleaseAndDestroyContact
        // is the graph-level teardown coordinator (decomp step 2 Task 3) -- it
        // detaches the island adjacency (via w.m_islandMgr), releases the persistent
        // color, and destroys the pool slot. Order is FROZEN (reads c before the
        // pool frees it; the RemoveBody color-leak assert gates the pairing).
        void ConstraintGraph::ReleaseAndDestroyContact(PhysicsWorld& w, std::uint32_t id, const Contact& c) noexcept
        {
            w.m_islandMgr.DetachContactAdjacency(w, id, c); // reads c before the pool frees the slot
            ReleaseContactColor(w, id); // free the color while c still holds it
            m_contactPool.Destroy(id);
        }
    } // namespace Physics
} // namespace Manifold2D
