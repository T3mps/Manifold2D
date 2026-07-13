#pragma once

// Island: the persistent island registry RECORD + shared constants (Phase A).
//
// The island TOPOLOGY (the id-indexed record pool + per-body island id + merge/
// split) AND the per-Step sleep pass moved into IslandManager (PhysicsWorld
// decomposition step 1, 2026-07-09). This header now provides only the pieces
// SHARED across that boundary -- the Island record struct + the registry/sleep
// constants (kInvalidIsland, kMaxSplitsPerStep, kSleepTime) -- which IslandManager
// #includes. It is a data header now, not a module: it declares no functions.
//
// An Island is a connected component of DYNAMIC bodies joined by touching
// dynamic-dynamic contacts (a joint is an island edge too). It SURVIVES across
// steps and is maintained INCREMENTALLY by IslandManager at the body/contact
// lifecycle seams:
//   * AddBody(Dynamic)  -> a new dynamic body is its own 1-body island.
//   * touch-BEGIN       -> the two islands merge (weighted union, O(smaller)).
//   * touch-END/destroy -> the island is flagged a split candidate; a quota-
//     limited deferred pass (kMaxSplitsPerStep per Step) re-derives its connected
//     components. The split is O(island) and amortized over a few Steps.
//   * Static/Kinematic bodies are NOT island members (they anchor; they are not
//     island nodes -- the dynamic-dynamic-only union rule).
//
// The per-Step sleep pass (IslandManager::UpdateSleep, Step stage 5) advances each
// awake dynamic body's idle timer, then sleeps an island AS A UNIT when EVERY
// awake-dynamic member has accumulated kSleepTime seconds of idle. WAKE paths live
// in PhysicsWorld (ApplyImpulse / SetVelocity / Wake), ConstraintGraph (WakeMoverPair) and
// IslandManager::WakeIsland; the sleep pass only puts islands to SLEEP. The
// thresholds + whole-island-unit-sleep behavior are UNCHANGED from the M6
// global-UF version this replaced.
//
// DETERMINISM: merges are applied in canonical (min,max)-slot order; splits
// iterate members by ascending slot index; ForEachIsland iterates ascending
// island id. No wall clock, no fast math.
//
// PRESENTATION-FREE + C++23-clean: Geometry::Vec2 + std + sibling Physics headers only.
// No SDL3/NVRHI/Batcher2D/ImGui. namespace Manifold2D::Physics, Core style.

#include <cstdint>
#include <vector>

#include <Manifold2D/Physics/PhysicsTypes.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        namespace Island
        {
            // ----------------------------------------------------------------
            // Persistent island registry (Phase A -- replaces the per-step UF).
            // ----------------------------------------------------------------
            //
            // An Island is a connected component of DYNAMIC bodies joined by
            // touching dynamic-dynamic contacts. It SURVIVES across steps and is
            // maintained INCREMENTALLY at the body/contact lifecycle seams (create
            // = a 1-body island; touch-begin = merge; touch-end/destroy = mark a
            // split candidate; a deferred quota-limited pass rebuilds candidates).
            // Static/Kinematic bodies are NOT members (they anchor, they are not
            // island nodes -- the same dynamic-dynamic-only rule the old UF used).
            //
            // The registry lives on IslandManager (m_islands, id-indexed, free-list
            // recycled). Sleep bookkeeping is DERIVED from members (per-body idle
            // timers stay per-body on PhysicsWorld in m_sleepTimer); an island holds
            // no timer.
            struct Island
            {
                // Member body SLOTS (dynamic only). Order is append-order; the
                // sleep + wake passes iterate it directly (no global scan).
                std::vector<std::uint32_t> bodies;

                // Set when a member's contact separated or a member was destroyed:
                // the deferred split pass (quota-limited, once per Step) re-derives
                // this island's connected components and clears the flag.
                bool splitCandidate = false;
            };

            // A body with no island (Static/Kinematic, or a transiently un-assigned
            // dynamic slot) has island id == kInvalidIsland (IslandManager::IslandOf).
            inline constexpr std::uint32_t kInvalidIsland = 0xFFFFFFFFu;

            // At most this many split-candidate islands are rebuilt per Step (Box2D
            // v3's value). Splits are deferred + amortized; the registry converges
            // over a few steps. Processed in ascending island-id order (determinism).
            inline constexpr std::uint32_t kMaxSplitsPerStep = 1u;

            // ---- sleep threshold --------------------------------------------
            //
            // The per-body VELOCITY gate is no longer a pair of separate Lua-ported
            // constants. UpdateSleep now uses Box2D v3's combined, extent-weighted
            // test: a body is idle when |v| + |w|*maxExtent < its per-body
            // sleepThreshold (PhysicsWorld::SleepThresholdSlot, defaulted from
            // WorldDef::sleepThreshold). The old kSleepLinVel2 (=4) / kSleepAngVel
            // (=0.05) gates were too strict + body-size-unaware for the gravity-900
            // world and never let a slowly-rolling pile sleep (see the never-settle
            // findings doc). Only the TIME gate remains a constant here.

            // Idle time a body (and every island member) must accumulate before
            // the island sleeps (Lua lines 437/442: sleepT > 0.5 seconds).
            inline constexpr Real kSleepTime = Real(0.5);

        } // namespace Island
    } // namespace Physics
} // namespace Manifold2D
