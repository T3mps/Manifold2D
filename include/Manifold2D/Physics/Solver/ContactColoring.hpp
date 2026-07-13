#pragma once

// ContactColoring.hpp -- greedy graph coloring for the SIMD constraint solve.
//
// The lane-wide (SoA) solve processes constraints 8-wide. Two constraints in
// the same 8-lane batch that share a DYNAMIC body would race on the read-
// modify-write of that body's velocity (both lanes load the old velocity, apply
// their impulse, and store -- one update is lost). Graph coloring removes that
// hazard: we partition the constraints into "colors" such that no two
// constraints in a color touch the same dynamic body, so every constraint in a
// color is independent and the whole color is safe to solve lane-wide.
//
// Static/kinematic bodies (invMass == 0) are READ-ONLY in the solve, so sharing
// one across many constraints in a color is harmless -- they do NOT constrain
// coloring. Each ColorEdge carries an aDyn/bDyn flag so coloring only treats
// dynamic endpoints as conflicting.
//
// This is Box2D-v3's per-step greedy coloring: walk the edges in input order,
// give each the lowest free color (one whose dynamic endpoints are unused so
// far), and if all kColorCount colors are taken for an endpoint, spill the edge
// to `overflow` (solved separately, e.g. scalar tail). Fixed walk order + fixed
// lowest-free rule => the coloring is a deterministic function of its input.
//
// SCOPE (Task 2): the pure algorithm + its property tests. NOTHING consumes the
// coloring yet (the solver wiring is a later task). Backend-agnostic: no SIMD,
// no PhysicsWorld -- just std::vector + a bitmask.
//
// PRESENTATION-FREE + C++20-clean.

#include <cstdint>
#include <vector>

namespace Manifold2D
{
    namespace Physics
    {
        // Box2D v3 color count. The first kColorCount colors are solved
        // lane-wide; anything that finds no free color spills to overflow.
        inline constexpr int kColorCount = 12;

        // One constraint expressed as an edge between two body slots. `aDyn` /
        // `bDyn` mark whether each endpoint is a dynamic (read-modify-write)
        // body -- only dynamic endpoints constrain coloring. `ref` is the
        // caller's own constraint index, echoed back in the Coloring output.
        struct ColorEdge
        {
            std::uint32_t a, b;     // body slots
            bool aDyn, bDyn;        // is each endpoint dynamic? (static -> shareable)
            std::uint32_t ref;      // caller's constraint index
        };

        // Result of coloring: colors[k] holds the refs assigned to color k (k in
        // [0, kColorCount)); overflow holds the refs that found no free color.
        struct Coloring
        {
            std::vector<std::vector<std::uint32_t>> colors;   // colors[k] = refs in color k
            std::vector<std::uint32_t> overflow;              // refs with no free color
        };

        // Greedily color `edges` (walked in input order). `bodyCount` sizes the
        // scratch per-body color-bitmask map; the caller passes valid in-range
        // body slots for every dynamic endpoint. Deterministic: same input
        // vector -> identical Coloring.
        Coloring ColorConstraints(const std::vector<ColorEdge>& edges,
                                  std::uint32_t bodyCount);

    } // namespace Physics
} // namespace Manifold2D
