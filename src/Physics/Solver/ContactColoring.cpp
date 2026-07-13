// ContactColoring.cpp -- per-step greedy graph coloring (see header for why).

#include <Manifold2D/Physics/Solver/ContactColoring.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        Coloring ColorConstraints(const std::vector<ColorEdge>& edges,
                                  std::uint32_t bodyCount)
        {
            // perBody[i] is a color bitmask: bit k set => body i is already
            // used in color k. Only dynamic endpoints set/test bits, so a
            // static body's mask stays 0 and never blocks a color.
            std::vector<std::uint32_t> perBody(bodyCount, 0u);

            Coloring result;
            result.colors.resize(kColorCount);

            for (const ColorEdge& e : edges)
            {
                // A dynamic endpoint is in range when its slot is valid; a
                // static endpoint never indexes perBody (it cannot conflict).
                const bool aActive = e.aDyn && e.a < bodyCount;
                const bool bActive = e.bDyn && e.b < bodyCount;

                // Find the lowest color free for BOTH dynamic endpoints. A
                // static endpoint is free in every color (mask treated as 0).
                int chosen = -1;
                for (int k = 0; k < kColorCount; ++k)
                {
                    const std::uint32_t bit = 1u << k;
                    const bool aFree = !aActive || !(perBody[e.a] & bit);
                    const bool bFree = !bActive || !(perBody[e.b] & bit);
                    if (aFree && bFree)
                    {
                        chosen = k;
                        break;
                    }
                }

                if (chosen >= 0)
                {
                    result.colors[chosen].push_back(e.ref);
                    // Mark the color used for each DYNAMIC endpoint only.
                    const std::uint32_t bit = 1u << chosen;
                    if (aActive) perBody[e.a] |= bit;
                    if (bActive) perBody[e.b] |= bit;
                }
                else
                {
                    // No free color for some endpoint -> spill to the tail.
                    result.overflow.push_back(e.ref);
                }
            }

            return result;
        }

    } // namespace Physics
} // namespace Manifold2D
