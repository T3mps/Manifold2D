// One translation unit so the Manifold2D static library is never empty (avoids
// MSVC LNK4221) in Task 1, before the Physics/Geometry .cpp populate src/ in Task 2.
#include <cstdint>

namespace Manifold2D
{
    // Bumped when the vendored snapshot changes; also proves the lib links.
    std::uint32_t LibraryVersion() noexcept { return 2000; } // Phase 2 == 2.0.0
}
