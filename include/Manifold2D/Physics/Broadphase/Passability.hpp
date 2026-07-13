#pragma once

// Passability seam for the Manifold2D 2D physics static broadphase (M6, Task P1.7).
//
// PORT NOTE: Client/src/physics/TileGrid.lua answers static-geometry queries
// straight from Map cell flags (map:isWalkable(cx,cy)) -- it stores NOTHING for
// walls. To keep the C++ TileGrid free of any Map / world / iso coupling (and so
// the SERVER can feed the same broadphase from ITS own world representation),
// the cell-flag lookup is abstracted behind IPassabilitySource: the engine's
// TileGrid asks only "is cell (cx,cy) solid, and what are the grid bounds?".
// The Map (client) and the server's authoritative grid each implement this seam,
// so client + server share one static broadphase from a single source of truth:
// flip a cell flag and physics follows instantly (spec: Broadphase
// single-source-of-truth).
//
// COORDINATE CONVENTION (MODERNIZE -- the C++ engine drops iso projection): the
// Lua's grid is 1-based and iso-projected. Here cells are 0-based plain
// CARTESIAN indices: cx in [0, Width), cy in [0, Height). The mapping from a
// cell to its world AABB is the TileGrid's job (Cartesian, see TileGrid.hpp);
// this seam only reports solidity + bounds.
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.
// No SDL3/NVRHI/Batcher2D/ImGui, no iso/Map/world coupling. Also compiled
// Arcane-only, /MD. namespace Manifold2D::Physics, Core style.

#include <cstdint>
#include <vector>

namespace Manifold2D
{
    namespace Physics
    {
        // ----------------------------------------------------------------
        // IPassabilitySource: the single-source-of-truth cell-flag seam.
        // ----------------------------------------------------------------
        //
        // PORT of map:isWalkable: IsSolid(cx,cy) == (not isWalkable) -- true
        // means the cell is a wall / blocked. Width()/Height() are the cell
        // grid bounds (the Lua's map.w / map.h). 0-based indices.
        class IPassabilitySource
        {
        public:
            virtual ~IPassabilitySource() = default;

            // True iff cell (cx,cy) is a wall / blocked. The Lua queried only
            // in-bounds cells (it clamped the cell range to the map); TileGrid
            // clamps its scan to [0,Width)x[0,Height) the same way, so an
            // implementation is never asked about an out-of-bounds cell on the
            // hot path. For belt-and-braces, OOB SHOULD report NOT solid (see
            // GridPassability) -- out-of-bounds is simply "not a wall", matching
            // the Lua's behavior of never visiting cells outside the map.
            [[nodiscard]] virtual bool IsSolid(int cx, int cy) const = 0;

            // OBSTACLE-CLASS TIERING SEAM (M6, Task P1.9 carry-forward).
            //
            // Does cell (cx,cy) block line-of-sight? Default: opaque iff solid
            // (TALL == solid). LOW-obstacle tiers override to return false for
            // movement-blocking-but-see-through cells.
            //
            // PORT NOTE: the binary IsSolid the static broadphase / movement uses
            // (map:isWalkable) cannot express the LOS rule. PhysicsWorld.lua's
            // raycast distinguished movement (`not map:isWalkable`) from sight
            // (`map:obstacleClass(cx,cy) == Map.OBSTACLE_TALL`): a LOW obstacle
            // BLOCKS MOVEMENT but NOT sight; a TALL obstacle blocks BOTH. The
            // harness encodes this as cell flags 1=walkable, 2=LOW (blocks
            // movement, not sight), 4=TALL (blocks both). This non-pure,
            // defaulted virtual is the minimal faithful seam: Raycast(tallOnly)
            // / LineOfSight call BlocksSight; everything else (movement,
            // TileGrid spans) keeps using IsSolid. NON-BREAKING: the default
            // delegates to IsSolid, so existing impls are unchanged and TileGrid
            // (movement only) never touches it.
            [[nodiscard]] virtual bool BlocksSight(int cx, int cy) const
            {
                return IsSolid(cx, cy);
            }

            // Cell-grid bounds (the Lua map.w / map.h). Cells are 0-based, so
            // valid indices are cx in [0, Width()) and cy in [0, Height()).
            [[nodiscard]] virtual int Width() const = 0;
            [[nodiscard]] virtual int Height() const = 0;
        };

        // ----------------------------------------------------------------
        // GridPassability: a simple in-memory IPassabilitySource (tests +
        // small static grids). Holds a width x height flat solid-flag buffer.
        // ----------------------------------------------------------------
        //
        // OOB CHOICE (documented per task): IsSolid returns FALSE for any cell
        // outside [0,Width)x[0,Height). Out-of-bounds is "not a wall", matching
        // the Lua, which clamped the scan range to the map and never queried
        // outside it. TileGrid clamps its scan identically, so the OOB branch is
        // belt-and-braces; making it return false avoids spurious walls at the
        // world edge if a caller ever over-scans.
        //
        // SIGHT LAYER (M6, Task P1.9): a SEPARATE per-cell sight-blocking flag
        // carries the obstacle-class tiering (the harness 1/2/4 flags). A cell
        // can be solid-for-movement yet see-through (a LOW obstacle): SetSolid
        // controls movement (IsSolid + TileGrid spans), SetBlocksSight controls
        // LOS (BlocksSight). The two layers are independent so a test can build:
        //   TALL = SetSolid(true)  + SetBlocksSight(true)  (blocks both)
        //   LOW  = SetSolid(true)  + SetBlocksSight(false) (blocks movement only)
        // SetSolid does NOT touch the sight layer (kept orthogonal); construct a
        // TALL cell by setting both. (A convenience would be easy to add, but
        // keeping the layers fully independent is clearer for the tiering test.)
        class GridPassability final : public IPassabilitySource
        {
        public:
            GridPassability(int width, int height)
                : m_width(width < 0 ? 0 : width),
                  m_height(height < 0 ? 0 : height),
                  m_solid(static_cast<std::size_t>(m_width) *
                              static_cast<std::size_t>(m_height),
                          std::uint8_t(0)),
                  m_sight(static_cast<std::size_t>(m_width) *
                              static_cast<std::size_t>(m_height),
                          std::uint8_t(0))
            {
            }

            // Mark / clear a cell's MOVEMENT-solid flag (no-op if OOB, like the
            // Lua bounds clamp). Does not affect the sight layer.
            void SetSolid(int cx, int cy, bool solid)
            {
                if (cx < 0 || cy < 0 || cx >= m_width || cy >= m_height)
                {
                    return;
                }
                m_solid[Index(cx, cy)] = solid ? std::uint8_t(1) : std::uint8_t(0);
            }

            // Mark / clear a cell's SIGHT-blocking flag (no-op if OOB). A TALL
            // obstacle sets both solid + sight; a LOW obstacle sets solid only.
            void SetBlocksSight(int cx, int cy, bool blocks)
            {
                if (cx < 0 || cy < 0 || cx >= m_width || cy >= m_height)
                {
                    return;
                }
                m_sight[Index(cx, cy)] = blocks ? std::uint8_t(1) : std::uint8_t(0);
            }

            [[nodiscard]] bool IsSolid(int cx, int cy) const override
            {
                if (cx < 0 || cy < 0 || cx >= m_width || cy >= m_height)
                {
                    return false; // OOB == not a wall (see OOB CHOICE above)
                }
                return m_solid[Index(cx, cy)] != std::uint8_t(0);
            }

            [[nodiscard]] bool BlocksSight(int cx, int cy) const override
            {
                if (cx < 0 || cy < 0 || cx >= m_width || cy >= m_height)
                {
                    return false; // OOB == not an occluder (matches IsSolid OOB)
                }
                return m_sight[Index(cx, cy)] != std::uint8_t(0);
            }

            [[nodiscard]] int Width() const override { return m_width; }
            [[nodiscard]] int Height() const override { return m_height; }

        private:
            [[nodiscard]] std::size_t Index(int cx, int cy) const
            {
                return static_cast<std::size_t>(cy) *
                           static_cast<std::size_t>(m_width) +
                       static_cast<std::size_t>(cx);
            }

            int                       m_width = 0;
            int                       m_height = 0;
            std::vector<std::uint8_t> m_solid;
            std::vector<std::uint8_t> m_sight;
        };

    } // namespace Physics
} // namespace Manifold2D
