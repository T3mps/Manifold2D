#pragma once

// Static-geometry broadphase for the Manifold2D 2D physics engine (M6, Task P1.7).
//
// PORT + MODERNIZE of Client/src/physics/TileGrid.lua. The Lua TileGrid stores
// NOTHING for walls: it answers queries straight from Map cell flags and
// MATERIALIZES transient "virtual fixtures" for the solid cells near a query,
// merging a per-row run of consecutive solid cells into ONE shape (the spec's
// greedy rectangle merge -- it kills the tile-seam internal-edge bug, where a
// shape sliding along a wall catches on the vertical edge between two adjacent
// solid cells). That algorithm is what we port faithfully:
//
//   * store nothing for walls -- the IPassabilitySource (Passability.hpp) is
//     the ONLY input;
//   * compute the cell range covering the query's world-space AABB, with
//     PAD = 1 cell of slack (so a shape near a boundary sees its walls);
//   * GREEDY PER-ROW HORIZONTAL RUN-MERGE: within each row, merge consecutive
//     solid cells into one virtual fixture. This is the internal-edge killer.
//   * ONLY horizontal (per-row) merging -- the Lua does NOT merge across rows
//     (the harness confirms a row with a gap yields TWO spans). Each row is
//     scanned independently, left to right.
//   * zero steady-state allocation -- the result fills the caller's reused
//     vector (clear() + push_back, capacity preserved), exactly like the Lua's
//     pooled `out` rows. No closures on the hot path.
//
// CARTESIAN REFORMULATION (MODERNIZE -- the part that DIFFERS from the Lua):
// the Lua is ISO-coupled (require "world.IsoProjection"): a solid cell is a
// DIAMOND in screen space and a merged row-run is a PARALLELOGRAM (4 verts:
// top/right/bottom/left). The C++ ENGINE DROPS ISO PROJECTION and uses a plain
// CARTESIAN grid: each solid cell is an axis-aligned square, and a per-row run
// of consecutive solid cells merges into ONE axis-aligned AABB RECTANGLE. The
// AABB rect is the faithful Cartesian analog of the Lua parallelogram, and it
// is directly collidable by the narrowphase (the unified Collide treats an Aabb
// as a 4-vert box) and the GJK shape-cast (ShapeCastPoly / ShapePolyDistance over
// the 4 corners). So the OUTPUT type is Aabb2 (the broadphase box, same min/max
// type the narrowphase + world consume), NOT a parallelogram vertex tuple.
//
// Because iso is dropped there is NO Lua bit-match for the rect COORDINATES.
// The merge ALGORITHM is identical (per-row runs, no cross-row merge); only the
// cell->world mapping is reformulated to Cartesian. The test oracle is a
// SELF-DEFINED Cartesian grid (a known cell-flag layout + analytically-expected
// merged rects) -- correct + expected for this task (the plan acknowledges the
// Cartesian reformulation).
//
// CELL -> WORLD (Cartesian, 0-based cells):
//   cell (cx,cy) -> AABB
//     min = origin + (cx*cellSize,     cy*cellSize)
//     max = origin + ((cx+1)*cellSize, (cy+1)*cellSize)
//   a merged run cx = runStart..runEnd in row cy ->
//     min = origin + (runStart*cellSize,    cy*cellSize)
//     max = origin + ((runEnd+1)*cellSize, (cy+1)*cellSize)
//
// NOT an IBroadphase: TileGrid has no body ids / Update / Pairs -- it is a
// STANDALONE static broadphase. The world composes an IBroadphase (movers) +
// a TileGrid (statics), so TileGrid deliberately does not implement IBroadphase.
//
// DETERMINISM: row-major scan, fixed PAD, integer cell math, no wall-clock, no
// fast-math.
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.
// No SDL3/NVRHI/Batcher2D/ImGui, no iso/Map/world coupling. Also compiled
// Arcane-only, /MD. namespace Manifold2D::Physics, Core style.

#include <vector>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Broadphase/Broadphase.hpp>
#include <Manifold2D/Physics/Broadphase/Passability.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        // ----------------------------------------------------------------
        // TileGrid: static broadphase over an IPassabilitySource.
        // ----------------------------------------------------------------
        class TileGrid
        {
        public:
            // `src` must outlive this TileGrid (the seam is the single source
            // of truth; TileGrid holds a reference and stores nothing for
            // walls). cellSize > 0 is the side length of one square cell;
            // origin is the world position of cell (0,0)'s min corner.
            TileGrid(const IPassabilitySource& src, Real cellSize,
                     Vec2 origin = Vec2(Real(0), Real(0)));

            // Fill `out` with the merged solid AABB rects (virtual fixtures)
            // intersecting the world-space query box. Per-row horizontal
            // run-merge: a row-run of consecutive solid cells becomes ONE rect;
            // a gap in a row yields a separate rect per run; rows are never
            // merged together. Returns the rect count.
            //
            // `out` is CLEARED then filled (clear() + push_back) so its capacity
            // is preserved across calls -- zero steady-state allocation, the
            // Lua's pooled-row discipline.
            int Query(const Aabb2& worldBox, std::vector<Aabb2>& out) const;

            // Map cell (cx,cy) to its world-space AABB (Cartesian). Public for
            // tests / callers that want a single cell's box.
            [[nodiscard]] Aabb2 CellBox(int cx, int cy) const;

            // Returns the passability source (non-null; src must outlive this TileGrid).
            [[nodiscard]] const IPassabilitySource& Source() const { return *m_src; }

            [[nodiscard]] Real CellSize() const { return m_cellSize; }
            [[nodiscard]] Vec2 Origin() const { return m_origin; }

        private:
            // Must outlive this TileGrid.
            const IPassabilitySource* m_src = nullptr;
            Real                      m_cellSize = Real(1);
            Vec2                      m_origin{ Real(0), Real(0) };
        };

    } // namespace Physics
} // namespace Manifold2D
