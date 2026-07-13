// TileGrid: static-geometry broadphase (M6, Task P1.7). See TileGrid.hpp for
// the full PORT + MODERNIZE note (faithful per-row run-merge from
// Client/src/physics/TileGrid.lua, reformulated cell->world to plain Cartesian).

#include <Manifold2D/Physics/Broadphase/TileGrid.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>

namespace Manifold2D
{
    namespace Physics
    {
        namespace
        {
            // floor(x) as an int. Carried in f64 to track the Lua's f64 cell
            // math; physics state is f32 but the cell index must be exact for
            // boxes that land on a cell boundary.
            inline int FloorToInt(double x)
            {
                return static_cast<int>(std::floor(x));
            }

            // PAD: cells of slack added around the query's cell range, so a
            // shape near a cell boundary still sees the walls just past it.
            // Ported verbatim from TileGrid.lua (PAD = 1). Implementation
            // constant only -- not part of the public interface.
            constexpr int kPad = 1;
        } // namespace

        TileGrid::TileGrid(const IPassabilitySource& src, Real cellSize,
                           Vec2 origin)
            : m_src(&src),
              m_cellSize(cellSize > Real(0) ? cellSize : Real(1)),
              m_origin(origin)
        {
            assert(cellSize > Real(0) && "TileGrid: cellSize must be positive");
        }

        Aabb2 TileGrid::CellBox(int cx, int cy) const
        {
            // Cartesian cell -> AABB (see TileGrid.hpp CELL -> WORLD):
            //   min = origin + (cx*cellSize,     cy*cellSize)
            //   max = origin + ((cx+1)*cellSize, (cy+1)*cellSize)
            Aabb2 box;
            box.min = Vec2(m_origin.x + static_cast<Real>(cx) * m_cellSize,
                           m_origin.y + static_cast<Real>(cy) * m_cellSize);
            box.max = Vec2(m_origin.x + static_cast<Real>(cx + 1) * m_cellSize,
                           m_origin.y + static_cast<Real>(cy + 1) * m_cellSize);
            return box;
        }

        int TileGrid::Query(const Aabb2& worldBox, std::vector<Aabb2>& out) const
        {
            // Zero steady-state allocation: clear() keeps the backing capacity,
            // push_back reuses it. (The Lua reused pooled `out` rows; an Aabb2
            // is a trivially-copyable value, so a flat vector is the C++ analog.)
            out.clear();

            const int width  = m_src->Width();
            const int height = m_src->Height();
            if (width <= 0 || height <= 0)
            {
                return 0;
            }

            // -------- cell range covering the query box (Cartesian) --------
            // The Lua took the min/max fractional cell coords over the 4 corners
            // and rounded (floor(f + 0.5)) with PAD slack, clamped to the map.
            // In Cartesian the cell containing world coord w is
            //   floor((w - origin) / cellSize)
            // The min corner gives the low cell, the max corner the high cell.
            // We add PAD = 1 cell of slack on every side and clamp the SCAN
            // range to [0,width) x [0,height) -- exactly the Lua's clamp to the
            // map, so out-of-bounds cells are simply never visited.
            const double invCell = 1.0 / static_cast<double>(m_cellSize);
            const double ox      = static_cast<double>(m_origin.x);
            const double oy      = static_cast<double>(m_origin.y);

            const int cxLo = FloorToInt(
                (static_cast<double>(worldBox.min.x) - ox) * invCell);
            const int cxHi = FloorToInt(
                (static_cast<double>(worldBox.max.x) - ox) * invCell);
            const int cyLo = FloorToInt(
                (static_cast<double>(worldBox.min.y) - oy) * invCell);
            const int cyHi = FloorToInt(
                (static_cast<double>(worldBox.max.y) - oy) * invCell);

            const int a = std::max(0, cxLo - kPad);          // first column
            const int c = std::min(width - 1, cxHi + kPad);  // last column
            const int b = std::max(0, cyLo - kPad);          // first row
            const int d = std::min(height - 1, cyHi + kPad); // last row

            if (a > c || b > d)
            {
                return 0; // query does not overlap any in-bounds cell
            }

            // -------- greedy per-row horizontal run-merge --------
            // Row-major scan (deterministic). Within each row, a maximal run of
            // consecutive solid cells [runStart..cx] merges into ONE AABB. Rows
            // are scanned independently -- NO cross-row merge (the Lua does not;
            // a row with a gap yields two rects). This eliminates the internal
            // vertical edges between adjacent cells in a run, which is the
            // tile-seam catch the merge exists to kill.
            for (int cy = b; cy <= d; ++cy)
            {
                int cx = a;
                while (cx <= c)
                {
                    if (m_src->IsSolid(cx, cy))
                    {
                        const int runStart = cx;
                        // Extend the run over consecutive solid cells.
                        while (cx + 1 <= c && m_src->IsSolid(cx + 1, cy))
                        {
                            ++cx;
                        }
                        const int runEnd = cx;

                        // Merged run -> one Cartesian AABB rect (see header):
                        //   min = origin + (runStart*cellSize,    cy*cellSize)
                        //   max = origin + ((runEnd+1)*cellSize,  (cy+1)*cellSize)
                        Aabb2 rect;
                        rect.min = Vec2(
                            m_origin.x + static_cast<Real>(runStart) * m_cellSize,
                            m_origin.y + static_cast<Real>(cy) * m_cellSize);
                        rect.max = Vec2(
                            m_origin.x +
                                static_cast<Real>(runEnd + 1) * m_cellSize,
                            m_origin.y +
                                static_cast<Real>(cy + 1) * m_cellSize);
                        out.push_back(rect);
                    }
                    ++cx;
                }
            }

            return static_cast<int>(out.size());
        }

    } // namespace Physics
} // namespace Manifold2D
