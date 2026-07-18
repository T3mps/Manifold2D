// MOSAIC_LOG_CATEGORY must be defined before the FIRST include of Mosaic/Log.hpp
// in this TU, so it leads ahead of this TU's own header (defensive: no current
// transitive Mosaic pull from SpatialGrid.hpp, but this keeps the convention
// uniform with the TUs where one exists).
#define MOSAIC_LOG_CATEGORY "Manifold2D.Broadphase"
#include <Mosaic/Log.hpp>

#include <Manifold2D/Physics/Broadphase/SpatialGrid.hpp>
#include <algorithm>
#include <cmath>

namespace {
    // Max cells per axis a single box may span. The bound is a cell COUNT, not a
    // length, so it is scale-relative -- but read at the MKS 1 m residency tile it
    // is a ~65 km per-axis magnitude bound (65536 cells x 1 m). All legitimate
    // content is far under it; anything larger is garbage input (non-finite ->
    // huge, or escaped coords) -> treat as empty.
    constexpr int kMaxCellsPerAxis = 1 << 16;
    // Total-cell budget: kMaxCellsPerAxis alone is a PER-AXIS bound, so a box
    // spanning e.g. ~30000 cells on EACH axis (under the per-axis budget AND
    // under the raw-magnitude bound) still authorizes ~9e8 total cell visits --
    // and a box at the per-axis budget on both axes authorizes ~4.3e9. 1<<22
    // (~4.2M) cells is generous: at the MKS 1 m tile that is a ~2 km x 2 km
    // (sqrt(4.2M) ~= 2048 tiles/axis) solid square, far beyond any legitimate
    // content, while still catching the pathological cases above.
    constexpr long long kMaxCellsTotal = 1LL << 22;
}

namespace Manifold2D { namespace Physics {

SpatialGrid::SpatialGrid(Real tileSize, Vec2 origin)
    : m_tileSize(tileSize > Real(0) ? tileSize : Real(1)), m_origin(origin) {}

void SpatialGrid::CellCoord(Vec2 p, int& cx, int& cy) const
{
    cx = static_cast<int>(std::floor((p.x - m_origin.x) / m_tileSize));
    cy = static_cast<int>(std::floor((p.y - m_origin.y) / m_tileSize));
}

void SpatialGrid::CellRange(const Aabb2& box, int& x0, int& y0, int& x1, int& y1) const
{
    CellCoord(box.min, x0, y0);
    CellCoord(box.max, x1, y1);
}

bool SpatialGrid::SaneBox(const Aabb2& b) const
{
    if (!std::isfinite(b.min.x) || !std::isfinite(b.min.y) ||
        !std::isfinite(b.max.x) || !std::isfinite(b.max.y))
        return false;
    // Reject raw coordinates whose (coord - origin)/tileSize would overflow an
    // int BEFORE calling CellRange: casting an out-of-int-range float is UB,
    // and on this toolchain (MSVC / SSE2 cvttss2si) BOTH overflow directions
    // saturate to the SAME sentinel (INT_MIN), so a huge-but-finite box (e.g.
    // +-1e30) can make x0 == x1 and slip past a span check done after the
    // cast. Bounding the raw magnitude here keeps CellRange's cast in-range.
    const Real bound = static_cast<Real>(kMaxCellsPerAxis) * m_tileSize;
    if (std::abs(b.min.x - m_origin.x) > bound || std::abs(b.min.y - m_origin.y) > bound ||
        std::abs(b.max.x - m_origin.x) > bound || std::abs(b.max.y - m_origin.y) > bound)
        return false;
    int x0, y0, x1, y1; CellRange(b, x0, y0, x1, y1);
    // Guard against a span so large the cell loop would never finish. Compute in
    // 64-bit to avoid int overflow in the subtraction.
    const long long spanX = static_cast<long long>(x1) - static_cast<long long>(x0);
    const long long spanY = static_cast<long long>(y1) - static_cast<long long>(y0);
    if (spanX < 0 || spanY < 0) return false;                 // wrapped -> garbage
    if (spanX > kMaxCellsPerAxis || spanY > kMaxCellsPerAxis) return false;
    // Total-cell budget: catches a box under BOTH per-axis bounds individually
    // but whose product (the cell loop's actual iteration count) is still
    // pathological. Already 64-bit (spanX/spanY), so no overflow risk here.
    if ((spanX + 1) * (spanY + 1) > kMaxCellsTotal) return false;
    return true;
}

void SpatialGrid::Insert(std::uint32_t id, const Aabb2& box)
{
    if (!SaneBox(box))
    {
        MOSAIC_LOG_WARN("non-finite AABB dropped from broadphase");
        Remove(id); // drop garbage; don't register
        return;
    }
    Remove(id); // self-heal: drop any prior registration so re-Insert is safe
                // (idempotent; Remove is a no-op for a fresh id).
    int x0, y0, x1, y1; CellRange(box, x0, y0, x1, y1);
    std::vector<std::uint64_t>& owned = m_idCells[id];
    owned.clear();
    for (int cy = y0; cy <= y1; ++cy)
        for (int cx = x0; cx <= x1; ++cx)
        {
            const std::uint64_t k = Key(cx, cy);
            m_cells[k].push_back(id);
            owned.push_back(k);
        }
}

void SpatialGrid::Remove(std::uint32_t id)
{
    auto it = m_idCells.find(id);
    if (it == m_idCells.end()) return;
    for (std::uint64_t k : it->second)
    {
        auto cit = m_cells.find(k);
        if (cit == m_cells.end()) continue;
        std::vector<std::uint32_t>& ids = cit->second;
        ids.erase(std::remove(ids.begin(), ids.end(), id), ids.end());
        if (ids.empty()) m_cells.erase(cit);
    }
    m_idCells.erase(it);
}

void SpatialGrid::Move(std::uint32_t id, const Aabb2& box)
{
    // Insert is self-healing (re-registers), so Move == Insert: drop the prior
    // cells + register the new ones in one pass.
    Insert(id, box);
}

void SpatialGrid::ForEachCell(
    Mosaic::FunctionRef<void(int, int, const std::vector<std::uint32_t>&)> fn) const
{
    // Unpack each occupied cell key with the EXACT inverse of Key():
    //   Key(cx,cy) = (uint32_t(cx) << 32) | uint32_t(cy)
    // so cx = int(uint32_t(k >> 32)), cy = int(uint32_t(k & 0xFFFFFFFF)).
    // Casting the uint32_t bit pattern back to int recovers the original signed
    // coord (two's-complement round-trip). Iteration order is unordered_map
    // order -- display only; no sim path depends on it.
    for (const auto& [k, ids] : m_cells)
    {
        const int cx = static_cast<int>(static_cast<std::uint32_t>(k >> 32));
        const int cy = static_cast<int>(static_cast<std::uint32_t>(k & 0xFFFFFFFFu));
        fn(cx, cy, ids);
    }
}

int SpatialGrid::QueryAABB(const Aabb2& box, std::vector<std::uint32_t>& out) const
{
    out.clear();
    if (!SaneBox(box)) return 0;
    int x0, y0, x1, y1; CellRange(box, x0, y0, x1, y1);
    for (int cy = y0; cy <= y1; ++cy)
        for (int cx = x0; cx <= x1; ++cx)
        {
            auto cit = m_cells.find(Key(cx, cy));
            if (cit == m_cells.end()) continue;
            out.insert(out.end(), cit->second.begin(), cit->second.end());
        }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return static_cast<int>(out.size());
}

}} // namespace Manifold2D::Physics
