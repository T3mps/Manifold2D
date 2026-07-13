#pragma once
// SpatialGrid: a per-shape fixed-tile spatial hash (Physics v2 collision rebuild).
// A FIRST-CLASS grid index: ids (fixtures or bodies) register into every cell
// their tight AABB overlaps; QueryAABB returns the union of those cells' ids,
// SORTED + de-duped (the caller applies the tight AABB test). Bounded memory
// (hash, not a dense array); cell coord = floor((p - origin)/tileSize).
// DETERMINISM: cells visited row-major; output sorted+unique; no fp in the cell
// index math beyond the floor.
// PRESENTATION-FREE + C++20 (Geometry::Vec2 + std + Physics only).
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <Manifold2D/Core/FunctionRef.hpp>
#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Broadphase/Broadphase.hpp> // Aabb2

namespace Manifold2D { namespace Physics {

// NOTE: deliberately NOT an IBroadphase -- this is the first-class statics +
// residency index (per-shape registration + region queries), not a swappable
// mover broadphase, so it exposes Insert/Remove/Move/QueryAABB, not Update/Pairs.
class SpatialGrid
{
public:
    explicit SpatialGrid(Real tileSize, Vec2 origin = Vec2(Real(0), Real(0)));

    void Insert(std::uint32_t id, const Aabb2& box); // register into overlapping cells
    void Remove(std::uint32_t id);                   // drop from all its cells
    void Move(std::uint32_t id, const Aabb2& box);   // remove + insert (re-register)

    // Ids whose cells overlap `box`, SORTED ascending + de-duped. out cleared
    // then filled (clear()+push_back -> capacity preserved). Returns out.size().
    int QueryAABB(const Aabb2& box, std::vector<std::uint32_t>& out) const;

    [[nodiscard]] Real TileSize() const { return m_tileSize; }
    [[nodiscard]] Vec2 Origin()   const { return m_origin; }

    // Integer cell coord of a world point (floor). Public for tests / residency.
    void CellCoord(Vec2 p, int& cx, int& cy) const;

    // ---- read-only debug-visualization accessor (Slice A) ----------------
    //
    // Visit each OCCUPIED cell, passing its integer (cx, cy) coord and the ids
    // resident in it. The key is unpacked with the EXACT inverse of the private
    // Key() pack (uint32 round-trip preserves the signed coord). Read-only;
    // unordered_map traversal -> iteration order is for DISPLAY only (no sim
    // path may depend on it). The ids vector is never empty for a live cell
    // (Remove drops a cell once it empties).
    void ForEachCell(
        FunctionRef<void(int cx, int cy,
                         const std::vector<std::uint32_t>& ids)> fn) const;

private:
    static std::uint64_t Key(int cx, int cy)
    { return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(cx)) << 32)
           |  static_cast<std::uint64_t>(static_cast<std::uint32_t>(cy)); }
    void CellRange(const Aabb2& box, int& x0, int& y0, int& x1, int& y1) const;
    // Rejects non-finite boxes and boxes whose cell span would blow the cell
    // budget (escaped coords / garbage input) -- callers treat these as empty.
    bool SaneBox(const Aabb2& b) const;

    Real m_tileSize;
    Vec2 m_origin;
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> m_cells;   // cell -> ids
    std::unordered_map<std::uint32_t, std::vector<std::uint64_t>> m_idCells; // id -> its cells
};

}} // namespace Manifold2D::Physics
