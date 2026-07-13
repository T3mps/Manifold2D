// SpatialHash escape/degenerate-box guard (E01-5 fold-in).
//
// SpatialHash::CellOf uses the same unguarded floor(v/cs) -> int32 pattern as
// SpatialGrid::CellCoord: a NaN / finite-but-huge box makes the int cast
// saturate (MSVC SSE2 cvttsd2si -> INT_MIN for BOTH overflow directions and
// NaN), and Update's / QueryAABB's cell loop over [kx0,kx1]x[ky0,ky1] then
// either registers garbage or iterates an unbounded number of cells -> hang /
// OOM. SpatialHash::SaneBox mirrors the shipped SpatialGrid::SaneBox guard.
//
// These mirror the three shipped SpatialGrid escape tests
// (PhysicsSpatialGridTest.cpp: "survives a non-finite AABB", "survives an
// absurdly large AABB", "rejects a box whose TOTAL cell count blows the
// budget") -- each queries the OFFENDING box itself (so the assertion is
// load-bearing: without the guard the box registers and the query finds it)
// then confirms the hash is not corrupted for a valid nearby id.
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <Manifold2D/Physics/Broadphase/SpatialHash.hpp>

using namespace Manifold2D::Physics;

namespace
{
    Aabb2 Box(Real x0, Real y0, Real x1, Real y1)
    { Aabb2 a; a.min = Vec2(x0, y0); a.max = Vec2(x1, y1); return a; }
}

TEST_CASE("SpatialHash survives escape/degenerate boxes", "[physics]")
{
    // cellSize 1 m: at this residency tile a coordinate value IS its cell index,
    // so the budget arithmetic below is easy to read.
    SpatialHash h(Real(1));
    std::vector<std::uint32_t> out;
    const Real inf = std::numeric_limits<Real>::infinity();
    const Real nan = std::numeric_limits<Real>::quiet_NaN();

    SECTION("non-finite box is dropped, not crashed")
    {
        h.Update(1u, Box(nan, Real(0), inf, Real(10)));            // must not hang/crash
        const int n = h.QueryAABB(Box(nan, Real(0), inf, Real(10)), out); // ditto
        REQUIRE(n == 0);
        REQUIRE(out.empty());
        // A valid id nearby still works (hash is not corrupted).
        h.Update(2u, Box(Real(0), Real(0), Real(1), Real(1)));
        h.QueryAABB(Box(Real(0), Real(0), Real(1), Real(1)), out);
        REQUIRE(std::find(out.begin(), out.end(), 2u) != out.end());
    }

    SECTION("finite-but-huge box is dropped")
    {
        // +-1e30 is far past int range: the pre-cast magnitude bound rejects it
        // BEFORE CellOf's cast is exercised. Without the guard the box registers
        // (MSVC collapses to a single INT_MIN cell) and the query below finds it.
        h.Update(2u, Box(Real(-1e30f), Real(-1e30f), Real(1e30f), Real(1e30f)));
        const int n = h.QueryAABB(Box(Real(-1e30f), Real(-1e30f), Real(1e30f), Real(1e30f)), out);
        REQUIRE(n == 0);                 // treated as empty (out-of-budget)
        REQUIRE(out.empty());
        // A valid id nearby still works (hash is not corrupted).
        h.Update(3u, Box(Real(0), Real(0), Real(1), Real(1)));
        h.QueryAABB(Box(Real(0), Real(0), Real(1), Real(1)), out);
        REQUIRE(std::find(out.begin(), out.end(), 3u) != out.end());
    }

    SECTION("total cell count blows the budget")
    {
        // half-extent 1100 -> 2200 cells/axis: UNDER the per-axis budget
        // (2200 < 65536) AND under the raw-magnitude bound (1100 < 65536), but
        // (2200+1)^2 ~= 4.84M cells > kMaxCellsTotal (1<<22 ~= 4.19M). Only the
        // TOTAL-cell budget branch rejects it. Without the guard the cell loop
        // registers ~4.84M buckets and the query re-walks them (found -> n>0).
        h.Update(4u, Box(Real(-1100), Real(-1100), Real(1100), Real(1100)));
        const int n = h.QueryAABB(Box(Real(-1100), Real(-1100), Real(1100), Real(1100)), out);
        REQUIRE(n == 0);                 // treated as empty (out-of-budget)
        REQUIRE(out.empty());
        // A valid id nearby still works (hash is not corrupted).
        h.Update(5u, Box(Real(0), Real(0), Real(1), Real(1)));
        h.QueryAABB(Box(Real(0), Real(0), Real(1), Real(1)), out);
        REQUIRE(std::find(out.begin(), out.end(), 5u) != out.end());
    }
}
