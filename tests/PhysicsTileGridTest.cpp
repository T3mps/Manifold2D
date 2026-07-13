// Physics M6 P1.7: TileGrid static broadphase -- the keystone.
//
// PORT + MODERNIZE of Client/src/physics/TileGrid.lua (see TileGrid.hpp). The
// Lua stores nothing for walls and answers queries straight from cell flags,
// merging a per-row run of consecutive solid cells into ONE virtual fixture
// (the greedy rectangle merge -- it kills the tile-seam internal-edge catch).
// We port that ALGORITHM; we REFORMULATE the cell->world coordinate math from
// iso (the Lua's diamonds/parallelograms) to plain CARTESIAN axis-aligned
// rects. Because iso is dropped there is NO Lua bit-match for the rect coords;
// the oracle is a SELF-DEFINED Cartesian grid with analytically-derived merged
// rects, which is correct + expected for this task.
//
// Two test groups:
//   1) MERGED-RECT SET -- the merge algorithm. Reformulates the harness
//      "== TileGrid ==" block (main.lua ~160-215) to Cartesian: an 8x4 grid
//      with a row-run + a two-span (gapped) row. Query the whole grid -> the
//      correct merged rects, asserted ANALYTICALLY from cellSize/origin + the
//      per-row-merge rule. Single run -> 1 rect; gap -> 2 rects; offscreen ->
//      0; partial overlap covered.
//   2) TILE-SEAM NO-CATCH -- the keystone property. A swept capsule glides
//      ACROSS a merged horizontal span with no internal-edge catch: because
//      consecutive solid cells merge into ONE rect there are no internal
//      vertical edges between cells, so a capsule sweeping parallel just above
//      the span clears its full length instead of catching at a cell boundary.

#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/Broadphase/Broadphase.hpp>
#include <Manifold2D/Physics/Broadphase/Passability.hpp>
#include <Manifold2D/Physics/Broadphase/TileGrid.hpp>
#include <Manifold2D/Physics/Narrowphase/Gjk.hpp>

using namespace Manifold2D::Physics;
using Catch::Approx;

namespace
{
    // Build the harness's 8x4 grid (reformulated to 0-based Cartesian).
    //
    // Lua (1-based) walls: (3,2)(4,2)(5,2) and (2,3)(6,3)(7,3).
    // 0-based: row 1 cols 2,3,4 (one merged run) and row 2 cols 1, 5, 6
    // (col 1 alone + a 5..6 run -> TWO spans). Total = 3 merged rects.
    constexpr int kW = 8;
    constexpr int kH = 4;

    GridPassability MakeHarnessGrid()
    {
        GridPassability g(kW, kH);
        // row 1 (0-based): a single merged run, cols 2..4
        g.SetSolid(2, 1, true);
        g.SetSolid(3, 1, true);
        g.SetSolid(4, 1, true);
        // row 2 (0-based): a gapped row -> two runs, col 1 and cols 5..6
        g.SetSolid(1, 2, true);
        g.SetSolid(5, 2, true);
        g.SetSolid(6, 2, true);
        return g;
    }

    // Find a rect in `rects` whose min/max match (within f32 tolerance).
    bool HasRect(const std::vector<Aabb2>& rects, Real minx, Real miny,
                 Real maxx, Real maxy)
    {
        for (const Aabb2& r : rects)
        {
            if (r.min.x == Approx(minx) && r.min.y == Approx(miny) &&
                r.max.x == Approx(maxx) && r.max.y == Approx(maxy))
            {
                return true;
            }
        }
        return false;
    }
} // namespace

// ====================================================================
// 1) Merged-rect set: the greedy per-row run-merge, analytic Cartesian oracle.
// ====================================================================
TEST_CASE("physics: tilegrid per-row merge yields the analytic rect set",
          "[physics]")
{
    const Real     cs = Real(3.2);
    const Vec2     origin(Real(0), Real(0));
    GridPassability grid = MakeHarnessGrid();
    TileGrid       tg(grid, cs, origin);

    // Whole-grid query: the world box is exactly the 8x4 grid extent. With
    // origin {0,0} and cellSize 3.2 that is [0,0]..[25.6,12.8]. PAD adds slack
    // but the scan still clamps to the grid bounds.
    const Aabb2 whole{ Vec2(Real(0), Real(0)), Vec2(Real(kW) * cs, Real(kH) * cs) };
    std::vector<Aabb2> rects;
    const int n = tg.Query(whole, rects);

    // THREE merged spans total (harness: eq(n, 3, "three merged spans total")).
    CHECK(n == 3);
    CHECK(rects.size() == 3u);

    // Analytic expectations (cell (cx,cy) -> [cx*cs, cy*cs]..[(cx+1)*cs,(cy+1)*cs];
    // a run runStart..runEnd in row cy -> [runStart*cs, cy*cs]..[(runEnd+1)*cs,(cy+1)*cs]):
    //
    //   row 1 run 2..4 -> [2*3.2, 1*3.2]..[5*3.2, 2*3.2] = [6.4,3.2]..[16,6.4]
    //   row 2 run 1..1 -> [1*3.2, 2*3.2]..[2*3.2, 3*3.2] = [3.2,6.4]..[6.4,9.6]
    //   row 2 run 5..6 -> [5*3.2, 2*3.2]..[7*3.2, 3*3.2] = [16,6.4]..[22.4,9.6]
    CHECK(HasRect(rects, 6.4f, 3.2f, 16.0f, 6.4f));   // the single merged run
    CHECK(HasRect(rects, 3.2f, 6.4f, 6.4f, 9.6f));    // gapped row, first run (col 1)
    CHECK(HasRect(rects, 16.0f, 6.4f, 22.4f, 9.6f));  // gapped row, second run (5..6)

    // The merged run spans EXACTLY 3 cells wide (9.6 m) -- one rect, not three.
    // (Per-cell rects would be 3.2 m each with internal seams; the merge fuses
    // them into a single 9.6 m rect with no internal vertical edges.)
    for (const Aabb2& r : rects)
    {
        if (r.min.x == Approx(6.4) && r.min.y == Approx(3.2))
        {
            CHECK((r.max.x - r.min.x) == Approx(Real(3) * cs));
            CHECK((r.max.y - r.min.y) == Approx(cs));
        }
    }
}

TEST_CASE("physics: tilegrid offscreen query is empty, partial overlap finds runs",
          "[physics]")
{
    const Real     cs = Real(3.2);
    GridPassability grid = MakeHarnessGrid();
    TileGrid       tg(grid, cs, Vec2(Real(0), Real(0)));

    std::vector<Aabb2> rects;

    // Far offscreen query (the harness's offscreen-empty check): no in-bounds
    // solid cell -> 0 rects.
    const Aabb2 offscreen{ Vec2(Real(-1000), Real(-1000)),
                           Vec2(Real(-900), Real(-900)) };
    CHECK(tg.Query(offscreen, rects) == 0);
    CHECK(rects.empty());

    // A query that only PARTIALLY overlaps the grid: a tight window over the
    // gapped row's second run only (cells 5..6 of row 2 = world [16,6.4]..[22.4,9.6]).
    // PAD = 1 widens the cell scan, so it can also pick up the row-1 run above
    // (cells 2..4 of row 1) and col 1 of row 2 -- assert the 5..6 run is found
    // and the result is a subset of the full 3 (not all the offscreen junk).
    const Aabb2 window{ Vec2(Real(17), Real(7)), Vec2(Real(20), Real(9)) };
    const int   n = tg.Query(window, rects);
    CHECK(n >= 1);
    CHECK(static_cast<std::size_t>(n) == rects.size());
    CHECK(HasRect(rects, 16.0f, 6.4f, 22.4f, 9.6f)); // the 5..6 run is present

    // An empty grid (no solids) returns 0 for any query.
    GridPassability emptyGrid(kW, kH);
    TileGrid        emptyTg(emptyGrid, cs);
    const Aabb2     whole{ Vec2(Real(0), Real(0)),
                       Vec2(Real(kW) * cs, Real(kH) * cs) };
    CHECK(emptyTg.Query(whole, rects) == 0);
    CHECK(rects.empty());
}

TEST_CASE("physics: tilegrid honors a non-zero origin", "[physics]")
{
    // The Cartesian cell->world mapping offsets by `origin`; verify the run
    // rect shifts by exactly the origin.
    const Real      cs = Real(3.2);
    const Vec2      origin(Real(100), Real(-50));
    GridPassability grid = MakeHarnessGrid();
    TileGrid        tg(grid, cs, origin);

    const Aabb2        whole{ origin, origin + Vec2(Real(kW) * cs, Real(kH) * cs) };
    std::vector<Aabb2> rects;
    CHECK(tg.Query(whole, rects) == 3);

    // row 1 run 2..4 shifted by origin: [100+6.4, -50+3.2]..[100+16, -50+6.4]
    CHECK(HasRect(rects, Real(106.4), Real(-46.8), Real(116), Real(-43.6)));
}

TEST_CASE("physics: tilegrid Query reuses the out vector (zero steady-state alloc)",
          "[physics]")
{
    const Real      cs = Real(3.2);
    GridPassability grid = MakeHarnessGrid();
    TileGrid        tg(grid, cs);

    const Aabb2 whole{ Vec2(Real(0), Real(0)),
                       Vec2(Real(kW) * cs, Real(kH) * cs) };
    std::vector<Aabb2> out;

    tg.Query(whole, out);
    const std::size_t cap0 = out.capacity();
    REQUIRE(out.size() == 3u);
    REQUIRE(cap0 >= 3u);

    // Repeated queries must NOT grow the capacity (clear() + push_back reuse).
    for (int i = 0; i < 200; ++i)
    {
        tg.Query(whole, out);
    }
    CHECK(out.capacity() == cap0);
    CHECK(out.size() == 3u);

    // An offscreen query clears but keeps the capacity (no shrink, no grow).
    tg.Query(Aabb2{ Vec2(Real(-1e6f), Real(-1e6f)), Vec2(Real(-9e5f), Real(-9e5f)) },
             out);
    CHECK(out.empty());
    CHECK(out.capacity() == cap0);
}

// ====================================================================
// 2) Tile-seam no-catch: the keystone property.
//
// A horizontal run of N solid cells merges into ONE rect, so there are NO
// internal vertical edges between the cells. A capsule swept HORIZONTALLY just
// above the span's top edge therefore travels its full length with no spurious
// early stop at an internal cell boundary -- it either clears the span
// entirely, or (if its path dips into the span) stops only at the span's OUTER
// extent. Contrast: per-cell (UN-merged) rects would expose a vertical edge at
// every cell boundary, and a grazing sweep could catch on one of them. The
// merge eliminates them. (The full CharacterController slide-down-a-column test
// is P1.10; here we prove the per-row merge removes internal horizontal-run edges.)
// ====================================================================
TEST_CASE("physics: tilegrid merged span has no internal seam to catch on",
          "[physics]")
{
    const Real cs = Real(3.2);
    // A single long horizontal run: row 0, cols 0..4 (5 cells). origin {0,0}.
    // Merged rect = [0,0]..[16,3.2]. Internal cell boundaries WOULD be at
    // x = 3.2, 6.4, 9.6, 12.8 -- the merge fuses them away.
    GridPassability grid(/*w*/ 8, /*h*/ 2);
    for (int cx = 0; cx <= 4; ++cx)
    {
        grid.SetSolid(cx, 0, true);
    }
    TileGrid tg(grid, cs);

    const Aabb2 whole{ Vec2(Real(0), Real(0)), Vec2(Real(8) * cs, Real(2) * cs) };
    std::vector<Aabb2> rects;
    const int          n = tg.Query(whole, rects);

    // EXACTLY ONE merged rect for the 5-cell run -- the seam-killer. If the
    // grid emitted per-cell rects we'd see 5 here (and the internal edges).
    REQUIRE(n == 1);
    const Aabb2 span = rects[0];
    REQUIRE(span.min.x == Approx(0));
    REQUIRE(span.min.y == Approx(0));
    REQUIRE(span.max.x == Approx(Real(5) * cs)); // 16 -- the OUTER extent
    REQUIRE(span.max.y == Approx(cs));           // 3.2

    // Build the merged rect as a collidable polygon (CCW box) for the cast.
    const Aabb2 r        = span;
    const Vec2  poly[4]  = { Vec2(r.min.x, r.min.y), Vec2(r.max.x, r.min.y),
                             Vec2(r.max.x, r.max.y), Vec2(r.min.x, r.max.y) };

    // A capsule grazing JUST ABOVE the span's top edge (y = 3.2). Capsule
    // half-length 0.6, radius 0.4; center placed so its bottom (y_center +
    // radius) sits a hair above the span top -> it should NOT touch the span
    // at all, and crucially must not catch at any internal x (there are none).
    const Shape cap = MakeCapsule(/*halfLen*/ Real(0.6), /*radius*/ Real(0.4));

    // Start to the LEFT of the span, sweep RIGHT across its full width, riding
    // at y just above the top edge: center y = top - radius - gap.
    const Real gap     = Real(0.1);
    const Real rideY   = span.min.y - Real(0.4) - gap; // above the top edge
    const Vec2 start(span.min.x - Real(2), rideY);
    const Vec2 endPt(span.max.x + Real(2), rideY);
    const Vec2 translation = endPt - start;

    Transform   xf;
    xf.position = start;
    const ShapeCastResult clearCast =
        ShapeCastPoly(cap, xf, translation, poly, 4);

    // Riding above the top edge with a clearance gap: the swept capsule never
    // reaches the span (no internal edge to catch on along the way) -> no hit,
    // it travels the full translation.
    CHECK_FALSE(clearCast.hit);

    // Now drive the capsule THROUGH the span's left face at mid-height: it must
    // stop at the span's OUTER left extent, NOT at an internal cell boundary
    // (x = 3.2/6.4/9.6/12.8 do not exist as edges anymore). Sweep left->right
    // at the span's vertical mid-line.
    const Real  midY = (span.min.y + span.max.y) * Real(0.5);
    Transform   xf2;
    xf2.position = Vec2(span.min.x - Real(4), midY);
    const Vec2  trans2 = Vec2(Real(8), Real(0)); // drive into the left face
    const ShapeCastResult faceCast =
        ShapeCastPoly(cap, xf2, trans2, poly, 4);

    REQUIRE(faceCast.hit);
    // Impact center x at TOI: start.x + trans2.x * t. It must land at the OUTER
    // left face (x = span.min.x), offset by the capsule's right extent
    // (halfLen + radius = 1.0) and the cast tolerance. The KEY assertion: the
    // stop is at the span's outer extent, far from any internal cell boundary
    // (the first internal boundary would have been x = 3.2, well to the right
    // of the outer left face at x = 0).
    const Real impactX = xf2.position.x + trans2.x * faceCast.t;
    // The capsule's leading (right) edge at impact ~ impactX + (halfLen+radius).
    const Real leadingX = impactX + (Real(0.6) + Real(0.4));
    // Leading edge stops at (or a tolerance short of) the OUTER left face x = 0.
    CHECK(leadingX <= Approx(span.min.x).margin(0.02));
    // And it is NOWHERE NEAR the first internal cell boundary x = 3.2 (which
    // the merge eliminated) -- it stopped at the outer extent, not mid-run.
    CHECK(leadingX < Real(3.2) - Real(1.0));
}
