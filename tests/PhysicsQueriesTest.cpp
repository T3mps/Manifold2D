// Physics M6 P1.9: PhysicsWorld queries -- Raycast / LineOfSight / ShapeCast /
// OverlapShape / QueryAABB.
//
// PORT NOTE: a BEHAVIORAL port of the physics_harness raycast / shapeCast / LOS
// blocks (Client/src/tests/physics_harness/main.lua ~420-456, ~608-684) and the
// world query methods (Client/src/physics/PhysicsWorld.lua raycast 502-604,
// queryAABB 473-484, Cast.lua shapeCast/rayVsBody). The harness blocks are
// ISO-COUPLED (Iso.toCellF lattice, iso world coords), so we DO NOT bit-match
// their iso numbers. Instead we REFORMULATE to a SELF-DEFINED CARTESIAN oracle
// (TileGrid cellSize/origin + GridPassability cells) and assert the SAME
// behavioral / analytic invariants:
//   * Raycast vs cells: a wall cell in the path is hit with 0 < t < 1 and the
//     correct cellX/cellY; a clear column misses; a degenerate ray returns none.
//   * Raycast vs bodies: AABB slab hit (exact); a near body beats a farther
//     wall (nearest wins); a diagonal ray through a circle center hits at
//     EXACTLY r from the center (the harness "exact surface hit r=10" invariant).
//   * LineOfSight: a TALL cell blocks LOS; a LOW cell (BlocksSight=false) does
//     NOT; a clear column has LOS; degenerate is false.
//   * ShapeCast: a circle swept toward a wall stops a radius SHORT of where a
//     ray hits (cast.t < ray.t); a clear cast misses; the hit reports a near-
//     touching distance + a push-back normal.
//   * OverlapShape: a query shape returns exactly the overlapping bodies.
//   * QueryAABB: both body kinds (P1.8 coverage + a non-overlap exclusion).
//
// The cell DDA is the Cartesian (min-corner) reformulation of the Lua's
// centered iso lattice -- consistent with TileGrid::CellBox / IsSolid.
//
// PRESENTATION-FREE + C++20-clean.

#include <cmath>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>
#include <Manifold2D/Physics/Broadphase/Passability.hpp>

using namespace Manifold2D::Physics;
using Catch::Approx;

namespace
{
    // A 16x16 Cartesian grid, cellSize 1, origin (0,0) -- a 16x16 m world.
    // Cell C spans world [C*1, (C+1)*1) on each axis (min-corner indexing,
    // matching TileGrid::CellBox). The center of cell (cx,cy) is
    // (cx*1+0.5, cy*1+0.5).
    constexpr int  kGridW    = 16;
    constexpr int  kGridH    = 16;
    constexpr Real kCellSize = Real(1);

    // World position of cell (cx,cy)'s center.
    Vec2 CellCenter(int cx, int cy)
    {
        return Vec2(static_cast<Real>(cx) * kCellSize + kCellSize * Real(0.5),
                    static_cast<Real>(cy) * kCellSize + kCellSize * Real(0.5));
    }

    WorldDef MakeWorldDef(const IPassabilitySource& src)
    {
        WorldDef def;
        def.passability  = &src;
        def.tileCellSize = kCellSize;
        def.tileOrigin   = Vec2(Real(0), Real(0));
        return def;
    }
}

// ---------------------------------------------------------------------------
// Raycast vs cells (Cartesian self-defined oracle)
// ---------------------------------------------------------------------------

TEST_CASE("Raycast: a wall cell in the path is hit with 0<t<1 + correct cell",
          "[physics][queries][raycast]")
{
    GridPassability grid(kGridW, kGridH);
    // Wall at cell (5, 2). A horizontal ray along row 2 must hit it.
    grid.SetSolid(5, 2, true);

    PhysicsWorld w(MakeWorldDef(grid));

    // Ray along the center of row 2, from cell 0 to cell 10 (x: 0.5 -> 10.5).
    const Vec2 from = CellCenter(0, 2);   // (0.5, 2.5)
    const Vec2 to   = CellCenter(10, 2);  // (10.5, 2.5)

    const auto hit = w.Raycast(from, to);
    REQUIRE(hit.has_value());
    REQUIRE(hit->isCell);
    REQUIRE(hit->cellX == 5);
    REQUIRE(hit->cellY == 2);
    REQUIRE(hit->t > Real(0));
    REQUIRE(hit->t < Real(1));
    // The hit enters cell 5 at its left boundary x = 5; t = (5-0.5)/(10.5-0.5) = 0.45.
    REQUIRE(hit->t == Approx(Real(0.45)));
    REQUIRE(hit->point.x == Approx(Real(5)));
    REQUIRE(hit->point.y == Approx(Real(2.5)));
}

TEST_CASE("Raycast: a clear column yields no cell hit", "[physics][queries][raycast]")
{
    GridPassability grid(kGridW, kGridH);
    // Wall on a DIFFERENT row -- not in the ray's path.
    grid.SetSolid(5, 7, true);

    PhysicsWorld w(MakeWorldDef(grid));

    const Vec2 from = CellCenter(0, 2);
    const Vec2 to   = CellCenter(10, 2);

    // cellsOnly so no bodies interfere; clear row -> no hit.
    RaycastOpts opts;
    opts.cellsOnly = true;
    const auto hit = w.Raycast(from, to, opts);
    REQUIRE_FALSE(hit.has_value());
}

TEST_CASE("Raycast: degenerate (zero-length) ray returns none",
          "[physics][queries][raycast]")
{
    GridPassability grid(kGridW, kGridH);
    grid.SetSolid(2, 2, true);
    PhysicsWorld w(MakeWorldDef(grid));

    const Vec2 p = CellCenter(2, 2);
    const auto hit = w.Raycast(p, p); // from == to
    REQUIRE_FALSE(hit.has_value());
}

TEST_CASE("Raycast: a diagonal wall cell is hit on the correct cell",
          "[physics][queries][raycast]")
{
    GridPassability grid(kGridW, kGridH);
    // Diagonal wall at (3,3). A 45-degree ray along the diagonal hits it.
    grid.SetSolid(3, 3, true);
    PhysicsWorld w(MakeWorldDef(grid));

    RaycastOpts opts;
    opts.cellsOnly = true;
    const Vec2 from = CellCenter(0, 0); // (0.5, 0.5)
    const Vec2 to   = CellCenter(8, 8); // (8.5, 8.5)
    const auto hit = w.Raycast(from, to, opts);
    REQUIRE(hit.has_value());
    REQUIRE(hit->isCell);
    REQUIRE(hit->cellX == 3);
    REQUIRE(hit->cellY == 3);
    REQUIRE(hit->t > Real(0));
    REQUIRE(hit->t < Real(1));
}

// ---------------------------------------------------------------------------
// Raycast vs bodies (coordinate-agnostic invariants)
// ---------------------------------------------------------------------------

TEST_CASE("Raycast: AABB body slab hit is exact", "[physics][queries][raycast]")
{
    // No passability source -> no cell pass; only the body test runs.
    WorldDef wd;
    PhysicsWorld w(wd);

    BodyDef bd;
    bd.type     = BodyType::Static;
    bd.position = Vec2(Real(10), Real(0));
    bd.shape    = MakeAabb(Real(1), Real(1)); // box spans x[9,11], y[-1,1]
    BodyHandle box = w.AddBody(bd);

    // Horizontal ray from x=0 to x=20 at y=0 -> enters the box at x=9.
    const auto hit = w.Raycast(Vec2(Real(0), Real(0)), Vec2(Real(20), Real(0)));
    REQUIRE(hit.has_value());
    REQUIRE_FALSE(hit->isCell);
    REQUIRE(hit->body == box);
    // Enters at x=9; t = 9/20 = 0.45.
    REQUIRE(hit->t == Approx(Real(0.45)));
    REQUIRE(hit->point.x == Approx(Real(9)));
    REQUIRE(hit->point.y == Approx(Real(0)));
}

TEST_CASE("Raycast: a near body beats a farther wall (nearest wins)",
          "[physics][queries][raycast]")
{
    GridPassability grid(kGridW, kGridH);
    // Wall FAR down the row at cell (10, 2).
    grid.SetSolid(10, 2, true);
    PhysicsWorld w(MakeWorldDef(grid));

    // A static AABB body NEARER than the wall: spans x[4,6] on row-2 center.
    BodyDef bd;
    bd.type     = BodyType::Static;
    bd.position = Vec2(Real(5), Real(2.5)); // row 2 center y = 2.5
    bd.shape    = MakeAabb(Real(1), Real(1)); // x[4,6], y[1.5,3.5]
    BodyHandle box = w.AddBody(bd);

    const Vec2 from = CellCenter(0, 2);   // (0.5, 2.5)
    const Vec2 to   = CellCenter(14, 2);  // (14.5, 2.5)

    const auto hit = w.Raycast(from, to);
    REQUIRE(hit.has_value());
    // The body (enters x=4, t~0.25) beats the wall cell 10 (enters x=10,
    // t~0.68): the NEARER hit wins, and it is the body, not a cell.
    REQUIRE_FALSE(hit->isCell);
    REQUIRE(hit->body == box);
    REQUIRE(hit->point.x == Approx(Real(4)));
}

TEST_CASE("Raycast: a farther body loses to a nearer wall (nearest wins, "
          "cell side)", "[physics][queries][raycast]")
{
    GridPassability grid(kGridW, kGridH);
    // Wall NEAR at cell (2, 2): entered at x=2.
    grid.SetSolid(2, 2, true);
    PhysicsWorld w(MakeWorldDef(grid));

    // A body FARTHER than the wall: spans x[9,11].
    BodyDef bd;
    bd.type     = BodyType::Static;
    bd.position = Vec2(Real(10), Real(2.5));
    bd.shape    = MakeAabb(Real(1), Real(1));
    w.AddBody(bd);

    const Vec2 from = CellCenter(0, 2);   // (0.5, 2.5)
    const Vec2 to   = CellCenter(14, 2);  // (14.5, 2.5)

    const auto hit = w.Raycast(from, to);
    REQUIRE(hit.has_value());
    // The wall (x=2) is nearer than the body (x=9): the cell wins.
    REQUIRE(hit->isCell);
    REQUIRE(hit->cellX == 2);
    REQUIRE(hit->cellY == 2);
    REQUIRE(hit->point.x == Approx(Real(2)));
}

TEST_CASE("Raycast: a diagonal ray through a circle center hits at exactly r",
          "[physics][queries][raycast]")
{
    WorldDef wd;
    PhysicsWorld w(wd); // no cells; body test only

    const Real r = Real(1);
    const Vec2 center(Real(10), Real(10));
    BodyDef bd;
    bd.type     = BodyType::Static;
    bd.position = center;
    bd.shape    = MakeCircle(r);
    w.AddBody(bd);

    // Diagonal ray heading straight at the circle center from the lower-left.
    const Vec2 from(Real(0), Real(0));
    const Vec2 to(Real(20), Real(20)); // passes through (10,10)

    const auto hit = w.Raycast(from, to);
    REQUIRE(hit.has_value());
    REQUIRE_FALSE(hit->isCell);
    // The harness "exact surface hit (r=1)" invariant: the hit point lies on
    // the circle surface, i.e. |hit.point - center| == r (within ~0.02 --
    // margin ~3.2x kShapeCastTol (1.25*kLinearSlop) (MKS P4)).
    const Real dx = hit->point.x - center.x;
    const Real dy = hit->point.y - center.y;
    const Real dist = std::sqrt(dx * dx + dy * dy);
    REQUIRE(dist == Approx(r).margin(Real(0.02)));
    // And it hit the NEAR surface (the lower-left side), so the hit is before
    // the center along the ray.
    REQUIRE(hit->point.x < center.x);
    REQUIRE(hit->point.y < center.y);
}

// ---------------------------------------------------------------------------
// LineOfSight (the obstacle-class tiering seam)
// ---------------------------------------------------------------------------

TEST_CASE("LineOfSight: a TALL cell blocks sight", "[physics][queries][los]")
{
    GridPassability grid(kGridW, kGridH);
    // TALL obstacle: blocks movement AND sight.
    grid.SetSolid(5, 2, true);
    grid.SetBlocksSight(5, 2, true);
    PhysicsWorld w(MakeWorldDef(grid));

    const Vec2 from = CellCenter(0, 2);
    const Vec2 to   = CellCenter(10, 2);
    REQUIRE_FALSE(w.LineOfSight(from, to)); // blocked
}

TEST_CASE("LineOfSight: a LOW cell (see-through) does NOT block sight",
          "[physics][queries][los]")
{
    GridPassability grid(kGridW, kGridH);
    // LOW obstacle: blocks MOVEMENT (IsSolid true) but NOT sight
    // (BlocksSight false). The seam tiering: LineOfSight uses BlocksSight, so a
    // movement raycast would be blocked here but sight passes through.
    grid.SetSolid(5, 2, true);
    grid.SetBlocksSight(5, 2, false);
    PhysicsWorld w(MakeWorldDef(grid));

    const Vec2 from = CellCenter(0, 2);
    const Vec2 to   = CellCenter(10, 2);

    // Sight passes (LOW is see-through)...
    REQUIRE(w.LineOfSight(from, to));

    // ...but a MOVEMENT raycast (IsSolid) IS blocked by the same LOW cell.
    RaycastOpts mov;
    mov.cellsOnly = true; // movement rule = IsSolid (tallOnly false)
    const auto blocked = w.Raycast(from, to, mov);
    REQUIRE(blocked.has_value());
    REQUIRE(blocked->isCell);
    REQUIRE(blocked->cellX == 5);
    REQUIRE(blocked->cellY == 2);
}

TEST_CASE("LineOfSight: a clear column has sight", "[physics][queries][los]")
{
    GridPassability grid(kGridW, kGridH);
    grid.SetSolid(5, 9, true);
    grid.SetBlocksSight(5, 9, true); // TALL, but on a different row
    PhysicsWorld w(MakeWorldDef(grid));

    const Vec2 from = CellCenter(0, 2);
    const Vec2 to   = CellCenter(10, 2);
    REQUIRE(w.LineOfSight(from, to)); // clear
}

TEST_CASE("LineOfSight: degenerate segment has no sight", "[physics][queries][los]")
{
    GridPassability grid(kGridW, kGridH);
    PhysicsWorld w(MakeWorldDef(grid));
    const Vec2 p = CellCenter(3, 3);
    REQUIRE_FALSE(w.LineOfSight(p, p)); // ports `if not x1 then return false`
}

// ---------------------------------------------------------------------------
// ShapeCast (world) -- reuses the P1.4 GJK conservative-advancement primitive
// ---------------------------------------------------------------------------

TEST_CASE("ShapeCast: a swept circle stops a radius short of where a ray hits",
          "[physics][queries][shapecast]")
{
    WorldDef wd;
    PhysicsWorld w(wd); // no cells; cast against a static body

    // A static wall AABB the cast moves toward: spans x[10,12], y[-5,5].
    BodyDef bd;
    bd.type     = BodyType::Static;
    bd.position = Vec2(Real(11), Real(0));
    bd.shape    = MakeAabb(Real(1), Real(5));
    w.AddBody(bd);

    const Real circleR = Real(0.8);
    const Shape circle = MakeCircle(circleR);
    const Vec2  start(Real(0), Real(0));
    const Vec2  delta(Real(20), Real(0)); // sweep right past the wall

    const auto cast = w.ShapeCast(circle, start, delta);
    REQUIRE(cast.has_value());
    REQUIRE(cast->body != kInvalidBody);

    // A zero-radius RAY along the same line hits the wall's near face at x=10,
    // t = 10/20 = 0.5. The radius-0.8 circle must stop ~its radius SHORT of
    // that (its center stops near x=9.2), so cast.t < ray.t.
    const auto ray = w.Raycast(start, Vec2(start.x + delta.x, start.y + delta.y));
    REQUIRE(ray.has_value());
    REQUIRE(cast->t < ray->t);

    // The cast reports a near-touching surface distance (< the cast TOL) and a
    // push-back normal pointing back toward the moving shape (i.e. -x, away
    // from the wall the circle is approaching from the left).
    // margin ~1.6x kShapeCastTol (1.25*kLinearSlop) (MKS P4)
    REQUIRE(cast->distance < Real(0.01));
    REQUIRE(cast->normal.x < Real(0)); // points back toward the mover
    // The circle center at TOI is ~radius short of the wall face (x=10).
    // margin ~3.2x kShapeCastTol (1.25*kLinearSlop) (MKS P4)
    REQUIRE(cast->point.x == Approx(Real(10) - circleR).margin(Real(0.02)));
}

TEST_CASE("ShapeCast: a clear cast misses", "[physics][queries][shapecast]")
{
    WorldDef wd;
    PhysicsWorld w(wd);

    // A static body well off the cast's path.
    BodyDef bd;
    bd.type     = BodyType::Static;
    bd.position = Vec2(Real(10), Real(50)); // far away in y
    bd.shape    = MakeAabb(Real(1), Real(1));
    w.AddBody(bd);

    const Shape circle = MakeCircle(Real(0.5));
    const auto cast =
        w.ShapeCast(circle, Vec2(Real(0), Real(0)), Vec2(Real(5), Real(0)));
    REQUIRE_FALSE(cast.has_value());
}

TEST_CASE("ShapeCast: degenerate (zero-length) delta returns none",
          "[physics][queries][shapecast]")
{
    WorldDef wd;
    PhysicsWorld w(wd);
    BodyDef bd;
    bd.type     = BodyType::Static;
    bd.position = Vec2(Real(1), Real(0));
    bd.shape    = MakeAabb(Real(1), Real(1));
    w.AddBody(bd);

    const Shape circle = MakeCircle(Real(0.5));
    const auto cast =
        w.ShapeCast(circle, Vec2(Real(0), Real(0)), Vec2(Real(0), Real(0)));
    REQUIRE_FALSE(cast.has_value());
}

TEST_CASE("ShapeCast: movers are obstacles only with opts.movers",
          "[physics][queries][shapecast]")
{
    WorldDef wd;
    PhysicsWorld w(wd);

    // A KINEMATIC body in the cast path -- a mover, not a static.
    BodyDef bd;
    bd.type     = BodyType::Kinematic;
    bd.position = Vec2(Real(10), Real(0));
    bd.shape    = MakeAabb(Real(1), Real(5));
    w.AddBody(bd);

    const Shape circle = MakeCircle(Real(0.8));
    const Vec2  start(Real(0), Real(0));
    const Vec2  delta(Real(20), Real(0));

    // Without opts.movers, the kinematic body is NOT an obstacle -> miss.
    REQUIRE_FALSE(w.ShapeCast(circle, start, delta).has_value());

    // With opts.movers, it IS -> hit.
    ShapeCastOpts opts;
    opts.movers = true;
    const auto cast = w.ShapeCast(circle, start, delta, opts);
    REQUIRE(cast.has_value());
    REQUIRE(cast->body != kInvalidBody);
}

TEST_CASE("ShapeCast: opts.exclude skips the caster's own body",
          "[physics][queries][shapecast]")
{
    WorldDef wd;
    PhysicsWorld w(wd);

    // The "caster" body overlapping the start, plus a wall down the path.
    BodyDef self;
    self.type     = BodyType::Kinematic;
    self.position = Vec2(Real(0), Real(0));
    self.shape    = MakeCircle(Real(0.8));
    BodyHandle selfH = w.AddBody(self);

    BodyDef wall;
    wall.type     = BodyType::Kinematic;
    wall.position = Vec2(Real(10), Real(0));
    wall.shape    = MakeAabb(Real(1), Real(5));
    w.AddBody(wall);

    const Shape circle = MakeCircle(Real(0.8));
    ShapeCastOpts opts;
    opts.movers  = true;
    opts.exclude = selfH;

    const auto cast =
        w.ShapeCast(circle, Vec2(Real(0), Real(0)), Vec2(Real(20), Real(0)), opts);
    REQUIRE(cast.has_value());
    // The hit is the wall, not the excluded self body.
    REQUIRE(cast->body != selfH);
}

// ---------------------------------------------------------------------------
// OverlapShape
// ---------------------------------------------------------------------------

TEST_CASE("OverlapShape: returns exactly the overlapping bodies",
          "[physics][queries][overlap]")
{
    WorldDef wd;
    PhysicsWorld w(wd);

    // Body A: overlaps the query. Body B: also overlaps. Body C: far away.
    BodyDef a;
    a.type     = BodyType::Static;
    a.position = Vec2(Real(0), Real(0));
    a.shape    = MakeCircle(Real(1));
    BodyHandle ha = w.AddBody(a);

    BodyDef b;
    b.type     = BodyType::Kinematic;
    b.position = Vec2(Real(1.5), Real(0));
    b.shape    = MakeCircle(Real(1));
    BodyHandle hb = w.AddBody(b);

    BodyDef c;
    c.type     = BodyType::Static;
    c.position = Vec2(Real(50), Real(50)); // far away -> excluded
    c.shape    = MakeCircle(Real(1));
    w.AddBody(c);

    // Query a circle centered at (0.5,0) radius 1.2: overlaps A (dist 0.5 <
    // 2.2) and B (dist 1.0 < 2.2), excludes C.
    const Shape query = MakeCircle(Real(1.2));
    const Transform xf{ Vec2(Real(0.5), Real(0)), Real(0) };

    std::vector<BodyHandle> hits;
    const int n = w.OverlapShape(query, xf, hits);
    REQUIRE(n == 2);
    // Index-ordered: A added first, then B.
    REQUIRE(hits[0] == ha);
    REQUIRE(hits[1] == hb);
}

TEST_CASE("OverlapShape: a non-overlapping query returns empty",
          "[physics][queries][overlap]")
{
    WorldDef wd;
    PhysicsWorld w(wd);

    BodyDef a;
    a.type     = BodyType::Static;
    a.position = Vec2(Real(0), Real(0));
    a.shape    = MakeCircle(Real(0.5));
    w.AddBody(a);

    const Shape query = MakeCircle(Real(0.5));
    const Transform xf{ Vec2(Real(10), Real(10)), Real(0) }; // far away

    std::vector<BodyHandle> hits;
    const int n = w.OverlapShape(query, xf, hits);
    REQUIRE(n == 0);
    REQUIRE(hits.empty());
}

// ---------------------------------------------------------------------------
// QueryAABB (P1.8 coverage + a non-overlap exclusion gap)
// ---------------------------------------------------------------------------

TEST_CASE("QueryAABB: includes overlapping, excludes non-overlapping",
          "[physics][queries][aabb]")
{
    WorldDef wd;
    PhysicsWorld w(wd);

    BodyDef in;
    in.type     = BodyType::Kinematic;
    in.position = Vec2(Real(1), Real(1));
    in.shape    = MakeCircle(Real(0.5));
    BodyHandle hin = w.AddBody(in);

    BodyDef out;
    out.type     = BodyType::Static;
    out.position = Vec2(Real(50), Real(50));
    out.shape    = MakeAabb(Real(0.5), Real(0.5));
    w.AddBody(out);

    std::vector<BodyHandle> hits;
    const int n =
        w.QueryAABB(Aabb{ Vec2(Real(0), Real(0)), Vec2(Real(5), Real(5)) }, hits);
    REQUIRE(n == 1);
    REQUIRE(hits[0] == hin);
}

// ---------------------------------------------------------------------------
// OverlapShape includes sensors (contract lock)
// ---------------------------------------------------------------------------

TEST_CASE("OverlapShape: a sensor body overlapping the query shape IS returned",
          "[physics][queries][overlap]")
{
    // OverlapShape's documented contract: sensor bodies ARE included (unlike
    // ShapeCast, which skips sensors). This test locks that contract so any
    // future refactor that accidentally silences sensors will fail here.
    WorldDef wd;
    PhysicsWorld w(wd);

    // A non-sensor body well outside the query (should not appear).
    BodyDef far;
    far.type     = BodyType::Static;
    far.position = Vec2(Real(50), Real(50));
    far.shape    = MakeCircle(Real(0.5));
    w.AddBody(far);

    // A SENSOR body squarely overlapping the query center.
    BodyDef sensor;
    sensor.type      = BodyType::Static;
    sensor.position  = Vec2(Real(0), Real(0));
    sensor.shape     = MakeCircle(Real(1));
    sensor.isSensor  = true;
    BodyHandle hs = w.AddBody(sensor);

    // A non-sensor body also overlapping (included for the non-sensor path too).
    BodyDef solid;
    solid.type     = BodyType::Kinematic;
    solid.position = Vec2(Real(0.5), Real(0));
    solid.shape    = MakeCircle(Real(1));
    BodyHandle hk = w.AddBody(solid);

    // Query circle at origin radius 0.8: overlaps both the sensor and the solid.
    const Shape query = MakeCircle(Real(0.8));
    const Transform xf{ Vec2(Real(0), Real(0)), Real(0) };

    std::vector<BodyHandle> hits;
    const int n = w.OverlapShape(query, xf, hits);
    REQUIRE(n == 2);
    // Both the sensor (hs) and the solid mover (hk) must appear; index-ordered
    // (sensor was added second, solid third -> hs < hk by index).
    bool foundSensor = false;
    bool foundSolid  = false;
    for (const BodyHandle h : hits)
    {
        if (h == hs) foundSensor = true;
        if (h == hk) foundSolid  = true;
    }
    REQUIRE(foundSensor); // sensor IS included
    REQUIRE(foundSolid);
    // Confirm via IsSensor that we are actually testing a sensor body.
    REQUIRE(w.IsSensor(hs));
    REQUIRE_FALSE(w.IsSensor(hk));
}

// ---------------------------------------------------------------------------
// ShapeCast vs a tile span (exercises the ShapeCastPoly-against-span path)
// ---------------------------------------------------------------------------

TEST_CASE("ShapeCast: a swept circle hits a solid tile span",
          "[physics][queries][shapecast]")
{
    // Build a world with a TileGrid that has a solid run at column 5, row 2.
    // The merged tile span for that cell is the AABB of cell (5,2):
    //   x in [5, 6), y in [2, 3)  (cellSize=1, origin=0).
    // Sweep a circle from the left toward the span; assert it hits (exercises
    // the ShapeCastPoly-against-tile-span path) and that the hit normal points
    // back toward the mover (i.e. away from the span, in the -x direction).
    GridPassability grid(kGridW, kGridH);
    grid.SetSolid(5, 2, true);

    PhysicsWorld w(MakeWorldDef(grid));

    const Real   circleR = Real(0.4);
    const Shape  circle  = MakeCircle(circleR);
    // Start left of the span, sweeping right along row-2 center (y=2.5).
    const Vec2   start(Real(1), Real(2.5));
    const Vec2   delta(Real(10), Real(0)); // sweeps from x=1 to x=11

    const auto cast = w.ShapeCast(circle, start, delta);
    REQUIRE(cast.has_value());
    // A tile span hit has body == kInvalidBody.
    REQUIRE(cast->body == kInvalidBody);
    // The circle must stop BEFORE the span's near face (x=5): its center
    // lands approximately at x = 5 - circleR (a radius short of the wall).
    REQUIRE(cast->point.x < Real(5));
    // margin ~8x kShapeCastTol (1.25*kLinearSlop) (MKS P4)
    REQUIRE(cast->point.x == Approx(Real(5) - circleR).margin(Real(0.05)));
    // The cast terminates at t < 1 (did not pass through).
    REQUIRE(cast->t > Real(0));
    REQUIRE(cast->t < Real(1));
    // Push-back normal faces back toward the mover (negative x direction).
    REQUIRE(cast->normal.x < Real(0));
    // Near-touching at impact: surface distance is small.
    // margin ~3.2x kShapeCastTol (1.25*kLinearSlop) (MKS P4)
    REQUIRE(cast->distance < Real(0.02));
}
