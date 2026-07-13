// Queries.cpp -- the PhysicsWorld spatial-query surface (same class, separate
// translation unit). Split out of PhysicsWorld.cpp in P2.1 (the orchestration
// TU was ~900 lines and the dynamics stages would push it past readable; the
// queries are a self-contained cluster -- Raycast / RayVsBody / LineOfSight /
// ShapeCast / OverlapShape / StaticCandidates -- so they move here verbatim,
// NO API change).
//
// PORTS the query path of Client/src/physics/PhysicsWorld.lua (raycast,
// shapeCast, lineOfSight, _staticCandidates) + the composed OverlapShape (no
// direct Lua method). The cell DDA is REFORMULATED from iso to plain Cartesian
// against the world's TileGrid lattice; the body slab + GJK point-cast tests
// are coordinate-agnostic and port faithfully. See PhysicsWorld.hpp for the
// per-method contract.
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.

#include <Manifold2D/Physics/PhysicsWorld.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include <Manifold2D/Physics/Narrowphase/GeometryKernel.hpp>
#include <Manifold2D/Physics/Narrowphase/Gjk.hpp>
#include <Manifold2D/Physics/Narrowphase/Collide.hpp>   // T7: unified rotation-aware
                                                     // narrowphase (OverlapShape).

namespace Manifold2D
{
    namespace Physics
    {
        // --------------------------------------------------------------------
        // Raycast -- PORT of PhysicsWorld.lua raycast (502-593). The cell DDA is
        // REFORMULATED from iso (Iso.toCellF) to plain CARTESIAN using the
        // world's TileGrid lattice (cellSize/origin). The body slab + GJK
        // point-cast tests are coordinate-agnostic and port faithfully.
        // --------------------------------------------------------------------
        std::optional<RaycastHit> PhysicsWorld::Raycast(const Vec2& from,
                                                        const Vec2& to,
                                                        const RaycastOpts& opts) const
        {
            const Real x0 = from.x, y0 = from.y;
            const Real x1 = to.x,   y1 = to.y;
            const Real dx = x1 - x0, dy = y1 - y0;

            // Degenerate (zero-length) ray -> no hit (ports `if not x1` + the
            // len<eps guards; here the ray has no direction so nothing to test).
            if (dx == Real(0) && dy == Real(0))
            {
                return std::nullopt;
            }

            bool       haveBest = false;
            RaycastHit best{};

            // ---- cell DDA (Cartesian reformulation of the iso lattice) ------
            //
            // The Lua used Iso.toCellF (centered-on-integer iso cells) to map
            // world->fractional-cell. The C++ engine REFORMULATES to TileGrid's
            // plain Cartesian, MIN-CORNER cell mapping so the DDA stays
            // internally consistent with IsSolid / TileGrid::CellBox (cell C
            // spans world [origin + C*cellSize, origin + (C+1)*cellSize), cell
            // boundaries at INTEGER fractional-cell values, not integer +/- 0.5).
            // fractional cell = (world - origin) / cellSize; integer cell =
            // floor(f); boundaries at integers. This is the standard
            // Amanatides-Woo DDA. The Lua's centered +/-0.5 boundary offsets drop
            // out (they were an iso artifact); the traversal structure
            // (tMax/tDelta, step, bounded budget, blocked-on-visit) ports
            // faithfully. The reported cellX/cellY match the cell IsSolid /
            // CellBox index exactly.
            if (m_tileGrid)
            {
                const IPassabilitySource& src   = m_tileGrid->Source();
                const Real                cell   = m_tileGrid->CellSize();
                const Vec2                origin = m_tileGrid->Origin();
                const Real                invCell =
                    cell != Real(0) ? Real(1) / cell : Real(0);

                // World -> fractional cell (min-corner Cartesian lattice).
                const Real fx0 = (x0 - origin.x) * invCell;
                const Real fy0 = (y0 - origin.y) * invCell;
                const Real fx1 = (x1 - origin.x) * invCell;
                const Real fy1 = (y1 - origin.y) * invCell;
                const Real cdx = fx1 - fx0;
                const Real cdy = fy1 - fy0;

                int cx = static_cast<int>(std::floor(fx0));
                int cy = static_cast<int>(std::floor(fy0));
                const int endCx = static_cast<int>(std::floor(fx1));
                const int endCy = static_cast<int>(std::floor(fy1));

                const int stepX = cdx > Real(0) ? 1 : -1;
                const int stepY = cdy > Real(0) ? 1 : -1;
                const Real inf = std::numeric_limits<Real>::infinity();
                // First cell boundary crossing: for +step the next boundary is at
                // (cx+1); for -step it is at cx. Distance in t-space, then
                // tDelta = |1/cd| per full cell. (Standard Amanatides-Woo.)
                Real tMaxX = inf;
                if (cdx != Real(0))
                {
                    const Real nextX = (stepX > 0)
                        ? (static_cast<Real>(cx) + Real(1))
                        : static_cast<Real>(cx);
                    tMaxX = (nextX - fx0) / cdx;
                }
                Real tMaxY = inf;
                if (cdy != Real(0))
                {
                    const Real nextY = (stepY > 0)
                        ? (static_cast<Real>(cy) + Real(1))
                        : static_cast<Real>(cy);
                    tMaxY = (nextY - fy0) / cdy;
                }
                const Real tDeltaX = cdx != Real(0) ? std::abs(Real(1) / cdx) : inf;
                const Real tDeltaY = cdy != Real(0) ? std::abs(Real(1) / cdy) : inf;

                Real t = Real(0);
                // (W+H)*2 budget: a diagonal visits at most W+H cells; a ray
                // starting >W+H cells OUTSIDE the grid can exhaust the budget
                // before reaching a solid cell (faithful to the Lua; fine for
                // on-map casters).
                const int budget = (src.Width() + src.Height()) * 2; // bounded
                for (int it = 0; it < budget; ++it)
                {
                    const bool blocked = opts.tallOnly ? src.BlocksSight(cx, cy)
                                                       : src.IsSolid(cx, cy);
                    if (blocked)
                    {
                        best.t      = t;
                        best.point  = Vec2(x0 + dx * t, y0 + dy * t);
                        best.isCell = true;
                        best.cellX  = cx;
                        best.cellY  = cy;
                        best.body   = kInvalidBody;
                        haveBest    = true;
                        break;
                    }
                    if (cx == endCx && cy == endCy)
                    {
                        break;
                    }
                    if (tMaxX < tMaxY)
                    {
                        t = tMaxX;
                        tMaxX += tDeltaX;
                        cx += stepX;
                    }
                    else
                    {
                        t = tMaxY;
                        tMaxY += tDeltaY;
                        cy += stepY;
                    }
                    if (t > Real(1))
                    {
                        break;
                    }
                }
            }

            // ---- body tests (coordinate-agnostic) ---------------------------
            if (opts.cellsOnly)
            {
                return haveBest ? std::optional<RaycastHit>(best) : std::nullopt;
            }

            for (std::uint32_t i = 0; i < m_count; ++i)
            {
                if (m_alive[i] == 0)
                {
                    continue;
                }
                const Shape& s = m_shape[i];
                if (s.kind == ShapeKind::Aabb)
                {
                    // Exact slab test (ports lines 554-582).
                    const Aabb2 box = SlotAabb(i);
                    const Real a0 = box.min.x, b0 = box.min.y;
                    const Real a1 = box.max.x, b1 = box.max.y;
                    Real tmin = Real(0), tmax = Real(1);
                    bool ok = true;
                    if (dx == Real(0))
                    {
                        if (x0 < a0 || x0 > a1) ok = false;
                    }
                    else
                    {
                        Real ta = (a0 - x0) / dx, tb = (a1 - x0) / dx;
                        if (ta > tb) std::swap(ta, tb);
                        if (ta > tmin) tmin = ta;
                        if (tb < tmax) tmax = tb;
                    }
                    if (ok)
                    {
                        if (dy == Real(0))
                        {
                            if (y0 < b0 || y0 > b1) ok = false;
                        }
                        else
                        {
                            Real ta = (b0 - y0) / dy, tb = (b1 - y0) / dy;
                            if (ta > tb) std::swap(ta, tb);
                            if (ta > tmin) tmin = ta;
                            if (tb < tmax) tmax = tb;
                        }
                    }
                    if (ok && tmin <= tmax && tmin <= Real(1) && tmax >= Real(0))
                    {
                        const Real t = std::max(Real(0), tmin);
                        if (!haveBest || t < best.t)
                        {
                            best.t      = t;
                            best.point  = Vec2(x0 + dx * t, y0 + dy * t);
                            best.isCell = false;
                            best.body   = BodyHandle{ i, m_gen[i] };
                            haveBest    = true;
                        }
                    }
                }
                else
                {
                    // Exact shape hit via GJK point-cast (rayVsBody): a ray is a
                    // zero-radius circle cast against this body's shape.
                    const std::optional<Real> t = RayVsBody(i, from, Vec2(dx, dy));
                    if (t && (!haveBest || *t < best.t))
                    {
                        best.t      = *t;
                        best.point  = Vec2(x0 + dx * *t, y0 + dy * *t);
                        best.isCell = false;
                        best.body   = BodyHandle{ i, m_gen[i] };
                        haveBest    = true;
                    }
                }
            }

            return haveBest ? std::optional<RaycastHit>(best) : std::nullopt;
        }

        // --------------------------------------------------------------------
        // RayVsBody -- PORT of Cast.rayVsBody. A ray (point) = zero-radius
        // circle cast against ONE body's shape via the GJK conservative-
        // advancement primitive. Returns the parametric t or nullopt.
        // --------------------------------------------------------------------
        std::optional<Real> PhysicsWorld::RayVsBody(std::uint32_t idx,
                                                    const Vec2& from,
                                                    const Vec2& delta) const
        {
            static const Shape ray = MakeCircle(Real(0)); // zero-radius circle
            const Transform start{ from, Real(0) };

            // T7 Part B (ROTATION + FIXTURE AWARE): cast the ray vs EACH of the
            // body's fixtures at its composed world transform (real body angle +
            // fixture local), keeping the NEAREST t. The free GJK ShapeCast is now
            // rotation-aware (Part 0: BuildCore honors xf.rotation). A body with
            // no fixtures falls back to its legacy single shape at the real angle.
            const Real bodyAngle = m_angle[idx];
            bool   have = false;
            Real   bestT = Real(0);
            auto consider = [&](const Shape& s, const Transform& xf)
            {
                // Qualified: the free GJK ShapeCast, not PhysicsWorld::ShapeCast.
                const ShapeCastResult r =
                    ::Manifold2D::Physics::ShapeCast(ray, start, delta, s, xf);
                if (r.hit && (!have || r.t < bestT))
                {
                    bestT = r.t;
                    have  = true;
                }
            };

            if (idx < m_bodyFixtures.size() && !m_bodyFixtures[idx].empty())
            {
                const Vec2 bodyPos(m_posX[idx], m_posY[idx]);
                for (const std::uint32_t fi : m_bodyFixtures[idx])
                {
                    if (fi >= m_fxCount || m_fxGen[fi] == 0u)
                    {
                        continue;
                    }
                    const Transform xf = ComposeFixtureXf(
                        bodyPos, bodyAngle,
                        Vec2(m_fxLocalPosX[fi], m_fxLocalPosY[fi]),
                        m_fxLocalAngle[fi]);
                    consider(m_fxShape[fi], xf);
                }
            }
            else
            {
                const Transform bodyXf{ Vec2(m_posX[idx], m_posY[idx]), bodyAngle };
                consider(m_shape[idx], bodyXf);
            }

            return have ? std::optional<Real>(bestT) : std::nullopt;
        }

        // --------------------------------------------------------------------
        // LineOfSight -- PORT of lineOfSight (601-604). Only TALL (sight-
        // blocking) cells block; LOW obstacles are see-through.
        // --------------------------------------------------------------------
        bool PhysicsWorld::LineOfSight(const Vec2& from, const Vec2& to) const
        {
            // Degenerate -> no LOS (ports `if not x1 then return false`).
            if (from.x == to.x && from.y == to.y)
            {
                return false;
            }
            RaycastOpts opts;
            opts.tallOnly  = true;
            opts.cellsOnly = true;
            return !Raycast(from, to, opts).has_value();
        }

        // --------------------------------------------------------------------
        // ShapeCast -- PORT of Cast.shapeCast. Swept-AABB candidate collection
        // (tile spans + non-sensor statics + optional movers) cast against each
        // via the P1.4 GJK conservative-advancement primitive (ShapeCastPoly for
        // spans, ShapeCast for body shapes). Keeps the NEAREST t. The hit point
        // is the shape CENTER at TOI.
        //
        // T7: rotation + fixture aware. `movingAngle` orients the swept shape;
        // each obstacle body is cast against EVERY one of its fixtures at the
        // composed world transform (fixtureless bodies fall back to the single
        // shape at the real body angle). SENSOR GATING IS BODY-LEVEL ONLY
        // (m_sensor[idx]); per-fixture m_fxSensor is intentionally NOT consulted
        // by this query this milestone (consistent with the M6 body-level sensor
        // model) -- a fixture on a non-sensor body still blocks the cast.
        // --------------------------------------------------------------------
        std::optional<ShapeCastHit> PhysicsWorld::ShapeCast(
            const Shape& shape, const Vec2& pos, const Vec2& delta,
            const ShapeCastOpts& opts, Real movingAngle) const
        {
            const Real len =
                std::sqrt(delta.x * delta.x + delta.y * delta.y);
            if (len < Real(1e-9))
            {
                return std::nullopt; // no travel (ports the Cast.lua guard)
            }

            // Swept AABB for candidate collection. Pad = kShapeCastTol -- the
            // cast's own hit criterion (MKS P1.iii; was a px-tuned +/-2 ported
            // from Cast.lua lines 40-43). Sufficient by construction: a hit
            // requires surface distance < kShapeCastTol, so pad >= kShapeCastTol
            // guarantees candidate culling cannot drop a reachable obstacle.
            //
            // T7 Part B/C: the moving shape is carried at `movingAngle` (default 0
            // keeps every existing caller byte-identical). ComputeAABB is
            // rotation-aware, so the swept-AABB candidate box already covers the
            // rotated extent; the conservative-advancement (free GJK ShapeCast)
            // holds movingAngle fixed during the sweep (translational CCD model).
            const Transform xf0{ pos, movingAngle };
            const Transform xf1{ Vec2(pos.x + delta.x, pos.y + delta.y), movingAngle };
            const Aabb2 a = shape.ComputeAABB(xf0);
            const Aabb2 b = shape.ComputeAABB(xf1);
            Aabb2 swept;
            swept.min = Vec2(std::min(a.min.x, b.min.x) - kShapeCastTol,
                             std::min(a.min.y, b.min.y) - kShapeCastTol);
            swept.max = Vec2(std::max(a.max.x, b.max.x) + kShapeCastTol,
                             std::max(a.max.y, b.max.y) + kShapeCastTol);

            StaticCandidates(swept, m_scratchSpans, m_scratchStatics, m_staticGridScratch);

            bool         haveBest = false;
            ShapeCastHit best{};
            auto consider = [&](const ShapeCastResult& r, BodyHandle body)
            {
                if (r.hit && (!haveBest || r.t < best.t))
                {
                    best.t        = r.t;
                    best.normal   = r.normal;
                    best.distance = r.distance;
                    best.body     = body;
                    haveBest      = true;
                }
            };

            // Tile spans (raw world-space AABB polys -> ShapeCastPoly). The span
            // is an axis-aligned box; AabbToCorners expands to the canonical
            // corner order shared with Depenetrate and WorldPoly (no drift).
            for (std::size_t i = 0; i < m_scratchSpans.size(); ++i)
            {
                Vec2 poly[4];
                AabbToCorners(m_scratchSpans[i], poly);
                consider(ShapeCastPoly(shape, xf0, delta, poly, 4), kInvalidBody);
            }

            // T7 Part B (ROTATION + FIXTURE AWARE): cast the moving shape vs each
            // of an obstacle body's fixtures at its composed world transform (real
            // body angle + fixture local), keeping the nearest hit per body. A
            // body with no fixtures falls back to its legacy single shape at the
            // real angle. The free GJK ShapeCast is rotation-aware via Part 0.
            auto castVsBody = [&](std::uint32_t idx)
            {
                const BodyHandle handle{ idx, m_gen[idx] };
                const Real bodyAngle = m_angle[idx];
                if (idx < m_bodyFixtures.size() && !m_bodyFixtures[idx].empty())
                {
                    const Vec2 bodyPos(m_posX[idx], m_posY[idx]);
                    for (const std::uint32_t fi : m_bodyFixtures[idx])
                    {
                        if (fi >= m_fxCount || m_fxGen[fi] == 0u)
                        {
                            continue;
                        }
                        const Transform xf = ComposeFixtureXf(
                            bodyPos, bodyAngle,
                            Vec2(m_fxLocalPosX[fi], m_fxLocalPosY[fi]),
                            m_fxLocalAngle[fi]);
                        consider(::Manifold2D::Physics::ShapeCast(shape, xf0, delta,
                                                              m_fxShape[fi], xf),
                                 handle);
                    }
                }
                else
                {
                    const Transform bodyXf{ Vec2(m_posX[idx], m_posY[idx]),
                                            bodyAngle };
                    consider(::Manifold2D::Physics::ShapeCast(shape, xf0, delta,
                                                          m_shape[idx], bodyXf),
                             handle);
                }
            };

            // Non-sensor static bodies.
            for (std::size_t i = 0; i < m_scratchStatics.size(); ++i)
            {
                const std::uint32_t idx = m_scratchStatics[i];
                if (m_sensor[idx] != 0)
                {
                    continue;
                }
                castVsBody(idx);
            }

            // Optional movers (kinematic/dynamic, non-sensor, not excluded).
            if (opts.movers)
            {
                for (std::uint32_t i = 0; i < m_count; ++i)
                {
                    if (m_alive[i] == 0 ||
                        static_cast<BodyType>(m_btype[i]) == BodyType::Static ||
                        m_sensor[i] != 0)
                    {
                        continue;
                    }
                    if (opts.exclude != kInvalidBody &&
                        opts.exclude.index == i &&
                        opts.exclude.generation == m_gen[i])
                    {
                        continue;
                    }
                    castVsBody(i);
                }
            }

            if (!haveBest)
            {
                return std::nullopt;
            }
            // Position = shape CENTER at TOI (ports best.x/best.y).
            best.point = Vec2(pos.x + delta.x * best.t, pos.y + delta.y * best.t);
            return best;
        }

        // --------------------------------------------------------------------
        // OverlapShape -- NEW (composed; no direct Lua method). Body handles
        // whose shape overlaps the query shape at xf, narrowed by the unified
        // rotation-aware Collide (pointCount > 0). Includes statics + movers;
        // index-ordered.
        //
        // T7 Part B (ROTATION + FIXTURE AWARE): each candidate body is tested by
        // iterating EVERY fixture at its composed world transform (real body angle
        // + fixture local) through Collide; the body is included if ANY fixture
        // overlaps the query shape. A body with no fixtures falls back to its
        // legacy single shape at the real angle. Replaces the rotation-blind
        // single-shape CollideShapes(angle=0) path. Per-fixture m_fxSensor is NOT
        // consulted here (sensor-ness is a body-level concept for queries this
        // milestone; callers filter sensor BODIES via IsSensor()).
        // --------------------------------------------------------------------
        int PhysicsWorld::OverlapShape(const Shape& shape, const Transform& xf,
                                       std::vector<BodyHandle>& out) const
        {
            out.clear();
            // Broadphase candidate pass via the query shape's tight AABB, then a
            // narrowphase overlap test. We do a deterministic INDEX-order linear
            // scan over alive slots (the broadphase QueryAABB returns ids sorted
            // ascending too, but a direct index scan with an AABB pre-filter is
            // simpler, index-ordered, and matches QueryAABB's contract). The AABB
            // pre-filter is the broadphase narrowing predicate.
            const Aabb2 qbox = shape.ComputeAABB(xf);
            for (std::uint32_t i = 0; i < m_count; ++i)
            {
                if (m_alive[i] == 0)
                {
                    continue;
                }
                if (!AabbOverlap(qbox, SlotAabb(i)))
                {
                    continue; // broad reject
                }

                const Real bodyAngle = m_angle[i];
                bool overlaps = false;
                if (i < m_bodyFixtures.size() && !m_bodyFixtures[i].empty())
                {
                    const Vec2 bodyPos(m_posX[i], m_posY[i]);
                    for (const std::uint32_t fi : m_bodyFixtures[i])
                    {
                        if (fi >= m_fxCount || m_fxGen[fi] == 0u)
                        {
                            continue;
                        }
                        const Transform fxXf = ComposeFixtureXf(
                            bodyPos, bodyAngle,
                            Vec2(m_fxLocalPosX[fi], m_fxLocalPosY[fi]),
                            m_fxLocalAngle[fi]);
                        if (Collide(shape, xf, m_fxShape[fi], fxXf,
                                    Real(0)).pointCount > 0)
                        {
                            overlaps = true;
                            break; // ANY fixture overlap includes the body
                        }
                    }
                }
                else
                {
                    const Transform bodyXf{ Vec2(m_posX[i], m_posY[i]), bodyAngle };
                    overlaps = Collide(shape, xf, m_shape[i], bodyXf,
                                       Real(0)).pointCount > 0;
                }

                if (overlaps)
                {
                    out.push_back(BodyHandle{ i, m_gen[i] });
                }
            }
            return static_cast<int>(out.size());
        }

        void PhysicsWorld::StaticCandidates(const Aabb2& box,
                                            std::vector<Aabb2>& spansOut,
                                            std::vector<std::uint32_t>& staticsOut,
                                            std::vector<std::uint32_t>& gridScratch) const
        {
            // Merged tile spans (TileGrid) + overlapping static-body slots.
            // Static-body lookup now goes through m_staticTree (a DynamicTree;
            // sorted slot-id order, ascending) instead of the O(dynamics*statics)
            // m_staticList scan that recomputed each static AABB per query. Ports
            // _staticCandidates. With no TileGrid spansOut is empty.
            spansOut.clear();
            if (m_tileGrid)
            {
                m_tileGrid->Query(box, spansOut);
            }
            staticsOut.clear();
            m_staticTree.QueryAABB(box, gridScratch);
            for (std::uint32_t idx : gridScratch)
                if (m_alive[idx] && AabbOverlap(box, SlotAabb(idx)))
                    staticsOut.push_back(idx);
        }

    } // namespace Physics
} // namespace Manifold2D
