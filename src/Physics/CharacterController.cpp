// CharacterController.cpp -- WASD slide + click-to-move (port of
// CharacterController.lua, the live gameplay layer).
//
// See CharacterController.hpp for the contract + the faithful-substep PORT NOTE
// (the Lua substep+depenetrate algorithm; NOT a true ShapeCast sweep).
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.

#include <Manifold2D/Physics/CharacterController.hpp>

#include <cmath>

#include <Manifold2D/Physics/Narrowphase/GeometryKernel.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        // (BoxCorners removed: use AabbToCorners from GeometryKernel.hpp instead.)

        Vec2 CharacterController::Depenetrate(Vec2 p)
        {
            const Shape* sp = m_world->GetShape(m_body);
            if (sp == nullptr)
            {
                return p; // stale handle -> nothing to resolve against
            }
            const Shape& s = *sp;
            // Circle = zero-length capsule (halfLen 0). Capsule carries its own
            // halfLen + radius. (The Lua: halfLen = s.halfLen or 0; s.r.)
            const Real halfLen = (s.kind == ShapeKind::Capsule) ? s.halfLen : Real(0);
            const Real r       = s.radius;

            for (int pass = 0; pass < kMaxPasses; ++pass)
            {
                // The body's world AABB at (x,y) drives the candidate query.
                const Transform xf{ p, Real(0) };
                const Aabb2 box = s.ComputeAABB(xf);
                m_world->StaticCandidates(box, m_spans, m_statics, m_staticGridScratch);

                bool hit    = false;
                Vec2 bn{ Real(0), Real(0) };
                Real bdepth = Real(0);

                const Vec2 segA(p.x - halfLen, p.y);
                const Vec2 segB(p.x + halfLen, p.y);

                // Tile spans: each merged span rect -> 4-corner poly -> CapsulePoly.
                for (std::size_t i = 0; i < m_spans.size(); ++i)
                {
                    AabbToCorners(m_spans[i], m_polyScratch.data());
                    const Hit h = CapsulePoly(segA, segB, r,
                                              m_polyScratch.data(), 4);
                    if (h.hit && h.depth > bdepth)
                    {
                        hit    = true;
                        bn     = h.normal;
                        bdepth = h.depth;
                    }
                }

                // Static bodies: skip sensors; AABB (-> 4-corner poly) and
                // POLYGON statics only (round statics produce no poly -> skipped,
                // matching the Lua). Circle/Capsule statics are intentionally not
                // resolved against here (the Lua _depenetrate handles only
                // aabb/polygon static kinds).
                for (std::size_t i = 0; i < m_statics.size(); ++i)
                {
                    const std::uint32_t idx = m_statics[i];
                    if (m_world->SensorSlot(idx))
                    {
                        continue;
                    }
                    const Shape& os  = m_world->ShapeSlot(idx);
                    const Vec2   opos = m_world->PosSlot(idx);

                    const Vec2* verts = nullptr;
                    int         n     = 0;
                    if (os.kind == ShapeKind::Aabb)
                    {
                        // Build the static box's corners around its world pos
                        // (the Lua aabbPoly(posX,posY,hw,hh)).
                        const Aabb2 obox{
                            Vec2(opos.x - os.halfW, opos.y - os.halfH),
                            Vec2(opos.x + os.halfW, opos.y + os.halfH)
                        };
                        AabbToCorners(obox, m_polyScratch.data());
                        verts = m_polyScratch.data();
                        n     = 4;
                    }
                    else if (os.kind == ShapeKind::Polygon)
                    {
                        // Static polygon verts are baked in local space; the Lua
                        // statics carry world-space verts because the engine is
                        // fixedRotation and static bodies live at the origin of
                        // their own verts. The C++ Polygon verts are local; offset
                        // by the body world pos to get world verts. (Static bodies
                        // are typically placed with their poly authored about the
                        // body position.)
                        // Note: reserve() is intentionally absent here -- the
                        // clear()+push_back pattern preserves capacity across
                        // calls (no per-pass realloc in steady state).
                        m_polyWorld.clear();
                        for (const Vec2& v : os.verts)
                        {
                            m_polyWorld.push_back(Vec2(opos.x + v.x, opos.y + v.y));
                        }
                        verts = m_polyWorld.data();
                        n     = static_cast<int>(m_polyWorld.size());
                    }
                    else
                    {
                        continue; // round static -> no poly -> skipped (Lua parity)
                    }

                    const Hit h = CapsulePoly(segA, segB, r, verts, n);
                    if (h.hit && h.depth > bdepth)
                    {
                        hit    = true;
                        bn     = h.normal;
                        bdepth = h.depth;
                    }
                }

                if (!hit)
                {
                    return p;
                }
                // Push out along the deepest normal, a hair past the surface.
                p.x += bn.x * (bdepth + kDepenetrationSkin);
                p.y += bn.y * (bdepth + kDepenetrationSkin);
            }
            return p;
        }

        Vec2 CharacterController::SlideMove(Real dx, Real dy)
        {
            // Stale-body guard: explicit early-out before reading position.
            // Matters once P3.3 pools entities (vs the silent zero-displacement
            // that the IsValid-gated Position fallback would otherwise produce).
            if (!m_world->IsValid(m_body)) { return Vec2(Real(0), Real(0)); }

            const Vec2 start = m_world->Position(m_body);
            Vec2       p     = start;

            const Real dist = std::sqrt(dx * dx + dy * dy);
            if (dist < Real(1e-9))
            {
                return Vec2(Real(0), Real(0));
            }
            const Real ux = dx / dist;
            const Real uy = dy / dist;

            Real remaining = dist;
            while (remaining > Real(0))
            {
                const Real stepLen = (remaining < kMaxSubstep) ? remaining : kMaxSubstep;
                remaining -= stepLen;
                p = Depenetrate(Vec2(p.x + ux * stepLen, p.y + uy * stepLen));
            }

            // Write back WITHOUT snapping prev (Step manages prev -- see the
            // integration contract). MovePosition updates pos + the mover
            // broadphase AABB, prev untouched. Velocity stays 0 in WASD mode.
            m_world->MovePosition(m_body, p);
            return Vec2(p.x - start.x, p.y - start.y);
        }

    } // namespace Physics
} // namespace Manifold2D
