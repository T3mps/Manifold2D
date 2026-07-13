#pragma once

// Contact manifold types: ManifoldPoint + Manifold.
//
// HISTORY: this file once also declared CollidePolygons -- the M6-era
// kind-dispatched poly/AABB narrowphase ported bit-for-bit from Manifold.lua.
// Physics v2 (Phase A, Task T3) replaced that path with the unified
// rotation-aware Narrowphase/Collide.cpp, and T8 strangled the now-dead
// CollidePolygons (+ its Sat/Specialized/Dispatch siblings). What survives here
// are the contact-manifold STRUCTS, which are the lingua franca of the
// narrowphase: Collide(...) RETURNS a Manifold and the Solver consumes one.
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.
// No SDL3/NVRHI/Batcher2D/ImGui, no C++23-only features.

#include <cstdint>

#include <Manifold2D/Physics/Narrowphase/NarrowphaseKind.hpp>
#include <Manifold2D/Physics/PhysicsTypes.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        // ----------------------------------------------------------------
        // ManifoldPoint: one contact point.
        // ----------------------------------------------------------------
        //
        // point      : world-space contact position (a contained vertex).
        // separation : penetration depth. POSITIVE when overlapping (the Lua's
        //              per-point `depth`); NEGATIVE for a speculative contact
        //              (a gap, within [-margin, 0)) when a skin margin is used.
        // normal     : THIS contact's own unit normal, points from B toward A
        //              (push A out of B) -- the Lua contact-row `nx,ny`. Each
        //              point carries its own normal because round-vs-poly
        //              endpoint contacts (e.g. a capsule whose two ends touch
        //              DIFFERENT polygon faces) can have DIFFERENT normals per
        //              point. For poly-poly (single SAT axis) every point's
        //              normal equals Manifold::normal.
        // id          : the Lua `key` analog -- a stable per-pair feature key the
        //              solver uses for warm-starting. We reproduce the Lua's
        //              scheme exactly: id = keyBase + slot, where slot is 1 for
        //              the deepest point and 2 for the second-deepest (so the
        //              first manifold point carries keyBase+1, the second
        //              keyBase+2). This matches manifold.json's `key` field.
        // normalImpulse / tangentImpulse :
        //              WARM-START STATE (Box2D-v3 form). The accumulated
        //              (normal, tangent) impulse the solver converged on LAST
        //              step, carried HERE on the persistent pool Contact's
        //              manifold point so it survives across steps without a
        //              solver-side cache. EmitContactConstraints seeds the
        //              emitted ContactConstraintPoint from these; PhysicsWorld
        //              writes them back after Solve(). Default 0 == cold start
        //              (a brand-new feature, or a transient tile span that has
        //              no persistent home). They ride the feature id: when
        //              UpdateContacts replaces the manifold each step it copies
        //              these forward by matching id (see UpdateContacts), so the
        //              same physical contact keeps its impulse history.
        struct ManifoldPoint
        {
            Vec2          point{ Real(0), Real(0) };
            Real          separation = Real(0);
            Vec2          normal{ Real(0), Real(0) };
            std::uint32_t id         = 0;
            Real          normalImpulse  = Real(0);
            Real          tangentImpulse = Real(0);
        };

        // ----------------------------------------------------------------
        // Manifold: representative normal + up to 2 contact points (each with
        // its own normal).
        // ----------------------------------------------------------------
        //
        // normal     : a REPRESENTATIVE unit contact normal, points from B
        //              toward A (push A out of B). It is the Box2D-v3-style
        //              single-axis representative -- the normal of the DEEPEST
        //              contact point (deterministic tiebreak: on equal
        //              separation the first-emitted / lowest-key point wins).
        //              For poly-poly (one SAT axis) every point shares this
        //              normal, so it is exactly the SAT axis. It is kept for
        //              convenience and for poly-poly consumers; the per-point
        //              ManifoldPoint::normal preserves the Lua's per-contact
        //              fidelity (round endpoints touching different faces) that
        //              the future solver needs.
        // pointCount : 0, 1, or 2.
        // kind       : DISPLAY-ONLY tag of the narrowphase path that produced
        //              this manifold (debug-viz Slice A). Default Separated (no
        //              contact). Set by Collide() at each resolving branch and
        //              copied to the ContactConstraint by GenerateContacts. The
        //              Step path never reads it -- determinism is unaffected.
        struct Manifold
        {
            Vec2            normal{ Real(0), Real(0) };
            int             pointCount = 0;
            ManifoldPoint   points[2]{};
            NarrowphaseKind kind = NarrowphaseKind::Separated;
        };

    } // namespace Physics
} // namespace Manifold2D
