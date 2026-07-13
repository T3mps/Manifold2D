#pragma once

// Core value types for the Manifold2D 2D physics engine (M6).
//
// PORT NOTE: this engine is a first-party port of the proven Lua physics
// engine at Client/src/physics/*.lua. The Lua engine + its physics_harness
// are the behavioral oracle. We port named modules; we do not invent
// algorithms. See docs/superpowers/specs/2026-06-14-* for the milestone plan.
//
// PRESENTATION-FREE: this header (and everything under Manifold2D/Physics/) pulls
// in only Geometry::Vec2 + the C++ standard library. No SDL3/NVRHI/Batcher2D/
// ImGui. It must compile both /MD (into Arcane.dll) and static-CRT/C++20 (the
// Arcane-only build).
//
// SCALAR CHOICE (determinism, P1.0 decision): stored physics state uses f32.
// Per-platform self-consistent determinism (fixed 60 Hz, stable iteration
// order, /fp:precise, no fast-math) does NOT require f64. f32 halves the SoA
// footprint. Switching to f64 if a later determinism test demands it is a
// ONE-TYPEDEF change here:
//   using Real = double; using Vec2 = Geometry::Vec2d;
// Geometry::Vec2<T> exposes both widths through the same call sites, so the
// math headers and SoA arrays follow automatically.

#include <cmath>
#include <cstdint>

#include <Manifold2D/Geometry/Vec2.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        // ----------------------------------------------------------------
        // Scalar + vector aliases
        // ----------------------------------------------------------------

        // Stored-state scalar. f32 by default (see SCALAR CHOICE above).
        using Real = float;

        // 2D vector: the first-party Geometry::Vec2 (Manifold2D Phase 1 --
        // glm severed 2026-07-10). If Real switches to double, change this
        // to Geometry::Vec2d in lockstep (the only edit).
        using Vec2 = Geometry::Vec2<float>;

        // ----------------------------------------------------------------
        // Rotation helpers (single source of truth)
        // ----------------------------------------------------------------
        //
        // Standard 2D rotation R(a)*v = (c*v.x - s*v.y, s*v.x + c*v.y),
        // c = cos(a), s = sin(a). This is the canonical definition for the
        // compound-COM math (the origin <-> COM round-trip in the solver and the
        // contact anchor bases) -- the SAME convention as the still-inlined copies
        // in PhysicsWorld::ComposeFixtureXf / GetFixtureWorldPos / Shape::ComputeAABB
        // / the T7 BuildCore (a future cleanup may delegate those onto RotateVec).
        [[nodiscard]] inline Vec2 RotateVec(Real angle, const Vec2& v) noexcept
        {
            const Real c = std::cos(angle);
            const Real s = std::sin(angle);
            return Vec2(c * v.x - s * v.y, s * v.x + c * v.y);
        }

        // World-space center of mass of a body given its ORIGIN pose
        // (pos, angle) and its LOCAL-frame center of mass (localCenter):
        //   com = pos + R(angle) * localCenter.
        // For localCenter == (0,0) (every single-fixture body and all
        // static/kinematic bodies) this reduces to `pos`, so any caller that
        // swaps an origin for WorldCom(...) is byte-identical there.
        [[nodiscard]] inline Vec2 WorldCom(const Vec2& pos, Real angle,
                                           const Vec2& localCenter) noexcept
        {
            return pos + RotateVec(angle, localCenter);
        }

        // ----------------------------------------------------------------
        // Body classification (ported from PhysicsWorld.lua body "type")
        // ----------------------------------------------------------------

        // Static  : never integrates; immovable world geometry.
        // Kinematic: integrated from velocity; pushes dynamics, is never
        //            pushed (spec: kinematics push, are never pushed).
        // Dynamic : full force/impulse-driven body (Phase 2+).
        enum class BodyType : std::uint8_t
        {
            Static    = 0,
            Kinematic = 1,
            Dynamic   = 2,
        };

        // ----------------------------------------------------------------
        // BodyHandle: packed {index, generation} over a free-list.
        // ----------------------------------------------------------------
        //
        // Body/contact state lives in SoA arrays on the world (filled in by
        // later tasks). A handle is NEVER a pointer: it is a slot index plus
        // a generation counter. When a slot is freed and later reused its
        // generation is bumped, so a stale handle (same index, old
        // generation) compares unequal and is rejected by World::IsValid.
        // This is the C++ port of the Lua handle/slot-reuse invariant
        // exercised by the harness ("stale handle stays invalid after slot
        // reuse").
        struct BodyHandle
        {
            std::uint32_t index      = 0;
            std::uint32_t generation = 0;

            friend constexpr bool operator==(const BodyHandle& a,
                                              const BodyHandle& b) noexcept
            {
                return a.index == b.index && a.generation == b.generation;
            }
            friend constexpr bool operator!=(const BodyHandle& a,
                                              const BodyHandle& b) noexcept
            {
                return !(a == b);
            }
        };

        // A null/invalid handle sentinel: index 0, generation 0. Live slots
        // begin their generation at 1 so the default-constructed handle is
        // always distinguishable from any real body.
        inline constexpr BodyHandle kInvalidBody{ 0u, 0u };

        // Sentinel SoA slot index meaning "no body" -- used by the solver's
        // ContactConstraint to flag a tile-span virtual fixture (bodyB has no
        // real SoA slot; its inverse mass/inertia are 0 and it is never written
        // back). Ports the Lua contact-row `b == -1` span marker.
        inline constexpr std::uint32_t kInvalidSlot = 0xFFFFFFFFu;

        // ----------------------------------------------------------------
        // Fixed constants
        // ----------------------------------------------------------------

        // Max vertices for a convex polygon collider. Ported verbatim from
        // shapes.lua MAX_POLY_VERTS (revised 2026-06-10: 8 -> 128 for
        // editor-era props). SAT poly-vs-poly is O(nA*nB) over edge axes, so
        // large polys are fine as occasional statics but not the common
        // mover shape.
        inline constexpr std::uint32_t kMaxPolyVerts = 128u;

        // Pi constant (f64 literal narrowed to Real; shared across physics math).
        inline constexpr Real kPi = Real(3.14159265358979323846);

        // Linear slop: the small overlap the solver tolerates so resting
        // contacts do not jitter (Box2D-style allowed penetration). Used by
        // the position-correction / contact code in later tasks.
        inline constexpr Real kLinearSlop = Real(0.005);

        // Box2D v3 B2_MAX_ROTATION: max rotation per FULL step (quarter turn).
        // constants.h:33 -> 0.25f * B2_PI. Manifold2D units == Box2D units.
        inline constexpr Real kMaxRotation = Real(0.25) * Real(3.14159265358979323846);

        // Box2D v3 B2_SPECULATIVE_DISTANCE (constants.h:38): 4 * linear slop.
        // Collision skin / speculative contact distance, in meters.
        inline constexpr Real kSkin = Real(4) * kLinearSlop; // = 0.02

    } // namespace Physics
} // namespace Manifold2D
