#pragma once

// Fixture.hpp -- Box2D-style fixture layer for the Manifold2D 2D physics engine
// (Physics v2, Phase A, Task 4).
//
// A Fixture is one convex collider attached to a Body. A body owns 1..N
// fixtures; each fixture carries its own Shape, local transform, material
// (density / friction / restitution), collision filter (categoryBits /
// maskBits), and isSensor flag.
//
// ADDITIVE (v2 Task 4): the Fixture layer is introduced alongside the
// existing single-shape AddBody path, which is now implemented as a
// convenience that creates one fixture from the BodyDef's shape + material.
// Task 5 wires fixture-pair contact generation; this task is data-model only.
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.
// No SDL3/NVRHI/Batcher2D/ImGui. Compiles both /MD (Arcane.dll) and
// pure-compute (Arcane-only, /MD).

#include <cstdint>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        // ----------------------------------------------------------------
        // FixtureHandle: packed {index, generation} over a free-list.
        //
        // Mirrors BodyHandle exactly (same SoA-slot + generation discipline).
        // A stale handle (bumped generation after DropFixture) is always
        // distinguishable from any live handle.
        // ----------------------------------------------------------------
        struct FixtureHandle
        {
            std::uint32_t index      = 0;
            std::uint32_t generation = 0;

            friend constexpr bool operator==(const FixtureHandle& a,
                                              const FixtureHandle& b) noexcept
            {
                return a.index == b.index && a.generation == b.generation;
            }
            friend constexpr bool operator!=(const FixtureHandle& a,
                                              const FixtureHandle& b) noexcept
            {
                return !(a == b);
            }
        };

        // Null / invalid fixture sentinel: index 0, generation 0. Live
        // fixture slots start their generation at 1.
        inline constexpr FixtureHandle kInvalidFixture{ 0u, 0u };

        // ----------------------------------------------------------------
        // FixtureDef: parameters for AddFixture (Box2D-style).
        // ----------------------------------------------------------------
        struct FixtureDef
        {
            // Collider geometry (the unified core+radius shape from T1).
            // Required; must be set before calling AddFixture.
            Shape shape{};

            // Local transform of this fixture in the body frame.
            //   worldPos   = bodyPos + R(bodyAngle) * localPos
            //   worldAngle = bodyAngle + localAngle
            Vec2 localPos  { Real(0), Real(0) };
            Real localAngle = Real(0);

            // Material: mass density (mass = density * area), friction
            // coefficient (combined friction = sqrt(fA * fB) at contacts),
            // and restitution (bounce coefficient; combined = max(rA, rB)).
            Real density     = Real(1);
            Real friction    = Real(0.4);
            Real restitution = Real(0);

            // Collision filter. A contact is generated between two fixtures
            // A and B iff (A.category & B.mask) != 0 AND (B.category & A.mask) != 0.
            // Default: every fixture is in category 1 and collides with everything.
            std::uint32_t categoryBits = 1u;
            std::uint32_t maskBits     = 0xFFFFFFFFu;

            // Sensor fixtures detect overlap events but do NOT generate
            // contact constraints (no collision response). Mirrors Box2D.
            bool isSensor = false;
        };

    } // namespace Physics
} // namespace Manifold2D
