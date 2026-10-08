#pragma once

// Events.hpp: per-step contact / hit / sensor event records (Box2D v3
// b2ContactEvents / b2SensorEvents shape, spec 2026-10-08 s6). Arrays are owned
// by PhysicsWorld and valid until the next Step. Every record carries BOTH
// fixture handles and BOTH body handles (amendment A5: a destroy-time End names
// a fixture that no longer exists).
//
// PRESENTATION-FREE, ASCII comments, C++23.

#include <cstdint>
#include <span>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Fixture.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        // Contact::eventFlags bits, fixed when the pool contact is created
        // (Box2D contact.c:253-256, :535-541).
        inline constexpr std::uint8_t kEvContact = 1u;
        inline constexpr std::uint8_t kEvHit     = 2u;

        struct ContactBeginEvent { FixtureHandle a{}, b{}; BodyHandle bodyA{}, bodyB{}; };
        struct ContactEndEvent   { FixtureHandle a{}, b{}; BodyHandle bodyA{}, bodyB{}; };
        // normal points from A to B (Box2D convention; Manifold2D's manifold
        // normal points B -> A, Manifold.hpp:33, so the producer negates it).
        struct ContactHitEvent   { FixtureHandle a{}, b{}; BodyHandle bodyA{}, bodyB{};
                                   Vec2 point{}; Vec2 normal{}; Real approachSpeed = Real(0); };
        struct SensorBeginEvent  { FixtureHandle sensor{}, visitor{}; BodyHandle sensorBody{}, visitorBody{}; };
        struct SensorEndEvent    { FixtureHandle sensor{}, visitor{}; BodyHandle sensorBody{}, visitorBody{}; };

        struct ContactEvents
        {
            std::span<const ContactBeginEvent> begin;
            std::span<const ContactEndEvent>   end;
            std::span<const ContactHitEvent>   hit;
        };
        struct SensorEvents
        {
            std::span<const SensorBeginEvent> begin;
            std::span<const SensorEndEvent>   end;
        };

        // One touching solver contact of a body (b2Body_GetContactData). normal
        // points from `self` outward to `other`.
        struct BodyContact
        {
            FixtureHandle self{}, other{};
            BodyHandle    selfBody{}, otherBody{};
            Vec2          normal{};
            int           pointCount = 0;
        };

        // Deterministic order for event arrays: (index, generation) lexicographic.
        [[nodiscard]] constexpr bool FixtureLess(FixtureHandle l, FixtureHandle r) noexcept
        {
            return l.index != r.index ? l.index < r.index : l.generation < r.generation;
        }
    } // namespace Physics
} // namespace Manifold2D
