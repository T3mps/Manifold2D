#pragma once

// Body: an ergonomic view over a {world, handle} pair (M6, Task P1.8).
//
// PORT NOTE: ports the Lua `Body` proxy methods (Client/src/physics/
// PhysicsWorld.lua getPosition/setPosition/setVelocity/getVelocity/
// drawPosition/shape/setEventsEnabled). In the Lua, Body is the ONLY surface
// (the world returns a Body from addBody). In the C++ engine the CANONICAL
// surface is HANDLE-BASED on PhysicsWorld (events carry BodyHandles, so handle
// methods are the natural payload-consuming API); Body is a thin, copyable
// ergonomic wrapper that forwards to those world methods. Pick whichever fits
// the call site.
//
// A Body is cheap (two words: a PhysicsWorld* + a BodyHandle) and copyable. It
// is valid only while world->IsValid(handle); methods on a stale view forward
// to world methods that no-op / return defaults for invalid handles.
//
// DYNAMICS (P2.1): applyImpulse, wake/isAwake, getAngle/setAngle now forward to
// the world's dynamics surface (the SoA carries mass/inertia/angle/awake as of
// P2.1). Dynamic-only effects no-op on Static/Kinematic, matching the world
// methods. (See PhysicsWorld.hpp PORT BOUNDARY.)
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.
// namespace Manifold2D::Physics, Core style.

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/Shapes.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        class Body
        {
        public:
            Body() = default;
            Body(PhysicsWorld* world, BodyHandle handle) noexcept
                : m_world(world), m_handle(handle)
            {
            }

            [[nodiscard]] BodyHandle    Handle() const noexcept { return m_handle; }
            [[nodiscard]] PhysicsWorld* World()  const noexcept { return m_world; }

            // True iff this view still refers to a live body.
            [[nodiscard]] bool IsValid() const noexcept
            {
                return m_world != nullptr && m_world->IsValid(m_handle);
            }

            // ---- ports of the Lua Body methods (forward to the world) -------

            [[nodiscard]] Vec2 GetPosition() const noexcept
            {
                return m_world->Position(m_handle);
            }

            // Teleport: prev snaps too (ports Body:setPosition).
            // Null-guarded: graceful no-op on a default-constructed / post-remove
            // view (matches the read path which routes through world methods that
            // already handle invalid handles).
            void SetPosition(Vec2 p) noexcept
            {
                if (m_world) m_world->SetPosition(m_handle, p);
            }

            [[nodiscard]] Vec2 GetVelocity() const noexcept
            {
                return m_world->Velocity(m_handle);
            }

            // Set velocity (Dynamic wakes; ports Body:setVelocity).
            void SetVelocity(Vec2 v) noexcept
            {
                if (m_world) m_world->SetVelocity(m_handle, v);
            }

            // ---- dynamics (P2.1; Dynamic-only effects) ---------------------

            // Linear impulse at the body center (ports Body:applyImpulse, the
            // no-point branch). Dynamic only; wakes.
            void ApplyImpulse(Vec2 impulse) noexcept
            {
                if (m_world) m_world->ApplyImpulse(m_handle, impulse);
            }

            // Linear impulse at a world point -> linear + angular (ports
            // Body:applyImpulse, the px,py branch). Dynamic only; wakes.
            void ApplyImpulse(Vec2 impulse, Vec2 worldPoint) noexcept
            {
                if (m_world) m_world->ApplyImpulse(m_handle, impulse, worldPoint);
            }

            // Wake a sleeping Dynamic body (ports Body:wake).
            void Wake() noexcept
            {
                if (m_world) m_world->Wake(m_handle);
            }

            [[nodiscard]] bool IsAwake() const noexcept
            {
                return m_world && m_world->IsAwake(m_handle);
            }

            [[nodiscard]] Real GetAngle() const noexcept
            {
                return m_world ? m_world->GetAngle(m_handle) : Real(0);
            }

            void SetAngle(Real angle) noexcept
            {
                if (m_world) m_world->SetAngle(m_handle, angle);
            }

            // Render-boundary lerp prev..curr (ports Body:drawPosition).
            [[nodiscard]] Vec2 DrawPosition(Real alpha) const noexcept
            {
                return m_world->DrawPosition(m_handle, alpha);
            }

            [[nodiscard]] const Shape* GetShape() const noexcept
            {
                return m_world->GetShape(m_handle);
            }

            [[nodiscard]] BodyType GetType() const noexcept
            {
                return m_world->GetType(m_handle);
            }

            [[nodiscard]] bool IsSensor() const noexcept
            {
                return m_world->IsSensor(m_handle);
            }

            // Per-body event gate (ports Body:setEventsEnabled).
            void SetEventsEnabled(bool on) noexcept
            {
                if (m_world) m_world->SetBodyEvents(m_handle, on);
            }

        private:
            PhysicsWorld* m_world  = nullptr;
            BodyHandle    m_handle{};
        };

    } // namespace Physics
} // namespace Manifold2D
