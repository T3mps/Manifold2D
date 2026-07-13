#pragma once

// JointMath: the shared velocity-constraint helpers for the joint set (M6,
// Task P2.5).
//
// PORT NOTE: a faithful port of the three free helpers at the top of
// Client/src/physics/Joints.lua (lines 11-42): `vat` (velocity at an offset
// point), `applyAt` (apply a linear impulse at an offset, with the invMass==0
// and invInertia!=0 static special-cases), and `solvePoint` (the 2x2 effective-
// mass solve for a point constraint). The Lua reached into the world SoA arrays
// directly (w.velX[i] / w.invMass[i] / ...); the C++ joints route through the
// PhysicsWorld slot accessors (VelSlot/AngVelSlot/InvMassSlot/InvInertiaSlot +
// SetVelSlot/SetAngVelSlot) so the raw vectors stay private. Behavior is
// identical: a slot index of kInvalidSlot is the Lua `i < 0` (static anchor,
// no DOF) sentinel; the math no-ops on it exactly as the Lua did.
//
// The Lua used i < 0 to mark "no body" (a static anchor with no SoA slot). The
// C++ joints store body slots and use kInvalidSlot for the same purpose (mouse
// has no body A). Helpers take a `bool valid` companion so a kInvalidSlot slot
// contributes zero inverse mass/inertia + receives no impulse, matching the Lua
// `if i < 0 then return ...`.
//
// PRESENTATION-FREE + C++20-clean: Geometry::Vec2 + std + sibling Physics headers only.
// namespace Manifold2D::Physics, Core style.

#include <cmath>
#include <cstdint>

#include <Manifold2D/Physics/PhysicsTypes.hpp>
#include <Manifold2D/Physics/PhysicsWorld.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        namespace JointMath
        {
            // True iff `i` names a real SoA slot (the Lua `i >= 0`). kInvalidSlot
            // is the static-anchor sentinel (mouse's missing body A); it has no
            // velocity/mass.
            [[nodiscard]] inline bool Valid(std::uint32_t i) noexcept
            {
                return i != kInvalidSlot;
            }

            // Inverse mass of slot i (0 for kInvalidSlot / static). Ports the
            // Lua `w.invMass[i] or 0`.
            [[nodiscard]] inline Real InvMass(const PhysicsWorld& w, std::uint32_t i) noexcept
            {
                return Valid(i) ? w.InvMassSlot(i) : Real(0);
            }

            // Inverse rotational inertia of slot i (0 for kInvalidSlot / static).
            [[nodiscard]] inline Real InvInertia(const PhysicsWorld& w, std::uint32_t i) noexcept
            {
                return Valid(i) ? w.InvInertiaSlot(i) : Real(0);
            }

            // Velocity at an offset point (rx, ry) from body i's center:
            //   vx = velX - angVel * ry ; vy = velY + angVel * rx
            // Ports Joints.lua `vat` (lines 11-14). kInvalidSlot -> (0, 0).
            [[nodiscard]] inline Vec2
            Vat(const PhysicsWorld& w, std::uint32_t i, Real rx, Real ry) noexcept
            {
                if (!Valid(i))
                {
                    return Vec2(Real(0), Real(0));
                }
                const Vec2 v   = w.VelSlot(i);
                const Real av  = w.AngVelSlot(i);
                return Vec2(v.x - av * ry, v.y + av * rx);
            }

            // Apply a linear impulse (jx, jy) to body i at offset (rx, ry).
            // Ports Joints.lua `applyAt` (lines 16-26), faithfully including the
            // two static special-cases:
            //   * i < 0 OR invMass == 0: skip the linear push; but if the body
            //     still has rotational DOF (invInertia != 0) apply the angular
            //     part. (This handles a fixed-position-but-spinnable body.)
            //   * otherwise: full linear + angular response.
            inline void ApplyAt(PhysicsWorld& w, std::uint32_t i,
                                 Real jx, Real jy, Real rx, Real ry) noexcept
            {
                if (!Valid(i))
                {
                    return;
                }
                const Real iM = w.InvMassSlot(i);
                const Real iI = w.InvInertiaSlot(i);
                if (iM == Real(0))
                {
                    if (iI != Real(0))
                    {
                        w.SetAngVelSlot(i, w.AngVelSlot(i) + (rx * jy - ry * jx) * iI);
                    }
                    return;
                }
                Vec2 v = w.VelSlot(i);
                v.x += jx * iM;
                v.y += jy * iM;
                w.SetVelSlot(i, v);
                w.SetAngVelSlot(i, w.AngVelSlot(i) + (rx * jy - ry * jx) * iI);
            }

            // Apply an angular-only impulse j to body i (no lever arm). Used by
            // the weld/prismatic angle lock + the motor. No linear effect.
            inline void ApplyAngular(PhysicsWorld& w, std::uint32_t i, Real j) noexcept
            {
                if (!Valid(i))
                {
                    return;
                }
                const Real iI = w.InvInertiaSlot(i);
                if (iI != Real(0))
                {
                    w.SetAngVelSlot(i, w.AngVelSlot(i) + j * iI);
                }
            }

            // 2x2 effective-mass solve for a point constraint between slots ia
            // and ib with arms (rax,ray) / (rbx,rby): returns the impulse that
            // realizes the desired relative velocity change (dvx, dvy). Ports
            // Joints.lua `solvePoint` (lines 30-42). A near-singular K returns 0.
            [[nodiscard]] inline Vec2
            SolvePoint(const PhysicsWorld& w, std::uint32_t ia, std::uint32_t ib,
                       Real rax, Real ray, Real rbx, Real rby,
                       Real dvx, Real dvy) noexcept
            {
                const Real ma  = InvMass(w, ia);
                const Real iaI = InvInertia(w, ia);
                const Real mb  = InvMass(w, ib);
                const Real ibI = InvInertia(w, ib);

                const Real k11 = ma + mb + iaI * ray * ray + ibI * rby * rby;
                const Real k12 = -iaI * rax * ray - ibI * rbx * rby;
                const Real k22 = ma + mb + iaI * rax * rax + ibI * rbx * rbx;
                const Real det = k11 * k22 - k12 * k12;
                if (std::fabs(det) < Real(1e-12))
                {
                    return Vec2(Real(0), Real(0));
                }
                const Real inv = Real(1) / det;
                return Vec2((k22 * dvx - k12 * dvy) * inv,
                            (k11 * dvy - k12 * dvx) * inv);
            }

            // Rotate a local point (lx, ly) by body i's current angle. Ports
            // Joints.lua `rotated` (lines 73-76).
            [[nodiscard]] inline Vec2
            Rotated(const PhysicsWorld& w, std::uint32_t i, Real lx, Real ly) noexcept
            {
                const Real a = Valid(i) ? w.GetAngle(w.HandleOf(i)) : Real(0);
                const Real c = std::cos(a);
                const Real s = std::sin(a);
                return Vec2(lx * c - ly * s, lx * s + ly * c);
            }

            // Body i's current angle (0 for kInvalidSlot / static-anchor).
            [[nodiscard]] inline Real Angle(const PhysicsWorld& w, std::uint32_t i) noexcept
            {
                return Valid(i) ? w.GetAngle(w.HandleOf(i)) : Real(0);
            }

            // Body i's current center position (origin for kInvalidSlot).
            [[nodiscard]] inline Vec2 Pos(const PhysicsWorld& w, std::uint32_t i) noexcept
            {
                return Valid(i) ? w.PosSlot(i) : Vec2(Real(0), Real(0));
            }

            // Body i's current angular velocity (0 for kInvalidSlot).
            [[nodiscard]] inline Real AngVel(const PhysicsWorld& w, std::uint32_t i) noexcept
            {
                return Valid(i) ? w.AngVelSlot(i) : Real(0);
            }

        } // namespace JointMath
    } // namespace Physics
} // namespace Manifold2D
