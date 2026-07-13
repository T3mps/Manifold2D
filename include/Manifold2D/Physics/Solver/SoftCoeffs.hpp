#pragma once

// SoftCoeffs.hpp -- shared b2MakeSoft soft-constraint coefficients (M6).
//
// Both the contact solver (SoftStep.cpp) and the joint set (Joints.cpp) need
// b2MakeSoft. Centralising the formula here prevents drift when new soft
// consumers are added (CCD / Astra in P3.x).
//
// FORMULA (Box2D v3 b2MakeSoft):
//   omega = 2 * pi * hertz
//   a1    = 2 * zeta + h * omega
//   a2    = h * omega * a1
//   a3    = 1 / (1 + a2)
//   -> { biasRate = omega/a1, massScale = a2*a3, impulseScale = a3 }
//   hertz <= 0 -> { 0, 1, 0 }  (hard / un-softened constraint)
//
// PRESENTATION-FREE: only PhysicsTypes.hpp (Real, kPi) + <cmath>.

#include <cmath>

#include <Manifold2D/Physics/PhysicsTypes.hpp>

namespace Manifold2D
{
    namespace Physics
    {
        // Soft-constraint coefficients produced by MakeSoft.
        struct SoftCoeffs
        {
            Real biasRate     = Real(0);
            Real massScale    = Real(1);
            Real impulseScale = Real(0);
        };

        // b2MakeSoft(hertz, dampingRatio, h): compute the three soft-constraint
        // scalars for a spring with the given frequency (hertz), damping ratio
        // (zeta), and sub-step size (h). hertz <= 0 returns the hard-constraint
        // identity { 0, 1, 0 }.
        inline SoftCoeffs MakeSoft(Real hertz, Real dampingRatio, Real h) noexcept
        {
            if (hertz <= Real(0) || h <= Real(0))
            {
                return SoftCoeffs{ Real(0), Real(1), Real(0) };
            }
            const Real omega = Real(2) * kPi * hertz;
            const Real a1 = Real(2) * dampingRatio + h * omega;
            const Real a2 = h * omega * a1;
            const Real a3 = Real(1) / (Real(1) + a2);
            return SoftCoeffs{ omega / a1, a2 * a3, a3 };
        }

    } // namespace Physics
} // namespace Manifold2D
