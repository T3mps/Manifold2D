#pragma once

// NarrowphaseKind: which narrowphase path produced a contact manifold.
//
// A pure display/inspection tag (debug-visualization Slice A). It is written
// once per Collide() branch at the point that path resolves a manifold, carried
// onto the Manifold, and copied to the ContactConstraint by GenerateContacts'
// emit. NOTHING in the Step path reads it -- it never changes simulation
// behavior (determinism is unaffected).
//
// `Separated` is the default: no contact was produced (an empty manifold leaves
// this value), so a ContactConstraint that still reads `Separated` means the
// kind was never set by a producing branch.
//
// SPLIT NOTE (debug-viz Slice B, Task 3): this enum lived in NarrowphaseTrace.hpp
// when it was tag-only. Task 3 added the richer NarrowphaseTrace struct (which
// embeds a Manifold) to NarrowphaseTrace.hpp; since Manifold.hpp must include
// the ENUM (Manifold::kind) but NarrowphaseTrace.hpp must include Manifold (the
// trace's `manifold` member), the enum was extracted HERE to break that include
// cycle. Manifold.hpp includes this minimal header; NarrowphaseTrace.hpp
// includes both this header and Manifold.hpp. This is a header-only split (no
// compiled TU), so it needs no premake regen.
//
// PRESENTATION-FREE + C++20-clean: <cstdint> only. Compiles both /MD
// (Arcane.dll) and pure-compute (Arcane-only, /MD).

#include <cstdint>

namespace Manifold2D
{
    namespace Physics
    {
        enum class NarrowphaseKind : std::uint8_t
        {
            Separated = 0,    // no contact (empty manifold; default)
            CircleCircle,     // round path, both cores are points (circles)
            CircleVsPolygon,  // round path, one circle core vs a polygon core
            Capsule,          // round path involving a 2-vert segment (capsule)
            SatPolygon,       // poly-poly SAT reference-face clip
            Epa,              // deep round overlap resolved by EPA
            Mpr,              // deep round overlap resolved by the MPR fallback
        };
    } // namespace Physics
} // namespace Manifold2D
