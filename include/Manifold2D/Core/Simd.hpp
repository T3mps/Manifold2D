#pragma once

// Manifold2D::Simd -- portable wide-float SIMD abstraction. This is the move
// origin for this type (lift design: docs/superpowers/specs/2026-07-10-manifold2d-phase2-lift-design.md,
// D3) -- the host engine's now-duplicate copy is deleted in Task 3 once
// Physics/Geometry retarget here.
// Original design: docs/superpowers/specs/2026-06-24-arcane-simd-wide-float-design.md.
//
// One value type per concept (f32w / b32w / i32w) with operator overloads + free
// functions, mapped per target to AVX2 (8-wide) / NEON (4-wide) / scalar (1-wide)
// at COMPILE time. Presentation-free, header-only; builds /MD and static-CRT.
//
// Determinism: within a build the lane order + op sequence is fixed (run-twice
// identical). Elementwise ops bit-match a scalar reference (mul_add uses fused
// multiply-add on every backend, incl. std::fma in scalar); rsqrt/recip are
// hardware-estimate approximations (tolerance-checked, not bit-matched).
//
// Op surface (each backend provides): splat/setzero/isplat/iota/iload, load/
// store/loadu/storeu, + - * / unary- (and the compound-assign forms), mul_add/
// mul_sub (the only fused ops under /fp:strict), min/max/abs/sqrt/rsqrt/recip,
// cmp_gt/ge/lt/le/eq -> b32w, select, any/all/none, gather/scatter. `iload`
// builds an i32w from a plain int32_t[width] array (the missing counterpart to
// load's f32w; the contact solver needs per-lane body indices as an i32w for
// gather/scatter). It is an EXACT load (bit-identical, no rounding).
//
// Define ARCANE_SIMD_SCALAR before including to force the scalar backend on any
// target (used by the forced-scalar test TU).
//
// Backend isolation strategy (ODR safety):
//   Each backend lives in its own sub-namespace (Manifold2D::Simd::Scalar,
//   Manifold2D::Simd::Avx2, Manifold2D::Simd::Neon).  Simd.hpp then imports the ACTIVE
//   backend into Manifold2D::Simd via "using namespace" -- exactly one backend is
//   compiled per TU, keeping the types distinct across TUs and eliminating ODR
//   violations when multiple TUs are linked into one binary.
//   To add a new backend: define its ops in Manifold2D::Simd::<Name> inside a new
//   Simd_<Name>.inl, then add a ladder arm below that includes it, re-exports it
//   with "using namespace <Name>", and sets ARCANE_SIMD_NS.

#include <cstdint>

#if defined(_MSC_VER)
    #define ARCANE_SIMD_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    #define ARCANE_SIMD_INLINE inline __attribute__((always_inline))
#else
    #define ARCANE_SIMD_INLINE inline
#endif

// Exactly one backend is included per TU, selected by the ladder below.
// ARCANE_SIMD_INLINE and ARCANE_SIMD_NS are intentionally not #undef'd here
// because the .inl bodies need the inline macro at parse time, and ARCANE_SIMD_NS
// is consumed by the test harness (SimdWideTests.inl) after this header is parsed.
// PRODUCTION code should use Manifold2D::Simd:: directly -- ARCANE_SIMD_NS is a
// test-harness seam that names a specific backend sub-namespace; never use it in
// non-test code.
#if defined(ARCANE_SIMD_SCALAR)
    #include <Manifold2D/Core/Simd_Scalar.inl>
    namespace Manifold2D { namespace Simd { using namespace Scalar; } }
    #define ARCANE_SIMD_NS ::Manifold2D::Simd::Scalar
#elif defined(__AVX2__)
    #include <Manifold2D/Core/Simd_AVX2.inl>
    namespace Manifold2D { namespace Simd { using namespace Avx2; } }
    #define ARCANE_SIMD_NS ::Manifold2D::Simd::Avx2
#elif defined(__ARM_NEON__) || defined(__ARM_NEON)
    #include <Manifold2D/Core/Simd_NEON.inl>
    namespace Manifold2D { namespace Simd { using namespace Neon; } }
    #define ARCANE_SIMD_NS ::Manifold2D::Simd::Neon
#else
    // Fallback: no explicit ARCANE_SIMD_SCALAR, no hardware intrinsics detected.
    #include <Manifold2D/Core/Simd_Scalar.inl>
    namespace Manifold2D { namespace Simd { using namespace Scalar; } }
    #define ARCANE_SIMD_NS ::Manifold2D::Simd::Scalar
#endif

static_assert(Manifold2D::Simd::f32w::width == 1 || Manifold2D::Simd::f32w::width == 4 ||
              Manifold2D::Simd::f32w::width == 8,
              "Manifold2D::Simd::f32w::width must be 1, 4, or 8");
