#pragma once
// Test-only smoke helper: sums an array through the Manifold2D::Simd wide-float
// wrapper. Lives in the test target, NOT the shipping library -- its sole
// purpose is to exercise the Simd wrapper in a real (non-inline) translation
// unit; no production physics/geometry code uses it. Defined in
// tests/Support/SimdSmoke.cpp.
namespace Manifold2D { namespace Simd { float SimdSmokeSum(const float* p, int count) noexcept; } }
