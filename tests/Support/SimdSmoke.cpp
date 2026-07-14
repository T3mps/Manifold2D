// SimdSmokeSum -- a tiny end-to-end smoke over the Manifold2D::Simd wide-float
// wrapper (sums an array via f32w). Exercised by SimdWideTests; compiling it as
// a real test TU also forces the wrapper to instantiate outside header-inline
// use. Test-only -- deliberately kept out of the shipping library (src/).
#include <Manifold2D/Core/Simd.hpp>
#include <Support/SimdSmoke.hpp>

namespace Manifold2D { namespace Simd {

float SimdSmokeSum(const float* p, int count) noexcept
{
    f32w acc = setzero();
    int i = 0;
    for (; i + f32w::width <= count; i += f32w::width)
        acc += loadu(p + i);
    alignas(32) float lanes[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    store(lanes, acc);                        // lanes is alignas(32) -> aligned store valid for width <= 8
    float sum = 0.0f;
    for (int k = 0; k < f32w::width; ++k) sum += lanes[k];
    for (; i < count; ++i) sum += p[i];       // scalar tail
    return sum;
}

} } // namespace Manifold2D::Simd
