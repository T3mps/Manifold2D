// 1-wide scalar reference backend (the oracle). Included by Simd.hpp; never
// compiled standalone.
#include <cmath>   // std::fma / std::sqrt in the scalar math ops (Tasks 2-3)

namespace Manifold2D { namespace Simd { namespace Scalar {

inline constexpr const char* kBackendName = "scalar";

struct f32w { float   v; static constexpr int width = 1; };
struct b32w { bool    m; static constexpr int width = 1; };  // m: bool here; lane bitmask in the wide backends
struct i32w { int32_t v; static constexpr int width = 1; };

ARCANE_SIMD_INLINE f32w splat(float x)      noexcept { return f32w{ x }; }
ARCANE_SIMD_INLINE f32w setzero()           noexcept { return f32w{ 0.0f }; }
ARCANE_SIMD_INLINE i32w isplat(int32_t x)   noexcept { return i32w{ x }; }
ARCANE_SIMD_INLINE i32w iota()              noexcept { return i32w{ 0 }; } // lane index 0..width-1
// 1-wide int load (the single lane = *p). Exact -> bit-identical. Mirrors the
// AVX2/NEON iload contract: build the gather/scatter index vector from a plain
// int32_t array. The scalar array need not be aligned (it is a plain deref).
ARCANE_SIMD_INLINE i32w iload(const int32_t* p) noexcept { return i32w{ *p }; }
ARCANE_SIMD_INLINE void istore(int32_t* p, i32w v) noexcept { *p = v.v; }

ARCANE_SIMD_INLINE f32w load(const float* p)    noexcept { return f32w{ *p }; }
ARCANE_SIMD_INLINE void store(float* p, f32w a) noexcept { *p = a.v; }
ARCANE_SIMD_INLINE f32w loadu(const float* p)   noexcept { return f32w{ *p }; }
ARCANE_SIMD_INLINE void storeu(float* p, f32w a) noexcept { *p = a.v; }

// Arithmetic operators
ARCANE_SIMD_INLINE f32w operator+(f32w a, f32w b) noexcept { return f32w{ a.v + b.v }; }
ARCANE_SIMD_INLINE f32w operator-(f32w a, f32w b) noexcept { return f32w{ a.v - b.v }; }
ARCANE_SIMD_INLINE f32w operator*(f32w a, f32w b) noexcept { return f32w{ a.v * b.v }; }
ARCANE_SIMD_INLINE f32w operator/(f32w a, f32w b) noexcept { return f32w{ a.v / b.v }; }
ARCANE_SIMD_INLINE f32w operator-(f32w a)         noexcept { return f32w{ -a.v }; }
ARCANE_SIMD_INLINE f32w& operator+=(f32w& a, f32w b) noexcept { a.v += b.v; return a; }
ARCANE_SIMD_INLINE f32w& operator-=(f32w& a, f32w b) noexcept { a.v -= b.v; return a; }
ARCANE_SIMD_INLINE f32w& operator*=(f32w& a, f32w b) noexcept { a.v *= b.v; return a; }
ARCANE_SIMD_INLINE f32w& operator/=(f32w& a, f32w b) noexcept { a.v /= b.v; return a; }
ARCANE_SIMD_INLINE f32w mul_add(f32w a, f32w b, f32w c) noexcept { return f32w{ std::fma(a.v, b.v,  c.v) }; }
ARCANE_SIMD_INLINE f32w mul_sub(f32w a, f32w b, f32w c) noexcept { return f32w{ std::fma(a.v, b.v, -c.v) }; }

// Math ops: min/max/abs/sqrt + estimate rsqrt/recip (Task 3)
ARCANE_SIMD_INLINE f32w min(f32w a, f32w b) noexcept { return f32w{ a.v < b.v ? a.v : b.v }; }
ARCANE_SIMD_INLINE f32w max(f32w a, f32w b) noexcept { return f32w{ a.v > b.v ? a.v : b.v }; }
ARCANE_SIMD_INLINE f32w abs(f32w a)         noexcept { return f32w{ std::fabs(a.v) }; }
ARCANE_SIMD_INLINE f32w sqrt(f32w a)        noexcept { return f32w{ std::sqrt(a.v) }; }
ARCANE_SIMD_INLINE f32w rsqrt(f32w a)       noexcept { return f32w{ 1.0f / std::sqrt(a.v) }; }
ARCANE_SIMD_INLINE f32w recip(f32w a)       noexcept { return f32w{ 1.0f / a.v }; }

// Compare / select / mask reductions (Task 4)
ARCANE_SIMD_INLINE b32w cmp_gt(f32w a, f32w b) noexcept { return b32w{ a.v >  b.v }; }
ARCANE_SIMD_INLINE b32w cmp_ge(f32w a, f32w b) noexcept { return b32w{ a.v >= b.v }; }
ARCANE_SIMD_INLINE b32w cmp_lt(f32w a, f32w b) noexcept { return b32w{ a.v <  b.v }; }
ARCANE_SIMD_INLINE b32w cmp_le(f32w a, f32w b) noexcept { return b32w{ a.v <= b.v }; }
ARCANE_SIMD_INLINE b32w cmp_eq(f32w a, f32w b) noexcept { return b32w{ a.v == b.v }; }
ARCANE_SIMD_INLINE f32w select(b32w m, f32w t, f32w f) noexcept { return m.m ? t : f; }
ARCANE_SIMD_INLINE bool any (b32w m) noexcept { return  m.m; }
ARCANE_SIMD_INLINE bool all (b32w m) noexcept { return  m.m; }
ARCANE_SIMD_INLINE bool none (b32w m) noexcept { return !m.m; }

// Gather / scatter (Task 5)
ARCANE_SIMD_INLINE f32w gather(const float* base, i32w idx) noexcept { return f32w{ base[idx.v] }; }
ARCANE_SIMD_INLINE void scatter(float* base, i32w idx, f32w a) noexcept { base[idx.v] = a.v; }

} } } // namespace Manifold2D::Simd::Scalar
