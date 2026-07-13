#pragma once

// Geometry::Vec2<T> -- the first-party 2D vector for Manifold2D (the physics
// engine) and the Geometry hull/predicate kernel. Replaces glm::vec2 /
// glm::vec<2,T> (Phase 1 of the Manifold2D extraction; spec:
// docs/superpowers/specs/2026-07-10-manifold2d-phase1-vec2-design.md).
//
// Rules this file lives by (spec D1-D4):
// - Constrained by the duck-typed Numeric concept: float, double, integers,
//   AND lane-wide types (Simd::f32w) all qualify WITHOUT this header
//   depending on the Simd module (the dependency arrow must keep pointing
//   away from Geometry so Manifold2D stays liftable).
// - NOT internally SIMD-vectorized. Per-object SIMD wastes lanes and fights
//   the solver's SoA architecture; Vec2<f32w> (8 lanes in .x, 8 in .y) IS
//   the wide form.
// - glm-parity layout and semantics: {T x, y}, standard layout, trivially
//   copyable, sizeof(Vec2<float>) == 8, and the default constructor leaves
//   members UNINITIALIZED (hot stack arrays like Collide.cpp's
//   Vec2 va[kMaxPolyVerts] must not gain zero-init cost).
// - Expression-order fidelity: the free functions compute the exact trees
//   the physics/geometry code computed against glm (Dot = a.x*b.x + a.y*b.y,
//   Cross = a.x*b.y - a.y*b.x, Perp = (v.y, -v.x)). Under /fp:strict the
//   whole existing suite is a value-identity gate over this file.
// - bool-returning / branching operations are scalar-only (concept-gated);
//   the arithmetic core compiles for every Numeric T. Wide comparisons /
//   mask-selects are deliberately absent until the solver wants them.

#include <cmath>
#include <concepts>
#include <type_traits>

namespace Manifold2D::Geometry
{
    // Duck-typed numeric constraint (spec D1). Requires closure under the
    // arithmetic the Vec2 core uses; deliberately does NOT require division,
    // comparison, or construction-from-int so lane-wide types qualify.
    // bool is excluded explicitly: bool+bool promotes to int and converts
    // back to bool, so it would otherwise satisfy the closure requirements
    // while producing nonsense components (review fix).
    template <typename T>
    concept Numeric = !std::same_as<T, bool> && requires(T a, T b) {
        { a + b } -> std::convertible_to<T>;
        { a - b } -> std::convertible_to<T>;
        { a * b } -> std::convertible_to<T>;
        { -a }    -> std::convertible_to<T>;
    };

    template <Numeric T>
    struct Vec2
    {
        T x, y;

        // glm-parity: default construction leaves x/y UNINITIALIZED.
        constexpr Vec2() = default;
        constexpr Vec2(T xIn, T yIn) noexcept : x(xIn), y(yIn) {}

        constexpr Vec2& operator+=(const Vec2& r) noexcept { x = x + r.x; y = y + r.y; return *this; }
        constexpr Vec2& operator-=(const Vec2& r) noexcept { x = x - r.x; y = y - r.y; return *this; }
        constexpr Vec2& operator*=(T s) noexcept { x = x * s; y = y * s; return *this; }
        constexpr Vec2& operator/=(T s) noexcept { x = x / s; y = y / s; return *this; }
    };

    using Vec2f = Vec2<float>;
    using Vec2d = Vec2<double>;

    // ------------------------------------------------------------------
    // Arithmetic core -- valid for every Numeric T (including Vec2<f32w>).
    // ------------------------------------------------------------------

    template <Numeric T>
    constexpr Vec2<T> operator+(const Vec2<T>& a, const Vec2<T>& b) noexcept { return Vec2<T>(a.x + b.x, a.y + b.y); }

    template <Numeric T>
    constexpr Vec2<T> operator-(const Vec2<T>& a, const Vec2<T>& b) noexcept { return Vec2<T>(a.x - b.x, a.y - b.y); }

    template <Numeric T>
    constexpr Vec2<T> operator-(const Vec2<T>& v) noexcept { return Vec2<T>(-v.x, -v.y); }

    template <Numeric T>
    constexpr Vec2<T> operator*(const Vec2<T>& v, T s) noexcept { return Vec2<T>(v.x * s, v.y * s); }

    template <Numeric T>
    constexpr Vec2<T> operator*(T s, const Vec2<T>& v) noexcept { return Vec2<T>(s * v.x, s * v.y); }

    template <Numeric T>
    constexpr Vec2<T> operator/(const Vec2<T>& v, T s) noexcept { return Vec2<T>(v.x / s, v.y / s); }

    // Dot product. EXACT tree: a.x*b.x + a.y*b.y (matches SoftStep's local
    // Dot and every inlined physics dot).
    template <Numeric T>
    constexpr T Dot(const Vec2<T>& a, const Vec2<T>& b) noexcept { return a.x * b.x + a.y * b.y; }

    // 2D scalar cross (perp-dot). EXACT tree: a.x*b.y - a.y*b.x (matches
    // Physics::Math::Cross2 and Geometry::detail::Cross's componentwise form).
    template <Numeric T>
    constexpr T Cross(const Vec2<T>& a, const Vec2<T>& b) noexcept { return a.x * b.y - a.y * b.x; }

    template <Numeric T>
    constexpr T LengthSq(const Vec2<T>& v) noexcept { return v.x * v.x + v.y * v.y; }

    // Left-hand perpendicular in the engine's y-down convention: (v.y, -v.x).
    // Matches Physics::Math::Perp exactly.
    template <Numeric T>
    constexpr Vec2<T> Perp(const Vec2<T>& v) noexcept { return Vec2<T>(v.y, -v.x); }

    // ------------------------------------------------------------------
    // Scalar-only surface (bool-returning or branching; spec D3).
    // ------------------------------------------------------------------

    template <Numeric T>
        requires std::floating_point<T> || std::integral<T>
    constexpr bool operator==(const Vec2<T>& a, const Vec2<T>& b) noexcept { return a.x == b.x && a.y == b.y; }

    template <Numeric T>
        requires std::floating_point<T> || std::integral<T>
    constexpr bool operator!=(const Vec2<T>& a, const Vec2<T>& b) noexcept { return !(a == b); }

    template <Numeric T>
        requires std::floating_point<T>
    inline T Length(const Vec2<T>& v) noexcept { return std::sqrt(v.x * v.x + v.y * v.y); }

    // Zero-guarded normalization: a zero-length vector normalizes to zero
    // (no NaN). New convenience -- no existing caller constrains this choice;
    // documented here as THE contract.
    template <Numeric T>
        requires std::floating_point<T>
    inline Vec2<T> Normalized(const Vec2<T>& v) noexcept
    {
        const T len = Length(v);
        if (len > T(0))
            return Vec2<T>(v.x / len, v.y / len);
        return Vec2<T>(T(0), T(0));
    }

    template <Numeric T>
        requires std::floating_point<T> || std::integral<T>
    constexpr Vec2<T> Min(const Vec2<T>& a, const Vec2<T>& b) noexcept
    { return Vec2<T>(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y); }

    template <Numeric T>
        requires std::floating_point<T> || std::integral<T>
    constexpr Vec2<T> Max(const Vec2<T>& a, const Vec2<T>& b) noexcept
    { return Vec2<T>(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y); }

    static_assert(sizeof(Vec2<float>) == 8, "Vec2<float> must be layout-identical to glm::vec2");
    static_assert(std::is_standard_layout_v<Vec2<float>>);
    static_assert(std::is_trivially_copyable_v<Vec2<float>>);
    static_assert(sizeof(Vec2<double>) == 16);
}
