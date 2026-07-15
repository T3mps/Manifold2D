// Shared [simd] test bodies. Included by SimdWideTest.cpp (active backend) and
// SimdWideScalarTest.cpp (forced scalar). Width-agnostic: loops f32w::width.
#include <Mosaic/Simd/Wide.hpp>
#include <Support/SimdSmoke.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm> // std::min / std::max (Task 3)
#include <cmath>     // std::fma / std::fabs / std::sqrt (Task 2+)
#include <cstdint>
#include <cstring>   // std::memcpy for index-vector construction (Task 5)

#ifndef MOSAIC_SIMD_TUTAG
    #define MOSAIC_SIMD_TUTAG "active"
#endif

// MOSAIC_SIMD_NS is defined by Simd.hpp to point at the active backend's
// sub-namespace (e.g. ::Mosaic::Simd::Avx2 or ::Mosaic::Simd::Scalar).
// This avoids ODR violations when both TUs are linked into the same binary.
namespace SimdT = MOSAIC_SIMD_NS;

TEST_CASE("Simd[" MOSAIC_SIMD_TUTAG "]: backend + width are sane", "[simd]")
{
    INFO("backend = " << SimdT::kBackendName << " width = " << SimdT::f32w::width);
    CHECK(SimdT::f32w::width >= 1);
    CHECK((SimdT::f32w::width == 1 || SimdT::f32w::width == 4 || SimdT::f32w::width == 8));
    CHECK(SimdT::b32w::width == SimdT::f32w::width);
    CHECK(SimdT::i32w::width == SimdT::f32w::width);
    CHECK(SimdT::kBackendName != nullptr);
}

TEST_CASE("Simd[" MOSAIC_SIMD_TUTAG "]: splat/load/store round-trip", "[simd]")
{
    constexpr int W = SimdT::f32w::width;
    alignas(32) float in[W];
    alignas(32) float out[W];
    for (int i = 0; i < W; ++i) in[i] = static_cast<float>(i) * 1.5f - 3.0f;

    // splat
    SimdT::f32w s = SimdT::splat(2.5f);
    SimdT::store(out, s);
    for (int i = 0; i < W; ++i) CHECK(out[i] == 2.5f);

    // setzero
    SimdT::store(out, SimdT::setzero());
    for (int i = 0; i < W; ++i) CHECK(out[i] == 0.0f);

    // load -> store identity (aligned)
    SimdT::store(out, SimdT::load(in));
    for (int i = 0; i < W; ++i) CHECK(out[i] == in[i]);

    // loadu/storeu identity (offset by 1 float)
    float ub[W + 1];
    for (int i = 0; i < W; ++i) ub[i + 1] = in[i];
    float uo[W + 1];
    SimdT::storeu(uo + 1, SimdT::loadu(ub + 1));
    for (int i = 0; i < W; ++i) CHECK(uo[i + 1] == in[i]);
}

TEST_CASE("Simd[" MOSAIC_SIMD_TUTAG "]: arithmetic matches scalar reference", "[simd]")
{
    constexpr int W = SimdT::f32w::width;
    alignas(32) float a[W], b[W], c[W], out[W];
    for (int i = 0; i < W; ++i) { a[i] = 1.0f + i; b[i] = 0.5f * (i + 2); c[i] = -2.0f + i; }

    SimdT::f32w va = SimdT::load(a), vb = SimdT::load(b), vc = SimdT::load(c);

    SimdT::store(out, va + vb); for (int i = 0; i < W; ++i) CHECK(out[i] == a[i] + b[i]);
    SimdT::store(out, va - vb); for (int i = 0; i < W; ++i) CHECK(out[i] == a[i] - b[i]);
    SimdT::store(out, va * vb); for (int i = 0; i < W; ++i) CHECK(out[i] == a[i] * b[i]);
    SimdT::store(out, va / vb); for (int i = 0; i < W; ++i) CHECK(out[i] == a[i] / b[i]);
    SimdT::store(out, -va);     for (int i = 0; i < W; ++i) CHECK(out[i] == -a[i]);

    SimdT::f32w acc = va; acc += vb; SimdT::store(out, acc);
    for (int i = 0; i < W; ++i) CHECK(out[i] == a[i] + b[i]);

    SimdT::f32w acc2 = va; acc2 -= vb; SimdT::store(out, acc2);
    for (int i = 0; i < W; ++i) CHECK(out[i] == a[i] - b[i]);
    SimdT::f32w acc3 = va; acc3 *= vb; SimdT::store(out, acc3);
    for (int i = 0; i < W; ++i) CHECK(out[i] == a[i] * b[i]);
    SimdT::f32w acc4 = va; acc4 /= vb; SimdT::store(out, acc4);
    for (int i = 0; i < W; ++i) CHECK(out[i] == a[i] / b[i]);

    // mul_add / mul_sub use fused multiply-add -> bit-match std::fma per lane.
    SimdT::store(out, SimdT::mul_add(va, vb, vc));
    for (int i = 0; i < W; ++i) CHECK(out[i] == std::fma(a[i], b[i], c[i]));
    SimdT::store(out, SimdT::mul_sub(va, vb, vc));
    for (int i = 0; i < W; ++i) CHECK(out[i] == std::fma(a[i], b[i], -c[i]));
}

TEST_CASE("Simd[" MOSAIC_SIMD_TUTAG "]: min/max/abs/sqrt exact; rsqrt/recip approx", "[simd]")
{
    constexpr int W = SimdT::f32w::width;
    alignas(32) float a[W], b[W], out[W];
    for (int i = 0; i < W; ++i) { a[i] = (i % 2 ? -1.0f : 1.0f) * (i + 1) * 1.25f; b[i] = (i + 1) * 0.75f; }

    SimdT::f32w va = SimdT::load(a), vb = SimdT::load(b);

    SimdT::store(out, SimdT::min(va, vb)); for (int i = 0; i < W; ++i) CHECK(out[i] == std::min(a[i], b[i]));
    SimdT::store(out, SimdT::max(va, vb)); for (int i = 0; i < W; ++i) CHECK(out[i] == std::max(a[i], b[i]));
    SimdT::store(out, SimdT::abs(va));     for (int i = 0; i < W; ++i) CHECK(out[i] == std::fabs(a[i]));
    SimdT::store(out, SimdT::sqrt(vb));    for (int i = 0; i < W; ++i) CHECK(out[i] == std::sqrt(b[i]));

    // rsqrt / recip are hardware-estimate ops -> relative tolerance (AVX2 rcp/rsqrt
    // ~12-bit; scalar backend is exact and well within tol).
    constexpr float kRelTol = 4.0e-3f;
    SimdT::store(out, SimdT::rsqrt(vb));
    for (int i = 0; i < W; ++i) {
        float ref = 1.0f / std::sqrt(b[i]);
        CHECK(std::fabs(out[i] - ref) <= kRelTol * std::fabs(ref));
    }
    SimdT::store(out, SimdT::recip(vb));
    for (int i = 0; i < W; ++i) {
        float ref = 1.0f / b[i];
        CHECK(std::fabs(out[i] - ref) <= kRelTol * std::fabs(ref));
    }
}

TEST_CASE("Simd[" MOSAIC_SIMD_TUTAG "]: compare/select/mask reductions", "[simd]")
{
    constexpr int W = SimdT::f32w::width;
    alignas(32) float a[W], b[W], t[W], f[W], out[W];
    for (int i = 0; i < W; ++i) { a[i] = float(i); b[i] = float(W - 1 - i); t[i] = 100.0f + i; f[i] = -100.0f - i; }

    SimdT::f32w va = SimdT::load(a), vb = SimdT::load(b), vt = SimdT::load(t), vf = SimdT::load(f);

    // select(cmp_gt(a,b), t, f) -> t where a>b else f
    SimdT::store(out, SimdT::select(SimdT::cmp_gt(va, vb), vt, vf));
    for (int i = 0; i < W; ++i) CHECK(out[i] == (a[i] > b[i] ? t[i] : f[i]));

    SimdT::store(out, SimdT::select(SimdT::cmp_ge(va, vb), vt, vf));
    for (int i = 0; i < W; ++i) CHECK(out[i] == (a[i] >= b[i] ? t[i] : f[i]));
    SimdT::store(out, SimdT::select(SimdT::cmp_lt(va, vb), vt, vf));
    for (int i = 0; i < W; ++i) CHECK(out[i] == (a[i] <  b[i] ? t[i] : f[i]));
    SimdT::store(out, SimdT::select(SimdT::cmp_le(va, vb), vt, vf));
    for (int i = 0; i < W; ++i) CHECK(out[i] == (a[i] <= b[i] ? t[i] : f[i]));
    SimdT::store(out, SimdT::select(SimdT::cmp_eq(va, va), vt, vf));
    for (int i = 0; i < W; ++i) CHECK(out[i] == t[i]);

    // cmp_eq false-lane coverage: va vs vb differ on most lanes (W>1 backends).
    SimdT::store(out, SimdT::select(SimdT::cmp_eq(va, vb), vt, vf));
    for (int i = 0; i < W; ++i) CHECK(out[i] == (a[i] == b[i] ? t[i] : f[i]));

    // mask reductions
    CHECK(SimdT::all(SimdT::cmp_eq(va, va)));
    CHECK(SimdT::none(SimdT::cmp_lt(va, vf)));     // a >= 0 > f, never <
    bool anyExpected = false; for (int i = 0; i < W; ++i) if (a[i] > b[i]) { anyExpected = true; break; }
    CHECK(SimdT::any(SimdT::cmp_gt(va, vb)) == anyExpected);

    // direct any-false path: a >= 0 > f, so cmp_lt(va,vf) is all-false.
    CHECK(SimdT::any(SimdT::cmp_lt(va, vf)) == false);
}

TEST_CASE("Simd[" MOSAIC_SIMD_TUTAG "]: gather/scatter round-trip", "[simd]")
{
    constexpr int W = SimdT::f32w::width;
    // base table large enough for any index pattern.
    float base[64];
    for (int i = 0; i < 64; ++i) base[i] = float(i) * 10.0f;

    // build an index vector via memcpy of a plain int array into i32w.
    alignas(32) int32_t idx[W];
    for (int i = 0; i < W; ++i) idx[i] = (i * 7 + 3) % 64;   // scattered, in-range
    SimdT::i32w vi;
    std::memcpy(&vi, idx, sizeof(vi));   // backends lay i32w out as `width` int32 lanes

    alignas(32) float out[W];
    SimdT::store(out, SimdT::gather(base, vi));
    for (int i = 0; i < W; ++i) CHECK(out[i] == base[idx[i]]);

    // scatter into a fresh table, then read back.
    float dst[64]; for (int i = 0; i < 64; ++i) dst[i] = -1.0f;
    alignas(32) float vals[W]; for (int i = 0; i < W; ++i) vals[i] = float(i) + 0.25f;
    SimdT::scatter(dst, vi, SimdT::load(vals));
    for (int i = 0; i < W; ++i) CHECK(dst[idx[i]] == vals[i]);

    // index-vector constructors (isplat / iota) -- the solver builds gather
    // indices from these, so verify their lane layout here.
    alignas(32) int32_t ib[W];
    SimdT::i32w vs5 = SimdT::isplat(5); std::memcpy(ib, &vs5, sizeof(vs5));
    for (int i = 0; i < W; ++i) CHECK(ib[i] == 5);
    SimdT::i32w vio = SimdT::iota();    std::memcpy(ib, &vio, sizeof(vio));
    for (int i = 0; i < W; ++i) CHECK(ib[i] == i);
}

TEST_CASE("Simd[" MOSAIC_SIMD_TUTAG "]: iload round-trips an int32 array -> i32w", "[simd]")
{
    // iload is the int counterpart to f32w `load`: the contact solver packs each
    // lane's body slot into an alignas(32) int32_t[width] array (ContactConstraint-
    // Simd::bodyIndexA/B) and turns it into the gather/scatter index vector with
    // iload. Build a known index pattern, iload it, and gather a base table by it
    // -> the gathered lanes must equal base[idx[i]] (proving iload's lane layout
    // matches what gather/scatter consume). Then exercise iload directly through a
    // scatter round-trip (the solver scatters velocities back by the same i32w).
    constexpr int W = SimdT::f32w::width;

    alignas(32) int32_t idx[W];
    for (int i = 0; i < W; ++i) idx[i] = (i * 5 + 2) % 60;   // scattered, in-range

    // iload the aligned index array (AVX2 _mm256_load_si256 asserts 32B-aligned).
    SimdT::i32w vi = SimdT::iload(idx);

    // Lane layout: iload(idx) must equal the memcpy-built i32w the existing
    // gather/scatter test relied on (same lanes in the same order).
    alignas(32) int32_t back[W];
    std::memcpy(back, &vi, sizeof(vi));
    for (int i = 0; i < W; ++i) CHECK(back[i] == idx[i]);

    // gather by the iload'd index -> base[idx[i]] per lane.
    float base[64]; for (int i = 0; i < 64; ++i) base[i] = float(i) * 3.0f + 1.0f;
    alignas(32) float out[W];
    SimdT::store(out, SimdT::gather(base, vi));
    for (int i = 0; i < W; ++i) CHECK(out[i] == base[idx[i]]);

    // scatter by the iload'd index -> dst[idx[i]] = vals[i].
    float dst[64]; for (int i = 0; i < 64; ++i) dst[i] = -7.0f;
    alignas(32) float vals[W]; for (int i = 0; i < W; ++i) vals[i] = float(i) * 2.0f - 0.5f;
    SimdT::scatter(dst, vi, SimdT::load(vals));
    for (int i = 0; i < W; ++i) CHECK(dst[idx[i]] == vals[i]);
}

TEST_CASE("Simd[" MOSAIC_SIMD_TUTAG "]: run-twice determinism (bit-identical)", "[simd]")
{
    constexpr int W = SimdT::f32w::width;
    auto run = [](float* out) {
        alignas(32) float a[W], b[W], c[W];
        for (int i = 0; i < W; ++i) { a[i] = 1.0f + i * 0.3f; b[i] = 2.0f - i * 0.1f; c[i] = 0.5f * i; }
        SimdT::f32w r = SimdT::mul_add(SimdT::load(a), SimdT::load(b),
                                       SimdT::max(SimdT::load(c), SimdT::setzero()));
        r = SimdT::select(SimdT::cmp_gt(r, SimdT::splat(3.0f)), SimdT::sqrt(r), r);
        SimdT::store(out, r);
    };
    alignas(32) float o1[W], o2[W];
    run(o1); run(o2);
    for (int i = 0; i < W; ++i)
    {
        std::uint32_t u1, u2;
        std::memcpy(&u1, &o1[i], 4); std::memcpy(&u2, &o2[i], 4);
        CHECK(u1 == u2);   // bit-identical across runs
    }
}

TEST_CASE("Simd[" MOSAIC_SIMD_TUTAG "]: Core pilot SimdSmokeSum matches scalar sum", "[simd]")
{
    float data[37];
    double ref = 0.0;
    for (int i = 0; i < 37; ++i) { data[i] = float(i) * 0.5f - 4.0f; ref += data[i]; }
    float got = Mosaic::Simd::SimdSmokeSum(data, 37);
    // SoA lane-sum reorders additions -> tolerance check, not bit-exact (documents
    // that horizontal sums are order-dependent across widths).
    CHECK(std::fabs(got - static_cast<float>(ref)) <= 1e-3f * (1.0f + std::fabs(static_cast<float>(ref))));
}
