// Manifold2D::Core primitive smoke (Phase 2 Task 1). Proves the lifted primitives
// compile + link from their new <Manifold2D/Core/...> home in namespace Manifold2D,
// with zero Arcane dependency. Deeper semantics are covered by the physics/geometry
// suites once they move (Task 2); this is a link+surface tripwire.
#include <cstdint>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <Manifold2D/Core/FunctionRef.hpp>
#include <Manifold2D/Core/BitSet.hpp>
#include <Manifold2D/Core/Simd.hpp>
#include <Manifold2D/Core/WorkScheduler.hpp>

TEST_CASE("Manifold2D::FunctionRef binds and calls", "[manifold2d]")
{
    int seen = 0;
    auto lam = [&](int v) { seen = v; };
    Manifold2D::FunctionRef<void(int)> ref = lam;
    ref(42);
    CHECK(seen == 42);
}

TEST_CASE("Manifold2D::BitSet set/ForEachSetBit reads back set bits only", "[manifold2d]")
{
    // Manifold2D::BitSet's real surface is Resize/Set/ClearAll/InPlaceUnion/
    // ForEachSetBit (no Test() accessor) -- adapted from the brief's sketch to
    // match Manifold2D/Core/BitSet.hpp. Assertion intent preserved: a set bit
    // reads back set, an unset bit reads back clear.
    Manifold2D::BitSet b;
    b.Resize(70);
    b.Set(3);
    b.Set(64);
    std::vector<std::uint32_t> got;
    b.ForEachSetBit([&](std::uint32_t i) { got.push_back(i); });
    CHECK(got == std::vector<std::uint32_t>{3u, 64u});
}

TEST_CASE("Manifold2D::SerialWorkScheduler runs a range inline as worker 0", "[manifold2d]")
{
    Manifold2D::SerialWorkScheduler sched;
    CHECK(sched.WorkerCount() == 1u);
    std::uint32_t calls = 0, lastWorker = 99; std::size_t total = 0;
    sched.ParallelFor(10, 1, [&](std::size_t b, std::size_t e, std::uint32_t w) {
        ++calls; lastWorker = w; total += (e - b);
    });
    CHECK(calls == 1u);
    CHECK(lastWorker == 0u);
    CHECK(total == 10u);
}

TEST_CASE("Manifold2D::Simd f32w lane round-trip", "[manifold2d][simd]")
{
    using Manifold2D::Simd::f32w;
    // Manifold2D::Simd's real surface is free functions (load/store, operator+)
    // over f32w::width lanes (not member Load/Store/kLanes) -- adapted from the
    // brief's sketch to match Manifold2D/Core/Simd.hpp + the house pattern in
    // Arcane/Core/src/Arcane/Math/SimdSmoke.cpp. Assertion intent preserved: a
    // per-lane load followed by an add reads back the expected per-lane value.
    alignas(32) float in[8] = {};
    alignas(32) float out[8] = {};
    for (int i = 0; i < f32w::width; ++i) in[i] = float(i);
    f32w a = Manifold2D::Simd::load(in);
    f32w b = a + a;
    Manifold2D::Simd::store(out, b);
    for (int i = 0; i < f32w::width; ++i) CHECK(out[i] == in[i] + in[i]);
}
