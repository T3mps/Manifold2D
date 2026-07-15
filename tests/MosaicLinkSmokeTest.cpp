// Mosaic link smoke. Manifold2D's Core primitives (FunctionRef, BitSet, Simd,
// IWorkScheduler) all moved OUT to Mosaic, the shared Starworks core -- there is no
// Manifold2D/Core/ any more. This is the tripwire that the vendored Mosaic compiles
// and links from here, and that the surface Manifold2D depends on is present.
// Semantics are owned by Mosaic's own suite; the physics/geometry suites exercise
// these types for real.
#include <cstdint>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <Mosaic/FunctionRef.hpp>
#include <Mosaic/BitSet.hpp>
#include <Mosaic/Simd/Wide.hpp>
#include <Mosaic/Jobs/WorkScheduler.hpp>

TEST_CASE("Mosaic::FunctionRef binds and calls", "[manifold2d]")
{
    int seen = 0;
    auto lam = [&](int v) { seen = v; };
    Mosaic::FunctionRef<void(int)> ref = lam;
    ref(42);
    CHECK(seen == 42);
}

TEST_CASE("Mosaic::BitSet set/ForEachSetBit reads back set bits only", "[manifold2d]")
{
    // BitSet's surface is Resize/Set/ClearAll/InPlaceUnion/ForEachSetBit (there is
    // no Test() accessor): a set bit reads back set, an unset bit reads back clear.
    Mosaic::BitSet b;
    b.Resize(70);
    b.Set(3);
    b.Set(64);
    std::vector<std::uint32_t> got;
    b.ForEachSetBit([&](std::uint32_t i) { got.push_back(i); });
    CHECK(got == std::vector<std::uint32_t>{3u, 64u});
}

TEST_CASE("Mosaic::SerialWorkScheduler runs a range inline as worker 0", "[manifold2d]")
{
    Mosaic::SerialWorkScheduler sched;
    CHECK(sched.WorkerCount() == 1u);
    std::uint32_t calls = 0, lastWorker = 99; std::size_t total = 0;
    sched.ParallelFor(10, 1, [&](std::size_t b, std::size_t e, std::uint32_t w) {
        ++calls; lastWorker = w; total += (e - b);
    });
    CHECK(calls == 1u);
    CHECK(lastWorker == 0u);
    CHECK(total == 10u);
}

TEST_CASE("Mosaic::Simd f32w lane round-trip", "[manifold2d][simd]")
{
    using Mosaic::Simd::f32w;
    // Mosaic::Simd's surface is free functions (load/store, operator+) over
    // f32w::width lanes, not member Load/Store/kLanes: a per-lane load followed by
    // an add reads back the expected per-lane value.
    alignas(32) float in[8] = {};
    alignas(32) float out[8] = {};
    for (int i = 0; i < f32w::width; ++i) in[i] = float(i);
    f32w a = Mosaic::Simd::load(in);
    f32w b = a + a;
    Mosaic::Simd::store(out, b);
    for (int i = 0; i < f32w::width; ++i) CHECK(out[i] == in[i] + in[i]);
}
