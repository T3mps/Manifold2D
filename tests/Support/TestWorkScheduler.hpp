#pragma once

// Test-only std::thread pool implementing Mosaic::IWorkScheduler, so the
// MT-invariance suites can exercise the parallel ParallelFor paths. The library
// itself creates no threads -- this lives in tests/ only (mirrors Astra's
// tests/Support/TestWorkerPool.hpp). It replaces the enki-backed
// Arcane::ArcaneWorkScheduler the tests used inside the Aphelyon monorepo, so
// the standalone repo has no Arcane dependency.
//
// Partitioning contract (kept identical to the enki executor the solver was
// tuned against): [0,count) is split into contiguous ascending ranges with
// grain = ceil(count/workers), so the range count never exceeds WorkerCount().
// Each range is assigned a DISTINCT worker id in [0,WorkerCount()) and runs
// concurrently (the calling thread runs range 0 as worker 0; ranges 1..k run on
// spawned threads). This is exactly what the solver-MT / broadphase-MT /
// narrowphase-MT paths rely on to index per-worker scratch without locking.

#include <Mosaic/Jobs/WorkScheduler.hpp>
#include <Mosaic/FunctionRef.hpp>

#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

namespace Manifold2D::Testing
{
    class TestWorkScheduler final : public Mosaic::IWorkScheduler
    {
    public:
        explicit TestWorkScheduler(std::uint32_t workers)
            : m_workers(workers < 1u ? 1u : workers) {}

        void ParallelFor(std::size_t count, std::size_t minBatch,
                         Mosaic::FunctionRef<void(std::size_t, std::size_t,
                                                      std::uint32_t)> fn) override
        {
            if (count == 0) return;

            const std::uint32_t w = m_workers;
            std::size_t grain = (count + w - 1) / w;          // ceil(count/workers)
            const std::size_t minB = minBatch ? minBatch : 1;
            if (grain < minB) grain = minB;                   // honor minimum batch

            // Spawn a thread per tail range (1..k); range 0 runs inline as worker 0.
            // grain = ceil(count/workers) bounds the range count at <= workers, so
            // every worker id stays in [0,WorkerCount()).
            std::vector<std::thread> pool;
            std::uint32_t wid = 0;
            for (std::size_t begin = 0; begin < count; begin += grain, ++wid)
            {
                std::size_t end = begin + grain;
                if (end > count) end = count;
                if (begin == 0) continue;                     // worker 0 runs inline below
                pool.emplace_back([&fn, begin, end, wid] { fn(begin, end, wid); });
            }

            const std::size_t firstEnd = grain < count ? grain : count;
            fn(0, firstEnd, 0);                               // worker 0, range [0, firstEnd)

            for (auto& t : pool) t.join();
        }

        std::uint32_t WorkerCount() const noexcept override { return m_workers; }

    private:
        std::uint32_t m_workers;
    };
}
