#pragma once

// Manifold2D::IWorkScheduler -- the injected data-parallel seam (Astra's
// IWorkScheduler model; design: docs/superpowers/specs/2026-07-10-manifold2d-phase2-lift-design.md,
// D2). Manifold2D creates no threads; SerialWorkScheduler is the deterministic
// default when the consumer injects none. The host engine adapts its own task
// executor to this interface -- a small forwarding copy, so this seam has no
// dependency back on any host engine type; the dependency arrow runs one way,
// host -> Manifold2D. See ParallelFor below for the per-worker-disjoint-writes
// contract the solver relies on (kept verbatim from the origin).

#include <Manifold2D/Core/FunctionRef.hpp>

#include <cstddef>
#include <cstdint>

namespace Manifold2D
{
    struct IWorkScheduler
    {
        // Partition [0,count) into DISJOINT sub-ranges (grain >= minBatch where
        // possible) covering it exactly once; invoke fn(begin,end,worker) on each.
        // worker in [0,WorkerCount()) names the running thread (per-worker scratch).
        // Each concurrently-running sub-range gets a DISTINCT worker id, so fn may use worker as an index into per-worker scratch without locking (the physics solver-MT and broadphase-MT rely on this for disjoint per-worker writes).
        // BLOCKS until all sub-ranges complete. count==0 is a no-op. Re-entrant:
        // legal to call from within an fn already running on this executor.
        // A minBatch of 0 is treated as 1.
        virtual void ParallelFor(std::size_t count, std::size_t minBatch,
                                 FunctionRef<void(std::size_t begin, std::size_t end,
                                                  std::uint32_t worker)> fn) = 0;

        // Inclusive of the calling thread; always >= 1. Batch-size denominator.
        virtual std::uint32_t WorkerCount() const noexcept = 0;

        virtual ~IWorkScheduler() = default;
    };

    // Single-threaded reference: runs the whole range inline as worker 0.
    class SerialWorkScheduler final : public IWorkScheduler
    {
    public:
        void ParallelFor(std::size_t count, std::size_t /*minBatch*/,
                         FunctionRef<void(std::size_t, std::size_t, std::uint32_t)> fn) override
        {
            if (count == 0) return;
            fn(0, count, 0);
        }

        std::uint32_t WorkerCount() const noexcept override { return 1; }
    };
}
