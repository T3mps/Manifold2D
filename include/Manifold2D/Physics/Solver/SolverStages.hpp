#pragma once

// SolverStages.hpp -- the Box2D-v3 b2SolverStage stage/block machinery that runs
// the whole SoftStep substep loop inside ONE worker region (Gap 1, solver-MT
// scaling rework). This header owns:
//   1. The stage/block descriptors (StageType / SolverBlock / SolverStage) the
//      driver pre-builds once per step into reused scratch (SoftStep::m_stages /
//      m_blocks).
//   2. SolverStageContext -- the per-step bundle ExecuteBlock / SolverWorker read
//      (the constraint data handles, the integrate/overflow/joint callbacks, and
//      the atomic stage-sync word).
//   3. The ring-CAS claim loop (ExecuteStage / ExecuteStageForTest), the per-stage
//      orchestration (ExecuteMainStage), the block dispatch (ExecuteBlock), and
//      the worker entry (SolverWorker).
//
// THREADING MODEL (Box2D src/solver.c). Box2D enqueues `workerCount` tasks ONCE
// per step; the workers run the entire substep loop inside a persistent region,
// advancing across colors/passes/substeps with a single atomic store
// (atomicSyncBits) + a spin-wait -- never re-entering the task system. A color
// barrier costs one atomic store + one spin, NOT an enqueue+join. This replaces
// SoftStep's ~190 fork-join ParallelFor dispatches/step with ONE region.
//
// GAP 1.2 (this task): the region is genuinely MULTITHREADED. SoftStep::Solve
// dispatches it ONCE per step via IWorkScheduler::ParallelFor(workerCount, 1, ...)
// (worker index = the partition start `begin`, Box2D b2SolverTask style):
// `begin == 0` is the main/orchestrator (the partition covering index 0 always
// exists -> exactly one main); every `begin > 0` is a thief that spins on the
// stageSync word and steals blocks via ring-CAS. The main self-completes every
// block alone (claim-at-execution), so thieves are pure optimization: liveness +
// correctness hold at ANY worker count (serial == enki(1) == enki(N)). After the
// last stage the main publishes a TERMINAL sentinel (kSolverStageSyncTerminal)
// the thieves check to EXIT, so ParallelFor's join never hangs. The stage walk
// reproduces SoftStep's exact D1 substep order, so the result is BYTE-IDENTICAL
// to the per-color ParallelFor it replaces, at any worker count.
//
// DETERMINISM. Stage order is fixed; a color boundary is a hard barrier (spin
// until completionCount == blockCount -- adjacent colors share bodies, Gauss-
// Seidel). Within a color, blocks partition the color's body-DISJOINT batches, so
// which-block-runs-which and in-what-order cannot change any float. Atomics are
// control-flow ONLY (syncIndex CAS, completionCount, stageSync) -- never in FP
// math -- so there is no atomic-ordering-induced FP non-determinism. Hence
// serial == enki(1) == enki(N), byte-identical.
//
// INCLUDE GRAPH. This header forward-declares SoftStep (only a pointer field
// references it) and pulls the constraint data + SIMD passes from
// ContactConstraintSimd.hpp; it does NOT include SoftStep.hpp (SoftStep.hpp
// includes THIS for its m_stages/m_blocks members). ExecuteBlock reaches the
// solver's data through SolverStageContext handles + FunctionRef callbacks, so
// it never dereferences SoftStep -- keeping the two files decoupled.
//
// PRESENTATION-FREE + C++20-clean: std + sibling Physics headers + Manifold2D::Simd
// (via ContactConstraintSimd.hpp). namespace Manifold2D::Physics.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <Manifold2D/Core/FunctionRef.hpp>
#include <Manifold2D/Physics/Solver/BodyState.hpp>            // BodyState row
#include <Manifold2D/Physics/Solver/ContactConstraintSimd.hpp> // SimdSolve range passes
#include <Manifold2D/Physics/Solver/Solver.hpp>                // SolverContext / ContactConstraint

namespace Manifold2D
{
    namespace Physics
    {
        class SoftStep; // forward decl -- only a pointer field references it here.

        // The seven stage kinds, in the D1 substep order they are walked. The five
        // colored types (WarmStart/Solve/Relax/Restitution/StoreImpulses) emit one
        // stage per non-empty color; the two body types (IntegrateVelocities/
        // IntegratePositions) emit one stage each over the awake-body range.
        enum class StageType : std::uint8_t
        {
            IntegrateVelocities,
            WarmStart,
            Solve,
            IntegratePositions,
            Relax,
            Restitution,
            StoreImpulses
        };

        // One claimable unit of a stage: a half-open [begin, end) sub-range (batch
        // indices for colored stages, awake-body indices for the body stages) plus
        // the atomic ring-CAS claim token. syncIndex starts at 0 and is CAS-advanced
        // from prevSync -> curSync each time the owning stage runs (it advances once
        // per substep re-visit), so a block is claimed by exactly one worker per run.
        //
        // The relaxed copy ctor/assign exist solely so the structs are vector-
        // storable (std::atomic is otherwise non-copyable, which would make
        // std::vector<SolverBlock>::resize ill-formed). Copies only ever happen
        // SINGLE-THREADED during the per-step scratch rebuild (never inside the
        // concurrent region), so a relaxed value copy is correct.
        struct SolverBlock
        {
            int              begin = 0;
            int              end = 0;
            std::atomic<int> syncIndex{ 0 };

            SolverBlock() noexcept = default;
            SolverBlock(const SolverBlock& o) noexcept
                : begin(o.begin), end(o.end),
                  syncIndex(o.syncIndex.load(std::memory_order_relaxed)) {}
            SolverBlock& operator=(const SolverBlock& o) noexcept
            {
                begin = o.begin;
                end = o.end;
                syncIndex.store(o.syncIndex.load(std::memory_order_relaxed),
                                std::memory_order_relaxed);
                return *this;
            }
        };

        // One stage = a run of body-disjoint blocks of a single type/color. blocks
        // points into the owning SoftStep::m_blocks scratch (a contiguous slice).
        // completionCount counts blocks finished THIS run (reset to 0 after the
        // barrier); syncIndex is the per-stage CAS round (main-advanced single-
        // threaded -- it increments each substep re-visit so the block CAS expects
        // prevSync == the value left by the previous run). Same relaxed-copy rationale
        // as SolverBlock.
        struct SolverStage
        {
            StageType        type = StageType::IntegrateVelocities;
            int              colorIndex = 0;   // active color (colored types); ignored otherwise
            SolverBlock*     blocks = nullptr; // -> m_blocks slice
            int              blockCount = 0;
            std::atomic<int> completionCount{ 0 };
            int              syncIndex = 0;    // per-stage CAS round (main-advanced)

            SolverStage() noexcept = default;
            SolverStage(const SolverStage& o) noexcept
                : type(o.type), colorIndex(o.colorIndex), blocks(o.blocks),
                  blockCount(o.blockCount),
                  completionCount(o.completionCount.load(std::memory_order_relaxed)),
                  syncIndex(o.syncIndex) {}
            SolverStage& operator=(const SolverStage& o) noexcept
            {
                type = o.type;
                colorIndex = o.colorIndex;
                blocks = o.blocks;
                blockCount = o.blockCount;
                completionCount.store(o.completionCount.load(std::memory_order_relaxed),
                                      std::memory_order_relaxed);
                syncIndex = o.syncIndex;
                return *this;
            }
        };

        // Per-step bundle the region reads. Built on the stack in SoftStep::Solve and
        // passed by reference to SolverWorker. The data handles let ExecuteBlock reach
        // the solver's per-color batches / dense body rows / refs without touching
        // SoftStep's definition; the FunctionRefs forward the body-integrate range +
        // the main-serial overflow/joint passes to SoftStep methods (bound as
        // lambdas in Solve). stageSync is the single "advance to next stage" word
        // (Box2D atomicSyncBits): the main publishes (syncIndex<<16)|stageIndex with
        // release on every stage advance and kSolverStageSyncTerminal at the end; the
        // thieves (workerIndex > 0) load it with acquire to find + steal the live stage
        // and to know when to exit.
        struct SolverStageContext
        {
            SolverContext* ctx = nullptr;
            SolverStage*   stages = nullptr;   // -> m_stages.data()
            int            substepCount = 0;
            int            activeColorStages = 0; // non-empty colors per colored phase
            std::atomic<std::uint32_t> stageSync{ 0 };

            // Data ExecuteBlock reads for the colored + store stages.
            std::vector<std::vector<ContactConstraintSimd>>* colorBatches = nullptr;
            std::vector<std::uint32_t>* colorRefs = nullptr; // -> m_colorRefs.data() (kColorCount vectors)
            BodyState*     bodyState = nullptr;
            float          h = 0.0f;
            float          maxBiasVel = 0.0f;
            float          threshold = 0.0f;

            // Body-integrate range callbacks (bound to SoftStep + ctx + h in Solve).
            Manifold2D::FunctionRef<void(std::size_t, std::size_t)> integrateVel{};
            Manifold2D::FunctionRef<void(std::size_t, std::size_t)> integratePos{};

            // Main-serial inline callbacks SolverWorker invokes between colored
            // phases (Box2D solves overflow + joints on the main worker while thieves
            // spin). overflowSolve takes useBias (true=biased solve, false=relax).
            Manifold2D::FunctionRef<void()>     overflowWarmStart{};
            Manifold2D::FunctionRef<void(bool)> overflowSolve{};
            Manifold2D::FunctionRef<void()>     overflowRestitution{};
            Manifold2D::FunctionRef<void()>     jointBridge{}; // Sync->Solve->Sync (no-op if no joints)
        };

        // -----------------------------------------------------------------------
        // Block sizing (Box2D b2ComputeBlockCount). Contact blocks min ~4 wide-
        // constraints (batches), body blocks min ~32, target ~= blocksPerWorker *
        // workerCount (Gap 1.2 sizes with the executor's real WorkerCount() so the
        // ring-CAS work-stealing has enough blocks to spread across the thieves;
        // the per-color/body partition is disjoint, so the block count -- hence the
        // worker count -- never changes any float: byte-identical at any width).
        // -----------------------------------------------------------------------
        inline int ComputeBlockCount(int itemCount, int minBlockSize, int targetBlocks) noexcept
        {
            if (itemCount <= 0) { return 0; }
            const int maxBlocks = (itemCount + minBlockSize - 1) / minBlockSize; // ceil
            int blocks = (targetBlocks < maxBlocks) ? targetBlocks : maxBlocks;
            return (blocks < 1) ? 1 : blocks;
        }

        // Partition [0, itemCount) into blockCount contiguous near-equal [begin,end)
        // ranges (remainder spread over the leading blocks) and reset each block's
        // syncIndex to 0 for a fresh step. blockCount must be > 0 and <= itemCount's
        // ComputeBlockCount result.
        inline void PartitionBlocks(SolverBlock* blocks, int blockCount, int itemCount) noexcept
        {
            const int base = itemCount / blockCount;
            const int rem = itemCount % blockCount;
            int start = 0;
            for (int b = 0; b < blockCount; ++b)
            {
                const int count = base + (b < rem ? 1 : 0);
                blocks[b].begin = start;
                blocks[b].end = start + count;
                blocks[b].syncIndex.store(0, std::memory_order_relaxed);
                start += count;
            }
        }

        // -----------------------------------------------------------------------
        // ExecuteBlock -- run one block of `stage` (dispatch on the stage type to the
        // matching SimdSolve range overload or the body-integrate callback).
        // -----------------------------------------------------------------------
        inline void ExecuteBlock(SolverStageContext& sc, const SolverStage& stage, int blockIndex)
        {
            const SolverBlock& block = stage.blocks[blockIndex];
            const std::size_t begin = static_cast<std::size_t>(block.begin);
            const std::size_t end = static_cast<std::size_t>(block.end);
            const std::size_t color = static_cast<std::size_t>(stage.colorIndex);

            switch (stage.type)
            {
            case StageType::IntegrateVelocities:
                sc.integrateVel(begin, end);
                break;
            case StageType::IntegratePositions:
                sc.integratePos(begin, end);
                break;
            case StageType::WarmStart:
                SimdSolve::WarmStart((*sc.colorBatches)[color], sc.bodyState, begin, end);
                break;
            case StageType::Solve:
                SimdSolve::SolveNormalAndFriction((*sc.colorBatches)[color], sc.bodyState,
                                                  sc.h, /*useBias=*/true, sc.maxBiasVel, begin, end);
                break;
            case StageType::Relax:
                SimdSolve::SolveNormalAndFriction((*sc.colorBatches)[color], sc.bodyState,
                                                  sc.h, /*useBias=*/false, sc.maxBiasVel, begin, end);
                break;
            case StageType::Restitution:
                SimdSolve::ApplyRestitution((*sc.colorBatches)[color], sc.bodyState,
                                            sc.threshold, begin, end);
                break;
            case StageType::StoreImpulses:
                SimdSolve::StoreImpulses((*sc.colorBatches)[color], sc.ctx->contacts,
                                         sc.colorRefs[color].data(), begin, end);
                break;
            }
        }

        // -----------------------------------------------------------------------
        // ExecuteStage -- the staggered ring-order CAS claim loop (Box2D
        // b2ExecuteStage). Each worker rings ONE full pass from `startIndex`,
        // CAS-claiming every block whose syncIndex == prevSync (prev -> cur). The CAS
        // winner runs the block via runBlock(blockIndex) and bumps the stage's
        // completionCount (release). `visited` advances every iteration, so the loop
        // is exactly one ring (blockCount visits): after it, every block has been
        // claimed by SOME worker. Returns the count this worker executed.
        //
        // The main rings from 0 and self-completes whatever the thieves did not grab;
        // each thief rings from its own workerIndex offset (Gap 1.2) so the claims
        // stagger instead of all hammering block 0.
        // -----------------------------------------------------------------------
        template <class RunBlock>
        inline int ExecuteStage(SolverStage& stage, int prevSync, int curSync,
                                RunBlock&& runBlock, int startIndex = 0) noexcept
        {
            const int blockCount = stage.blockCount;
            if (blockCount <= 0) { return 0; }

            int executed = 0;
            int visited = 0;
            int blockIndex = startIndex % blockCount;
            while (visited < blockCount)
            {
                int expected = prevSync;
                if (stage.blocks[blockIndex].syncIndex.load(std::memory_order_relaxed) == prevSync &&
                    stage.blocks[blockIndex].syncIndex.compare_exchange_strong(
                        expected, curSync,
                        std::memory_order_acquire, std::memory_order_relaxed))
                {
                    runBlock(blockIndex);
                    stage.completionCount.fetch_add(1, std::memory_order_release);
                    ++executed;
                }
                ++visited;
                ++blockIndex;
                if (blockIndex >= blockCount) { blockIndex = 0; }
            }
            return executed;
        }

        // Test-only shim: expose the ring-CAS claim loop with an injectable per-block
        // callback so the claim protocol is unit-tested in isolation (no SoftStep).
        template <class RunBlock>
        inline int ExecuteStageForTest(SolverStage& stage, int prevSync, int curSync,
                                       RunBlock&& runBlock) noexcept
        {
            return ExecuteStage(stage, prevSync, curSync, std::forward<RunBlock>(runBlock), 0);
        }

        // -----------------------------------------------------------------------
        // ExecuteMainStage -- the main worker drives one stage (Box2D
        // b2ExecuteMainStage): publish the stage-advance word, participate in
        // claiming/running blocks, then spin on the completion barrier and reset.
        // -----------------------------------------------------------------------
        inline void ExecuteMainStage(SolverStageContext& sc, int stageIndex)
        {
            SolverStage& stage = sc.stages[stageIndex];
            const int blockCount = stage.blockCount;
            if (blockCount <= 0) { return; } // empty stage (e.g. no awake bodies)

            const int prevSync = stage.syncIndex;
            const int curSync = prevSync + 1;

            // Single-store "advance to next stage" signal (syncIndex round + stage
            // index) for the thieves (Gap 1.2). Release so a thief that observes it
            // sees the stage setup (and, transitively, the prior stages' block writes).
            const std::uint32_t syncBits =
                (static_cast<std::uint32_t>(curSync) << 16) |
                (static_cast<std::uint32_t>(stageIndex) & 0xFFFFu);
            sc.stageSync.store(syncBits, std::memory_order_release);

            // The main participates in claiming + running blocks (ring from 0). It
            // self-completes every block a thief did not grab, so liveness holds for
            // any worker count (Gap 1.1: it grabs them all).
            ExecuteStage(stage, prevSync, curSync,
                         [&](int blockIndex) { ExecuteBlock(sc, stage, blockIndex); },
                         /*startIndex=*/0);

            // Barrier: wait for every block to finish (self + thieves), then reset
            // the per-run count and advance the per-stage CAS round.
            while (stage.completionCount.load(std::memory_order_acquire) != blockCount)
            {
                // spin (Gap 1.1: already complete -> exits immediately)
            }
            stage.completionCount.store(0, std::memory_order_relaxed);
            stage.syncIndex = curSync;
        }

        // Terminal stage-sync sentinel (Box2D's UINT_MAX). The main publishes this
        // after the LAST stage; a thief loop exits the moment it observes it. It is
        // unambiguous: a real publish is (curSync << 16) | stageIndex with both
        // halves small (curSync <= substepCount, stageIndex < stageCount), so it can
        // never equal 0xFFFFFFFF. This is liveness-critical -- a thief that never
        // exits would hang ParallelFor's join (it never returns from its partition).
        inline constexpr std::uint32_t kSolverStageSyncTerminal = 0xFFFFFFFFu;

        // -----------------------------------------------------------------------
        // SolverWorker -- the persistent region entry, dispatched once per step via
        // ParallelFor(workerCount, 1, ...) with worker index = the partition start
        // `begin` (Box2D b2SolverTask style).
        //
        // workerIndex == 0 -> the MAIN/orchestrator (the partition covering index 0
        //   always exists, so there is exactly one). It walks the flat stage list in
        //   D1's exact substep order, driving each colored/body stage via
        //   ExecuteMainStage (publish stageSync -> claim+run blocks -> spin the
        //   completion barrier -> reset+advance) and the overflow + joint passes
        //   main-serial inline between them, then publishes the terminal sentinel.
        //
        // workerIndex > 0 -> a THIEF. It spins reading sc.stageSync (acquire); when it
        //   names a live stage it runs that stage's ring-CAS claim loop (ExecuteStage)
        //   from its own worker offset, stealing whatever blocks the main has not yet
        //   self-completed. A thief NEVER advances a stage, NEVER runs overflow/joints
        //   (main-only), and NEVER resets completionCount -- it only CAS-claims blocks
        //   and release-bumps completionCount on each win. Re-reading the SAME syncBits
        //   is a harmless no-op (every block's CAS expects prevSync and is already at
        //   curSync), so the loop needs no last-seen tracking for correctness; the CAS
        //   protocol makes re-execution idempotent. The thief exits on the terminal
        //   sentinel. Because the main self-completes every block alone, a thief that
        //   starts late, runs nothing, or exits late only wastes spins -- it can never
        //   change a result or deadlock the join.
        //
        // Stage layout in sc.stages (C = activeColorStages):
        //   [0]            IntegrateVelocities
        //   [1 .. C]       WarmStart per active color
        //   [C+1 .. 2C]    Solve  per active color
        //   [2C+1]         IntegratePositions
        //   [2C+2 .. 3C+1] Relax  per active color
        //   [3C+2 .. 4C+1] Restitution   per active color (once)
        //   [4C+2 .. 5C+1] StoreImpulses per active color (once)
        // The substep-reused stages advance their own syncIndex each substep; the
        // restitution/store stages run once.
        // -----------------------------------------------------------------------
        inline void SolverWorker(SolverStageContext& sc, std::uint32_t workerIndex)
        {
            if (workerIndex != 0)
            {
                // ----- Thief (Gap 1.2) -----------------------------------------
                // Ring from this worker's own offset so concurrent thieves stagger
                // their CAS claims instead of all starting at block 0.
                const int startIndex = static_cast<int>(workerIndex);
                std::uint32_t syncBits;
                while ((syncBits = sc.stageSync.load(std::memory_order_acquire)) !=
                       kSolverStageSyncTerminal)
                {
                    // Decode (curSync<<16)|stageIndex. The pre-start value 0 decodes to
                    // stageIndex 0 / curSync 0 -> prevSync = -1, which no reset block
                    // (syncIndex == 0) ever matches, so a thief that wakes before the
                    // main's first publish just no-op-spins until a real stage appears.
                    const int stageIndex = static_cast<int>(syncBits & 0xFFFFu);
                    const int curSync    = static_cast<int>(syncBits >> 16);
                    SolverStage& stage = sc.stages[static_cast<std::size_t>(stageIndex)];
                    ExecuteStage(stage, curSync - 1, curSync,
                                 [&](int blockIndex) { ExecuteBlock(sc, stage, blockIndex); },
                                 startIndex);
                }
                return;
            }

            // ----- Main / orchestrator (workerIndex == 0) ----------------------
            const int C = sc.activeColorStages;
            for (int s = 0; s < sc.substepCount; ++s)
            {
                ExecuteMainStage(sc, 0); // IntegrateVelocities

                for (int c = 0; c < C; ++c) { ExecuteMainStage(sc, 1 + c); }          // WarmStart
                if (sc.overflowWarmStart) { sc.overflowWarmStart(); }
                if (sc.jointBridge) { sc.jointBridge(); }                              // joint pass #1

                for (int c = 0; c < C; ++c) { ExecuteMainStage(sc, 1 + C + c); }      // Solve (bias)
                if (sc.overflowSolve) { sc.overflowSolve(true); }

                ExecuteMainStage(sc, 1 + 2 * C);                                       // IntegratePositions
                if (sc.jointBridge) { sc.jointBridge(); }                              // joint pass #2

                for (int c = 0; c < C; ++c) { ExecuteMainStage(sc, 2 + 2 * C + c); }  // Relax (no bias)
                if (sc.overflowSolve) { sc.overflowSolve(false); }
            }

            for (int c = 0; c < C; ++c) { ExecuteMainStage(sc, 2 + 3 * C + c); }      // Restitution (once)
            if (sc.overflowRestitution) { sc.overflowRestitution(); }
            for (int c = 0; c < C; ++c) { ExecuteMainStage(sc, 2 + 4 * C + c); }      // StoreImpulses (once)

            // Liveness: tell every thief to exit so ParallelFor's join completes.
            // Release so a thief observing it has seen all of the region's writes.
            sc.stageSync.store(kSolverStageSyncTerminal, std::memory_order_release);
        }

    } // namespace Physics
} // namespace Manifold2D
