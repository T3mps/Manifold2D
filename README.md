# Manifold2D

Manifold2D is Starworks' standalone 2D physics + geometry library -- one of
three independently-usable components in the Starworks stack (Astra = ECS,
Manifold2D = 2D physics, Arcane = full engine). A consumer that wants only 2D
physics can take Manifold2D on its own, without Arcane. The library has **zero
external dependencies** (standard library only) and is modeled on Box2D v3:
a compacted, graph-colored soft-step solver with persistent islands, a
DynamicTree / SpatialGrid broadphase, GJK/EPA/MPR + SAT narrowphase, CCD, and
an exact-predicate geometry kernel.

## Layout

```
include/Manifold2D/        # public headers (the include root)
  Core/                    # namespace Manifold2D:: primitives
                           #   FunctionRef, BitSet, Simd (+ Scalar/AVX2/NEON .inl),
                           #   WorkScheduler (the IWorkScheduler seam), SimdSmoke
  Physics/                 # namespace Manifold2D::Physics -- solver, broadphase,
                           #   narrowphase, joints, islands, contacts, queries, CCD
  Geometry/                # namespace Manifold2D::Geometry -- exact-predicate
                           #   convex-hull / orientation kernel (header-only)
src/                       # implementation .cpp (NOT on the public include path)
  Physics/**               # solver + collision .cpp
  Version.cpp  SimdSmoke.cpp
tests/                     # Catch2 + rapidcheck suites (the physics test gate)
  Support/TestWorkScheduler.hpp   # std::thread IWorkScheduler for the MT tests
scripts/                   # generate + sync + vendor helpers
vendor/                    # Catch2, rapidcheck, premake5 (self-contained build)
```

Headers are included as `#include <Manifold2D/Physics/PhysicsWorld.hpp>`.

## Threading -- an injected seam

Manifold2D **creates no threads**. Data-parallel work goes through
`Manifold2D::IWorkScheduler`, a `ParallelFor`-only interface the host supplies
an adapter for (e.g. an enkiTS- or std::thread-backed pool). When no scheduler
is injected, `Manifold2D::SerialWorkScheduler` runs each range inline as worker
0 -- the deterministic default. `ParallelFor(count, minBatch, fn)` guarantees
each concurrently-running sub-range a distinct `worker` id in
`[0, WorkerCount())`, so the solver-MT and broadphase-MT paths index per-worker
scratch without locking.

## Units

Manifold2D is authored in **MKS** (meters / kilograms / seconds). Bodies are
0.1--10 m, default gravity is `(0, 10)` (y-down), velocities are m/s. Never
author pixel-scale content; map world -> screen at the display layer.

## Build + test

```bash
scripts/build.sh Release && scripts/run-tests.sh Release --rng-seed 1      # Linux / macOS
```

```powershell
scripts/build.ps1 Release; scripts/run-tests.ps1 Release --rng-seed 1      # Windows
```

Configurations: `Debug` / `Release` / `Dist`. Windows uses the vendored
`premake5.exe` and the newest Visual Studio; Linux / macOS fetch the same
premake version (SHA-256 pinned) and build with `gmake`. CI covers Windows
(MSVC), Linux (GCC 14, Clang 19) and macOS (Apple Clang, arm64) and diffs a
cross-platform determinism fixture across all of them -- see
[docs/ci.md](docs/ci.md).

## License

MIT -- see [LICENSE](LICENSE).
