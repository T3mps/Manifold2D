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

## Build + test (Windows)

```bat
scripts\generate_vs2022.bat                            :: premake5 vs2022 -> Manifold2D.sln
msbuild Manifold2D.sln /p:Configuration=Release /m
bin\Release-windows-x86_64\Manifold2DTests\Manifold2DTests.exe
```

Configurations: `Debug` / `Release` / `Dist`. The build is self-contained --
the vendored `premake5.exe` and test deps mean no other repo is required.

Linux (`scripts/generate_linux.sh` -> `gmake2`) is scaffolded but not yet
green; the cross-platform burn-down is a follow-up.

## License

MIT -- see [LICENSE](LICENSE).
