# CLAUDE.md

Guidance for Claude Code when working in the Manifold2D repository.

## What this is

Manifold2D is Starworks' standalone, dependency-free 2D physics + geometry
library (Box2D v3 model). It is developed here and vendored into consumers
(e.g. the Arcane engine) as a synced source copy -- there is no submodule.

## Layout + namespaces

- `include/Manifold2D/Core/**` -- flat `namespace Manifold2D` primitives:
  `FunctionRef`, `BitSet`, `Simd` (+ `Scalar`/`Avx2`/`Neon` `.inl`),
  `WorkScheduler` (the `IWorkScheduler` seam), `SimdSmoke`.
- `include/Manifold2D/Physics/**` -- `namespace Manifold2D::Physics` (solver,
  broadphase, narrowphase, joints, islands, contacts, queries, CCD).
- `include/Manifold2D/Geometry/**` -- `namespace Manifold2D::Geometry`
  (exact-predicate convex hull / orientation kernel; header-only).
- `src/**` -- implementation `.cpp`, NOT on the public include path; reach
  headers via `<Manifold2D/...>`.
- `tests/**` -- Catch2 + rapidcheck. `tests/Support/TestWorkScheduler.hpp` is a
  test-only `std::thread` `IWorkScheduler` for the MT-invariance suites.
- `vendor/{Catch2,rapidcheck,premake5}` -- self-contained build; no external repo.

## Invariants

- **Zero external dependencies** -- standard library only. Do not add includes
  of any `Arcane/`, third-party, or system-specific header into `include/` or
  `src/`. `rg 'Arcane/' include src` must be empty.
- **The library creates no threads.** All data-parallel work goes through the
  injected `Manifold2D::IWorkScheduler`; `SerialWorkScheduler` is the default.
- **Determinism:** `/fp:strict` (MSVC) / `-ffp-contract=off -fno-fast-math`
  (gcc/clang). No `/fp:fast`. UTF-8 without BOM, ASCII comments.
- **Units are MKS** (meters/kg/seconds); default gravity `(0, 10)` y-down.

## Build + test (Windows)

```bat
scripts\generate_vs2022.bat
msbuild Manifold2D.sln /p:Configuration=Debug /m
bin\Debug-windows-x86_64\Manifold2DTests\Manifold2DTests.exe
```

Configurations: Debug / Release / Dist. Linux (`gmake2`) is scaffolded, not yet
green.
