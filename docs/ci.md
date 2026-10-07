# CI

Manifold2D follows the shared Starworks CI standard (identical in Arcane,
Astra and Manifold2D). This file uses the standard's section order:
Overview, Matrix, Entry points, Toolchains, Seeds and test results,
Determinism, Supply chain, Reproducing locally, Deviations.

## Overview

One workflow, `.github/workflows/ci.yml`, triggered on pushes to `main`, on
every pull request and on `workflow_dispatch`. It has two jobs:

- `build-test` -- the OS x compiler x configuration matrix. Every leg runs the
  same two entry points (build, then tests with two seeds) and uploads
  `test-results/` as an artifact named `<leg>-<Config>`.
- `cross-platform-determinism` -- downloads every leg's artifact and runs
  `scripts/compare-determinism.py` over the determinism dumps (see
  [Determinism](#determinism)). It runs even when a leg failed, and fails if
  fewer than 8 legs produced dumps.

`permissions: contents: read`; one concurrent run per ref (a newer push
cancels the older run).

## Matrix

| Leg                | Image          | Compiler                              | Generator          |
|--------------------|----------------|---------------------------------------|--------------------|
| `windows-msvc`     | `windows-2025` | MSVC from the newest VS on the image  | premake5 `vs2022`/`vs2026` + MSBuild |
| `linux-gcc14`      | `ubuntu-24.04` | `g++-14`                              | premake5 `gmake` + make |
| `linux-clang19`    | `ubuntu-24.04` | `clang++-19` (apt if not preinstalled) | premake5 `gmake` + make |
| `macos-appleclang` | `macos-15`     | Apple Clang from Xcode `16.4` (`xcode-select`) | premake5 `gmake` + make |

x `Debug`, `Release` = 8 legs. Images are pinned; never `*-latest`.
`Dist` builds through the same scripts but is not in the matrix (it is
`Release` without symbols).

## Entry points

One script per shell, identical arguments on every OS:

```
scripts/build.ps1     <Debug|Release|Dist>                  # Windows (pwsh)
scripts/build.sh      <Debug|Release|Dist>                  # Linux / macOS (bash)
scripts/run-tests.ps1 <Debug|Release|Dist> [--rng-seed N]
scripts/run-tests.sh  <Debug|Release|Dist> [--rng-seed N]
```

- `build.ps1` verifies the vendored `vendor/premake5/premake5.exe` (SHA-256),
  finds the newest Visual Studio with `vswhere`, generates `vs2022` (VS 17) or
  `vs2026` (VS 18) and builds the solution with that MSBuild.
- `build.sh` fetches premake via `scripts/fetch-premake.sh`, generates `gmake`
  (with `--cc=clang` when `$CXX` is clang) and runs `make config=<config>`.
  The compiler comes from `CC`/`CXX` (CI sets them per leg).
- Both build scripts fail the build if a generated project/makefile carries
  fast-math or FP contraction, or if the library/tests lose `/fp:strict`
  (`-ffp-contract=off` on gcc/clang).
- `run-tests.*` locate `bin/<Config>-*/Manifold2DTests/Manifold2DTests[.exe]`
  and run it with the seed, writing JUnit and the determinism dump.

## Toolchains

- **Floating point**: `floatingpoint "Strict"` (MSVC `/fp:strict`) on the
  library and tests; gcc/clang get `-ffp-contract=off -fno-fast-math`. No
  `/fp:fast`, `-ffast-math` or `-Ofast` anywhere (enforced by the build
  scripts). Apple clang defaults to `-ffp-contract=on`, so the explicit `off`
  matters there.
- **ISA**: x64 builds use AVX2 + FMA3 (`/arch:AVX2`; `-mavx2 -mfma`, the
  gcc/clang equivalent -- `-mavx2` alone does not enable the FMA intrinsics
  `Wide_AVX2.inl` uses). macOS builds arm64 (`architecture "ARM64"`, NEON
  backend). Explicit fused ops (`std::fma`, Simd `mul_add`) are fused on every
  backend, so the FMA choice is not a determinism hazard.
- **C++23**, static CRT on Windows.

## Seeds and test results

Each leg runs the full suite twice: `--rng-seed 1` and `--rng-seed 24301`.
The seed is passed to Catch2 (`--rng-seed`) and to rapidcheck
(`RC_PARAMS=seed=N`), so every property-based failure is reproducible from the
log. The seeds are fixed (not run-derived) so a red run reproduces bit-for-bit
locally; rotating them is a one-line change in `ci.yml` (`SEED_A`/`SEED_B`).

Outputs, per run, under `test-results/`:

- `junit-<Config>-seed<N>.xml` -- Catch2 JUnit reporter.
- `determinism-<Config>-seed<N>.txt` -- the cross-platform determinism dump.

## Determinism

The pre-existing determinism fixtures (`PhysicsDeterminismTest.cpp`, the P2.6
harness, the solver/broadphase/narrowphase MT-invariance suites) assert
run-twice and serial-vs-MT identity **inside one binary**, so they pass on any
platform by construction and say nothing about cross-platform identity.

`tests/CrossPlatformDeterminismTest.cpp` adds that: it steps fixed scenes and
folds every body's raw float bits (position, angle, linear + angular velocity)
into an FNV-1a hash per step, asserts the in-process identities, and appends
`scene <name> <class> <hash>` (+ final states as hex floats) to
`$MANIFOLD2D_DETERMINISM_OUT`. The CI job then requires:

| Class      | Scenes                                   | Requirement |
|------------|------------------------------------------|-------------|
| `trigfree` | `pile-tree`, `pile-hash`, `pile-sap`     | bit-identical on **all** legs: every OS, compiler, config, seed |
| `trig`     | `rotating-mixed`                         | bit-identical **within each OS**; a difference **between** OSes is reported with its magnitude but does not fail |

Why the split: the `trigfree` scenes use only fixedRotation bodies, so the
only transcendentals evaluated are `sin/cos(0)` (exact everywhere); the rest
is IEEE-754 `+ - * / sqrt` and explicit `fma`, which are correctly rounded and
-- with contraction and fast-math banned -- identical on MSVC, GCC, Clang and
Apple Clang, AVX2 and NEON. Free rotation calls `std::sin`/`std::cos` on
arbitrary angles (`PhysicsTypes.hpp`, `PhysicsWorld.cpp`, `Joints.cpp`,
narrowphase transforms). Those go to the platform libm (MSVC UCRT, glibc,
Apple libm), which C++ does not require to be correctly rounded, so the last
bit can differ between OSes and the soft-step solver amplifies it. All Linux
legs share glibc, so GCC and Clang must still agree exactly. Making rotating
scenes bit-identical across OSes would need the library to ship its own
sin/cos (a behaviour change, out of scope for CI work).

Measured results: see the latest `cross-platform-determinism` job summary;
the verdict at the time of introduction is recorded in the PR that added this
file.

## Supply chain

- Actions pinned by full commit SHA (version in a trailing comment):
  `actions/checkout` v7.0.1, `actions/upload-artifact` v7.0.2,
  `actions/download-artifact` v8.0.2. Checkout does not persist credentials.
- premake **5.0.0-beta8** everywhere:
  - Windows: vendored `vendor/premake5/premake5.exe`, byte-identical to the
    upstream `premake-5.0.0-beta8-windows.zip` binary, SHA-256
    `2301e3e23ff3074cb83a5ea6103d68c7ea81dad56b786807c84b0643cddea31b`
    (checked by `build.ps1`).
  - Linux: `premake-5.0.0-beta8-linux.tar.gz`, SHA-256
    `63edd3e7461eebdd45b500a3c7e8ad4e7a67d68f230010f9a97cbb71b4ec59c8`.
  - macOS: `premake-5.0.0-beta8-macosx.tar.gz` (arm64), SHA-256
    `fa73a46f093fa6f17494a3d063421aa6cae3ea825a61c62dd59fc2f07a256d03`.
  `scripts/fetch-premake.sh` downloads from the GitHub release, verifies the
  hash before extracting, and caches in `.tools/` (gitignored).
- Clang 19 comes from the Ubuntu 24.04 archive (signed apt) only when the
  image lacks it.

## Reproducing locally

```bash
CC=gcc-14   CXX=g++-14     scripts/build.sh Release && scripts/run-tests.sh Release --rng-seed 1
CC=clang-19 CXX=clang++-19 scripts/build.sh Release && scripts/run-tests.sh Release --rng-seed 24301
```

```powershell
scripts/build.ps1 Release; scripts/run-tests.ps1 Release --rng-seed 1
```

Switching compiler in one checkout: `rm -rf bin bin-int` first (object dirs
are per config, not per compiler). Compare dumps from several machines with
`python3 scripts/compare-determinism.py <dir-of-leg-dirs>`.

## Deviations

From the shared standard, with reasons:

1. **Extra job `cross-platform-determinism`.** The standard is one matrix; the
   determinism diff needs every leg's output, so it is a dependent job in the
   same workflow (still one `ci.yml`).
2. **`trig` scenes are not required to match across OSes** -- libm `sin/cos`
   differ (see [Determinism](#determinism)); they are required to match
   within an OS and the cross-OS delta is reported.
3. **Windows generator is chosen at run time** (`vs2022` or `vs2026`) to honour
   "newest VS on the image" without editing the workflow when the image moves.
4. **`Dist` not in the matrix** -- identical code generation to `Release` minus
   symbols; it still builds through the same scripts.
