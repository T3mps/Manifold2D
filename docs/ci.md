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

`tests/CrossPlatformDeterminismTest.cpp` adds that. It steps fixed scenes and
folds every body's raw float bits (position, angle, linear + angular velocity)
into an FNV-1a hash per step, asserts the in-process identities, and writes to
`$MANIFOLD2D_DETERMINISM_OUT`:

- `scene <name> <class> <hash>` -- the final hash;
- `check <name> <step> <hash>` -- the running hash at checkpoint steps, so a
  divergence is pinned to the first step where legs part company;
- `state ...` -- final positions/angles as hex floats (gives the magnitude);
- `probe <name> <hash>` -- the leg's **trig fingerprint**: `sin`/`cos`/`sqrt`
  of 12 angles read through `volatile` (`runtime`) and as literals the
  optimizer may fold (`folded`), plus 2^18-angle sweeps of `sin`, of `cos`,
  and of `sin`+`cos` on the same argument (`sincos-sweep`; an optimizer may
  merge that pair into one sincos call, e.g. Darwin's `__sincosf_stret`).

`scripts/compare-determinism.py` (the `cross-platform-determinism` job)
enforces:

| Class      | Scenes                               | Requirement |
|------------|--------------------------------------|-------------|
| `trigfree` | `pile-tree`, `pile-hash`, `pile-sap` | bit-identical on **every** leg: OS, compiler, config, seed |
| `trig`     | `rotating-mixed`                     | bit-identical between legs with the **same trig fingerprint**; a difference between legs whose fingerprints differ is reported with its magnitude and first diverging step, not failed |

Why the split: the `trigfree` scenes use only fixedRotation bodies, so the
only transcendentals evaluated are `sin/cos(0)` (exact everywhere); the rest
is IEEE-754 `+ - * / sqrt` and explicit `fma`, correctly rounded and -- with
contraction and fast-math banned -- identical on MSVC, GCC, Clang and Apple
Clang, AVX2 and NEON. Free rotation feeds `std::sin`/`std::cos` of arbitrary
angles back into the state (`PhysicsTypes.hpp`, `PhysicsWorld.cpp`,
`Joints.cpp`, narrowphase transforms). C++ does not require those to be
correctly rounded, and *which* implementation runs depends on the platform
libm and on the optimizer, so a 1-ulp difference is legitimate between legs
whose fingerprint differs -- and the soft-step solver amplifies it. Legs whose
fingerprint is identical evaluate identical trig on every sampled input, so
any divergence between them is a real determinism bug and fails the job.
Making rotating scenes bit-identical everywhere would need the library to ship
its own sin/cos (a behaviour change, out of scope for CI work).

### Measured verdict

Measured on the PR that introduced this check (8 legs x 2 seeds = 16 dumps):

- `trigfree` (`pile-tree`, `pile-hash`, `pile-sap`): **bit-identical on all
  16 dumps** -- Windows/MSVC, Linux/GCC 14, Linux/Clang 19, macOS/Apple Clang
  arm64, Debug and Release, both seeds -- and identical across the three
  broadphases (`20539a820ac0dc3c`).
- `rotating-mixed` (`trig`): bit-identical on all 16 dumps through step 30.
  By step 60 it splits into {all Linux legs, macOS Debug} and {all Windows
  legs, macOS Release}; the final states differ by up to ~1e-3 m (Linux vs
  macOS) and ~0.73 m (vs Windows) after 300 steps. The 12-angle probes are
  identical everywhere, which is why the dense sweeps were added; the job
  summary of the latest run attributes each group to its fingerprint.
- Every leg is identical across its two seeds, and GCC 14 == Clang 19 on
  every scene.

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
2. **`trig` scenes are gated on the trig fingerprint, not on "all legs"** --
   platform `sin/cos` may differ by an ulp (see [Determinism](#determinism));
   they must match between legs with the same fingerprint, and any other
   difference is reported with its magnitude and first diverging step.
3. **Windows generator is chosen at run time** (`vs2022` or `vs2026`) to honour
   "newest VS on the image" without editing the workflow when the image moves.
4. **`Dist` not in the matrix** -- identical code generation to `Release` minus
   symbols; it still builds through the same scripts.
