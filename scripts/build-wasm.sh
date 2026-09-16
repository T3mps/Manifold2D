#!/usr/bin/env bash
# Build Manifold2D (+ the wasm bindings) to WebAssembly with em++.
#
# emsdk must be on PATH: source <emsdk>/emsdk_env.sh (or emsdk_env.bat on
# cmd/PowerShell) in this shell first. NOT wired into premake -- premake's
# Manifold2D project globs include/** + src/** only, so bindings/wasm/*.cpp is
# invisible to the MSVC build and this script is the SECOND build target.
#
#   scripts/build-wasm.sh                 library + bindings, linked to out/manifold.js
#   scripts/build-wasm.sh --no-bindings    compile the library TUs only (no link):
#                                          the toolchain proof, needs no bindings file
set -euo pipefail
shopt -s globstar

cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="$ROOT/out"
BINDINGS="$ROOT/bindings/wasm/ManifoldWasm.cpp"

NO_BINDINGS=0
if [ "${1:-}" = "--no-bindings" ]; then
    NO_BINDINGS=1
elif [ -n "${1:-}" ]; then
    echo "error: unknown argument '$1' (expected --no-bindings or nothing)" >&2
    exit 2
fi

if ! command -v em++ >/dev/null 2>&1; then
    echo "error: em++ is not on PATH. Run: source <emsdk>/emsdk_env.sh" >&2
    exit 1
fi
echo "emsdk: $(em++ --version 2>&1 | head -n 1)"

REV="$(git -C "$ROOT" rev-parse --short HEAD)"
echo "rev: $REV"

# -ffp-contract=off -fno-fast-math: the repo's determinism invariant (CLAUDE.md
#   "Determinism: /fp:strict (MSVC) / -ffp-contract=off -fno-fast-math (gcc/clang)").
# -DMOSAIC_SIMD_SCALAR: force Wide_Scalar.inl. wasm gets neither MOSAIC_HAS_AVX2
#   nor MOSAIC_HAS_NEON (Platform.hpp's SIMD block is gated on the x86/ARM arch
#   macros), so Wide.hpp's final #else would pick the scalar backend anyway --
#   the define makes it explicit and also suppresses the <immintrin.h> include in
#   ContactConstraintSimd.hpp:90.
# -DNDEBUG: disables <cassert> and, per Mosaic/Assert.hpp:192-198
#   (MOSAIC_ASSERTS_ACTIVE 0 when NDEBUG is set and neither MOSAIC_DEBUG nor
#   MOSAIC_ENABLE_ASSERTS is), every MOSAIC_ASSERT. It also drops the debug-only
#   coloring validator in SoftStep::Solve. -DMOSAIC_DISABLE_ASSERTS forces the
#   same off-state unconditionally (belt and suspenders).
# RTTI stays ON (no -fno-rtti): PhysicsWorld.hpp:716's FixtureBroadphaseTree()
#   uses an unguarded dynamic_cast.
# No -fexceptions: the only throw in the library is MakePolygon's
#   std::invalid_argument (src/Physics/Shapes.cpp:332) and the bindings validate
#   the vertex count themselves, so it is unreachable from JS.
CXXFLAGS=(
    -std=c++23 -O2
    -ffp-contract=off -fno-fast-math
    -DNDEBUG -DMOSAIC_DISABLE_ASSERTS -DMANIFOLD2D_RELEASE -DMOSAIC_SIMD_SCALAR
    "-DMANIFOLD_WASM_REV=\"$REV\""
    -I"$ROOT/include" -I"$ROOT/ThirdParty/Mosaic/include"
)

SRCS=( src/**/*.cpp )
echo "sources: ${#SRCS[@]} translation units"

mkdir -p "$OUT"

if [ "$NO_BINDINGS" = "1" ]; then
    mkdir -p "$OUT/obj"
    for f in "${SRCS[@]}"; do
        o="$OUT/obj/$(printf '%s' "${f%.cpp}" | tr '/' '_').o"
        em++ "${CXXFLAGS[@]}" -c "$f" -o "$o"
    done
    echo "compile-only OK: ${#SRCS[@]} translation units -> $OUT/obj"
    exit 0
fi

if [ ! -f "$BINDINGS" ]; then
    echo "error: $BINDINGS not found. Add the bindings (Task 4) or pass --no-bindings." >&2
    exit 1
fi

em++ "${CXXFLAGS[@]}" --bind --no-entry \
    "${SRCS[@]}" "$BINDINGS" \
    -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createManifold \
    -sENVIRONMENT=web -sALLOW_MEMORY_GROWTH=1 -sFILESYSTEM=0 \
    -o "$OUT/manifold.js"

printf '%s\n' "$REV" > "$OUT/manifold.rev"
# Build stamp INSIDE the JS glue. embind registers MANIFOLD_WASM_REV from a
# string in the wasm data segment, so the literal does NOT appear in
# manifold.js; the web repo's verify-deploy greps this trailing comment to
# check manifold.rev against the shipped JS without instantiating the module.
printf '\n// MANIFOLD_WASM_REV=%s\n' "$REV" >> "$OUT/manifold.js"

raw=$(wc -c < "$OUT/manifold.wasm")
gz=$(gzip -9 -c "$OUT/manifold.wasm" | wc -c)
echo "manifold.wasm: $raw bytes raw, $gz bytes gzipped"
limit=$((400 * 1024))
if [ "$gz" -gt "$limit" ]; then
    echo "error: gzipped wasm is $gz bytes, over the $limit byte (400 KB) budget" >&2
    exit 1
fi
echo "wasm build OK: $OUT/manifold.js + manifold.wasm + manifold.rev (rev $REV)"
