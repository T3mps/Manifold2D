#!/usr/bin/env bash
# Build Manifold2D + Manifold2DTests on Linux / macOS.
#   scripts/build.sh <Debug|Release|Dist>
# Toolchain comes from CC / CXX (defaults: gcc/g++ on Linux, clang/clang++ on
# macOS). Generator: premake5 gmake, pinned + SHA-256 checked by fetch-premake.sh.
set -euo pipefail

config="${1:-}"
case "$config" in
    Debug|Release|Dist) ;;
    *) echo "usage: $0 <Debug|Release|Dist>" >&2; exit 2 ;;
esac

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

premake="$("$root/scripts/fetch-premake.sh")"

if [[ "$(uname -s)" == "Darwin" ]]; then
    export CC="${CC:-clang}" CXX="${CXX:-clang++}"
    jobs="$(sysctl -n hw.ncpu)"
else
    export CC="${CC:-gcc}" CXX="${CXX:-g++}"
    jobs="$(nproc)"
fi

# Tell premake which flag dialect to emit (clang rejects some gcc-only flags).
if "$CXX" --version 2>/dev/null | grep -qi clang; then cc_family=clang; else cc_family=gcc; fi

echo "== premake5 $("$premake" --version | awk '{print $NF}') gmake (--cc=$cc_family)"
echo "== $("$CXX" --version | head -n1)"
"$premake" --cc="$cc_family" gmake

# Determinism guard: no fast-math / FP contraction may reach any generated makefile.
shopt -s nullglob
makefiles=(Makefile ide/*.make vendor/*/ide/*.make)
if grep -nE -- '-ffast-math|-Ofast|-ffp-contract=(fast|on)|-funsafe-math-optimizations' "${makefiles[@]}"; then
    echo "build.sh: fast-math / FP-contraction flag in generated makefiles (banned, see docs/ci.md)" >&2
    exit 1
fi
if ! grep -q -- '-ffp-contract=off' ide/Manifold2D.make; then
    echo "build.sh: -ffp-contract=off missing from ide/Manifold2D.make" >&2
    exit 1
fi

lower="$(echo "$config" | tr '[:upper:]' '[:lower:]')"
make -j"$jobs" config="$lower"
