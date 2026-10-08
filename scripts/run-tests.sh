#!/usr/bin/env bash
# Run the Manifold2D test suite built by scripts/build.sh.
#   scripts/run-tests.sh <Debug|Release|Dist> [--rng-seed N]
# Seeds both Catch2 (--rng-seed) and rapidcheck (RC_PARAMS seed=N); default 1.
# Writes test-results/junit-<Config>-seed<N>.xml and, via the cross-platform
# determinism fixture, test-results/determinism-<Config>-seed<N>.txt.
set -euo pipefail

config="${1:-}"
case "$config" in
    Debug|Release|Dist) ;;
    *) echo "usage: $0 <Debug|Release|Dist> [--rng-seed N]" >&2; exit 2 ;;
esac
shift
seed=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rng-seed) seed="${2:?--rng-seed needs a value}"; shift 2 ;;
        *) echo "run-tests.sh: unknown argument $1" >&2; exit 2 ;;
    esac
done
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "run-tests.sh: seed must be a non-negative integer" >&2; exit 2; }

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

shopt -s nullglob
exes=(bin/"$config"-*/Manifold2DTests/Manifold2DTests)
if [[ ${#exes[@]} -ne 1 ]]; then
    echo "run-tests.sh: expected exactly one Manifold2DTests under bin/$config-*, found ${#exes[@]} (run scripts/build.sh $config)" >&2
    exit 1
fi

mkdir -p test-results
junit="test-results/junit-$config-seed$seed.xml"
det="test-results/determinism-$config-seed$seed.txt"
rm -f "$det"

export RC_PARAMS="seed=$seed"
export MANIFOLD2D_DETERMINISM_OUT="$root/$det"
echo "== ${exes[0]} --rng-seed $seed (RC_PARAMS=$RC_PARAMS)"
"${exes[0]}" --rng-seed "$seed" --reporter console --reporter "JUnit::out=$junit"
