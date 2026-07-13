#!/usr/bin/env bash
# Generate Linux gmake2 makefiles. Part B scaffold -- the cross-platform
# burn-down (g++/clang green + Linux CI) is a follow-up; this target exists so
# that work is a drop-in. Requires a Linux premake5 on PATH (the vendored
# premake5.exe is Windows-only).
set -euo pipefail
premake5 gmake2
