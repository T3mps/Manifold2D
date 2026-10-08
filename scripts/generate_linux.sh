#!/usr/bin/env bash
# Generate GNU makefiles (Linux / macOS) with the pinned, SHA-256-verified
# premake5 from scripts/fetch-premake.sh. scripts/build.sh does this + builds.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
"$("$root/scripts/fetch-premake.sh")" gmake
