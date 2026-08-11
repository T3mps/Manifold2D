# Vendor the standalone Manifold2D into a consumer repo as a SOURCE-ONLY
# mirror: include/ + src/ + LICENSE. Nothing else -- the consumer owns its
# ThirdParty/Manifold2D/premake5.lua (a project-only wrapper, e.g. Arcane's
# enkiTS-style static-lib project), its own Mosaic at ThirdParty/Mosaic, and
# its own test deps. The pre-2026-08 /MIR full-tree sync is retired: it
# mirrored the standalone workspace premake5.lua + scripts/ over the
# consumer's wrapper, which the 2026-07-20 wrapper decision forbids.
#
# Usage: .\scripts\vendor.ps1 [-Consumer <path-to-consumer-repo-root>]
param([string]$Consumer = "D:\dev\starworks\Arcane")
$ErrorActionPreference = "Stop"
$src = Split-Path -Parent $PSScriptRoot   # scripts/ -> standalone repo root
$dst = "$Consumer\ThirdParty\Manifold2D"

if (-not (Test-Path "$src\include\Manifold2D")) {
    throw "Standalone Manifold2D not found at '$src' (no include\Manifold2D)."
}
if (-not (Test-Path "$dst\premake5.lua")) {
    throw "Consumer wrapper not found at '$dst\premake5.lua' -- refusing to vendor into a directory that lacks the consumer's own premake wrapper."
}

# Source dirs mirror WITH orphan deletion (a header removed upstream must
# disappear downstream too); everything outside include/ + src/ is untouched.
robocopy "$src\include" "$dst\include" /MIR /NFL /NDL /NJH
if ($LASTEXITCODE -ge 8) { Write-Error "robocopy include failed ($LASTEXITCODE)"; exit 1 }
robocopy "$src\src" "$dst\src" /MIR /NFL /NDL /NJH
if ($LASTEXITCODE -ge 8) { Write-Error "robocopy src failed ($LASTEXITCODE)"; exit 1 }
Copy-Item "$src\LICENSE" "$dst\LICENSE" -Force

# Provenance stamp (anti-drift, same mechanism as the Astra sync).
$commit = (git -C $src rev-parse HEAD 2>$null)
if (-not $commit) { $commit = "(source not a git checkout)" }
"Vendored from github.com/T3mps/Manifold2D @ $commit on $(Get-Date -Format s) -- include/ + src/ + LICENSE only; the premake5.lua here is the CONSUMER's wrapper, never synced." |
    Out-File -Encoding ascii "$dst\VENDORED.txt"

Write-Host "Vendored $src {include,src,LICENSE} -> $dst (wrapper premake untouched)"
exit 0
