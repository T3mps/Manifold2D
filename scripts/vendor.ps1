# Vendor the standalone Manifold2D into a consumer repo (e.g. Aphelyon) as a
# faithful source mirror, like ThirdParty/Astra (which carries its own tests/).
# Excludes only .git/.github, build outputs, and the standalone test-dep vendor/
# tree (the consumer supplies its own Catch2/rapidcheck). The vendored
# premake5.lua is a standalone workspace and is INERT in the consumer -- the
# consumer builds Manifold2D via its own inline project (see Arcane's premake).
#
# Usage: .\scripts\vendor.ps1 [-Aphelyon <path-to-consumer-repo-root>]
param([string]$Aphelyon = "D:\dev\starworks\Gacha")
$ErrorActionPreference = "Stop"
$src = "D:\dev\starworks\Manifold2D"
$dst = "$Aphelyon\ThirdParty\Manifold2D"

robocopy $src $dst /MIR /NFL /NDL /NJH `
    /XD .git .github bin bin-int ide ide-md .vs vendor `
    /XF *.user *.slnx *.sln

if ($LASTEXITCODE -ge 8) { Write-Error "robocopy failed ($LASTEXITCODE)"; exit 1 }
Write-Host "Vendored $src -> $dst"
exit 0
