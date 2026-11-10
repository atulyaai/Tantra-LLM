[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'High')]
param(
    [ValidateSet('artifacts', 'fresh-github')]
    [string]$Mode = 'artifacts'
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root

function Remove-LocalArtifact {
    param([string]$RelativePath)
    $target = Join-Path $root $RelativePath
    if (Test-Path -LiteralPath $target) {
        # $PSCmdlet is unreliable inside a plain function scope (it can throw
        # NullReferenceException), so let WhatIfPreference propagate instead.
        Remove-Item -LiteralPath $target -Recurse -Force -WhatIf:$WhatIfPreference
    }
}

if ($Mode -eq 'artifacts') {
    # These paths are generated locally and are intentionally not source code.
    # Do not add source/config/tests here: restore those through Git instead.
    @('runs', 'Datasets', 'Model/checkpoints', 'Model/downloads', '.pytest_cache') |
        ForEach-Object { Remove-LocalArtifact $_ }
    # Never walk the virtual environment or Git metadata.  They are managed by
    # Python/Git, respectively, and are outside the project artifact contract.
    $venvRoot = Join-Path $root '.venv'
    $gitRoot = Join-Path $root '.git'
    Get-ChildItem -LiteralPath $root -Recurse -Directory -Force -Filter '__pycache__' |
        Where-Object {
            -not $_.FullName.StartsWith($venvRoot, [System.StringComparison]::OrdinalIgnoreCase) -and
            -not $_.FullName.StartsWith($gitRoot, [System.StringComparison]::OrdinalIgnoreCase)
        } |
        ForEach-Object {
            Remove-Item -LiteralPath $_.FullName -Recurse -Force -WhatIf:$WhatIfPreference
        }
    Write-Output 'Local artifacts cleaned. Source, documentation, tests, and tracked tokenizer files were preserved.'
    exit 0
}

# This is the complete reset used for a fresh download. It removes every local
# change and ignored artifact inside this repository, then restores origin/main.
if ($WhatIfPreference) {
    Write-Output ('What if: resetting tracked files of {0} to origin/main and removing all ignored local files.' -f $root)
    exit 0
}
git fetch --prune origin
if ($LASTEXITCODE -ne 0) { throw 'git fetch failed' }
git reset --hard origin/main
if ($LASTEXITCODE -ne 0) { throw 'git reset failed' }
git clean -ffdx
if ($LASTEXITCODE -ne 0) { throw 'git clean failed' }
Write-Output 'Fresh GitHub checkout complete.'
