<#
.SYNOPSIS
Starts Tantra Studio if needed, then opens it in the default browser.

This script is safe to run repeatedly: it reuses an existing server on port
8000 instead of starting a second process.
#>

$repoRoot = Split-Path -Parent $PSScriptRoot
$port = 8000
$url = "http://127.0.0.1:$port/"

function Get-TantraPython {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) { return $pythonCommand.Source }

    $pyCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCommand) { return $pyCommand.Source }

    $codexPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if (Test-Path -LiteralPath $codexPython) { return $codexPython }

    throw "Python was not found. Install Python 3.10+ or create a project virtual environment."
}

if (-not (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)) {
    $python = Get-TantraPython
    $logDir = Join-Path $repoRoot "logs"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    Start-Process -FilePath $python -ArgumentList "server.py" -WorkingDirectory $repoRoot -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logDir "webui.out.log") `
        -RedirectStandardError (Join-Path $logDir "webui.err.log")

    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $deadline) {
        if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { break }
        Start-Sleep -Milliseconds 500
    }
}

if (-not (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)) {
    throw "Tantra Studio did not start. Check logs\webui.err.log for details."
}

Start-Process $url
