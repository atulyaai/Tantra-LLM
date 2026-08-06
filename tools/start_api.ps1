# One-click start: activate venv and run the NP-DNA API server.
Param(
    [string]$Checkpoint = "model/latest",
    [string]$Host = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

# Activate local venv if present
$ActivatePath = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $ActivatePath) {
    . $ActivatePath
} else {
    Write-Host "Venv not found. Run: python -m venv .venv && .venv\Scripts\pip install -e '.[api]'"
    exit 1
}

$env:PYTHONPATH = (Get-Location).Path
$env:NPDNA_CHECKPOINT = $Checkpoint

Write-Host "Starting NP-DNA API on $Host:$Port (checkpoint=$Checkpoint)..."
uvicorn npdna.serving:app --host $Host --port $Port
