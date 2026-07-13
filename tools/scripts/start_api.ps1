# One-click start: venv + env + API
Param(
  [string]$ModelPath = "D:\models\longvita-16k",
  [string]$DialogueModel = "microsoft/DialoGPT-medium",
  [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

# Activate venv
$ActivatePath = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $ActivatePath) {
    . $ActivatePath
} else {
    Write-Host "Venv not found. Run: python -m venv .venv"
    exit 1
}

# Set env
$env:PYTHONPATH = (Get-Location).Path
$env:TANTRA_LV_DIR = $ModelPath
$env:TANTRA_SPB = $DialogueModel

# Start API
Write-Host "Starting API on port $Port..."
uvicorn demos.api_server:app --host 0.0.0.0 --port $Port
