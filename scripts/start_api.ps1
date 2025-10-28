# One-click start: venv + env + API
Param(
  [string] = "D:\models\longvita-16k",
  [string] = "microsoft/DialoGPT-medium",
  [int] = 8000
)

Continue = "Stop"

# Activate venv
 = ".\.venv\Scripts\Activate.ps1"
if (Test-Path ) { .  } else { Write-Host "Venv not found. Run: python -m venv .venv"; exit 1 }

# Set env
D:\Atulya\Tantra-LLM = (Get-Location).Path
D:\models\longvita-16k = 
microsoft/DialoGPT-medium = 

# Start API
uvicorn demos.api_server:app --host 0.0.0.0 --port 
