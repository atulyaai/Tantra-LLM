@echo off
REM One-click start: activate venv and run the NP-DNA API server (CMD)
setlocal enableextensions enabledelayedexpansion

set CHECKPOINT=%1
if "%CHECKPOINT%"=="" set CHECKPOINT=model/latest
set HOST=%2
if "%HOST%"=="" set HOST=127.0.0.1
set PORT=%3
if "%PORT%"=="" set PORT=8000

if not exist .\.venv\Scripts\activate.bat (
  echo Venv not found. Run: python -m venv .venv ^&^& .venv\Scripts\pip install -e ".[api]"
  exit /b 1
)

call .\.venv\Scripts\activate.bat
set PYTHONPATH=%cd%
set NPDNA_CHECKPOINT=%CHECKPOINT%

echo Starting NP-DNA API on %HOST%:%PORT% (checkpoint=%CHECKPOINT%)...
uvicorn npdna.serving:app --host %HOST% --port %PORT%
