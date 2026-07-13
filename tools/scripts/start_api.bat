@echo off
REM One-click start: venv + env + API (CMD)
setlocal enableextensions enabledelayedexpansion

if not exist .\.venv\Scripts\activate.bat (
  echo Venv not found. Run: python -m venv .venv
  exit /b 1
)

call .\.venv\Scripts\activate.bat
set PYTHONPATH=%cd%
if "%TANTRA_LV_DIR%"=="" set TANTRA_LV_DIR=D:\models\longvita-16k
if "%TANTRA_SPB%"=="" set TANTRA_SPB=microsoft/DialoGPT-medium

uvicorn demos.api_server:app --host 0.0.0.0 --port 8000
