@echo off
chcp 65001 >nul
title Tantra
cd /d "%~dp0"

rem Use the private environment made by install.ps1 when it exists, else the system Python.
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

rem "tantra.bat serve" (desktop shortcut) starts the WebUI directly.
if /i "%~1"=="serve" goto serve

:menu
echo.
echo   तन्त्र  TANTRA
echo   ------------------------------------------
echo   1  WebUI          (opens http://127.0.0.1:8000)
echo   2  Train          (continues automatically from the last save)
echo   3  Chat           (terminal)
echo   4  Test the model (50 questions + speed)
echo   5  Export small file for use (Model\tantra.pt)
echo   6  Check ^& repair everything (installs what is missing)
echo   7  Update Tantra  (latest code + packages)
echo   8  Run code tests
echo   9  WebUI for phone (same Wi-Fi, key required)
echo   0  Exit
echo.
set /p choice=Choose:

if "%choice%"=="1" goto serve
if "%choice%"=="2" %PY% main.py --mode train
if "%choice%"=="3" %PY% main.py --mode chat --int8
if "%choice%"=="4" %PY% main.py --mode eval --int8
if "%choice%"=="5" %PY% main.py --mode export
if "%choice%"=="6" %PY% main.py --mode doctor --fix
if "%choice%"=="7" powershell -ExecutionPolicy Bypass -File "%~dp0install.ps1" -Dir "%~dp0." -NoStart
if "%choice%"=="8" %PY% -m pytest Tests -q
if "%choice%"=="9" start "" http://127.0.0.1:8000 & %PY% main.py --mode serve --lan
if "%choice%"=="0" exit /b 0
goto menu

:serve
start "" http://127.0.0.1:8000
%PY% main.py --mode serve
pause
