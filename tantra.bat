@echo off
chcp 65001 >nul
title Tantra
cd /d "%~dp0"

:menu
echo.
echo   तन्त्र  TANTRA
echo   ------------------------------------------
echo   1  Train          (continues automatically from the last save)
echo   2  Chat           (terminal)
echo   3  WebUI          (opens http://127.0.0.1:8000)
echo   4  Test the model (50 questions + speed)
echo   5  Export small file for use (Model\tantra.pt)
echo   6  Build tokenizer (ONLY ONCE, before the first real training)
echo   7  Install / update requirements
echo   8  Run code tests
echo   0  Exit
echo.
set /p choice=Choose: 

if "%choice%"=="1" python main.py --mode train
if "%choice%"=="2" python main.py --mode chat --int8
if "%choice%"=="3" start "" http://127.0.0.1:8000 & python main.py --mode serve --int8
if "%choice%"=="4" python main.py --mode eval --int8
if "%choice%"=="5" python main.py --mode export
if "%choice%"=="6" python main.py --mode tokenizer
if "%choice%"=="7" python -m pip install -r requirements.txt
if "%choice%"=="8" python -m pytest Tests -q
if "%choice%"=="0" exit /b 0
goto menu
