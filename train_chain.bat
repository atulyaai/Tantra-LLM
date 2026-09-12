@echo off
REM ============================================================
REM  TANTRA TRAINING SESSION CHAIN
REM  Runs bounded chunks per Kaggle session, auto-resumes.
REM  Each session: ~1000 steps, auto-detects checkpoint.
REM  Just run this .bat and repeat on each Kaggle session.
REM ============================================================
cd /d "%~dp0"

setlocal enabledelayedexpansion

set SESSION_STEPS=1000
set TOTAL_STEPS=100000
set STEP_COUNT=0

:LOOP
python main.py --mode auto-pilot --dataset Datasets/tantra_master_train.jsonl --steps %SESSION_STEPS% --auto-config --device cuda --optimizer lion --lr 5e-5 --max-grad-norm 5.0 --grad-accum 4 --data-workers 4 --dim 1024 --layers 24 --heads 16 --mask-non-assistant --training-stage sft --warmup 100 --eval-every 1000 --checkpoint-every 1000 --val-dataset Datasets/tantra_val.jsonl
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Training failed with exit code %ERRORLEVEL%. Retrying...
    timeout /t 5 /nobreak >nul
    goto LOOP
)
set /a STEP_COUNT+=SESSION_STEPS
echo [SESSION CHAIN] Completed step %STEP_COUNT% of %TOTAL_STEPS%.
if %STEP_COUNT% GEQ %TOTAL_STEPS% (
    echo [SESSION CHAIN] Training complete! %TOTAL_STEPS% steps reached.
    pause
    goto END
)
echo [SESSION CHAIN] Next session: continuing from checkpoint (step %STEP_COUNT%).
echo [SESSION CHAIN] Press any key to start next session, or close this window to stop.
pause >nul
goto LOOP

:END
echo [SESSION CHAIN] Done.
