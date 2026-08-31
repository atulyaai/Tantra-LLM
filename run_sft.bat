@echo off
title Tantra LLM - SFT Instruction Fine-Tuning
color 0B
echo ============================================================
echo        TANTRA NEUROCORE INSTRUCTION FINE-TUNING (SFT)
echo ============================================================
cd /d "%~dp0"
python main.py --mode dataset --dataset Datasets/staged_master.jsonl --steps 10000 --resume --lr 1e-4 --grad-accum 4 --seq-len 512 --eval-every 500 --checkpoint-every 500 --training-stage sft
pause
