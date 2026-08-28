@echo off
title Tantra LLM - SFT Instruction Fine-Tuning
color 0B
echo ============================================================
echo        TANTRA NEUROCORE INSTRUCTION FINE-TUNING (SFT)
echo ============================================================
cd /d "%~dp0"
python main.py --mode auto-pilot --dataset Datasets/expert_conversation.jsonl --steps 10000 --resume --lr 1e-4 --grad-accum 2 --batch-size 16 --seq-len 512 --eval-every 500 --checkpoint-every 500 --auto-growth --device auto
pause
