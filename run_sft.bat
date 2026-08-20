@echo off
title Tantra LLM - SFT Instruction Fine-Tuning
color 0B
echo ============================================================
echo        TANTRA NEUROCORE INSTRUCTION FINE-TUNING (SFT)
echo ============================================================
cd /d "D:\Atulya Tantra\Tantra-LLM"
python main.py --mode dataset --dataset Datasets --steps 3000 --resume --lr 2e-5 --grad-accum 4 --seq-len 512 --eval-every 250 --log-every 25 --training-stage sft
pause
