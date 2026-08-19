@echo off
title Tantra LLM - SFT Instruction Fine-Tuning
color 0B
echo ============================================================
echo        TANTRA NEUROCORE INSTRUCTION FINE-TUNING (SFT)
echo ============================================================
cd /d "D:\Atulya Tantra\Tantra-LLM"
python main.py --mode dataset --dataset Datasets/instructions --steps 5000 --resume --lr 1e-4 --grad-accum 4 --seq-len 512 --eval-every 500 --log-every 25 --training-stage sft
pause
