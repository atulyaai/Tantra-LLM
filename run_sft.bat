@echo off
title Tantra LLM - SFT Instruction Fine-Tuning
color 0B
echo ============================================================
echo        TANTRA NEUROCORE INSTRUCTION FINE-TUNING (SFT)
echo        optimizer RESET on stage transition, 512/18 + auto-grow to 24
echo ============================================================
cd /d "%~dp0"
python main.py --mode auto-pilot --dataset Datasets/tantra_master_train.jsonl --steps 100000 --resume --lr 5e-5 --optimizer lion --grad-accum 4 --batch-size 4 --seq-len 256 --eval-every 2000 --checkpoint-every 5000 --auto-growth --device auto --dim 512 --layers 18 --heads 8 --mask-non-assistant --training-stage sft --warmup 1000
pause