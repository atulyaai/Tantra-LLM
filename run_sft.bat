@echo off
title Tantra LLM - SFT Instruction Fine-Tuning
color 0B
echo ============================================================
echo        TANTRA NEUROCORE INSTRUCTION FINE-TUNING (SFT)
echo ============================================================
cd /d "%~dp0"
python main.py --mode auto-pilot --dataset Datasets/tantra_master_dataset.jsonl --steps 50000 --resume --lr 1e-4 --grad-accum 4 --batch-size 1 --seq-len 128 --eval-every 2000 --checkpoint-every 5000 --auto-growth --max-layers 16 --device auto --layers 16
pause