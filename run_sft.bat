@echo off
title Tantra LLM - Fresh 0.5B SFT Training
color 0B
echo ============================================================
echo        TANTRA NEUROCORE FRESH 0.5B SFT
echo        dim=1024, layers=24, heads=16, No Resume
echo ============================================================
cd /d "%~dp0"
python main.py --mode auto-pilot --dataset Datasets/tantra_master_train.jsonl --steps 100000 --fresh --lr 5e-5 --optimizer lion --grad-accum 8 --batch-size 1 --seq-len 256 --eval-every 2000 --checkpoint-every 5000 --auto-growth --device auto --dim 1024 --layers 24 --heads 16 --mask-non-assistant --training-stage sft --warmup 100
pause