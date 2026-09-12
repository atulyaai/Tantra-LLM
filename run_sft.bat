@echo off
title Tantra LLM - 0.5B SFT Training (2 GPU DataParallel)
color 0B
echo ============================================================
echo        TANTRA NEUROCORE 0.5B SFT - 2 GPU DataParallel
echo        dim=1024, layers=24, heads=16, --fresh
echo ============================================================
cd /d "%~dp0"
python main.py --mode auto-pilot --dataset Datasets/tantra_master_train.jsonl --steps 100000 --auto-config --device cuda --optimizer lion --lr 5e-5 --max-grad-norm 5.0 --grad-accum 4 --data-workers 4 --dim 1024 --layers 24 --heads 16 --mask-non-assistant --training-stage sft --warmup 100 --eval-every 1000 --checkpoint-every 1000 --val-dataset Datasets/tantra_val.jsonl
pause
