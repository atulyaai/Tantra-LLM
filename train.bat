@echo off
title Tantra LLM - Hindi Training
cd /d "%~dp0"
chcp 65001 >nul 2>&1
python main.py --mode auto-pilot --dataset Datasets/tantra_combined_train.jsonl --val-dataset Datasets/tantra_combined_val.jsonl --steps 5000 --auto-config --lr 3e-4 --optimizer lion --grad-accum 4 --batch-size 2 --seq-len 128 --eval-every 200 --checkpoint-every 500 --device cpu --dim 512 --layers 8 --heads 8 --mask-non-assistant --training-stage sft --warmup 100 --data-workers 0 --max-grad-norm 1.0
pause
