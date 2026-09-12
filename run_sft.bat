@echo off
title Tantra LLM - Fresh 0.5B SFT Training
color 0B
echo ============================================================
echo        TANTRA NEUROCORE FRESH 0.5B SFT
echo        dim=1024, layers=24, heads=16, No Resume
echo ============================================================
cd /d "%~dp0"
python main.py --mode auto-pilot --dataset Datasets/tantra_master_train.jsonl --steps 100000 --fresh --device cuda --single-gpu --compile --optimizer lion --lr 5e-5 --grad-accum 4 --batch-size 4 --seq-len 256 --eval-every 1000 --checkpoint-every 1000 --auto-growth --dim 1024 --layers 24 --heads 16 --mask-non-assistant --training-stage sft --warmup 100
pause