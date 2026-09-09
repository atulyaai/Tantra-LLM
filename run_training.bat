@echo off
cd /d "D:\Atulya Tantra\Tantra-LLM"
python main.py --mode auto-pilot --dataset Datasets/tantra_master_train.jsonl --steps 100000 --resume --lr 1e-4 --grad-accum 4 --batch-size 1 --seq-len 128 --eval-every 2000 --checkpoint-every 5000 --auto-growth --device auto --layers 18
pause
