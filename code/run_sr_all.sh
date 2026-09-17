#!/bin/sh
PY="C:/ProgramData/Anaconda3/python.exe"
cd "$(dirname "$0")"
set -x
$PY -u train_sr.py --dataset ucmerced --model srcnn   --epochs 60  --iters_per_epoch 100 --batch 32 --lr 1e-3 --max_minutes 20
$PY -u train_sr.py --dataset ucmerced --model edsr    --epochs 120 --iters_per_epoch 100 --batch 16 --max_minutes 35
$PY -u train_sr.py --dataset ucmerced --model hfgan_p --epochs 120 --iters_per_epoch 100 --batch 8  --max_minutes 75
$PY -u train_sr.py --dataset ucmerced --model hfgan_g --epochs 120 --iters_per_epoch 100 --batch 8  --max_minutes 75
$PY -u train_sr.py --dataset eurosat  --model srcnn   --epochs 60  --iters_per_epoch 100 --batch 64 --lr 1e-3 --max_minutes 15
$PY -u train_sr.py --dataset eurosat  --model edsr    --epochs 120 --iters_per_epoch 100 --batch 32 --max_minutes 25
$PY -u train_sr.py --dataset eurosat  --model hfgan_p --epochs 120 --iters_per_epoch 100 --batch 16 --max_minutes 45
$PY -u train_sr.py --dataset eurosat  --model hfgan_g --epochs 120 --iters_per_epoch 100 --batch 16 --max_minutes 45
echo ALL_SR_DONE
