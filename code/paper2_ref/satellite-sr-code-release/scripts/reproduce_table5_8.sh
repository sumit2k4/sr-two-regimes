#!/usr/bin/env bash
# Reproduce Tables 5-8 (main benchmarks): UC Merced, NWPU-RESISC45, AID, RSSCN7 at x4 bicubic.
# Trains the proposed model (Variant IV) with three seeds on each dataset.
# Baselines (EDSR, ESRGAN, SwinIR) are retrained from their official repos under the
# identical protocol; add their commands below once their configs are in place.
set -euo pipefail

SEEDS=(42 123 999)
DATASETS=(ucmerced nwpu aid rsscn7)

for DS in "${DATASETS[@]}"; do
  for S in "${SEEDS[@]}"; do
    python train.py --config configs/default.yaml --variant IV --seed "$S" \
      --train-split "splits/${DS}_train.txt" --val-split "splits/${DS}_val.txt" \
      --out "runs/${DS}_proposed_seed${S}"
    python evaluate.py --config configs/default.yaml \
      --ckpt "runs/${DS}_proposed_seed${S}/generator_best.pth" --variant IV \
      --test-split "splits/${DS}_test.txt"
  done
done

echo "Aggregate mean +/- std across seeds for Tables 5-8."
