#!/usr/bin/env bash
# Reproduce Table 9 (ablation study) on UC Merced.
#   Variant I   = CNN only (L1)
#   Variant II  = CNN + Transformer fusion
#   Variant III = CNN + GAN
#   Variant IV  = full proposed model
# Each variant is trained with the three paper seeds; report mean +/- std.
set -euo pipefail

TRAIN=splits/ucmerced_train.txt
VAL=splits/ucmerced_val.txt
SEEDS=(42 123 999)

for V in I II III IV; do
  for S in "${SEEDS[@]}"; do
    python train.py --config configs/default.yaml --variant "$V" --seed "$S" \
      --train-split "$TRAIN" --val-split "$VAL" \
      --out "runs/ablation_${V}_seed${S}"
  done
done

echo "Evaluate each run on the test split and aggregate into Table 9."
echo "Expected: Variant IV gives the highest PSNR/SSIM (highest pixel consistency);"
echo "          Variant I gives the highest LPIPS (most perceptual smoothing)."
