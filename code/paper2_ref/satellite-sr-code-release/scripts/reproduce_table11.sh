#!/usr/bin/env bash
# Reproduce Table 11 (recent-methods comparison) on UC Merced x4.
#
# Two kinds of entries, kept strictly separate:
#   1. Published figures (MBGPIN, DEGAN)  -> configs/recent/published_figures.yaml, "*".
#      NOT produced here, never overwritten.
#   2. retrain entries (BD-VITGAN, DTWSTSR, HAM) -> produced ONLY by this script, by
#      cloning each official repo and retraining under our common protocol.
set -euo pipefail

SEEDS=(42 123 999)
RETRAIN=(bd_vitgan dtwstsr ham)   # methods without a citable UC Merced x4 figure
mkdir -p external runs

# 0) Our own model on UC Merced x4 (provides the "Proposed" row in Table 11).
#    Skip if you already trained it for Tables 5-8 (reuse those runs/checkpoints).
for S in "${SEEDS[@]}"; do
  if [[ ! -f "runs/ucmerced_proposed_seed${S}/eval.txt" ]]; then
    python train.py --config configs/default.yaml --variant IV --seed "$S" \
      --train-split splits/ucmerced_train.txt --val-split splits/ucmerced_val.txt \
      --out "runs/ucmerced_proposed_seed${S}"
    python evaluate.py --config configs/default.yaml --variant IV \
      --ckpt "runs/ucmerced_proposed_seed${S}/generator_best.pth" \
      --test-split splits/ucmerced_test.txt
  fi
done

# 1) Retrain the recent methods that have no citable UC Merced x4 figure.
for M in "${RETRAIN[@]}"; do
  REPO=$(python -c "import yaml;print(yaml.safe_load(open('configs/recent/$M.yaml'))['official_repo'])")
  if [[ "$REPO" == "<fill"* ]]; then
    echo "[$M] official_repo not set in configs/recent/$M.yaml -- fill it, then re-run."
    continue
  fi
  [[ -d external/$M ]] || git clone "$REPO" "external/$M"
  for S in "${SEEDS[@]}"; do
    python scripts/retrain_external.py --method "$M" \
      --split splits/ucmerced_train.txt --testlist splits/ucmerced_test.txt --seed "$S"
  done
done

# Note: the aggregator expects runs/ucmerced_proposed_seed<seed>/eval.txt for the
# Proposed row -- that name matches the "ucmerced_proposed" key in aggregate_table11.py.
# Aggregate: published rows (with '*') + retrained rows (mean +/- std) + proposed row
python scripts/aggregate_table11.py \
  --published configs/recent/published_figures.yaml \
  --runs runs --out runs/table11.csv

echo "Table 11 written to runs/table11.csv"
echo "Published rows carry '*'; retrained rows are protocol-identical to ours."
