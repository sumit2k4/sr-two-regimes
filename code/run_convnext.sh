#!/bin/sh
# Rerun the full classification factorial with ConvNeXt-Tiny branches, then
# refit the gate, rebuild every table and figure, and compile the manuscript.
# The ResNet-18 runs already on disk are kept as a backbone-robustness check.
PY="C:/ProgramData/Anaconda3/python.exe"
cd "$(dirname "$0")"

for DS in ucmerced eurosat; do
  if [ "$DS" = "ucmerced" ]; then EP=30; else EP=20; fi
  for DOM in hr bic_p1 srcnn_p1 edsr_p1 hfgan_p_p1 hfgan_g_p1 lr_p1; do
    for SEED in 42 123 999; do
      $PY -u train_cls.py --dataset $DS --domain $DOM --backbone convnext_tiny \
          --seed $SEED --epochs $EP
    done
  done
done

for DS in ucmerced eurosat; do
  $PY -u qge.py --dataset $DS --protocol p1
  $PY -u qge.py --dataset $DS --protocol p2
  for AB in no_realism no_crossview no_posterior no_branch_sup no_halluc_pen; do
    $PY -u qge.py --dataset $DS --protocol p1 --ablation $AB
  done
done

$PY -u gate_diag.py
echo "CONVNEXT_STAGE_DONE"
