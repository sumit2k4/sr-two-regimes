#!/bin/sh
# Stage 2: materialise SR domains, score reconstruction quality, train the
# classification factorial, fit the quality-gated ensemble, build all tables.
PY="C:/ProgramData/Anaconda3/python.exe"
cd "$(dirname "$0")"
set -x

for DS in ucmerced eurosat; do
  $PY -u make_sr_cache.py --dataset $DS
  $PY -u sr_eval.py       --dataset $DS
done

for DS in ucmerced eurosat; do
  if [ "$DS" = "ucmerced" ]; then EP=30; else EP=20; fi
  for DOM in hr bic_p1 srcnn_p1 edsr_p1 hfgan_p_p1 hfgan_g_p1 lr_p1; do
    for SEED in 42 123 999; do
      $PY -u train_cls.py --dataset $DS --domain $DOM --backbone resnet18 --seed $SEED --epochs $EP
    done
  done
  for BB in efficientnet_b0 densenet121; do
    for SEED in 42 123 999; do
      $PY -u train_cls.py --dataset $DS --domain hfgan_g_p1 --backbone $BB --seed $SEED --epochs $EP
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

$PY -u analyze.py
echo STAGE2_DONE
