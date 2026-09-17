#!/bin/sh
PY="C:/ProgramData/Anaconda3/python.exe"
cd "$(dirname "$0")"
for DS in ucmerced eurosat; do
  $PY -u qge.py --dataset $DS --protocol p1
  $PY -u qge.py --dataset $DS --protocol p2
  for AB in no_realism no_crossview no_posterior no_branch_sup no_halluc_pen; do
    $PY -u qge.py --dataset $DS --protocol p1 --ablation $AB
  done
done
$PY -u analyze.py
$PY -u make_tables.py
$PY -u figures.py
echo QGE_STAGE_DONE
