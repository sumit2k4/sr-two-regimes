#!/bin/sh
# Final build: regenerate every table, macro and figure from the cached results,
# then compile the manuscript. Order matters, because make_tables.py rewrites
# numbers.tex and make_tables_transfer.py appends to it.
PY="C:/ProgramData/Anaconda3/python.exe"
cd "$(dirname "$0")"

$PY -u analyze.py > ../runs/final_analyze.log 2>&1
$PY -u analyze_transfer.py >> ../runs/final_analyze.log 2>&1
$PY -u gate_diag.py >> ../runs/final_analyze.log 2>&1
$PY -u make_tables.py
$PY -u make_tables_transfer.py
$PY -u make_tables_backbone.py
$PY -u figures.py
$PY -u figures_transfer.py
$PY -u figures_arch.py

cd ../paper
pdflatex -interaction=batchmode main.tex > /dev/null 2>&1 || true
pdflatex -interaction=batchmode main.tex > /dev/null 2>&1 || true
pdflatex -interaction=batchmode main.tex > /dev/null 2>&1 || true
echo "BUILD_DONE"
grep -c "^! " main.log || true
