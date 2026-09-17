#!/bin/sh
# Wait for the super-resolution stage, then run everything else end to end.
PY="C:/ProgramData/Anaconda3/python.exe"
cd "$(dirname "$0")"

while ! grep -q ALL_SR_DONE ../runs/sr_all.log 2>/dev/null; do
  sleep 60
done
echo "SR stage complete, starting stage 2"

sh run_stage2.sh
$PY -u bench.py ucmerced
$PY -u analyze.py
$PY -u make_tables.py
$PY -u figures.py

cd ../paper
pdflatex -interaction=batchmode main.tex > /dev/null 2>&1
pdflatex -interaction=batchmode main.tex > /dev/null 2>&1
pdflatex -interaction=batchmode main.tex > /dev/null 2>&1
echo "PIPELINE_DONE"
