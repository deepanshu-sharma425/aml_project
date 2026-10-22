#!/usr/bin/env bash
# Train every deep model sequentially (one at a time keeps GPU memory low).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
mkdir -p results/logs

$PY scripts/train_deep.py --model gwnet --variant full --epochs 35 --patience 8 --tag main | tee results/logs/gwnet_full_main.log
$PY scripts/train_deep.py --model lstm --epochs 60 --patience 10 --tag main | tee results/logs/lstm_main.log

# Ablation: identical 8-epoch budget for every Graph WaveNet variant.
for v in full no_adaptive adaptive_only no_graph; do
  $PY scripts/train_deep.py --model gwnet --variant "$v" --epochs 8 --patience 100 --tag ablation | tee "results/logs/gwnet_${v}_ablation.log"
done
