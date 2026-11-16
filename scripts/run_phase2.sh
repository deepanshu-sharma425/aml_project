#!/usr/bin/env bash
# Phase 2: per-segment XGBoost, ST-GAT, the XGBoost -> GAT hybrid, their ablations
# and the evaluation. Deep runs go one at a time (keeps GPU / unified memory low).
#
#   bash scripts/run_phase2.sh             # everything (about 6 hours of compute on an M3 laptop)
#   FORCE=1 bash scripts/run_phase2.sh     # retrain runs that already have results
#
# Resumable: a run whose results/metrics/<name>.json exists is skipped, so after an
# interruption (sleep, out-of-memory kill) just start the script again. A failed run
# is reported at the end instead of aborting the remaining runs.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
mkdir -p results/logs
FAILED=()

# Main runs: 40 epochs, early stopping after 8 flat epochs.
# Ablations: every variant gets the same fixed budget (ABL_EPOCHS, no early stopping),
# and the full model is re-run under that same budget, so rows are comparable.
ABL_EPOCHS=${ABL_EPOCHS:-12}

done_already() { [[ -z "${FORCE:-}" && -f "results/metrics/$1.json" ]]; }

train() {  # train <run name> <train_deep.py arguments...>
  local name=$1; shift
  if done_already "$name"; then echo "skip $name (results exist)"; return; fi
  echo "=== $name  $(date '+%H:%M')"
  $PY -W ignore scripts/train_deep.py --name "$name" --epoch-fraction 0.5 "$@" 2>&1 \
    | grep --line-buffered -v -i warn > "results/logs/$name.log"
  if [[ ${PIPESTATUS[0]} -ne 0 ]]; then FAILED+=("$name"); echo "FAILED $name"; fi
}
run() { train "$1" --model stgat "${@:2}"; }

run stgat_attention_main --spatial attention --epochs 40 --patience 8
if [[ -n "${FORCE:-}" || ! -f results/predictions/xgb_seg_base.npy ]]; then
  $PY scripts/run_segment_xgb.py > results/logs/xgb_seg.log 2>&1 || FAILED+=(xgb_seg)
fi
run hybrid_attention_main --spatial attention --base xgb_oof --epochs 40 --patience 8

A="--epochs $ABL_EPOCHS --patience 1000"
# ST-GAT: is the graph needed? is attention needed? is the road prior needed? is masking needed?
run stgat_attention_ablation --spatial attention $A
run stgat_none_ablation      --spatial none      $A
run stgat_mean_ablation      --spatial mean      $A
run stgat_dense_ablation     --spatial dense     $A
run stgat_rawinput_ablation  --spatial attention --inputs raw $A
# Hybrid: graph correction vs per-segment correction, residual vs input-only, leakage control
run hybrid_attention_ablation --spatial attention --base xgb_oof      $A
run hybrid_none_ablation      --spatial none      --base xgb_oof      $A
run hybrid_noresid_ablation   --spatial attention --base xgb_oof --no-residual $A
run hybrid_insample_ablation  --spatial attention --base xgb_insample $A

echo "=== evaluate  $(date '+%H:%M')"
$PY -W ignore scripts/evaluate_phase2.py > results/logs/evaluate_phase2.log 2>&1 || FAILED+=(evaluate)
if (( ${#FAILED[@]} )); then echo "FAILED: ${FAILED[*]}"; exit 1; fi
echo "=== all done  $(date '+%H:%M')"
