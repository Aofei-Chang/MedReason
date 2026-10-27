#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-/data/aofei/conda/env/med_edit/bin/python}"
"$PYTHON_BIN" prepare.py
"$PYTHON_BIN" theory_check.py > theory_check.log
CUDA_VISIBLE_DEVICES="${HUATUO_GPU:-1}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PYTHON_BIN" run_primary.py --model huatuo >> huatuo.log 2>&1 &
huatuo_pid=$!
CUDA_VISIBLE_DEVICES="${MEDSIGHT_GPU:-2}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PYTHON_BIN" run_primary.py --model medsight >> medsight.log 2>&1 &
medsight_pid=$!
status=0
wait "$huatuo_pid" || status=1
wait "$medsight_pid" || status=1
if [ "$status" -ne 0 ]; then
  echo 'A model run failed; inspect huatuo.log and medsight.log.' >&2
  exit 1
fi
"$PYTHON_BIN" analyze.py > analysis.log
"$PYTHON_BIN" prepare_adaptive.py
CUDA_VISIBLE_DEVICES="${HUATUO_GPU:-1}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PYTHON_BIN" run.py --model huatuo --tasks fact,anatomy,generation \
  --policies evidence_first --policy-variant evidence_first --output-dir results/adaptive_prompt \
  >> adaptive_huatuo.log 2>&1 &
huatuo_pid=$!
CUDA_VISIBLE_DEVICES="${MEDSIGHT_GPU:-2}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PYTHON_BIN" run.py --model medsight --tasks fact,anatomy,generation \
  --policies evidence_first --policy-variant evidence_first --output-dir results/adaptive_prompt \
  >> adaptive_medsight.log 2>&1 &
medsight_pid=$!
status=0
wait "$huatuo_pid" || status=1
wait "$medsight_pid" || status=1
if [ "$status" -ne 0 ]; then
  echo 'An adaptive run failed; inspect adaptive model logs.' >&2
  exit 1
fi
"$PYTHON_BIN" analyze_adaptive.py > adaptive_analysis.log
"$PYTHON_BIN" conclude.py > conclusions.log
"$PYTHON_BIN" report.py
"$PYTHON_BIN" validate.py > validation.log
