#!/bin/bash
# Ablation suite for paper_v3 (2026-07-22): vary one flag at a time from the
# winner config (s1s2, mean agg + offline shrink, calibrated prompt v1,
# share-mode arguments, evidence pooling, BM25 routing, rho=0.5, J=3, R=2,
# temp 0.7, seed 42). Outputs land in data/results_v2/ following the
# existing naming convention; every run is resumable via --resume.
set -u
cd "$(dirname "$0")/.."

BASE="python3 run_workflow.py --dataset-jsonl data/processed/polymarket_250_with_evidence.jsonl \
  --limit 250 --scenario s1s2 --aggregator mean --calibrated-prompt \
  --evidence-pooling --temperature 0.7 --max-workers 8"

RUNS=(
  "rho03|--share-mode arguments --bm25 --public-ratio 0.3"
  "rho07|--share-mode arguments --bm25 --public-ratio 0.7"
  "numshare|--share-mode numbers --bm25"
  "randroute|--share-mode arguments"
  "r3|--share-mode arguments --bm25 --num-rounds 3"
  "seed1|--share-mode arguments --bm25 --seed 1"
  "seed2|--share-mode arguments --bm25 --seed 2"
)

for spec in "${RUNS[@]}"; do
  name="${spec%%|*}"; flags="${spec#*|}"
  out="data/results_v2/polymarket_250_s1s2_argpool_cal_${name}_5.4mini.jsonl"
  log="data/results_v2/ablation_${name}.log"
  echo "=== $(date '+%F %T') starting ${name} ==="
  # shellcheck disable=SC2086
  $BASE $flags --output "$out" --resume >> "$log" 2>&1
  echo "=== $(date '+%F %T') finished ${name} exit=$? ==="
done
echo "ALL DONE $(date '+%F %T')"
