#!/usr/bin/env bash
# B5 on the second model family: Llama 3.3 70B Instruct through OpenRouter,
# pinned to Novita (bf16, runner.OPENROUTER). Pre-registered in
# prereg/CONFIRMATORY.md section 3 ("the same five commands with --models
# meta-llama/llama-3.3-70b-instruct --workers 8 --recap 1500 and tags
# _llamaconf*"), analysed by analyze_confirmatory.py as family `llama` and
# reported separately as a replication. SPENDS CREDIT: 0.97 USD estimated on
# 2026-09-27 from the measured per-call spend of the same conditions on the
# exploratory n600 Llama runs (0.91 main, 0.06 re-asks); hard caps 1.30 USD
# over the five runs.
#
#   bash scripts/run_b5_llama.sh
#
# The draw is run_confirmatory.sh's, so every item, label and prompt is the one
# GPT-4.1 was sent. Temperature 0, 700 output tokens; every answer cut off at
# 700 is asked again at 1,500 and written to pilot_raw{tag}_recap1500.csv beside
# the main file, which stays as it was (prereg section 6).

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
COMMON=(--models meta-llama/llama-3.3-70b-instruct --workers 8 --recap 1500
        --n 1000 --seed 20260925 --kmax 3 --sample-kmax 1 --types DR
        --drop-nonsense --causal-only --exclude-ids prereg/excluded_ids.txt)

# A run stops when more than 1% of its calls fail (runner.guard_errors). The
# provider rate-limits in bursts, so a stopped run is tried again after a
# pause: what succeeded is cached and only the failed calls are sent again. A
# run stopped by its --max-usd cap is NOT tried again.
run () {
  local log="$1"; shift
  local rc
  for attempt in 1 2 3; do
    echo ">>> $log   attempt $attempt   [$(date '+%H:%M:%S')]"
    rc=0
    python scripts/pilot.py "${COMMON[@]}" "$@" > "results/cladder/logs/$log" 2>&1 || rc=$?
    if [ "$rc" -eq 0 ]; then
      echo "<<< $log done   [$(date '+%H:%M:%S')]"
      return 0
    fi
    if grep -q "SPENDING CAP REACHED" "results/cladder/logs/$log"; then
      echo "!!! $log reached its spending cap. Stopping; nothing more is sent." >&2
      exit "$rc"
    fi
    echo "... $log stopped, exit=$rc; trying again in 60 s" >&2
    sleep 60
  done
  echo "!!! $log FAILED, exit=$rc. See results/cladder/logs/$log" >&2
  exit "$rc"
}

run "llamaconf_KEEP_log.txt" --lexicon KEEP --conds RAW,ORACLE,DR_k1,DR_k2,DR_k3 \
    --max-usd 0.50 --tag _llamaconfKEEP
run "llamaconf_PSEUDO_log.txt" --lexicon PSEUDO --conds RAW,ORACLE,DR_k1,DR_k2,DR_k3 \
    --max-usd 0.50 --tag _llamaconfPSEUDO
run "llamaconf_ladder_PERMUTE_log.txt" --lexicon PERMUTE --conds RAW --max-usd 0.10 \
    --tag _llamaconfladderPERMUTE
run "llamaconf_ladder_IRRELEVANT_log.txt" --lexicon IRRELEVANT --conds RAW --max-usd 0.10 \
    --tag _llamaconfladderIRRELEVANT
run "llamaconf_ladder_SYMBOL_log.txt" --lexicon SYMBOL --conds RAW --max-usd 0.10 \
    --tag _llamaconfladderSYMBOL

echo "HOAN TAT $(date '+%Y-%m-%d %H:%M:%S')"
