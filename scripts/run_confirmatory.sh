#!/usr/bin/env bash
# B5: the confirmatory phase, GPT-4.1 part. Pre-registered in
# prereg/CONFIRMATORY.md, which was committed and pushed before the first call.
# SPENDS CREDIT: 19.22 USD estimated from the token counts of the same models,
# conditions and query types on n600; hard cap 20.20 USD over the five runs.
#
#   bash scripts/run_confirmatory.sh
#
# A fresh CLadder sample: the draw leaves out every id in prereg/excluded_ids.txt
# (all 1,121 ids of the exploratory phase), so no question asked here was asked
# before. 986 items, of which the causal-query jobs keep about 484.
#
#   KEEP, PSEUDO                     RAW, ORACLE, DR_k1, DR_k2, DR_k3
#   PERMUTE, IRRELEVANT, SYMBOL      RAW
#
# Analysed by scripts/analyze_confirmatory.py, whose functions reproduce the
# published n600 rows exactly before they touch these records.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
COMMON=(--models gpt-4.1-nano,gpt-4.1-mini,gpt-4.1 --n 1000 --seed 20260925
        --kmax 3 --sample-kmax 1 --types DR --drop-nonsense --causal-only
        --exclude-ids prereg/excluded_ids.txt)

# A run stops when more than 1% of its calls fail (runner.guard_errors); what
# succeeded is cached, so it is tried again after a pause and pays only for the
# calls that failed. A run stopped by its --max-usd cap is NOT tried again: a
# fresh attempt would start a fresh cap.
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

run "conf_KEEP_log.txt" --lexicon KEEP --conds RAW,ORACLE,DR_k1,DR_k2,DR_k3 \
    --max-usd 8.35 --tag _confKEEP
run "conf_PSEUDO_log.txt" --lexicon PSEUDO --conds RAW,ORACLE,DR_k1,DR_k2,DR_k3 \
    --max-usd 8.30 --tag _confPSEUDO
run "conf_ladder_PERMUTE_log.txt" --lexicon PERMUTE --conds RAW --max-usd 1.35 \
    --tag _confladderPERMUTE
run "conf_ladder_IRRELEVANT_log.txt" --lexicon IRRELEVANT --conds RAW --max-usd 1.10 \
    --tag _confladderIRRELEVANT
run "conf_ladder_SYMBOL_log.txt" --lexicon SYMBOL --conds RAW --max-usd 1.10 \
    --tag _confladderSYMBOL

echo "HOAN TAT $(date '+%Y-%m-%d %H:%M:%S')"
