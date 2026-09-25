#!/usr/bin/env bash
# The second benchmark, CaLM's ATE items (prereg/CALM.md, committed and pushed
# before the first call). SPENDS CREDIT.
#
#   bash scripts/run_calm.sh llama    # about 0.41 USD, cap 0.60 (OpenRouter)
#   bash scripts/run_calm.sh gpt      # about 8.0 USD, cap 9.0 (OpenAI)
#
# 520 REAL-mode items from 92 stories: KEEP under RAW and ORACLE, PSEUDO under
# RAW, ORACLE and DR_k1. Estimates from scripts/calm_run.py --dry-run, which
# prices each new call at the model's mean output length over every CLadder
# record on disk. Analysed by scripts/analyze_calm.py.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8

case "${1:-}" in
  llama) M="meta-llama/llama-3.3-70b-instruct"; P=llama; EXTRA=(--workers 8 --recap 1500)
         CAP_K=0.25; CAP_P=0.35 ;;
  gpt)   M="gpt-4.1-nano,gpt-4.1-mini,gpt-4.1"; P=gpt; EXTRA=()
         CAP_K=3.8; CAP_P=5.2 ;;
  *) echo "usage: bash scripts/run_calm.sh llama|gpt" >&2; exit 2 ;;
esac

# Retried after a pause when more than 1% of calls fail (what succeeded is
# cached); never retried after the spending cap stopped it.
run () {
  local log="$1"; shift
  local rc
  for attempt in 1 2 3; do
    echo ">>> $log   attempt $attempt   [$(date '+%H:%M:%S')]"
    rc=0
    python scripts/calm_run.py --models "$M" "${EXTRA[@]}" "$@" > "results/logs/$log" 2>&1 || rc=$?
    if [ "$rc" -eq 0 ]; then
      echo "<<< $log done   [$(date '+%H:%M:%S')]"
      return 0
    fi
    if grep -q "SPENDING CAP REACHED" "results/logs/$log"; then
      echo "!!! $log reached its spending cap. Stopping; nothing more is sent." >&2
      exit "$rc"
    fi
    echo "... $log stopped, exit=$rc; trying again in 60 s" >&2
    sleep 60
  done
  echo "!!! $log FAILED, exit=$rc. See results/logs/$log" >&2
  exit "$rc"
}

run "calm_${P}_KEEP_log.txt" --lexicon KEEP --conds RAW,ORACLE --max-usd "$CAP_K" --tag "_${P}KEEP"
run "calm_${P}_PSEUDO_log.txt" --lexicon PSEUDO --conds RAW,ORACLE,DR_k1 --max-usd "$CAP_P" \
    --tag "_${P}PSEUDO"

echo "HOAN TAT $(date '+%Y-%m-%d %H:%M:%S')"
