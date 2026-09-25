#!/usr/bin/env bash
# Four additions to the n600 sample (580 items, all 10 graph families), run
# 2026-09-23. SPENDS CREDIT: about 16 USD at GPT-4.1 list prices, measured
# with --dry-run --cost-ref before the run.
#
#   bash scripts/run_n600_extensions.sh
#
#   1. DR_k2, DR_k3    the dose line on all 10 families; price400 has 7
#   2. NAMES_ONLY      the matched control for ORACLE: names, no arrows
#   3. SCRAMBLE        a random DAG on the same names, no true edge kept
#   4. the lexical ladder (PERMUTE, IRRELEVANT, SYMBOL) at RAW, which with the
#      KEEP and PSEUDO runs already on disk puts section 7 on 580 items, not 86
#
# Every run reproduces the n600 sample exactly (--n 600 --sample-kmax 1
# --drop-nonsense) and keeps only its NEW conditions (--conds), so no existing
# results file changes and no old row is written twice. Every call is cached,
# so re-running costs nothing and an interrupted run resumes.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
M="gpt-4.1-nano,gpt-4.1-mini,gpt-4.1"

run () {
  local log="$1"; shift
  echo ">>> $log   [$(date '+%H:%M:%S')]"
  local rc=0
  python scripts/pilot.py "$@" > "results/logs/$log" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "!!! $log FAILED, exit=$rc. See results/logs/$log" >&2
    exit "$rc"
  fi
  echo "<<< $log done   [$(date '+%H:%M:%S')]"
}

for LEX in KEEP PSEUDO; do
  run "n600arms_${LEX}_log.txt" --n 600 --models "$M" --kmax 3 --sample-kmax 1 \
      --types DR --drop-nonsense --lexicon "$LEX" --names-only --scramble \
      --conds DR_k2,DR_k3,NAMES_ONLY,SCRAMBLE --tag "_n600arms$LEX"
done

for LEX in PERMUTE IRRELEVANT SYMBOL; do
  run "n600ladder_${LEX}_log.txt" --n 600 --models "$M" --kmax 1 \
      --types DR --drop-nonsense --lexicon "$LEX" --conds RAW \
      --tag "_n600ladder$LEX"
done

echo "HOAN TAT $(date '+%Y-%m-%d %H:%M:%S')"
