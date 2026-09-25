#!/usr/bin/env bash
# RAW_CLEAN on all three samples, and the old RAW asked again in the same
# sitting, run 2026-09-24. SPENDS CREDIT: about 3.2 USD at GPT-4.1 list prices,
# measured with --dry-run before the run.
#
#   bash scripts/run_clean_raw.sh
#
# Why a new condition and not a wider strip_structure(). The sentence "X is
# unobserved." survives the strip in every IV, arrowhead and frontdoor item. It
# is structure, so RAW should not carry it. But it is also the only place a
# latent node is marked as latent - the ORACLE block lists edges, not latency,
# and under PSEUDO the latent is just another pseudoword - so the graph arms
# SHOULD carry it. Widening the strip would take it out of every arm (about
# 23 USD of re-runs) and leave ORACLE an incomplete structure. RAW_CLEAN takes it
# out of the no-graph baseline only. See src/prompts.py.
#
# Why RAW again. RAW_CLEAN is answered today; the RAW it is compared with was
# answered on 2026-09-08 (lex, price400) or about 2026-09-16 (n600). Asking the
# old RAW prompts again now, into cache_drift/, makes RAW_CLEAN minus RAW a
# same-run contrast and extends the drift test to the two older samples. The
# n600 RAW prompts were already asked again by the first drift check.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
M="gpt-4.1-nano,gpt-4.1-mini,gpt-4.1"

run () {
  local log="$1"; shift
  echo ">>> $log   [$(date '+%H:%M:%S')]"
  local rc=0
  python "$@" > "results/cladder/logs/$log" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "!!! $log FAILED, exit=$rc. See results/cladder/logs/$log" >&2
    exit "$rc"
  fi
  echo "<<< $log done   [$(date '+%H:%M:%S')]"
}

for LEX in KEEP PSEUDO; do
  run "cleanraw_lex_${LEX}_log.txt" scripts/pilot.py --n 200 --models "$M" --kmax 1 \
      --types DR --drop-nonsense --lexicon "$LEX" --clean-raw --conds RAW_CLEAN \
      --tag "_cleanrawlex$LEX"
  run "cleanraw_n600_${LEX}_log.txt" scripts/pilot.py --n 600 --models "$M" --kmax 1 \
      --types DR --drop-nonsense --lexicon "$LEX" --clean-raw --conds RAW_CLEAN \
      --tag "_cleanrawn600$LEX"
  run "cleanraw_price400_${LEX}_log.txt" scripts/pilot.py --n 400 --models "$M" --kmax 3 \
      --types DR,ED,FE --drop-nonsense --lexicon "$LEX" --clean-raw --conds RAW_CLEAN \
      --tag "_cleanrawprice400$LEX"
done

run "drift_check_price400_log.txt" scripts/check_drift.py --sample price400 --conds RAW
run "drift_check_lex_log.txt" scripts/check_drift.py --sample lex --conds RAW

echo "HOAN TAT $(date '+%Y-%m-%d %H:%M:%S')"
