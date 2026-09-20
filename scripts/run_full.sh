#!/usr/bin/env bash
# Full experiment at n=800, both lexical branches, three model tiers.
#
#   bash scripts/run_full.sh
#
# Every call is cached by (model, temperature, prompt), so re-running costs
# nothing and an interrupted run resumes where it stopped. Safe to kill and
# restart.
#
# Scale: ~59k calls, ~70 USD, several hours. Stages are ordered so the most
# informative one lands first - if the run is cut short, the main-branch pilot
# alone still answers the primary question.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8

N=800
MODELS="gpt-4.1-nano,gpt-4.1-mini,gpt-4.1"
LOG=results/full_run.log

# A stage that fails must stop the run. The 2026-09-07 log shows all six stages
# exiting 127 (python not on PATH) while the script still printed HOAN TAT after
# 7 minutes and zero bytes of data - the exit code was written to the log and
# then never read.
stage () {
  local name="$1"; shift
  echo "" | tee -a "$LOG"
  echo "=================================================================" | tee -a "$LOG"
  echo ">>> $name   [$(date '+%H:%M:%S')]" | tee -a "$LOG"
  echo "=================================================================" | tee -a "$LOG"
  local rc=0
  "$@" >> "$LOG" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "!!! $name THAT BAI, exit=$rc   [$(date '+%H:%M:%S')]" | tee -a "$LOG"
    echo "!!! Aborting the whole run. See $LOG." | tee -a "$LOG"
    exit "$rc"
  fi
  echo "<<< $name done, exit=0   [$(date '+%H:%M:%S')]" | tee -a "$LOG"
}

command -v python >/dev/null 2>&1 || {
  echo "python is not on PATH. Activate the environment and run again." >&2
  exit 127
}

mkdir -p results
: > "$LOG"
echo "BAT DAU $(date '+%Y-%m-%d %H:%M:%S')  n=$N  models=$MODELS" | tee -a "$LOG"

# 1. Main branch. The primary result: what each error type costs, and where the
#    curve crosses the no-graph floor.
stage "1/5 pilot commonsense n=$N" \
  python scripts/pilot.py --n $N --models "$MODELS" --kmax 3 --types DR,ED,FE \
                          --tag _n800

# 2. Where a real agent's own graph lands on that curve.
stage "2/5 induction commonsense n=$N" \
  python scripts/induction.py --n $N --models "$MODELS" --tag _n800

# 3. The lexical contrast, done WITHIN item. The earlier version of this stage
#    passed --data test-noncommonsense-v1.5.csv; that file carries no question at
#    all (0% of its prompts contain a '?') and make_items now refuses it. See
#    REPORT.md section 2. The four lexicons below are the replacement design.
for LEX in KEEP PERMUTE SYMBOL PSEUDO; do
  stage "3/6 pilot lexicon=$LEX n=$N"     python scripts/pilot.py --n $N --models "$MODELS" --kmax 1 --types DR                             --drop-nonsense --lexicon "$LEX" --tag "_n800_lex$LEX"
done

# 4. Whether induction quality survives losing the lexical anchors.
for LEX in KEEP PERMUTE SYMBOL PSEUDO; do
  stage "4/6 induction lexicon=$LEX n=$N"     python scripts/induction.py --n $N --models "$MODELS" --drop-nonsense                                 --lexicon "$LEX" --tag "_n800_lex$LEX"
done

# 5. Pricing the error types on the main branch.
stage "5/6 pricing the error types"   python scripts/analyze_types.py --pilot pilot_raw_n800.csv                                   --induction induction_raw_n800.csv --tag _n800

# 6. Headline analysis. analyze_querygroup.py carries the stratification and the
#    interaction tests that review round 6 required; analyze_lexical.py is the
#    pooled view kept for comparison.
stage "6/7 lexical analysis"   python scripts/analyze_lexical.py
stage "6/7 stratify by query group"   python scripts/analyze_querygroup.py

# 7. Checks that do not depend on the run above and cost nothing, but that the
#    report leans on. verify_groundtruth enumerates all 7,064 SCMs rather than
#    trusting CLadder's labels; analyze_chains is the only direct evidence that
#    the supplied graph reaches the model's reasoning and not just its answer.
stage "7/7 verify the answer key over every SCM"   python scripts/verify_groundtruth.py
stage "7/7 do the chains use the graph"   python scripts/analyze_chains.py
stage "7/7 bat thuong va residue"   python scripts/analyze_anomaly_residue.py
stage "7/7 ngan sach ghep cap"   python scripts/analyze_budget_paired.py

echo "" | tee -a "$LOG"
echo "HOAN TAT $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
