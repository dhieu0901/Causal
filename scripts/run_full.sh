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

set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8

N=800
MODELS="gpt-4.1-nano,gpt-4.1-mini,gpt-4.1"
LOG=results/full_run.log

stage () {
  local name="$1"; shift
  echo "" | tee -a "$LOG"
  echo "=================================================================" | tee -a "$LOG"
  echo ">>> $name   [$(date '+%H:%M:%S')]" | tee -a "$LOG"
  echo "=================================================================" | tee -a "$LOG"
  "$@" >> "$LOG" 2>&1
  echo "<<< $name done, exit=$?   [$(date '+%H:%M:%S')]" | tee -a "$LOG"
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

# 3. Contamination control. Same items, variable names replaced by pseudowords.
stage "3/5 pilot pseudoword n=$N" \
  python scripts/pilot.py --n $N --models "$MODELS" --kmax 3 --types DR,ED,FE \
                          --data test-noncommonsense-v1.5.csv --tag _n800_noncs

# 4. Whether induction quality survives losing the lexical anchors.
stage "4/5 induction pseudoword n=$N" \
  python scripts/induction.py --n $N --models "$MODELS" \
                              --data test-noncommonsense-v1.5.csv --tag _n800_noncs

# 5. Analysis of both branches.
stage "5/5 analyse commonsense" \
  python scripts/analyze_types.py --pilot pilot_raw_n800.csv \
                                  --induction induction_raw_n800.csv --tag _n800
stage "5/5 analyse pseudoword" \
  python scripts/analyze_types.py --pilot pilot_raw_n800_noncs.csv \
                                  --induction induction_raw_n800_noncs.csv \
                                  --tag _n800_noncs

echo "" | tee -a "$LOG"
echo "HOAN TAT $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
