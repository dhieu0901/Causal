#!/usr/bin/env bash
# The closed-world clause and which names carry the gain, on Llama 3.3 70B
# Instruct (OpenRouter, pinned to Novita, bf16). Pre-registered in
# prereg/NAMES_PREMISE.md, committed and pushed before the first call. SPENDS
# CREDIT: about 0.22 USD for 1,636 new calls, estimated from the measured spend
# of Study 1's RAW ladder runs on Llama (0.06 USD per 484 calls), plus
# re-asks; hard caps 4 x 0.12 = 0.48 USD.
#
#   bash scripts/run_names_premise.sh
#
# The draw is run_b5_llama.sh's, so every question is one of Study 1's 484 and
# the RAW answers under KEEP and PSEUDO are reused from it. Four new cells:
# RAW_OPEN under KEEP and PSEUDO, and RAW under PSEUDO_XY and PSEUDO_THIRD.
# Temperature 0, 700 output tokens; every answer cut off at 700 is asked again
# at 1,500 into pilot_raw{tag}_recap1500.csv. Analysed by
# analyze_names_premise.py.
#
# A cap covers every attempt of its run together: an attempt gets what the
# earlier attempts left ("spent so far" in their logs). A run stopped by its
# cap is not tried again. A run stopped by provider errors is tried again
# after a pause; what succeeded is cached and costs nothing to replay.

set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
COMMON=(--models meta-llama/llama-3.3-70b-instruct --workers 8 --recap 1500
        --n 1000 --seed 20260925 --kmax 3 --sample-kmax 1 --types DR
        --drop-nonsense --causal-only --exclude-ids prereg/excluded_ids.txt)

spent_in () {
  grep -o "spent so far [0-9.]* USD" "$1" | tail -1 | awk '{print $4}'
}

# run LOG CAP COMMAND...
run () {
  local log="$1" left="$2"; shift 2
  local rc s f
  for attempt in 1 2 3 4; do
    f="results/cladder/logs/${log}_a${attempt}.txt"
    echo ">>> $log   attempt $attempt   cap left $left USD   [$(date '+%H:%M:%S')]"
    rc=0
    "$@" --max-usd "$left" > "$f" 2>&1 || rc=$?
    s=$(spent_in "$f"); s=${s:-0}
    left=$(python -c "print(round(max(0.0, $left - $s), 2))")
    if [ "$rc" -eq 0 ]; then
      echo "<<< $log done, this attempt spent $s USD   [$(date '+%H:%M:%S')]"
      return 0
    fi
    if grep -q "SPENDING CAP REACHED" "$f"; then
      echo "!!! $log reached its spending cap. Stopping; nothing more is sent." >&2
      exit "$rc"
    fi
    echo "... $log stopped, exit=$rc; trying again in 60 s" >&2
    sleep 60
  done
  echo "!!! $log FAILED, exit=$rc. See $f" >&2
  exit "$rc"
}

run premise_KEEP 0.12 python scripts/pilot.py "${COMMON[@]}" --lexicon KEEP --open-raw \
    --conds RAW_OPEN --tag _llamapremiseKEEP
run premise_PSEUDO 0.12 python scripts/pilot.py "${COMMON[@]}" --lexicon PSEUDO --open-raw \
    --conds RAW_OPEN --tag _llamapremisePSEUDO
run premise_PSEUDO_XY 0.12 python scripts/pilot.py "${COMMON[@]}" --lexicon PSEUDO_XY \
    --conds RAW --tag _llamapremisePSEUDO_XY
run premise_PSEUDO_THIRD 0.12 python scripts/pilot.py "${COMMON[@]}" --lexicon PSEUDO_THIRD \
    --conds RAW --tag _llamapremisePSEUDO_THIRD

echo "HOAN TAT $(date '+%Y-%m-%d %H:%M:%S')"
