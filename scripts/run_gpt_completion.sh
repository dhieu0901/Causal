#!/usr/bin/env bash
# Completion of the GPT-4.1 family's PSEUDO_THIRD cell (prereg/FOLLOWUPS_GPT_COMPLETION.md),
# committed and pushed before the first call. Runs after B8 on the same 7 USD of
# OpenAI credit. SPENDS CREDIT: estimated 0.75 USD (gpt-4.1-mini 288 calls,
# gpt-4.1 334; gpt-4.1-nano's 484 answers are all cached and it is not called).
#
#   bash scripts/run_gpt_completion.sh B8_SPENT_USD
#
# B8_SPENT_USD is B8's recorded OpenAI spend (the GPT-4.1 pair and gpt-5.6-luna,
# from their logs). The cap is 0.90 USD or, if less, what B8 left of the 7 USD
# less 0.02; below 0.80 the run does not start. A run stopped by its cap or by
# the provider is not resumed.

set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
export OPENAI_KEY_VAR=OPENAI_API_KEY

spent="${1:?give the recorded OpenAI spend of B8 in USD}"
cap=$(python -c "print(round(min(0.90, 7.00 - float('$spent') - 0.02), 2))")
if python -c "import sys; sys.exit(0 if $cap >= 0.80 else 1)"; then
  echo ">>> completion, cap $cap USD   [$(date '+%H:%M:%S')]"
else
  echo "!!! B8 left too little credit (cap would be $cap USD, below 0.80): not run." >&2
  exit 3
fi

DRAW=(--n 1000 --seed 20260925 --kmax 3 --sample-kmax 1 --types DR --drop-nonsense
      --causal-only --exclude-ids prereg/excluded_ids.txt)
python scripts/pilot.py "${DRAW[@]}" --models gpt-4.1-nano,gpt-4.1-mini,gpt-4.1 \
  --lexicon PSEUDO_THIRD --conds RAW --tag _premisePSEUDO_THIRD --max-usd "$cap" \
  > results/cladder/logs/gpt_completion_PSEUDO_THIRD.txt 2>&1
rc=$?
grep -o "spent so far [0-9.]* USD" results/cladder/logs/gpt_completion_PSEUDO_THIRD.txt | tail -1
echo "<<< completion exit $rc   [$(date '+%H:%M:%S')]"
exit $rc
