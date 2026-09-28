#!/usr/bin/env bash
# CLadder's questions told in new stories (src/new_stories.py), on Llama 3.3 70B
# Instruct (OpenRouter, pinned to Novita, bf16) and the GPT-4.1 family.
# Pre-registered in prereg/NEW_STORIES.md, committed and pushed before the first
# call. SPENDS CREDIT, estimated from the measured tokens of Study 1's KEEP RAW
# answers on the same 484 questions: Llama about 0.07 USD (cap 0.15); GPT-4.1
# family about 1.18 USD (nano 0.07, mini 0.20, gpt-4.1 0.91; cap 1.40, approved
# over the default OpenAI budget on 28/09).
#
#   bash scripts/run_new_stories.sh
#
# The draws are run_b5_llama.sh's and run_confirmatory.sh's, so every question is
# one of Study 1's 484; KEEP and SYMBOL under RAW are reused from Study 1. One new
# cell per family: NEWSTORY under RAW. Temperature 0, 700 output tokens; Llama's
# answers cut off at 700 are asked again at 1,500, as in Study 1 on Llama.
#
# A cap covers every attempt of its run together: an attempt gets what the
# earlier attempts left ("spent so far" in their logs). A run stopped by its
# cap is not tried again. A run stopped by provider errors is tried again
# after a pause; what succeeded is cached and costs nothing to replay.

set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
DRAW=(--n 1000 --seed 20260925 --kmax 3 --sample-kmax 1 --types DR --drop-nonsense
      --causal-only --exclude-ids prereg/excluded_ids.txt --lexicon NEWSTORY --conds RAW)

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

run newstory_llama 0.15 python scripts/pilot.py "${DRAW[@]}" \
    --models meta-llama/llama-3.3-70b-instruct --workers 8 --recap 1500 --tag _llamaconfNEWSTORY
export OPENAI_KEY_VAR=back_up
run newstory_gpt 1.40 python scripts/pilot.py "${DRAW[@]}" \
    --models gpt-4.1-nano,gpt-4.1-mini,gpt-4.1 --tag _confNEWSTORY

echo "HOAN TAT $(date '+%Y-%m-%d %H:%M:%S')"
