#!/usr/bin/env bash
# The new-story tests on gpt-5.6-luna, and the names-premise tests on
# gpt-5.6-luna and the GPT-4.1 family. Pre-registered in
# prereg/FOLLOWUPS_FAMILIES.md, committed and pushed before the first call.
# SPENDS CREDIT, estimated with pilot.py --dry-run --cost-ref from the measured
# tokens of the reused RAW cells: luna new stories 0.16 USD (cap 0.40); luna
# names-premise 0.67 USD (caps 4 x 0.30); GPT-4.1 family names-premise 4.26 USD
# (caps 1.45, 1.35, 1.15, 1.30). Approved on 28/09 over the default OpenAI budget.
#
#   bash scripts/run_followups_families.sh
#
# Study 1's draw, so every question is one of its 484. gpt-5.6-luna at its
# defaults as in B7 (temperature not set, 4,000 output tokens); the GPT-4.1
# family at temperature 0, 700 output tokens. No re-asks. Analysed by
# analyze_new_stories.py (family luna) and analyze_names_premise.py (families
# luna and gpt).
#
# A cap covers every attempt of its run together: an attempt gets what the
# earlier attempts left ("spent so far" in their logs). A run stopped by its
# cap is not tried again. A run stopped by provider errors is tried again
# after a pause; what succeeded is cached and costs nothing to replay.

set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
export OPENAI_KEY_VAR=back_up
DRAW=(--n 1000 --seed 20260925 --kmax 3 --sample-kmax 1 --types DR --drop-nonsense
      --causal-only --exclude-ids prereg/excluded_ids.txt)
LUNA=(--models gpt-5.6-luna --temperature 1.0 --max-tokens 4000)
GPT=(--models gpt-4.1-nano,gpt-4.1-mini,gpt-4.1)

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

P=(python scripts/pilot.py "${DRAW[@]}")

run fam_luna_NEWSTORY 0.40 "${P[@]}" "${LUNA[@]}" --lexicon NEWSTORY --conds RAW --tag _lunaconfNEWSTORY

run fam_luna_KEEP 0.30 "${P[@]}" "${LUNA[@]}" --lexicon KEEP --open-raw --conds RAW_OPEN --tag _lunapremiseKEEP
run fam_luna_PSEUDO 0.30 "${P[@]}" "${LUNA[@]}" --lexicon PSEUDO --open-raw --conds RAW_OPEN --tag _lunapremisePSEUDO
run fam_luna_PSEUDO_XY 0.30 "${P[@]}" "${LUNA[@]}" --lexicon PSEUDO_XY --conds RAW --tag _lunapremisePSEUDO_XY
run fam_luna_PSEUDO_THIRD 0.30 "${P[@]}" "${LUNA[@]}" --lexicon PSEUDO_THIRD --conds RAW --tag _lunapremisePSEUDO_THIRD

run fam_gpt_KEEP 1.45 "${P[@]}" "${GPT[@]}" --lexicon KEEP --open-raw --conds RAW_OPEN --tag _premiseKEEP
run fam_gpt_PSEUDO 1.35 "${P[@]}" "${GPT[@]}" --lexicon PSEUDO --open-raw --conds RAW_OPEN --tag _premisePSEUDO
run fam_gpt_PSEUDO_XY 1.15 "${P[@]}" "${GPT[@]}" --lexicon PSEUDO_XY --conds RAW --tag _premisePSEUDO_XY
run fam_gpt_PSEUDO_THIRD 1.30 "${P[@]}" "${GPT[@]}" --lexicon PSEUDO_THIRD --conds RAW --tag _premisePSEUDO_THIRD

echo "HOAN TAT $(date '+%Y-%m-%d %H:%M:%S')"
