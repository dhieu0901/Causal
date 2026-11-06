#!/usr/bin/env bash
# B8: a graph offered as possibly wrong, and deleted and spurious edges, on
# fresh ate and ett questions with real names, in three families with separate
# verdicts. Pre-registered in prereg/B8.md, committed and pushed before the
# first call. The calls sent are exactly those listed in prereg/b8_selection.csv
# (scripts/b8_select.py), 1,650 per model.
# SPENDS CREDIT. Estimated from the measured tokens of the closest prompts
# already paid for (B6's instruction-free graph prompts for the GPT-4.1 models
# and Llama, B7's ate/ett graph prompts for gpt-5.6-luna): gpt-4.1-mini and
# gpt-4.1 4.70 USD (cap 5.10) and gpt-5.6-luna 1.32 USD (cap 1.75) on the
# OpenAI key (OPENAI_API_KEY), funded with 7 USD of credit; Llama 3.3 70B
# 0.61 USD (cap 0.90) on the second OpenRouter key (backup_openrouter, limited to
# 2 USD), pinned to CoreWeave (fp16): Novita, which served every earlier Llama
# run in bf16, no longer lists the model.
# gpt-4.1-nano is no longer served (shutdown_date 2026-10-23). Approved 06/11.
#
#   bash scripts/run_b8.sh
#
# The families run one at a time, in this order. The GPT-4.1 pair at
# temperature 0 and 700 output tokens; gpt-5.6-luna at temperature 1, the only
# value it accepts, and 4,000 output tokens, as in B7; Llama at temperature 0
# and 700 output tokens on its pinned provider. No re-asks. Analysed by
# analyze_b8.py.
#
# A cap covers every attempt of its run together: an attempt gets what the
# earlier attempts left ("spent so far" in their logs). A run stopped by its
# cap is not tried again, and nothing after it runs; that family is reported
# without verdicts (prereg/B8.md, section 8). A run stopped by provider errors
# is tried again after a pause; what succeeded is cached and costs nothing to
# replay.

set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
export OPENAI_KEY_VAR=OPENAI_API_KEY
DRAW=(--n 340 --seed 20260927 --kmax 3 --sample-kmax 1 --types DR,ED,FE --drop-nonsense
      --query-types ate,ett --exclude-ids prereg/excluded_ids_b8.txt --no-instr --uncertain
      --lexicon KEEP --only prereg/b8_selection.csv)
GPT=(--models gpt-4.1-mini,gpt-4.1)
LUNA=(--models gpt-5.6-luna --temperature 1.0 --max-tokens 4000)
LLAMA=(--models meta-llama/llama-3.3-70b-instruct --workers 8)

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

run b8_gpt 5.10 "${P[@]}" "${GPT[@]}" --tag _b8KEEP
run b8_luna 1.75 "${P[@]}" "${LUNA[@]}" --tag _b8lunaKEEP
# Llama on the second OpenRouter key, pinned to CoreWeave (runner.provider).
export OPENROUTER_KEY_VAR=backup_openrouter
export OPENROUTER_PIN="meta-llama/llama-3.3-70b-instruct=CoreWeave"
run b8_llama 0.90 "${P[@]}" "${LLAMA[@]}" --tag _b8llamaKEEP

echo "HOAN TAT $(date '+%Y-%m-%d %H:%M:%S')"
