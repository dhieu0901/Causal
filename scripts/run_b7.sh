#!/usr/bin/env bash
# B7: the six confirmatory tests on a current reasoning model, gpt-5.6-luna.
# Pre-registered in prereg/B7.md, committed and pushed before the first call.
# SPENDS CREDIT: about 5.76 USD estimated from 12 uncached feasibility calls
# (mean 209 input and 760 output tokens); hard caps 3.30 + 3.30 + 3 x 0.80 =
# 9.00 USD.
#
#   bash scripts/run_b7.sh
#
# The confirmatory sample exactly (prereg/CONFIRMATORY.md): the same draw, the
# same 484 causal items, the same prompts and conditions,
#
#   KEEP, PSEUDO                     RAW, ORACLE, DR_k1, DR_k2, DR_k3
#   PERMUTE, IRRELEVANT, SYMBOL      RAW
#
# on one model at its defaults: reasoning effort not set, temperature not set
# (the model refuses 0; runner.call sends no temperature when it is 1.0, and the
# cache key records 1.0), at most 4,000 output tokens, reasoning included.
# Analysed by scripts/analyze_confirmatory.py as family "luna".
#
# Keys, caps and retries work as in scripts/run_b6.sh: a cap covers every
# attempt of its run, and a key out of credit hands over to `back_up`.

set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
COMMON=(--models gpt-5.6-luna --n 1000 --seed 20260925 --kmax 3 --sample-kmax 1
        --types DR --drop-nonsense --causal-only --exclude-ids prereg/excluded_ids.txt
        --temperature 1.0 --max-tokens 4000)

spent_in () {
  grep -o "spent so far [0-9.]* USD" "$1" | tail -1 | awk '{print $4}'
}

run () {
  local log="$1" left="$2"; shift 2
  local key="OPENAI_API_KEY" rc s f
  for attempt in 1 2 3 4; do
    f="results/cladder/logs/${log}_a${attempt}.txt"
    echo ">>> $log   attempt $attempt   key variable $key   cap left $left USD   [$(date '+%H:%M:%S')]"
    rc=0
    OPENAI_KEY_VAR="$key" "$@" --max-usd "$left" > "$f" 2>&1 || rc=$?
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
    if grep -q "OUT OF CREDIT" "$f"; then
      if [ "$key" = "back_up" ]; then
        echo "!!! both keys are out of credit. Stopping." >&2
        exit "$rc"
      fi
      echo "... the first key is out of credit; moving to the key in back_up" >&2
      key="back_up"
      continue
    fi
    echo "... $log stopped, exit=$rc; trying again in 60 s" >&2
    sleep 60
  done
  echo "!!! $log FAILED, exit=$rc. See $f" >&2
  exit "$rc"
}

P=(python scripts/pilot.py "${COMMON[@]}")
run b7_KEEP 3.30 "${P[@]}" --lexicon KEEP --conds RAW,ORACLE,DR_k1,DR_k2,DR_k3 --tag _lunaconfKEEP
run b7_PSEUDO 3.30 "${P[@]}" --lexicon PSEUDO --conds RAW,ORACLE,DR_k1,DR_k2,DR_k3 --tag _lunaconfPSEUDO
run b7_PERMUTE 0.80 "${P[@]}" --lexicon PERMUTE --conds RAW --tag _lunaconfladderPERMUTE
run b7_IRRELEVANT 0.80 "${P[@]}" --lexicon IRRELEVANT --conds RAW --tag _lunaconfladderIRRELEVANT
run b7_SYMBOL 0.80 "${P[@]}" --lexicon SYMBOL --conds RAW --tag _lunaconfladderSYMBOL

echo "HOAN TAT $(date '+%Y-%m-%d %H:%M:%S')"
