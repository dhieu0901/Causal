#!/usr/bin/env bash
# B6: reliance or compliance, and where the harm of a wrong graph lives.
# Pre-registered in prereg/B6.md, committed and pushed before the first call.
# SPENDS CREDIT: about 20.54 USD estimated for the main runs (token counts of
# the same models, conditions and query types on the confirmatory sample) and
# at most 1.19 USD for the probe. Hard caps: 11.30 + 11.30 + 1.30 = 23.90 USD.
#
#   bash scripts/run_b6.sh
#
# 335 fresh CLadder items, `ate` and `ett` only, drawn with every id of the
# exploratory phase and of the confirmatory draw left out
# (prereg/excluded_ids_b6.txt). Under KEEP and PSEUDO:
#
#   RAW, ORACLE, DR_k1, DR_k2, DR_k3            as in every earlier phase
#   ORACLE_NI, DR_k1_NI, DR_k2_NI, DR_k3_NI     the same prompts without the line
#                                               "Use this causal structure when
#                                               reasoning." (same draws)
#
# then the direction probe on the confirmatory items (scripts/probe_direction.py).
# Analysed by scripts/analyze_b6.py, whose functions reproduce the published
# exploratory rows before they read these records.
#
# Two accounts. Each run starts on OPENAI_API_KEY; when the API says that
# account has no credit left (runner.OUT_OF_CREDIT), the run stops sending and
# is started again on the key in the .env variable `back_up`
# (runner.make_client reads OPENAI_KEY_VAR). No key is typed or printed.
#
# A cap covers every attempt of its run together: an attempt gets what the
# earlier attempts left ("spent so far" in their logs). A run stopped by its
# cap is not tried again.

set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
COMMON=(--models gpt-4.1-nano,gpt-4.1-mini,gpt-4.1 --n 340 --seed 20260926
        --kmax 3 --sample-kmax 1 --types DR --drop-nonsense --query-types ate,ett
        --exclude-ids prereg/excluded_ids_b6.txt --no-instr
        --conds RAW,ORACLE,ORACLE_NI,DR_k1,DR_k1_NI,DR_k2,DR_k2_NI,DR_k3,DR_k3_NI)

spent_in () {
  grep -o "spent so far [0-9.]* USD" "$1" | tail -1 | awk '{print $4}'
}

# run LOG CAP COMMAND...
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

run b6_KEEP 11.30 python scripts/pilot.py "${COMMON[@]}" --lexicon KEEP --tag _b6KEEP
run b6_PSEUDO 11.30 python scripts/pilot.py "${COMMON[@]}" --lexicon PSEUDO --tag _b6PSEUDO
run b6_probe 1.30 python scripts/probe_direction.py

echo "HOAN TAT $(date '+%Y-%m-%d %H:%M:%S')"
