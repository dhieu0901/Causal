#!/usr/bin/env bash
# B6 and the path probe on Llama 3.3 70B Instruct (OpenRouter, pinned to
# Novita, bf16). Pre-registered in prereg/B6_LLAMA.md, committed and pushed
# before the first call. SPENDS CREDIT: about 1.12 USD estimated from the
# measured per-call spend of B5 on Llama (0.49 per lexicon, 0.14 for the
# probe); hard caps 0.65 + 0.65 + 0.30 = 1.60 USD.
#
#   bash scripts/run_b6_llama.sh
#
# The draw is run_b6.sh's, so every item, graph and prompt is the one the
# GPT-4.1 models were sent. Temperature 0, 700 output tokens; every B6 answer
# cut off at 700 is asked again at 1,500 into pilot_raw{tag}_recap1500.csv.
# Analysed by analyze_b6.py (section 4) and analyze_path_probe.py.
#
# A cap covers every attempt of its run together: an attempt gets what the
# earlier attempts left ("spent so far" in their logs). A run stopped by its
# cap is not tried again. A run stopped by provider errors is tried again
# after a pause; what succeeded is cached and costs nothing to replay.

set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
LLAMA=meta-llama/llama-3.3-70b-instruct
COMMON=(--models "$LLAMA" --workers 8 --recap 1500 --n 340 --seed 20260926
        --kmax 3 --sample-kmax 1 --types DR --drop-nonsense --query-types ate,ett
        --exclude-ids prereg/excluded_ids_b6.txt --no-instr
        --conds RAW,ORACLE,ORACLE_NI,DR_k1,DR_k1_NI,DR_k2,DR_k2_NI,DR_k3,DR_k3_NI)

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

run b6llama_KEEP 0.65 python scripts/pilot.py "${COMMON[@]}" --lexicon KEEP --tag _b6llamaKEEP
run b6llama_PSEUDO 0.65 python scripts/pilot.py "${COMMON[@]}" --lexicon PSEUDO --tag _b6llamaPSEUDO
run b6llama_probe 0.30 python scripts/probe_path.py --models "$LLAMA" --tag _llama --workers 8

echo "HOAN TAT $(date '+%Y-%m-%d %H:%M:%S')"
