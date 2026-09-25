#!/usr/bin/env bash
# B1: a second model family, Llama 3.3 70B Instruct through OpenRouter, pinned
# to one provider (runner.OPENROUTER). Run 2026-09-24. SPENDS CREDIT: about
# 1.1 USD in all, measured on a 16-call smoke test (0.0002 USD per call).
#
#   bash scripts/run_llama70b.sh
#
# The same prompts GPT-4.1 was sent, on all three samples, so the pooled
# headline DiD (+5.98 pp, pool_samples.py) can be recomputed for Llama like for
# like. One model on the lex sample alone has an interval about 15 points wide
# either side, too wide to see an effect of that size.
#
#   lex        all 174 items             PROSE, RAW, ORACLE, DR_k1
#   n600       the 292 causal items      same four
#   price400   the 199 causal items      same four
#   n600       the 292 causal items      RAW under PERMUTE, IRRELEVANT, SYMBOL:
#                                        with KEEP and PSEUDO above, the five-rung
#                                        lexical ladder of analyze_ladder5.py
#
# Each command reproduces its sample exactly (checked 2026-09-24: every item
# id and label equal to the GPT-4.1 file, every prompt found in the GPT-4.1
# cache). Temperature 0 and 700 output tokens, as for GPT-4.1. --max-usd stops
# a run that costs far more than measured; every call is cached, so a re-run
# costs nothing and an interrupted run resumes. The first attempt stopped on
# n600 KEEP at 18 of 1,168 calls rate-limited (429) by the provider, which was
# serving R1 at the same time; 8 workers and longer retries fixed it, and the
# re-run paid only for the calls that had failed.
#
# Llama writes longer than GPT-4.1: 3 to 7% of its answers ran past 700 tokens,
# more under RAW than under PROSE. --recap 1500 asks exactly those again and
# writes pilot_raw{tag}_recap1500.csv beside each main file, which stays as it
# was; analyze_second_family.py reports every headline contrast both ways.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
M="meta-llama/llama-3.3-70b-instruct"

# A run stops when more than 1% of its calls fail (runner.guard_errors). For a
# batch of 30 re-asks one 429 is already 3%, and the provider rate-limits in
# bursts, so a stopped run is tried again after a pause: what succeeded is
# cached, and only the failed calls are sent again.
run () {
  local log="$1"; shift
  local rc
  for attempt in 1 2 3; do
    echo ">>> $log   attempt $attempt   [$(date '+%H:%M:%S')]"
    rc=0
    python scripts/pilot.py --models "$M" --workers 8 --recap 1500 "$@" \
        > "results/cladder/logs/$log" 2>&1 || rc=$?
    if [ "$rc" -eq 0 ]; then
      echo "<<< $log done   [$(date '+%H:%M:%S')]"
      return 0
    fi
    echo "... $log stopped, exit=$rc; trying again in 60 s" >&2
    sleep 60
  done
  echo "!!! $log FAILED, exit=$rc. See results/cladder/logs/$log" >&2
  exit "$rc"
}

for LEX in KEEP PSEUDO; do
  run "llama70b_lex_${LEX}_log.txt" --n 200 --kmax 1 --types DR --drop-nonsense \
      --lexicon "$LEX" --max-usd 0.3 --tag "_llama70b$LEX"
  run "llama70b_n600_${LEX}_log.txt" --n 600 --kmax 1 --types DR --drop-nonsense \
      --lexicon "$LEX" --causal-only --max-usd 0.6 --tag "_llama70b_n600$LEX"
  run "llama70b_price400_${LEX}_log.txt" --n 400 --kmax 3 --types DR,ED,FE \
      --drop-nonsense --lexicon "$LEX" --conds PROSE,RAW,ORACLE,DR_k1 --causal-only \
      --max-usd 0.4 --tag "_llama70b_price400$LEX"
done

for LEX in PERMUTE IRRELEVANT SYMBOL; do
  run "llama70b_n600ladder_${LEX}_log.txt" --n 600 --kmax 1 --types DR --drop-nonsense \
      --lexicon "$LEX" --conds RAW --causal-only --max-usd 0.15 \
      --tag "_llama70b_n600ladder$LEX"
done

echo "HOAN TAT $(date '+%Y-%m-%d %H:%M:%S')"
