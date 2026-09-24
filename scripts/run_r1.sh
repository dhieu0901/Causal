#!/usr/bin/env bash
# B2: a model that reasons at length, DeepSeek-R1 through OpenRouter, pinned to
# one provider (runner.OPENROUTER). Run 2026-09-24. SPENDS CREDIT.
#
#   bash scripts/run_r1.sh [CONDS] [MAX_USD]      default RAW,ORACLE,DR_k1 3.8
#
# The 86 causal items of the lex sample, anonymised (PSEUDO): the setting where
# the names say nothing, so any gain from ORACLE is the graph's own. KEEP is not
# run: at the measured 0.014 USD per call (3,500 to 5,600 reasoning tokens on
# the smoke test) the 5 USD budget does not cover both lexicons, so R1 gets no
# DiD, only its contrasts against RAW.
#
# Temperature 0.6, DeepSeek's recommendation: R1 falls into repetition at 0, so
# unlike every GPT-4.1 and Llama number this one carries sampling noise. Output
# capped at 8,000 tokens, which covers the reasoning in 7 of 8 smoke-test
# calls; an answer cut off is kept, scored unparsed, and counted in
# results/second_family_runs.csv. The 2,000-token cap first proposed would have
# cut off every one of those 8 calls before it answered.
#
# Staged on 2026-09-24: RAW,ORACLE first (1.76 USD, 0.010 per call), then
# DR_k1 with what was left - it tests whether R1 gains from the CORRECT graph
# or from any structure block. KEEP did not fit in the budget after that. The
# last stage names all three conditions, so its file holds all three; the
# earlier calls come back from the cache at no cost.
#
# 12 of the 258 answers ran past 8,000 tokens (8 ORACLE, 4 DR_k1, no RAW): R1
# reasons about 60% longer when it is handed a graph, so the cut-offs fall on
# one side of every contrast. --recap 16000 (Novita's own maximum) asks exactly
# those 12 again and writes pilot_raw_r1PSEUDO_recap16000.csv beside the main
# file, which stays as it was.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
CONDS="${1:-RAW,ORACLE,DR_k1}"
MAXUSD="${2:-3.8}"

python scripts/pilot.py --models "deepseek/deepseek-r1" --n 200 --kmax 1 --types DR \
    --drop-nonsense --lexicon PSEUDO --causal-only --conds "$CONDS" \
    --temperature 0.6 --max-tokens 8000 --max-usd "$MAXUSD" --workers 8 --recap 16000 --tag "_r1PSEUDO" \
    > results/r1_PSEUDO_log.txt 2>&1
echo "HOAN TAT $(date '+%Y-%m-%d %H:%M:%S')"
