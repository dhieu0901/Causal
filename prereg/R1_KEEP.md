# R1 on KEEP: do the names and the graph interact in a model that reasons at length?

Nguyễn Dương Hiếu, 27 September 2026.

This file was committed and pushed **before the first call of this run was sent**. It is not edited after that commit; deviations are reported with the results. Every analysis below is **exploratory**: the items were used in the exploratory phase, so nothing here confirms or contradicts a pre-registered hypothesis.

## 1. Why

B2 (`scripts/run_r1.sh`, 24 September 2026) sent DeepSeek-R1 the 86 causal items of the lex sample under `PSEUDO` only; the budget did not cover `KEEP`. Without `KEEP`, R1 has no difference-in-differences: we cannot say whether a correct graph offsets anonymised names for a model that reasons before it answers, the question H1 asked of GPT-4.1 (not confirmed, `prereg/CONFIRMATORY.md`). This run adds `KEEP`.

## 2. Sample, model and conditions: those of B2, with the other lexicon

- The same 86 items: `pilot.py --n 200 --kmax 1 --types DR --drop-nonsense --causal-only`, the lex sample, identical ids and labels to `results/cladder/raw/pilot_raw_r1PSEUDO.csv`.
- `KEEP` lexicon, conditions `RAW`, `ORACLE`, `DR_k1`, the same three as B2.
- The same model settings: `deepseek/deepseek-r1` through OpenRouter pinned to Novita (fp8, `runner.OPENROUTER`), temperature 0.6, at most 8,000 output tokens; every answer cut off at 8,000 is asked again at 16,000 (`--recap 16000`) and written beside the main file.
- Exact command: `bash scripts/run_r1.sh RAW,ORACLE,DR_k1 3.80 KEEP`.

## 3. What will be computed

Per item, on the causal items R1 answered in every cell a contrast needs; an unparsed answer drops out of every pair it belongs to (primary), and each estimate is also reported with the cut-offs re-asked at 16,000 tokens and with unparsed answers scored as wrong. Bootstrap over items, 4,000 draws, seed 20260907, 95% interval. No p-value decides anything.

| | Quantity |
|---|---|
| E1 | DiD `[(KEEP - PSEUDO) under RAW] - [(KEEP - PSEUDO) under ORACLE]`: does the correct graph shrink the cost of anonymised names for R1? |
| E2 | `KEEP - PSEUDO` under `RAW`: what the real names are worth to R1 with no graph |
| E3 | Under `KEEP`: `ORACLE - RAW`, `DR_k1 - RAW`, `ORACLE - DR_k1`, as B2 reported them under `PSEUDO` |
| E4 | E1 and E2 for `gpt-4.1` and Llama 3.3 70B on the same items, and R1 minus `gpt-4.1` paired by item: descriptive |

Code: `scripts/analyze_second_family.py`, section 3.

## 4. Cost

Estimated 3.10 USD: B2's billed spend under `PSEUDO` on the same items and conditions (2.87 main, 0.23 re-asks). Hard cap 3.80 USD for the run, re-asks included. The OpenRouter key has 5.45 USD left of its own 10 USD limit, shared with B5 on Llama (`scripts/run_b5_llama.sh`, cap 1.30).

## 5. Limits known in advance

- 86 items, one model: an interval on E1 is expected to be 15 points or more wide either side, so only a large interaction can be seen. A null E1 is not evidence of no interaction.
- Temperature 0.6: each arm carries sampling noise that the GPT-4.1 and Llama runs at temperature 0 do not.
- The `PSEUDO` arm was answered on 24 September, the `KEEP` arm on 27 September. The provider and quantisation are pinned, but a change in the served model between the two dates would enter E1 and E2.
- About a third of R1's answers come back in the reasoning field rather than the answer (B2); the share differs by condition. Reported as in B2.
