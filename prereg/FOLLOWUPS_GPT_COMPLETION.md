# FOLLOWUPS_GPT_COMPLETION: finishing the GPT-4.1 family's `PSEUDO_THIRD` cell

Nguyễn Dương Hiếu, Nguyễn Đại Quân, Trương Hoàng Tùng, Triệu Hải Đăng Trình. Registered 6 November 2026: the commit that adds this file is the registration, made before any call of this run. It is not edited after that commit; deviations are reported with the results.

## 1. What, and why

`prereg/FOLLOWUPS_FAMILIES.md` registered P1, P2, N1 and N2 on the GPT-4.1 family. Its run stopped on 29 September 2026 in the family's last cell, `PSEUDO_THIRD` under `RAW`, when the OpenAI key reached its project's spending limit, and it was not resumed. That family's result stays as reported: P1, P2 and N1 with intervals and no verdicts, N2 not computable, "incomplete run, no verdict".

This file registers a completion of that one cell, so that the registered Holm family of four can be tested on the GPT-4.1 family. It is reported as a completion registered after the other results were known, beside the original result, not in its place.

## 2. What exists and is known when this is written

- `gpt-4.1-nano` has answered all 484 questions of the cell. `gpt-4.1-mini` has 196 answers and `gpt-4.1` 150: on 150 questions the `PSEUDO_THIRD` prompt equals the `KEEP` prompt under `RAW`, whose answer is reused as registered, and `gpt-4.1-mini` answered 46 more before the stop. These answers are used as they are; nothing is re-asked.
- `gpt-4.1-nano` is no longer served by OpenAI (`shutdown_date` 2026-10-23 in its model list); it is not called, and none of its calls is missing. `gpt-4.1-mini` and `gpt-4.1` have the `shutdown_date` 2027-01-14.
- Every result of `prereg/FOLLOWUPS_FAMILIES.md` is known, including this family's P1 +5.03, P2 +2.45 and N1 +12.86 (no verdicts) and N2 on Llama 3.3 70B (borderline) and `gpt-5.6-luna` (not confirmed).

## 3. Calls and cost

- The command of `scripts/run_followups_families.sh` for this cell, unchanged: Study 1's draw (`--n 1000 --seed 20260925 --kmax 3 --sample-kmax 1 --types DR --drop-nonsense --causal-only --exclude-ids prereg/excluded_ids.txt`), `--lexicon PSEUDO_THIRD --conds RAW --tag _premisePSEUDO_THIRD`, the three GPT-4.1 models, temperature 0, at most 700 output tokens, no re-asks. `pilot.py --dry-run` on 6 November: `gpt-4.1-nano` 484 cached, 0 new; `gpt-4.1-mini` 196 cached, 288 new; `gpt-4.1` 150 cached, 334 new.
- Cost, from the measured tokens of the same models' `KEEP` cell under `RAW`: 0.12 USD for `gpt-4.1-mini` and 0.63 USD for `gpt-4.1`, 0.75 USD in all.
- It runs after B8 (`prereg/B8.md`), on the same OpenAI credit of 7 USD. The cap is 0.90 USD or, if less, the credit B8 leaves (7 USD less B8's recorded spend) less 0.02 USD; if that is below 0.80 USD the run does not start, and the results say so (`scripts/run_gpt_completion.sh`). A run stopped by its cap or by the provider is not resumed, and the family stays without verdicts.

## 4. Tests and decision rule

P1, P2, N1 and N2 exactly as in `prereg/NAMES_PREMISE.md` section 3 and `prereg/FOLLOWUPS_FAMILIES.md` sections 3 and 4: every difference per (model, question) on parsed answers, averaged over every cell; stories resampled with all their questions and model rows, 4,000 draws, percentile intervals; two-sided bootstrap p; Holm across the four, family-wise alpha 0.05; seeds 20260907 (primary), 1, 2, 3, 4, a verdict that changes with the seed reported as borderline. Sensitivity: questions resampled instead of stories; unparsed answers scored as wrong. Computed by `scripts/analyze_names_premise.py`, whose tests are not changed.

## 5. Limits known in advance

- The family's other results were known; only the decision to complete the cell is new, not the tests.
- The completed cell mixes answers given on 28 and 29 September with answers given in November by the same model snapshots.
