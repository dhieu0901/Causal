# FOLLOWUPS_FAMILIES: the new-story and names-premise tests on more families

Nguyễn Dương Hiếu, Nguyễn Đại Quân, Trương Hoàng Tùng, Triệu Hải Đăng Trình. 28 September 2026.

Registered before the first call of this study: this file, the run script
(`scripts/run_followups_families.sh`) and the analysis code
(`scripts/analyze_new_stories.py`, which gains the family `luna`, and
`scripts/analyze_names_premise.py`, which gains the families `luna` and `gpt`)
are committed and pushed together. Nothing here is changed after that commit;
any deviation is reported as one.

## 1. Why, and what is already known

Two registered follow-ups ran on 28 September 2026 on fewer families than Study 1:

- `prereg/NEW_STORIES.md` (GPT-4.1 family and Llama 3.3 70B): S1 confirmed in
  both (new stories beat letters by +11.09 and +15.73); S2 confirmed for the
  GPT-4.1 family (-0.29), not for Llama, which scores higher on the new
  stories (+4.10); S3 not confirmed in either.
- `prereg/NAMES_PREMISE.md` (Llama 3.3 70B only): P1 confirmed (+5.53), P2
  not confirmed (-1.31), N1 confirmed (+14.26), N2 borderline and negative
  (-5.36).

This registration puts the same tests, unchanged, to the families that did not
run them: S1 to S3 on `gpt-5.6-luna`; P1, P2, N1 and N2 on `gpt-5.6-luna` and
on the GPT-4.1 family. Each is a separate family with its own verdicts; nothing
is pooled with, or replaces, the earlier results. The earlier results were known
when this file was written.

## 2. Sample, models, conditions

- Questions: Study 1's 484 CLadder questions, the same draw for every family
  (`--n 1000 --seed 20260925 --kmax 3 --sample-kmax 1 --types DR
  --drop-nonsense --causal-only --exclude-ids prereg/excluded_ids.txt`).
- `gpt-5.6-luna` at its defaults, as in B7 (`prereg/B7.md`): temperature not
  set (the model refuses 0; the cache records 1.0), at most 4,000 output
  tokens, one answer per question, no re-asks.
- The GPT-4.1 family (`gpt-4.1-nano`, `gpt-4.1-mini`, `gpt-4.1`; snapshots
  `2025-04-14`): temperature 0, at most 700 output tokens, no re-asks, as in
  Study 1.
- One user message, no system prompt. OpenAI key `back_up`.
- Reused, already answered: `gpt-5.6-luna`'s `KEEP`, `PSEUDO` and `SYMBOL`
  cells under `RAW` (B7, `pilot_raw_lunaconf*`); the GPT-4.1 family's `KEEP`
  and `PSEUDO` cells under `RAW` (Study 1, `pilot_raw_conf*`).
- New cells, built by the code of the two earlier registrations
  (`src/new_stories.py`, `src/lexical.py`, `src/prompts.py`):
  - `gpt-5.6-luna`: `NEWSTORY` under `RAW` (484 calls).
  - `gpt-5.6-luna` and each GPT-4.1 model: `RAW_OPEN` under `KEEP` and under
    `PSEUDO`; `PSEUDO_XY` and `PSEUDO_THIRD` under `RAW`. 1,636 new calls per
    model: on 150 questions the `PSEUDO_XY` prompt equals the `PSEUDO` prompt
    and the `PSEUDO_THIRD` prompt equals the `KEEP` prompt under `RAW`, and
    the cached answer is reused, as for Llama.

## 3. Tests

Identical to the earlier registrations: S1, S2 and S3 as in
`prereg/NEW_STORIES.md` section 4; P1, P2, N1 and N2 as in
`prereg/NAMES_PREMISE.md` section 3. For the GPT-4.1 family every difference is
taken per (model, question) on parsed answers and averaged over every cell, as
H3b is; for `gpt-5.6-luna` there is one model per question.

## 4. Decision rule

As in the earlier registrations: stories resampled with all their questions
(and all their model rows), 4,000 draws, percentile intervals; two-sided
bootstrap p for S1, P1, P2, N1, N2, the larger one-sided p for the
equivalence tests S2 and S3 (margin 5 points, 90% interval). Holm within each
study and family: across S1 to S3 for `gpt-5.6-luna`; across P1, P2, N1, N2
for `gpt-5.6-luna` and, separately, for the GPT-4.1 family; family-wise alpha
0.05. Confirmed, contradicted and not confirmed as in the earlier
registrations. Seeds 20260907 (primary), 1, 2, 3, 4; a verdict that changes
across them is reported as borderline.

Sensitivity, beside the primary analysis: questions resampled instead of
stories (a question with all its model rows); unparsed answers scored as
wrong. No re-ask analysis: these models' answers are not re-asked.

## 5. Checks before the new data are read

- `analyze_new_stories.py` must reproduce `gpt-5.6-luna`'s published H3b
  (13.02, `results/cladder/confirmatory.csv`) from the reused `KEEP` and
  `SYMBOL` answers, or stop. At this commit it does.
- `analyze_names_premise.py` must reproduce each family's Study 1 `KEEP` and
  `PSEUDO` cells under `RAW` (n, accuracy and share of Yes answers in
  `results/cladder/accuracy_cells.csv`), or stop. At this commit it does for
  all three families (Llama 476 and 474 rows, `gpt-5.6-luna` 484 and 484, the
  GPT-4.1 family 1,414 and 1,422).
- The code change leaves every earlier number unchanged: rerun on the earlier
  records, both scripts give the same Llama and GPT-4.1 rows as before (the
  names-premise files gain a `family` column).

## 6. How the result is read

As in the earlier registrations. In addition: a test that holds in a new
family extends the earlier reading to it; a test that fails in a new family is
reported beside the earlier ones and limits them to the families where they
held.

## 7. Cost and order

Estimated with `pilot.py --dry-run --cost-ref`, each new call priced at the
same question's measured output tokens in the reused cell: `gpt-5.6-luna`
new stories 0.16 USD; `gpt-5.6-luna` names-premise 0.67 USD (0.16, 0.21,
0.17, 0.13); GPT-4.1 family names-premise 4.26 USD (1.20, 1.09, 0.93, 1.04).
Hard caps, enforced by the run script: 0.40; 4 x 0.30; 1.45, 1.35, 1.15,
1.30. The runs go one after another in that order, and a run stopped by its
cap is not resumed.

## 8. Limits known in advance

- `gpt-5.6-luna` gives one answer per question at temperature 1.
- The tests, their direction and the decision rule are those of the earlier
  registrations, whose results were known when this file was written.
- The new stories keep CLadder's template (`prereg/NEW_STORIES.md` section 8).
