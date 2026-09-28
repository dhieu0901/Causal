# NAMES_PREMISE: the closed-world clause, and which names carry the gain

Registered before the first call of this study: this file, the code that sends
the calls (`scripts/run_names_premise.sh`, `scripts/pilot.py --open-raw`, the
lexicons `PSEUDO_XY` and `PSEUDO_THIRD` in `src/lexical.py`, the condition
`RAW_OPEN` in `src/prompts.py`) and the code that analyses them
(`scripts/analyze_names_premise.py`) are committed and pushed together. Nothing
in this file is changed after that commit; any deviation is reported as one.

## 1. Questions

Study 1 left two questions open.

**Q1, the premise.** CLadder's opening sentence reads "Imagine a
self-contained, hypothetical world with only the following conditions, and
without any unmentioned factors or causal relationships". `RAW` removes every
sentence that states a causal relationship, so read literally a `RAW` question
implies "no effect". With pseudowords and no graph, Llama 3.3 70B says Yes to
28.27% of Study 1's questions, where 51% of the labels are Yes. Does the clause
produce that lean towards No, and is part of the cost of anonymising (`KEEP`
minus `PSEUDO` under `RAW`) a cost of the clause?

**Q2, which names.** Real names beat pseudowords under `RAW`. Is the gain
carried by the names of treatment and outcome, or by the names of the other
variables, which can signal their role (for instance "unobserved confounders")?

## 2. Sample, model, conditions

- Questions: Study 1's 484 CLadder questions, the draw of
  `scripts/run_b5_llama.sh` (`--n 1000 --seed 20260925 --sample-kmax 1
  --drop-nonsense --causal-only --exclude-ids prereg/excluded_ids.txt`).
- Model: Llama 3.3 70B Instruct through OpenRouter, pinned to one provider
  (Novita, bf16) as in Study 1, temperature 0, one user message, no system
  prompt, 700 output tokens; every answer cut off at 700 is asked again at
  1,500 into a separate file.
- Reused cells (Study 1, already answered): `KEEP` and `PSEUDO` under `RAW`.
- New cells (1,636 new calls; 150 prompts of `PSEUDO_XY` and 150 of
  `PSEUDO_THIRD` equal an already answered `RAW` prompt and are served from the
  cache):
  - `RAW_OPEN` under `KEEP` and under `PSEUDO`: `RAW` with ", and without any
    unmentioned factors or causal relationships" removed from the opening
    sentence, nothing else changed.
  - `PSEUDO_XY` under `RAW`: `PSEUDO`'s own pseudowords on treatment and outcome
    only; every other variable keeps CLadder's name.
  - `PSEUDO_THIRD` under `RAW`: `PSEUDO`'s own pseudowords on every variable
    except treatment and outcome.
  With the same pseudowords as `PSEUDO`, the cells `KEEP`, `PSEUDO_XY`,
  `PSEUDO_THIRD` and `PSEUDO` form a 2 x 2: treatment and outcome named or not,
  crossed with the other variables named or not.

## 3. Tests

Every difference is taken per question, on parsed answers, as in Study 1.

| Test | Quantity | Predicted |
|---|---|---|
| P1 | share of Yes answers under `PSEUDO`: `RAW_OPEN` minus `RAW` | positive |
| P2 | accuracy: (`KEEP` - `PSEUDO` under `RAW`) - (`KEEP` - `PSEUDO` under `RAW_OPEN`), questions answered in all four cells | positive |
| N1 | accuracy: `KEEP` - `PSEUDO_XY` under `RAW` | positive |
| N2 | accuracy: `KEEP` - `PSEUDO_THIRD` under `RAW`, on the questions whose `RAW` prompt names another variable (334 of 484) | positive |

On 150 questions the other variables appear only in the sentences of structure
that `RAW` removes, so the `RAW` prompt is the same under `KEEP` and
`PSEUDO_THIRD`; N2 leaves them out, and N2 on every question is reported as
descriptive.

## 4. Decision rule

- Bootstrap: stories resampled with all their questions (the names are the
  story's), 4,000 draws; percentile 95% interval; two-sided bootstrap p
  (`stats.boot_p`).
- Holm across the four tests, family-wise alpha 0.05. A test is confirmed when
  its Holm-adjusted p is below 0.05 and its sign is the predicted one,
  contradicted when the p is below 0.05 and the sign is the other one, and not
  confirmed otherwise.
- Seeds 20260907 (primary), 1, 2, 3, 4; a verdict that changes across them is
  reported as borderline.
- Sensitivity analyses, reported beside the primary one: questions resampled
  instead of stories; unparsed answers scored as wrong; answers cut off at 700
  tokens replaced by their re-ask at 1,500.

## 5. Checks before the new data are read

`analyze_names_premise.py` must reproduce Study 1's Llama cells `KEEP` and
`PSEUDO` under `RAW` (their n, accuracy and share of Yes answers in
`results/cladder/accuracy_cells.csv`) or stop.

## 6. Descriptive, no verdicts

Every cell's accuracy, share of Yes answers and balanced accuracy; each test on
gold-Yes and gold-No questions and as a change in the share of Yes answers;
`KEEP` - `PSEUDO` under `RAW` and under `RAW_OPEN`; `RAW_OPEN` minus `RAW` under
each name condition; `PSEUDO_XY` and `PSEUDO_THIRD` against `PSEUDO`.

## 7. Scope and cost

One model family. The GPT-4.1 family is not run: three models on four cells
would cost several times the budget left for this study. Cost: about 0.22 USD
estimated from the measured spend of Study 1's `RAW` ladder runs on Llama
(0.06 USD per 484 calls), plus re-asks; hard caps 0.12 USD per cell, 0.48 USD
in all, enforced by the run script.
