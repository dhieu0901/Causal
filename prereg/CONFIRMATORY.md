# Confirmatory phase: pre-registration

Nguyễn Dương Hiếu, 25 September 2026.

This file was committed and pushed **before the first call of this phase was sent**. The commit that adds it is the registration, and its timestamp is on GitHub. It is not edited after that commit; deviations are reported with the results, not written into this file.

## 1. Why this phase exists

The hypotheses tested below were formed during the exploratory phase, after its data were seen, on the same CLadder questions that produced them. Those questions cannot test them again. This phase asks the same models questions they were never asked, with the tests, their direction and the decision rule fixed here.

## 2. Sample

- Source: CLadder v1.5, `data/full_v1.5_default.csv`, stories whose `story_id` starts with `nonsense` left out, as in the exploratory phase.
- Excluded: every CLadder id in `prereg/excluded_ids.txt`, 1,121 ids. That is the union of the `id` column over `results/raw/*.csv`, `results/_itemmap_*.csv` and `results/drift_check_raw*.csv` at commit `3c973a4`, i.e. every question asked in the exploratory phase. SHA-256 of the id lines (without the three comment lines): `bc057c8a2904fe97c997fbfe9b741dd6f053a40018dc9f1e4242ff8cdc6b62f0`.
- Draw: `pilot.make_items(n=1000, seed=20260925, kmax=1, drop_nonsense=True, exclude_ids=...)`, the exploratory sampler: an equal number of questions from each (graph family, rung) stratum. 986 questions, all ten graph families.
- Only the causal query types are sent (`pilot.py --causal-only`: every type except `marginal`, `correlation` and `backadj`), about 484 questions per condition.
- The runner answers each distinct prompt once. A few prompts are textually identical to an earlier prompt from a different CLadder id, mostly after relabelling to letters, and get the cached answer. The dry run counted 4 to 5 of 2,292 `KEEP` prompts per model and 9 of 484 `SYMBOL` prompts.

## 3. Models and conditions

- `gpt-4.1-nano`, `gpt-4.1-mini`, `gpt-4.1` through the OpenAI API, temperature 0, at most 700 output tokens, prompts built by `src/prompts.py` and `src/lexical.py` as of this commit.
- `KEEP` and `PSEUDO` lexicons: `RAW`, `ORACLE`, `DR_k1`, `DR_k2`, `DR_k3` (k reversed edges, drawn per question with seed 20260925; `DR_k3` exists on the seven families that have three reversible edges).
- `PERMUTE`, `IRRELEVANT`, `SYMBOL` lexicons: `RAW`.
- Exact commands: `scripts/run_confirmatory.sh`. Estimated cost 19.22 USD, from the measured token counts of the same models, conditions and query types on the exploratory n600 sample. Hard cap 20.20 USD across the five runs; a run stopped by its cap is not retried.
- Second model family, not part of the decision below: Llama 3.3 70B Instruct through OpenRouter, pinned to Novita (bf16), the same five commands with `--models meta-llama/llama-3.3-70b-instruct --workers 8 --recap 1500` and tags `_llamaconf*`, run when the OpenRouter budget allows (about 1 USD). Analysed by the same script as family `llama` and reported separately as a replication.

## 4. The six tests

Common definitions. An answer is correct (1) or not (0). The causal group is every query type except `marginal`, `correlation` and `backadj`. Primary analysis: an unparsed answer drops out of every pair it belongs to, as in the exploratory phase. All effects are in percentage points.

| Test | Quantity | Prediction |
|---|---|---|
| H1 | Per question, `[(KEEP - PSEUDO) under RAW] - [(KEEP - PSEUDO) under ORACLE]`, one value per model, averaged over every (question, model) cell | positive: giving the correct graph shrinks the cost of anonymised names |
| H2 | Per question, `ORACLE - DR_k1` under `PSEUDO`, same averaging | positive: the model uses the direction of the edges it is given |
| H3a | Per question, `IRRELEVANT - PERMUTE` under `RAW` | equivalent to zero within 5 points: a wrong prior costs no more than no prior |
| H3b | Per question, `KEEP - SYMBOL` under `RAW` | positive: the correct prior carried by real names is worth something |
| H4-KEEP | On the seven families with k = 3, on questions whose one-edge reversal already changes the estimand: per question, the slope `(D3 - D1) / 2` with `Dk = (DR_k - RAW)` averaged over models; mean over questions | negative: harm grows with the number of reversed edges |
| H4-PSEUDO | the same under `PSEUDO` | negative |

"Changes the estimand" is the rule of `scripts/classify_perturbations.py` at this commit: in the corrupted graph Y stops (or starts) being a descendant of X, or the set of valid backdoor adjustment sets for (X, Y) is different. For `nde`, `nie` and `det-counterfactual` a reversal also counts as changing the estimand when it alters an edge that lies on a directed path from X to Y in either graph.

## 5. Decision rule

- Bootstrap over questions, 4,000 draws. Two-sided p is `stats.boot_p` (floor 2/4000). H3a is two one-sided tests against -5 and +5; its p is the larger one-sided bootstrap p (floor 1/4000), which amounts to the 90% bootstrap interval lying inside (-5, +5).
- The six p-values are adjusted by Holm, family-wise alpha = 0.05.
- **Confirmed**: adjusted p < 0.05 and the estimate in the predicted direction (H3a: adjusted TOST p < 0.05). **Contradicted**: adjusted p < 0.05 in the opposite direction. Otherwise **not confirmed**.
- Seed rule: the whole procedure is repeated at bootstrap seeds 20260907 (primary), 1, 2, 3 and 4. A verdict that is not the same at all five is reported as borderline, not as confirmed.
- Every estimate is reported with its 95% interval whatever its verdict.

## 6. Sensitivity analyses, fixed now, which do not change a verdict

- Unparsed answers scored as wrong instead of dropped.
- Llama only: every answer cut off at 700 tokens asked again with 1,500 (`--recap`), both versions reported.

## 7. The analysis code is the exploratory analysis

`scripts/analyze_confirmatory.py` computes all six tests. Before it reads a confirmatory record it runs the same functions on the exploratory n600 sample and stops unless they return the published n600 rows exactly. At this commit they do:

| Test | n600, recomputed = published | Source |
|---|---|---|
| H1 | +6.78 [+1.65 ; +12.08] | `results/pooled_headline.csv` |
| H2 | +8.19 [+4.37 ; +12.02] | `results/structure_arms.csv` |
| H3a | -0.83 [-4.49 ; +2.70] | `results/ladder5_steps_n600.csv` |
| H3b | +10.93 [+7.17 ; +14.89] | `results/ladder5_steps_n600.csv` |
| H4-KEEP | -3.28 [-5.90 ; -0.66] | `results/perturbation_conditional_slope.csv` |
| H4-PSEUDO | -1.76 [-4.28 ; +0.68] | `results/perturbation_conditional_slope.csv` |

These are exploratory numbers, not evidence for the hypotheses. They say what to expect: run through the decision rule above as a code test, the n600 data would give five of six confirmed and H4-PSEUDO not confirmed, so H4-PSEUDO is the least powered of the six.

## 8. What will not be done

- No test is added to or taken out of the six after this commit. Any other analysis of these records is labelled exploratory.
- No data are replaced. A run that stops resumes from the cache, which returns the answers already received.
- If the spending cap stops a run before all five finish, a test is computed only if all of its data exist; the others are reported as not run.

## 9. Limits known in advance

- Same benchmark and same prompt formats. This phase guards against hypotheses fitted to the questions that suggested them, not against CLadder itself.
- The model names are aliases, so the served models can change. The exploratory drift check found no drift; a difference between the exploratory and confirmatory estimates can carry drift, the tests within this phase cannot.
- At temperature 0, 15% of answers changed when the same prompt was asked again in the exploratory drift check. This is noise inside every estimate, in both phases.
