# Path probe: does the model read a path-cutting graph correctly, and does its reading predict its answer?

Nguyễn Dương Hiếu, 27 September 2026.

This file was committed and pushed **before the first call of this probe was sent** (a pilot of 12 calls, sent to a separate cache to measure answer length, is not data). It is not edited after that commit; deviations are reported with the results.

## 1. Why

B6 confirmed that the harm of a wrong graph sits in reversals that leave no directed path from treatment X to outcome Y (M1, M3; REPORT section 2.2). Two readings fit that result:

- **Trust.** The model reads the wrong graph correctly, sees that no path is left, and concludes there is no effect.
- **Misreading.** The model misreads graphs, and the path-cutting reversals are where misreading happens to show.

This probe separates them by asking each model, of every graph B6 showed it, whether that graph has a directed path from X to Y.

## 2. Graphs, prompt, models

- **Graphs:** every graph of B6, replayed exactly as `pilot.build_jobs` drew it (items of `prereg/B6.md`, seed 20260926): the correct graph (`ORACLE`) and each reversal `DR_k1`, `DR_k2`, `DR_k3`, under `KEEP` and `PSEUDO`. 2,520 (item, lexicon, graph) rows, 1,786 distinct prompts per model. Each replayed graph rebuilds a B6 prompt that is a key in the API cache, or `scripts/probe_path.py` stops; all 2,520 pass.
- **X and Y** are the item's own names for treatment and outcome in that lexicon, found through the cladder-meta variable mapping; no item is left out.
- **Prompt:** the graph block B6 carried (`prompts.describe_graph`, the same sentences) and one question, nothing else (no story, no numbers, no causal question):

  > Here is a causal graph, given as statements of direct effects: {block}
  > In this graph, is there a directed path from {X} to {Y}? A directed path is a chain of one or more direct effects, each one pointing forward, that starts at {X} and ends at {Y}.
  > Think briefly, then answer on the last line in exactly this format: ANSWER: yes or ANSWER: no

- **Models:** `gpt-4.1-nano`, `gpt-4.1-mini`, `gpt-4.1`, temperature 0, at most 300 output tokens, the OpenAI key `back_up` (the primary key has reached its project spend limit).
- **Cost:** estimated 3.20 USD from a pilot of 4 calls per model (mean 120 to 160 output tokens); upper bound 5.88 USD if every answer ran to 300 tokens. Hard cap 3.50 USD for the whole run. Exact command: `python scripts/probe_path.py --max-usd 3.50`.

## 3. The unit

A **cell** is one (model, item, lexicon, reversal dose k). Its **harm** h is the model's B6 answer under `DR_kk` minus its answer under `ORACLE` (correct = 1, instructed prompts, parsed answers only). Its **reading** is the model's probe answer for the shown reversal and for the item's correct graph (unparsed probe answers drop out). The cell's group is the one B6 used (`analyze_answer_change.py`: "answer changed, path cut" when the shown graph has no X -> Y path and the implied answer differs from the label).

`scripts/analyze_path_probe.py` computes everything. Before it reads a probe record it rebuilds the cells from B6's records and stops unless, averaged over models and lexicons, they give `analyze_b6.b6_draws()`'s harm on every draw. At this commit they do: 925 draws from 5,456 cells.

## 4. What is computed

| | Quantity | Prediction |
|---|---|---|
| PP1 | Per model, share of graphs read correctly: the correct graphs, reversals with a path, reversals without; and on B6's path-cut draws, the share read as "no path" | descriptive, no verdict |
| PP2 | Mean h over path-cut cells that the model read as "no path" while reading the item's correct graph as "path" | negative: the harm is there when the model reads the graph right |
| PP3 | Mean h over path-cut cells read as "path" (a misreading; correct graph read as "path") minus mean h over those read as "no path" | positive: a model that misreads the cut is harmed less, so its reading drives its answer |

PP3 is run only with at least 20 misread cells on at least 10 items; otherwise it is reported as a description with no verdict, and PP2 is the only test.

Sensitivity, no verdict: PP2 with the B6 answers given without the instruction line (`DR_kk_NI` minus `ORACLE_NI`).

## 5. Decision rule

Bootstrap over items, 4,000 draws, every cell of an item resampled with it (`analyze_b6.boot_mean`; PP3 takes both means from the same resample, `analyze_b6.boot_diff`). Two-sided p is `stats.boot_p`. Holm across the tests run, family-wise alpha 0.05. **Confirmed**: adjusted p < 0.05 and the estimate in the predicted direction; **contradicted**: adjusted p < 0.05 the other way; otherwise **not confirmed**. Repeated at bootstrap seeds 20260907 (primary), 1, 2, 3, 4; a verdict that changes with the seed is reported as borderline.

## 6. How the result is read

- PP2 confirmed, with PP1 high on path-cut draws: the harm is trust in a graph read correctly, not misreading.
- PP3 confirmed: the model's own reading of the graph predicts its answer, not only the graph's true structure.
- If most path-cut draws are misread (PP1 low), the "trust" reading of B6 is weakened, and that is reported.

## 7. Limits known in advance

- The probe shows the graph alone. In the B6 task the graph came after a story with numbers, so a model might read the same graph differently there; the probe measures whether it *can* read it.
- Misreading is likely rare for `gpt-4.1` and `gpt-4.1-mini`, so PP3 rests mostly on `gpt-4.1-nano` and may not run.
- Temperature 0 is not deterministic on this API (15% of answers change when asked again, REPORT section 6); the reading of a cell is one sample.
- The cells were built from B6 data already analysed, so PP2 and PP3 test an interpretation of a known result, not a new effect.
