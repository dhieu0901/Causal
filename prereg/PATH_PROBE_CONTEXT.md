# The path probe inside the question: do the models read a path-cutting graph correctly when it sits in the full question?

Nguyễn Dương Hiếu, Nguyễn Đại Quân, Trương Hoàng Tùng, Triệu Hải Đăng Trình. 28 September 2026.

This file is committed and pushed **before the first call of this probe is sent** (a pilot of a few calls per model, sent to the separate pilot cache to measure answer length, is not data). It is not edited after that commit; deviations are reported with the results.

## 1. Why

The path probe (`prereg/PATH_PROBE.md`) showed each graph of B6 alone, with no story and no numbers, and found that `gpt-4.1` and `gpt-4.1-mini` read path existence correctly on every graph; PP2 was confirmed (the harm is there on path-cut draws the models read correctly). Its own section 7 named the limit: in the B6 task the graph came after a story with numbers, so a model might read the same graph differently there. This probe asks the same question inside the question B6 sent.

## 2. Prompts, graphs, models

- **Prompt:** the exact B6 prompt (instructed condition: story, stated probabilities, the block "The causal structure of this world is: ... Use this causal structure when reasoning."), with the causal question at the end of the story and the answer line removed, and in their place: "In this causal structure, is there a directed path from {X} to {Y}? A directed path is a chain of one or more direct effects, each one pointing forward, that starts at {X} and ends at {Y}. Think briefly, then answer on the last line in exactly this format: ANSWER: yes or ANSWER: no". Built by `scripts/probe_path.py --context` (`context_prompt`), which stops unless each prompt has that shape.
- **Graphs:** only those PP2 uses: every path-cut draw of B6 (answer changed, no directed X -> Y path; 307 draws on 164 items) and the correct graph of each of those items, under `KEEP` and `PSEUDO`: 942 rows, 937 distinct prompts per model. Each graph is replayed as `scripts/probe_path.py` replays it and proved by the API cache.
- **Models:** `gpt-4.1-nano`, `gpt-4.1-mini`, `gpt-4.1`, temperature 0, at most 300 output tokens, the OpenAI key `back_up`.
- **Cost:** estimated 1.95 USD at 150 output tokens per answer (to be checked by the pilot); upper bound 3.35 USD if every answer ran to 300 tokens. Hard cap 2.90 USD for the whole run. Exact command: `OPENAI_KEY_VAR=back_up python scripts/probe_path.py --context --max-usd 2.90`.

## 3. Tests

The unit, the groups and the construction are those of `prereg/PATH_PROBE.md` section 3, with the reading taken from this probe instead of the probe of the graph alone:

| | Quantity | Prediction |
|---|---|---|
| PC1 | Per model, share of path-cut graphs read as "no path" and of correct graphs read as "path", inside the question | descriptive, no verdict |
| PC2 | Mean h over path-cut cells the model read, inside the question, as "no path" while reading the item's correct graph as "path" | negative |
| PC3 | Mean h over path-cut cells read as "path" minus mean h over those read as "no path"; run only with at least 20 misread cells on at least 10 items, otherwise a description with no verdict | positive |

Computed by `scripts/analyze_path_probe.py` (the functions that computed PP2 and PP3, fed `probe_path_context_raw.csv`), which still first rebuilds B6's draws from the cells and stops if they differ. Holm across the tests run, family-wise alpha 0.05; bootstrap over items, 4,000 draws; seeds 20260907 (primary), 1, 2, 3, 4, a verdict that changes with the seed being borderline. Sensitivity, no verdict: PC2 with B6's answers given without the instruction line.

## 4. How the result is read

- PC1 high for `gpt-4.1` and `gpt-4.1-mini` and PC2 confirmed: inside the question, too, the models read that the wrong graph leaves no path, and the harm sits on the draws they read correctly. The "deference, not misreading" reading of B6 then holds in the task's own context.
- PC1 clearly lower than the probe of the graph alone: the story or the numbers change how the graph is read, and the paper says so.

## 5. Limits known in advance

- The causal question is removed, so this is the graph read in the question's context, not in the act of answering it.
- Only path-cut draws and their correct graphs are asked; the reading of other reversals inside the question is not measured.
- The B6 data were analysed before this probe was designed; PC2 tests an interpretation of a known result.
