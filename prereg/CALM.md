# Second benchmark (CaLM): pre-registration

Nguyễn Dương Hiếu, 25 September 2026.

Committed and pushed **before the first call on these items was sent**, like `prereg/CONFIRMATORY.md`. Not edited after that commit; deviations are reported with the results.

## 1. Why

Every result of the study so far is on CLadder, whose graphs come from ten fixed families. This phase asks whether the two central findings hold on a benchmark built by other people, on random graphs: (i) a correct graph makes up for the prior that anonymised names take away, and (ii) the model follows the direction of the edges it is given.

## 2. Items

- Source: CaLM (Chen et al. 2024, arXiv:2405.00622), `OpenCausaLab/CaLM` at commit `1c1e93a80c3f9b3250bd79faf4dee8f8bc3479dd`, file `calm_dataset/intervention/average_treatment_effect/ATE-B_ATE-natural_EN.json` (1,600 items; SHA-256 `0928be65f495862fab154dd206342ecbc8000cb76d814d22f7b7fcc3a7194e62`). Its Lite release (`calm_lite_dataset/...`, SHA-256 `70809205b956dcd38895e37df2d079ce71057c0569f4b507410093e3fc6947f1`) is used only to check the labels below. Apache-2.0.
- Not built from CLadder: random DAGs on 3 to 5 nodes. CaLM tasks named after CLadder query types (`backadj`, `det-counterfactual`, `collider-bias`, `correlation`, `exp-away`) are not used.
- Mode. CaLM labels its items REAL (real names, a sensible story), RANDOM (real names from a fixed pool on a random graph) or FAKE (pseudowords) in the Lite release only. `src/calm.mode()` recovers the label for all 1,600 items (a RANDOM item's names each recur in 40 or more items) and agrees with all 100 Lite labels.
- Gold. The full release has no answers. `src/calm.gold()` recomputes them as CaLM's worked solutions do (no directed path from treatment to outcome: No; otherwise the sign of the backdoor-adjusted difference in the direction the question asks) and agrees with all 100 Lite answers.
- Used: the 534 REAL items, minus those whose edge sentences do not parse back to the item's own graph under both lexicons (the rule `pilot.build_jobs` applies to CLadder). That leaves **520 items from 92 stories** (one story with the name "feeling more alert and energized" is dropped); 107 answers Yes, 413 No.
- `scripts/analyze_calm.py` re-checks the SHA-256 of both files and both 100/100 agreements on every run, and stops if any fails.

## 3. Conditions

Built by `src/prompts.py` exactly as for CLadder. The CaLM text uses CLadder's own frame (the same preamble and one "X has a direct effect on Y." sentence per edge), so:

- `RAW`: the edge sentences removed; the probabilities, CaLM's instruction line and the question kept.
- `ORACLE`: `RAW` plus the structure block over the item's names.
- `DR_k1`: `RAW` plus a structure block with one edge reversed, acyclic, drawn per item with `random.Random(f"20260925:{item}:DR:1")`.
- Lexicons: `KEEP` (CaLM's names) with `RAW` and `ORACLE`; `PSEUDO` (every name replaced by a pseudoword from `src/lexical.PSEUDOWORDS`, seeded per item) with `RAW`, `ORACLE` and `DR_k1`.
- Gold is the true world's answer under every condition.

## 4. Models

- Llama 3.3 70B Instruct through OpenRouter, pinned to Novita (bf16), temperature 0, at most 700 output tokens, `--recap 1500`. Estimated 0.41 USD, hard cap 0.60 USD (`scripts/run_calm.sh llama`).
- GPT-4.1 nano, mini and full through OpenAI, temperature 0, at most 700 output tokens: estimated 8.0 USD, hard cap 9.0 USD (`scripts/run_calm.sh gpt`), run only if the author approves that cost. Each family gets its own verdicts; nothing is pooled across families.

## 5. Tests

One value per (item, model) cell, averaged over every cell. An unparsed answer drops out of every pair it belongs to.

| Test | Quantity | Prediction |
|---|---|---|
| C1 | `[(KEEP - PSEUDO) under RAW] - [(KEEP - PSEUDO) under ORACLE]` | positive: the correct graph makes up for part of what anonymised names cost |
| C2 | `ORACLE - DR_k1` under `PSEUDO` | positive: reversing one edge of the graph given costs accuracy |
| C3 | `KEEP - PSEUDO` under `RAW` | positive: real names carry a usable prior |

## 6. Decision rule

- Bootstrap, 4,000 draws, **resampling stories** (the 92 name-and-graph sets the items share). Two-sided p is `stats.boot_p`.
- Holm over the three tests within a family, family-wise alpha = 0.05. Confirmed: adjusted p < 0.05 in the predicted direction; contradicted: adjusted p < 0.05 the other way; otherwise not confirmed.
- Repeated at bootstrap seeds 20260907 (primary), 1, 2, 3, 4; a verdict that is not the same at all five is borderline.

## 7. Sensitivity analyses, fixed now, which do not change a verdict

- Items resampled instead of stories.
- Unparsed answers scored as wrong.
- Llama only: every answer cut off at 700 tokens asked again with 1,500 tokens, both versions reported.
- Exploratory and labelled as such: accuracy by condition, split by whether the question needs the probabilities or only the graph.

## 8. Limits known in advance

- 79% of the gold answers are No, mostly items with no directed path from treatment to outcome, where the graph alone decides the answer. A model biased towards No scores well under every condition; the tests are within-item differences, which that bias does not create, but it can compress them.
- CaLM's REAL stories were generated with a language model; how sensible each one is varies.
- One benchmark more is still one benchmark more: same prompt frame, same answer format.
