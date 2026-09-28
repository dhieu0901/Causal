# B6 and the path probe on Llama 3.3 70B: does the path-cut mechanism hold outside the GPT-4.1 family?

Nguyễn Dương Hiếu, Nguyễn Đại Quân, Trương Hoàng Tùng, Triệu Hải Đăng Trình. 28 September 2026.

This file is committed and pushed **before the first call of this phase is sent** (one test call of the probe, sent to the separate pilot cache to check the key, is not data). It is not edited after that commit; deviations are reported with the results.

## 1. Why

B6 (`prereg/B6.md`) and the path probe (`prereg/PATH_PROBE.md`) ran on the three GPT-4.1 models only. Their two confirmed results are the core of the mechanism reading: the harm of a wrong graph sits in reversals that leave no directed path from treatment X to outcome Y (M1, M3), it persists without the instruction to use the graph (R1), and it is there when the model reads the cut correctly (PP2). A reader can fairly say this is one provider's family. This phase runs the same tests, unchanged, on Llama 3.3 70B Instruct, an open-weight model from another developer.

What is already known about Llama, so the reader can weigh the result: on the `ate`/`ett` questions of B5 (different questions), Llama lost 20.29 points on path-cutting reversals and showed little reaction to a graph otherwise (`results/cladder/b7_descriptive.csv`, descriptive). M1 is therefore expected to be negative; M3, R1, R2 and PP2 have not been measured on Llama.

## 2. Items, conditions, model

- **Items:** exactly B6's 335 fresh `ate`/`ett` questions (`pilot.py --n 340 --seed 20260926 --sample-kmax 1 --kmax 3 --drop-nonsense --query-types ate,ett --exclude-ids prereg/excluded_ids_b6.txt`). None of them was ever sent to Llama.
- **Conditions:** B6's nine, under `KEEP` and `PSEUDO`: `RAW`, `ORACLE`, `ORACLE_NI`, `DR_k1`, `DR_k1_NI`, `DR_k2`, `DR_k2_NI`, `DR_k3`, `DR_k3_NI`. The prompts are byte-identical to the ones B6 sent to the GPT-4.1 models (same draws, same seed).
- **Probe:** the 2,520 (item, lexicon, graph) rows of `prereg/PATH_PROBE.md`, 1,786 distinct prompts, the same template, at most 300 output tokens.
- **Model:** `meta-llama/llama-3.3-70b-instruct` through OpenRouter, pinned to the provider Novita (bf16) with fallbacks off, temperature 0, request seed 20260907, at most 700 output tokens for the B6 prompts. Every B6 answer cut off at 700 tokens is asked again at 1,500 (`--recap 1500`), as in B5 on Llama.

Exact commands: `bash scripts/run_b6_llama.sh`, which runs

```
python scripts/pilot.py --models meta-llama/llama-3.3-70b-instruct --n 340 --seed 20260926 --kmax 3 --sample-kmax 1 --types DR --drop-nonsense --query-types ate,ett --exclude-ids prereg/excluded_ids_b6.txt --no-instr --conds RAW,ORACLE,ORACLE_NI,DR_k1,DR_k1_NI,DR_k2,DR_k2_NI,DR_k3,DR_k3_NI --workers 8 --recap 1500 --lexicon {KEEP|PSEUDO} --tag _b6llama{KEEP|PSEUDO} --max-usd 0.65
python scripts/probe_path.py --models meta-llama/llama-3.3-70b-instruct --tag _llama --workers 8 --max-usd 0.30
```

## 3. Tests

The tests, their construction and their decision rules are those of the two earlier registrations, unchanged:

| | Quantity (B6 draw unit: harm per draw averaged over the KEEP and PSEUDO answers) | Prediction |
|---|---|---|
| M1 | path cut: DR minus ORACLE | negative |
| M2 | estimand changed, answer kept: DR minus ORACLE | within +-5 points (TOST) |
| M3 | path cut minus answer changed with a path kept | negative |
| R1 | path cut, without the instruction line: DR_NI minus ORACLE_NI | negative |
| R2 | path cut: harm with the instruction minus harm without | within +-5 points (TOST) |
| PP2 | mean harm over path-cut cells Llama read as "no path" while reading the correct graph as "path" | negative |
| PP3 | misread minus read-correctly path-cut cells; run only with at least 20 misread cells on at least 10 items | positive |

- Code: `analyze_b6.run_tests` and `verdicts` on `b6_draws(prefix="b6llama")`; `analyze_path_probe.tests` on the Llama cells and `probe_path_raw_llama.csv`. The parametrisation added for this phase leaves every GPT-4.1 output byte for byte unchanged (checked before this commit), and both scripts still reproduce their published exploratory and B6 rows before reading Llama's.
- Decision rule: bootstrap over items, 4,000 draws; Holm across M1 to R2 (five tests), and separately across the probe tests run, family-wise alpha 0.05; confirmed / contradicted / not confirmed as in `prereg/B6.md` section 5; seeds 20260907 (primary), 1, 2, 3, 4, and a verdict that changes with the seed is borderline.
- **Primary analysis:** parsed answers, answers cut off at 700 tokens left as they are (unparsed, dropped from their pairs), as in B5 on Llama. **Sensitivity**, same tests: unparsed answers scored as wrong; cut-off answers replaced by their 1,500-token re-ask.
- Descriptive, no verdict: the registered B6 descriptive quantities, and PP1 (share of graphs Llama reads correctly).

## 4. How the result is read

- M1 and M3 confirmed on Llama: the path-cut mechanism is not specific to the GPT-4.1 family.
- R1 confirmed: Llama too relies on the graph without being told to use it.
- PP2 confirmed with PP1 high on path-cut draws: Llama's harm is deference to a graph it reads correctly, as for GPT-4.1.
- M3 not confirmed: the mechanism reading rests on one provider's models, and the paper says so.

## 5. Cost

Measured on B5's Llama runs: 0.36 to 0.37 USD per 2,292 calls, plus about 0.03 USD for the re-asks. Estimate 0.49 USD per lexicon (2,855 calls each) and 0.14 USD for the probe (upper bound 0.24 USD if every answer ran to 300 tokens): about 1.12 USD in all. Hard caps: 0.65 + 0.65 + 0.30 = 1.60 USD. The OpenRouter key had 2.02 USD of its limit left at this commit.

## 6. Limits known in advance

- One open-weight model; a second provider, not a second architecture class.
- Llama's B5 behaviour on path cuts is known (section 1), so M1 replicates a known pattern on new questions.
- Temperature 0 is not fully deterministic through this API; each answer is one sample.
- The probe shows the graph alone, as in `prereg/PATH_PROBE.md`.
