"""Pilot run: does structure guidance help on CLadder, and how fast does a
corrupted graph burn that benefit off?

    python scripts/pilot.py --n 100 --models gpt-4.1-nano,gpt-4.1-mini

Conditions, all paired on the same items:
  PROSE   untouched CLadder prompt (the DAG is stated in prose)
  RAW     prose DAG stripped out - the real no-graph floor
  ORACLE  stripped, plus the true DAG restated as a structure block
  DR_k    stripped, plus the DAG with k edge directions reversed
"""
from __future__ import annotations
import argparse, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from perturb import (FAMILY_STRUCTURE, to_edges, max_k, ENUMERATORS,
                     enumerate_scramble)
from prompts import build, parse_answer, strip_structure, parse_prose_graph, ANSWER_RE
from lexical import relabel_item, residue
from runner import run_batch, usage_summary, guard_errors, is_ok, OPENROUTER
from stats import mcnemar_exact_p


def make_items(n, seed, kmax=3, data="full_v1.5_default.csv", pair_to=None,
               drop_nonsense=False):
    """Draw a stratified item sample.

    `data` should stay `full_v1.5_default.csv`. CLadder also ships six
    test-*-v1.5.csv files - commonsense, anticommonsense and noncommonsense
    (pseudoword) names, plus balanced, easy and hard - and reaching for the
    first three looks like the natural way to run a lexical contrast. But all
    six carry no question (see the guard below), so none can be scored. The lexical contrast is done
    within item instead, by `--lexicon` / `src/lexical.py`, which rewrites the
    variable names of a complete prompt and leaves its graph, its numbers, its
    question, and its gold label untouched.

    That is also the stronger design. Comparing two separately drawn splits is a
    between-items contrast that McNemar cannot touch, on the one comparison
    where pairing matters most. Caliper (arXiv:2606.04915) perturbs within item
    for the same reason. Caliper reports a 7.6 pp gap on CLadder's
    interventional rung over its five local models, and Caliper's 29.6 pp is on
    CRASS over nine frontier models - two benchmarks and two model sets, not one
    range. Both figures are Caliper's, not this project's.

    `pair_to` reuses the ids sampled from another file, for the case where two
    files really are item-matched.
    """
    d = pd.read_csv(ROOT / "data" / data)

    # The three test-*-v1.5.csv files ship the background and the given
    # quantities but NOT the question - 0% of their prompts contain a '?', while
    # full_v1.5_default.csv is 100%. Running on them asks the model for a yes/no
    # answer to a prompt that poses no question, so it can only guess, and the
    # result is a guaranteed 50% that reads exactly like a collapse in ability.
    # The first pseudoword branch was scored this way; see REPORT.md.
    has_q = d.prompt.str.contains(r"\?", regex=True).mean()
    if has_q < 0.99:
        raise SystemExit(
            f"{data}: only {100*has_q:.1f}% of prompts contain a question. This file "
            f"has had its question removed and cannot be scored. Use "
            f"full_v1.5_default.csv and change the lexicon with --lexicon.")

    if drop_nonsense:
        d = d[~d.story_id.astype(str).str.startswith("nonsense")]
    fams = [f for f in FAMILY_STRUCTURE if max_k(f, "DR") >= kmax]
    d = d[d.graph_id.isin(fams)].reset_index(drop=True)

    if pair_to:
        ref = make_items(n, seed, kmax, pair_to)
        keep = d[d.id.isin(set(ref.id))]
        missing = len(ref) - len(keep)
        if missing:
            raise SystemExit(
                f"{data} is missing {missing} of the {len(ref)} ids sampled from "
                f"{pair_to}; it cannot be paired to it.")
        # Reference order, so `item` indexes the same question in both runs.
        return (keep.set_index("id").loc[ref.id].reset_index()
                    .reindex(columns=d.columns))

    # proportional stratification over graph family x rung
    rng = random.Random(seed)
    groups = list(d.groupby(["graph_id", "rung"]))
    per = max(1, n // len(groups))
    picked = []
    for _, g in groups:
        picked += rng.sample(list(g.index), min(per, len(g)))
    picked = picked[:n] if len(picked) >= n else picked
    return d.loc[picked].reset_index(drop=True)


def build_jobs(items, kmax=3, seed=0, types=("DR",), lexicon="KEEP",
               drop_residue=False, with_instr=False, with_names=False,
               with_scramble=False, with_clean_raw=False):
    """Every graph is built over the story's own variable names.

    Using the symbol DAG (X -> V2 -> Y) beside a body about husbands and wives
    leaves the model two unlinked namespaces, which makes the structure block
    unusable and silently collapses ORACLE onto RAW. The first pilot run did
    exactly that; see REPORT.md.

    Which particular corruption gets drawn matters. Measured across two draws of
    the same design, DR accuracy moved by 1.36 pp on average and 3.40 pp at worst
    - large next to the 2-3 pp effects being estimated. A single shared RNG makes
    that worse: adding ED and FE to the run shifts the stream and silently
    re-draws every DR perturbation too, so only 36% of them stayed the same and a
    p-value fell from 0.011 to 0.150. Seeding per (item, type, k) keeps each draw
    independent of what else the run happens to request.
    """
    jobs, dropped, unrelabelled, dirty = [], 0, 0, 0
    for i, r in items.iterrows():
        # KEEP returns the prompt unchanged, so a default run stays byte-identical
        # to earlier ones and reuses their cache entries.
        prompt, clean = relabel_item(r.prompt, lexicon, seed=str(r.id))
        if not clean:
            # Half-relabelled text would leave real world knowledge in an item
            # scored as lexicon-free, which is the confound the condition exists
            # to remove. Drop rather than score it.
            unrelabelled += 1
            continue

        # `clean` only proves no variable_mapping phrase survived. Grammatical
        # variants outside the mapping survive it; see src/lexical.py. Reporting
        # is the default and dropping is opt-in, because dropping changes the
        # sample and would silently break comparability with every result
        # already in REPORT.md.
        if residue(r.prompt, prompt, lexicon):
            dirty += 1
            if drop_residue:
                continue

        _, removed = strip_structure(prompt)
        edges = parse_prose_graph(removed)
        expected = len(to_edges(FAMILY_STRUCTURE[r.graph_id]))
        if len(edges) != expected:          # parse disagreed with the known DAG
            dropped += 1
            continue
        # `id` is CLadder's own key; `item` only indexes this draw of the sample.
        meta = dict(item=i, id=r.id, gold=r.label, graph_id=r.graph_id, rung=r.rung,
                    query_type=r.get("query_type", ""),
                    story_id=r.get("story_id", ""),
                    # CLadder's own lexical label. --drop-nonsense keeps every
                    # real-word row, but 45% of those are CLadder's
                    # anticommonsense items, whose names already point the wrong
                    # way - so an unstratified KEEP baseline has the correct
                    # prior removed on nearly half its items and dilutes every
                    # lexical effect measured against it.
                    question_property=r.get("question_property", ""))
        jobs.append(dict(cond="PROSE", prompt=build(prompt, "PROSE"), **meta))
        jobs.append(dict(cond="RAW", prompt=build(prompt, "RAW"), **meta))
        if with_clean_raw:
            # RAW without the latent-confounder sentence (src/prompts.py). Equal
            # to RAW, and so free, on every family that has no latent.
            jobs.append(dict(cond="RAW_CLEAN", prompt=build(prompt, "RAW_CLEAN"), **meta))
        jobs.append(dict(cond="ORACLE", prompt=build(prompt, "ORACLE", edges), **meta))
        if with_instr:
            # Opt-in: it adds a paid call per item and no existing analysis
            # reads it. See src/prompts.py for what the condition separates.
            jobs.append(dict(cond="RAW_INSTR", prompt=build(prompt, "RAW_INSTR"),
                             **meta))
        if with_names:
            # The matched control for ORACLE: same block, same instruction, same
            # set of variable names, NOT ONE ARROW. ORACLE minus NAMES_ONLY
            # isolates what the edges are worth. Opt-in, because it adds a call
            # per item and no existing analysis reads it.
            jobs.append(dict(cond="NAMES_ONLY",
                             prompt=build(prompt, "NAMES_ONLY", edges), **meta))
        nodes = sorted({n for e in edges for n in e})
        if with_scramble:
            # A random DAG on the same names, with as many arrows as ORACLE and
            # not one true edge (perturb.enumerate_scramble). Seeded per item
            # like the DR draws, so adding it re-draws nothing else. Opt-in, for
            # the same reason as NAMES_ONLY.
            bad = random.Random(f"{seed}:{i}:SCRAMBLE").choice(
                enumerate_scramble(edges, nodes))
            jobs.append(dict(cond="SCRAMBLE",
                             prompt=build(prompt, "PERTURB", bad), **meta))
        for t in types:
            for k in range(1, kmax + 1):
                opts = ENUMERATORS[t](edges, nodes, k)
                if not opts:
                    continue
                bad = random.Random(f"{seed}:{i}:{t}:{k}").choice(opts)
                jobs.append(dict(cond=f"{t}_k{k}",
                                 prompt=build(prompt, "PERTURB", bad), **meta))
    if unrelabelled:
        print(f"    [lexicon {lexicon}] dropped {unrelabelled} items that could not be "
              f"fully relabelled")
    if dirty:
        verb = "DROPPED" if drop_residue else "KEPT (reported only)"
        print(f"    [lexicon {lexicon}] {dirty} items carry residue outside variable_mapping"
              f" - {verb}")
        if not drop_residue:
            print(f"    [lexicon {lexicon}] residue biases this condition TOWARDS KEEP;"
                  f" see REPORT.md section 7.2")
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--models", default="gpt-4.1-nano,gpt-4.1-mini")
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--kmax", type=int, default=3)
    ap.add_argument("--data", default="full_v1.5_default.csv",
                    help="lexical split, e.g. test-noncommonsense-v1.5.csv")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--types", default="DR", help="perturbation types, e.g. DR,ED,FE")
    ap.add_argument("--pair-to", default=None, dest="pair_to",
                    help="reuse the ids sampled from this split, so the two runs "
                         "are paired item by item")
    ap.add_argument("--lexicon", default="KEEP",
                    choices=["KEEP", "PERMUTE", "IRRELEVANT", "SYMBOL", "PSEUDO"],
                    help="swap variable names within item (see src/lexical.py)")
    ap.add_argument("--drop-nonsense", action="store_true", dest="drop_nonsense",
                    help="keep only the real-word stories; full_v1.5_default.csv "
                         "is 38%% pseudoword otherwise")
    ap.add_argument("--with-instr", action="store_true", dest="with_instr",
                    help="add the RAW_INSTR condition: a causal-reasoning instruction with NO graph block, to split the two effects inside Delta_struct")
    ap.add_argument("--names-only", action="store_true", dest="with_names",
                    help="add the NAMES_ONLY condition: the variable names listed and not one arrow. The matched control for ORACLE, isolating what the EDGES are worth")
    ap.add_argument("--drop-residue", action="store_true", dest="drop_residue",
                    help="drop items that still carry real nouns outside "
                         "variable_mapping. This CHANGES the sample, so results are "
                         "NOT directly comparable with earlier runs")
    ap.add_argument("--clean-raw", action="store_true", dest="with_clean_raw",
                    help="add RAW_CLEAN: RAW without the 'X is unobserved.' sentence "
                         "that strip_structure leaves in the three latent families")
    ap.add_argument("--scramble", action="store_true", dest="with_scramble",
                    help="add the SCRAMBLE condition: a random DAG on the same names "
                         "with as many edges as ORACLE and none of the true ones")
    ap.add_argument("--sample-kmax", type=int, default=None, dest="sample_kmax",
                    help="the kmax used to DRAW the items (default: --kmax). Items "
                         "come only from families with a valid k=kmax reversal, so "
                         "raising --kmax alone changes the sample; pass the old "
                         "value here to add deeper perturbations to an existing one")
    ap.add_argument("--conds", default="",
                    help="comma list; keep only these conditions. Every condition "
                         "is still built first, so no draw changes. Use it to add "
                         "conditions to a sample already run without writing its "
                         "old rows into a second file")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="send nothing; print how many calls are already cached and "
                         "what the rest would cost")
    ap.add_argument("--cost-ref", default="", dest="cost_ref",
                    help="FILE:COND, with --dry-run. Price each new call at the SAME "
                         "item's output tokens under COND in results/raw/FILE, which must "
                         "be a run of this very sample, e.g. "
                         "pilot_raw_n600PSEUDO.csv:DR_k1")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 for every GPT-4.1 run. DeepSeek recommends 0.6 for R1, "
                         "whose output degrades into repetition at 0")
    ap.add_argument("--max-tokens", type=int, default=700, dest="max_tokens",
                    help="output cap. For a reasoning model it also caps the "
                         "reasoning, and an answer cut off is scored unparsed")
    ap.add_argument("--causal-only", action="store_true", dest="causal_only",
                    help="send only the causal group (every query type except "
                         "marginal, correlation, backadj). Items are drawn and "
                         "perturbed first, so no draw changes")
    ap.add_argument("--recap", type=int, default=None,
                    help="OpenRouter models: re-ask every answer the --max-tokens "
                         "cap cut off, with this larger cap, and write those rows "
                         "to pilot_raw{tag}_recap{N}.csv. A sensitivity check; the "
                         "main file is unchanged")
    ap.add_argument("--workers", type=int, default=16,
                    help="concurrent calls. Lower it when two runs share one "
                         "OpenRouter provider, which rate-limits (429) under load")
    ap.add_argument("--max-usd", type=float, default=None, dest="max_usd",
                    help="per model: stop sending once this run has been billed "
                         "this much (OpenRouter models only; see runner.run_batch)")
    a = ap.parse_args()

    sample_kmax = a.kmax if a.sample_kmax is None else a.sample_kmax
    items = make_items(a.n, a.seed, sample_kmax, a.data, a.pair_to, a.drop_nonsense)
    jobs = build_jobs(items, a.kmax, a.seed,
                      tuple(t.strip() for t in a.types.split(",")), a.lexicon,
                      drop_residue=a.drop_residue, with_instr=a.with_instr,
                      with_names=a.with_names, with_scramble=a.with_scramble,
                      with_clean_raw=a.with_clean_raw)
    if a.conds:
        keep = {c.strip() for c in a.conds.split(",") if c.strip()}
        unknown = keep - {j["cond"] for j in jobs}
        if unknown:
            raise SystemExit(f"--conds names conditions this run does not build: "
                             f"{sorted(unknown)}")
        jobs = [j for j in jobs if j["cond"] in keep]
    if a.causal_only:
        jobs = [j for j in jobs
                if j["query_type"] not in {"marginal", "correlation", "backadj"}]
    print(f"items={len(items)}  jobs/model={len(jobs)}  lexicon={a.lexicon}  "
          f"paired_to={a.pair_to}  families={sorted(items.graph_id.unique())}")

    if a.dry_run:
        from runner import estimate_cost
        ref = None
        if a.cost_ref:
            fname, cond = a.cost_ref.rsplit(":", 1)
            ref = pd.read_csv(ROOT / "results" / "raw" / fname)
            # `item` is a row index into THIS sample, so a file from another
            # sample would price every call at an unrelated question.
            meta = {j["item"]: (j["graph_id"], j["gold"]) for j in jobs}
            chk = ref.drop_duplicates("item").set_index("item")
            bad = [i for i, m in meta.items()
                   if i in chk.index and (chk.graph_id[i], chk.gold[i]) != m]
            if bad:
                raise SystemExit(f"--cost-ref {fname}: {len(bad)} items disagree on "
                                 f"graph or label; it is not a run of this sample")
            ref = ref[ref.cond == cond]
        total, known = 0.0, True
        for model in a.models.split(","):
            model = model.strip()
            ro = (None if ref is None else
                  ref[ref.model == model].set_index("item").out_tok)
            e = estimate_cost(jobs, model, a.temperature, ref_out=ro,
                              max_tokens=a.max_tokens)
            print(f"  {e['model']:14s} calls={e['calls']:5d} distinct={e['distinct']:5d} "
                  f"cached={e['cached']:5d} new={e['new']:5d}  usd={e['usd']}  "
                  f"{e.get('new_by_cond', '')}")
            if e["usd"] is None:
                known = False
            else:
                total += e["usd"]
        print(f"  ESTIMATE, nothing sent: {total:.2f} USD"
              + ("" if known else "  (plus models with no price or no cache)"))
        return

    def rows_of(recs, model, cap):
        allrows = []
        for r in recs:
            # A call that never reached the model is dropped, not scored. Scoring
            # it would enter an infrastructure failure as a wrong answer.
            if not is_ok(r["result"]):
                continue
            pred = parse_answer(r["result"]["text"])
            source = "content"
            if model in OPENROUTER and not r["result"]["text"].strip():
                # DeepSeek-R1 on OpenRouter sometimes returns an empty answer with
                # the final "ANSWER: no" block at the end of the reasoning field
                # instead (3 of 8 smoke-test calls, 2026-09-24). Only the strict
                # format is taken, and only from the tail: the reasoning says a
                # bare "yes" or "no" many times on its way to the answer.
                m = ANSWER_RE.findall(r["result"].get("reasoning", "")[-400:])
                pred, source = (m[-1].lower(), "reasoning") if m else (None, "")
            allrows.append({"model": model, "item": r["item"], "id": r["id"],
                            "cond": r["cond"],
                            "graph_id": r["graph_id"], "rung": r["rung"],
                            # Carried through so the lexical analysis can split by
                            # query type without re-deriving the sample.
                            "query_type": r["query_type"], "story_id": r["story_id"],
                            "question_property": r["question_property"],
                            "lexicon": a.lexicon,
                            "gold": r["gold"], "pred": pred,
                            "correct": int(pred == r["gold"]) if pred else 0,
                            "parsed": int(pred is not None),
                            "in_tok": r["result"]["in_tok"],
                            "out_tok": r["result"]["out_tok"],
                            "error": r["result"].get("error", ""),
                            # OpenRouter only, so the GPT-4.1 files keep their
                            # columns: who served the call, whether the cap cut
                            # it off, how much of it was reasoning, what it cost.
                            **({k: r["result"].get(k, "") for k in
                                ("provider", "finish", "reason_tok", "usd")}
                               | {"answer_from": source,
                                  "temperature": a.temperature,
                                  "max_tokens": cap}
                               if model in OPENROUTER else {})})
        return allrows

    allrows, recaprows = [], []
    for model in a.models.split(","):
        model = model.strip()
        print(f"\n>>> {model}: {len(jobs)} calls")
        recs = run_batch(jobs, model, temperature=a.temperature, workers=a.workers,
                         max_tokens=a.max_tokens, max_usd=a.max_usd,
                         on_tick=lambda d, t: print(f"    {d}/{t}", flush=True))
        u = usage_summary(recs)
        print(f"    done: {u}")
        guard_errors(recs, label=model)
        allrows += rows_of(recs, model, a.max_tokens)
        if a.recap:
            # An answer the cap cut off is unparsed and drops out of every paired
            # contrast. When the cut-offs pile up in one condition (R1 with the
            # graph reasons about 60% longer than without), that is a bias, so
            # the same prompts are asked again with room to finish.
            cut = [{k: v for k, v in r.items() if k != "result"} for r in recs
                   if r["result"].get("finish") == "length"]
            print(f"\n>>> {model}: re-asking the {len(cut)} answers cut off at "
                  f"{a.max_tokens} tokens, with {a.recap}")
            if cut:
                rec2 = run_batch(cut, model, temperature=a.temperature, workers=a.workers,
                                 max_tokens=a.recap, max_usd=a.max_usd)
                print(f"    done: {usage_summary(rec2)}")
                guard_errors(rec2, label=f"{model} recap")
                recaprows += rows_of(rec2, model, a.recap)

    df = pd.DataFrame(allrows)
    out = ROOT / "results" / "raw" / f"pilot_raw{a.tag}.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved {out}  ({len(df)} rows)")
    if a.recap:
        out2 = ROOT / "results" / "raw" / f"pilot_raw{a.tag}_recap{a.recap}.csv"
        pd.DataFrame(recaprows).to_csv(out2, index=False)
        print(f"saved {out2}  ({len(recaprows)} rows)")

    print("\n" + "=" * 70)
    print("PILOT: DO CHINH XAC THEO DIEU KIEN")
    print("=" * 70)
    order = ["PROSE", "RAW", "ORACLE"] + sorted(set(df.cond) - {"PROSE", "RAW", "ORACLE"})
    piv = (df.groupby(["model", "cond"])
             .agg(acc=("correct", "mean"), parse=("parsed", "mean"), n=("correct", "size"))
             .reset_index())
    piv["acc"] = (piv["acc"] * 100).round(2)
    piv["parse"] = (piv["parse"] * 100).round(1)
    for m in df.model.unique():
        sub = piv[piv.model == m].set_index("cond").reindex(order)
        print(f"\n--- {m} ---")
        print(sub[["n", "acc", "parse"]].to_string())
        raw = sub.loc["RAW", "acc"]
        print(f"  Delta_struct = ORACLE - RAW = {sub.loc['ORACLE','acc'] - raw:+.2f} pp")
        print(f"  Cost of stripping prose (PROSE - RAW) = {sub.loc['PROSE','acc'] - raw:+.2f} pp")
        for k in range(1, a.kmax + 1):
            c = f"DR_k{k}"
            if c in sub.index and pd.notna(sub.loc[c, "acc"]):
                print(f"  {c}: {sub.loc[c,'acc']:.2f}  (vs RAW {sub.loc[c,'acc']-raw:+.2f} pp)")

    print("\n" + "=" * 70)
    print("McNEMAR EXACT (paired, on the same items)")
    print("=" * 70)
    for m in df.model.unique():
        s = df[df.model == m].pivot_table(index="item", columns="cond",
                                          values="correct", aggfunc="first")
        for a_, b_ in [("ORACLE", "RAW"), ("DR_k1", "RAW"), ("DR_k2", "RAW"),
                       ("DR_k3", "RAW"), ("ORACLE", "DR_k1")]:
            if a_ in s and b_ in s:
                x, y = s[a_].dropna(), s[b_].dropna()
                idx = x.index.intersection(y.index)
                b = int(((x[idx] == 1) & (y[idx] == 0)).sum())
                c = int(((x[idx] == 0) & (y[idx] == 1)).sum())
                print(f"  {m:16s} {a_:8s} vs {b_:8s}  b={b:3d} c={c:3d}  "
                      f"p={mcnemar_exact_p(b, c):.4f}")


if __name__ == "__main__":
    main()
