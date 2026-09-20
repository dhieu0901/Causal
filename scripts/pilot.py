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
from perturb import FAMILY_STRUCTURE, to_edges, max_k, ENUMERATORS
from prompts import build, parse_answer, strip_structure, parse_prose_graph
from lexical import relabel_item, residue
from runner import run_batch, usage_summary, guard_errors, is_ok
from stats import mcnemar_exact_p


def make_items(n, seed, kmax=3, data="full_v1.5_default.csv", pair_to=None,
               drop_nonsense=False):
    """Draw a stratified item sample.

    `data` should stay `full_v1.5_default.csv`. CLadder also ships three
    test-*-v1.5.csv files whose variable names are commonsense,
    anticommonsense, or pseudowords, and reaching for those looks like the
    natural way to run a lexical contrast - but they carry no question (see the
    guard below), so they cannot be scored at all. The lexical contrast is done
    within item instead, by `--lexicon` / `src/lexical.py`, which rewrites the
    variable names of a complete prompt and leaves its graph, its numbers, its
    question, and its gold label untouched.

    That is also the stronger design. Comparing two separately drawn splits is a
    between-items contrast that McNemar cannot touch, on the one comparison
    where pairing matters most. Caliper (arXiv:2606.04915) perturbs within item
    for the same reason, and reports 7.6 to 29.6 pp drops from anonymising
    names.

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
               drop_residue=False, with_instr=False, with_names=False):
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
        meta = dict(item=i, gold=r.label, graph_id=r.graph_id, rung=r.rung,
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
    a = ap.parse_args()

    items = make_items(a.n, a.seed, a.kmax, a.data, a.pair_to, a.drop_nonsense)
    jobs = build_jobs(items, a.kmax, a.seed,
                      tuple(t.strip() for t in a.types.split(",")), a.lexicon,
                      drop_residue=a.drop_residue, with_instr=a.with_instr,
                      with_names=a.with_names)
    print(f"items={len(items)}  jobs/model={len(jobs)}  lexicon={a.lexicon}  "
          f"paired_to={a.pair_to}  families={sorted(items.graph_id.unique())}")

    allrows = []
    for model in a.models.split(","):
        model = model.strip()
        print(f"\n>>> {model}: {len(jobs)} calls")
        recs = run_batch(jobs, model, temperature=0.0, workers=16,
                         on_tick=lambda d, t: print(f"    {d}/{t}", flush=True))
        u = usage_summary(recs)
        print(f"    done: {u}")
        guard_errors(recs, label=model)
        for r in recs:
            # A call that never reached the model is dropped, not scored. Scoring
            # it would enter an infrastructure failure as a wrong answer.
            if not is_ok(r["result"]):
                continue
            pred = parse_answer(r["result"]["text"])
            allrows.append({"model": model, "item": r["item"], "cond": r["cond"],
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
                            "error": r["result"].get("error", "")})

    df = pd.DataFrame(allrows)
    out = ROOT / "results" / f"pilot_raw{a.tag}.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved {out}  ({len(df)} rows)")

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
