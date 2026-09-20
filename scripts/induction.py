"""Where does a real agent's own graph land on the break-even curve?

    python scripts/induction.py --n 150 --models gpt-4.1-nano,gpt-4.1-mini,gpt-4.1

Two steps per item:
  1. INDUCE   the model is given the body (structure stripped) plus the variable
              list, and asked for the edges. Scored against the true DAG.
  2. INDUCED  the model then answers the question using its OWN induced graph.

Reuses the same items and seed as pilot.py, so INDUCED is paired with RAW,
ORACLE and the DR_k conditions already measured and the comparison is exact.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
from perturb import FAMILY_STRUCTURE, to_edges
from prompts import build, parse_answer, strip_structure, parse_prose_graph
from induce import build_induce_prompt, parse_edges, edge_f1
from runner import run_batch, usage_summary, guard_errors, is_ok
from stats import mcnemar_exact_p
from pilot import make_items
from lexical import relabel_item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--models", default="gpt-4.1-nano,gpt-4.1-mini,gpt-4.1")
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--kmax", type=int, default=3)
    ap.add_argument("--data", default="full_v1.5_default.csv",
                    help="lexical split; the pseudoword split separates structure "
                         "inference from world knowledge")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--lexicon", default="KEEP",
                    choices=["KEEP", "PERMUTE", "SYMBOL", "PSEUDO"],
                    help="swap variable names within item (see src/lexical.py)")
    ap.add_argument("--drop-nonsense", action="store_true", dest="drop_nonsense",
                    help="keep only real-word stories")
    a = ap.parse_args()

    items = make_items(a.n, a.seed, a.kmax, a.data, None, a.drop_nonsense)

    # Same items and same lexicon as the matching pilot run, so induction quality
    # is paired across lexicons the way accuracy already is. Comparing induction
    # on real-word items against induction on CLadder's own nonsense stories is a
    # between-items contrast; relabelling in place makes it a paired one.
    recs = []
    for i, r in items.iterrows():
        prompt, clean = relabel_item(r.prompt, a.lexicon, seed=str(r.id))
        if not clean:
            continue
        _, removed = strip_structure(prompt)
        edges = parse_prose_graph(removed)
        if len(edges) != len(to_edges(FAMILY_STRUCTURE[r.graph_id])):
            continue
        nodes = sorted({n for e in edges for n in e})
        body, _ = strip_structure(prompt)
        recs.append({"item": i, "gold": r.label, "graph_id": r.graph_id,
                     "rung": r.rung, "true_edges": edges, "nodes": nodes,
                     "body": body, "prompt": prompt})
    print(f"items={len(recs)}  families={sorted({x['graph_id'] for x in recs})}")

    rows = []
    for model in [m.strip() for m in a.models.split(",")]:
        # ---- step 1: induce the graph --------------------------------------
        jobs = [{"item": r["item"],
                 "prompt": build_induce_prompt(r["body"], r["nodes"])} for r in recs]
        print(f"\n>>> {model} step 1 INDUCE: {len(jobs)} calls")
        got = run_batch(jobs, model, temperature=0.0, workers=16, max_tokens=1400,
                        on_tick=lambda d, t: print(f"    {d}/{t}", flush=True))
        print(f"    {usage_summary(got)}")
        guard_errors(got, label=f"{model} INDUCE")
        # A failed INDUCE call is far more corrosive here than in pilot.py. Its
        # empty text parses to zero edges, which (a) scores as an F1 the model
        # never earned and (b) sends the item down the `not e` branch below,
        # silently turning its PERTURB condition into RAW. The experimental
        # condition itself would change without a trace. Drop those items.
        failed_induce = {g["item"] for g in got if not is_ok(g["result"])}
        induced = {g["item"]: parse_edges(g["result"]["text"],
                                          next(r["nodes"] for r in recs if r["item"] == g["item"]))
                   for g in got if is_ok(g["result"])}
        recs = [r for r in recs if r["item"] not in failed_induce]

        # ---- step 2: reason using that graph -------------------------------
        jobs2 = []
        for r in recs:
            e = induced.get(r["item"], [])
            if not e:                      # nothing parsed: fall back to no graph
                jobs2.append({"item": r["item"],
                              "prompt": build(r["prompt"], "RAW")})
            else:
                jobs2.append({"item": r["item"],
                              "prompt": build(r["prompt"], "PERTURB", e)})
        print(f">>> {model} step 2 INDUCED: {len(jobs2)} calls")
        got2 = run_batch(jobs2, model, temperature=0.0, workers=16,
                         on_tick=lambda d, t: print(f"    {d}/{t}", flush=True))
        print(f"    {usage_summary(got2)}")
        guard_errors(got2, label=f"{model} INDUCED")
        answers = {g["item"]: g["result"]["text"] for g in got2 if is_ok(g["result"])}

        for r in recs:
            if r["item"] not in answers:      # step-2 call failed: not a wrong answer
                continue
            e = induced.get(r["item"], [])
            sc = edge_f1(e, r["true_edges"])
            pred = parse_answer(answers[r["item"]])
            rows.append({"model": model, "item": r["item"], "graph_id": r["graph_id"],
                         "rung": r["rung"], "gold": r["gold"], "pred": pred,
                         "correct": int(pred == r["gold"]) if pred else 0,
                         "parsed_answer": int(pred is not None),
                         "parsed_graph": int(bool(e)),
                         **{k: sc[k] for k in ("precision", "recall", "f1",
                                               "n_reversed", "n_spurious",
                                               "n_missing", "n_wrong_edges",
                                               "exact_match", "n_true")}})

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / f"induction_raw{a.tag}.csv", index=False)

    print("\n" + "=" * 78)
    print("CHAT LUONG DO THI TU DUNG (doi chieu Table 3 cua NoisyCausal)")
    print("=" * 78)
    q = (df.groupby("model")
           .agg(n=("f1", "size"), parse_graph=("parsed_graph", "mean"),
                precision=("precision", "mean"), recall=("recall", "mean"),
                f1=("f1", "mean"), exact=("exact_match", "mean"),
                canh_sai=("n_wrong_edges", "mean"),
                dao_chieu=("n_reversed", "mean"),
                thua=("n_spurious", "mean"), thieu=("n_missing", "mean"))
           .round(3))
    print(q.to_string())
    q.to_csv(ROOT / "results" / f"induction_quality{a.tag}.csv")

    print("\n" + "=" * 78)
    print("INDUCED ROI VAO DAU TREN DUONG CONG?")
    print("=" * 78)
    try:
        pilot = pd.read_csv(ROOT / "results" / f"pilot_raw{a.tag}.csv")
    except FileNotFoundError:
        print("  chua co results/pilot_raw.csv - chay scripts/pilot.py truoc")
        return

    acc_ind = df.groupby("model").correct.mean() * 100
    acc_pil = pilot.groupby(["model", "cond"]).correct.mean().unstack() * 100
    cols = ["RAW", "ORACLE"] + sorted(c for c in acc_pil.columns if c.startswith("DR_k"))
    comp = acc_pil[cols].copy()
    comp["INDUCED"] = acc_ind
    print(comp.round(2).to_string())
    comp.round(2).to_csv(ROOT / "results" / f"induction_vs_curve{a.tag}.csv")

    print("\n  INDUCED against the reference points:")
    for m in comp.index:
        if m not in acc_ind:
            continue
        r = comp.loc[m]
        print(f"    {m:14s} INDUCED={r['INDUCED']:.2f}  "
              f"vs RAW {r['INDUCED']-r['RAW']:+.2f}  "
              f"vs ORACLE {r['INDUCED']-r['ORACLE']:+.2f}  "
              f"vs DR_k1 {r['INDUCED']-r['DR_k1']:+.2f}")

    print("\n" + "=" * 78)
    print("McNEMAR: INDUCED co khac RAW khong? (paired)")
    print("=" * 78)
    for m in df.model.unique():
        pi = pilot[(pilot.model == m) & (pilot.cond == "RAW")].set_index("item").correct
        ii = df[df.model == m].set_index("item").correct
        idx = pi.index.intersection(ii.index)
        b = int(((ii[idx] == 1) & (pi[idx] == 0)).sum())
        c = int(((ii[idx] == 0) & (pi[idx] == 1)).sum())
        p = mcnemar_exact_p(b, c)
        print(f"  {m:14s} n={len(idx)}  b={b:3d} c={c:3d}  p={p:.4f}"
              f"{'  *' if p < .05 else ''}")


if __name__ == "__main__":
    main()
