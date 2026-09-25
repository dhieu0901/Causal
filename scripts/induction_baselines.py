"""What does an F1 of 0.41 on graph induction actually mean?

    python scripts/induction_baselines.py

Section 9 of REPORT.md reports edge-F1 for the induced graphs and a reversal
count per item, and reads a mechanism off the gap between lexicons. Neither
number is interpretable without a floor, and two floors matter here:

  RANDOM         emit as many edges as the true graph has, drawn uniformly from
                 the ordered node pairs. On CLadder's 3-5 node graphs this is
                 not a small number - there are only six ordered pairs on three
                 nodes - so a mediocre-looking F1 can be at chance.

  WORLD-KNOWLEDGE  under PERMUTE the variable names are the item's own, moved to
                 different graph positions. An agent that ignores the text and
                 answers purely from what it knows about the world would emit
                 the KEEP graph. Scoring that against the PERMUTE target gives
                 the reversal count such an agent would score, which is the
                 ceiling the "the model follows its prior, not the text" reading
                 predicts.

Both are computed from the data alone and cost nothing to run. The real models
sit between them, which is the point: they neither read the text cleanly nor
ignore it.
"""
from __future__ import annotations
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
from induce import edge_f1
from lexical import relabel_item
from pilot import make_items
from prompts import parse_prose_graph, strip_structure

LEXICONS = ["KEEP", "PERMUTE", "SYMBOL", "PSEUDO"]
TIER = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]


def main():
    items = make_items(200, 20260907, 1, "full_v1.5_default.csv", drop_nonsense=True)

    rows = []
    for _, r in items.iterrows():
        keep, _ = relabel_item(r.prompt, "KEEP", seed=str(r.id))
        perm, ok = relabel_item(r.prompt, "PERMUTE", seed=str(r.id))
        if not ok:
            continue
        e_keep = parse_prose_graph(strip_structure(keep)[1])
        e_perm = parse_prose_graph(strip_structure(perm)[1])
        if not e_keep or not e_perm:
            continue

        # World knowledge: emit the graph the names imply in the real world,
        # score it against the permuted truth.
        wk = edge_f1(e_keep, e_perm)

        nodes = sorted({n for e in e_perm for n in e})
        pairs = [(a, b) for a in nodes for b in nodes if a != b]
        rng = random.Random(str(r.id))
        rnd = edge_f1(rng.sample(pairs, min(len(e_perm), len(pairs))), e_perm)

        rows.append({"wk_rev": wk["n_reversed"], "wk_f1": wk["f1"],
                     "rnd_rev": rnd["n_reversed"], "rnd_f1": rnd["f1"]})
    b = pd.DataFrame(rows)

    print("=" * 78)
    print(f"DUONG SAN MO PHONG, n={len(b)} item, 0 luot goi API")
    print("=" * 78)
    print(f"{'Hypothetical agent':46s} {'reversed':>10s} {'F1':>8s}")
    print(f"{'Doan ngau nhien (cung so canh)':46s} "
          f"{b.rnd_rev.mean():10.3f} {b.rnd_f1.mean():8.3f}")
    print(f"{'Chi dung tri thuc the gioi (xuat do thi KEEP)':46s} "
          f"{b.wk_rev.mean():10.3f} {b.wk_f1.mean():8.3f}")
    # REPORT section 9 quotes both floors (0.966 reversals, F1 0.362), and they
    # were only ever printed. Persisted so the two endpoints of the "percent of
    # the way to the knowledge agent" scale have a file behind them - including
    # the one that complicates it: a random guesser reverses MORE edges than the
    # knowledge agent, so many reversals do not by themselves mean prior-driven.
    pd.DataFrame([
        {"agent": "random, same edge count", "reversed": round(b.rnd_rev.mean(), 3),
         "f1": round(b.rnd_f1.mean(), 3), "n_items": len(b)},
        {"agent": "world knowledge only (emit KEEP graph)", "reversed": round(b.wk_rev.mean(), 3),
         "f1": round(b.wk_f1.mean(), 3), "n_items": len(b)},
    ]).to_csv(ROOT / "results" / "cladder" / "induction_agent_floors.csv", index=False)

    ind = {l: pd.read_csv(ROOT / "results" / "cladder" / "raw" / f"induction_raw_lex{l}.csv")
           for l in LEXICONS
           if (ROOT / "results" / "cladder" / "raw" / f"induction_raw_lex{l}.csv").exists()}
    if not ind:
        return

    print("\n" + "=" * 78)
    print("MODEL THAT, DAT CANH HAI DUONG SAN")
    print("=" * 78)
    out = []
    for m in [x for x in TIER if x in set(next(iter(ind.values())).model)]:
        for l in LEXICONS:
            if l not in ind:
                continue
            s = ind[l][ind[l].model == m]
            emitted = (s.n_true - s.n_missing + s.n_spurious).mean()
            rev = s.n_reversed.mean()
            out.append({
                "model": m, "lexicon": l,
                "f1": round(s.f1.mean(), 3),
                "f1_over_random_floor": round(s.f1.mean() - b.rnd_f1.mean(), 3),
                "spurious_edges": round(emitted, 2),
                "reversed": round(rev, 3),
                # Normalised, because a lexicon that made the model emit more
                # edges would raise the raw count on its own.
                "reversed_on_spurious": round(rev / emitted, 3) if emitted else None,
                "percent_of_way_to_knowledge": (
                    round(100 * rev / b.wk_rev.mean(), 1) if b.wk_rev.mean() else None),
                "pct_items_with_reversal": round(100 * (s.n_reversed > 0).mean(), 1),
            })
    o = pd.DataFrame(out)
    print(o.to_string(index=False))
    o.to_csv(ROOT / "results" / "cladder" / "induction_baselines.csv", index=False)

    print("\n  f1_over_random_floor at or below 0 means the self-built graph is no")
    print("  better than guessing, however respectable the raw F1 looks.")
    print("  percent_of_way_to_knowledge: 100% is the agent that ignores the text")
    print("  entirely and answers from knowledge alone. Real models sit in between,")
    print("  so they read the text PARTLY.")


if __name__ == "__main__":
    main()
