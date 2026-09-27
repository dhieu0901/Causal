"""Deleted and spurious edges in the groups of analyze_answer_change.py, exploratory.

    python scripts/analyze_edge_types.py

Why. Every pre-registered wrong graph so far is a reversal (DR). price400 also
sent deleted edges (ED, k = 1 to 3) and spurious edges (FE, k = 1 and 2) on
the same items, three GPT-4.1 models, both lexicons, with the instruction line.
This file places each of those draws in the groups of REPORT section 4.8 - is
the effect kept, is the estimand changed, does the answer the shown graph
implies change, and if so is every X -> Y path gone - and reports the harm,
cond minus ORACLE, per group and per corruption type. It is what the next
pre-registration (B8) was written from; nothing here is a test.

The unit differs from the B6 unit in one way. FE draws differ between the two
lexicons (classify_perturbations.classify: candidate edges are walked in
sorted-name order), so a draw is one (item, lexicon, type, k): its harm is
the mean over the three models, parsed answers only. The bootstrap resamples
items, all draws of an item together.

Check that stops the script: the DR draws of price400, classified here with
the ED and FE arms added, must fall in the same groups as in
results/cladder/answer_change.csv.

Writes: results/cladder/edge_types.csv (one row per draw)
        results/cladder/edge_types_harm.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from analyze_answer_change import Matcher, TYPES, draws_for
from stats import boot_p, cluster_boot

RESULTS = ROOT / "results" / "cladder"
NBOOT = 4000
SEEDS = [20260907, 1, 2, 3, 4]
ARMS = ("DR", "ED", "FE")


def harm(lex):
    """Per (item, cond): cond minus ORACLE, averaged over models, parsed only."""
    d = pd.read_csv(RESULTS / "raw" / f"pilot_raw_price400{lex}.csv")
    d = d[(d.parsed == 1) & d.query_type.isin(TYPES)]
    out = []
    for m in sorted(d.model.unique()):
        dm = d[d.model == m]
        r = dm[dm.cond == "ORACLE"].drop_duplicates("item").set_index("item").correct
        for cond in sorted(c for c in dm.cond.unique() if c[:2] in ARMS):
            c = dm[dm.cond == cond].drop_duplicates("item").set_index("item").correct
            i = r.index.intersection(c.index)
            out.append(pd.DataFrame({"item": i, "cond": cond, "diff": (c[i] - r[i]).values}))
    h = pd.concat(out).groupby(["item", "cond"])["diff"].mean().rename("h").reset_index()
    return h.assign(lexicon=lex)


def boot(x, seed):
    keys = list(x.groupby("item").indices.values())
    v = x.h.values
    dr = cluster_boot(len(keys), lambda i: 100 * v[np.concatenate([keys[j] for j in i])].mean(),
                      seed, NBOOT)
    return 100 * v.mean(), np.percentile(dr, 2.5), np.percentile(dr, 97.5), boot_p(dr, NBOOT)


def cell(x):
    per = [boot(x, s) for s in SEEDS]
    e, lo, hi, p = per[0]
    ps = [q[3] for q in per]
    return dict(harm_pp=round(e, 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2), p_boot=round(p, 4),
                p_min_seeds=round(min(ps), 4), p_max_seeds=round(max(ps), 4),
                n_draws=len(x), n_items=x.item.nunique())


def main() -> int:
    D = draws_for("price400", Matcher(), arms=ARMS)
    D = D.assign(arm=D.cond.str[:2])

    # the DR draws must land where analyze_answer_change.py put them
    pub = pd.read_csv(RESULTS / "answer_change.csv")
    pub = pub[pub["sample"].eq("price400") & pub.lexicon.isin(["KEEP", "PSEUDO"])]
    key = ["item", "lexicon", "cond"]
    a = D[D.arm.eq("DR")].set_index(key)[["group", "route", "implied_answer"]].sort_index()
    b = pub.set_index(key)[["group", "route", "implied_answer"]].sort_index()
    if not (a.index.equals(b.index) and a.fillna("").equals(b.fillna(""))):
        raise SystemExit("  DR draws do not reproduce answer_change.csv; stopping")
    print(f"  check: {len(a)} DR draws reproduce answer_change.csv exactly")

    H = pd.concat([harm("KEEP"), harm("PSEUDO")], ignore_index=True)
    D = D.merge(H, on=["item", "lexicon", "cond"], how="left")
    D.to_csv(RESULTS / "edge_types.csv", index=False)

    changed = D.group.eq("answer changed")
    cells = [("estimand kept", D.group.eq("estimand kept")),
             ("estimand changed, answer kept", D.group.eq("estimand changed, answer kept")),
             ("answer changed, path cut", changed & D.route.eq("no path")),
             ("answer changed, path kept", changed & D.route.ne("no path")),
             ("all draws", D.group.notna())]
    rows = []
    for arm in ARMS:
        for kk in ("all", "k1"):
            for g, sel in cells:
                x = D[sel & D.arm.eq(arm) & D.h.notna()]
                if kk == "k1":
                    x = x[x.cond.str.endswith("_k1")]
                if x.item.nunique() < 10:
                    continue
                rows.append(dict(arm=arm, k=kk, group=g) | cell(x))
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "edge_types_harm.csv", index=False)

    print("\n  draws per group, k = 1 (both lexicons):")
    k1 = D[D.cond.str.endswith("_k1")]
    print(pd.crosstab([k1.group, k1.route], k1.arm).to_string())
    print("\n  harm, cond minus ORACLE, pp (items resampled; p range over five seeds):")
    for r in R.itertuples():
        print(f"    {r.arm} {r.k:3s} {r.group:32s} {r.harm_pp:+7.2f} [{r.ci_lo:+6.2f} ; "
              f"{r.ci_hi:+6.2f}]  p {r.p_min_seeds:.4f}-{r.p_max_seeds:.4f}  "
              f"draws {r.n_draws:4d}  items {r.n_items}")
    print("\n  wrote results/cladder/edge_types.csv, edge_types_harm.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
