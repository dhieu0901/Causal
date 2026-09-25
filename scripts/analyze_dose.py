"""Is one reversed edge one dose, or is the dose the FRACTION of edges corrupted?

The objection, raised 2026-09-20: reversing one edge of a three-edge graph
corrupts 33% of it, but only 20% of a five-edge graph. So reporting "DR_k1" as a
single condition pools four different doses. The objection is correct, and the
existing k sweep has enough data to answer it.

Seven families in results/cladder/raw/pilot_raw_price400KEEP.csv - 399 items, 2.7x the
pilot_raw.csv file an earlier version of this script used - with k = 1, 2, 3:

    three-edge families (confounding, mediation)               33%, 67%, 100%
    four-edge families (IV, diamond, diamondcut, frontdoor)    25%, 50%,  75%
    five-edge family (arrowhead)                               20%, 40%,  60%

which is nine dose levels from 20% to 100%. The testable question: does the loss
track the NUMBER of edges reversed, or the FRACTION? If it is the fraction, the
per-family curves should collapse onto one line when plotted against k/E rather
than against k.

Measurement. Each item is compared against itself under ORACLE - same item, same
model - so every number is a loss relative to having the correct graph. Cluster
bootstrap over items. Then one very simple question: how much variance does a
regression on k explain, and how much does one on k/E?

Run:  python scripts/analyze_dose.py
Writes: results/cladder/dose_response.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from stats import boot_interval, boot_items, cluster_boot

import numpy as np
import pandas as pd

SEED = 20260920
NBOOT = 4000
KIND = {"DR": "reversed edge", "ED": "missing edge", "FE": "spurious edge"}


def edges_per_family():
    """Edge count per family, read straight off CLadder's parameter keys."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "vg", str(ROOT / "scripts" / "verify_groundtruth.py"))
    vg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vg)
    meta = json.loads((ROOT / "data" / "cladder" / "cladder-meta.json").read_text(encoding="utf-8"))
    out = {}
    for m in meta:
        if m["graph_id"] in out:
            continue
        s = vg.SCM(m["params"])
        out[m["graph_id"]] = sum(len(s.parents[n]) for n in s.nodes)
    return out


def loss(d, family, cond):
    """Paired loss against ORACLE, per item, averaged over the models.

    Returns a Series indexed by item whose value is the mean over the models that
    have both conditions. Pairing within item is the design's main strength, so
    it is preserved here.
    """
    sub = d[(d.graph_id == family) & (d.parsed == 1)]
    cols = []
    for m in sorted(sub.model.unique()):
        o = sub[(sub.model == m) & (sub.cond == "ORACLE")].set_index("item").correct
        c = sub[(sub.model == m) & (sub.cond == cond)].set_index("item").correct
        i = o.index.intersection(c.index)
        if len(i) < 5:
            continue
        cols.append(pd.Series((o[i] - c[i]).values, index=i, name=m))
    if not cols:
        return pd.Series(dtype=float)
    return pd.concat(cols, axis=1).mean(axis=1)


def boot(v, seed=SEED, n=NBOOT):
    """Cluster bootstrap over items for the mean. Returns (estimate, lo, hi)."""
    x = v.dropna().values
    if len(x) < 5:
        return np.nan, np.nan, np.nan
    est, lo, hi, _ = boot_items(x, seed, n)          # convention A, src/stats.py
    return est, lo, hi


def main():
    d = pd.read_csv(ROOT / "results" / "cladder" / "raw" / "pilot_raw_price400KEEP.csv")
    E = edges_per_family()
    families = sorted(d.graph_id.unique(), key=lambda g: (E[g], g))

    print("=" * 86)
    print("DOSE RESPONSE: loss against ORACLE, by edge COUNT and by edge FRACTION")
    print("=" * 86)
    print("  paired within item, averaged over three models, cluster bootstrap over",
          NBOOT, "draws, seed", SEED)
    print("  each number is pp lost relative to the same item given the CORRECT graph\n")

    rows = []
    for kind in ("DR", "ED", "FE"):
        conds = sorted(c for c in d.cond.unique() if c.startswith(kind + "_k"))
        if not conds:
            continue
        print(f"\n{kind} - {KIND[kind]}")
        print(f"  {'family':13s} {'edges':>5s} {'k':>2s} {'fraction':>9s} "
              f"{'loss pp':>8s}  {'95% CI':>18s} {'n':>4s}")
        print("  " + "-" * 68)
        for fam in families:
            for c in conds:
                k = int(c.split("_k")[1])
                if k > E[fam]:
                    continue
                v = loss(d, fam, c)
                if v.empty:
                    continue
                est, lo, hi = boot(v)
                if np.isnan(est):
                    continue
                frac = k / E[fam]
                print(f"  {fam:13s} {E[fam]:5d} {k:2d} {frac:8.0%} "
                      f"{est:+8.2f}  [{lo:+7.2f} ; {hi:+7.2f}] {len(v):4d}")
                rows.append({"error_type": kind, "family": fam, "n_edges": E[fam],
                             "k": k, "frac_edges_corrupted": round(frac, 3),
                             "loss_pp": round(est, 2), "ci_lo": round(lo, 2),
                             "ci_hi": round(hi, 2), "n_items": len(v)})

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "cladder" / "dose_response.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 86)
    print("WHICH DOSE EXPLAINS MORE: edge count k, or edge fraction k/E?")
    print("=" * 86)
    print("  Linear regression of the loss on each variable, over the (family x k) cells.")
    print("  If the fraction is the right variable, the families collapse onto one line")
    print("  when plotted against k/E.\n")
    print("  The third column is a model of FAMILY dummies alone, with no dose term.")
    print("  Adjusted R-squared, to penalise the model with more parameters.\n")
    print(f"  {'type':6s} {'cells':>5s} {'adjR2 on k':>12s} {'adjR2 on k/E':>14s} "
          f"{'adjR2 on family':>16s} {'winner'}")
    print("  " + "-" * 72)

    def adj_r2(A, y):
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        ss = ((y - y.mean()) ** 2).sum()
        if ss <= 0:
            return np.nan
        r2 = 1 - (resid ** 2).sum() / ss
        n, p = len(y), A.shape[1] - 1
        return 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else np.nan

    # These nine adjusted R-squared values are quoted in PROPOSAL section 7.2 and
    # in its abstract, and until 2026-09-23 they lived only in the print below -
    # scripts/check_numbers.py could not vouch for a single one of them. The
    # abstract now leans on "the highest dose column is 0.11", which is not a
    # number an abstract may carry without a file behind it. Persist them.
    r2_rows = []
    for kind in ("DR", "ED", "FE"):
        s = df[df.error_type == kind]
        if len(s) < 4 or s.family.nunique() < 2:
            print(f"  {kind:6s} {len(s):5d}  too few cells to compare")
            continue
        y = s.loss_pp.values
        one = np.ones((len(s), 1))
        r2 = {
            "k": adj_r2(np.hstack([s.k.values.astype(float)[:, None], one]), y),
            "k/E": adj_r2(np.hstack([s.frac_edges_corrupted.values[:, None], one]), y),
            "family": adj_r2(np.hstack([pd.get_dummies(s.family, drop_first=True)
                                        .values.astype(float), one]), y),
        }
        win = max(r2, key=lambda k: -np.inf if np.isnan(r2[k]) else r2[k])
        print(f"  {kind:6s} {len(s):5d} {r2['k']:12.3f} {r2['k/E']:14.3f} "
              f"{r2['family']:16.3f} {win:>8s}")
        r2_rows.append({"error_type": kind, "n_cells": len(s),
                        "adj_r2_k": round(float(r2["k"]), 3),
                        "adj_r2_k_over_E": round(float(r2["k/E"]), 3),
                        "adj_r2_family": round(float(r2["family"]), 3),
                        "winner": win})
    if r2_rows:
        pd.DataFrame(r2_rows).to_csv(
            ROOT / "results" / "cladder" / "dose_variance_explained.csv", index=False)

    print("\n  HOW TO READ THIS. The seven families give only three edge counts (3, 4,")
    print("  5), and the five-edge group is the single family arrowhead - the one whose")
    print("  CLadder labels are broken. So this comparison is directional evidence, not")
    print("  enough to settle which dose variable is right.")
    print("\n  What it does show: EVERY R-squared here is low - the highest of the two")
    print("  dose columns is 0.11. So the answer to the objection is not 'k or k/E is")
    print("  the right one', it is that DOSE EXPLAINS VERY LITTLE either way.")
    print("  For ED and FE the family dummies dominate (0.36 and 0.61): what matters is")
    print("  WHICH structure is broken, not how much of it. For DR, k/E and family are")
    print("  comparable and both weak, so DR alone is unsettled.")
    print("\n  CORRECTION. An earlier version of this block ran on results/cladder/raw/pilot_raw.csv")
    print("  (147 items) and reported that BOTH dose variables had negative adjusted")
    print("  R-squared. On 399 items that is no longer true for DR: k/E rises to 0.11.")
    print("  The strong claim 'dose is not an explanatory variable' holds for ED and FE")
    print("  only.")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
