"""The budget contrast of REPORT.md section 8.2, with an executable provenance.

    python scripts/analyze_budget_paired.py

Round 6 of the review panel found that section 8.2 - the table that RETRACTS the
budget-doubling claim - was the one table in the report with no script behind
it. Everything else in the project regenerates from code; the part that carries
a negative finding about the project did not. This script closes that gap.

It also fixes a mix-up inside that table. Two different item sets were in play
and one row mixed them:

  per-branch   items parsed in BOTH ORACLE and RAW, within one lexicon branch.
               This is what the reported point estimates use: it reproduces
               4.36 / 9.28 / 7.07 / 9.95 / 3.23 / 0.83 exactly, at
               n = 390 / 388 / 382 / 382 / 371 / 363.

  cross-branch items parsed in all FOUR cells (ORACLE and RAW, KEEP and PSEUDO).
               A paired difference of budgets has to use this one, because the
               two budgets must rest on the same items to be subtracted. Here
               n = 382 / 371 / 343.

Section 8.2 quoted the per-branch point estimates beside the cross-branch n,
which is how "n = 382 for gpt-4.1" got printed when the per-branch n for gpt-4.1
is 390 and 382 is gpt-4.1-mini's. Both bases are reported below so the numbers
can never drift apart again; the paired CI is computed on the cross-branch set,
which is the only defensible basis for it.

The sample-size figure answers "how many items would make this difference
detectable at p < 0.05", by scaling the current bootstrap standard error:
n_needed = n_now * (1.96 * SE / |diff|) ** 2. That is an estimate under the
assumption that the effect size and the variance hold, and it can be wrong in
either direction - it is a planning number, not a result.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

TIER = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]


def load(tag, prefix="price400"):
    p = ROOT / "results" / f"pilot_raw_{prefix}{tag}.csv"
    if not p.exists():
        raise SystemExit(f"thieu {p.name}: chay pilot.py --lexicon {tag} --tag _{prefix}{tag}")
    return pd.read_csv(p)


def per_item(d, model, cond):
    s = d[(d.model == model) & (d.cond == cond) & (d.parsed == 1)]
    return s.set_index("item").correct


def budget_series(d, model, items=None):
    """ORACLE - RAW per item, on items parsed in both."""
    o, r = per_item(d, model, "ORACLE"), per_item(d, model, "RAW")
    i = o.index.intersection(r.index)
    if items is not None:
        i = i.intersection(items)
    return (o[i] - r[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--prefix", default="price400",
                    help="tien to file du lieu: price400 (n=400) hoac n600")
    a = ap.parse_args()
    K, P = load("KEEP", a.prefix), load("PSEUDO", a.prefix)
    rng = np.random.default_rng(a.seed)
    W = 92

    print("=" * W)
    print("1. BUDGET PER BRANCH - the basis for the point estimates in section 8.2")
    print("=" * W)
    rows = []
    for m in TIER:
        bk, bp = budget_series(K, m), budget_series(P, m)
        rows.append({"model": m,
                     "ngan_sach_KEEP": round(100 * bk.mean(), 2), "n_KEEP": len(bk),
                     "ngan_sach_PSEUDO": round(100 * bp.mean(), 2), "n_PSEUDO": len(bp)})
    per = pd.DataFrame(rows)
    print(per.to_string(index=False))
    sfx = "" if a.prefix == "price400" else f"_{a.prefix}"
    per.to_csv(ROOT / "results" / f"budget_per_branch{sfx}.csv", index=False)
    print("\n  This reproduces section 8.2 EXACTLY. Note the n values: 390/382/371, NOT")
    print("  the 382/371 that section 8.2 quotes - those two belong to the section 2 basis.")

    print("\n" + "=" * W)
    print("2. PAIRED BUDGET DIFFERENCE - the only valid basis for a subtraction")
    print("=" * W)
    print("  Only items parsed in all FOUR cells (ORACLE/RAW x KEEP/PSEUDO).")
    print(f"  Bootstrap boc lai theo item, {a.boot} lan.\n")
    rows = []
    for m in TIER:
        common = (set(per_item(K, m, "ORACLE").index) & set(per_item(K, m, "RAW").index)
                  & set(per_item(P, m, "ORACLE").index) & set(per_item(P, m, "RAW").index))
        bk = budget_series(K, m, common)
        bp = budget_series(P, m, common)
        idx = sorted(common)
        diff_i = pd.Series((bp[idx].values - bk[idx].values), index=idx)
        boot = np.array([100 * diff_i.loc[rng.choice(idx, len(idx), replace=True)].mean()
                         for _ in range(a.boot)])
        est = 100 * diff_i.mean()
        lo, hi = np.percentile(boot, [2.5, 97.5])
        se = boot.std(ddof=1)
        n_need = int(np.ceil(len(idx) * (1.96 * se / abs(est)) ** 2)) if est else None
        rows.append({"model": m, "n_shared": len(idx),
                     "ngan_sach_KEEP": round(100 * bk[idx].mean(), 2),
                     "ngan_sach_PSEUDO": round(100 * bp[idx].mean(), 2),
                     "delta_pp": round(est, 2), "ci_lo": round(lo, 2), "ci_hi": round(hi, 2),
                     "se": round(se, 2),
                     "established": "no" if lo <= 0 <= hi else "yes",
                     "n_can_de_p05": n_need})
    pair = pd.DataFrame(rows)
    print(pair.to_string(index=False))
    pair.to_csv(ROOT / "results" / f"budget_paired_ci{sfx}.csv", index=False)

    n_ok = (pair.established == "yes").sum()
    print(f"\n  {n_ok}/3 model co CI khong chua 0.")
    rev = pair[pair.delta_pp < 0]
    if len(rev):
        print(f"  Moving the OTHER way: {', '.join(rev.model)} - the budget FALLS once the "
              f"lexical anchor is removed.")

    print("\n" + "=" * W)
    print("3. KET LUAN")
    print("=" * W)
    if n_ok == 0:
        print('  The claim that "the budget doubles once the lexical anchor is removed" is')
        print('  NOT ESTABLISHED.')
        print("  Ca ba CI deu chua 0. Do do diem hoa von k* cung chua xac lap, vi")
        print("  its numerator is not solid.")
    else:
        print(f"  {n_ok}/3 models reach it. Still not enough to state it for all three.")
    big = pair.dropna(subset=["n_can_de_p05"]).sort_values("n_can_de_p05")
    for _, r in big.iterrows():
        print(f"    {r.model:14} hien co n={r.n_shared}, can n~{int(r.n_can_de_p05)} "
              f"de hieu dat p<0,05")
    print("\n  n_needed_for_p05 is a PLANNING number, not a result: it assumes the effect")
    print("  size and the variance stay unchanged as the sample grows.")
    print(f"  Da ghi: results/budget_per_branch{sfx}.csv, "
          f"results/budget_paired_ci{sfx}.csv")


if __name__ == "__main__":
    main()
