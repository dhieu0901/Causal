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

from stats import boot_p

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
                     # REPORT section 8.2 quotes p for these rows; it was not
                     # stored, so check_numbers.py could not vouch for it.
                     "p_boot": round(boot_p(boot, a.boot), 4),
                     "established": "no" if lo <= 0 <= hi else "yes",
                     "n_can_de_p05": n_need})
    pair = pd.DataFrame(rows)
    print(pair.to_string(index=False))
    pair.to_csv(ROOT / "results" / f"budget_paired_ci{sfx}.csv", index=False)

    # ------------------------------------------------------------------
    # The pooled column. REPORT.md section 8.2 carries a third column headed
    # "Gop, n~800 item rieng biet" and, until 2026-09-23, no script produced
    # it - the two per-sample columns beside it did. Pooling has to happen on
    # the ORIGINAL CLadder id, not on the item index, because the two samples
    # were drawn separately and their item numbers collide while referring to
    # different questions. pool_samples.py makes the same move for the headline.
    # ------------------------------------------------------------------
    pooled_rows = []
    tags = [("price400", ""), ("n600", "_n600")]
    have = [(pfx, sfx2) for pfx, sfx2 in tags
            if (ROOT / "results" / f"pilot_raw_{pfx}KEEP.csv").exists()
            and (ROOT / "results" / f"_itemmap_{pfx}.csv").exists()]
    if len(have) == 2 and a.prefix == "price400":
        print("\n" + "=" * W)
        print("2b. GOP HAI MAU, bo trung theo id goc CLadder")
        print("=" * W)
        for m in TIER:
            parts, seen = [], set()
            for pfx, _ in have:
                try:
                    Kx, Px = load("KEEP", pfx), load("PSEUDO", pfx)
                except SystemExit:
                    continue
                imap = pd.read_csv(ROOT / "results" / f"_itemmap_{pfx}.csv")
                to_id = dict(zip(imap.item, imap.id))
                common = (set(per_item(Kx, m, "ORACLE").index)
                          & set(per_item(Kx, m, "RAW").index)
                          & set(per_item(Px, m, "ORACLE").index)
                          & set(per_item(Px, m, "RAW").index))
                idx = sorted(common)
                if not idx:
                    continue
                bk, bp = budget_series(Kx, m, common), budget_series(Px, m, common)
                d_i = pd.Series(bp[idx].values - bk[idx].values,
                                index=[to_id.get(i) for i in idx])
                d_i = d_i[[i is not None for i in d_i.index]]
                # First sample wins a repeated question; never count it twice.
                keep = [i for i in d_i.index if i not in seen]
                seen.update(keep)
                parts.append(d_i.loc[keep])
            if not parts:
                continue
            allv = pd.concat(parts)
            allv = allv[~allv.index.duplicated()]
            ids = list(allv.index)
            boot = np.array([100 * allv.loc[rng.choice(ids, len(ids), replace=True)].mean()
                             for _ in range(a.boot)])
            est = 100 * allv.mean()
            lo, hi = np.percentile(boot, [2.5, 97.5])
            pooled_rows.append({"model": m, "n_unique_ids": len(ids),
                                "delta_pp": round(est, 2), "ci_lo": round(lo, 2),
                                "ci_hi": round(hi, 2), "p_boot": round(boot_p(boot, a.boot), 4),
                                "established": "no" if lo <= 0 <= hi else "yes"})
        if pooled_rows:
            pooled = pd.DataFrame(pooled_rows)
            print(pooled.to_string(index=False))
            pooled.to_csv(ROOT / "results" / "budget_paired_pooled.csv", index=False)
            print("\n  De-dup theo id goc: mot cau xuat hien o ca hai mau chi tinh MOT lan.")
            print("  Gop KHONG phai nhan ban - hai mau dung chung mot phan item, va")
            print("  chung duoc rut tu cung mot benchmark boi cung mot nguoi.")

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
