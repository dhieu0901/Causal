"""Does the cost of scrambling the names depend on how good the prior was?

    python scripts/analyze_prior_strength.py

`full_v1.5_default.csv` carries a `question_property` column that the three
test-*-v1.5.csv files drop, and it labels each item's lexical category:

    nonsense          3842   invented words (zory, xevu)
    anticommonsense   3129   real words, implausible causal direction
    easy              1437   real words, plausible direction
    hard              1353   real words, plausible direction
    commonsense        351   real words, plausible direction

That matters for how the lexical ladder is read. `--drop-nonsense` keeps every
real-word row, which sounds like "commonsense naming" but is 45% CLadder's own
anticommonsense items - so the KEEP baseline already has the correct prior
removed on nearly half its items, and the KEEP -> PERMUTE contrast is measured
against a partly-scrambled reference.

Splitting KEEP by that column turns the confound into a third condition. It
gives a prior-strength gradient for free, and CLadder's anticommonsense items
serve as an independent replication of PERMUTE: a different team, a different
method, the same construct.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from pilot import make_items
from stats import mcnemar_exact_p

LEXICONS = ["KEEP", "PERMUTE", "SYMBOL", "PSEUDO"]
TIER = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]
# easy / hard are difficulty tags on plausibly-named items, not a separate
# lexical class - they share 33 of their 37 stories with commonsense - so they
# group with it.
PLAUSIBLE = {"commonsense", "easy", "hard"}


def label_items(n=200, seed=20260907, kmax=1):
    full = pd.read_csv(ROOT / "data" / "full_v1.5_default.csv").set_index("id")
    it = make_items(n, seed, kmax, "full_v1.5_default.csv",
                    drop_nonsense=True).reset_index(drop=True)
    qp = it.id.map(full.question_property)
    it["qp"] = qp
    it["nhom"] = np.where(qp.isin(PLAUSIBLE), "prior dung", "prior sai san co")
    return it


def main():
    it = label_items()
    d = {}
    for l in LEXICONS:
        p = ROOT / "results" / f"pilot_raw_lex{l}.csv"
        if p.exists():
            x = pd.read_csv(p)
            x["nhom"] = x.item.map(it.nhom)
            d[l] = x
    if "KEEP" not in d:
        raise SystemExit("thieu results/pilot_raw_lexKEEP.csv")

    models = [m for m in TIER if m in set(d["KEEP"].model)]
    print("=" * 84)
    print("0. THANH PHAN MAU THEO NHAN GOC CUA CLADDER")
    print("=" * 84)
    print(it.qp.value_counts().to_string())
    print(f"\n  -> prior dung: {(it.nhom == 'prior dung').sum()}"
          f"   prior sai san co: {(it.nhom == 'prior sai san co').sum()}")

    print("\n" + "=" * 84)
    print("1. DUONG SAN KEEP: anticommonsense CUA CLADDER da tu lam mat bao nhieu?")
    print("=" * 84)
    print("  Day la phep nhan doc lap cho PERMUTE: cung mot cau truc, hai nhom")
    print("  lam ra no khac nhau, tren hai tap item khac nhau.\n")
    rows = []
    for m in models:
        s = d["KEEP"][(d["KEEP"].model == m) & (d["KEEP"].cond == "RAW") &
                      (d["KEEP"].parsed == 1)]
        a = 100 * s[s.nhom == "prior dung"].correct.mean()
        b = 100 * s[s.nhom == "prior sai san co"].correct.mean()
        rows.append({"model": m, "prior_dung": round(a, 1),
                     "prior_sai_san_co": round(b, 1), "chenh_pp": round(a - b, 1)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 84)
    print("2. HIEU UNG DOI TU VUNG, TACH THEO DO MANH CUA PRIOR BAN DAU")
    print("=" * 84)
    out = []
    for m in models:
        for c in ["RAW", "ORACLE"]:
            for lex in [l for l in LEXICONS if l != "KEEP" and l in d]:
                for g in ["prior dung", "prior sai san co"]:
                    k = d["KEEP"]
                    a = k[(k.model == m) & (k.cond == c) & (k.parsed == 1) &
                          (k.nhom == g)].set_index("item").correct
                    b = d[lex][(d[lex].model == m) & (d[lex].cond == c) &
                               (d[lex].parsed == 1) &
                               (d[lex].nhom == g)].set_index("item").correct
                    i = a.index.intersection(b.index)
                    if len(i) < 10:
                        continue
                    nb = int(((b[i] == 1) & (a[i] == 0)).sum())
                    nc = int(((b[i] == 0) & (a[i] == 1)).sum())
                    p = mcnemar_exact_p(nb, nc)
                    out.append({"model": m, "cond": c, "so_sanh": f"{lex} - KEEP",
                                "prior_ban_dau": g, "n": len(i),
                                "delta_pp": round(100 * (b[i].mean() - a[i].mean()), 2),
                                "p": round(p, 4),
                                "y_nghia": "*" if p < .05 else ""})
    o = pd.DataFrame(out)
    print(o.to_string(index=False))
    o.to_csv(ROOT / "results" / "prior_strength.csv", index=False)

    print("\n" + "=" * 84)
    print("3. TRUNG BINH THEO NHOM PRIOR (gop 3 bo an danh x 3 model)")
    print("=" * 84)
    for c in ["RAW", "ORACLE"]:
        s = o[o.cond == c]
        print(f"\n--- {c} ---")
        for g in ["prior dung", "prior sai san co"]:
            t = s[s.prior_ban_dau == g]
            print(f"  {g:18s} hai TB {t.delta_pp.mean():7.2f} pp   "
                  f"p<0.05: {int((t.p < .05).sum())}/{len(t)}")
    print("\n  Xoa mot prior dung dat hon xoa mot prior von da sai. Do la dieu")
    print("  phai xay ra neu co che dung, va no la mot phep kiem co the that bai.")


if __name__ == "__main__":
    main()
