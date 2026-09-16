"""Does the ranking of graph-error types survive losing the lexical anchors?

    python scripts/analyze_errortypes_lexical.py

Section 6 prices three kinds of graph corruption on the main branch:

    DR  one edge direction reversed
    ED  one spurious edge added
    FE  one true edge deleted

and finds DR the most expensive. But that was measured with the ordinary
variable names in place, where the model has a usable prior to fall back on.
The lexical experiment only ever ran DR, so the ranking had never been checked
under anonymisation - which is precisely the regime the rest of the project is
about.

What comes out is a cleaner statement of the project's own mechanism than the
main branch gives:

  With the correct prior available (KEEP), NO error type has a measurable cost.
  The model has something else to lean on, so a corrupted graph barely matters.

  Once the prior is gone (SYMBOL, PSEUDO), the graph starts to matter, and what
  matters is EDGE DIRECTION specifically. DR is the only type whose cost clears
  zero, and on SYMBOL it is established as more expensive than either of the
  other two.

Two limits, both real. The causal subset leaves roughly 60 items per cell, so
the equivalence bounds here are wide and a null means "not resolved at this
sample size", not "zero". And k is fixed at 1: this prices the FIRST wrong edge
of each kind, not a slope, so nothing here speaks to how cost accumulates.

Bootstrap resamples ITEMS, pooling the three models, for the reason given in
analyze_querygroup.py.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

LEXICONS = ["KEEP", "PERMUTE", "SYMBOL", "PSEUDO"]
TIER = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]
TYPES = ["DR", "ED", "FE"]
LABEL = {"DR": "dao chieu canh", "ED": "thua mot canh", "FE": "thieu mot canh"}
ARITH = {"marginal", "correlation"}
IDENT = {"backadj"}


def load(lex):
    p = ROOT / "results" / f"pilot_raw_edfe{lex}.csv"
    if not p.exists():
        raise SystemExit(
            f"thieu {p.name}. Chay:\n"
            f"  python scripts/pilot.py --n 200 --models gpt-4.1-nano,gpt-4.1-mini,"
            f"gpt-4.1 --kmax 1 --types DR,ED,FE --drop-nonsense --lexicon {lex} "
            f"--tag _edfe{lex}")
    return pd.read_csv(p)


def paired(d, model, a, b, causal=True):
    """Per-item (a - b) on items parsed in both conditions."""
    s = d[(d.model == model) & (d.parsed == 1)]
    if causal:
        s = s[~s.query_type.isin(ARITH | IDENT)]
    w = s.pivot_table(index="item", columns="cond", values="correct", aggfunc="first")
    if a not in w.columns or b not in w.columns:
        return None
    w = w[[a, b]].dropna()
    if not len(w):
        return None
    return pd.Series(w[a].values - w[b].values, index=w.index)


def boot(by_model, rng, n=4000):
    items = sorted(set().union(*[set(s.index) for s in by_model.values()]))
    M = np.vstack([by_model[m].reindex(items).values for m in by_model])
    out = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(items), len(items))
        out[i] = np.nanmean(M[:, pick])
    return 100 * np.nanmean(M), 100 * out


def summarise(by_model, rng, boots):
    est, bs = boot(by_model, rng, boots)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi, ("co" if lo > 0 or hi < 0 else "khong")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--all-queries", action="store_true",
                    help="dung ca mau thay vi chi nhom cau hoi nhan qua")
    a = ap.parse_args()
    causal = not a.all_queries
    D = {lex: load(lex) for lex in LEXICONS}
    rng = np.random.default_rng(a.seed)
    W = 88

    print("=" * W)
    print("1. GIA CUA TUNG LOAI LOI DO THI, THEO BO TU VUNG")
    print("=" * W)
    scope = "chi nhom cau hoi nhan qua that" if causal else "ca mau"
    print(f"  Do bang ORACLE tru <loai>_k1, {scope}, k co dinh bang 1.")
    print(f"  Gop ba model, bootstrap boc lai theo ITEM {a.boot} lan.\n")

    rows = []
    for lex in LEXICONS:
        for t in TYPES:
            by = {}
            for m in TIER:
                s = paired(D[lex], m, "ORACLE", f"{t}_k1", causal)
                if s is not None:
                    by[m] = s
            if not by:
                continue
            est, lo, hi, ok = summarise(by, rng, a.boot)
            rows.append({"lexicon": lex, "loai": t, "nghia": LABEL[t],
                         "chi_phi_pp": round(est, 2), "ci_lo": round(lo, 2),
                         "ci_hi": round(hi, 2), "xac_lap": ok,
                         "can_tuong_duong": round(max(abs(lo), abs(hi)), 2)})
    cost = pd.DataFrame(rows)
    print(cost[["lexicon", "loai", "chi_phi_pp", "ci_lo", "ci_hi", "xac_lap"]]
          .to_string(index=False))
    cost.to_csv(ROOT / "results" / "errortype_by_lexicon.csv", index=False)

    print("\n" + "=" * W)
    print("2. CANH BI DAO CHIEU CO DAT HON HAI LOAI KIA KHONG")
    print("=" * W)
    print("  Hieu ghep cap trong cung item, nen khong phai suy tu hai CI chong nhau.\n")
    rows = []
    for lex in LEXICONS:
        for other in ("ED", "FE"):
            by = {}
            for m in TIER:
                s = paired(D[lex], m, f"{other}_k1", f"DR_k1", causal)
                if s is not None:
                    by[m] = s
            if not by:
                continue
            est, lo, hi, ok = summarise(by, rng, a.boot)
            rows.append({"lexicon": lex, "so_sanh": f"DR dat hon {other}",
                         "hieu_pp": round(est, 2), "ci_lo": round(lo, 2),
                         "ci_hi": round(hi, 2), "xac_lap": ok})
    rank = pd.DataFrame(rows)
    print(rank.to_string(index=False))
    rank.to_csv(ROOT / "results" / "errortype_ranking.csv", index=False)

    print("\n" + "=" * W)
    print("3. DOC KET QUA")
    print("=" * W)
    keep_any = (cost[cost.lexicon == "KEEP"].xac_lap == "co").any()
    anon = cost[cost.lexicon.isin(["SYMBOL", "PSEUDO"])]
    dr_anon = anon[(anon.loai == "DR") & (anon.xac_lap == "co")]
    if not keep_any:
        kb = cost[cost.lexicon == "KEEP"].can_tuong_duong.max()
        print(f"  Tren KEEP: KHONG loai loi nao co chi phi do duoc "
              f"(can tuong duong toi da {kb:.2f} pp).")
        print("  Con prior dung de dua vao thi do thi hong it quan trong.")
    if len(dr_anon):
        print(f"\n  Tren bo an danh: DR xac lap o {len(dr_anon)}/2 bo, "
              f"trong khi ED va FE khong o bo nao.")
        print("  Bo prior di thi do thi moi co gia, va thu co gia la CHIEU CANH.")
    strict = rank[rank.xac_lap == "co"]
    if len(strict):
        print(f"\n  DR dat hon loai khac mot cach xac lap o: "
              f"{', '.join(sorted(set(strict.lexicon)))}.")
    print("\n  CANH BAO CO MAU. Nhom nhan qua chi con khoang 60 item moi o, nen mot")
    print("  ket qua 'khong xac lap' o day nghia la CHUA PHAN GIAI DUOC o co mau")
    print("  nay, khong phai bang khong. Doc cot can_tuong_duong trong file CSV.")
    print("\n  Da ghi: results/errortype_by_lexicon.csv, results/errortype_ranking.csv")


if __name__ == "__main__":
    main()
