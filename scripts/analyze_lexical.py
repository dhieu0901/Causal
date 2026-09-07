"""Does anonymising the variable names cost accuracy, on the same items?

    python scripts/analyze_lexical.py

Four runs over one item sample, differing only in what the variables are
called. Same graphs, same probabilities, same questions, same gold labels, same
perturbation draws:

  KEEP     as CLadder wrote them          real words, plausible direction
  PERMUTE  the item's own names, shuffled real words, implausible direction
  SYMBOL   single letters A, B, C         no words, easy to bind
  PSEUDO   CLadder's pseudowords          no words, hard to bind

That makes the lexical manipulation paired, so McNemar applies to it. The first
attempt at this contrast compared CLadder's commonsense split against its
noncommonsense split, which shares no item with it - and, worse, those
test-*-v1.5.csv files carry no question at all, so that run could only ever
return chance. See REPORT.md.

The ladder is built so each step isolates one thing. KEEP -> PERMUTE keeps the
vocabulary identical and only scrambles which name sits at which graph position,
so it removes the usable prior while holding prompt length, tokenisation and
binding difficulty fixed - PERMUTE runs 9 characters longer than KEEP on
average, against 169 shorter for SYMBOL. PERMUTE -> SYMBOL then drops real
words entirely, and SYMBOL -> PSEUDO makes the tokens hard to tell apart while
leaving them equally meaningless.

Without PERMUTE, any KEEP -> PSEUDO gap is confounded with the prompt getting
much shorter and tokenising differently.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from stats import mcnemar_exact_p

LEXICONS = ["KEEP", "PERMUTE", "SYMBOL", "PSEUDO"]
TIER = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]


def load(tagfmt="pilot_raw_lex{}.csv"):
    out = {}
    for lex in LEXICONS:
        p = ROOT / "results" / tagfmt.format(lex)
        if p.exists():
            out[lex] = pd.read_csv(p)
    return out


def paired_mcnemar(a, b, model, cond):
    """a vs b on the same items, parsed in both. Returns (n, delta_pp, p)."""
    def wide(d):
        s = d[(d.model == model) & (d.cond == cond) & (d.parsed == 1)]
        return s.set_index("item").correct
    x, y = wide(a), wide(b)
    i = x.index.intersection(y.index)
    if len(i) < 10:
        return len(i), np.nan, np.nan
    x, y = x[i], y[i]
    nb = int(((x == 1) & (y == 0)).sum())
    nc = int(((x == 0) & (y == 1)).sum())
    return len(i), 100 * (x.mean() - y.mean()), mcnemar_exact_p(nb, nc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    d = load()
    missing = [l for l in LEXICONS if l not in d]
    if missing:
        raise SystemExit(f"thieu ket qua cho: {missing}")

    models = [m for m in TIER if m in set(d["KEEP"].model)]
    conds = ["PROSE", "RAW", "ORACLE", "DR_k1"]

    print("=" * 88)
    print("1. DO CHINH XAC THEO BO TU VUNG  (chi cau parse duoc, cung item)")
    print("=" * 88)
    rows = []
    for m in models:
        for c in conds:
            r = {"model": m, "cond": c}
            for lex in LEXICONS:
                s = d[lex]
                s = s[(s.model == m) & (s.cond == c) & (s.parsed == 1)]
                r[lex] = round(100 * s.correct.mean(), 2) if len(s) else None
                r[f"parse_{lex}"] = round(100 * d[lex][(d[lex].model == m) &
                                                       (d[lex].cond == c)].parsed.mean(), 1)
            rows.append(r)
    acc = pd.DataFrame(rows)
    print(acc.to_string(index=False))
    acc.to_csv(ROOT / "results" / f"lexical_accuracy{a.tag}.csv", index=False)

    print("\n" + "=" * 88)
    print("2. McNEMAR GHEP CAP: doi tu vung tren CUNG item co lam giam do chinh xac khong?")
    print("=" * 88)
    out = []
    for m in models:
        for c in conds:
            for lex in ["PERMUTE", "SYMBOL", "PSEUDO"]:
                n, delta, p = paired_mcnemar(d[lex], d["KEEP"], m, c)
                out.append({"model": m, "cond": c, "so_sanh": f"{lex} - KEEP",
                            "n": n, "delta_pp": round(delta, 2) if pd.notna(delta) else None,
                            "p": round(p, 4) if pd.notna(p) else None,
                            "y_nghia": "*" if pd.notna(p) and p < .05 else ""})
            for hi, lo in [("PSEUDO", "SYMBOL"), ("SYMBOL", "PERMUTE")]:
                n, delta, p = paired_mcnemar(d[hi], d[lo], m, c)
                out.append({"model": m, "cond": c, "so_sanh": f"{hi} - {lo}",
                            "n": n,
                            "delta_pp": round(delta, 2) if pd.notna(delta) else None,
                            "p": round(p, 4) if pd.notna(p) else None,
                            "y_nghia": "*" if pd.notna(p) and p < .05 else ""})
    mc = pd.DataFrame(out)
    print(mc.to_string(index=False))
    mc.to_csv(ROOT / "results" / f"lexical_mcnemar{a.tag}.csv", index=False)

    print("\n" + "=" * 88)
    print("3. PHAN RA THEO query_type  (dieu kien ORACLE: do thi dung 100%)")
    print("=" * 88)
    print("  correlation va marginal la rung-1, thuan so hoc, do thi khong tham gia.")
    print("  Neu muc roi o day bang hoac hon cac loai co nhan qua, thi thu bi pha huy")
    print("  khong phai nang luc suy luan nhan qua.\n")
    qt_rows = []
    for m in models:
        for qt in sorted(d["KEEP"].query_type.dropna().unique()):
            r = {"model": m, "query_type": qt}
            base = None
            for lex in LEXICONS:
                s = d[lex]
                s = s[(s.model == m) & (s.cond == "ORACLE") &
                      (s.parsed == 1) & (s.query_type == qt)]
                v = 100 * s.correct.mean() if len(s) else np.nan
                r[lex] = round(v, 1) if pd.notna(v) else None
                r[f"n_{lex}"] = len(s)
                if lex == "KEEP":
                    base = v
            for lex in ["PERMUTE", "SYMBOL", "PSEUDO"]:
                r[f"roi_{lex}"] = (round(base - r[lex], 1)
                                   if r.get(lex) is not None and pd.notna(base) else None)
            qt_rows.append(r)
    qd = pd.DataFrame(qt_rows)
    keep = ["model", "query_type", "n_KEEP", "KEEP", "PERMUTE", "SYMBOL", "PSEUDO",
            "roi_PERMUTE", "roi_SYMBOL", "roi_PSEUDO"]
    print(qd[keep].to_string(index=False))
    qd.to_csv(ROOT / "results" / f"lexical_by_querytype{a.tag}.csv", index=False)

    print("\n" + "=" * 88)
    print("4. GIA CUA MOT CANH DAO CHIEU, THEO TUNG BO TU VUNG")
    print("=" * 88)
    print("  Neu cau truc that su duoc dung de suy luan, bo neo tu vung phai lam")
    print("  do thi SAI tro nen dat hon, chu khong phai re di.\n")
    pr = []
    for m in models:
        for lex in LEXICONS:
            s = d[lex][(d[lex].model == m) & (d[lex].parsed == 1)]
            g = s.groupby("cond").correct.mean() * 100
            if not {"ORACLE", "RAW", "DR_k1"} <= set(g.index):
                continue
            pr.append({"model": m, "tu_vung": lex,
                       "RAW": round(g["RAW"], 2), "ORACLE": round(g["ORACLE"], 2),
                       "DR_k1": round(g["DR_k1"], 2),
                       "delta_struct": round(g["ORACLE"] - g["RAW"], 2),
                       "gia_1_canh_dao": round(g["ORACLE"] - g["DR_k1"], 2)})
    pd_ = pd.DataFrame(pr)
    print(pd_.to_string(index=False))
    pd_.to_csv(ROOT / "results" / f"lexical_price{a.tag}.csv", index=False)


if __name__ == "__main__":
    main()
