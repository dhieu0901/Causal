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
from scipy import stats
from stats import mcnemar_exact_p

LEXICONS = ["KEEP", "PERMUTE", "SYMBOL", "PSEUDO"]
TIER = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]


def load(tagfmt="pilot_raw_lex{}.csv"):
    out = {}
    for lex in LEXICONS:
        p = ROOT / "results" / "cladder" / "raw" / tagfmt.format(lex)
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
        raise SystemExit(f"missing results for: {missing}")

    models = [m for m in TIER if m in set(d["KEEP"].model)]
    conds = ["PROSE", "RAW", "ORACLE", "DR_k1"]

    print("=" * 88)
    print("1. ACCURACY BY LEXICON  (parsed answers only, same items)")
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
    acc.to_csv(ROOT / "results" / "cladder" / f"lexical_accuracy{a.tag}.csv", index=False)

    print("\n" + "=" * 88)
    print("2. PAIRED McNEMAR: does swapping the lexicon on the SAME item cost accuracy?")
    print("=" * 88)
    out = []
    for m in models:
        for c in conds:
            for lex in ["PERMUTE", "SYMBOL", "PSEUDO"]:
                n, delta, p = paired_mcnemar(d[lex], d["KEEP"], m, c)
                out.append({"model": m, "cond": c, "n_compared": f"{lex} - KEEP",
                            "n": n, "delta_pp": round(delta, 2) if pd.notna(delta) else None,
                            "p": round(p, 4) if pd.notna(p) else None,
                            "meaning": "*" if pd.notna(p) and p < .05 else ""})
            for hi, lo in [("PSEUDO", "SYMBOL"), ("SYMBOL", "PERMUTE")]:
                n, delta, p = paired_mcnemar(d[hi], d[lo], m, c)
                out.append({"model": m, "cond": c, "n_compared": f"{hi} - {lo}",
                            "n": n,
                            "delta_pp": round(delta, 2) if pd.notna(delta) else None,
                            "p": round(p, 4) if pd.notna(p) else None,
                            "meaning": "*" if pd.notna(p) and p < .05 else ""})
    mc = pd.DataFrame(out)
    print(mc.to_string(index=False))
    mc.to_csv(ROOT / "results" / "cladder" / f"lexical_mcnemar{a.tag}.csv", index=False)
    # REPORT section 4.0 quotes the mean harm per lexicon across the three
    # models (e.g. PERMUTE under ORACLE -6.61, 2/3 significant). Those means
    # were computed from this table by hand and never written down.
    summ = (mc.assign(sig=mc.meaning == "*")
              .groupby(["cond", "n_compared"])
              .agg(mean_delta_pp=("delta_pp", "mean"), n_sig=("sig", "sum"),
                   n_models=("delta_pp", "size"))
              .round({"mean_delta_pp": 2}).reset_index())
    summ.to_csv(ROOT / "results" / "cladder" / f"lexical_mcnemar_summary{a.tag}.csv", index=False)

    print("\n" + "=" * 88)
    print("3. PHAN RA THEO query_type  (dieu kien ORACLE: do thi dung 100%)")
    print("=" * 88)
    print("  correlation and marginal are rung-1, pure arithmetic, with no role for the")
    print("  graph. If the drop here matches or exceeds the causal query types, then what")
    print("  was destroyed is not causal reasoning ability.\n")
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
    qd.to_csv(ROOT / "results" / "cladder" / f"lexical_by_querytype{a.tag}.csv", index=False)

    print("\n" + "=" * 88)
    print("4. THE PRICE OF ONE REVERSED EDGE, BY LEXICON")
    print("=" * 88)
    print("  If the structure really is used for reasoning, removing the lexical anchor")
    print("  should make a WRONG graph more expensive, not cheaper.\n")
    pr = []
    for m in models:
        for lex in LEXICONS:
            s = d[lex][(d[lex].model == m) & (d[lex].parsed == 1)]
            g = s.groupby("cond").correct.mean() * 100
            if not {"ORACLE", "RAW", "DR_k1"} <= set(g.index):
                continue
            pr.append({"model": m, "lexicon": lex,
                       "RAW": round(g["RAW"], 2), "ORACLE": round(g["ORACLE"], 2),
                       "DR_k1": round(g["DR_k1"], 2),
                       "delta_struct": round(g["ORACLE"] - g["RAW"], 2),
                       "price_one_reversed_edge": round(g["ORACLE"] - g["DR_k1"], 2)})
    pd_ = pd.DataFrame(pr)
    print(pd_.to_string(index=False))
    pd_.to_csv(ROOT / "results" / "cladder" / f"lexical_price{a.tag}.csv", index=False)

    # ---- 4b. PROSE minus RAW, per lexicon ---------------------------------
    # REPORT section 10 leans on these four numbers for its contamination
    # argument: the PROSE/KEEP cell is the only one matching CLadder character
    # for character, so if the model were living off string memory that cell
    # should gain the MOST from prose. It gains the least. The four figures were
    # quoted with no file behind them; paired within item, as the design allows.
    pv = []
    for lex in LEXICONS:
        cols = []
        for m in models:
            s = d[lex][(d[lex].model == m) & (d[lex].parsed == 1)]
            w = s.pivot_table(index="item", columns="cond", values="correct",
                              aggfunc="first")
            if not {"PROSE", "RAW"} <= set(w.columns):
                continue
            w = w[["PROSE", "RAW"]].dropna()
            if len(w) < 10:
                continue
            cols.append(pd.Series((w.PROSE - w.RAW).values, index=w.index, name=m))
        if not cols:
            continue
        v = pd.concat(cols, axis=1).mean(axis=1).dropna()
        pv.append({"lexicon": lex, "n_items": len(v),
                   "prose_minus_raw_pp": round(100 * v.mean(), 2)})
    if pv:
        pvd = pd.DataFrame(pv)
        print("\n  4b. PROSE - RAW, ghep cap trong tung item, gop ba model:")
        print(pvd.to_string(index=False))
        pvd.to_csv(ROOT / "results" / "cladder" / f"prose_gain_by_lexicon{a.tag}.csv", index=False)

    # ---- 4c. clustering by story ------------------------------------------
    # REPORT section 10b argues against story-level memorisation from an ICC
    # of correctness by story_id near zero. The three ICCs were quoted and never
    # computed by any script. One-way ANOVA ICC(1) per model, KEEP / RAW.
    ic = []
    for m in models:
        s = d["KEEP"][(d["KEEP"].model == m) & (d["KEEP"].cond == "RAW") &
                      (d["KEEP"].parsed == 1)]
        g = s.groupby("story_id").correct
        k = g.size()
        n, n_st = len(s), len(k)          # not `a`: that is the argparse namespace
        if n_st < 2:
            continue
        grand = s.correct.mean()
        msb = float((k * (g.mean() - grand) ** 2).sum() / (n_st - 1))
        msw = float(((s.correct - s.story_id.map(g.mean())) ** 2).sum() / (n - n_st))
        k0 = (n - float((k ** 2).sum()) / n) / (n_st - 1)
        icc = (msb - msw) / (msb + (k0 - 1) * msw)
        ic.append({"model": m, "n_items": n, "n_stories": n_st,
                   "icc_story": round(icc, 3),
                   "design_effect": round(1 + (float(k.mean()) - 1) * max(icc, 0.0), 2)})
    if ic:
        icd = pd.DataFrame(ic)
        print("\n  4c. ICC of correctness by story_id, KEEP / RAW:")
        print(icd.to_string(index=False))
        icd.to_csv(ROOT / "results" / "cladder" / f"story_icc{a.tag}.csv", index=False)

    # ---- 5. does the lexicon change the graph the model BUILDS? ------------
    ind = {}
    for lex in LEXICONS:
        f = ROOT / "results" / "cladder" / "raw" / f"induction_raw_lex{lex}.csv"
        if f.exists():
            ind[lex] = pd.read_csv(f)
    if len(ind) < 2:
        print("\n(no per-lexicon induction results - skipping section 5)")
        return

    print("\n" + "=" * 88)
    print("5. QUALITY OF THE MODEL'S SELF-BUILT GRAPH, PAIRED ON THE SAME ITEMS")
    print("=" * 88)
    print("  Sections 1-4 measure REASONING over a graph that was supplied.")
    print("  This section measures EXTRACTING the graph from text - a different ability.\n")
    rows, base_lex = [], "KEEP"
    for m in models:
        for lex in [l for l in LEXICONS if l in ind]:
            s_ = ind[lex][ind[lex].model == m]
            b = ind[base_lex][ind[base_lex].model == m].set_index("item")
            cur = s_.set_index("item")
            i = b.index.intersection(cur.index)
            # Wilcoxon on the paired per-item F1: the same item is scored under
            # both lexicons, so the pairing is exact and a rank test on the
            # differences is the right call for a bounded, non-normal score.
            try:
                p_f1 = (stats.wilcoxon(cur.loc[i, "f1"], b.loc[i, "f1"]).pvalue
                        if lex != base_lex else np.nan)
            except ValueError:
                p_f1 = np.nan
            rev = s_.n_reversed.mean()
            rev_base = ind[base_lex][ind[base_lex].model == m].n_reversed.mean()
            rows.append({
                "model": m, "lexicon": lex, "n": len(i),
                "f1": round(s_.f1.mean(), 3),
                "f1_vs_KEEP": round(s_.f1.mean() - rev_base * 0 - b.f1.mean(), 3),
                "p_wilcoxon": round(p_f1, 4) if pd.notna(p_f1) else None,
                "exactly_correct": round(s_.exact_match.mean(), 3),
                "reversed_edges": round(rev, 3),
                "ratio": round(rev / rev_base, 1) if rev_base else None,
                "missing_edges": round(s_.n_missing.mean(), 2),
            })
    idf = pd.DataFrame(rows)
    # Nine Wilcoxon tests (3 models x 3 anonymised lexicons) were reported
    # uncorrected. Benjamini-Hochberg over that family (review round 10, M6).
    pv = idf.p_wilcoxon
    ok = pv.notna()
    order = pv[ok].sort_values()
    m_ = len(order)
    cut = 0
    for r_, (idx, val) in enumerate(order.items(), start=1):
        if val <= 0.05 * r_ / m_:
            cut = r_
    idf["survives_BH"] = False
    idf.loc[order.index[:cut], "survives_BH"] = True
    print(idf.to_string(index=False))
    idf.to_csv(ROOT / "results" / "cladder" / f"lexical_induction{a.tag}.csv", index=False)
    print("\n  Prior SAI (PERMUTE) gay dao chieu nhieu gap may lan prior VANG MAT")
    print("  (SYMBOL/PSEUDO) is the key dissociation: if edge direction is read off the")
    print("  text, the two cases have to look the same.")


if __name__ == "__main__":
    main()
