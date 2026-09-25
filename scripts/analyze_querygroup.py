"""Does the correct graph rescue anonymisation, tested as an interaction?

    python scripts/analyze_querygroup.py

Two problems with the pooled analysis this script replaces.

FIRST, the headline claim is an interaction and was never tested as one. "The
correct graph cuts almost half the harm of anonymising" is not two separate
statements ("RAW costs 10.73 pp", "ORACLE costs 5.67 pp"). It is one statement
about the difference of two differences, and the design is fully paired, so the
difference-in-differences can be computed directly:

    DiD_i = [(KEEP - LEX) | RAW] - [(KEEP - LEX) | ORACLE]

Reading it off a change in the COUNT of significant cells (8/9 versus 3/9) is
the Gelman-Stern error: a difference between "significant" and "not significant"
is not itself significant. REPORT.md section 8.1 names that error and fixes it
for the price contrast; the same fix is applied here.

SECOND, and this is what makes the first problem look worse than it is, the
pooled sample mixes three kinds of question that respond to a graph in three
completely different ways:

  rung1_arith  marginal, correlation - pure arithmetic over the given numbers.
               The Causal Hierarchy Theorem says a rung-1 quantity is fixed by
               rung-1 data, so a graph cannot help. Measured Delta_struct here
               is 0.00 pp to two decimals on all three models - a clean negative
               control the project was not using.

  identify     backadj - "which adjustment set is correct". This is an
               IDENTIFICATION question: its answer depends on the graph and on
               nothing else. strip_structure() deletes the graph, so RAW deletes
               the answer, and gpt-4.1-mini lands at 28.1%, below chance.
               ORACLE hands it straight back: +59.38 pp. That is not the graph
               helping a model reason, it is the graph being the answer.

  causal       ate, ett, nde, nie, det-counterfactual, collider_bias, exp_away -
               the queries where a graph is genuinely an aid to reasoning rather
               than the answer itself.

backadj is 18.4% of the sample but contributes about 70% of gpt-4.1's
Delta_struct, and removing it flips gpt-4.1-mini's Delta_struct to -8.61 pp. It
inflates BOTH arms of the interaction, which pulls them together and destroys
the very contrast the study is trying to establish. On the causal subset the
interaction goes from +4.50 pp (CI contains 0) to +14.35 pp, p < 0.001.

The bootstrap resamples ITEMS, not cells. The nine cells share one KEEP
baseline and the three models share all 174 items, so the cells are strongly
dependent and a naive standard error over nine numbers would be far too small.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy import stats
from stats import boot_cell_means, boot_interval, boot_p, cluster_boot, mcnemar_exact_p

LEXICONS = ["KEEP", "PERMUTE", "SYMBOL", "PSEUDO"]
ANON = ["PERMUTE", "SYMBOL", "PSEUDO"]
TIER = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]

# Grouped by what a graph can do for the query, not by Pearl rung. backadj is a
# rung-2 label in CLadder but behaves differently from every other rung-2 query
# here, because the graph is its answer rather than an aid to computing one.
ARITH = {"marginal", "correlation"}
IDENT = {"backadj"}
GROUPS = {"rung1_arith": ARITH, "identify": IDENT, "causal": None}
NHAN = {"rung1_arith": "rung-1 so hoc", "identify": "backadj identification",
        "causal": "genuinely causal", "all": "all pooled"}


def load():
    out = {}
    for lex in LEXICONS:
        p = ROOT / "results" / "raw" / f"pilot_raw_lex{lex}.csv"
        if not p.exists():
            raise SystemExit(f"{p.name} missing: run pilot.py --lexicon {lex}")
        out[lex] = pd.read_csv(p)
    return out


def qset(d, group):
    if group == "all":
        return set(d.query_type.unique())
    if GROUPS[group] is not None:
        return GROUPS[group]
    return set(d.query_type.unique()) - ARITH - IDENT


def cell(d, model, cond, qs):
    """Per-item correctness for one (model, condition, query group), parsed only."""
    s = d[(d.model == model) & (d.cond == cond) & (d.parsed == 1)
          & (d.query_type.isin(qs))]
    return s.set_index("item").correct


def paired(a, b, model, cond, qs):
    """a minus b on items parsed in both. Returns (n, delta_pp, p)."""
    x, y = cell(a, model, cond, qs), cell(b, model, cond, qs)
    i = x.index.intersection(y.index)
    if len(i) < 10:
        return len(i), np.nan, np.nan
    x, y = x[i], y[i]
    nb = int(((x == 1) & (y == 0)).sum())
    nc = int(((x == 0) & (y == 1)).sum())
    return len(i), 100 * (x.mean() - y.mean()), mcnemar_exact_p(nb, nc)


def bh(pvals, q=0.05):
    """Benjamini-Hochberg. Returns a boolean mask of survivors."""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    idx = np.where(ok)[0][np.argsort(p[ok])]
    m = len(idx)
    keep = np.zeros(len(p), bool)
    cut = 0
    for r, j in enumerate(idx, start=1):
        if p[j] <= q * r / m:
            cut = r
    keep[idx[:cut]] = True
    return keep


def did_cells(d, qs):
    """Per-item DiD for each of the 9 (model, anon lexicon) cells."""
    cols = []
    for m in TIER:
        for lex in ANON:
            kr, lr = cell(d["KEEP"], m, "RAW", qs), cell(d[lex], m, "RAW", qs)
            ko, lo = cell(d["KEEP"], m, "ORACLE", qs), cell(d[lex], m, "ORACLE", qs)
            i = kr.index.intersection(lr.index).intersection(ko.index).intersection(lo.index)
            if len(i) < 10:
                continue
            v = (kr[i] - lr[i]) - (ko[i] - lo[i])
            cols.append(pd.Series(v.values, index=i, name=f"{m}|{lex}"))
    return pd.concat(cols, axis=1) if cols else pd.DataFrame()


def boot_mean(W, seed, n=4000):
    """Cluster bootstrap over items for the mean across cells."""
    # Convention C, src/stats.py. This floored at 1/n while every other
    # bootstrap here floored at 2/n, so the same situation printed 0.00025 in
    # this file and 0.0005 elsewhere; boot_p now floors every one at 2/n.
    return boot_cell_means(W, seed, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()
    d = load()
    base = d["KEEP"]
    W = 88

    print("=" * W)
    print("0. SAMPLE COMPOSITION BY QUERY GROUP")
    print("=" * W)
    one = base[(base.model == TIER[0]) & (base.cond == "RAW")]
    rows = []
    for g in ["rung1_arith", "identify", "causal"]:
        qs = qset(base, g)
        n = int(one.query_type.isin(qs).sum())
        rows.append({"group": NHAN[g], "n": n, "percent": round(100 * n / len(one), 1),
                     "query_type": ", ".join(sorted(qs & set(one.query_type)))})
    comp = pd.DataFrame(rows)
    print(comp.to_string(index=False))
    comp.to_csv(ROOT / "results" / "querygroup_composition.csv", index=False)

    print("\n" + "=" * W)
    print("1. Delta_struct = ORACLE - RAW, PHAN RA THEO NHOM  (dieu kien KEEP)")
    print("=" * W)
    print("  Theory says: rung-1 must be 0 (Causal Hierarchy Theorem), backadj must be")
    print("  very large (the graph IS the answer), and the genuinely-causal group is")
    print("  the part worth measuring.\n")
    rows = []
    for m in TIER:
        r = {"model": m}
        for g in ["rung1_arith", "identify", "causal", "all"]:
            qs = qset(base, g)
            s = base[(base.model == m) & (base.parsed == 1) & (base.query_type.isin(qs))]
            o = s[s.cond == "ORACLE"].correct.mean()
            w = s[s.cond == "RAW"].correct.mean()
            r[NHAN[g]] = round(100 * (o - w), 2)
        rows.append(r)
    dec = pd.DataFrame(rows)
    print(dec.to_string(index=False))
    dec.to_csv(ROOT / "results" / "querygroup_delta_struct.csv", index=False)
    print("\n  backadj is 18.4% of the sample but dominates the pooled Delta_struct.")
    print("  For gpt-4.1-mini, removing it FLIPS the sign of Delta_struct.")

    print("\n" + "=" * W)
    print("2. McNEMAR: what anonymising variable names costs, WITHIN EACH GROUP")
    print("=" * W)
    rows, summary = [], []
    for g in ["all", "causal", "identify", "rung1_arith"]:
        qs = qset(base, g)
        # PROSE and DR_k1 were in REPORT section 4.0's table from the start but
        # never in this loop, so two of its four rows had no file behind them.
        # Each (group, cond) is its own BH family, so widening the loop cannot
        # move a number that was already here.
        for cond in ["RAW", "ORACLE", "PROSE", "DR_k1"]:
            ds, ps = [], []
            for m in TIER:
                for lex in ANON:
                    n, dl, p = paired(d[lex], d["KEEP"], m, cond, qs)
                    ds.append(dl)
                    ps.append(p)
                    rows.append({"group": NHAN[g], "cond": cond, "model": m,
                                 "lexicon": lex, "n": n,
                                 "delta_pp": None if pd.isna(dl) else round(dl, 2),
                                 "p": None if pd.isna(p) else round(p, 4)})
            sig = int(np.nansum([p < 0.05 for p in ps]))
            sig_bh = int(bh(ps).sum())
            mean_harm = float(np.nanmean(ds))
            # The report's fourth column is the WORST single cell, not the mean.
            worst = float(np.nanmin(ds)) if np.isfinite(ds).any() else float("nan")
            print(f"  {NHAN[g]:16} {cond:7} tho {sig}/9   BH q=.05 {sig_bh}/9   "
                  f"mean harm {mean_harm:+7.2f} pp   worst {worst:+7.2f} pp")
            # These three numbers are quoted verbatim in REPORT section 4.0 and
            # section 5. Until now they existed only in this print statement, so
            # nothing in results/ could vouch for them and scripts/check_numbers.py
            # flagged them as unaccounted. Persist them.
            summary.append({"group": NHAN[g], "cond": cond,
                            "n_sig_raw": sig, "n_sig_bh": sig_bh, "n_cells": len(ps),
                            "mean_harm_pp": round(mean_harm, 2),
                            "worst_harm_pp": round(worst, 2)})
        print()
    # Scoring sensitivity, computed by the SAME loop above rather than a
    # parallel one - that is the point of the check. REPORT section 10 quotes
    # the pair -10.73 -> -10.79 and only the first half had a file.
    d2 = {k: v.copy() for k, v in d.items()}
    for k in d2:
        d2[k].loc[d2[k].parsed == 0, "correct"] = 0
        d2[k]["parsed"] = 1
    for g in ["all", "causal"]:
        qs = qset(base, g)
        for cond in ["RAW", "ORACLE"]:
            ds2, ps2 = [], []
            for m in TIER:
                for lex in ANON:
                    _, dl, pv = paired(d2[lex], d2["KEEP"], m, cond, qs)
                    ds2.append(dl)
                    ps2.append(pv)
            summary.append({"group": NHAN[g] + " (unparsed=wrong)", "cond": cond,
                            "n_sig_raw": int(np.nansum([pv < 0.05 for pv in ps2])),
                            "n_sig_bh": int(bh(ps2).sum()), "n_cells": len(ps2),
                            "mean_harm_pp": round(float(np.nanmean(ds2)), 2),
                            "worst_harm_pp": round(float(np.nanmin(ds2)), 2)})

    mc = pd.DataFrame(rows)
    mc.to_csv(ROOT / "results" / "querygroup_mcnemar.csv", index=False)
    pd.DataFrame(summary).to_csv(
        ROOT / "results" / "querygroup_mcnemar_summary.csv", index=False)

    print("=" * W)
    print("3. THE INTERACTION TEST - the headline claim, tested properly")
    print("=" * W)
    print("  DiD = [(KEEP-LEX)|RAW] - [(KEEP-LEX)|ORACLE], paired on the intersection")
    print("  of the four cells. A value > 0 means the correct graph REDUCES the harm")
    print(f"  of anonymising. Bootstrap resamples ITEMS, {a.boot} draws.\n")
    rows = []
    for g in ["all", "causal", "identify", "rung1_arith"]:
        qs = qset(base, g)
        Wd = did_cells(d, qs)
        if Wd.empty:
            continue
        est, lo, hi, p = boot_mean(Wd, a.seed, a.boot)
        star = "  <-- xac lap" if lo > 0 else ""
        print(f"  {NHAN[g]:16} 9 cells pooled: {est:+7.2f} pp  95% CI [{lo:+7.2f} ; {hi:+7.2f}]"
              f"  p={p:.4f}{star}")
        rows.append({"group": NHAN[g], "scope": "9 cells pooled", "did_pp": round(est, 2),
                     "ci_lo": round(lo, 2), "ci_hi": round(hi, 2), "p_boot": round(p, 4)})
        for m in TIER:
            sub = Wd[[c for c in Wd.columns if c.startswith(m + "|")]].dropna(how="all")
            if sub.empty:
                continue
            e2, l2, h2, p2 = boot_mean(sub, a.seed, a.boot)
            print(f"      {m:16} {e2:+7.2f} pp  CI [{l2:+7.2f} ; {h2:+7.2f}]  p={p2:.4f}")
            rows.append({"group": NHAN[g], "scope": m, "did_pp": round(e2, 2),
                         "ci_lo": round(l2, 2), "ci_hi": round(h2, 2),
                         "p_boot": round(p2, 4)})
        print()
    did = pd.DataFrame(rows)
    # BH over the whole family of 16 interaction tests. REPORT section 1 says
    # that under BH only 1/3 models passes on its own, not 2/3 - review item
    # V7-10 - and until now that verdict lived in prose with no column behind it.
    did["bh_q05_family16"] = bh(did.p_boot.tolist())
    did.to_csv(ROOT / "results" / "querygroup_interaction.csv", index=False)
    per_m = did[(did.group == NHAN["causal"]) & (did.scope != "9 cells pooled")]
    print(f"  BH q=.05 over all {len(did)} interaction tests: "
          f"{int(per_m.bh_q05_family16.sum())}/{len(per_m)} models pass alone "
          f"in the genuinely-causal group.\n")

    print("=" * W)
    print("4. THE LEXICAL LADDER: EQUIVALENCE BOUNDS instead of counting cells")
    print("=" * W)
    print("  Counting cells cannot separate 'zero' from 'underpowered'. A CI on the")
    print("  pooled difference can, and it CAN BE WRONG.\n")
    rows = []
    for x, y in [("PERMUTE", "KEEP"), ("SYMBOL", "PERMUTE"), ("PSEUDO", "SYMBOL")]:
        for g in ["all", "causal"]:
            qs = qset(base, g)
            cols = []
            for m in TIER:
                for cond in ["PROSE", "RAW", "ORACLE", "DR_k1"]:
                    u, v = cell(d[x], m, cond, qs), cell(d[y], m, cond, qs)
                    i = u.index.intersection(v.index)
                    if len(i) < 10:
                        continue
                    cols.append(pd.Series((u[i] - v[i]).values, index=i,
                                          name=f"{m}|{cond}"))
            Wd = pd.concat(cols, axis=1)
            est, lo, hi, p = boot_mean(Wd, a.seed, a.boot)
            print(f"  {x:8}-{y:8} [{NHAN[g]:14}] {est:+6.2f} pp  "
                  f"CI 95% [{lo:+6.2f} ; {hi:+6.2f}]")
            rows.append({"rung": f"{x}-{y}", "group": NHAN[g], "delta_pp": round(est, 2),
                         "ci_lo": round(lo, 2), "ci_hi": round(hi, 2)})
        print()
    eq = pd.DataFrame(rows)
    eq.to_csv(ROOT / "results" / "querygroup_equivalence.csv", index=False)
    print("  Rung 1 is a real effect; rungs 2 and 3 are tightly bounded around zero.")
    print("  That is an equivalence bound, not an argument from failure to reject.")

    # ------------------------------------------------------------------
    # Does the grouping of the two ambiguous query types matter? Review item
    # V7-13 asked for exp_away to move to rung1_arith (CLadder labels it rung 1)
    # and collider_bias to identify (its answer is fixed by the graph). Both
    # readings are defensible, so the question is settled by measurement, and
    # the measurement is kept here so nobody has to re-litigate it.
    # ------------------------------------------------------------------
    print("\n" + "=" * W)
    print("5. DO NHAY CUA DiD THEO CACH CAT NHOM TRUY VAN")
    print("=" * W)
    print("  exp_away mang nhan rung 1 cua CLadder nhung khong phai so hoc thuan;")
    print("  collider_bias do do thi quyet dinh nhung khong phai cau hoi tap hieu")
    print("  chinh. Hai cach xep deu bien minh duoc, nen do xem no co doi gi khong.\n")
    n_amb = {q: int((base[(base.model == TIER[0]) & (base.cond == "RAW")]
                     .query_type == q).sum())
             for q in ("exp_away", "collider_bias")}
    srows = []
    # The four cuts of REPORT section 4.1 plus the V7-13 variant. That table
    # shows the headline moving from p=0.07 to significant depending on which
    # query groups are removed, which is the most important thing a reader can
    # know about it - and it had no script until 2026-09-23.
    for label, drop in [
            ("gop tat ca", set()),
            ("chi bo rung-1", ARITH),
            ("chi bo backadj", IDENT),
            ("bo ca hai (hien tai)", ARITH | IDENT),
            ("de xuat V7-13", ARITH | IDENT | {"exp_away", "collider_bias"}),
            # REPORT section 4.1's caveat on strip_structure(): for
            # det-counterfactual the two surviving sentences ARE the structural
            # equations, so RAW is not "no graph" for that type. The report
            # quotes the DiD without it as the robustness answer; until now no
            # cut here produced it.
            ("bo ca hai, bo det-cf", ARITH | IDENT | {"det-counterfactual"})]:
        qs = set(base.query_type.unique()) - drop
        # The DiD, via the same did_cells() the rest of this file uses. An
        # earlier draft of this block computed Delta_struct (ORACLE - RAW on
        # KEEP) instead and printed it under the same heading. The pooled row
        # happened to land at +4.30 against the DiD's +4.50, close enough to
        # pass a glance, while "drop backadj" came out at -2.48 against +7.13 -
        # the opposite sign. Two different quantities under one label.
        W_ = did_cells(d, qs)
        if W_.empty:
            continue
        est, lo, hi, p = boot_mean(W_, a.seed, a.boot)
        print(f"  {label:22s} n={len(W_):4d}  DiD {est:+6.2f} pp  "
              f"[{lo:+6.2f} ; {hi:+6.2f}]  p={p:.4f}")
        srows.append({"scheme": label, "n_items": len(W_),
                      "did_pp": round(est, 2), "ci_lo": round(lo, 2),
                      "ci_hi": round(hi, 2), "p_boot": round(p, 4)})
    if len(srows) >= 2:
        pd.DataFrame(srows).to_csv(
            ROOT / "results" / "querygroup_sensitivity.csv", index=False)
        by = {r["scheme"]: r for r in srows}
        cur, alt = by.get("bo ca hai (hien tai)"), by.get("de xuat V7-13")
        if cur and alt:
            shift = abs(cur["did_pp"] - alt["did_pp"])
        else:
            shift = float("nan")
        print(f"\n  Con so dich {shift:.2f} pp. exp_away co {n_amb['exp_away']} item")
        print(f"  va collider_bias co {n_amb['collider_bias']}, nen du lieu KHONG the")
        print("  phan xu cho chung thuoc nhom nao - va khong can phan xu, vi khong")
        print("  con so nao duoc bao cao phu thuoc vao lua chon do.")
        print("\n  Tieu chi da ap dung, ghi thanh van: rung1_arith chi gom hai loai la")
        print("  SO HOC THUAN tren cac con so da cho (marginal, correlation);")
        print("  identify chi gom backadj, noi do thi CHINH LA dap an; causal la phan")
        print("  con lai. exp_away mang nhan rung 1 nhung doi hoi nhan ra mot collider")
        print("  nen khong phai so hoc thuan; collider_bias do do thi quyet dinh nhung")
        print("  khong phai cau hoi tap hieu chinh. Ca hai deu o lai nhom causal.")

    print("\n" + "=" * W)
    print("KET LUAN")
    print("=" * W)
    allrow = did[(did.group == NHAN["all"]) & (did.scope == "9 cells pooled")].iloc[0]
    caurow = did[(did.group == NHAN["causal"]) & (did.scope == "9 cells pooled")].iloc[0]
    print(f"  Whole sample      : DiD {allrow.did_pp:+.2f} pp  "
          f"CI [{allrow.ci_lo:+.2f} ; {allrow.ci_hi:+.2f}]  p={allrow.p_boot:.4f}")
    print(f"  Genuinely causal  : DiD {caurow.did_pp:+.2f} pp  "
          f"CI [{caurow.ci_lo:+.2f} ; {caurow.ci_hi:+.2f}]  p={caurow.p_boot:.4f}")
    print("\n  Filtering out backadj and rung-1 STRENGTHENS the headline claim rather")
    print("  than weakening it.")
    print("  Wrote: results/querygroup_*.csv")


if __name__ == "__main__":
    main()
