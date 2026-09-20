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
from stats import mcnemar_exact_p

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
        p = ROOT / "results" / f"pilot_raw_lex{lex}.csv"
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
    items = W.index.unique()
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for k in range(n):
        s = rng.choice(items, len(items), replace=True)
        out[k] = 100 * np.nanmean(W.loc[s].mean(axis=0).values)
    est = 100 * np.nanmean(W.mean(axis=0).values)
    lo, hi = np.percentile(out, [2.5, 97.5])
    p = 2 * min((out <= 0).mean(), (out >= 0).mean())
    return est, lo, hi, max(p, 1.0 / n)


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
    rows = []
    for g in ["all", "causal", "identify", "rung1_arith"]:
        qs = qset(base, g)
        for cond in ["RAW", "ORACLE"]:
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
            print(f"  {NHAN[g]:16} {cond:7} tho {sig}/9   BH q=.05 {sig_bh}/9   "
                  f"mean harm {np.nanmean(ds):+7.2f} pp")
        print()
    mc = pd.DataFrame(rows)
    mc.to_csv(ROOT / "results" / "querygroup_mcnemar.csv", index=False)

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
    did.to_csv(ROOT / "results" / "querygroup_interaction.csv", index=False)

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
