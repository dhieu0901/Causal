"""Every structure arm against RAW - the contrast the project was missing.

Why this file exists. The project's headline estimator is a difference in
differences:

    DiD_i(C) = [(KEEP - PSEUDO) | RAW] - [(KEEP - PSEUDO) | C]

A DiD is an INTERACTION, and an interaction is blind to anything that moves both
of its branches by the same amount. That blindness produced the misleading
reading in results/structure_arms.csv, where

    ORACLE minus DR_k1 = +1.49  CI [-1.86 ; +5.01]  p=0.41

was taken to mean "a wrong graph helps about as much as a right one". It means no
such thing. Reversing edges costs accuracy on BOTH lexical branches, so the
interaction barely moves while the accuracies themselves move a great deal.

This file measures the plain paired contrast instead:

    delta_i(C) = acc_i(C) - acc_i(RAW)

against RAW, the arm that supplies no structure block at all. That is the
comparison a practitioner actually faces: is it better to hand the model a graph,
or nothing? And it is the comparison that reveals the dose-response curve,
because delta is measured against a real no-structure baseline.

STRATIFICATION IS NOT OPTIONAL, and the first version of this file got it wrong.
REPORT section 4.0 prints the words "bat buoc doc truoc moi con so khac" - read
before every other number - because the three query groups respond in OPPOSITE
directions. `backadj` is the group where the graph IS the answer: strip_structure
deletes the answer and ORACLE hands it back, worth +27.36 to +43.77 pp. On
genuinely causal group the same block is worth about ZERO on the KEEP branch.
Pooling them produced a headline of "+4.56 to +6.43 pp" that was almost entirely
backadj, and an unstratified average is blind to two subgroups moving in opposite
directions in exactly the way a DiD is blind to two branches moving together.

So every table here reports all three groups. Benjamini-Hochberg is applied
WITHIN each group, because the groups are three different scientific questions
rather than one family.

Honesty note, to be carried into the write-up. This contrast was specified AFTER
the DiD results were in hand, i.e. it is post hoc. Two things are done about
that, and neither should be dropped when reporting: the tests are corrected by
Benjamini-Hochberg, and the estimates are reported separately for all three
samples so a reader can see which effects reproduce across samples.

Run:  python scripts/analyze_vs_raw.py
Writes: results/vs_raw.csv
        results/vs_raw_rq2.csv
        results/vs_raw_trend.csv
        results/vs_raw_moderator.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
RESULTS = ROOT / "results"

from analyze_querygroup import ARITH, IDENT

# The three groups of REPORT section 4.0. `causal` is the complement, computed
# per sample because not every sample carries every query type.
GROUPS = ["causal", "backadj", "rung1_arith"]

SEED = 20260907
NBOOT = 4000
ALPHA = 0.05

# (tag, n_items, the conditions that sample actually ran besides RAW)
SAMPLES = [
    ("lex", 174, ["ORACLE", "PROSE", "DR_k1"]),
    ("n600", 580, ["ORACLE", "PROSE", "DR_k1"]),
    ("price400", 399, ["ORACLE", "PROSE",
                       "DR_k1", "DR_k2", "DR_k3",
                       "ED_k1", "ED_k2", "ED_k3",
                       "FE_k1", "FE_k2"]),
]
LEXICONS = ["KEEP", "PSEUDO"]

# price400 was drawn with k up to 3, which the three 2-edge families cannot
# support, so it silently excludes them. n600 has all ten. Any comparison
# between the two samples has to hold this constant.
PRICE400_FAMILIES = ["IV", "arrowhead", "confounding", "diamond",
                     "diamondcut", "frontdoor", "mediation"]
SIMPLE_FAMILIES = ["chain", "collision", "fork"]


def by_group(d: pd.DataFrame, group: str) -> pd.DataFrame:
    """Restrict to one of REPORT section 4.0's three query groups."""
    if group == "backadj":
        return d[d.query_type.isin(IDENT)]
    if group == "rung1_arith":
        return d[d.query_type.isin(ARITH)]
    if group == "causal":
        return d[~d.query_type.isin(ARITH | IDENT)]
    return d


def paired(d: pd.DataFrame, cond: str, families=None, group=None) -> pd.Series:
    """acc(cond) - acc(RAW), per item, averaged over the models that ran both.

    Pairing within item is the design's main strength: the same item, the same
    story, the same gold label, differing only in the structure block. It is
    preserved here rather than comparing two group means.
    """
    if group is not None:
        d = by_group(d, group)
    if families is not None:
        d = d[d.graph_id.isin(families)]
    cols = []
    for m in sorted(d.model.unique()):
        r = d[(d.model == m) & (d.cond == "RAW")].set_index("item").correct
        c = d[(d.model == m) & (d.cond == cond)].set_index("item").correct
        i = r.index.intersection(c.index)
        if len(i) < 5:
            continue
        cols.append(pd.Series((c[i] - r[i]).values, index=i, name=m))
    if not cols:
        return pd.Series(dtype=float)
    return pd.concat(cols, axis=1).mean(axis=1).dropna()


def boot(v: pd.Series, seed: int = SEED, n: int = NBOOT):
    """Cluster bootstrap over items. Returns (estimate_pp, lo, hi, p)."""
    x = v.values
    if len(x) < 5:
        return (np.nan,) * 4
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for b in range(n):
        out[b] = x[rng.integers(0, len(x), len(x))].mean()
    # Clamp to 1. Draws that land exactly on 0 are counted by BOTH tails, so
    # 2*min(...) can exceed 1 - results/family_breakdown.csv carried a p of
    # 1.0255 from this. A probability above 1 is never a rounding curiosity to a
    # reader opening the CSV; it discredits the whole table.
    p = min(1.0, 2 * min((out <= 0).mean(), (out >= 0).mean()))
    return (100 * x.mean(), 100 * np.percentile(out, 2.5),
            100 * np.percentile(out, 97.5), max(p, 2.0 / n))


def benjamini_hochberg(p: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Step-up procedure. Returns a boolean mask of the rejected hypotheses."""
    m = len(p)
    order = np.argsort(p)
    thresh = (np.arange(1, m + 1) / m) * alpha
    below = p[order] <= thresh
    keep = np.zeros(m, dtype=bool)
    if below.any():
        keep[order[: np.where(below)[0].max() + 1]] = True
    return keep


def load(tag: str, lexicon: str) -> pd.DataFrame:
    d = pd.read_csv(RESULTS / f"pilot_raw_{tag}{lexicon}.csv")
    return d[d.parsed == 1]


def main_table():
    rows, keep_series = [], {}
    for tag, _, conds in SAMPLES:
        for lex in LEXICONS:
            d = load(tag, lex)
            for group in GROUPS:
                for cond in conds:
                    v = paired(d, cond, group=group)
                    if v.empty:
                        continue
                    est, lo, hi, p = boot(v)
                    rows.append(dict(query_group=group, sample=tag, lexicon=lex,
                                     cond=cond, delta_pp=round(est, 2),
                                     ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                                     p_boot=round(p, 4), n_items=len(v)))
                    if group == "causal":
                        keep_series[(tag, lex, cond)] = v
    R = pd.DataFrame(rows)
    # Corrected WITHIN each query group: the three groups are three different
    # questions, not one family, and pooling them is what broke the first
    # version of this analysis.
    R["survives_BH"] = False
    for group in GROUPS:
        m = (R.query_group == group).values
        if m.any():
            R.loc[m, "survives_BH"] = benjamini_hochberg(R.loc[m, "p_boot"].values)
    return R, keep_series


def trend_table(series) -> pd.DataFrame:
    """Slope of the per-item loss on k, for the arms that ran k = 1, 2, 3.

    One slope is fitted PER ITEM across its own three k values, then the slopes
    are bootstrapped over items. Fitting per item keeps the pairing; fitting one
    line through three group means would throw it away.
    """
    rows = []
    for arm in ["DR", "ED"]:
        for lex in LEXICONS:
            need = [("price400", lex, f"{arm}_k{k}") for k in (1, 2, 3)]
            if not all(key in series for key in need):
                continue
            M = pd.concat({k: series[need[k - 1]] for k in (1, 2, 3)}, axis=1).dropna()
            ks = np.array([1.0, 2.0, 3.0])
            kc = ks - ks.mean()
            slope = (M.values * kc).sum(axis=1) / (kc ** 2).sum()
            est, lo, hi, p = boot(pd.Series(slope))
            rows.append(dict(arm=arm, lexicon=lex, sample="price400",
                             estimate=round(est, 2), ci_lo=round(lo, 2),
                             ci_hi=round(hi, 2), p_boot=round(p, 4),
                             n_items=len(slope),
                             unit="pp per extra reversed edge"))
    # reversal against omission, head to head at k = 1
    for lex in LEXICONS:
        a = series.get(("price400", lex, "DR_k1"))
        b = series.get(("price400", lex, "ED_k1"))
        if a is None or b is None:
            continue
        i = a.index.intersection(b.index)
        est, lo, hi, p = boot(pd.Series((a[i] - b[i]).values))
        rows.append(dict(arm="DR_k1 minus ED_k1", lexicon=lex, sample="price400",
                         estimate=round(est, 2), ci_lo=round(lo, 2),
                         ci_hi=round(hi, 2), p_boot=round(p, 4), n_items=len(i),
                         unit="pp, reversal against omission"))
    T = pd.DataFrame(rows)
    # Corrected within this table. Without it the omission slope on KEEP reads as
    # significant at 0.026 and the error-type dissociation below dissolves.
    T["survives_BH"] = benjamini_hochberg(T.p_boot.values)
    return T


def rq2_table() -> pd.DataFrame:
    """ORACLE minus DR_k1, paired within item, on the causal group.

    This is the DIRECT test of RQ2 - must the structure block be CORRECT? - and
    for a long time the project answered it from results/structure_arms.csv,
    where the same contrast reads +1.49, CI [-1.86 ; +5.01], p=0.41, i.e. "not
    distinguishable". That row is a difference of two DIFFERENCES-IN-DIFFERENCES.
    An interaction cannot see a shift that moves both lexical branches, and
    reversing an edge moves both.

    Taking the contrast directly, item by item, on the query group that needs
    causal reasoning, reverses the answer: the correct graph wins in every cell.
    """
    rows = []
    for tag, _, conds in SAMPLES:
        if "DR_k1" not in conds:
            continue
        for lex in LEXICONS:
            d = load(tag, lex)
            a, b = paired(d, "ORACLE", group="causal"), paired(d, "DR_k1", group="causal")
            i = a.index.intersection(b.index)
            if len(i) < 5:
                continue
            est, lo, hi, p = boot(pd.Series((a[i] - b[i]).values))
            rows.append(dict(sample=tag, lexicon=lex, quantity="ORACLE minus DR_k1",
                             query_group="causal", delta_pp=round(est, 2),
                             ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                             p_boot=round(p, 4), n_items=len(i)))
    R = pd.DataFrame(rows)
    R["survives_BH"] = benjamini_hochberg(R.p_boot.values)
    return R


def moderator_table() -> pd.DataFrame:
    """Does graph complexity moderate the damage from a wrong graph?

    On the CAUSAL group only, for the reason given at the top of this file.

    DR_k1 does not reproduce across samples: price400 KEEP gives -3.23 while n600
    KEEP gives +0.69. The samples differ in composition, because price400 cannot
    contain the three 2-edge families. This table holds composition constant.
    """
    rows = []
    for lex in LEXICONS:
        d = load("n600", lex)
        for label, fams in [("all 10 families", None),
                            ("7 families, the price400 set", PRICE400_FAMILIES),
                            ("3 simple 2-edge families", SIMPLE_FAMILIES)]:
            for cond in ["ORACLE", "DR_k1"]:
                v = paired(d, cond, fams, group="causal")
                if v.empty:
                    continue
                est, lo, hi, p = boot(v)
                rows.append(dict(sample="n600", lexicon=lex, subset=label, cond=cond,
                                 delta_pp=round(est, 2), ci_lo=round(lo, 2),
                                 ci_hi=round(hi, 2), p_boot=round(p, 4), n_items=len(v)))
    return pd.DataFrame(rows)


def main() -> None:
    R, series = main_table()
    T = trend_table(series)
    Q = rq2_table()
    M = moderator_table()

    print("=" * 78)
    print("1. EVERY STRUCTURE ARM AGAINST RAW  -  acc(cond) - acc(RAW), paired")
    print("=" * 78)
    print("\n  Positive means the structure block HELPED relative to supplying none.")
    print(f"  Benjamini-Hochberg at {ALPHA} WITHIN each query group.\n")
    for group in GROUPS:
        g = R[R.query_group == group]
        if g.empty:
            continue
        print(f"  ######## QUERY GROUP: {group}  ({len(g)} tests) ########")
        for tag, _, _ in SAMPLES:
            s = g[g["sample"] == tag]
            if s.empty:
                continue
            print(f"  --- {tag} ---")
            print(f"  {'lex':8s} {'cond':8s} {'delta_pp':>9s} {'ci_lo':>8s} {'ci_hi':>8s}"
                  f" {'p':>8s} {'n':>5s}  BH")
            for _, r in s.iterrows():
                print(f"  {r.lexicon:8s} {r.cond:8s} {r.delta_pp:>9.2f} {r.ci_lo:>8.2f}"
                      f" {r.ci_hi:>8.2f} {r.p_boot:>8.4f} {r.n_items:>5d}"
                      f"  {'yes' if r.survives_BH else 'no'}")
            print()

    print("=" * 78)
    print("2. DOSE - slope across k, and reversal against omission")
    print("=" * 78 + "\n")
    print(f"  Benjamini-Hochberg at {ALPHA} within this table, {len(T)} tests.\n")
    print(f"  {'arm':20s} {'lex':8s} {'estimate':>9s} {'ci_lo':>8s} {'ci_hi':>8s}"
          f" {'p':>8s} {'n':>5s}  BH")
    for _, r in T.iterrows():
        print(f"  {r.arm:20s} {r.lexicon:8s} {r.estimate:>9.2f}"
              f" {r.ci_lo:>8.2f} {r.ci_hi:>8.2f} {r.p_boot:>8.4f} {r.n_items:>5d}"
              f"  {'yes' if r.survives_BH else 'no'}")

    print("\n" + "=" * 78)
    print("3. RQ2 DIRECT - ORACLE minus DR_k1, paired, causal group")
    print("=" * 78 + "\n")
    print("  Must the structure block be CORRECT? This is the contrast that answers it.")
    print(f"  Benjamini-Hochberg at {ALPHA} within this table.\n")
    print(f"  {'sample':10s} {'lex':8s} {'delta_pp':>9s} {'ci_lo':>8s} {'ci_hi':>8s}"
          f" {'p':>8s} {'n':>5s}  BH")
    for _, r in Q.iterrows():
        print(f"  {r['sample']:10s} {r.lexicon:8s} {r.delta_pp:>9.2f} {r.ci_lo:>8.2f}"
              f" {r.ci_hi:>8.2f} {r.p_boot:>8.4f} {r.n_items:>5d}"
              f"  {'yes' if r.survives_BH else 'no'}")
    print(f"""
  Positive in {int((Q.delta_pp > 0).sum())} of {len(Q)} cells, surviving the
  correction in {int(Q.survives_BH.sum())}. The correct graph beats a graph with
  one reversed edge across three independently drawn samples.

  Compare results/structure_arms.csv, which puts the same contrast at +1.49,
  CI [-1.86 ; +5.01], p=0.41 - "not distinguishable". That row is a difference of
  two INTERACTIONS and cannot see a shift that moves both lexical branches.
  Reversing an edge moves both. Taken directly, item by item, the answer flips.

  Note the two cells that do NOT survive: lex KEEP (n=86, the smallest sample)
  and price400 PSEUDO. Both are still positive. The pattern is consistent; the
  per-cell power is not.""")

    print("\n" + "=" * 78)
    print("4. MODERATOR - is the damage from a wrong graph larger in complex graphs?")
    print("=" * 78 + "\n")
    print(f"  {'lex':8s} {'subset':30s} {'cond':8s} {'delta_pp':>9s} {'ci_lo':>8s}"
          f" {'ci_hi':>8s} {'p':>8s} {'n':>5s}")
    for _, r in M.iterrows():
        print(f"  {r.lexicon:8s} {r.subset:30s} {r.cond:8s} {r.delta_pp:>9.2f}"
              f" {r.ci_lo:>8.2f} {r.ci_hi:>8.2f} {r.p_boot:>8.4f} {r.n_items:>5d}")

    print("""
  HOW TO READ THIS FILE.

  READ THE `causal` GROUP. It is the one this project exists to measure. The
  `backadj` group is not a nuisance but it is a different question: there the
  graph IS the answer, strip_structure deletes it, ORACLE hands it back, and the
  block is worth +27.36 to +43.77 pp - six cells out of six, every one surviving
  the correction. Quote that separately or not at all; never average it in.

  THE RESULT, on the causal group, is an ASYMMETRY, not a sign flip.

  No upside with familiar variable names. A correct graph is worth about zero on
  the KEEP branch: -5.04, -0.17, -0.08 across the three samples, not one of them
  surviving the correction. PROSE behaves the same way.

  Real downside, at one edge. A single reversed edge costs -8.72 (lex, survives),
  -6.11 (n600, survives) and -5.16 (price400, nominal only). Two and three
  reversed edges cost -7.36 and -6.90, both surviving. So the damage is
  established while the benefit is not - handing a self-extracted DAG to a model
  has NEGATIVE expected value at any extraction error rate above zero.

  The upside appears only once the lexical anchor is gone. On PSEUDO the correct
  graph is worth +7.75, +6.30 (survives) and +1.01. That is the project's
  original interaction claim, and it is the half that survives.

  WHAT WAS WITHDRAWN when this table was stratified. The unstratified version of
  this file reported "+4.56 to +6.43 pp on KEEP, all surviving BH" and called the
  curve a sign flip. Both are gone: that range was almost entirely backadj, and
  there is no flip because the starting point is zero, not positive. The claim
  that "no omission cell is established anywhere" is also gone - ED_k2 on KEEP
  reaches -5.39 nominally, so reversal against omission is a difference of degree
  rather than a qualitative dissociation.

  DOSE IS CONFOUNDED WITH COMPOSITION. See scripts/classify_perturbations.py:
  20.6% of the reversals actually drawn at k=1 leave the ATE estimand unchanged,
  against 0.0% at k=2 and k=3. The apparent slope across k is largely that share
  collapsing, not the dose biting harder. Do not quote a pp-per-edge slope from
  the trend table without that column beside it.""")

    R.to_csv(RESULTS / "vs_raw.csv", index=False)
    T.to_csv(RESULTS / "vs_raw_trend.csv", index=False)
    Q.to_csv(RESULTS / "vs_raw_rq2.csv", index=False)
    M.to_csv(RESULTS / "vs_raw_moderator.csv", index=False)
    print(f"\n  wrote results/vs_raw.csv ({len(R)} rows), "
          f"vs_raw_trend.csv ({len(T)}), vs_raw_rq2.csv ({len(Q)}), vs_raw_moderator.csv ({len(M)})")


if __name__ == "__main__":
    main()
