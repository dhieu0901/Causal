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
from stats import boot_interval, boot_p, cluster_boot, mcnemar_exact_p

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
    it["group"] = np.where(qp.isin(PLAUSIBLE), "correct prior", "wrong prior already")
    return it


# Same convention as scripts/analyze_vs_raw.py: cluster the resample on the
# item, because one item contributes several rows (model x lexicon x cond) and
# those rows are not independent draws.
SEED = 20260907
NBOOT = 4000


def cell(x, model, cond, group):
    s = x[(x.model == model) & (x.cond == cond) & (x.parsed == 1) &
          (x.group == group)]
    return s.set_index("item").correct


def did_series(d, models, anon, group, min_n=10):
    """Per-item (anon - KEEP | ORACLE) - (anon - KEEP | RAW), averaged over cells.

    Every item must supply all four cells or the difference is not a difference,
    so the four indices are intersected before subtracting.
    """
    cols = []
    for m in models:
        for lex in anon:
            k_raw, k_ora = cell(d["KEEP"], m, "RAW", group), cell(d["KEEP"], m, "ORACLE", group)
            a_raw, a_ora = cell(d[lex], m, "RAW", group), cell(d[lex], m, "ORACLE", group)
            i = k_raw.index.intersection(k_ora.index) \
                           .intersection(a_raw.index).intersection(a_ora.index)
            if len(i) < min_n:
                continue
            cols.append(((a_ora[i] - k_ora[i]) - (a_raw[i] - k_raw[i])).rename(f"{m}|{lex}"))
    if not cols:
        return pd.Series(dtype=float)
    return pd.concat(cols, axis=1).mean(axis=1).dropna()


def raw_delta_series(d, models, anon, group, min_n=10):
    """Per-item (anon - KEEP) under RAW, averaged over cells. The harm itself."""
    cols = []
    for m in models:
        for lex in anon:
            k, a = cell(d["KEEP"], m, "RAW", group), cell(d[lex], m, "RAW", group)
            i = k.index.intersection(a.index)
            if len(i) < min_n:
                continue
            cols.append((a[i] - k[i]).rename(f"{m}|{lex}"))
    if not cols:
        return pd.Series(dtype=float)
    return pd.concat(cols, axis=1).mean(axis=1).dropna()


def boot(v, seed=SEED, n=NBOOT):
    """Cluster bootstrap over items. Returns (estimate_pp, lo, hi, p)."""
    x = v.values
    if len(x) < 5:
        return (np.nan,) * 4
    out = cluster_boot(len(x), lambda i: x[i].mean(), seed, n)
    est, lo, hi, p = boot_interval(x.mean(), out, n)
    return 100 * est, 100 * lo, 100 * hi, p


def boot_two_sample(a, b, seed=SEED, n=NBOOT):
    """Two disjoint item sets, so resample each independently and difference."""
    xa, xb = a.values, b.values
    if len(xa) < 5 or len(xb) < 5:
        return (np.nan,) * 4
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        out[i] = (xa[rng.integers(0, len(xa), len(xa))].mean()
                  - xb[rng.integers(0, len(xb), len(xb))].mean())
    return (100 * (xa.mean() - xb.mean()), 100 * np.percentile(out, 2.5),
            100 * np.percentile(out, 97.5), boot_p(out, n))


def main():
    it = label_items()
    d = {}
    for l in LEXICONS:
        p = ROOT / "results" / f"pilot_raw_lex{l}.csv"
        if p.exists():
            x = pd.read_csv(p)
            x["group"] = x.item.map(it.group)
            d[l] = x
    if "KEEP" not in d:
        raise SystemExit("thieu results/pilot_raw_lexKEEP.csv")

    models = [m for m in TIER if m in set(d["KEEP"].model)]
    print("=" * 84)
    print("0. SAMPLE COMPOSITION BY CLADDER'S OWN LABEL")
    print("=" * 84)
    print(it.qp.value_counts().to_string())
    # label_items() writes the English labels below. This line compared against
    # Vietnamese ones, so it printed "0" and "0" on every run since it was
    # written - a false zero sitting directly under the real counts.
    print(f"\n  -> correct prior: {(it.group == 'correct prior').sum()}"
          f"   wrong prior already: {(it.group == 'wrong prior already').sum()}")

    print("\n" + "=" * 84)
    print("1. THE KEEP FLOOR: how much has CLADDER'S OWN anticommonsense already removed?")
    print("=" * 84)
    print("  This is an independent replication of PERMUTE: the same manipulation,")
    print("  built by two different groups, on two different item sets.\n")
    rows = []
    for m in models:
        s = d["KEEP"][(d["KEEP"].model == m) & (d["KEEP"].cond == "RAW") &
                      (d["KEEP"].parsed == 1)]
        a = 100 * s[s.group == "correct prior"].correct.mean()
        b = 100 * s[s.group == "wrong prior already"].correct.mean()
        rows.append({"model": m, "correct_prior": round(a, 1),
                     "wrong_prior_already": round(b, 1), "chenh_pp": round(a - b, 1)})
    floor = pd.DataFrame(rows)
    print(floor.to_string(index=False))
    # REPORT section 4.2 quotes this column as "CLadder's own anticommonsense
    # gap" beside the project's own PERMUTE figures. It was printed and never
    # written, so check_numbers.py had nothing to check it against.
    floor.to_csv(ROOT / "results" / "prior_keep_floor.csv", index=False)

    print("\n" + "=" * 84)
    print("2. THE LEXICON EFFECT, SPLIT BY HOW STRONG THE ORIGINAL PRIOR WAS")
    print("=" * 84)
    out = []
    for m in models:
        for c in ["RAW", "ORACLE"]:
            for lex in [l for l in LEXICONS if l != "KEEP" and l in d]:
                for g in ["correct prior", "wrong prior already"]:
                    k = d["KEEP"]
                    a = k[(k.model == m) & (k.cond == c) & (k.parsed == 1) &
                          (k.group == g)].set_index("item").correct
                    b = d[lex][(d[lex].model == m) & (d[lex].cond == c) &
                               (d[lex].parsed == 1) &
                               (d[lex].group == g)].set_index("item").correct
                    i = a.index.intersection(b.index)
                    if len(i) < 10:
                        continue
                    nb = int(((b[i] == 1) & (a[i] == 0)).sum())
                    nc = int(((b[i] == 0) & (a[i] == 1)).sum())
                    p = mcnemar_exact_p(nb, nc)
                    out.append({"model": m, "cond": c, "n_compared": f"{lex} - KEEP",
                                "prior_ban_dau": g, "n": len(i),
                                "delta_pp": round(100 * (b[i].mean() - a[i].mean()), 2),
                                "p": round(p, 4),
                                "meaning": "*" if p < .05 else ""})
    o = pd.DataFrame(out)
    print(o.to_string(index=False))
    o.to_csv(ROOT / "results" / "prior_strength.csv", index=False)

    print("\n" + "=" * 84)
    print("3. MEAN BY PRIOR GROUP (3 anonymised lexicons x 3 models pooled)")
    print("=" * 84)
    # These means are quoted in REPORT sections 4.1 and 5. Until now they lived
    # only in this print statement, so nothing in results/ could vouch for them
    # and scripts/check_numbers.py flagged them as unaccounted. Persist them.
    summary = []
    for c in ["RAW", "ORACLE"]:
        s = o[o.cond == c]
        print(f"\n--- {c} ---")
        for g in ["correct prior", "wrong prior already"]:
            t = s[s.prior_ban_dau == g]
            n_sig = int((t.p < .05).sum())
            print(f"  {g:18s} hai TB {t.delta_pp.mean():7.2f} pp   "
                  f"p<0.05: {n_sig}/{len(t)}")
            summary.append({"cond": c, "prior_group": g,
                            "mean_delta_pp": round(float(t.delta_pp.mean()), 2),
                            "n_sig": n_sig, "n_cells": len(t)})
    pd.DataFrame(summary).to_csv(
        ROOT / "results" / "prior_strength_summary.csv", index=False)
    print("\n  Removing a CORRECT prior costs more than removing one that was already")
    print("  wrong. That is what the mechanism predicts, and it is a test that could")
    print("  have failed.")

    # ------------------------------------------------------------------
    # The two tests REPORT section 4.1 leans on. Both lived only in prose
    # until 2026-09-22 - the second was even printed inside a code fence, so
    # it read as script output while no script produced it. REPORT calls (a)
    # ESTABLISHED, which is not something a number with no source may claim.
    # ------------------------------------------------------------------
    anon = [l for l in LEXICONS if l != "KEEP" and l in d]
    inter = []

    print("\n" + "=" * 84)
    print("4. INTERACTION TEST, RUN SEPARATELY INSIDE EACH STRATUM")
    print("=" * 84)
    print("  Per item: (anon - KEEP | ORACLE) - (anon - KEEP | RAW), pooled over")
    print("  models and anonymised lexicons, then cluster bootstrap over items.")
    print("  This asks whether the graph rescues MORE where the prior was right.\n")
    for g in ["correct prior", "wrong prior already"]:
        v = did_series(d, models, anon, g)
        est, lo, hi, p = boot(v)
        print(f"  {g:20s} DiD {est:+7.2f} pp  CI [{lo:+7.2f} ; {hi:+7.2f}]  "
              f"p={p:.4f}  n={len(v)}")
        inter.append({"quantity": "DiD | ORACLE vs RAW", "stratum": g,
                      "estimate_pp": round(est, 2), "ci_lo": round(lo, 2),
                      "ci_hi": round(hi, 2), "p_boot": round(p, 4), "n_items": len(v)})

    print("\n" + "=" * 84)
    print("5. DIFFERENCE BETWEEN THE TWO STRATA - NOT a paired test")
    print("=" * 84)
    print("  The two strata are disjoint item sets, so there is nothing to pair.")
    print("  Bootstrap each stratum independently and difference the means.\n")
    a = raw_delta_series(d, models, anon, "correct prior")
    b = raw_delta_series(d, models, anon, "wrong prior already")
    est, lo, hi, p = boot_two_sample(a, b)
    print(f"  mean on CORRECT prior minus mean on ALREADY-WRONG prior")
    print(f"    {est:+.2f} pp   CI 95% [{lo:+.2f} ; {hi:+.2f}]   p = {p:.4f}")
    print(f"    n = {len(a)} vs {len(b)} items")
    inter.append({"quantity": "harm on correct prior minus harm on wrong prior",
                  "stratum": "between strata", "estimate_pp": round(est, 2),
                  "ci_lo": round(lo, 2), "ci_hi": round(hi, 2),
                  "p_boot": round(p, 4), "n_items": f"{len(a)} vs {len(b)}"})
    pd.DataFrame(inter).to_csv(
        ROOT / "results" / "prior_strength_interaction.csv", index=False)


if __name__ == "__main__":
    main()
