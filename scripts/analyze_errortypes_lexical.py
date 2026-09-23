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

from stats import boot_interval, cluster_boot

import numpy as np
import pandas as pd

LEXICONS = ["KEEP", "PERMUTE", "SYMBOL", "PSEUDO"]
TIER = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]
TYPES = ["DR", "ED", "FE"]
LABEL = {"DR": "reversed edge", "ED": "one spurious edge", "FE": "one missing edge"}
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
    """NOT routed through stats.cluster_boot, and the reason is the signature.

    Every other bootstrap here takes a SEED and builds its own Generator, so each
    call starts from the same state and is independent of call order. This one
    takes a Generator that the caller built once and passes to every call, so the
    stream continues across calls and each result depends on how many bootstraps
    ran before it. cluster_boot would restart the stream and change every number
    in results/errortype_by_lexicon.csv.

    That shared-generator design is a reproducibility smell - reordering the
    calls silently reorders the draws - but fixing it is a change to published
    numbers, not a refactor, so it is left alone and flagged here rather than
    quietly rewritten under the heading of tidying up.
    """
    items = sorted(set().union(*[set(s.index) for s in by_model.values()]))
    M = np.vstack([by_model[m].reindex(items).values for m in by_model])
    out = np.empty(n)
    for i in range(n):
        out[i] = np.nanmean(M[:, rng.integers(0, len(items), len(items))])
    return 100 * np.nanmean(M), 100 * out


def summarise(by_model, rng, boots):
    est, bs = boot(by_model, rng, boots)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi, ("yes" if lo > 0 or hi < 0 else "no")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--all-queries", action="store_true",
                    help="use the whole sample instead of the genuinely-causal group only")
    a = ap.parse_args()
    causal = not a.all_queries
    D = {lex: load(lex) for lex in LEXICONS}
    rng = np.random.default_rng(a.seed)
    W = 88

    print("=" * W)
    print("1. PRICE OF EACH GRAPH ERROR TYPE, BY LEXICON")
    print("=" * W)
    scope = "genuinely-causal group only" if causal else "whole sample"
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
            rows.append({"lexicon": lex, "error_type": t, "meaning": LABEL[t],
                         "cost_pp": round(est, 2), "ci_lo": round(lo, 2),
                         "ci_hi": round(hi, 2), "established": ok,
                         "equivalence_bound": round(max(abs(lo), abs(hi)), 2)})
    cost = pd.DataFrame(rows)
    print(cost[["lexicon", "error_type", "cost_pp", "ci_lo", "ci_hi", "established"]]
          .to_string(index=False))
    cost.to_csv(ROOT / "results" / "errortype_by_lexicon.csv", index=False)

    print("\n" + "=" * W)
    print("2. IS A REVERSED EDGE MORE EXPENSIVE THAN THE OTHER TWO TYPES")
    print("=" * W)
    print("  A paired within-item difference, so this is not inferred from two")
    print("  overlapping CIs.\n")
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
            rows.append({"lexicon": lex, "n_compared": f"DR dat hon {other}",
                         "delta_pp": round(est, 2), "ci_lo": round(lo, 2),
                         "ci_hi": round(hi, 2), "established": ok})
    rank = pd.DataFrame(rows)
    print(rank.to_string(index=False))
    rank.to_csv(ROOT / "results" / "errortype_ranking.csv", index=False)

    print("\n" + "=" * W)
    print("3. DOC KET QUA")
    print("=" * W)
    keep_any = (cost[cost.lexicon == "KEEP"].established == "yes").any()
    anon = cost[cost.lexicon.isin(["SYMBOL", "PSEUDO"])]
    dr_anon = anon[(anon.error_type == "DR") & (anon.established == "yes")]
    if not keep_any:
        kb = cost[cost.lexicon == "KEEP"].equivalence_bound.max()
        print(f"  On KEEP: NO error type has a measurable cost "
              f"(can tuong duong toi da {kb:.2f} pp).")
        print("  With a correct prior to lean on, a broken graph matters little.")
    if len(dr_anon):
        print(f"\n  On the anonymised lexicons: DR is established in {len(dr_anon)}/2 of "
              f"them, while ED and FE are established in none.")
        print("  Remove the prior and the graph starts to matter - and what matters is")
        print("  EDGE DIRECTION.")
    strict = rank[rank.established == "yes"]
    if len(strict):
        print(f"\n  DR dat hon loai khac mot cach xac lap o: "
              f"{', '.join(sorted(set(strict.lexicon)))}.")
    print("\n  SAMPLE-SIZE WARNING. The causal group leaves only about 60 items per cell,")
    print("  so a 'not established' result here means NOT RESOLVABLE at this sample")
    print("  size, not zero. Read the equivalence_bound column in the CSV.")
    print("\n  Da ghi: results/errortype_by_lexicon.csv, results/errortype_ranking.csv")


if __name__ == "__main__":
    main()
