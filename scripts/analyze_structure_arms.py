"""The structure arms ORACLE, PROSE, DR_k1 - on the POOLED n=490 sample.

Why this file exists, and why it was rewritten. REPORT sections 4.3 and 4.4
reported six numbers (ORACLE +14.31, PROSE +13.91, DR_k1 +8.64, ...) that were in
no results/*.csv and that no script produced. The first version of this file
reproduced that computation - but on the exploratory sample n=86, the very sample
REPORT section 4.0 RETRACTED for winner's curse.

`pool_samples.py` recovers the item -> CLadder id map, so the three samples pool
to n=490. This version runs on that pooled sample. The numbers therefore differ
sharply from the earlier version, and that is the point rather than a side effect.

The estimator, for an arbitrary structure block C:
    DiD_i(C) = [(KEEP - PSEUDO) | RAW] - [(KEEP - PSEUDO) | C]
Split into its two halves:
    lift = acc(PSEUDO | C) - acc(PSEUDO | RAW)      the anonymised branch
    drag = acc(KEEP   | C) - acc(KEEP   | RAW)      the original-words branch
    DiD  = lift - drag

PSEUDO is the only anonymised lexicon present in all three samples. The
exploratory sample also has PERMUTE and SYMBOL, but using them would break
comparability across samples.

Run:  python scripts/analyze_structure_arms.py
Writes: results/structure_arms.csv (make_figures.py reads this file)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from analyze_querygroup import ARITH, IDENT, TIER
from pool_samples import (POOL_LEXICON, SAMPLES, SEED, boot, cell, did_sample,
                          load, verify_item_map, _pilot)

ARMS = ["ORACLE", "PROSE", "DR_k1", "NAMES_ONLY"]

# NAMES_ONLY exists only once pilot.py has been run with --names-only. If the
# data is absent the arm is skipped silently rather than crashing, so this file
# runs both before and after that experiment.
OPTIONAL = {"NAMES_ONLY"}


def present(arm):
    """Is condition `arm` available in ALL THREE samples?"""
    for tag, *_ in SAMPLES:
        for lex in ("KEEP", POOL_LEXICON):
            p = ROOT / "results" / f"pilot_raw_{tag}{lex}.csv"
            if not p.exists():
                return False
            if arm not in set(pd.read_csv(p, usecols=["cond"]).cond.unique()):
                return False
    return True


def all_item_maps():
    pilot = _pilot()
    full = pd.read_csv(ROOT / "data" / "full_v1.5_default.csv")
    return {tag: verify_item_map(tag, n, kmax, drop, full, pilot)
            for tag, n, kmax, drop in SAMPLES}


def pooled_did(imaps, arm):
    """The DiD matrix (id x cell) pooled over all three samples, for one arm."""
    per = []
    for tag, *_ in SAMPLES:
        W = did_sample(tag, imaps[tag], [POOL_LEXICON], arm=arm)
        if not W.empty:
            per.append(W.rename(columns=lambda c: f"{tag}|{c}"))
    return pd.concat(per, axis=1) if per else pd.DataFrame()


def pooled_diff(imaps, a, b):
    """Paired per-item difference DiD(a) minus DiD(b), on cells present in both."""
    A, B = pooled_did(imaps, a), pooled_did(imaps, b)
    cols = [c for c in A.columns if c in B.columns]
    i = A.index.intersection(B.index)
    if not cols or len(i) < 10:
        return pd.DataFrame()
    return A.loc[i, cols] - B.loc[i, cols]


def pooled_split(imaps, arm):
    """Split into lift (anonymised branch) and drag (KEEP branch), pooled."""
    up, dn = [], []
    for tag, *_ in SAMPLES:
        imap = imaps[tag]
        keep = load(tag, "KEEP", imap)
        lex = load(tag, POOL_LEXICON, imap)
        qs = set(keep.query_type.unique()) - ARITH - IDENT
        for m in TIER:
            kr, ka = cell(keep, m, "RAW", qs), cell(keep, m, arm, qs)
            j = kr.index.intersection(ka.index)
            if len(j) >= 10:
                dn.append(pd.Series((ka[j] - kr[j]).values, index=j,
                                    name=f"{tag}|{m}"))
            lr, la = cell(lex, m, "RAW", qs), cell(lex, m, arm, qs)
            i = lr.index.intersection(la.index)
            if len(i) >= 10:
                up.append(pd.Series((la[i] - lr[i]).values, index=i,
                                    name=f"{tag}|{m}"))

    def gather(cols):
        if not cols:
            return pd.DataFrame()
        return pd.concat(cols, axis=1).groupby(level=0).mean()

    return gather(up), gather(dn)


def report(label, W, rows):
    if W.empty:
        print(f"  {label:28s} not enough data")
        return
    est, lo, hi, p = boot(W)
    n = len(W.index.unique())
    print(f"  {label:28s} {est:+7.2f}  [{lo:+7.2f} ; {hi:+7.2f}]  p={p:.4f}  n={n}")
    rows.append({"quantity": label, "estimate_pp": round(est, 2),
                 "ci_lo": round(lo, 2), "ci_hi": round(hi, 2),
                 "p_boot": round(p, 4), "n_items": n})


def main():
    imaps = all_item_maps()

    print("=" * 78)
    print("STRUCTURE ARMS, pooled n=490, genuinely-causal query group")
    print("=" * 78)
    print("  cluster bootstrap over items, 4000 draws, seed", SEED)
    print(f"  anonymised lexicon: {POOL_LEXICON} (the only one in all three samples)\n")

    usable = [a for a in ARMS if a not in OPTIONAL or present(a)]
    missing = [a for a in ARMS if a not in usable]
    if missing:
        print(f"  no data yet for: {', '.join(missing)} "
              f"(run pilot.py --names-only to produce it)\n")

    rows = []
    print("1. DiD per structure block  (against RAW)")
    for arm in usable:
        report(f"DiD | {arm}", pooled_did(imaps, arm), rows)

    print("\n2. Paired difference between blocks")
    report("ORACLE minus DR_k1", pooled_diff(imaps, "ORACLE", "DR_k1"), rows)
    report("ORACLE minus PROSE", pooled_diff(imaps, "ORACLE", "PROSE"), rows)
    if "NAMES_ONLY" in usable:
        # The decisive test. If this difference does not separate from zero, what
        # helps the model is not the CONTENT of the structure but merely having a
        # set of symbols to anchor on.
        report("ORACLE minus NAMES_ONLY",
               pooled_diff(imaps, "ORACLE", "NAMES_ONLY"), rows)

    print("\n3. Lift / drag split for the ORACLE block")
    U, D = pooled_split(imaps, "ORACLE")
    report("lift, anonymised branch", U, rows)
    report("drag, KEEP branch", D, rows)

    if not U.empty and not D.empty:
        up = 100 * np.nanmean(U.values)
        dn = 100 * np.nanmean(D.values)
        tot = up - dn
        print(f"\n  additivity check: lift {up:+.2f} minus drag {dn:+.2f} = {tot:+.2f} pp")
        if abs(dn) > 1e-9 and tot > 0:
            print(f"  share of the effect that is HARM to the KEEP branch: "
                  f"{100 * abs(min(dn, 0.0)) / tot:.0f}%")

    out = ROOT / "results" / "structure_arms.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")
    print("\nNOTE. The earlier version of this file ran on the exploratory sample")
    print("n=86 and gave ORACLE +14.35 / DR_k1 +8.69. These pooled n=490 numbers")
    print("are markedly lower. The exploratory sample is the RETRACTED one - see")
    print("REPORT section 4.0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
