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


# Conditions by the run that produced them on n600. A contrast between two
# conditions of one run compares answers given at the same time; a contrast
# across runs would also carry any change in the served model between
# 2026-09-16 and 2026-09-24. Nothing in this file can see one; the test is
# scripts/check_drift.py, which found none (results/drift_check.csv).
N600_OLD = {"RAW", "PROSE", "ORACLE", "DR_k1"}
N600_NEW = {"NAMES_ONLY", "SCRAMBLE", "DR_k2", "DR_k3"}


def load_n600(lex, imap):
    """n600 with the conditions scripts/run_n600_extensions.sh added."""
    d = load("n600", lex, imap)
    extra = ROOT / "results" / f"pilot_raw_n600arms{lex}.csv"
    if extra.exists():
        e = pd.read_csv(extra).merge(imap, on="item", how="left")
        if e.id.isna().any():
            raise SystemExit(f"n600arms{lex}: some items could not be mapped to an id")
        d = pd.concat([d, e], ignore_index=True)
    return d


def matched_controls(imap, rows):
    """Section 6: NAMES_ONLY and SCRAMBLE on n600, the only sample that has them.

    The ladder, from least to most structure in the block:
        RAW         no block
        NAMES_ONLY  the block, the names, the instruction, not one arrow
        SCRAMBLE    as many arrows as ORACLE, none of them true
        DR_k1..3    the true graph with k arrows reversed
        ORACLE      the true graph
    ORACLE minus NAMES_ONLY is what the arrows are worth once the names are
    there; SCRAMBLE minus NAMES_ONLY is what WRONG arrows cost against none.
    """
    extra = ROOT / "results" / "pilot_raw_n600armsKEEP.csv"
    if not extra.exists():
        print("\n6. NAMES_ONLY / SCRAMBLE: no data (run scripts/run_n600_extensions.sh)")
        return
    print("\n6. Matched controls on n600, genuinely-causal group, paired within item")
    print("   'same run' contrasts compare answers given at the same time; 'crosses")
    print("   runs' ones also carry any drift in the served model between runs.\n")
    pairs = [("NAMES_ONLY", "RAW"), ("ORACLE", "NAMES_ONLY"), ("SCRAMBLE", "RAW"),
             ("SCRAMBLE", "NAMES_ONLY"), ("ORACLE", "SCRAMBLE"), ("DR_k1", "SCRAMBLE"),
             ("DR_k2", "NAMES_ONLY"), ("DR_k3", "DR_k2"), ("SCRAMBLE", "DR_k3"),
             ("ORACLE", "DR_k1")]
    L = {lx: load_n600(lx, imap) for lx in ("KEEP", POOL_LEXICON)}
    qs = set(L["KEEP"].query_type.unique()) - ARITH - IDENT
    for lx in ("KEEP", POOL_LEXICON):
        for a_, b_ in pairs:
            cols = []
            for m in TIER:
                x, y = cell(L[lx], m, a_, qs), cell(L[lx], m, b_, qs)
                i = x.index.intersection(y.index)
                if len(i) >= 10:
                    cols.append(pd.Series((x[i] - y[i]).values, index=i, name=m))
            if not cols:
                continue
            W = pd.concat(cols, axis=1).groupby(level=0).mean()
            est, lo, hi, p = boot(W)
            same = ({a_, b_} <= N600_OLD) or ({a_, b_} <= N600_NEW)
            run = "same run" if same else "crosses runs"
            label = f"{lx} | {a_} minus {b_} | n600"
            print(f"  {label:36s} {est:+7.2f}  [{lo:+7.2f} ; {hi:+7.2f}]  p={p:.4f}"
                  f"  n={len(W)}  {run}")
            rows.append({"quantity": label, "sample": "n600", "run": run,
                         "estimate_pp": round(est, 2), "ci_lo": round(lo, 2),
                         "ci_hi": round(hi, 2), "p_boot": round(p, 4), "n_items": len(W)})
    # The interaction for the two new blocks, same estimator as section 1.
    for arm in ("NAMES_ONLY", "SCRAMBLE"):
        cols = []
        for m in TIER:
            kr, lr = cell(L["KEEP"], m, "RAW", qs), cell(L[POOL_LEXICON], m, "RAW", qs)
            ka, la = cell(L["KEEP"], m, arm, qs), cell(L[POOL_LEXICON], m, arm, qs)
            i = kr.index.intersection(lr.index).intersection(ka.index).intersection(la.index)
            if len(i) >= 10:
                cols.append(pd.Series(((kr[i] - lr[i]) - (ka[i] - la[i])).values,
                                      index=i, name=m))
        W = pd.concat(cols, axis=1).groupby(level=0).mean()
        est, lo, hi, p = boot(W)
        label = f"DiD | {arm} | n600"
        print(f"  {label:36s} {est:+7.2f}  [{lo:+7.2f} ; {hi:+7.2f}]  p={p:.4f}"
              f"  n={len(W)}  crosses runs")
        rows.append({"quantity": label, "sample": "n600", "run": "crosses runs",
                     "estimate_pp": round(est, 2), "ci_lo": round(lo, 2),
                     "ci_hi": round(hi, 2), "p_boot": round(p, 4), "n_items": len(W)})


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

    # Per sample, not only pooled. REPORT section 4.3 carries the three arms on
    # the 174-item lexical sample alone, and until 2026-09-23 that table had no
    # script: the only trace of it in this repository was the printed sentence
    # below, describing the run rather than performing it. A number that exists
    # only inside a sentence about a number is not a source.
    print("\n4. Per sample, so the exploratory figures can be checked too")
    for tag, *_ in SAMPLES:
        for arm in usable:
            W = did_sample(tag, imaps[tag], [POOL_LEXICON], arm=arm)
            if W.empty:
                continue
            est, lo, hi, p = boot(W)
            n = len(W.index.unique())
            print(f"  {tag:9s} DiD | {arm:12s} {est:+7.2f}  "
                  f"[{lo:+7.2f} ; {hi:+7.2f}]  p={p:.4f}  n={n}")
            # The label carries the sample. make_figures.py keys this file by
            # `quantity` into a dict, so a per-sample row sharing a label with
            # the pooled row would overwrite it and the figure would quietly
            # show price400's numbers under the pooled caption.
            rows.append({"quantity": f"DiD | {arm} | {tag}", "sample": tag,
                         "estimate_pp": round(est, 2), "ci_lo": round(lo, 2),
                         "ci_hi": round(hi, 2), "p_boot": round(p, 4),
                         "n_items": n})

    # The direct contrast on the anonymised branch. The manuscript rejects the
    # symbol-rebinding reading ("the block helps only because it repeats the
    # names") by setting ORACLE +6.30 beside DR_k1 -1.26 on n600 - one estimate
    # significant, one not, which is not a test of their difference. Review
    # round 10 (finding M3). This is the test: per item, ORACLE minus DR_k1 on
    # PSEUDO, causal group, same blocks and names, one edge's direction apart.
    print("\n4b. Direct contrast on the anonymised branch: ORACLE minus DR_k1")
    per_direct = []
    for tag, *_ in SAMPLES:
        L = load(tag, POOL_LEXICON, imaps[tag])
        qs = set(L.query_type.unique()) - ARITH - IDENT
        cols = []
        for m in TIER:
            o, d_ = cell(L, m, "ORACLE", qs), cell(L, m, "DR_k1", qs)
            i = o.index.intersection(d_.index)
            if len(i) >= 10:
                cols.append(pd.Series((o[i] - d_[i]).values, index=i, name=f"{tag}|{m}"))
        if not cols:
            continue
        W = pd.concat(cols, axis=1).groupby(level=0).mean()
        per_direct.append(W)
        est, lo, hi, p = boot(W)
        print(f"  {tag:9s} {est:+7.2f}  [{lo:+7.2f} ; {hi:+7.2f}]  p={p:.4f}  n={len(W)}")
        rows.append({"quantity": f"PSEUDO branch, ORACLE minus DR_k1 | {tag}", "sample": tag,
                     "estimate_pp": round(est, 2), "ci_lo": round(lo, 2),
                     "ci_hi": round(hi, 2), "p_boot": round(p, 4), "n_items": len(W)})
    if per_direct:
        P = pd.concat(per_direct, axis=1)
        est, lo, hi, p = boot(P)
        print(f"  {'pooled':9s} {est:+7.2f}  [{lo:+7.2f} ; {hi:+7.2f}]  p={p:.4f}  n={len(P)}")
        rows.append({"quantity": "PSEUDO branch, ORACLE minus DR_k1", "sample": "pooled",
                     "estimate_pp": round(est, 2), "ci_lo": round(lo, 2),
                     "ci_hi": round(hi, 2), "p_boot": round(p, 4), "n_items": len(P)})

    # Arms that exist in ONE sample only. price400 ran the full error ladder, so
    # it alone carries DR_k2 and DR_k3. REPORT section 4.3 answers the objection
    # "one reversed edge out of 2-5 is still nearly right" with DR_k3 on this
    # sample, and that figure had no script until 2026-09-23. It is reported per
    # sample and never pooled, because pooling would silently mean "price400".
    print("\n5. Arms present in one sample only (not pooled)")
    for tag, *_ in SAMPLES:
        for arm in ("DR_k2", "DR_k3"):
            W = did_sample(tag, imaps[tag], [POOL_LEXICON], arm=arm)
            if W.empty:
                continue
            B = did_sample(tag, imaps[tag], [POOL_LEXICON], arm="ORACLE")
            cols = [c for c in B.columns if c in W.columns]
            i = B.index.intersection(W.index)
            for label, M in ((f"DiD | {arm} | {tag}", W),
                             (f"ORACLE minus {arm} | {tag}", B.loc[i, cols] - W.loc[i, cols])):
                est, lo, hi, p_ = boot(M)
                n = len(M.index.unique())
                print(f"  {label:28s} {est:+7.2f}  [{lo:+7.2f} ; {hi:+7.2f}]  "
                      f"p={p_:.4f}  n={n}")
                rows.append({"quantity": label, "sample": tag,
                             "estimate_pp": round(est, 2), "ci_lo": round(lo, 2),
                             "ci_hi": round(hi, 2), "p_boot": round(p_, 4),
                             "n_items": n})

    matched_controls(imaps["n600"], rows)

    out = ROOT / "results" / "structure_arms.csv"
    df = pd.DataFrame(rows)
    if "sample" in df.columns:
        df["sample"] = df["sample"].fillna("pooled")
    df.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")
    print("\nNOTE. The lex-sample rows above are the EXPLORATORY ones, and section")
    print("4.0 of REPORT records why they were retracted as a headline. They are")
    print("reported so the figures quoted in section 4.3 can be checked, not so")
    print("they can be quoted again.")
    print("""
DO NOT READ SECTION 2 AS EQUIVALENCE. "ORACLE minus DR_k1" is a difference of two
INTERACTIONS, and an interaction cannot see anything that moves both of its
branches by the same amount. Reversing edges lowers accuracy on the KEEP branch
and on the PSEUDO branch alike, so this row stays near zero while the accuracies
themselves move a great deal. Its interval also runs to +5.01 against a total
effect of +5.98, so it would not support an equivalence claim even on its own
terms.

For the question this row is repeatedly mistaken for - does a wrong graph help as
much as a right one - run scripts/analyze_vs_raw.py, which measures each arm
against RAW directly AND stratifies by query group. On the genuinely causal group
with real variable names it finds a correct graph worth about zero (-5.04, -0.17,
-0.08, none surviving the correction) while a single reversed edge costs -8.72,
-6.11 and -5.16. The damage is established and the benefit is not, which is an
asymmetry rather than an equivalence.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
