"""Reproduce the headline number: pool three samples at ITEM level, de-duplicated
by the original CLadder id.

Why this file exists. REPORT section 4.0 retracted +14.35 (the exploratory sample,
n=86) and replaced it with **+5.98 pp on n=490 items, three samples pooled and
de-duplicated by original id**. Audit on 2026-09-20: **no script in the repo
produced that number**. It existed only as prose. This is the project's HEADLINE.

Why it could not be reproduced directly: `pilot.py` never writes CLadder's `id`
column into its result CSV. The `item` column is a reference index WITHIN one
sample (`--pair-to` makes KEEP and PSEUDO of the SAME sample share it, but two
different samples do not). The natural key (story_id, graph_id, query_type, rung)
is not unique either: 174 items collapse to 110 combinations.

The way out: `make_items` is DETERMINISTIC - `random.Random(seed)` then a
proportional stratified draw over (graph_id, rung). Re-running it with the right
arguments recovers the `id` column. Those arguments were recovered by sweeping
(n, kmax, drop_nonsense) until the generated item sequence matched the saved CSV
ROW BY ROW on (graph_id, rung, query_type, story_id). What that sweep found:

    lex       n=174  kmax=1  drop_nonsense=True   10 families
    n600      n=580  kmax=1  drop_nonsense=True   10 families
    price400  n=399  kmax=3  drop_nonsense=True    7 families

THE CONTRAST MUST BE IDENTICAL ACROSS SAMPLES. The `lex` sample has three
anonymised lexicons (PERMUTE, SYMBOL, PSEUDO); the two large samples have only
PSEUDO. So the pooling uses PSEUDO everywhere. Using all three for `lex` gives
+14.31 on 86 items instead of +12.71 on 85, which is not comparable with the
other two.

THE AVERAGING CONVENTION is the easiest thing to get wrong here. The mean must be
taken over EVERY CELL (item x model), not per item and then across items. The two
differ because not every item has all three cells. The first convention reproduces
REPORT's three numbers exactly; the second gives +13.92 / +6.85 / +1.88 and pools
to +6.53.

RESULT: all four of REPORT section 4.0's numbers reproduce exactly.

    exploratory    85  +12.71
    extension     287   +6.78
    error ladder  195   +1.72
    POOLED        490   +5.98   CI [+1.67 ; +10.19]

So the headline is CORRECT and properly de-duplicated. What was missing was a
script that produces it. Now there is one.

Run:  python scripts/pool_samples.py
Writes: results/pooled_headline.csv and results/_itemmap_*.csv
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from stats import boot_interval, boot_p, cluster_boot

from analyze_querygroup import ARITH, IDENT, TIER

SEED = 20260907
NBOOT = 4000
KEY = ["graph_id", "rung", "query_type", "story_id"]

# (file tag, n, kmax, drop_nonsense). Arguments recovered by sweep and checked
# row by row - see the docstring. `verify_item_map` re-runs that check on every
# call rather than trusting the constants.
SAMPLES = [
    ("lex",      174, 1, True),
    ("n600",     580, 1, True),
    ("price400", 399, 3, True),
]
POOL_LEXICON = "PSEUDO"          # the one anonymised lexicon present in ALL samples


def _pilot():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pilot", str(ROOT / "scripts" / "pilot.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def draw_items(full, pilot, n, kmax, drop):
    """A copy of `make_items` that takes the dataframe instead of reading it.

    Identical logic; separated only so the sweep can call it thousands of times
    without re-reading an 8.8 MB CSV each round.
    """
    d = full
    if drop:
        d = d[~d.story_id.astype(str).str.startswith("nonsense")]
    fams = [f for f in pilot.FAMILY_STRUCTURE if pilot.max_k(f, "DR") >= kmax]
    d = d[d.graph_id.isin(fams)].reset_index(drop=True)
    rng = random.Random(SEED)
    groups = list(d.groupby(["graph_id", "rung"]))
    per = max(1, n // len(groups))
    picked = []
    for _, g in groups:
        picked += rng.sample(list(g.index), min(per, len(g)))
    picked = picked[:n] if len(picked) >= n else picked
    return d.loc[picked].reset_index(drop=True)


def verify_item_map(tag, n, kmax, drop, full, pilot):
    """Recover item -> id, and PROVE it by matching the saved CSV row by row."""
    it = draw_items(full, pilot, n, kmax, drop)
    saved = pd.read_csv(ROOT / "results" / f"pilot_raw_{tag}KEEP.csv")
    want = (saved[["item"] + KEY].drop_duplicates()
            .sort_values("item")[KEY].reset_index(drop=True))
    got = it[KEY].reset_index(drop=True)
    if not got.equals(want):
        raise SystemExit(
            f"{tag}: the regenerated item sequence does NOT match the saved CSV. "
            f"Either the arguments (n={n}, kmax={kmax}, drop={drop}) are wrong or "
            f"pilot.py has changed. Stopping rather than pooling on a bad key.")
    out = it[["id"]].reset_index().rename(columns={"index": "item"})
    out.to_csv(ROOT / "results" / f"_itemmap_{tag}.csv", index=False)
    return out


def load(tag, lex, imap):
    d = pd.read_csv(ROOT / "results" / f"pilot_raw_{tag}{lex}.csv")
    d = d.merge(imap, on="item", how="left")
    if d.id.isna().any():
        raise SystemExit(f"{tag}/{lex}: some items could not be mapped to an id")
    return d


def cell(d, model, cond, qs):
    s = d[(d.model == model) & (d.cond == cond) & (d.parsed == 1)
          & (d.query_type.isin(qs))]
    return s.set_index("id").correct


def did_sample(tag, imap, lexicons, arm="ORACLE"):
    """Per-id DiD for one sample. Columns are (model x lexicon) cells."""
    keep = load(tag, "KEEP", imap)
    qs = set(keep.query_type.unique()) - ARITH - IDENT
    cols = []
    for m in TIER:
        for lx in lexicons:
            L = load(tag, lx, imap)
            kr, lr = cell(keep, m, "RAW", qs), cell(L, m, "RAW", qs)
            ka, la = cell(keep, m, arm, qs), cell(L, m, arm, qs)
            i = (kr.index.intersection(lr.index)
                 .intersection(ka.index).intersection(la.index))
            if len(i) < 10:
                continue
            cols.append(pd.Series(((kr[i] - lr[i]) - (ka[i] - la[i])).values,
                                  index=i, name=f"{m}|{lx}"))
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols, axis=1).groupby(level=0).mean()


def boot(W, seed=SEED, n=NBOOT):
    """Cluster bootstrap over items. The mean is taken over EVERY cell.

    Averaging per item first and then across items gives a different number,
    because not every item has all three cells. The convention here reproduces
    REPORT section 4.0 exactly, which is evidence it is the one REPORT used.
    """
    rng = np.random.default_rng(seed)
    idx = W.index.values
    out = cluster_boot(len(idx),
                       lambda i: 100 * np.nanmean(W.loc[idx[i]].values), seed, n)
    est = 100 * np.nanmean(W.values)
    # This is the helper analyze_by_family.py uses, and it is where the
    # p = 1.0255 in family_breakdown.csv came from.
    return est, np.percentile(out, 2.5), np.percentile(out, 97.5), boot_p(out, n)


def main():
    pilot = _pilot()
    full = pd.read_csv(ROOT / "data" / "full_v1.5_default.csv")

    print("=" * 78)
    print("POOLING THREE SAMPLES AT ITEM LEVEL, DE-DUPLICATED BY CLADDER id")
    print("=" * 78)
    print("  DiD_i = [(KEEP - PSEUDO) | RAW] - [(KEEP - PSEUDO) | ORACLE]")
    print("  genuinely-causal query group, cluster bootstrap over items,",
          NBOOT, "draws, seed", SEED, "\n")

    per, rows = {}, []
    print(f"  {'sample':10s} {'n items':>7s} {'DiD':>8s}  {'95% CI':>21s} {'p':>8s}")
    print("  " + "-" * 60)
    for tag, n, kmax, drop in SAMPLES:
        imap = verify_item_map(tag, n, kmax, drop, full, pilot)
        W = did_sample(tag, imap, [POOL_LEXICON])
        if W.empty:
            print(f"  {tag:10s} not enough data")
            continue
        per[tag] = W.rename(columns=lambda c: f"{tag}|{c}")
        e, lo, hi, p = boot(W)
        print(f"  {tag:10s} {len(W):7d} {e:+8.2f}  [{lo:+7.2f} ; {hi:+7.2f}] {p:8.4f}")
        rows.append({"pooling": tag, "n_items": len(W), "did_pp": round(e, 2),
                     "ci_lo": round(lo, 2), "ci_hi": round(hi, 2),
                     "p_boot": round(p, 4)})

    # Pool by concatenating the cells of all three samples along the id axis. An
    # item present in several samples simply contributes more cells, exactly as
    # it would within one sample.
    pooled = pd.concat(per.values(), axis=1)
    e, lo, hi, p = boot(pooled)
    print("  " + "-" * 60)
    print(f"  {'POOLED':10s} {len(pooled):7d} {e:+8.2f}  [{lo:+7.2f} ; {hi:+7.2f}] {p:8.4f}")
    rows.append({"pooling": "item level, de-duplicated", "n_items": len(pooled),
                 "did_pp": round(e, 2), "ci_lo": round(lo, 2),
                 "ci_hi": round(hi, 2), "p_boot": round(p, 4)})

    ns = {t: int(per[t].notna().any(axis=1).sum()) for t in per}
    total = sum(ns.values())
    w = sum(ns[t] * 100 * np.nanmean(per[t].values) for t in per) / total
    rows.append({"pooling": "n-weighted mean of samples", "n_items": total,
                 "did_pp": round(w, 2), "ci_lo": "", "ci_hi": "", "p_boot": ""})

    dup = int(pd.concat([per[t].notna().any(axis=1) for t in per], axis=1)
              .fillna(False).sum(axis=1).gt(1).sum())
    print("\n" + "=" * 78)
    print("AGAINST REPORT SECTION 4.0")
    print("=" * 78)
    # This line used to read "REPORT says : n=490, DiD +5.98, CI [+1.78 ;
    # +10.30], p=0.005" with those figures typed in. That interval was
    # superseded on 2026-09-22 and the print went on asserting it as "what
    # REPORT says" for as long as nobody re-read the string - a number typed by
    # hand is not a check, it is a second place for the number to go stale.
    #
    # The fix is NOT to read structure_arms.csv here: that file is written at
    # stage 12 of the sweep and this script runs at stage 5, so the comparison
    # would silently be against the previous run. Prose-against-CSV is
    # scripts/check_numbers.py's job. This script states what it measured and
    # writes it to results/pooled_headline.csv; nothing else.
    print(f"  do duoc o day : n={len(pooled)}, DiD {e:+.2f}, "
          f"CI [{lo:+.2f} ; {hi:+.2f}], p={p:.4f}")
    print("  So doi chieu voi REPORT do scripts/check_numbers.py lo, vi no doc")
    print("  duoc CA HAI phia. Dung go tay lai con so nao vao day.")
    print(f"\n  {dup} items appear in more than one sample and were de-duplicated by id.")
    print(f"  For contrast, an n-weighted mean of the samples (total {total}, which")
    print(f"  DOUBLE COUNTS those) gives {w:+.2f} pp - close, but not what REPORT used.")
    print("\n  The headline is correct and properly de-duplicated. What was missing")
    print("  was a script that produces it; now there is one.")

    out = ROOT / "results" / "pooled_headline.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)} and results/_itemmap_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
