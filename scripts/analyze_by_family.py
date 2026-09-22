"""Break the structure result down BY GRAPH FAMILY, on the POOLED n=490 sample.

Why this file exists, and why it was rewritten. It originally ran on the
exploratory sample, n=86 - the very sample REPORT section 4.0 RETRACTED for
winner's curse: its DiD is +12.71 against a pooled estimate of +5.98. With 7 to
10 items per family the confidence intervals were +-20 to +-40 pp, so that table
was a lead, not a result.

`pool_samples.py` recovers the item -> CLadder id map, so the three samples can
be pooled. On the pooled sample each family has roughly 30 to 70 items instead of
7 to 10. That is the test that turns the lead into a result, or kills it cleanly.

THE CONTRAST USES PSEUDO in all three samples, because the two large samples have
only that lexicon. The exploratory sample also has PERMUTE and SYMBOL, but using
all three would not be comparable across samples.

SCOPE WARNING. The `price400` sample ran with kmax=3 and therefore covers only 7
families; the three two-edge families (chain, collision, fork) come from `lex` and
`n600` alone. Items per family are uneven, so the n column has to be read with
every cell.

Run:  python scripts/analyze_by_family.py
Writes: results/family_breakdown.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from pool_samples import (POOL_LEXICON, SAMPLES, SEED, boot, did_sample,
                          verify_item_map, _pilot)

ALPHA = 0.05


def benjamini_hochberg(p: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Step-up procedure. Returns a boolean mask of the rejected hypotheses.

    This table sweeps ten families and then re-groups the same items by edge
    count and by node count. Reporting per-cell p values from a sweep that wide
    without correcting invites exactly the reading the sweep cannot support - and
    the four-edge group was in fact FOUND by looking at this table, so it is the
    textbook case for a correction rather than an optional extra.
    """
    m = len(p)
    order = np.argsort(p)
    thresh = (np.arange(1, m + 1) / m) * alpha
    below = p[order] <= thresh
    keep = np.zeros(m, dtype=bool)
    if below.any():
        keep[order[: np.where(below)[0].max() + 1]] = True
    return keep

# The one family where CLadder miscomputes ALL FOUR causal quantities, not just
# P(Y=1). See scripts/audit_cladder_arithmetic.py.
SUSPECT = {"arrowhead"}


def structure():
    """(n_nodes, n_edges) per family, read straight off CLadder's parameter keys."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "vg", str(ROOT / "scripts" / "verify_groundtruth.py"))
    vg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vg)
    meta = json.loads((ROOT / "data" / "cladder-meta.json").read_text(encoding="utf-8"))
    out = {}
    for m in meta:
        if m["graph_id"] in out:
            continue
        s = vg.SCM(m["params"])
        out[m["graph_id"]] = (len(s.nodes), sum(len(s.parents[n]) for n in s.nodes))
    return out


def pooled_matrix():
    """The DiD matrix (id x cell) pooled over all three samples, plus id -> family."""
    pilot = _pilot()
    full = pd.read_csv(ROOT / "data" / "full_v1.5_default.csv")
    per = {}
    for tag, n, kmax, drop in SAMPLES:
        imap = verify_item_map(tag, n, kmax, drop, full, pilot)
        W = did_sample(tag, imap, [POOL_LEXICON])
        if not W.empty:
            per[tag] = W.rename(columns=lambda c: f"{tag}|{c}")
    if not per:
        raise SystemExit("could not build the DiD matrix")
    return pd.concat(per.values(), axis=1), full.set_index("id").graph_id


def report(label, W, extra, rows):
    n = len(W.index.unique())
    if W.empty or n < 5:
        print(f"  {label:30s} only {n} items, skipped")
        return
    est, lo, hi, p = boot(W)
    flag = "  <- CLadder labels broken" if extra.get("suspect_labels") else ""
    print(f"  {label:30s} {est:+7.2f}  [{lo:+7.2f} ; {hi:+7.2f}]  p={p:.4f}  n={n}{flag}")
    rows.append({"slice": label, **extra, "estimate_pp": round(est, 2),
                 "ci_lo": round(lo, 2), "ci_hi": round(hi, 2),
                 "p_boot": round(p, 4), "n_items": n})


def main():
    G, fam_of = pooled_matrix()
    st = structure()
    fam = fam_of.reindex(G.index)
    if fam.isna().any():
        raise SystemExit("some ids have no graph_id")

    print("=" * 88)
    print("DiD BY GRAPH STRUCTURE - three samples pooled, de-duplicated by id")
    print("=" * 88)
    print("  DiD_i = [(KEEP - PSEUDO) | RAW] - [(KEEP - PSEUDO) | ORACLE]")
    print("  genuinely-causal query group, cluster bootstrap over items, 4000 draws,"
          " seed", SEED)
    print(f"  {len(G)} items over {G.shape[1]} cells (sample x model)\n")

    rows = []
    print("1. BY FAMILY  (ordered by edge count, then name)")
    print(f"  {'family':30s} {'DiD':>7s}  {'95% CI':>20s}")
    for g in sorted(st, key=lambda x: (st[x][1], st[x][0], x)):
        nn, ne = st[g]
        idx = fam.index[fam == g]
        report(f"{g} ({nn} nodes, {ne} edges)", G.loc[G.index.isin(idx)],
               {"section": 1, "family": g, "n_nodes": nn, "n_edges": ne,
                "arm": "ORACLE", "suspect_labels": g in SUSPECT}, rows)

    print("\n2. GROUPED BY EDGE COUNT")
    for ne in sorted({v[1] for v in st.values()}):
        gs = [g for g in st if st[g][1] == ne]
        idx = fam.index[fam.isin(gs)]
        report(f"{ne} edges: {', '.join(sorted(gs))}"[:30], G.loc[G.index.isin(idx)],
               {"section": 2, "family": "+".join(sorted(gs)), "n_nodes": "",
                "n_edges": ne, "arm": "ORACLE",
                "suspect_labels": bool(set(gs) & SUSPECT)}, rows)

    print("\n3. GROUPED BY NODE COUNT")
    for nn in sorted({v[0] for v in st.values()}):
        gs = [g for g in st if st[g][0] == nn]
        idx = fam.index[fam.isin(gs)]
        report(f"{nn} nodes ({len(gs)} families)", G.loc[G.index.isin(idx)],
               {"section": 3, "family": "+".join(sorted(gs)), "n_nodes": nn,
                "n_edges": "", "arm": "ORACLE",
                "suspect_labels": bool(set(gs) & SUSPECT)}, rows)

    print("\n4. SENSITIVITY: drop arrowhead from the four-node group")
    gs = [g for g in st if g not in SUSPECT and st[g][0] == 4]
    idx = fam.index[fam.isin(gs)]
    report("4 nodes, without arrowhead", G.loc[G.index.isin(idx)],
           {"section": 4, "family": "+".join(sorted(gs)), "n_nodes": 4,
            "n_edges": "", "arm": "ORACLE", "suspect_labels": False}, rows)

    R = pd.DataFrame(rows)

    # Sections 1 to 3 are one sweep over the same items, cut three ways, and the
    # four-edge finding was discovered inside it. They are corrected together.
    # Section 4 is a sensitivity re-analysis of a section-3 slice with a known
    # bad family removed, not an independent hypothesis, so it stays out of the
    # family and is marked as such.
    fam = R.section.isin([1, 2, 3]).values
    keep = benjamini_hochberg(R.loc[fam, "p_boot"].values)
    # object dtype, because section 4 is deliberately outside the family and is
    # recorded as "not tested" rather than as a failure
    col, j = [], 0
    for in_family in fam:
        if in_family:
            col.append(bool(keep[j]))
            j += 1
        else:
            col.append("not in correction family")
    R["survives_BH"] = col

    print("\n" + "=" * 88)
    print(f"5. AFTER BENJAMINI-HOCHBERG at {ALPHA} over the {int(fam.sum())} tests"
          " in sections 1 to 3")
    print("=" * 88 + "\n")
    surv = R[fam & (R.survives_BH == True)]
    if surv.empty:
        print("  NOTHING in this table survives the correction.")
    else:
        for _, r in surv.sort_values("p_boot").iterrows():
            print(f"  {r['slice']:32s} {r.estimate_pp:+7.2f}"
                  f"  [{r.ci_lo:+7.2f} ; {r.ci_hi:+7.2f}]  p={r.p_boot:.4f}"
                  f"  n={r.n_items}")
    dropped = R[fam & (R.survives_BH == False) & (R.p_boot < ALPHA)]
    if not dropped.empty:
        print("\n  Nominally significant but DOES NOT survive - do not report as"
              " a finding:")
        for _, r in dropped.sort_values("p_boot").iterrows():
            print(f"  {r['slice']:32s} {r.estimate_pp:+7.2f}  p={r.p_boot:.4f}"
                  f"  n={r.n_items}")

    print("""
  Read this table as the sweep it is. With ten families cut three ways, some cell
  will look significant whatever the truth, and the four-edge group was found by
  reading this very table rather than predicted before it. Only the rows listed
  above as surviving may be stated as results; the rest are leads for a larger
  sample. The per-family rows in particular carry 29 to 69 items each, which is
  not enough to rank families against one another.""")

    out = ROOT / "results" / "family_breakdown.csv"
    R.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")
    print("\nHOW TO READ THIS. These are the POOLED n=490 numbers, not the n=86")
    print("exploratory sample earlier versions of this file used. The intervals are")
    print("markedly tighter, and section 5 now corrects the 16 slices together, so")
    print("the per-cell p values above must be read through that correction: exactly")
    print("ONE slice survives it. Five further cells are nominally significant and")
    print("are listed as such precisely so they are not quoted as results. Cells that")
    print("include arrowhead must also be read with the CLadder label caveat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
