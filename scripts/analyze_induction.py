"""Analyze stored graph-induction responses without making API calls.

The collection step lives in ``scripts/induction.py`` and is intentionally not
part of the offline analysis pipeline.  This script consumes its stored raw
CSV and regenerates the two derived induction tables.

    python scripts/analyze_induction.py [--tag SUFFIX]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from stats import mcnemar_exact_p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="suffix used by the stored raw files")
    a = ap.parse_args()

    raw = ROOT / "results" / "cladder" / "raw" / f"induction_raw{a.tag}.csv"
    pilot_raw = ROOT / "results" / "cladder" / "raw" / f"pilot_raw{a.tag}.csv"
    if not raw.exists():
        raise SystemExit(
            f"missing stored input: {raw.relative_to(ROOT)}\n"
            "Collect it explicitly with scripts/induction.py; the offline analysis "
            "pipeline never calls an API."
        )
    if not pilot_raw.exists():
        raise SystemExit(f"missing stored input: {pilot_raw.relative_to(ROOT)}")

    df = pd.read_csv(raw)
    pilot = pd.read_csv(pilot_raw)

    print("\n" + "=" * 78)
    print("CHAT LUONG DO THI TU DUNG (doi chieu Table 3 cua NoisyCausal)")
    print("=" * 78)
    q = (df.groupby("model")
           .agg(n=("f1", "size"), parse_graph=("parsed_graph", "mean"),
                precision=("precision", "mean"), recall=("recall", "mean"),
                f1=("f1", "mean"), exact=("exact_match", "mean"),
                canh_sai=("n_wrong_edges", "mean"),
                dao_chieu=("n_reversed", "mean"),
                thua=("n_spurious", "mean"), thieu=("n_missing", "mean"))
           .round(3))
    print(q.to_string())
    q.to_csv(ROOT / "results" / "cladder" / f"induction_quality{a.tag}.csv")

    print("\n" + "=" * 78)
    print("INDUCED ROI VAO DAU TREN DUONG CONG?")
    print("=" * 78)
    acc_ind = df.groupby("model").correct.mean() * 100
    acc_pil = pilot.groupby(["model", "cond"]).correct.mean().unstack() * 100
    cols = ["RAW", "ORACLE"] + sorted(c for c in acc_pil.columns if c.startswith("DR_k"))
    comp = acc_pil[cols].copy()
    comp["INDUCED"] = acc_ind
    print(comp.round(2).to_string())
    comp.round(2).to_csv(ROOT / "results" / "cladder" / f"induction_vs_curve{a.tag}.csv")

    print("\n  INDUCED against the reference points:")
    for model in comp.index:
        if model not in acc_ind:
            continue
        row = comp.loc[model]
        print(f"    {model:14s} INDUCED={row['INDUCED']:.2f}  "
              f"vs RAW {row['INDUCED']-row['RAW']:+.2f}  "
              f"vs ORACLE {row['INDUCED']-row['ORACLE']:+.2f}  "
              f"vs DR_k1 {row['INDUCED']-row['DR_k1']:+.2f}")

    print("\n" + "=" * 78)
    print("McNEMAR: INDUCED co khac RAW khong? (paired)")
    print("=" * 78)
    for model in df.model.unique():
        pi = pilot[(pilot.model == model) & (pilot.cond == "RAW")].set_index("item").correct
        ii = df[df.model == model].set_index("item").correct
        idx = pi.index.intersection(ii.index)
        b = int(((ii[idx] == 1) & (pi[idx] == 0)).sum())
        c = int(((ii[idx] == 0) & (pi[idx] == 1)).sum())
        p = mcnemar_exact_p(b, c)
        print(f"  {model:14s} n={len(idx)}  b={b:3d} c={c:3d}  p={p:.4f}"
              f"{'  *' if p < .05 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
