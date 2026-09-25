"""Turn results/raw/pilot_raw.csv into the numbers that decide whether the study runs.

    python scripts/analyze_pilot.py

Answers four questions, in the order they can kill the project:

  Q1  Does an explicit graph help at all on CLadder? If Delta_struct <= 0 there is
      no benefit to burn off and the break-even question is empty.
  Q2  How much does one reversed edge cost? That slope sets the sample size.
  Q3  Where does the curve cross the no-graph floor, and is that inside reach?
  Q4  Does the effect track model tier, as the weak-model hypothesis predicts?
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from stats import mcnemar_exact_p, bootstrap_break_even

TIER = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]
TAG = ""


def paired(df, model):
    return df[df.model == model].pivot_table(
        index="item", columns="cond", values="correct", aggfunc="first")


def mcnemar(s, a, b):
    if a not in s or b not in s:
        return None
    x, y = s[a].dropna(), s[b].dropna()
    i = x.index.intersection(y.index)
    nb = int(((x[i] == 1) & (y[i] == 0)).sum())
    nc = int(((x[i] == 0) & (y[i] == 1)).sum())
    return nb, nc, mcnemar_exact_p(nb, nc), len(i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="pilot_raw.csv",
                    help="which results/*.csv to analyse")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    args = ap.parse_args()
    global TAG
    TAG = args.tag
    df = pd.read_csv(ROOT / "results" / "raw" / args.file)
    models = [m for m in TIER if m in set(df.model)] or sorted(df.model.unique())
    kcols = sorted([c for c in df.cond.unique() if c.startswith("DR_k")])
    kmax = len(kcols)

    print("=" * 78)
    print("PILOT - results")
    print("=" * 78)
    print(f"items={df.item.nunique()}  models={models}  "
          f"conditions={sorted(df.cond.unique())}")
    print(f"parse rate tong: {100*df.parsed.mean():.2f}%   "
          f"API errors: {(df.error.astype(str) != '').sum() - (df.error.isna()).sum()}")

    # ---- accuracy table ---------------------------------------------------
    print("\n" + "-" * 78)
    print("Q1/Q2  DO CHINH XAC THEO DIEU KIEN (%)")
    print("-" * 78)
    order = ["PROSE", "RAW", "ORACLE"] + kcols
    acc = (df.groupby(["model", "cond"]).correct.mean().unstack() * 100)
    acc = acc.reindex(columns=order).reindex(index=models).round(2)
    print(acc.to_string())
    acc.to_csv(ROOT / "results" / f"pilot_accuracy{TAG}.csv")

    # ---- deltas -----------------------------------------------------------
    print("\n" + "-" * 78)
    print("Q1/Q2  THE DECISIVE DIFFERENCES")
    print("-" * 78)
    rows = []
    for m in models:
        r = acc.loc[m]
        d_struct = r["ORACLE"] - r["RAW"]
        slope = (r["ORACLE"] - r[kcols[-1]]) / kmax if kmax else np.nan
        rows.append({
            "model": m,
            "RAW": r["RAW"], "ORACLE": r["ORACLE"],
            "Delta_struct": round(d_struct, 2),
            "strip_cost(PROSE-RAW)": round(r["PROSE"] - r["RAW"], 2),
            "doc_DR_pp_moi_canh": round(slope, 2),
        })
    dl = pd.DataFrame(rows)
    print(dl.to_string(index=False))
    dl.to_csv(ROOT / "results" / f"pilot_deltas{TAG}.csv", index=False)

    # ---- break-even -------------------------------------------------------
    print("\n" + "-" * 78)
    print("Q3  BREAK-EVEN POINT k* (bootstrap 4000, resampled at item level)")
    print("-" * 78)
    be_rows = []
    for m in models:
        s = paired(df, m)
        if "RAW" not in s:
            continue
        cbk, ok = {}, True
        for k, c in enumerate([("ORACLE")] + kcols):
            if c not in s:
                ok = False
                break
            cbk[k] = s[c].fillna(0).values
        if not ok:
            continue
        b = bootstrap_break_even(cbk, s["RAW"].fillna(0).values, reps=4000, seed=11)
        reach = "within reach" if b["k_star"] <= kmax else "OUT of reach"
        be_rows.append({"model": m, "k*": round(b["k_star"], 2),
                        "CI_lo": round(b["lo"], 2), "CI_hi": round(b["hi"], 2),
                        "rong": round(b["hi"] - b["lo"], 2),
                        "k_da_do_toi": kmax, "verdict": reach})
    if be_rows:
        bd = pd.DataFrame(be_rows)
        print(bd.to_string(index=False))
        bd.to_csv(ROOT / "results" / f"pilot_break_even{TAG}.csv", index=False)

    # ---- significance -----------------------------------------------------
    print("\n" + "-" * 78)
    print("McNEMAR EXACT  (b = only cond 1 correct, c = only cond 2 correct)")
    print("-" * 78)
    pairs = [("ORACLE", "RAW")] + [(k, "RAW") for k in kcols]
    if kcols:
        pairs.append((kcols[0], "ORACLE"))

    sig = []
    for m in models:
        s = paired(df, m)
        for a, b in pairs:
            r = mcnemar(s, a, b)
            if r:
                nb, nc, p, n = r
                sig.append({"model": m, "n_compared": f"{a} vs {b}", "n": n,
                            "b": nb, "c": nc, "p": round(p, 4),
                            "meaning": "*" if p < .05 else ""})
    sg = pd.DataFrame(sig)
    print(sg.to_string(index=False))
    sg.to_csv(ROOT / "results" / f"pilot_mcnemar{TAG}.csv", index=False)

    # ---- tier trend -------------------------------------------------------
    print("\n" + "-" * 78)
    print("Q4  DOES THE EFFECT SCALE WITH MODEL TIER?")
    print("-" * 78)
    print(dl[["model", "Delta_struct", "doc_DR_pp_moi_canh"]].to_string(index=False))
    if len(dl) >= 2:
        d0, d1 = dl.Delta_struct.iloc[0], dl.Delta_struct.iloc[-1]
        print(f"\n  Delta_struct: {dl.model.iloc[0]} = {d0:+.2f} pp  ->  "
              f"{dl.model.iloc[-1]} = {d1:+.2f} pp")
        print("  Hypothesis H4 predicts the weaker model gains less, or negatively.")

    print("\nDa ghi 5 file CSV vao results/")


if __name__ == "__main__":
    main()
