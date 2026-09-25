"""Is Delta_struct the graph, or is it the sentence telling the model to use it?

    python scripts/analyze_instruction.py

The ORACLE prompt adds two things to RAW at the same time:

    "The causal structure of this world is: <edges>"      the graph CONTENT
    "Use this causal structure when reasoning."           an INSTRUCTION

Delta_struct = ORACLE - RAW has been charging the whole gap to the first. This
script separates them with the RAW_INSTR condition, which carries an instruction
to reason causally and no graph at all:

    RAW_INSTR - RAW      what the instruction alone is worth
    ORACLE - RAW_INSTR   what the graph content adds on top of it

The decomposition alone is not the point. The project's headline is an
INTERACTION - that a correct graph erases the cost of anonymising the variable
names - and the honest threat to it is that an instruction with no graph in it
might erase that cost just as well. If it does, the finding is about telling a
model to think causally, not about giving it a causal structure, and it lands
much closer to Caliper's scaffold result (arXiv:2606.04915 section 4.9), where
scaffolding narrowed the gap mainly by lowering P0 rather than recovering P1.

So the test that matters is run on the same footing as the headline: the same
difference-in-differences, on the same causal-query subset, with RAW_INSTR
substituted for one arm.

    A  [(KEEP - PSEUDO) | RAW]       - [(KEEP - PSEUDO) | ORACLE]      headline
    B  [(KEEP - PSEUDO) | RAW]       - [(KEEP - PSEUDO) | RAW_INSTR]   instruction
    C  [(KEEP - PSEUDO) | RAW_INSTR] - [(KEEP - PSEUDO) | ORACLE]      graph content

A is what REPORT.md section 4.0 reports. B is how much of A an instruction with
no graph can reproduce. C is what survives for the graph once the instruction is
already in the baseline, and C is the number the headline has to rest on.

Bootstrap resamples ITEMS, not cells, for the reason given in
analyze_querygroup.py: the cells share one KEEP baseline and all 174 items.

A here will not equal the +14.35 pp in REPORT.md section 4.0, and should not.
That figure pools three anonymised lexicons over nine cells; this one uses the
single PSEUDO branch, because RAW_INSTR was only paid for on KEEP and PSEUDO.
The two are the same contrast on different amounts of data, so A is the
like-for-like baseline for B and C and not a restatement of the headline.
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

from stats import boot_cells, boot_interval, boot_p, cluster_boot

TIER = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]
ARITH = {"marginal", "correlation"}
IDENT = {"backadj"}
GROUPS = {"rung1_arith": ARITH, "identify": IDENT, "causal": None}


def load(tag):
    p = ROOT / "results" / "raw" / f"pilot_raw_instr{tag}.csv"
    if not p.exists():
        raise SystemExit(
            f"thieu {p.name}. Chay:\n"
            f"  python scripts/pilot.py --n 200 --models gpt-4.1-nano,gpt-4.1-mini,"
            f"gpt-4.1 --kmax 1 --types DR --drop-nonsense --lexicon {tag} "
            f"--with-instr --tag _instr{tag}")
    return pd.read_csv(p)


def subset(d, qs):
    if qs is None:
        return d[~d.query_type.isin(ARITH | IDENT)]
    return d[d.query_type.isin(qs)]


def cell(d, model, cond):
    s = d[(d.model == model) & (d.cond == cond) & (d.parsed == 1)]
    return s.set_index("item").correct


def did(K, P, model, lo_cond, hi_cond, qs):
    """Per-item [(KEEP-PSEUDO)|lo] - [(KEEP-PSEUDO)|hi], on items in all four."""
    k_lo, k_hi = cell(subset(K, qs), model, lo_cond), cell(subset(K, qs), model, hi_cond)
    p_lo, p_hi = cell(subset(P, qs), model, lo_cond), cell(subset(P, qs), model, hi_cond)
    idx = k_lo.index.intersection(k_hi.index).intersection(
        p_lo.index).intersection(p_hi.index)
    if not len(idx):
        return pd.Series(dtype=float)
    return pd.Series((k_lo[idx].values - p_lo[idx].values)
                     - (k_hi[idx].values - p_hi[idx].values), index=idx)


def boot(series_by_model, seed, n=4000):
    """Cluster bootstrap over items, averaged across models."""
    items = sorted(set().union(*[set(s.index) for s in series_by_model.values()]))
    arr = {m: s.reindex(items) for m, s in series_by_model.items()}
    M = np.vstack([arr[m].values for m in arr])
    return boot_cells(M, seed, n, axis=1, return_draws=True)   # convention B, src/stats.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()
    K, P = load("KEEP"), load("PSEUDO")
    W = 94

    print("=" * W)
    print("1. SPLITTING Delta_struct: how much is the INSTRUCTION, how much the GRAPH")
    print("=" * W)
    print("  On the KEEP branch, genuinely-causal query group only.\n")
    rows = []
    for m in TIER:
        s = subset(K, None)
        acc = {c: 100 * cell(s, m, c).mean() for c in
               ("RAW", "RAW_INSTR", "ORACLE")}
        rows.append({"model": m,
                     "RAW": round(acc["RAW"], 2),
                     "RAW_INSTR": round(acc["RAW_INSTR"], 2),
                     "ORACLE": round(acc["ORACLE"], 2),
                     "instruction": round(acc["RAW_INSTR"] - acc["RAW"], 2),
                     "graph_content": round(acc["ORACLE"] - acc["RAW_INSTR"], 2),
                     "tong": round(acc["ORACLE"] - acc["RAW"], 2)})
    dec = pd.DataFrame(rows)
    print(dec.to_string(index=False))
    dec.to_csv(ROOT / "results" / "instruction_decomposition.csv", index=False)

    print("\n" + "=" * W)
    print("2. THE DECISIVE TEST: can the instruction alone, with no graph, close")
    print("   the lexical gap?")
    print("=" * W)
    print("  Same interaction test, same query group, one arm swapped.")
    print(f"  Bootstrap boc lai theo ITEM, {a.boot} lan.\n")

    tests = [("A. tieu de   RAW  vs ORACLE", "RAW", "ORACLE"),
             ("B. cau lenh  RAW  vs RAW_INSTR", "RAW", "RAW_INSTR"),
             ("C. do thi    RAW_INSTR vs ORACLE", "RAW_INSTR", "ORACLE")]
    rows = []
    for gname, qs in GROUPS.items():
        for label, lo, hi in tests:
            by = {m: did(K, P, m, lo, hi, qs) for m in TIER}
            by = {m: s for m, s in by.items() if len(s)}
            if not by:
                continue
            est, bs = boot(by, a.seed, a.boot)
            clo, chi = np.percentile(bs, [2.5, 97.5])
            # Clamped: with an estimate sitting exactly on zero both tails can
            # round above 0.5 and the doubled value exceeds 1, which is not a
            # p-value and reads as a bug to anyone checking the table.
            p = boot_p(bs)
            rows.append({"group": gname, "test": label.split(".")[1].strip(),
                         "ma": label[0], "DiD_pp": round(est, 2),
                         "ci_lo": round(clo, 2), "ci_hi": round(chi, 2),
                         "p": round(p, 4),
                         "established": "yes" if clo > 0 or chi < 0 else "no"})
    res = pd.DataFrame(rows)
    for g in GROUPS:
        sub = res[res.group == g]
        if not len(sub):
            continue
        print(f"  -- {g} --")
        print(sub[["ma", "test", "DiD_pp", "ci_lo", "ci_hi", "p", "established"]]
              .to_string(index=False))
        print()
    res.to_csv(ROOT / "results" / "instruction_interaction.csv", index=False)

    print("=" * W)
    print("3. DOC KET QUA")
    print("=" * W)
    c = res[(res.group == "causal")].set_index("ma")
    if {"A", "B", "C"} <= set(c.index):
        A, B, C = c.loc["A"], c.loc["B"], c.loc["C"]
        share = 100 * B.DiD_pp / A.DiD_pp if A.DiD_pp else float("nan")
        print(f"  Tuong tac tieu de (A):            {A.DiD_pp:+.2f} pp  "
              f"CI [{A.ci_lo:+.2f}; {A.ci_hi:+.2f}]  p={A.p}")
        print(f"  Instruction with NO graph (B):    {B.DiD_pp:+.2f} pp  "
              f"CI [{B.ci_lo:+.2f}; {B.ci_hi:+.2f}]  p={B.p}")
        print(f"  Graph content on top of it (C):   {C.DiD_pp:+.2f} pp  "
              f"CI [{C.ci_lo:+.2f}; {C.ci_hi:+.2f}]  p={C.p}")
        print(f"\n  The instruction alone reproduces {share:.0f}% of the headline interaction.")
        if C.established == "yes":
            print("  C STILL HOLDS: the graph content does something of its own, beyond")
            print("  what the instruction does. The headline claim survives, but must be")
            print("  restated in terms of C rather than A.")
        else:
            print("  C DOES NOT HOLD: once the instruction is in the baseline, adding the")
            print("  graph content contributes nothing measurable. The headline claim must")
            print("  become a statement about ASKING the model to reason causally, not")
            print("  about SUPPLYING it with structure. This is a negative finding, and it")
            print("  has to go into the report before any draft is submitted anywhere.")
    print("\n  Da ghi: results/instruction_decomposition.csv,")
    print("          results/instruction_interaction.csv")


if __name__ == "__main__":
    main()
