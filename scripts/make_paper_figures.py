"""The paper's figures, every number read from a results CSV.

    python scripts/make_paper_figures.py

  fig_prereg.pdf      the six pre-registered tests of B5 on three families
                      (confirmatory.csv, primary rows), estimate and 95% CI; for
                      H3a, an equivalence test, the 90% interval that its two
                      one-sided tests at 0.05 correspond to
  fig_harm_groups.pdf where the harm of a reversed edge lives, per model, on the
                      ate and ett items of B5 (b7_descriptive.csv, part 3)
  fig_edge_types.pdf  reversed, deleted and spurious edges by group, price400,
                      exploratory (edge_types_harm.csv, all doses)

Colours: the first three slots of the dataviz reference categorical palette
(#2a78d6, #eb6834, #1baf7a; validated light mode, CVD separation passes), each
family also carrying its own marker, so identity never rests on colour alone.
PDFs are written without creation dates, so a re-run reproduces them.

Writes figures/paper/*.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "cladder"
OUT = ROOT / "figures" / "paper"

SERIES = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^")]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#d9d8d2"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Palatino Linotype", "Palatino", "DejaVu Serif"],
    "font.size": 8, "axes.linewidth": 0.6, "axes.edgecolor": MUTED, "xtick.color": MUTED,
    "ytick.color": INK, "xtick.major.width": 0.6, "ytick.major.width": 0, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
})


def dotplot(ax, rows, series, ylabels, xlabel, band=None):
    """rows: list over y of list over series of (est, lo, hi) or None."""
    n = len(series)
    off = [(j - (n - 1) / 2) * 0.22 for j in range(n)]
    if band:                                  # (lo, hi, row): an equivalence margin on one row
        lo, hi, row = band
        ax.add_patch(plt.Rectangle((lo, row - 0.42), hi - lo, 0.84, color=GRID, alpha=0.6,
                                   lw=0, zorder=0))
    ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
    for y, per in enumerate(rows):
        for j, v in enumerate(per):
            if v is None:
                continue
            est, lo, hi = v
            c, mk = SERIES[j]
            yy = y + off[j]
            ax.plot([lo, hi], [yy, yy], color=c, lw=1.4, solid_capstyle="round", zorder=2)
            ax.plot([est], [yy], marker=mk, ms=4.2, color=c, mec="white", mew=0.6, zorder=3)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], color=SERIES[j][0], marker=SERIES[j][1], lw=1.4, ms=4.2,
                          mec="white", mew=0.6, label=s) for j, s in enumerate(series)]
    ax.legend(handles=handles, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=n, handlelength=1.6, columnspacing=1.2, fontsize=7.5)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.02,
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(f"  wrote figures/paper/{name}")


def fig_prereg():
    d = pd.read_csv(RES / "confirmatory.csv")
    d = d[d.analysis.eq("primary")]
    tests = [("H1", "H1  graph offsets anonymised names"),
             ("H2", "H2  correct graph beats one reversal"),
             ("H3a", "H3a  unrelated words vs permuted (90% CI)"),
             ("H3b", "H3b  real names vs symbols"),
             ("H4-KEEP", "H4  slope per reversal, real names"),
             ("H4-PSEUDO", "H4  slope per reversal, anonymised")]
    fams = [("gpt", "GPT-4.1 family (mean of 3)"), ("luna", "gpt-5.6-luna"), ("llama", "Llama 3.3 70B")]
    rows = []
    for t, _ in tests:
        per = []
        cols = ["estimate_pp", "ci90_lo", "ci90_hi"] if t == "H3a" else ["estimate_pp", "ci_lo", "ci_hi"]
        for f, _ in fams:
            r = d[d.family.eq(f) & d.test.eq(t)]
            per.append(None if r.empty else tuple(float(x) for x in r.iloc[0][cols]))
        rows.append(per)
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    dotplot(ax, rows, [n for _, n in fams], [n for _, n in tests],
            "estimate, percentage points (95% CI; H3a 90%)", band=(-5, 5, 2))
    save(fig, "fig_prereg.pdf")


def fig_harm_groups():
    d = pd.read_csv(RES / "b7_descriptive.csv")
    d = d[d.part.eq("3 where the harm lives")]
    groups = [("estimand kept: DR minus ORACLE", "estimand kept"),
              ("estimand changed, answer kept: DR minus ORACLE", "estimand changed, answer kept"),
              ("answer changed, path cut: DR minus ORACLE", "answer changed, every X-Y path cut"),
              ("answer changed, path kept: DR minus ORACLE", "answer changed, a path kept")]
    fams = [("gpt-4.1", "gpt-4.1"), ("luna", "gpt-5.6-luna"), ("llama", "Llama 3.3 70B")]
    rows = [[tuple(d[d.family.eq(f) & d.quantity.eq(q)].iloc[0][["estimate_pp", "ci_lo", "ci_hi"]])
             for f, _ in fams] for q, _ in groups]
    fig, ax = plt.subplots(figsize=(5.2, 2.2))
    dotplot(ax, rows, [n for _, n in fams], [n for _, n in groups],
            "reversed graph minus correct graph, percentage points (95% CI)")
    save(fig, "fig_harm_groups.pdf")


def fig_edge_types():
    d = pd.read_csv(RES / "edge_types_harm.csv")
    d = d[d.k.eq("all")]
    groups = [("estimand kept", "estimand kept"),
              ("estimand changed, answer kept", "estimand changed, answer kept"),
              ("answer changed, path cut", "answer changed, every X-Y path cut"),
              ("all draws", "all corruptions")]
    arms = [("DR", "reversed edge"), ("ED", "deleted edge"), ("FE", "spurious edge")]
    rows = []
    for g, _ in groups:
        per = []
        for a, _ in arms:
            r = d[d.arm.eq(a) & d.group.eq(g)]
            per.append(None if r.empty else tuple(r.iloc[0][["harm_pp", "ci_lo", "ci_hi"]]))
        rows.append(per)
    fig, ax = plt.subplots(figsize=(5.2, 2.2))
    dotplot(ax, rows, [n for _, n in arms], [n for _, n in groups],
            "corrupted graph minus correct graph, percentage points (95% CI)")
    save(fig, "fig_edge_types.pdf")


def main() -> int:
    fig_prereg()
    fig_harm_groups()
    fig_edge_types()
    return 0


if __name__ == "__main__":
    sys.exit(main())
