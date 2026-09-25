"""Generate every figure used by slides.tex.

Rule: every number drawn on a figure must declare where it came from.
  - Read it straight from results/*.csv wherever possible
  - If it exists only in REPORT.md, declare it in FROM_REPORT with its section

Run:  python scripts/make_figures.py
Writes: figures/*.pdf, plus a provenance table printed to the console
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
FIGDIR = os.path.join(HERE, "figures")

# colours kept in sync with slides.tex
C_UP = "#006E64"     # cUp   - positive direction
C_DOWN = "#AA2823"   # cDown - negative direction
C_KEY = "#234682"    # cKey  - neutral emphasis
C_GREY = "#777777"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Palatino", "DejaVu Serif"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})

# ---------------------------------------------------------------------
# Values that exist ONLY in REPORT.md, with no script producing them.
# This is the gap recorded in REPORT section 11 ("a script for section 8.1").
# ---------------------------------------------------------------------
FROM_REPORT = {
    # The six values from sections 4.3 and 4.4 HAVE been moved to
    # scripts/analyze_structure_arms.py -> results/structure_arms.csv.
    # This table is kept as a record; nothing is drawn from it any more.
}

PROVENANCE = []


def note(what, value, source):
    PROVENANCE.append((what, value, source))


def read_csv(name):
    path = os.path.join(RESULTS, name)
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, name)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    print("  wrote", os.path.relpath(out, HERE))


# ---------------------------------------------------------------------
# Figure 1. Forest plot: does the graph have to be CORRECT?
# ---------------------------------------------------------------------
def fig_forest():
    """Read straight from results/structure_arms.csv, written by analyze_structure_arms.py."""
    arms = {r["quantity"]: r for r in read_csv("structure_arms.csv")}
    want = [
        ("DiD | ORACLE",    "true graph, as arrows",    C_UP),
        ("DiD | PROSE",     "true graph, inside prose", C_UP),
        ("DiD | DR_k1",     "one edge reversed",        C_DOWN),
        ("ORACLE minus DR_k1", "difference of the two", "#222222"),
    ]
    rows = []
    for key, lab, col in want:
        if key not in arms:
            sys.exit("%r missing from structure_arms.csv - run "
                     "scripts/analyze_structure_arms.py first" % key)
        r = arms[key]
        est, lo, hi = (float(r["estimate_pp"]), float(r["ci_lo"]), float(r["ci_hi"]))
        pv = float(r["p_boot"])
        ptxt = "<0.001" if pv < 0.001 else f"{pv:.3f}".rstrip("0").rstrip(".")
        rows.append((lab, est, lo, hi, ptxt, col))
        note(lab, f"{est:+.2f} [{lo:+.2f}, {hi:+.2f}]", "results/structure_arms.csv")

    fig, ax = plt.subplots(figsize=(4.45, 1.85))
    ys = [3, 2, 1, 0]
    for y, (lab, est, lo, hi, ptxt, col) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color=col, lw=2.0,
                solid_capstyle="butt", zorder=3)
        for x in (lo, hi):
            ax.plot([x, x], [y - 0.13, y + 0.13], color=col, lw=1.1, zorder=3)
        ax.plot([est], [y], "o", color=col, ms=5.0, zorder=4)
        ax.annotate(f"{est:+.2f}", (est, y + 0.19), ha="center",
                    va="bottom", fontsize=6.5, color=col, fontweight="bold")
        ax.annotate(ptxt, (26.6, y), ha="left", va="center", fontsize=6.5,
                    color="#333333", annotation_clip=False)

    ax.axvline(0, color=C_DOWN, lw=0.8, ls=(0, (3, 2)), alpha=0.65, zorder=2)
    ax.annotate("$p$", (26.6, 3.55), ha="left", va="center", fontsize=6.5,
                style="italic", color=C_GREY, annotation_clip=False)

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xlim(-6.5, 25.5)
    ax.set_ylim(-0.6, 3.75)
    ax.set_xticks([-5, 0, 5, 10, 15, 20, 25])
    ax.tick_params(axis="x", labelsize=6.5, length=2.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("difference-in-differences (percentage points)",
                  fontsize=6.5, color=C_GREY, labelpad=2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    save(fig, "fig_forest.pdf")


# ---------------------------------------------------------------------
# Figure 2. Lifting the weak branch, or lowering the strong one
# ---------------------------------------------------------------------
def fig_lift_drag():
    """Read straight from results/structure_arms.csv."""
    arms = {r["quantity"]: r for r in read_csv("structure_arms.csv")}
    lift = float(arms["lift, anonymised branch"]["estimate_pp"])
    drag = float(arms["drag, KEEP branch"]["estimate_pp"])
    note("anonymised branch lifted", f"{lift:+.2f}", "results/structure_arms.csv")
    note("original-words branch lowered", f"{drag:+.2f}", "results/structure_arms.csv")

    fig, ax = plt.subplots(figsize=(4.45, 1.12))
    ax.barh([1], [lift], height=0.52, color=C_UP, zorder=3)
    ax.barh([0], [drag], height=0.52, color=C_DOWN, zorder=3)
    ax.annotate(f"{lift:+.2f} anonymised branch lifted", (lift + 0.5, 1),
                ha="left", va="center", fontsize=7.5, color=C_UP)
    ax.annotate(f"{drag:+.2f} original-words branch lowered", (0.5, 0),
                ha="left", va="center", fontsize=7.5, color=C_DOWN)
    ax.axvline(0, color="#333333", lw=1.1, zorder=4)

    ax.set_xlim(-6.5, 11)
    ax.set_ylim(-0.55, 1.55)
    ax.set_yticks([])
    ax.set_xticks([-5, 0, 5, 10])
    ax.tick_params(axis="x", labelsize=6.5, length=2.5)
    ax.set_xlabel("percentage points", fontsize=6.5, color=C_GREY, labelpad=2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    save(fig, "fig_liftdrag.pdf")


# ---------------------------------------------------------------------
# Figure 3. The lexical ladder - read STRAIGHT from ladder5_steps.csv
# ---------------------------------------------------------------------
def fig_ladder():
    steps = {r["step"]: r for r in read_csv("ladder5_steps.csv")}

    chain = ["KEEP -> IRRELEVANT", "IRRELEVANT -> SYMBOL", "SYMBOL -> PSEUDO"]
    labels = ["KEEP", "IRRELEVANT", "SYMBOL", "PSEUDO"]

    cum = [0.0]
    for key in chain:
        if key not in steps:
            sys.exit("step %r missing from ladder5_steps.csv" % key)
        cum.append(cum[-1] - float(steps[key]["delta_pp"]))
        note(key, f"{-float(steps[key]['delta_pp']):+.2f} pp",
             "results/ladder5_steps.csv")

    first = steps[chain[0]]
    lo, hi = -float(first["ci_hi"]), -float(first["ci_lo"])

    fig, ax = plt.subplots(figsize=(4.0, 2.0))
    xs = list(range(4))
    ax.plot(xs[:2], cum[:2], color=C_DOWN, lw=2.2, zorder=3)
    ax.plot(xs[1:], cum[1:], color="#909090", lw=2.2, zorder=3)
    ax.plot(xs, cum, "o", color="#333333", ms=4.2, zorder=4)
    ax.axhline(0, color="#bbbbbb", lw=0.7, ls=(0, (3, 2)), zorder=1)

    ax.annotate(f"{cum[1]:+.2f} pp", (1.12, cum[1] / 2 + 1.1), ha="left",
                va="center", fontsize=8, color=C_DOWN, fontweight="bold")
    ax.annotate(f"CI [{lo:+.2f}, {hi:+.2f}]", (1.12, cum[1] / 2 - 1.7),
                ha="left", va="center", fontsize=6.3, color=C_DOWN)

    rest = ", ".join(
        f"{-float(steps[k]['delta_pp']):+.2f}" for k in chain[1:])
    ax.annotate(f"next steps {rest} pp; every CI contains zero",
                (1.5, -22.6), ha="center", va="top", fontsize=6.3, color=C_GREY,
                annotation_clip=False)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7, fontweight="bold", color="#444444")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(-21.5, 2.5)
    ax.set_yticks([0, -5, -10, -15, -20])
    ax.tick_params(axis="both", labelsize=6.5, length=2.5)
    ax.set_ylabel("cumulative pp from KEEP", fontsize=6.5, color=C_GREY)
    for side in ("bottom", "right"):
        ax.spines[side].set_visible(False)
    for side in ("top", "left"):
        ax.spines[side].set_color("#999999")
    save(fig, "fig_ladder.pdf")


# ---------------------------------------------------------------------
def count_calls():
    """Count the responses actually scored, skipping quarantined data."""
    import glob
    total = {"pilot_raw": 0, "induction_raw": 0}
    for f in glob.glob(os.path.join(RESULTS, "raw", "*.csv")):
        b = os.path.basename(f)
        if "INVALID" in b or "BUGGY" in b:
            continue
        for prefix in total:
            if b.startswith(prefix):
                with open(f, encoding="utf-8", errors="ignore") as fh:
                    total[prefix] += sum(1 for _ in fh) - 1
    n = sum(total.values())
    note("scored model responses", f"{n:,}", "results/raw/*_raw*.csv (counted directly)")
    return n


def main():
    print("generating figures:")
    fig_forest()
    fig_lift_drag()
    fig_ladder()
    count_calls()

    print("\nPROVENANCE OF EVERY NUMBER ON THE FIGURES")
    print("-" * 78)
    w = max(len(a) for a, _, _ in PROVENANCE)
    for what, val, src in PROVENANCE:
        flag = "csv " if src.startswith("results/") else "PROSE"
        print(f"  [{flag}] {what:<{w}}  {val:>22}   {src}")
    print("-" * 78)
    n_prose = sum(1 for _, _, s in PROVENANCE if not s.startswith("results/"))
    print(f"  {len(PROVENANCE)} values, of which {n_prose} exist only in REPORT.md")
    if n_prose:
        print("  => the gap recorded in REPORT section 11: no script produces these")


if __name__ == "__main__":
    main()
