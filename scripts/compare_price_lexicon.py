"""Is the cost of a wrong edge a property of the model, or of the domain?

    python scripts/compare_price_lexicon.py

The full error ladder (DR / ED / FE at k = 1, 2, 3) was run twice at n = 400 on
the same items: once with CLadder's own variable names, once with those names
replaced by pseudowords. That separates two things the break-even point mixes
together.

  price   pp of accuracy lost per erroneous edge - the slope of the
          degradation line
  budget  ORACLE - RAW, how much benefit the correct graph confers in the
          first place - the height that slope has to eat through

k* is solved from the fitted line as (intercept - RAW) / price, NOT as
budget / price - the fitted intercept sits 1.8 to 2.2 pp below the measured
ORACLE point, so the two formulas differ by up to 2.7 edges. See
analyze_types.py:175 for the expression actually used.

Round 5 of the review panel rejected the reading that the budget doubles when
the lexical anchor is removed: all three paired difference CIs contain 0 and
gpt-4.1-nano moves the other way. REPORT.md section 8.3 records that
retraction. This script therefore reports the price contrast, which stands,
and prints the budget numbers without a conclusion attached.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

TIER = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]
TYPES = ["ED", "FE", "DR"]


def load(tag):
    p = ROOT / "results" / "cladder" / f"types_price_price400{tag}.csv"
    if not p.exists():
        raise SystemExit(f"thieu {p.name}: chay analyze_types.py --tag _price400{tag}")
    return pd.read_csv(p)


def main():
    K, P = load("KEEP"), load("PSEUDO")
    models = [m for m in TIER if m in set(K.model)]

    print("=" * 84)
    print("1. PRICE PER WRONG EDGE - does it depend on the lexical regime?")
    print("=" * 84)
    rows = []
    for m in models:
        for t in TYPES:
            k = K[(K.model == m) & (K.error_type.str.startswith(t))].iloc[0]
            p = P[(P.model == m) & (P.error_type.str.startswith(t))].iloc[0]
            # Two CIs that overlap do not prove equality, but a price that moved
            # would have to show up as CIs pulling apart. None of them do.
            overlap = not (k.price_hi < p.price_lo or p.price_hi < k.price_lo)
            rows.append({
                "model": m, "error_type": t,
                "gia_KEEP": k.price_pp_per_edge,
                "KEEP_CI": f"[{k.price_lo:.2f}, {k.price_hi:.2f}]",
                "gia_PSEUDO": p.price_pp_per_edge,
                "PSEUDO_CI": f"[{p.price_lo:.2f}, {p.price_hi:.2f}]",
                "CI_overlap": "yes" if overlap else "no",
            })
    g = pd.DataFrame(rows)
    print(g.to_string(index=False))
    n_ov = (g.CI_overlap == "yes").sum()
    print(f"\n  {n_ov}/{len(g)} CI pairs overlap -> no pair separates.")
    g.to_csv(ROOT / "results" / "cladder" / "price_by_lexicon.csv", index=False)

    print("\n" + "=" * 84)
    print("2. BUDGET AND BREAK-EVEN - this is where things change")
    print("=" * 84)
    out = []
    for m in models:
        k = K[(K.model == m) & (K.error_type.str.startswith("DR"))].iloc[0]
        p = P[(P.model == m) & (P.error_type.str.startswith("DR"))].iloc[0]
        fmt = lambda r: ("not established" if pd.isna(r.breakeven_k)
                         else f"{r.breakeven_k:.2f} [{r.breakeven_lo:.2f}, {r.breakeven_hi:.2f}]")
        out.append({"model": m,
                    "ngan_sach_KEEP": k.budget_pp,
                    "ngan_sach_PSEUDO": p.budget_pp,
                    "kstar_KEEP_DR": fmt(k),
                    "kstar_PSEUDO_DR": fmt(p),
                    # Carried in the file itself, not only in the printed text:
                    # anyone reading this CSV without the script must see that
                    # the contrast between the two budgets was retracted.
                    "warning": "budget difference NOT ESTABLISHED, all three CIs "
                                "contain 0, nano moves the other way - see REPORT.md "
                                "section 8.3"})
    o = pd.DataFrame(out)
    print(o.to_string(index=False))
    o.to_csv(ROOT / "results" / "cladder" / "breakeven_by_lexicon.csv", index=False)

    print("""
  HOW TO READ THIS, per REPORT.md section 8.3.

  THE PRICE DIFFERENCE IS NOT COMPUTED HERE. The overlap column above is not
  evidence: two overlapping CIs do not show two quantities are equal, they show
  the comparison was never made. Until 2026-09-23 this block printed a paired
  difference of -0.11 [-1.62 ; 1.42] pp as though it were a result, while no
  function in this repository resampled that difference. It is now computed by
  scripts/analyze_price_paired.py, which writes results/cladder/price_paired_difference.csv.

  What that file says: 0 of 9 cells separate from zero, so the price cannot be
  told apart between lexical regimes. But the equivalence bound is only tight
  for DR on the strong models (about 1.7 pp). Across all nine cells the data
  rule out only a difference larger than about 8.86 pp, because FE fits three
  points on the smallest item counts. Quote the bound from the WEAKEST cell.

  NOT ESTABLISHED: the claim that "the budget doubles once the lexical anchor is
  removed" was rejected in review round 5. All three CIs on the budget difference
  contain 0, and gpt-4.1-nano moves the OTHER way - its budget FALLS. The table
  above is printed for reference and carries no conclusion.

  The break-even point k* is therefore also unestablished, because its numerator
  is not solid. The pair 0.74 and 1.93 must not be read as a regularity.

  Run scripts/analyze_budget_paired.py for a CI on the budget difference and the
  sample size needed to settle it.""")


if __name__ == "__main__":
    main()
