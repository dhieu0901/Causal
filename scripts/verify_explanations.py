"""Errors 3 and 4 in CLadder v1.5: the step-by-step explanations of `marginal` items.

    python scripts/verify_explanations.py

The manuscript lists five errors in CLadder v1.5. Errors 1, 2 and 5 each have a
script; errors 3 and 4 were described only in docs/CLADDER_DATA_ERRORS.md,
which is not part of the released repository. Review round 10 (finding P1)
flagged that a reader could not reproduce them. This script does.

Every `marginal` item carries an explanation of the form

    P(Y | X=1)*P(X=1) + P(Y | X=0)*P(X=0)      <- the formula: a PLUS
    ...
    0.77*0.76 - 0.23*0.26 = 0.64               <- the arithmetic: a MINUS
    0.64 > 0                                   <- the conclusion: against 0

  error 3  the arithmetic line uses a minus where its own formula has a plus
  error 4  the conclusion compares with 0, which a probability always exceeds,
           instead of with 0.5

Positive control. If the LABELS are right and only the explanations are wrong,
then comparing the published value with 0.5, and flipping for questions that
ask "less likely", must reproduce the label. It is run here so that errors 3
and 4 are not mistaken for label errors.

Writes results/explanation_errors.csv.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FORMULA_PLUS = re.compile(r"P\(Y \| X=1\)\*P\(X=1\) \+ P\(Y \| X=0\)\*P\(X=0\)")
ARITH = re.compile(r"^\s*[\d.]+\*[\d.]+\s*([+-])\s*[\d.]+\*[\d.]+\s*=\s*(-?[\d.]+)\s*$")
CONCL = re.compile(r"^\s*(-?[\d.]+)\s*([<>])\s*([\d.]+)\s*$")


def main() -> int:
    d = pd.read_csv(ROOT / "data" / "full_v1.5_default.csv", low_memory=False)
    m = d[d.query_type == "marginal"].copy()
    rows = []
    for r in m.itertuples():
        lines = str(r.reasoning).splitlines()
        plus = any(FORMULA_PLUS.search(x) for x in lines)
        ar = [ARITH.match(x) for x in lines]
        ar = next((a for a in ar if a), None)
        co = CONCL.match(lines[-1]) if lines else None
        prompt = str(r.prompt).lower()
        polarity = ("less" if "less likely" in prompt
                    else "more" if "more likely" in prompt else "unknown")
        value = float(ar.group(2)) if ar else float("nan")
        # Label under the 0.5 threshold, flipped for "less likely" questions.
        pred = (value > 0.5) if polarity == "more" else (value < 0.5)
        rows.append({
            "id": r.id, "label": r.label, "polarity": polarity,
            "formula_has_plus": plus,
            "arith_sign": ar.group(1) if ar else None,
            "published_value": value,
            "conclusion_threshold": float(co.group(3)) if co else None,
            "conclusion_reads": ("yes" if co and co.group(2) == ">" else
                                 "no" if co else None),
            "label_from_threshold_0_5": "yes" if pred else "no",
        })
    t = pd.DataFrame(rows)
    n = len(t)
    e3 = int((t.formula_has_plus & (t.arith_sign == "-")).sum())
    e4 = int((t.conclusion_threshold == 0).sum())
    opposite = int(((t.conclusion_threshold == 0) & (t.conclusion_reads != t.label)).sum())
    # The explanation prints the value to two decimals, so an item showing
    # exactly 0.50 cannot be decided against a 0.5 threshold from the text.
    # All 23 control "misses" on the first run were such items; count them
    # apart instead of calling them mismatches.
    undecidable = t.published_value == 0.5
    dec = t[~undecidable]
    reproduced = int((dec.label_from_threshold_0_5 == dec.label).sum())
    # Is the minus only a display error? Recompute both readings from the
    # printed factors and see which one the printed result matches.
    fac = re.compile(r"^\s*([\d.]+)\*([\d.]+)\s*[+-]\s*([\d.]+)\*([\d.]+)\s*=\s*(-?[\d.]+)\s*$")
    plus_ok = minus_ok = 0
    for r in m.itertuples():
        for x in str(r.reasoning).splitlines():
            g = fac.match(x)
            if g:
                a, b, c, e, res = map(float, g.groups())
                # Tolerance from rounding, not tuned: each printed factor is
                # rounded to 2 dp (+-0.005), so each product can be off by
                # about 0.01, the sum by 0.02, and the printed result adds
                # 0.005 of its own rounding.
                plus_ok += abs(a * b + c * e - res) <= 0.025
                minus_ok += abs(a * b - c * e - res) <= 0.025
                break
    print(f"marginal explanations: {n}")
    print(f"  printed result equals the PLUS computation : {plus_ok}/{n}")
    print(f"  printed result equals the MINUS computation: {minus_ok}/{n}")
    print(f"  error 3  formula has +, arithmetic uses -      : {e3}/{n}")
    print(f"  error 4  conclusion compares with 0            : {e4}/{n}")
    print(f"           of which read literally contradict the label: {opposite}")
    print(f"  control  0.5 threshold with question polarity reproduces the label: "
          f"{reproduced}/{len(dec)} decidable ({int(undecidable.sum())} show exactly 0.50)")
    print(f"           polarity: {t.polarity.value_counts().to_dict()}")
    summary = pd.DataFrame([
        {"check": "error 3: minus in arithmetic, plus in formula", "count": e3, "n": n},
        {"check": "error 4: conclusion compares with 0", "count": e4, "n": n},
        {"check": "error 4: conclusion contradicts own label", "count": opposite, "n": n},
        {"check": "control: 0.5 threshold with polarity reproduces label", "count": reproduced, "n": len(dec)},
        {"check": "control: value printed as exactly 0.50, undecidable", "count": int(undecidable.sum()), "n": n},
        {"check": "printed result equals the plus computation", "count": int(plus_ok), "n": n},
        {"check": "printed result equals the minus computation", "count": int(minus_ok), "n": n},
    ])
    summary.to_csv(ROOT / "results" / "explanation_errors.csv", index=False)
    print("\n  wrote results/explanation_errors.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
