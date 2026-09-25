"""Does the asymmetry of REPORT section 4.2 hold on the confirmatory sample?

    python scripts/analyze_b5_vs_raw.py

What is being re-checked. On the three exploratory samples, against RAW (no
structure block), on the causal query group:

    KEEP    ORACLE  about 0          DR_k1  -5 to -9, significant in 2 of 3
    PSEUDO  ORACLE  +1 to +8         DR_k1  about 0

i.e. with familiar names a correct graph adds nothing while one reversed edge
costs several points, and only once the names are anonymised does the correct
graph help. REPORT section 4.2 turns that into advice for pipelines that feed a
graph to a model.

The confirmatory run (prereg/CONFIRMATORY.md) sent RAW, ORACLE and DR_k1..3
under both lexicons on 484 fresh causal items, so the same contrasts can be
read off it for free. They are NOT among the six pre-registered tests, so what
this file prints is exploratory: a second look at an exploratory pattern on
data that did not produce it, not a confirmation. Benjamini-Hochberg is
applied across the rows of this table, and every p is reported as a range over
the project's five bootstrap seeds.

The estimator is analyze_vs_raw's, imported rather than copied: per item,
cond minus RAW averaged over the models that answered both, parsed answers
only, then the mean over items. The confirmatory run sent no backadj or
rung-1 arithmetic items, so the "graph is the answer" rows of section 4.2
cannot be re-checked here, and it sent no ED arm, so neither can the advice to
drop an edge rather than guess its direction.

Writes: results/cladder/b5_vs_raw.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from stats import boot_items
from analyze_querygroup import ARITH, IDENT
from analyze_vs_raw import benjamini_hochberg, paired

RESULTS = ROOT / "results" / "cladder"
NBOOT = 4000
SEEDS = [20260907, 1, 2, 3, 4]
# (lexicon, condition, reference): the rows of section 4.2 plus the DR doses
# and the familiar-name twin of H2.
ROWS = [(lex, cond, "RAW") for lex in ("KEEP", "PSEUDO")
        for cond in ("ORACLE", "DR_k1", "DR_k2", "DR_k3")]
ROWS += [("KEEP", "ORACLE", "DR_k1")]


def paired_vs(d: pd.DataFrame, cond: str, ref: str) -> pd.Series:
    """cond minus ref per item, models averaged; analyze_vs_raw.paired when ref is RAW."""
    if ref == "RAW":
        return paired(d, cond, group="causal")
    d = d[~d.query_type.isin(ARITH | IDENT)]
    cols = []
    for m in sorted(d.model.unique()):
        r = d[(d.model == m) & (d.cond == ref)].set_index("item").correct
        c = d[(d.model == m) & (d.cond == cond)].set_index("item").correct
        i = r.index.intersection(c.index)
        if len(i) >= 5:
            cols.append(pd.Series((c[i] - r[i]).values, index=i, name=m))
    return pd.concat(cols, axis=1).mean(axis=1).dropna()


def exploratory(lex: str, cond: str) -> str:
    """The same row on the three exploratory samples, from vs_raw.csv."""
    v = pd.read_csv(RESULTS / "vs_raw.csv")
    v = v[(v.query_group == "causal") & (v.lexicon == lex) & (v.cond == cond)]
    return "; ".join(f"{r.sample} {r.delta_pp:+.2f}" for r in v.itertuples())


def main() -> int:
    data = {}
    for lex in ("KEEP", "PSEUDO"):
        d = pd.read_csv(RESULTS / "raw" / f"pilot_raw_conf{lex}.csv")
        data[lex] = d[d.parsed == 1]

    rows = []
    for lex, cond, ref in ROWS:
        v = paired_vs(data[lex], cond, ref)
        per = [boot_items(v.values, s, NBOOT) for s in SEEDS]
        est, lo, hi, p = per[0]
        ps = [x[3] for x in per]
        rows.append(dict(sample="conf", lexicon=lex, cond=cond, reference=ref,
                         delta_pp=round(est, 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                         p_boot=round(p, 4), p_min_seeds=round(min(ps), 4),
                         p_max_seeds=round(max(ps), 4), n_items=len(v),
                         exploratory_samples=exploratory(lex, cond) if ref == "RAW" else ""))
    R = pd.DataFrame(rows)
    R["survives_BH"] = benjamini_hochberg(R.p_boot.values)
    R.to_csv(RESULTS / "b5_vs_raw.csv", index=False)

    print("=" * 96)
    print("THE SECTION 4.2 ASYMMETRY ON THE CONFIRMATORY SAMPLE (exploratory re-check)")
    print("=" * 96)
    for r in R.itertuples():
        star = "*" if r.survives_BH else " "
        print(f"  {r.lexicon:6s} {r.cond:7s} - {r.reference:6s} {r.delta_pp:+6.2f} "
              f"[{r.ci_lo:+6.2f} ; {r.ci_hi:+6.2f}]{star} p {r.p_min_seeds:.4f}-"
              f"{r.p_max_seeds:.4f}  n={r.n_items}   {r.exploratory_samples}")
    print("\n  * survives Benjamini-Hochberg across the rows of this table.")

    k = R.set_index(["lexicon", "cond", "reference"])
    ko, kd = k.loc[("KEEP", "ORACLE", "RAW")], k.loc[("KEEP", "DR_k1", "RAW")]
    print(f"\n  Familiar names: the correct graph {ko.delta_pp:+.2f} "
          f"[{ko.ci_lo:+.2f} ; {ko.ci_hi:+.2f}], one reversed edge {kd.delta_pp:+.2f} "
          f"[{kd.ci_lo:+.2f} ; {kd.ci_hi:+.2f}] against RAW.")
    print("  Wrote results/cladder/b5_vs_raw.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
