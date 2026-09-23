"""Re-run the analysis and prove every results CSV comes back byte-identical.

Why this file exists. Numbers in this project have moved between write-ups more
than once, and each time the first question was "did the code change, or is it
not reproducible?". Twice the answer was that the ANALYSIS changed - a query-group
filter was added, an item set was widened - and once it was that a document
carried an older run's numbers. None of it was non-determinism. But answering
that question took a manual snapshot-and-diff each time, so it is worth having as
a command.

Every analysis script here is seeded (SEED = 20260907) and bootstraps with
numpy's default_rng, so a rerun must reproduce its output exactly. If this script
reports a difference, one of three things is true and all of them matter:

  1. someone edited an analysis script and the CSVs on disk are now stale,
  2. a script picked up an unseeded source of randomness,
  3. an input under results/ changed underneath the analysis.

Note what this does NOT check: that the numbers are CORRECT. It checks only that
they are stable. scripts/check_numbers.py covers the other half, by asking whether
every quantity quoted in prose traces back to a cell in results/.

Run:  python scripts/verify_determinism.py           # the fast scripts
      python scripts/verify_determinism.py --all     # everything, several minutes
      python scripts/verify_determinism.py analyze_vs_raw classify_perturbations
Exit code 1 if anything differs.
"""

from __future__ import annotations

import filecmp
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# ORDER MATTERS, and getting it wrong is what this script caught on its first
# run. `analyze_types.py` defaults to reading results/induction_raw.csv, which
# `induction.py` WRITES. Run them alphabetically and analyze_types reads a stale
# file, so the comparison reports a spurious difference. Any list here must put
# a writer before its readers.
FAST = [
    "analyze_ladder5", "analyze_pilot", "analyze_instruction",
    "analyze_prior_strength", "compare_price_lexicon", "verify_labels",
    "verify_counterfactual", "classify_perturbations", "analyze_dose",
    "analyze_chains", "analyze_errortypes_lexical",
    "induction",                 # writes induction_raw.csv
    "analyze_lexical", "analyze_types",   # both read it
]
# pilot.py is excluded on purpose: it spends API credit and is not an analysis.

# --all used to mean FAST plus a second hand-kept list (SLOW), a copy of the pipeline order
# that had fallen seven scripts behind (moderators, falsification, price_paired,
# raw_leak, explanations, provenance, figures) - so "everything" was not
# everything. Review round 10 (finding P1). --all now reads the canonical ORDER
# from check_pipeline_order.py, minus the three gates, which do not write
# results/. One list, so it cannot drift again.
def _canonical_order() -> list[str]:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cpo", str(Path(__file__).resolve().parent / "check_pipeline_order.py"))
    cpo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cpo)
    gates = {"check_numbers", "check_pipeline_order", "verify_determinism"}
    return [s for s in cpo.ORDER if s not in gates]


def snapshot(dst: Path) -> list[str]:
    names = sorted(p.name for p in RESULTS.glob("*.csv"))
    for n in names:
        shutil.copy2(RESULTS / n, dst / n)
    return names


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    scripts = args or (_canonical_order() if "--all" in sys.argv else FAST)

    print("=" * 72)
    print("DETERMINISM CHECK - rerun, then diff every results CSV byte for byte")
    print("=" * 72)
    print(f"\n  {len(scripts)} script(s): {', '.join(scripts)}\n")

    with tempfile.TemporaryDirectory(prefix="noisycausal-det-") as tmp:
        tmpd = Path(tmp)
        before = snapshot(tmpd)
        print(f"  snapshot: {len(before)} CSV files\n")

        for s in scripts:
            path = ROOT / "scripts" / f"{s}.py"
            if not path.exists():
                print(f"  {s:28s} MISSING")
                return 1
            r = subprocess.run([sys.executable, str(path)],
                               capture_output=True, text=True, cwd=str(ROOT))
            status = "ran" if r.returncode == 0 else f"EXIT {r.returncode}"
            print(f"  {s:28s} {status}")
            if r.returncode != 0:
                print("      " + (r.stderr.strip().splitlines() or ["(no stderr)"])[-1][:100])

        after = sorted(p.name for p in RESULTS.glob("*.csv"))
        changed, new = [], [n for n in after if n not in before]
        for n in before:
            if n not in after:
                changed.append((n, "DELETED"))
            elif not filecmp.cmp(RESULTS / n, tmpd / n, shallow=False):
                changed.append((n, "DIFFERS"))

    print(f"\n  {len(before)} files compared, {len(changed)} differ, {len(new)} new")
    if new:
        print("\n  NEW (not an error - a script wrote a file that did not exist):")
        for n in new:
            print(f"    {n}")
    if changed:
        print("\n  NOT REPRODUCIBLE:")
        for n, how in changed:
            print(f"    {n:44s} {how}")
        print("""
  A rerun changed the output. Before trusting any number in the write-up, find
  out which of these it was:
    - an analysis script was edited and results/ was never regenerated,
    - a script uses randomness that is not seeded,
    - an input file changed under the analysis.""")
        return 1

    print("\n  Every CSV came back identical. The analysis is reproducible.")
    print("  This says nothing about whether the numbers are RIGHT - run")
    print("  scripts/check_numbers.py for the other half.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
