"""Cross-check every effect size quoted in prose against the results CSVs.

Why this file exists. Numbers in this project have gone stale more than once: a
headline of +14.35 survived in the prose after the pooled reanalysis moved it to
+5.98, and scripts/compare_price_lexicon.py printed a retracted budget claim four
lines below a table that contradicted it. Prose and CSV drift apart silently,
because nothing reads both.

What this does. It builds a pool of every numeric cell in results/*.csv, then
scans the documents and scripts for quantities written as percentage points and
asks whether each one exists in that pool. A number that matches nothing is not
necessarily wrong - it may be a sample size, a count, or a figure computed inline
and never written to disk - but it is a number no file can vouch for, and it has
to be checked by hand.

Deliberately narrow. Only quantities attached to "pp" are scanned. Widening the
pattern to every decimal in the repository produces hundreds of false positives
(version numbers, seeds, section numbers, years, p values) and the report becomes
unreadable, which is the same as having no report.

Decimal commas are handled, because the documents are written in Vietnamese and
use "+5,98 pp" where the CSVs hold 5.98.

Run:  python scripts/check_numbers.py
Exit code 1 if any quantity is unaccounted for, so it can gate a commit.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

# The documents are Vietnamese and the Windows console defaults to cp1252, which
# cannot encode them. Without this the report dies partway through printing.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

TOL = 0.005          # CSVs are rounded to 2 dp, so this is an exact-match window

TARGETS = (sorted((ROOT / "docs").glob("*.md"))
           + [ROOT / "README.md"]
           + sorted((ROOT / "scripts").glob("*.py"))
           + sorted((ROOT / "src").glob("*.py")))

# A number followed by "pp" within a short window, sign optional, comma or dot.
QUANTITY = re.compile(r"([+-]?\d{1,3}[.,]\d{1,2})\s*(?:pp\b|percentage points)")

SELF = Path(__file__).name


def number_pool() -> set[float]:
    """Every numeric cell in every results CSV, rounded to 2 dp."""
    pool: set[float] = set()
    for f in sorted(RESULTS.glob("*.csv")):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        for col in d.columns:
            v = pd.to_numeric(d[col], errors="coerce").dropna()
            for x in v:
                pool.add(round(float(x), 2))
    return pool


def matches(x: float, pool: set[float]) -> bool:
    """A quoted quantity may be written with either sign convention.

    A loss of 5.57 pp is stored as -5.57 in vs_raw.csv but is often written in
    prose as "costs 5.57 pp". Both readings count as accounted for, so the sign
    is not used to reject.
    """
    return any(abs(x - c) < TOL or abs(abs(x) - abs(c)) < TOL for c in pool)


def scan(path: Path) -> list[tuple[int, str, float]]:
    out = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return out
    for i, line in enumerate(text.splitlines(), 1):
        for raw in QUANTITY.findall(line):
            out.append((i, raw, float(raw.replace(",", "."))))
    return out


def main() -> int:
    pool = number_pool()
    n_csv = len(list(RESULTS.glob("*.csv")))
    print("=" * 78)
    print("NUMBER CHECK - quantities in pp, against every results CSV")
    print("=" * 78)
    print(f"\n  pool: {len(pool)} distinct values from {n_csv} CSV files")
    print(f"  scanning {len(TARGETS)} files\n")

    total = 0
    unmatched: list[tuple[Path, int, str]] = []
    for path in TARGETS:
        if path.name == SELF:
            continue
        found = scan(path)
        if not found:
            continue
        bad = [(ln, raw) for ln, raw, val in found if not matches(val, pool)]
        total += len(found)
        rel = path.relative_to(ROOT).as_posix()
        mark = "OK " if not bad else "??  "
        print(f"  {mark} {rel:42s} {len(found):>4d} quantities,"
              f" {len(bad):>3d} unaccounted")
        for ln, raw in bad:
            unmatched.append((path, ln, raw))

    print(f"\n  {total} quantities scanned, {len(unmatched)} unaccounted for")

    if unmatched:
        print("\n" + "=" * 78)
        print("UNACCOUNTED FOR - each of these must be traced by hand")
        print("=" * 78 + "\n")
        for path, ln, raw in unmatched:
            rel = path.relative_to(ROOT).as_posix()
            line = path.read_text(encoding="utf-8").splitlines()[ln - 1].strip()
            print(f"  {rel}:{ln}  {raw} pp")
            print(f"      {line[:110]}")
        print("""
  A number here is not automatically wrong. It may be a figure computed inline
  and never written to a CSV, a quantity from a source outside this project, or
  a hypothetical in a plan. What it is NOT is a number any file in results/ can
  vouch for, so each one needs either a source or a correction.""")
        return 1

    print("\n  Every quantity in pp traces to a value in results/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
