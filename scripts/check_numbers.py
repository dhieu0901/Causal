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

# Confidence bounds are written "[-11,41 ; -3,81]" and almost never carry a "pp",
# so QUANTITY above walked straight past them. That is not a small gap: on
# 2026-09-22 REVIEW.md was still showing [-11,21 ; -3,69] for a table whose CSV
# had moved to [-11,41 ; -3,81], and nothing in this file could see it. A CI is
# the part a reviewer checks first, so it gets its own pattern.
INTERVAL = re.compile(
    r"\[\s*([+-]?\d{1,3}[.,]\d{1,2})\s*[;,]\s*([+-]?\d{1,3}[.,]\d{1,2})\s*\]")

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


# A quantity with no CSV behind it is not automatically a defect. Two kinds are
# legitimate, and both are detected from context rather than from a hand-written
# list of values - a hand list goes stale the moment a number is edited, and a
# gate that needs manual upkeep stops being run.
#
# HISTORY  a value inside a blockquote whose OPENING line announces a
#          correction. The project records superseded numbers on purpose, so
#          that a reader who saw the old figure can find out what happened to
#          it. Deleting them would be worse than keeping them.
#
# EXTERNAL a value on a line that cites another paper by name. Caliper's 7.6 and
#          29.6 pp belong to Caliper; no CSV here can or should vouch for them.
#
# Everything else is a live claim of this project with nothing behind it, and
# that is what the exit code is for.
HISTORY_MARK = (
    "đã có script", "trước đây", "bản trước", "bản đầu", "đã rút", "đã bỏ",
    "cảnh báo", "đã sửa", "đã đo", "khép lại", "sửa ngày", "đổi cách trình bày",
    "đã khớp lại", "đã hạ cấp", "đã đóng", "không script nào", "đừng trích",
)
EXTERNAL_MARK = ("Caliper", "CausalGraph2LLM", "Corr2Cause", "GSM-Symbolic",
                 "Vernier", "NoisyCausal", "RE-IMAGINE", "Yamin", "Lopiano")

# REVIEW.md is a dated log of review rounds. A round records what the panel was
# shown AT THE TIME, and several of those figures have since been superseded on
# purpose - that is the document's job. Holding it to the current CSVs would
# force the log to be rewritten backwards, which destroys the only record of how
# a number changed. It is still SCANNED, and its count is printed, so a reader
# can see how much of it has gone out of date; it just does not fail the gate.
#
# This exemption is by file, and it is the only one. README.md, REPORT.md,
# PROPOSAL.md and WALKTHROUGH.md all describe the project as it stands now and
# are held to the CSVs - WALKTHROUGH was still carrying the retracted headline
# CI [+1.78 ; +10.30] on 2026-09-23, three documents after it was corrected
# everywhere else, and that is exactly what this gate is for.
LOG_FILES = {"REVIEW.md"}


def classify(lines: list[str], i: int) -> str:
    """HISTORY, EXTERNAL or LIVE for the 1-indexed line `i`.

    The note a value belongs to is the nearest bold heading ABOVE it inside the
    same blockquote, not the first line of the blockquote. A long quoted block
    here holds several separate notes, each opening with `> **...**`, so reading
    only the block's first line attributed every value to whichever note came
    first - line 187 of REPORT.md announces "da do va khep lai" on its own line
    and was still being called LIVE.
    """
    line = lines[i - 1]
    low = line.lower()
    if any(k in line for k in EXTERNAL_MARK):
        return "EXTERNAL"
    if any(k.lower() in low for k in HISTORY_MARK):
        return "HISTORY"
    if not line.lstrip().startswith(">"):
        # A table is not a blockquote, but a warning block sitting directly
        # above one governs it. WALKTHROUGH.md has the retracted structure-arms
        # table introduced by exactly such a block, and without this the three
        # rows underneath were still counted as live claims.
        if line.lstrip().startswith("|"):
            j = i - 1
            while j > 0 and (not lines[j - 1].strip()
                             or lines[j - 1].lstrip().startswith("|")):
                j -= 1
            if j > 0 and lines[j - 1].lstrip().startswith(">"):
                gov = lines[j - 1].lower()
                if any(k.lower() in gov for k in HISTORY_MARK):
                    return "HISTORY"
                if any(k in lines[j - 1] for k in EXTERNAL_MARK):
                    return "EXTERNAL"
        return "LIVE"
    # Walk up inside the blockquote to the nearest heading line.
    j = i - 1
    while j > 0 and lines[j - 1].lstrip().startswith(">"):
        head = lines[j - 1]
        if "**" in head:
            hl = head.lower()
            if any(k.lower() in hl for k in HISTORY_MARK):
                return "HISTORY"
            if any(k in head for k in EXTERNAL_MARK):
                return "EXTERNAL"
            break          # a heading that says neither: this note is live
        j -= 1
    return "LIVE"


def scan(path: Path) -> list[tuple[int, str, float, str]]:
    out = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return out
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        kind = classify(lines, i)
        for raw in QUANTITY.findall(line):
            out.append((i, raw, float(raw.replace(",", ".")), kind))
        for lo, hi in INTERVAL.findall(line):
            for raw in (lo, hi):
                out.append((i, f"CI {raw}", float(raw.replace(",", ".")), kind))
    return out


def out_of_range_probabilities() -> list[str]:
    """Any p outside [0, 1] sitting in a shipped CSV.

    This is a class-level guard, not a spot fix. A two-tailed bootstrap p is
    2*min(left, right), and a draw landing exactly on 0 is counted by BOTH
    tails, so the product can exceed 1 whenever an effect sits near zero. Three
    separate helpers in this repo computed it that way; two were clamped and the
    third kept shipping p=1.0255 in results/family_breakdown.csv for weeks. The
    number is never wrong in an interesting way - it is wrong in the way that
    tells a reviewer the table was never read.
    """
    bad = []
    for f in sorted(RESULTS.glob("*.csv")):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        for col in d.columns:
            name = col.lower()
            if not (name == "p" or "p_boot" in name or "pval" in name
                    or "p_value" in name):
                continue
            v = pd.to_numeric(d[col], errors="coerce").dropna()
            for x in v[(v > 1.0) | (v < 0.0)]:
                bad.append(f"{f.name}:{col} = {x}")
    return bad


def main() -> int:
    pool = number_pool()
    n_csv = len(list(RESULTS.glob("*.csv")))
    print("=" * 78)
    print("NUMBER CHECK - quantities in pp, against every results CSV")
    print("=" * 78)
    print(f"\n  pool: {len(pool)} distinct values from {n_csv} CSV files")
    print(f"  scanning {len(TARGETS)} files\n")

    total = 0
    live: list[tuple[Path, int, str]] = []
    excused = {"HISTORY": 0, "EXTERNAL": 0, "LOG": 0}
    for path in TARGETS:
        if path.name == SELF:
            continue
        found = scan(path)
        if not found:
            continue
        is_log = path.name in LOG_FILES
        bad = [(ln, raw, "LOG" if is_log and kind == "LIVE" else kind)
               for ln, raw, val, kind in found if not matches(val, pool)]
        total += len(found)
        n_live = sum(1 for _, _, k in bad if k == "LIVE")
        for _, _, k in bad:
            if k != "LIVE":
                excused[k] += 1
        rel = path.relative_to(ROOT).as_posix()
        mark = "OK " if not n_live else "??  "
        print(f"  {mark} {rel:42s} {len(found):>4d} quantities,"
              f" {n_live:>3d} live, {len(bad) - n_live:>3d} excused")
        for ln, raw, k in bad:
            if k == "LIVE":
                live.append((path, ln, raw))

    print(f"\n  {total} quantities scanned")
    print(f"  {excused['HISTORY']:>4d} mien vi nam trong khoi ghi nhan sua doi")
    print(f"  {excused['EXTERNAL']:>4d} mien vi trich tu cong trinh khac")
    print(f"  {excused['LOG']:>4d} mien vi nam trong REVIEW.md, nhat ky cac vong")
    print(f"  {len(live):>4d} KHANG DINH SONG khong co file nao dung sau")
    unmatched = live

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
  Each of these is a LIVE claim of this project. Values inside a block that
  announces a correction, and values on a line citing another paper, were
  already excused above - so these are neither history nor borrowed. Each one
  needs a script, a citation, or a retraction.""")

    bad_p = out_of_range_probabilities()
    print("\n" + "=" * 78)
    print("PROBABILITIES OUTSIDE [0, 1]")
    print("=" * 78)
    if bad_p:
        for line in bad_p:
            print(f"  {line}")
        print("\n  A two-tailed bootstrap p is 2*min(left, right) and a draw landing")
        print("  exactly on 0 is counted by both tails. Clamp it with min(1.0, ...).")
    else:
        print("\n  Khong co. Moi cot p deu nam trong [0, 1].")

    if unmatched or bad_p:
        return 1

    print("\n  Every quantity in pp traces to a value in results/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
