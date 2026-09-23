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
    "bản cũ", "đã bị thay thế", "con số dịch", "đã rút",
)
# Scripts carry their history in English docstrings and printed notes, and the
# marks above are Vietnamese, so a script saying "Until 2026-09-23 this block
# printed -0.11 [-1.62 ; 1.42] as though it were a result" was read as a live
# claim. These are only consulted for .py files, over the whole paragraph the
# value sits in, because a docstring sentence runs across several lines.
HISTORY_MARK_EN = ("until 2026-", "retracted", "as though it were a result",
                   "no longer stands", "was rejected")
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
                # Read the WHOLE governing blockquote, not just its last line. A
                # warning block usually opens with the marker and then explains
                # itself over several lines, so looking only at the nearest one
                # found the explanation and missed the warning.
                k0 = j
                while k0 > 1 and lines[k0 - 2].lstrip().startswith(">"):
                    k0 -= 1
                gov = " ".join(lines[k0 - 1:j])
                if any(k.lower() in gov.lower() for k in HISTORY_MARK):
                    return "HISTORY"
                if any(k in gov for k in EXTERNAL_MARK):
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


def paragraph_kind(lines: list[str], i: int) -> str | None:
    """HISTORY if the blank-line-delimited paragraph around line `i` says so.

    Only used for .py files - see HISTORY_MARK_EN.
    """
    a = i - 1
    while a > 0 and lines[a - 1].strip():
        a -= 1
    b = i
    while b < len(lines) and lines[b].strip():
        b += 1
    para = " ".join(lines[a:b]).lower()
    if any(k in para for k in HISTORY_MARK_EN):
        return "HISTORY"
    return None


def column_kind(lines: list[str], i: int, pos: int) -> str | None:
    """HISTORY/EXTERNAL if the COLUMN this value sits in is marked, else None.

    Rule four, and the first one that is not about blockquotes. REPORT.md
    section 4.4 holds a table whose header reads

        | | Do thi lam thay doi | mau n=86, **da rut** | mau gop n=490, **hien hanh** |

    so one row carries a retracted value and a live value side by side:
    `+9,75 pp` is history, `+5,12 pp` next to it is the current headline. A
    line-level marker cannot express that - marking the row would excuse the
    live number too, which is worse than the miss it fixes. Status here is a
    property of the COLUMN, so that is what gets read.
    """
    line = lines[i - 1]
    if not line.lstrip().startswith("|"):
        return None
    # Walk up to the first row of this table; that row is the header.
    j = i
    while j > 1 and lines[j - 2].lstrip().startswith("|"):
        j -= 1
    if j == i:
        return None                      # this IS the header row
    head = lines[j - 1]
    # Which cell does `pos` fall into? Count the pipes before it. Both header
    # and row are split the same way, so a ragged row simply misses and the
    # function falls through to None rather than guessing a neighbour.
    col = line.count("|", 0, pos)
    cells = head.split("|")
    if col >= len(cells):
        return None
    cell = cells[col]
    if any(k.lower() in cell.lower() for k in HISTORY_MARK):
        return "HISTORY"
    if any(k in cell for k in EXTERNAL_MARK):
        return "EXTERNAL"
    return None


def scan(path: Path) -> list[tuple[int, str, float, str]]:
    out = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return out
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        kind = classify(lines, i)
        if kind == "LIVE" and path.suffix == ".py":
            kind = paragraph_kind(lines, i) or kind
        for m in QUANTITY.finditer(line):
            raw = m.group(1)
            k = kind if kind != "LIVE" else (column_kind(lines, i, m.start(1)) or kind)
            out.append((i, raw, float(raw.replace(",", ".")), k))
        for m in INTERVAL.finditer(line):
            for gi in (1, 2):
                raw = m.group(gi)
                k = kind if kind != "LIVE" else (column_kind(lines, i, m.start(gi)) or kind)
                out.append((i, f"CI {raw}", float(raw.replace(",", ".")), k))
    return out


# The pool above is DENSE, and that makes a single-value match weak evidence.
# Measured on 2026-09-23 over 3,236 distinct values: 99% of every 2-dp number
# below 1 is somewhere in results/, 75% of those in [1, 5), 48% in [5, 10).
# Most of this project's effects live in exactly that range, so a made-up +4.37
# pp "traces to a file" three times in four. The gate reported 0 live claims
# while WALKTHROUGH still carried -0.11 [-1.62 ; +1.42], a CI that no function
# here ever computed: -1.62 and 1.42 each matched some unrelated cell.
#
# An interval is a much stronger fingerprint. Its two bounds - and the estimate
# written just before it - must sit in ONE ROW of ONE CSV. Three numbers landing
# in the same row by chance is negligible, so a claim that passes this test was
# produced by the computation it claims to come from.
TAIL = re.compile(r"([+-]?\d{1,3}[.,]\d{1,2})\s*(?:pp)?\s*\**\s*(?:,?\s*CI\s*)?$")


def row_index() -> tuple[dict[float, set[int]], dict[int, str]]:
    """|value| rounded to 2 dp -> the set of (file, row) ids holding it."""
    idx: dict[float, set[int]] = {}
    where: dict[int, str] = {}
    rid = 0
    for f in sorted(RESULTS.glob("*.csv")):
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        num = d.apply(pd.to_numeric, errors="coerce").to_numpy()
        for r in num:
            for x in r:
                if x == x:                      # not NaN
                    idx.setdefault(round(abs(float(x)), 2), set()).add(rid)
            where[rid] = f.name
            rid += 1
    return idx, where


def _rows(idx: dict[float, set[int]], v: float) -> set[int]:
    # One unit of rounding either way: a CSV holding 5.975 prints as 5.98 in
    # prose but may round to 5.97 here.
    out: set[int] = set()
    for k in (round(v - .01, 2), round(v, 2), round(v + .01, 2)):
        out |= idx.get(k, set())
    return out


def intervals_without_a_row(idx: dict[float, set[int]]) -> list[tuple[Path, int, str]]:
    """LIVE intervals whose bounds and estimate never share a CSV row."""
    f2 = lambda t: abs(float(t.replace(",", ".")))
    bad = []
    for path in TARGETS:
        if path.name == SELF or path.name in LOG_FILES:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(lines, 1):
            kind = classify(lines, i)
            if kind == "LIVE" and path.suffix == ".py":
                kind = paragraph_kind(lines, i) or kind
            for m in INTERVAL.finditer(line):
                k = kind if kind != "LIVE" else (column_kind(lines, i, m.start(1)) or kind)
                if k != "LIVE":
                    continue
                lo, hi = f2(m.group(1)), f2(m.group(2))
                if (lo, hi) == (2.5, 97.5):     # np.percentile(..., [2.5, 97.5])
                    continue
                head = line[:m.start()]
                t = TAIL.search(head) or TAIL.search(head.rstrip(" |*"))
                common = _rows(idx, lo) & _rows(idx, hi)
                if t:
                    common &= _rows(idx, f2(t.group(1)))
                if not common:
                    bad.append((path, i, (t.group(1) + " " if t else "") + m.group(0)))
    return bad


# Table cells are the third blind spot. Every results table in this project is
# written as "| gpt-4.1 | +43,75 | ..." - no "pp", no interval - so neither test
# above ever looked at them. On 2026-09-23 that let through a whole moderator
# table in WALKTHROUGH (+22,44, +27,44, +16,29 ...) and a per-model table in
# REPORT (-12,39 / -7,68 / +6,27) that no file had produced for days. A signed
# decimal is how this project writes an effect; counts and sample sizes carry
# no sign. So: every signed decimal in a LIVE table cell must match the pool.
# This is only as strong as the single-value test - the pool is dense - but it
# is the difference between checking those cells weakly and not at all.
SIGNED_CELL = re.compile(r"(?<![\w.,/])([+-]\d{1,3}[.,]\d{1,2})(?![\d%])")


def signed_table_cells_unmatched(pool: set[float]) -> list[tuple[Path, int, str]]:
    bad = []
    for path in TARGETS:
        if path.suffix != ".md" or path.name in LOG_FILES:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(lines, 1):
            if not line.lstrip().startswith("|"):
                continue
            kind = classify(lines, i)
            for m in SIGNED_CELL.finditer(line):
                k = kind if kind != "LIVE" else (column_kind(lines, i, m.start(1)) or kind)
                if k != "LIVE":
                    continue
                if not matches(float(m.group(1).replace(",", ".")), pool):
                    bad.append((path, i, m.group(1)))
    return bad


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

    idx, _ = row_index()
    rowless = intervals_without_a_row(idx)
    print("\n" + "=" * 78)
    print("INTERVALS WHOSE ESTIMATE AND BOUNDS SHARE NO ROW")
    print("=" * 78)
    absp = {round(abs(x), 2) for x in pool}
    dens = []
    for lo_, hi_ in ((1, 5), (5, 10), (10, 20)):
        grid = [round(lo_ + k / 100, 2) for k in range((hi_ - lo_) * 100)]
        dens.append(f"[{lo_},{hi_}) {100 * sum(g in absp for g in grid) / len(grid):.0f}%")
    print("\n  Why this test exists: a lone value matching the pool is weak evidence.")
    print("  Share of all 2-dp values already in the pool: " + ", ".join(dens) + ".")
    if rowless:
        print()
        for path, ln, txt in rowless:
            print(f"  {path.relative_to(ROOT).as_posix()}:{ln}  {txt}")
        print("\n  Each bound may match SOMETHING, but no single CSV row holds the")
        print("  interval with its estimate. Trace it to the script that computed it,")
        print("  or mark it as history.")
    else:
        print("\n  Khong co. Moi khoang tin cay song deu nam tron tren mot dong CSV.")

    cells = signed_table_cells_unmatched(pool)
    print("\n" + "=" * 78)
    print("SIGNED TABLE CELLS WITH NOTHING BEHIND THEM")
    print("=" * 78)
    if cells:
        print()
        for path, ln, txt in cells:
            print(f"  {path.relative_to(ROOT).as_posix()}:{ln}  {txt}")
        print("\n  A signed decimal in a live table is an effect size, and it matches")
        print("  no value in results/. Regenerate the table from its script.")
    else:
        print("\n  Khong co. Moi o bang co dau deu khop mot gia tri trong results/.")

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

    if unmatched or bad_p or rowless or cells:
        return 1

    print("\n  Every quantity in pp traces to a value in results/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
