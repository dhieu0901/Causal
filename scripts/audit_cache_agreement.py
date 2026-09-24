"""Does every answer in results/pilot_raw_*.csv equal the answer in the API cache?

    python scripts/audit_cache_agreement.py      # about 10 minutes, reads ~91k files

Every analysis reads the per-response CSVs, never the cache. But every run that
adds conditions to an old sample reads old prompts back FROM the cache, and so
does every re-run of the experiment. If the two disagree, a condition answered
from the cache is not the condition the CSV analysed.

They can disagree because temperature 0 is not deterministic on this API (15% of
answers flip on re-asking, scripts/check_drift.py) and, until 2026-09-24,
src/runner.py let a second call to one prompt overwrite the cache while the
first caller had already written its own answer to its CSV. Result of the audit
on 2026-09-24: 90,901 cells agree, 494 differ (0.54%), 3 cache files are
corrupt. 472 of the 494 are the n600 run of 2026-09-16, whose cache entries were
written twice within about forty minutes that evening; every run made on
2026-09-24 agrees 100%. The CSVs are the canonical record. runner.call now lets
the first writer win and returns the cached answer to every later caller.

Outside the analysis order: it needs the cache, which is not in the repository.
It writes nothing.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

import pilot
from prompts import parse_answer
from runner import _key

SEED = 20260907

# tag -> (n, kmax, sample_kmax, types, lexicon, build_jobs flags): how each file was run
RUNS = {}
for L in ["KEEP", "PERMUTE", "IRRELEVANT", "SYMBOL", "PSEUDO"]:
    RUNS[f"lex{L}"] = (200, 1, 1, ("DR",), L, {})
for L in ["KEEP", "PERMUTE", "SYMBOL", "PSEUDO"]:
    RUNS[f"edfe{L}"] = (200, 1, 1, ("DR", "ED", "FE"), L, {})
for L in ["KEEP", "PSEUDO"]:
    RUNS[f"instr{L}"] = (200, 1, 1, ("DR",), L, {"with_instr": True})
    RUNS[f"n600{L}"] = (600, 1, 1, ("DR",), L, {})
    RUNS[f"price400{L}"] = (400, 3, 3, ("DR", "ED", "FE"), L, {})
    RUNS[f"n600arms{L}"] = (600, 3, 1, ("DR",), L, {"with_names": True, "with_scramble": True})
    for s, (n, k, t) in {"lex": (200, 1, ("DR",)), "n600": (600, 1, ("DR",)),
                         "price400": (400, 3, ("DR", "ED", "FE"))}.items():
        RUNS[f"cleanraw{s}{L}"] = (n, k, k, t, L, {"with_clean_raw": True})
for L in ["PERMUTE", "IRRELEVANT", "SYMBOL"]:
    RUNS[f"n600ladder{L}"] = (600, 1, 1, ("DR",), L, {})


def main() -> int:
    if not any((ROOT / "cache").glob("*.json")):
        print("cache/ is empty (it is not in the repository); nothing to audit.")
        return 0
    tot = Counter()
    for tag, (n, kmax, skmax, types, lex, flags) in RUNS.items():
        f = ROOT / "results" / f"pilot_raw_{tag}.csv"
        if not f.exists():
            continue
        items = pilot.make_items(n, SEED, skmax, "full_v1.5_default.csv", None, True)
        jobs = pilot.build_jobs(items, kmax, SEED, types, lex, **flags)
        prompts = {(j["item"], j["cond"]): j["prompt"] for j in jobs}
        d = pd.read_csv(f)
        c, when = Counter(), Counter()
        for m, i, cond, pr in zip(d.model, d.item, d.cond, d.pred):
            p = prompts.get((i, cond))
            if p is None:
                c["no prompt rebuilt"] += 1
                continue
            k = _key(m, 0.0, p)
            if not k.exists():
                c["not in cache"] += 1
                continue
            try:
                rec = json.loads(k.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                c["corrupt cache file"] += 1
                continue
            cp = parse_answer(rec["text"])
            if ("" if pd.isna(pr) else str(pr)) == ("" if cp is None else cp):
                c["agree"] += 1
            else:
                c["DIFFER"] += 1
                when[dt.datetime.fromtimestamp(os.path.getmtime(k)).strftime("%m-%d")] += 1
        tot.update(c)
        print(f"{tag:22s} rows={len(d):6d} {dict(c)}  cache dates of differing cells={dict(when)}",
              flush=True)
    print("\nTOTAL", dict(tot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
