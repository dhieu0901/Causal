"""Add the minimum DeepSeek-R1 chains to the public response snapshot.

This release-preparation helper reads the private API cache and is deliberately
not part of the zero-cost analysis pipeline. For a completion cut off at 8,000
tokens, it stores the paid 16,000-token re-ask selected by the analysis. The
public analysis never creates an API client.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_querygroup import ARITH, IDENT
from analyze_r1_chains import CAP, LEX, RECAP, SEED, TEMP
from analyze_second_family import R1
from pilot import build_jobs, make_items
from runner import _key, read_cached

SNAPSHOT = ROOT / "results" / "cladder" / "raw" / "analysis_response_snapshot.csv"


def main() -> int:
    old = pd.read_csv(SNAPSHOT, keep_default_na=False)
    if set(old.columns) != {"cache_key", "model", "text"}:
        raise SystemExit("The existing response snapshot has an unexpected schema")
    if old.cache_key.duplicated().any():
        raise SystemExit("The existing response snapshot has duplicate keys")

    items = make_items(LEX["n_items"], SEED, 1, "full_v1.5_default.csv", None, True)
    jobs = [
        job for job in build_jobs(items, 1, SEED, ("DR",), "PSEUDO")
        if job["cond"] in ("RAW", "ORACLE", "DR_k1")
        and job["query_type"] not in ARITH | IDENT
    ]
    added = {}
    recaps = 0
    for job in jobs:
        record = read_cached(R1, TEMP, job["prompt"], max_tokens=CAP)
        if not record:
            raise SystemExit(f"Missing cached R1 response for item {job['item']}, {job['cond']}")
        if record.get("finish") == "length":
            record = read_cached(R1, TEMP, job["prompt"], max_tokens=RECAP)
            if not record:
                raise SystemExit(f"Missing R1 re-ask for item {job['item']}, {job['cond']}")
            recaps += 1
        text = (record.get("reasoning") or "") + "\n" + (record.get("text") or "")
        if not text.strip():
            raise SystemExit(f"Empty R1 chain for item {job['item']}, {job['cond']}")
        key = _key(R1, TEMP, job["prompt"], max_tokens=CAP).stem
        added[key] = (R1, text)

    extra = pd.DataFrame(
        ((key, model, text) for key, (model, text) in added.items()),
        columns=["cache_key", "model", "text"],
    )
    combined = pd.concat([old, extra], ignore_index=True).drop_duplicates("cache_key", keep="last")
    combined = combined.sort_values("cache_key").reset_index(drop=True)
    combined.to_csv(SNAPSHOT, index=False, lineterminator="\n")
    print(
        f"snapshot: {len(old)} -> {len(combined)} rows "
        f"({len(combined) - len(old)} added, {recaps} re-asks selected)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
