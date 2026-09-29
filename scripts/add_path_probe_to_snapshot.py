"""Add the minimum path-probe response texts to the public snapshot.

This release-preparation helper reads the private API cache and is deliberately
not part of the zero-cost analysis pipeline. The analysis itself only reads the
resulting three-column snapshot and never creates an API client.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_path import MAX_TOKENS, context_jobs, jobs
from runner import _key, read_cached

RAW = ROOT / "results" / "cladder" / "raw"
SNAPSHOT = RAW / "analysis_response_snapshot.csv"
SOURCES = (("probe_path_raw.csv", False),
           ("probe_path_raw_llama.csv", False),
           ("probe_path_context_raw.csv", True))


def main() -> int:
    old = pd.read_csv(SNAPSHOT, keep_default_na=False)
    if set(old.columns) != {"cache_key", "model", "text"}:
        raise SystemExit("The existing response snapshot has an unexpected schema")
    if old.cache_key.duplicated().any():
        raise SystemExit("The existing response snapshot has duplicate keys")

    added = {}
    for filename, context in SOURCES:
        probes = pd.read_csv(RAW / filename)
        rebuilt = pd.DataFrame(context_jobs() if context else jobs())[
            ["item", "lexicon", "cond", "prompt"]
        ]
        unparsed = probes[probes.parsed == 0].merge(
            rebuilt, on=["item", "lexicon", "cond"], validate="many_to_one"
        )
        if len(unparsed) != int((probes.parsed == 0).sum()):
            raise SystemExit(f"Not every unparsed row in {filename} has a rebuilt prompt")
        for row in unparsed.itertuples():
            record = read_cached(row.model, 0.0, row.prompt, max_tokens=MAX_TOKENS)
            if not record or not record.get("text"):
                raise SystemExit(
                    f"Missing cached response for {filename}: "
                    f"{row.model}, item {row.item}, {row.lexicon}, {row.cond}"
                )
            key = _key(row.model, 0.0, row.prompt, max_tokens=MAX_TOKENS).stem
            value = (row.model, record["text"])
            if key in added and added[key] != value:
                raise SystemExit(f"Conflicting cached records for key {key}")
            added[key] = value

    extra = pd.DataFrame(
        ((key, model, text) for key, (model, text) in added.items()),
        columns=["cache_key", "model", "text"],
    )
    combined = pd.concat([old, extra], ignore_index=True).drop_duplicates("cache_key", keep="first")
    combined = combined.sort_values("cache_key").reset_index(drop=True)
    combined.to_csv(SNAPSHOT, index=False, lineterminator="\n")
    print(f"snapshot: {len(old)} -> {len(combined)} rows ({len(combined) - len(old)} added)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
