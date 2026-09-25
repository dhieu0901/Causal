"""Give every per-response record its CLadder `id`, and prove each one.

    python scripts/backfill_ids.py            # verify; insert the column where missing

Why. pilot.py, induction.py and check_drift.py wrote `item`, a row index into
the sample make_items() drew, and never the CLadder `id`. Every analysis that
pools samples had to re-draw the sample to map one to the other
(pool_samples.verify_item_map, analyze_prior_strength.label_items). That is
right today and would go silently wrong the day make_items changed. Since
2026-09-24 the three writers record `id` themselves; this file adds it to the
records written before that.

How, without touching anything else. The column is inserted into each line of
text right after `item`. Every field before `item` (model, lexicon, cond) is a
bare token with no comma or quote, so the split is exact, and every other byte
of the file stays as it was. Re-reading the file must give the old columns back
unchanged, or nothing is written.

Proof, per row, not per file. Each row's graph_id, rung and gold label - and
query_type and story_id where the file has them - must equal CLadder's own row
for the id assigned to it. For the first pilot sample, whose arguments were
recovered by matching (not written down), the RAW prompt rebuilt for every item
must also be a key in the API cache when the cache is present.

Idempotent: a file that already has `id` is verified, not rewritten.
Outside the analysis order: it changes inputs, not outputs.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

RES = ROOT / "results" / "cladder"
SEED = 20260907

# The sample each file was drawn from. lex, n600 and price400 have row-verified
# maps written by pool_samples.py. "pilot" is the first run (n=150, kmax=3,
# nonsense stories kept): the defaults of induction.py, and the only arguments
# whose draw reproduces pilot_raw.csv item by item (swept 2026-09-24).
SAMPLE_OF = {"pilot_raw.csv": "pilot", "induction_raw.csv": "pilot",
             "drift_check_raw.csv": "n600", "drift_check_raw_lex.csv": "lex",
             "drift_check_raw_price400.csv": "price400"}
for f in (RES / "raw").glob("pilot_raw_*.csv"):
    t = f.name.removeprefix("pilot_raw_").removesuffix(".csv")
    SAMPLE_OF[f.name] = ("n600" if t.startswith(("n600", "cleanrawn600")) else
                         "price400" if t.startswith(("price400", "cleanrawprice400")) else
                         "lex")
for f in (RES / "raw").glob("induction_raw_lex*.csv"):
    SAMPLE_OF[f.name] = "lex"


def item_to_id(sample):
    if sample == "pilot":
        from pilot import make_items
        it = make_items(150, SEED, 3, "full_v1.5_default.csv", None, False)
        return it.id
    return pd.read_csv(RES / f"_itemmap_{sample}.csv").set_index("item").id


def prove_pilot(ids) -> str:
    """Every rebuilt RAW prompt of the first pilot must be in the cache."""
    from pilot import make_items
    from prompts import build
    from runner import _key, CACHE
    if not any(CACHE.glob("*.json")):
        return "cache absent, prompt proof skipped"
    it = make_items(150, SEED, 3, "full_v1.5_default.csv", None, False)
    hit = sum(_key("gpt-4.1-nano", 0.0, build(p, "RAW")).exists() for p in it.prompt)
    if hit != len(it):
        raise SystemExit(f"first pilot: only {hit}/{len(it)} RAW prompts are in the cache; "
                         f"the recovered sample is not the one that was run")
    return f"{hit}/{len(it)} RAW prompts found in the cache"


def main() -> int:
    full = pd.read_csv(ROOT / "data" / "cladder" / "full_v1.5_default.csv", low_memory=False).set_index("id")
    maps, bad = {}, 0
    for name in sorted(SAMPLE_OF):
        f = RES / ("raw" if name.startswith(("pilot_raw", "induction_raw")) else "") / name
        if not f.exists():
            continue
        sample = SAMPLE_OF[name]
        if sample not in maps:
            maps[sample] = item_to_id(sample)
        m = maps[sample]
        d = pd.read_csv(f)
        ids = d.item.map(m)
        if ids.isna().any():
            raise SystemExit(f"{name}: {int(ids.isna().sum())} items have no id in sample {sample}")
        ids = ids.astype(int)
        if "id" in d.columns and not (d.id == ids).all():
            raise SystemExit(f"{name}: the id column disagrees with the {sample} map")
        # per-row proof against CLadder itself
        checks = [("graph_id", "graph_id"), ("rung", "rung"), ("gold", "label"),
                  ("query_type", "query_type"), ("story_id", "story_id")]
        for mine, theirs in checks:
            if mine in d.columns:
                ok = d[mine].astype(str).values == full.loc[ids, theirs].astype(str).values
                if not ok.all():
                    raise SystemExit(f"{name}: {int((~ok).sum())} rows disagree with CLadder "
                                     f"on {mine} for the id assigned to them")
        verified = [c for c, _ in checks if c in d.columns]
        if "id" in d.columns:
            print(f"  {name:40s} has id; {len(d)} rows verified on {verified}")
            continue
        # newline="" keeps each line's own ending; read_text() would turn CRLF
        # into LF and change every line of the file (it did, on the first run).
        with f.open(encoding="utf-8", newline="") as fh:
            lines = fh.read().splitlines(keepends=True)
        header = lines[0].rstrip("\r\n").split(",")
        pos = header.index("item")
        if any(c in ("".join(header[:pos])) for c in '"'):
            raise SystemExit(f"{name}: quoted header before item; not safe to split")
        out = []
        for k, line in enumerate(lines):
            parts = line.split(",", pos + 1)
            val = "id" if k == 0 else str(ids.iloc[k - 1])
            if k > 0 and parts[pos].strip() != str(d.item.iloc[k - 1]):
                raise SystemExit(f"{name}: line {k} does not split cleanly at item")
            out.append(",".join(parts[:pos + 1] + [val] + parts[pos + 1:]))
        new = "".join(out)
        back = pd.read_csv(io.StringIO(new))
        if not back.drop(columns="id").equals(d) or not (back.id == ids).all():
            raise SystemExit(f"{name}: inserting the column would change other values; nothing written")
        f.write_text(new, encoding="utf-8", newline="")
        print(f"  {name:40s} id inserted; {len(d)} rows verified on {verified}")
    print(f"\n  first pilot sample: {prove_pilot(maps.get('pilot'))}")
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
