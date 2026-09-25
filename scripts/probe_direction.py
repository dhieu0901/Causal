"""Ask the model outright which way each edge's two names run in the real world.

    python scripts/probe_direction.py --dry-run
    python scripts/probe_direction.py --max-usd 2.0          # SPENDS CREDIT

Registered in prereg/B6.md as a descriptive probe, not a test.

Why. H3a (prereg/CONFIRMATORY.md) was confirmed: permuted real names, which
put the item's own variable names on the wrong nodes, cost no more than
unrelated real words. It was read as "a wrong prior costs nothing". That
reading assumes the permuted names carry a wrong prior - a real-world direction
that runs against the item's edge. On CaLM (REPORT section 3) real names looked
like they carry an association rather than a direction. This probe measures the
direction directly: for every edge u -> v of the confirmatory items, the names
the model saw on u and v, under KEEP (the story's own names) and under PERMUTE
(the H3a names), and one question: in the real world, does a change in one
cause a change in the other, the other way round, or neither?

The two names are shown in an order flipped at random per pair (seeded), so
answer a is not always "agrees with the edge". Edges touching a node named as
unobserved ("unobserved confounders") are left out: that is not a variable a
direction can be asked about.

Writes results/cladder/raw/probe_direction_raw.csv, one row per (model, item,
lexicon, edge). scripts/analyze_b6.py summarises it.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from analyze_querygroup import ARITH, IDENT
from classify_perturbations import canonical_structures
from lexical import build_lexicon, find_mapping, relabel_item
from pilot import make_items, read_ids
from runner import (CHARS_PER_TOKEN, PRICES_PER_M, billed_usd, guard_errors, is_ok,
                    run_batch, usage_summary)

MODELS = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]
MAX_TOKENS = 200
OUT = ROOT / "results" / "cladder" / "raw" / "probe_direction_raw.csv"
ANSWER = re.compile(r"ANSWER:\s*\(?([abc])\)?", re.IGNORECASE)

TEMPLATE = (
    'Two quantities: "{a}" and "{b}".\n\n'
    "In the real world, which of these is true?\n"
    "a) A change in {a} tends to cause a change in {b}.\n"
    "b) A change in {b} tends to cause a change in {a}.\n"
    "c) Neither causes the other.\n\n"
    "Answer with a single letter on the last line, in exactly this format:\n"
    "ANSWER: a\nor\nANSWER: b\nor\nANSWER: c")


def jobs() -> list[dict]:
    """One job per (item, lexicon, edge) of the confirmatory items that were sent."""
    items = make_items(1000, 20260925, 1, "full_v1.5_default.csv", None, True,
                       read_ids("prereg/excluded_ids.txt"))
    items = items[~items.query_type.isin(ARITH | IDENT)]
    canon = canonical_structures()
    out = []
    for i, r in items.iterrows():
        vm = find_mapping(r.prompt)
        if vm is None:
            continue
        names = {"KEEP": {k[:-4]: v for k, v in vm.items() if k.endswith("name")}}
        _, clean = relabel_item(r.prompt, "PERMUTE", seed=str(r.id))
        if clean:
            pm = build_lexicon(vm, "PERMUTE", seed=str(r.id))
            names["PERMUTE"] = {k[:-4]: v for k, v in pm.items() if k.endswith("name")}
        for u, v in canon[r.graph_id]:
            if "unobserved" in names["KEEP"].get(u, "").lower() or \
                    "unobserved" in names["KEEP"].get(v, "").lower():
                continue
            flip = random.Random(f"probe:{r.id}:{u}:{v}").random() < 0.5
            for lex, nm in names.items():
                if u not in nm or v not in nm:
                    continue
                a, b = (nm[v], nm[u]) if flip else (nm[u], nm[v])
                out.append(dict(item=i, id=int(r.id), query_type=r.query_type,
                                question_property=r.question_property, lexicon=lex,
                                u=u, v=v, first=a, second=b, flipped=flip,
                                prompt=TEMPLATE.format(a=a, b=b)))
    return out


def relation(pred, flipped):
    if pred is None:
        return "unparsed"
    if pred == "c":
        return "neither"
    first_causes_second = pred == "a"
    # unflipped: first is u, so "a" says u -> v, the item's own edge
    return "agrees" if first_causes_second != flipped else "reversed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    ap.add_argument("--max-usd", type=float, default=None, dest="max_usd")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    J = jobs()
    distinct = len({j["prompt"] for j in J})
    print(f"{len(J)} (item, lexicon, edge) rows, {distinct} distinct prompts per model")
    if a.dry_run:
        tin = sum(len(p) for p in {j["prompt"] for j in J}) / CHARS_PER_TOKEN
        tot = 0.0
        for m in MODELS:
            pi, po = PRICES_PER_M[m]
            usd = (tin * pi + distinct * MAX_TOKENS * po) / 1e6
            tot += usd
            print(f"  {m:14s} at most {usd:.2f} USD (every answer at the {MAX_TOKENS}-token cap)")
        print(f"  UPPER BOUND, nothing sent: {tot:.2f} USD")
        return 0

    spent, rows = 0.0, []
    for m in MODELS:
        left = None if a.max_usd is None else max(0.0, a.max_usd - spent)
        recs = run_batch(J, m, temperature=0.0, workers=a.workers, max_tokens=MAX_TOKENS,
                         max_usd=left)
        # Once per DISTINCT prompt: many rows share a prompt (the same two names
        # recur across items) and a result. Summing over rows, as the first run
        # on 2026-09-25 did, printed 1.39 USD for calls that cost 0.23; sending
        # was never affected, run_batch caps per prompt.
        spent += sum(billed_usd(m, r) for r in {x["prompt"]: x["result"] for x in recs}.values()
                     if not r.get("cached"))
        if any("USD cap" in r["result"].get("error", "") for r in recs):
            print(f"    SPENDING CAP REACHED: {spent:.2f} USD spent, --max-usd {a.max_usd}")
        print(f"  {m}: {usage_summary(recs)}  spent so far {spent:.2f} USD")
        guard_errors(recs, label=m)
        for r in recs:
            if not is_ok(r["result"]):
                continue
            hits = ANSWER.findall(r["result"]["text"])
            pred = hits[-1].lower() if hits else None
            rows.append({k: r[k] for k in ("item", "id", "query_type", "question_property",
                                           "lexicon", "u", "v", "first", "second", "flipped")}
                        | dict(model=m, pred=pred, parsed=int(pred is not None),
                               relation=relation(pred, r["flipped"]),
                               in_tok=r["result"]["in_tok"], out_tok=r["result"]["out_tok"]))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"  wrote {OUT.relative_to(ROOT)}  ({len(rows)} rows, {spent:.2f} USD)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
