"""Ask the model outright whether the graph it was shown has a directed X -> Y path.

    python scripts/probe_path.py --dry-run
    python scripts/probe_path.py --pilot 4          # a few calls to measure answer length
    python scripts/probe_path.py --max-usd 3.0      # SPENDS CREDIT

Registered in prereg/PATH_PROBE.md.

Why. B6 confirmed that the harm of a wrong graph sits in reversals that leave no
directed path from treatment X to outcome Y (REPORT section 2.2). Two readings
fit that result. The model reads the wrong graph correctly - it sees that no
path is left - and trusts it; or it misreads graphs in general and the
path-cutting ones happen to be where misreading shows. This probe separates
them: for every graph B6 showed (the correct one and each reversal, both
lexicons), the same graph block, the item's own names for X and Y, and one
question: is there a directed path from X to Y?

Every graph is replayed exactly as pilot.build_jobs drew it for B6 (relabel,
parse, enumerate over the name graph, choose with the per-(item, type, k) seed)
and proved by the API cache: the B6 prompt rebuilt from the replayed graph must
be a key that B6 really sent, or the script stops. X and Y are named through
the cladder-meta variable mapping (classify_perturbations.meta_mappings); an
item whose mappings disagree on which node is X or Y is left out.

The graph block is describe_graph's prose, the very sentences the B6 prompt
carried; the story, the numbers and the causal question are not shown.

Writes results/cladder/raw/probe_path_raw{tag}.csv, one row per (model, item,
lexicon, graph). scripts/analyze_path_probe.py reads it. `--models` and
`--tag` run the same probe on another family (prereg/B6_LLAMA.md:
--models meta-llama/llama-3.3-70b-instruct --tag _llama).
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

from classify_perturbations import (canonical_structures, descendants, meta_mappings,
                                    prose_graph)
from lexical import relabel_item
from pilot import ENUMERATORS, make_items, read_ids
from prompts import build, describe_graph
from runner import (CHARS_PER_TOKEN, PRICES_PER_M, _key, billed_usd, guard_errors, is_ok,
                    run_batch, usage_summary)

MODELS = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]
MAX_TOKENS = 300
# the B6 draw, exactly as prereg/B6.md and analyze_b6.B6 give it
B6 = dict(n=340, seed=20260926, sample_kmax=1, kmax=3,
          exclude="prereg/excluded_ids_b6.txt", query_types=["ate", "ett"])
RAWDIR = ROOT / "results" / "cladder" / "raw"
ANSWER = re.compile(r"ANSWER:\s*(yes|no)\b", re.IGNORECASE)

TEMPLATE = (
    "Here is a causal graph, given as statements of direct effects:\n{block}\n\n"
    "In this graph, is there a directed path from {x} to {y}? A directed path is a "
    "chain of one or more direct effects, each one pointing forward, that starts at "
    "{x} and ends at {y}.\n\n"
    "Think briefly, then answer on the last line in exactly this format:\n"
    "ANSWER: yes\nor\nANSWER: no")


def jobs():
    items = make_items(B6["n"], B6["seed"], B6["sample_kmax"], "full_v1.5_default.csv", None,
                       True, read_ids(B6["exclude"]), B6["query_types"])
    canon, maps = canonical_structures(), meta_mappings()
    out, skipped, missing = [], 0, 0
    for i, r in items.iterrows():
        sym = canon[r.graph_id]
        keep_prompt, _ = relabel_item(r.prompt, "KEEP", seed=str(r.id))
        keep_edges = prose_graph(keep_prompt)
        n2s_all = []
        for m in maps.get((r.story_id, r.graph_id), ()):
            s2n = dict(m)
            if (all(a in s2n and b in s2n for a, b in sym)
                    and sorted((s2n[a], s2n[b]) for a, b in sym) == sorted(keep_edges)):
                n2s_all.append({v: s for s, v in s2n.items()})
        for lex in ("KEEP", "PSEUDO"):
            prompt, clean = relabel_item(r.prompt, lex, seed=str(r.id))
            if not clean:
                continue
            edges = prose_graph(prompt)
            if len(edges) != len(sym) or not n2s_all:
                skipped += 1
                continue
            pos = {}
            for (a, b), (ka, kb) in zip(edges, keep_edges):
                pos.setdefault(a, ka)
                pos.setdefault(b, kb)
            nodes = sorted({n for e in edges for n in e})
            xy = {tuple(next(n for n in nodes if n2s[pos[n]] == s) for s in ("X", "Y"))
                  for n2s in n2s_all}
            if len(xy) != 1:
                skipped += 1              # the mappings disagree on which node is X or Y
                continue
            (x, y), = xy
            graphs = [("ORACLE", edges, build(prompt, "ORACLE", edges))]
            for k in range(1, B6["kmax"] + 1):
                opts = ENUMERATORS["DR"](edges, nodes, k)
                if opts:
                    bad = random.Random(f"{B6['seed']}:{i}:DR:{k}").choice(opts)
                    graphs.append((f"DR_k{k}", bad, build(prompt, "PERTURB", bad)))
            for cond, g, b6_prompt in graphs:
                if not _key("gpt-4.1-nano", 0.0, b6_prompt).exists():
                    missing += 1
                out.append(dict(item=i, id=int(r.id), query_type=r.query_type,
                                family=r.graph_id, lexicon=lex, cond=cond, x=x, y=y,
                                path_true=int(y in descendants(g, x)),
                                prompt=TEMPLATE.format(block=describe_graph(g), x=x, y=y)))
    if missing:
        raise SystemExit(f"{missing} replayed graphs are not the prompts B6 sent; stopping")
    print(f"  replay check: all {len(out)} graphs rebuild a B6 prompt found in the API cache;"
          f" {skipped} (item, lexicon) pairs left out")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    ap.add_argument("--pilot", type=int, default=0,
                    help="send this many prompts per model, into a separate cache, to "
                         "measure answer length; not data")
    ap.add_argument("--max-usd", type=float, default=None, dest="max_usd")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--tag", default="", help="output probe_path_raw{tag}.csv")
    ap.add_argument("--assume-out", type=float, default=None, dest="assume_out",
                    help="with --dry-run: mean output tokens per call, from the pilot")
    a = ap.parse_args()
    models = a.models.split(",")
    OUT = RAWDIR / f"probe_path_raw{a.tag}.csv"
    J = jobs()
    distinct = len({j["prompt"] for j in J})
    print(f"  {len(J)} (item, lexicon, graph) rows, {distinct} distinct prompts per model; "
          f"path present in {sum(j['path_true'] for j in J)}")
    if a.dry_run:
        tin = sum(len(p) for p in {j["prompt"] for j in J}) / CHARS_PER_TOKEN
        tot = cap = 0.0
        for m in models:
            pi, po = PRICES_PER_M[m]
            hi = (tin * pi + distinct * MAX_TOKENS * po) / 1e6
            cap += hi
            if a.assume_out is not None:
                est = (tin * pi + distinct * a.assume_out * po) / 1e6
                tot += est
                print(f"  {m:14s} estimate {est:.2f} USD, at most {hi:.2f}")
            else:
                print(f"  {m:14s} at most {hi:.2f} USD (every answer at the {MAX_TOKENS}-token cap)")
        if a.assume_out is not None:
            print(f"  ESTIMATE, nothing sent: {tot:.2f} USD")
        print(f"  UPPER BOUND, nothing sent: {cap:.2f} USD")
        return 0
    if a.pilot:
        P = [j for j in J if j["cond"] != "ORACLE"][:: max(1, len(J) // (4 * a.pilot))][:a.pilot]
        for m in models:
            recs = run_batch(P, m, temperature=0.0, workers=4, max_tokens=MAX_TOKENS,
                             cache=ROOT / "cache_probe_pilot")
            outs = [r["result"]["out_tok"] for r in recs if is_ok(r["result"])]
            print(f"  pilot {m:14s} out_tok {outs}  mean {sum(outs) / max(1, len(outs)):.1f}"
                  f"  usd {sum(billed_usd(m, r['result']) for r in recs):.4f}")
        return 0

    spent, rows = 0.0, []
    for m in models:
        left = None if a.max_usd is None else max(0.0, a.max_usd - spent)
        recs = run_batch(J, m, temperature=0.0, workers=a.workers, max_tokens=MAX_TOKENS,
                         max_usd=left)
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
            rows.append({k: r[k] for k in ("item", "id", "query_type", "family", "lexicon",
                                           "cond", "x", "y", "path_true")}
                        | dict(model=m, pred=pred, parsed=int(pred is not None),
                               correct=int(pred is not None
                                           and (pred == "yes") == bool(r["path_true"])),
                               in_tok=r["result"]["in_tok"], out_tok=r["result"]["out_tok"]))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"  wrote {OUT.relative_to(ROOT)}  ({len(rows)} rows, {spent:.2f} USD)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
