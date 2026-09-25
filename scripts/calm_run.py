"""Ask a model CaLM's ATE items under the graph conditions of the CLadder study.

    python scripts/calm_run.py --models meta-llama/llama-3.3-70b-instruct \
        --lexicon PSEUDO --conds RAW,ORACLE,DR_k1 --tag _llamaPSEUDO --dry-run

The second benchmark (src/calm.py, prereg/CALM.md). Items: CaLM's REAL-mode ATE
questions - real names on a story that makes sense - whose edge sentences parse
back to the item's own graph under every lexicon (the rule pilot.build_jobs
applies to CLadder). Conditions are built by src/prompts.py exactly as for
CLadder:

    RAW     the edge sentences removed; probabilities, instruction and question kept
    ORACLE  RAW plus the structure block over the item's names
    DR_k1   RAW plus a structure block with one edge reversed, acyclic, drawn per
            item with random.Random(f"{seed}:{item}:DR:1") as in pilot.py

The gold answer is the true world's under every condition. Writes
results/calm/raw/calm_raw{tag}.csv, and with --recap a re-ask of every answer cut off
at the token cap to results/calm/raw/calm_raw{tag}_recap{N}.csv. SPENDS CREDIT unless
--dry-run.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

import calm
from perturb import ENUMERATORS
from prompts import build, parse_answer, parse_prose_graph, strip_structure
from runner import (CHARS_PER_TOKEN, OPENROUTER, PRICES_PER_M, billed_usd, guard_errors,
                    is_ok, read_cached, run_batch, usage_summary)

SEED = 20260925


def items() -> list[dict]:
    """The REAL-mode items whose graph parses back under both lexicons."""
    full = calm.load(calm.FULL)
    freq = calm.name_freq(full)
    out = []
    for it in full:
        if calm.mode(it, freq) != "REAL" or calm.gold(it) is None:
            continue
        ok = True
        for lex in ("KEEP", "PSEUDO"):
            p, clean = calm.relabel(it, lex, seed=str(it["index"]))
            if not clean or len(parse_prose_graph(strip_structure(p)[1])) != len(calm.sym_edges(it)):
                ok = False
        if ok:
            out.append(it)
    return sorted(out, key=lambda it: it["index"])


def jobs_for(its, lexicon, conds, seed=SEED) -> list[dict]:
    jobs = []
    for it in its:
        p, _ = calm.relabel(it, lexicon, seed=str(it["index"]))
        edges = parse_prose_graph(strip_structure(p)[1])
        nodes = sorted({n for e in edges for n in e})
        to = calm._treatment_outcome(it)
        meta = dict(item=int(it["index"]), story=calm.story(it), lexicon=lexicon,
                    gold=calm.gold(it), n_nodes=len(calm.node_names(it)),
                    has_data=bool(it["Background"]["data_info"].strip()),
                    path=calm._path(calm.sym_edges(it), *to))
        for c in conds:
            if c == "RAW":
                pr = build(p, "RAW")
            elif c == "ORACLE":
                pr = build(p, "ORACLE", edges)
            elif c == "DR_k1":
                bad = random.Random(f"{seed}:{it['index']}:DR:1").choice(
                    ENUMERATORS["DR"](edges, nodes, 1))
                pr = build(p, "PERTURB", bad)
            else:
                raise SystemExit(f"unknown condition {c}")
            jobs.append(meta | {"cond": c, "prompt": pr})
    return jobs


def mean_out_tokens(model) -> float | None:
    """This model's mean output tokens over every CLadder record on disk: the price
    of an answer depends on the model far more than on the benchmark."""
    vals = []
    for f in (ROOT / "results" / "cladder" / "raw").glob("pilot_raw*.csv"):
        d = pd.read_csv(f, usecols=lambda c: c in {"model", "out_tok"})
        if "out_tok" in d:
            vals.append(d.loc[d.model == model, "out_tok"])
    s = pd.concat(vals) if vals else pd.Series(dtype=float)
    return float(s.mean()) if len(s) else None


def estimate(jobs, model, temperature, max_tokens):
    prompts = list(dict.fromkeys(j["prompt"] for j in jobs))
    new = [p for p in prompts if read_cached(model, temperature, p, max_tokens=max_tokens) is None]
    out_tok = mean_out_tokens(model)
    if model not in PRICES_PER_M or out_tok is None:
        return len(prompts), len(new), None
    pi, po = PRICES_PER_M[model]
    usd = sum(len(p) / CHARS_PER_TOKEN * pi + out_tok * po for p in new) / 1e6
    return len(prompts), len(new), usd


def rows_of(recs, model, cap):
    rows = []
    for r in recs:
        res = r["result"]
        pred = parse_answer(res.get("text", "")) if is_ok(res) else None
        rows.append({k: r[k] for k in ("item", "story", "cond", "lexicon", "gold",
                                        "n_nodes", "has_data", "path")}
                    | {"model": model, "pred": pred, "parsed": int(pred is not None),
                       "correct": int(pred == r["gold"]), "in_tok": res.get("in_tok", 0),
                       "out_tok": res.get("out_tok", 0), "error": res.get("error", "")}
                    | ({k: res.get(k, "") for k in ("provider", "finish", "usd")}
                       | {"max_tokens": cap} if model in OPENROUTER else {}))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--lexicon", choices=["KEEP", "PSEUDO"], required=True)
    ap.add_argument("--conds", default="RAW,ORACLE,DR_k1")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=700, dest="max_tokens")
    ap.add_argument("--recap", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-usd", type=float, default=None, dest="max_usd",
                    help="for the whole run, all models and re-asks together")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    a = ap.parse_args()

    its = items()
    jobs = jobs_for(its, a.lexicon, [c.strip() for c in a.conds.split(",")])
    print(f"items={len(its)}  stories={len({j['story'] for j in jobs})}  "
          f"jobs/model={len(jobs)}  lexicon={a.lexicon}")
    if a.dry_run:
        tot = 0.0
        for m in a.models.split(","):
            n, new, usd = estimate(jobs, m.strip(), a.temperature, a.max_tokens)
            print(f"  {m:36s} distinct={n:5d} new={new:5d}  usd={usd if usd is None else round(usd, 2)}")
            tot += usd or 0.0
        print(f"  ESTIMATE, nothing sent: {tot:.2f} USD")
        return 0

    spent = [0.0]
    budget = lambda: None if a.max_usd is None else max(0.0, a.max_usd - spent[0])

    def charge(recs, model):
        spent[0] += sum(billed_usd(model, r["result"]) for r in recs
                        if not r["result"].get("cached"))
        if any("USD cap" in r["result"].get("error", "") for r in recs):
            print(f"    SPENDING CAP REACHED: {spent[0]:.2f} USD spent, --max-usd {a.max_usd}")

    allrows, recaprows = [], []
    for model in (m.strip() for m in a.models.split(",")):
        print(f"\n>>> {model}: {len(jobs)} calls", flush=True)
        recs = run_batch(jobs, model, temperature=a.temperature, workers=a.workers,
                         max_tokens=a.max_tokens, max_usd=budget(),
                         on_tick=lambda d, t: print(f"    {d}/{t}", flush=True))
        charge(recs, model)
        print(f"    done: {usage_summary(recs)}   spent so far {spent[0]:.2f} USD", flush=True)
        guard_errors(recs, label=model)
        allrows += rows_of(recs, model, a.max_tokens)
        if a.recap:
            cut = [{k: v for k, v in r.items() if k != "result"} for r in recs
                   if r["result"].get("finish") == "length"]
            print(f"\n>>> {model}: re-asking the {len(cut)} answers cut off at "
                  f"{a.max_tokens} tokens, with {a.recap}", flush=True)
            if cut:
                rec2 = run_batch(cut, model, temperature=a.temperature, workers=a.workers,
                                 max_tokens=a.recap, max_usd=budget())
                charge(rec2, model)
                guard_errors(rec2, label=f"{model} recap")
                recaprows += rows_of(rec2, model, a.recap)

    out = ROOT / "results" / "calm" / "raw" / f"calm_raw{a.tag}.csv"
    pd.DataFrame(allrows).sort_values(["model", "item", "cond"]).to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}  ({len(allrows)} rows, {spent[0]:.2f} USD)")
    if a.recap:
        out2 = out.with_name(f"calm_raw{a.tag}_recap{a.recap}.csv")
        cols = list(allrows[0]) if allrows else []
        pd.DataFrame(recaprows, columns=cols).to_csv(out2, index=False)
        print(f"wrote {out2.relative_to(ROOT)}  ({len(recaprows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
