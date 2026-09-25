"""Has the served model drifted between the n600 runs? Ask old prompts again.

    python scripts/check_drift.py --dry-run      # cost, nothing sent
    python scripts/check_drift.py                # SPENDS CREDIT

Why this file exists. n600's RAW, PROSE, ORACLE and DR_k1 were answered around
2026-09-16. The conditions scripts/run_n600_extensions.sh added - DR_k2, DR_k3,
NAMES_ONLY, SCRAMBLE, and the PERMUTE / IRRELEVANT / SYMBOL rungs - were
answered on 2026-09-24. Every contrast between the two runs (k=1 against k=2,
ORACLE against NAMES_ONLY, KEEP against IRRELEVANT) assumes the model answering
on the 24th is the model that answered on the 16th. The API serves a model
alias, the cache hides any change, and nothing else in the repository can see
one. analyze_structure_arms.py and analyze_ladder5.py label every contrast
"same run" or "crosses runs"; this is the test the second label waits on.

The test. Send prompts the OLD run already answered, again, into a separate
cache (cache_drift/, gitignored) so the main cache cannot answer them, and
compare within item: accuracy now minus accuracy then, per model, cluster
bootstrap over items. A difference that separates from zero is drift, and the
cross-run contrasts must then be re-run in one sitting rather than corrected by
subtraction - drift need not be the same for every condition.

Temperature 0 is not deterministic on this API, so some answers flip with no
drift at all. Flips in both directions cancel in the accuracy difference; the
flip rate is reported beside it so the two are not confused.

Result, 2026-09-24 (5.54 USD). No drift: accuracy now minus then is -0.38 pp
[-1.33, +0.55] over all 5,256 cells, and no single condition or model separates
from zero. But 15.0% of answers flipped on re-asking (22.4% nano, 12.6% mini,
10.1% gpt-4.1): temperature 0 is far from deterministic on this API, so a
single run's per-item correctness carries noise that paired designs average
out and that no cache can reveal.

Writes: results/drift_check.csv, results/drift_check_raw.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from pilot import make_items, build_jobs
from prompts import parse_answer
from runner import run_batch, guard_errors, is_ok, PRICES_PER_M, read_cached
from stats import boot_items

SEED = 20260907
NBOOT = 4000
TIER = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]
ARITH = {"marginal", "correlation"}
IDENT = {"backadj"}
DRIFT_CACHE = ROOT / "cache_drift"

# How each sample was drawn (pool_samples.SAMPLES) and which file holds its old
# answers. The n600 check came first and keeps its unsuffixed output names.
SAMPLE_ARGS = {"n600": (600, 1, ("DR",)), "price400": (400, 3, ("DR", "ED", "FE")),
               "lex": (200, 1, ("DR",))}


def jobs_for(sample, lexicon, conds):
    """The old run's own jobs, rebuilt exactly, causal query group only."""
    n, kmax, types = SAMPLE_ARGS[sample]
    items = make_items(n, SEED, kmax, "full_v1.5_default.csv", None, True)
    jobs = build_jobs(items, kmax, SEED, types, lexicon)
    return [j | {"lexicon": lexicon} for j in jobs
            if j["cond"] in conds and j["query_type"] not in ARITH | IDENT]


def old_rows(sample, lexicon):
    d = pd.read_csv(ROOT / "results" / "raw" / f"pilot_raw_{sample}{lexicon}.csv")
    return d.set_index(["model", "cond", "item"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="n600", choices=sorted(SAMPLE_ARGS))
    ap.add_argument("--conds", default="RAW,ORACLE,DR_k1")
    ap.add_argument("--lexicons", default="KEEP,PSEUDO")
    ap.add_argument("--models", default=",".join(TIER))
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    a = ap.parse_args()
    conds = {c.strip() for c in a.conds.split(",")}
    lexes = [x.strip() for x in a.lexicons.split(",")]
    models = [m.strip() for m in a.models.split(",")]

    jobs = [j for lx in lexes for j in jobs_for(a.sample, lx, conds)]
    old = {lx: old_rows(a.sample, lx) for lx in lexes}
    sfx = "" if a.sample == "n600" else f"_{a.sample}"
    # Every job must be one the old run really sent: it is in the main cache.
    miss = sum(read_cached(models[0], 0.0, j["prompt"]) is None for j in jobs)
    if miss:
        raise SystemExit(f"{miss} rebuilt prompts are not in the main cache; this is "
                         f"not the old run's sample. Stopping before spending anything.")
    print(f"{len(jobs)} prompts per model ({sorted(conds)} x {lexes}, causal group), "
          f"all verified as sent in the old run.")

    # Cost at the SAME prompt's token counts in the old run.
    total = 0.0
    for m in models:
        pi, po = PRICES_PER_M[m]
        usd, fresh = 0.0, 0
        for j in jobs:
            if read_cached(m, 0.0, j["prompt"], DRIFT_CACHE) is not None:
                continue
            r = old[j["lexicon"]].loc[(m, j["cond"], j["item"])]
            usd += (r.in_tok * pi + r.out_tok * po) / 1e6
            fresh += 1
        total += usd
        print(f"  {m:14s} new calls={fresh:5d}  usd={usd:.2f}")
    print(f"  TOTAL {total:.2f} USD at the old run's token counts")
    if a.dry_run:
        return 0

    DRIFT_CACHE.mkdir(exist_ok=True)
    rows = []
    for m in models:
        print(f"\n>>> {m}")
        recs = run_batch(jobs, m, cache=DRIFT_CACHE,
                         on_tick=lambda d, t: print(f"    {d}/{t}", flush=True))
        guard_errors(recs, label=m)
        for r in recs:
            if not is_ok(r["result"]):
                continue
            o = old[r["lexicon"]].loc[(m, r["cond"], r["item"])]
            pred = parse_answer(r["result"]["text"])
            rows.append(dict(model=m, lexicon=r["lexicon"], cond=r["cond"], item=r["item"],
                             id=r["id"],
                             old_pred=o.pred, new_pred=pred,
                             old_correct=int(o.correct),
                             new_correct=int(pred == r["gold"]) if pred else 0))
    R = pd.DataFrame(rows)
    R.to_csv(ROOT / "results" / f"drift_check_raw{sfx}.csv", index=False)

    out = []
    groups = [(lx, c) for lx in lexes for c in sorted(conds)] + [("all", "all")]
    for lx, c in groups:
        s = R if lx == "all" else R[(R.lexicon == lx) & (R.cond == c)]
        for m in models + ["pooled"]:
            t = s if m == "pooled" else s[s.model == m]
            # one value per item (mean over its cells), bootstrap over items
            v = (t.new_correct - t.old_correct).groupby(t.item).mean()
            x = v.values
            est, lo, hi, p = boot_items(x, SEED, NBOOT)      # convention A, src/stats.py
            flips = (t.old_pred.fillna("").astype(str)
                     != t.new_pred.fillna("").astype(str)).mean()
            out.append(dict(lexicon=lx, cond=c, model=m, n_cells=len(t),
                            acc_old=round(100 * t.old_correct.mean(), 2),
                            acc_new=round(100 * t.new_correct.mean(), 2),
                            drift_pp=round(est, 2), ci_lo=round(lo, 2),
                            ci_hi=round(hi, 2), p_boot=round(p, 4),
                            flip_pct=round(100 * flips, 1)))
    D = pd.DataFrame(out)
    D.to_csv(ROOT / "results" / f"drift_check{sfx}.csv", index=False)
    print("\n" + D.to_string(index=False))
    print("\n  drift_pp = accuracy now minus accuracy then, same prompt, same item.")
    print("  flip_pct counts answers that changed in either direction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
