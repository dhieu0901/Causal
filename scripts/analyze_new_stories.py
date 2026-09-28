"""CLadder's questions told in new stories (GPT-4.1 family and Llama 3.3 70B).

    python scripts/analyze_new_stories.py

Pre-registered in prereg/NEW_STORIES.md, committed and pushed before the first
call. Study 1's 484 questions, under RAW, in one new condition: NEWSTORY, the
question told in a new story (src/new_stories.py) with the same graph, numbers,
query and label. KEEP and SYMBOL under RAW are Study 1's own answers.

Why. Real names beat letters by 11 to 13 points on CLadder (H3b). CLadder's text
has been public since 2023, so part of that gain could be recall of the
benchmark's own wording. New stories are text no CLadder question contains.

Tests, per family (Holm across the three, alpha 0.05; stories resampled with
all their questions, 4,000 draws, five seeds, a verdict that changes with the
seed is borderline). Every difference is taken per (model, question) on parsed
answers and averaged over every cell, as H3b is.
  S1  NEWSTORY - SYMBOL                                             (+)
  S2  NEWSTORY - KEEP, within +-5 points (two one-sided tests; 90% interval)
  S3  balanced accuracy of NEWSTORY - SYMBOL, commonsense minus
      anticommonsense questions, within +-5 points (TOST; 90% interval)
Sensitivity: questions resampled; unparsed answers scored as wrong; for Llama,
answers cut off at 700 tokens re-asked at 1,500. Descriptive: every cell's
accuracy, Yes share and balanced accuracy; S1 and S2 on gold-Yes and gold-No
questions and as a change in the share of Yes answers; S1 and S2 by model.

Before it reads a new file the script checks that it reproduces each family's
published H3b (results/cladder/confirmatory.csv). With no new records it says so
and writes nothing.

Writes results/cladder/new_stories.csv and new_stories_cells.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from analyze_confirmatory import ALPHA, LLAMA, MARGIN, SEEDS, TIER, holm, with_recap
from analyze_querygroup import ARITH, IDENT
from stats import boot_p, cluster_boot

RESULTS = ROOT / "results" / "cladder"
RAW = RESULTS / "raw"
NBOOT = 4000
PLAUSIBLE = {"commonsense", "easy", "hard"}
FAMILIES = {"gpt": (list(TIER), "conf", None), "llama": ([LLAMA], "llamaconf", 1500)}
FILES = {"KEEP": "pilot_raw_{p}KEEP.csv", "SYMBOL": "pilot_raw_{p}ladderSYMBOL.csv",
         "NEWSTORY": "pilot_raw_{p}NEWSTORY.csv"}
TESTS = [("S1", "NEWSTORY - SYMBOL under RAW", "two", +1),
         ("S2", "NEWSTORY - KEEP under RAW, within +-5", "tost", 0),
         ("S3", "balanced (NEWSTORY - SYMBOL), commonsense minus anticommonsense, within +-5",
          "tost", 0)]


def load(prefix, models, mode, recap):
    """lexicon -> rows (model, item) with correct, yes, gold, story, sense; RAW, parsed."""
    out = {}
    for lex, f in FILES.items():
        p = RAW / f.format(p=prefix)
        if not p.exists():
            return None
        d = pd.read_csv(p)
        if mode == "re-asked at 1500":
            d = with_recap(d, p.with_name(p.stem + f"_recap{recap}.csv"))
            if d is None:
                raise SystemExit(f"{p.name}: its re-ask file is missing")
        d = d[d.model.isin(models) & (d.cond == "RAW") & ~d.query_type.isin(ARITH | IDENT)]
        if mode == "unparsed as wrong":
            d = d.assign(correct=np.where(d.parsed == 1, d.correct, 0), parsed=1)
        d = d[d.parsed == 1]
        d = d.assign(yes=(d.pred.astype(str).str.lower() == "yes").astype(float),
                     sense=np.where(d.question_property.isin(PLAUSIBLE), "commonsense",
                                    d.question_property))
        out[lex] = d.drop_duplicates(["model", "item"]).set_index(["model", "item"])
    return out


def diff(cells, a, b, col):
    """a - b per (model, item) answered in both, with the item's labels."""
    x, y = cells[a][col], cells[b][col]
    i = x.index.intersection(y.index)
    meta = cells[a].loc[i, ["gold", "story_id", "sense"]]
    return meta.assign(v=(x[i] - y[i]).values).reset_index()


def draws(x, stat, stories, seed):
    """stat(rows) with stories resampled (or questions, when stories=False)."""
    unit = x.story_id.to_numpy() if stories else x.item.to_numpy()
    keys = sorted(set(unit))
    idx = [np.flatnonzero(unit == k) for k in keys]
    return stat(np.arange(len(x))), cluster_boot(
        len(idx), lambda i: stat(np.concatenate([idx[j] for j in i])), seed, NBOOT)


def mean_of(x):
    v = x.v.to_numpy(dtype=float)
    return lambda r: 100 * np.nanmean(v[r])


def balanced_gap(x):
    v, g, s = x.v.to_numpy(dtype=float), x.gold.to_numpy(), x.sense.to_numpy()

    def bal(r, grp):
        m = s[r] == grp
        return 50 * (np.nanmean(v[r][m & (g[r] == "yes")]) + np.nanmean(v[r][m & (g[r] == "no")]))
    return lambda r: bal(r, "commonsense") - bal(r, "anticommonsense")


def tost_p(d):
    return max(max(float((d <= -MARGIN).mean()), 1 / NBOOT),
               max(float((d >= MARGIN).mean()), 1 / NBOOT))


def run_tests(cells, stories=True):
    s1, s2 = diff(cells, "NEWSTORY", "SYMBOL", "correct"), diff(cells, "NEWSTORY", "KEEP", "correct")
    stats = {"S1": (s1, mean_of(s1)), "S2": (s2, mean_of(s2)), "S3": (s1, balanced_gap(s1))}
    per_seed = {}
    for seed in SEEDS:
        res = {}
        for t, what, kind, sign in TESTS:
            x, f = stats[t]
            e, d = draws(x, f, stories, seed)
            d = d[np.isfinite(d)]
            r = dict(estimate_pp=e, ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5),
                     p_boot=tost_p(d) if kind == "tost" else boot_p(d, NBOOT), n_items=x.item.nunique(),
                     n_stories=x.story_id.nunique())
            if kind == "tost":
                r |= dict(ci90_lo=np.percentile(d, 5), ci90_hi=np.percentile(d, 95))
            res[t] = r
        adj = holm([res[t]["p_boot"] for t, *_ in TESTS])
        for (t, _, kind, sign), pa in zip(TESTS, adj):
            if kind == "tost":
                v = "confirmed" if pa < ALPHA else "not confirmed"
            elif pa >= ALPHA:
                v = "not confirmed"
            else:
                v = "confirmed" if np.sign(res[t]["estimate_pp"]) == sign else "contradicted"
            res[t] |= dict(p_holm=pa, verdict=v)
        per_seed[seed] = res
    return per_seed


def rows_of(family, analysis, per_seed):
    out, prim = [], per_seed[SEEDS[0]]
    for t, what, kind, sign in TESTS:
        r = prim[t]
        vs = {per_seed[s][t]["verdict"] for s in SEEDS}
        ps = [per_seed[s][t]["p_boot"] for s in SEEDS]
        out.append(dict(family=family, analysis=analysis, test=t, quantity=what,
                        predicted={"two": "positive", "tost": "within +-5"}[kind],
                        estimate_pp=round(r["estimate_pp"], 2), ci_lo=round(r["ci_lo"], 2),
                        ci_hi=round(r["ci_hi"], 2),
                        ci90_lo=round(r["ci90_lo"], 2) if "ci90_lo" in r else None,
                        ci90_hi=round(r["ci90_hi"], 2) if "ci90_hi" in r else None,
                        p_boot=round(r["p_boot"], 4), p_holm=round(r["p_holm"], 4),
                        p_min_seeds=round(min(ps), 4), p_max_seeds=round(max(ps), 4),
                        n_items=r["n_items"], n_stories=r["n_stories"],
                        verdict=vs.pop() if len(vs) == 1 else "borderline (changes with the bootstrap seed)"))
    return out


def check_h3b(family, cells):
    conf = pd.read_csv(RESULTS / "confirmatory.csv")
    want = conf[(conf.analysis == "primary") & (conf.family == family) & (conf.test == "H3b")].estimate_pp.iloc[0]
    got = round(100 * np.nanmean(diff(cells, "KEEP", "SYMBOL", "correct").v), 2)
    if got != want:
        raise SystemExit(f"{family}: KEEP - SYMBOL is {got}, not the published H3b {want}")
    print(f"  {family}: check, KEEP - SYMBOL reproduces H3b ({got})")


def descriptive(family, cells, models):
    out = []

    def add(what, part, x, model="family"):
        if len(x) < 10:
            return
        e, d = draws(x, mean_of(x), True, SEEDS[0])
        out.append(dict(family=family, analysis="descriptive", test=what, quantity=part, model=model,
                        estimate_pp=round(e, 2), ci_lo=round(np.percentile(d, 2.5), 2),
                        ci_hi=round(np.percentile(d, 97.5), 2), p_boot=round(boot_p(d, NBOOT), 4),
                        n_items=x.item.nunique(), n_stories=x.story_id.nunique()))

    for what, a, b in (("NEWSTORY - SYMBOL", "NEWSTORY", "SYMBOL"), ("NEWSTORY - KEEP", "NEWSTORY", "KEEP"),
                       ("KEEP - SYMBOL", "KEEP", "SYMBOL")):
        x, y = diff(cells, a, b, "correct"), diff(cells, a, b, "yes")
        add(what, "gold yes", x[x.gold == "yes"])
        add(what, "gold no", x[x.gold == "no"])
        add(what, "says yes", y)
        for g in ("commonsense", "anticommonsense"):
            add(what, f"{g}", x[x.sense == g])
            add(what, f"{g}, says yes", y[y.sense == g])
        if len(models) > 1:
            for m in models:
                add(what, "all", x[x.model == m], model=m)
    return out


def cell_rows(family, cells):
    out = []
    for lex, d in cells.items():
        ay, an = d[d.gold == "yes"].correct.mean(), d[d.gold == "no"].correct.mean()
        out.append(dict(family=family, lexicon=lex, n=len(d), accuracy_pct=round(100 * d.correct.mean(), 2),
                        says_yes_pct=round(100 * d.yes.mean(), 2), balanced_pct=round(50 * (ay + an), 2)))
    return out


def main() -> int:
    rows, cells_out = [], []
    for family, (models, prefix, recap) in FAMILIES.items():
        cells = load(prefix, models, "primary", recap)
        if cells is None:
            print(f"  {family}: no records yet ({FILES['NEWSTORY'].format(p=prefix)}; "
                  f"scripts/run_new_stories.sh)")
            continue
        check_h3b(family, cells)
        rows += rows_of(family, "primary", run_tests(cells))
        rows += rows_of(family, "questions resampled", run_tests(cells, stories=False))
        rows += rows_of(family, "unparsed as wrong", run_tests(load(prefix, models, "unparsed as wrong", recap)))
        if recap:
            rows += rows_of(family, "re-asked at 1500", run_tests(load(prefix, models, "re-asked at 1500", recap)))
        rows += descriptive(family, cells, models)
        cells_out += cell_rows(family, cells)
    if not rows:
        return 0
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "new_stories.csv", index=False)
    C = pd.DataFrame(cells_out)
    C.to_csv(RESULTS / "new_stories_cells.csv", index=False)
    with pd.option_context("display.width", 230, "display.max_rows", 300, "display.max_columns", 20):
        print(C.to_string(index=False))
        print()
        print(R.to_string(index=False))
    print("\n  wrote results/cladder/new_stories.csv, new_stories_cells.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
