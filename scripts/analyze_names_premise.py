"""The closed-world clause, and which names carry the gain (Llama 3.3 70B,
gpt-5.6-luna, the GPT-4.1 family).

    python scripts/analyze_names_premise.py

Pre-registered in prereg/NAMES_PREMISE.md (Llama) and prereg/FOLLOWUPS_FAMILIES.md
(gpt-5.6-luna and the GPT-4.1 family), each committed and pushed before its
first call. Study 1's 484 questions. The RAW answers under KEEP and PSEUDO are
Study 1's own (pilot_raw_{llamaconf,lunaconf,conf}*); the new cells come from
scripts/run_names_premise.sh (Llama) and scripts/run_followups_families.sh:

  RAW_OPEN      RAW without ", and without any unmentioned factors or causal
                relationships" in CLadder's opening sentence (src/prompts.py),
                under KEEP and PSEUDO
  PSEUDO_XY     RAW with PSEUDO's own pseudowords on treatment and outcome only
  PSEUDO_THIRD  RAW with PSEUDO's own pseudowords on the other variables only
                (src/lexical.py)

Why. CLadder's opening sentence says the world holds no causal relationship
beyond those stated. RAW removes every stated one, so read literally a RAW
question implies "no effect", and with pseudowords Llama says Yes to 28% of
Study 1's questions where 51% of the labels are Yes. And real names help, but
which names: those of treatment and outcome, or those of the other variables,
which can signal their role?

Tests, per family (Holm across the four, alpha 0.05; stories resampled with all
their questions, 4,000 draws, five seeds; a verdict that changes with the seed
is borderline). Every difference is taken per (model, question), parsed answers
only, and averaged over every cell (one model per question for Llama and
gpt-5.6-luna, three for the GPT-4.1 family).
  P1  share of Yes answers under PSEUDO, RAW_OPEN minus RAW            (+)
  P2  (KEEP - PSEUDO | RAW) - (KEEP - PSEUDO | RAW_OPEN), accuracy       (+)
  N1  KEEP - PSEUDO_XY under RAW, accuracy                              (+)
  N2  KEEP - PSEUDO_THIRD under RAW, accuracy, on the questions whose RAW
      prompt names another variable at all                            (+)
On 150 of the 484 questions the other variables appear only in the
sentences of structure that RAW removes, so their RAW prompt is the same
under KEEP and PSEUDO_THIRD; N2 leaves them out, and is also reported on
every question (descriptive). Sensitivity: questions resampled; unparsed
answers scored as wrong; for Llama, answers cut off at 700 tokens re-asked at
1,500. Descriptive: every cell's accuracy, Yes share and balanced accuracy, and
each test on gold-Yes and gold-No questions and as a change in the share of Yes
answers.

Before it reads a family's new files the script checks that it reproduces that
family's Study 1 RAW cells (accuracy_cells.csv). A family with no new records
is skipped.

Writes results/cladder/names_premise.csv and names_premise_cells.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from analyze_confirmatory import ALPHA, LLAMA, SEEDS, TIER, holm, with_recap
from analyze_querygroup import ARITH, IDENT
from stats import boot_p, cluster_boot

RESULTS = ROOT / "results" / "cladder"
RAW = RESULTS / "raw"
NBOOT = 4000
LUNA = "gpt-5.6-luna"
# family -> (models, prefix of Study 1's files, prefix of the new files, re-ask cap)
FAMILIES = {"llama": ([LLAMA], "llamaconf", "llamapremise", 1500),
            "luna": ([LUNA], "lunaconf", "lunapremise", None),
            "gpt": (list(TIER), "conf", "premise", None)}
FILES = {("KEEP", "RAW"): "{s}KEEP", ("PSEUDO", "RAW"): "{s}PSEUDO",
         ("KEEP", "RAW_OPEN"): "{n}KEEP", ("PSEUDO", "RAW_OPEN"): "{n}PSEUDO",
         ("PSEUDO_XY", "RAW"): "{n}PSEUDO_XY", ("PSEUDO_THIRD", "RAW"): "{n}PSEUDO_THIRD"}
TESTS = [("P1", "share of Yes answers under PSEUDO, RAW_OPEN minus RAW", +1),
         ("P2", "(KEEP - PSEUDO | RAW) - (KEEP - PSEUDO | RAW_OPEN), accuracy", +1),
         ("N1", "KEEP - PSEUDO_XY under RAW, accuracy", +1),
         ("N2", "KEEP - PSEUDO_THIRD under RAW, accuracy, questions naming another variable", +1)]
DRAW = dict(n=1000, seed=20260925, sample_kmax=1, data="full_v1.5_default.csv",
            exclude="prereg/excluded_ids.txt")            # Study 1's draw, every family


def third_shown() -> set:
    """Questions whose RAW prompt changes when only the other variables are
    anonymised, i.e. whose RAW text names another variable at all."""
    from pilot import build_jobs, make_items, read_ids
    items = make_items(DRAW["n"], DRAW["seed"], DRAW["sample_kmax"], DRAW["data"], None, True,
                       read_ids(DRAW["exclude"]), None)
    items = items[~items.query_type.isin(ARITH | IDENT)]
    raw = lambda lex: {j["item"]: j["prompt"] for j in build_jobs(items, 3, DRAW["seed"], ("DR",), lex)
                       if j["cond"] == "RAW"}
    k, t = raw("KEEP"), raw("PSEUDO_THIRD")
    return {i for i in k if i in t and k[i] != t[i]}
K, P, KO, PO = ("KEEP", "RAW"), ("PSEUDO", "RAW"), ("KEEP", "RAW_OPEN"), ("PSEUDO", "RAW_OPEN")
XY, TH = ("PSEUDO_XY", "RAW"), ("PSEUDO_THIRD", "RAW")


def path_of(family, key):
    models, s, n, recap = FAMILIES[family]
    return RAW / f"pilot_raw_{FILES[key].format(s=s, n=n)}.csv"


def load(family, mode):
    """(lexicon, cond) -> one row per question (per (model, question) for the
    GPT-4.1 family): correct, yes, gold, story."""
    models, s, n, recap = FAMILIES[family]
    cells = {}
    for key in FILES:
        p = path_of(family, key)
        if not p.exists():
            return None
        d = pd.read_csv(p)
        if mode == "re-asked at 1500":
            d = with_recap(d, p.with_name(p.stem + f"_recap{recap}.csv"))
            if d is None:
                raise SystemExit(f"{p.name}: its re-ask file is missing")
        d = d[d.model.isin(models) & (d.cond == key[1]) & ~d.query_type.isin(ARITH | IDENT)]
        if mode == "unparsed as wrong":
            d = d.assign(correct=np.where(d.parsed == 1, d.correct, 0), parsed=1)
        d = d[d.parsed == 1]
        d = d.assign(yes=(d.pred.astype(str).str.lower() == "yes").astype(float))
        if len(models) == 1:
            cells[key] = d.drop_duplicates("item").set_index("item")
        else:
            cells[key] = d.drop_duplicates(["model", "item"]).set_index(["model", "item"])
    return cells


def item_of(index):
    """The question of each row: the index itself, or its `item` level."""
    return index.get_level_values("item") if isinstance(index, pd.MultiIndex) else index


def check_study1(family, cells):
    """The RAW cells under KEEP and PSEUDO must be Study 1's published ones."""
    ac = pd.read_csv(RESULTS / "accuracy_cells.csv")
    models = FAMILIES[family][0]
    ac = ac[(ac.benchmark == "CLadder B5") & (ac.family == family) & (ac.cond == "RAW")
            & (ac.model == (models[0] if len(models) == 1 else "all"))].set_index("lexicon")
    for key in (K, P):
        c = cells[key]
        got = (len(c), round(100 * c.correct.mean(), 2), round(100 * c.yes.mean(), 2))
        want = (int(ac.loc[key[0], "n"]), ac.loc[key[0], "accuracy_pct"], ac.loc[key[0], "says_yes_pct"])
        if got != want:
            raise SystemExit(f"{family} {key}: {got} is not Study 1's {want}; the loader is wrong")
    print(f"  {family}: check, the KEEP and PSEUDO RAW cells reproduce Study 1 "
          f"({len(cells[K])} and {len(cells[P])} rows)")


SHOWN: set = set()


def vectors(cells, col, every=False):
    """Per-row differences for the four tests on column col; every=True
    takes N2 over every question."""
    c = {k: v[col] for k, v in cells.items()}
    both = lambda a, b: a.index.intersection(b.index)
    i1 = both(c[PO], c[P])
    i2 = c[K].index.intersection(c[P].index).intersection(c[KO].index).intersection(c[PO].index)
    i3, i4 = both(c[K], c[XY]), both(c[K], c[TH])
    if not every:
        i4 = i4[item_of(i4).isin(SHOWN)]
    return {"P1": (c[PO][i1] - c[P][i1]) if col == "yes" else None,
            "P2": (c[K][i2] - c[P][i2]) - (c[KO][i2] - c[PO][i2]),
            "N1": c[K][i3] - c[XY][i3],
            "N2": c[K][i4] - c[TH][i4]}


def draws(v, unit_of, seed):
    """Mean of v, resampling units with all their rows: stories (unit_of maps a
    question to its story), or questions (unit_of None; one row per question
    for a one-model family, as registered for Llama)."""
    if unit_of is not None:
        lab = np.asarray(item_of(v.index).map(unit_of))
    elif isinstance(v.index, pd.MultiIndex):
        lab = np.asarray(item_of(v.index))
    else:
        lab = np.arange(len(v))
    units = sorted(set(lab))
    idx = [np.flatnonzero(lab == u) for u in units]
    x = v.to_numpy(dtype=float)
    d = cluster_boot(len(idx), lambda i: 100 * np.nanmean(x[np.concatenate([idx[j] for j in i])]),
                     seed, NBOOT)
    return 100 * np.nanmean(x), d, len(units)


def run_tests(cells, story_of, stories=True):
    vy, vc = vectors(cells, "yes"), vectors(cells, "correct")
    V = {"P1": vy["P1"], "P2": vc["P2"], "N1": vc["N1"], "N2": vc["N2"]}
    per_seed = {}
    for seed in SEEDS:
        res = {}
        for t, _, _ in TESTS:
            e, d, nu = draws(V[t], story_of if stories else None, seed)
            res[t] = dict(estimate_pp=e, ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5),
                          p_boot=boot_p(d, NBOOT), n_items=len(set(item_of(V[t].index))),
                          n_stories=len(set(item_of(V[t].index).map(story_of))))
        adj = holm([res[t]["p_boot"] for t, _, _ in TESTS])
        for (t, _, sign), pa in zip(TESTS, adj):
            if pa >= ALPHA:
                v = "not confirmed"
            else:
                v = "confirmed" if np.sign(res[t]["estimate_pp"]) == sign else "contradicted"
            res[t] |= dict(p_holm=pa, verdict=v)
        per_seed[seed] = res
    return per_seed


def rows_of(family, analysis, per_seed):
    out = []
    prim = per_seed[SEEDS[0]]
    for t, what, sign in TESTS:
        r = prim[t]
        vs = {per_seed[s][t]["verdict"] for s in SEEDS}
        ps = [per_seed[s][t]["p_boot"] for s in SEEDS]
        out.append(dict(family=family, analysis=analysis, test=t, quantity=what,
                        predicted="positive" if sign > 0 else "negative",
                        estimate_pp=round(r["estimate_pp"], 2), ci_lo=round(r["ci_lo"], 2),
                        ci_hi=round(r["ci_hi"], 2), p_boot=round(r["p_boot"], 4),
                        p_holm=round(r["p_holm"], 4), p_min_seeds=round(min(ps), 4),
                        p_max_seeds=round(max(ps), 4), n_items=r["n_items"], n_stories=r["n_stories"],
                        verdict=vs.pop() if len(vs) == 1 else "borderline (changes with the bootstrap seed)"))
    return out


def descriptive(family, cells, story_of, gold_of):
    """Each accuracy test on gold-Yes and gold-No questions, the change in the
    share of Yes answers, and P2's two halves; stories resampled, first seed."""
    vy, vc = vectors(cells, "yes"), vectors(cells, "correct")
    out = []

    def add(test, part, v):
        if len(v) < 10:
            return
        e, d, nu = draws(v, story_of, SEEDS[0])
        out.append(dict(family=family, analysis="descriptive", test=test, quantity=part,
                        estimate_pp=round(e, 2), ci_lo=round(np.percentile(d, 2.5), 2),
                        ci_hi=round(np.percentile(d, 97.5), 2), p_boot=round(boot_p(d, NBOOT), 4),
                        n_items=len(set(item_of(v.index))), n_stories=nu))

    for t in ("P2", "N1", "N2"):
        v = vc[t]
        g = np.asarray(item_of(v.index).map(gold_of))
        add(t, "gold yes", v[g == "yes"])
        add(t, "gold no", v[g == "no"])
        add(t, "says yes", vy[t])
    add("N2", "every question", vectors(cells, "correct", every=True)["N2"])
    add("N2", "every question, says yes", vectors(cells, "yes", every=True)["N2"])
    c = {k: v.correct for k, v in cells.items()}
    y = {k: v.yes for k, v in cells.items()}
    for name, a, b in (("KEEP - PSEUDO under RAW", K, P), ("KEEP - PSEUDO under RAW_OPEN", KO, PO),
                       ("KEEP: RAW_OPEN minus RAW", KO, K), ("PSEUDO: RAW_OPEN minus RAW", PO, P),
                       ("PSEUDO_XY - PSEUDO under RAW", XY, P), ("PSEUDO_THIRD - PSEUDO under RAW", TH, P)):
        i = c[a].index.intersection(c[b].index)
        add(name, "accuracy", c[a][i] - c[b][i])
        add(name, "says yes", y[a][i] - y[b][i])
    return out


def cell_rows(family, cells):
    out = []
    for (lex, cond), d in cells.items():
        ay, an = d[d.gold == "yes"].correct.mean(), d[d.gold == "no"].correct.mean()
        out.append(dict(family=family, lexicon=lex, cond=cond, n=len(d),
                        accuracy_pct=round(100 * d.correct.mean(), 2),
                        says_yes_pct=round(100 * d.yes.mean(), 2), acc_gold_yes_pct=round(100 * ay, 2),
                        acc_gold_no_pct=round(100 * an, 2), balanced_pct=round(50 * (ay + an), 2),
                        gold_yes_pct=round(100 * (d.gold == "yes").mean(), 2)))
    return out


def main() -> int:
    rows, cells_out = [], []
    for family, (models, s, n, recap) in FAMILIES.items():
        cells = load(family, "primary")
        if cells is None:
            missing = [path_of(family, k).name for k in FILES if not path_of(family, k).exists()]
            print(f"  {family}: no records yet: {', '.join(missing)}")
            continue
        check_study1(family, cells)
        if not SHOWN:
            SHOWN.update(third_shown())
        first = pd.concat(cells.values())
        story_of = first.groupby(item_of(first.index)).story_id.first()
        gold_of = first.groupby(item_of(first.index)).gold.first()
        print(f"  {family}: {len(SHOWN)} of {len(story_of)} questions name another variable "
              f"in their RAW prompt")
        rows += rows_of(family, "primary", run_tests(cells, story_of))
        rows += rows_of(family, "questions resampled", run_tests(cells, story_of, stories=False))
        rows += rows_of(family, "unparsed as wrong", run_tests(load(family, "unparsed as wrong"), story_of))
        if recap:
            rows += rows_of(family, "re-asked at 1500", run_tests(load(family, "re-asked at 1500"), story_of))
        rows += descriptive(family, cells, story_of, gold_of)
        cells_out += cell_rows(family, cells)
    if not rows:
        return 0
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "names_premise.csv", index=False)
    C = pd.DataFrame(cells_out)
    C.to_csv(RESULTS / "names_premise_cells.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 300, "display.max_columns", 20):
        print(C.to_string(index=False))
        print()
        print(R.to_string(index=False))
    print("\n  wrote results/cladder/names_premise.csv, names_premise_cells.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
