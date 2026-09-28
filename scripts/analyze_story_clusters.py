"""The registered CLadder tests with stories, not questions, as the resampling
unit, and the name effects split by CLadder's own commonsense label.

    python scripts/analyze_story_clusters.py

Exploratory sensitivity analyses, added after the registered ones.

  1. STORIES. The names are a property of the story: every question of one
     story carries the same variable names, and CLadder reuses a few dozen
     stories for thousands of questions. Resampling questions treats those
     questions as independent and can make the intervals of a name effect too
     narrow. The six tests of B5 (each family) and the five of B6 are rerun
     with stories resampled, with every other part of the registered
     construction kept: the estimate is unchanged, the interval and p are
     not. Holm across each phase's tests, five bootstrap seeds, as registered.
     CaLM's registered tests already resample stories.
  2. PRIOR. CLadder labels each real-word question commonsense (with its easy
     and hard difficulty tags) or anticommonsense, where the names are real
     but the story's causal direction is implausible. If real names help only
     because they carry a correct prior, the gain over symbols should shrink
     or vanish on anticommonsense questions. H3b and KEEP minus PSEUDO are
     split by that label, questions resampled as registered, with the
     difference between the two groups.

Writes results/cladder/story_clusters.csv and prior_split.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

import analyze_b6 as b6
from analyze_confirmatory import (ALPHA, CONF, FAMILIES, MARGIN, SEEDS, SEVEN, TESTS,
                                  cell_item, classes, conf_data, h1_matrix, h2_matrix, holm,
                                  paired_causal)
from pilot import read_ids
from stats import boot_p, cluster_boot

RESULTS = ROOT / "results" / "cladder"
RAW = RESULTS / "raw"
NBOOT = 4000
PLAUSIBLE = {"commonsense", "easy", "hard"}        # as analyze_prior_strength.py


# ---------------------------------------------------------------- matrices with an index
def step_df(hi, lo, models):
    cols = []
    for m in models:
        x, y = cell_item(hi, m), cell_item(lo, m)
        i = x.index.intersection(y.index)
        if len(i):
            cols.append(pd.Series(x[i].values - y[i].values, index=i, name=m))
    return pd.concat(cols, axis=1).sort_index()


def h4_series(d, cls, lex):
    d = d[(d.parsed == 1) & d.graph_id.isin(SEVEN)]
    M = pd.concat({k: paired_causal(d, f"DR_k{k}") for k in (1, 2, 3)}, axis=1).dropna()
    flag = cls[(cls.cond == "DR_k1") & (cls.lexicon == lex)].set_index("item").unchanged_qt
    M = M[flag.reindex(M.index).eq(False).values]
    return ((M[3] - M[1]) / 2).to_frame("slope")


def story_boot(W, story_of, seed):
    """Mean over every cell; resample stories with all their rows."""
    lab = np.asarray(W.index.map(story_of))
    groups = [W.values[lab == s] for s in sorted(set(lab))]
    d = cluster_boot(len(groups), lambda i: 100 * np.nanmean(
        np.concatenate([groups[j] for j in i]).ravel()), seed, NBOOT)
    return 100 * np.nanmean(W.values), d, len(groups)


def tost_p(d):
    return max(max(float((d <= -MARGIN).mean()), 1 / NBOOT),
               max(float((d >= MARGIN).mean()), 1 / NBOOT))


# ---------------------------------------------------------------- 1. stories
def b5_family(family, models, prefix, cls):
    S = conf_data(prefix, None)
    K = S["main"]["KEEP"]
    by_id = K.drop_duplicates("id").set_index("id").story_id
    by_item = K.drop_duplicates("item").set_index("item").story_id
    Lad = S["ladder"]
    mats = {"H1": (h1_matrix(S["main"]["KEEP"], S["main"]["PSEUDO"], models), by_id),
            "H2": (h2_matrix(S["main"]["PSEUDO"], models), by_id),
            "H3a": (step_df(Lad["IRRELEVANT"], Lad["PERMUTE"], models), by_item),
            "H3b": (step_df(Lad["KEEP"], Lad["SYMBOL"], models), by_item),
            "H4-KEEP": (h4_series(S["slope"]["KEEP"], cls, "KEEP"), by_item),
            "H4-PSEUDO": (h4_series(S["slope"]["PSEUDO"], cls, "PSEUDO"), by_item)}
    per_seed = {}
    for seed in SEEDS:
        res = {}
        for t, (W, st) in mats.items():
            e, d, ns = story_boot(W, st, seed)
            r = dict(estimate_pp=e, ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5),
                     p_boot=tost_p(d) if t == "H3a" else boot_p(d, NBOOT), n_items=len(W),
                     n_stories=ns)
            if t == "H3a":
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


def b6_stories():
    X = b6.b6_draws()
    st = pd.read_csv(RAW / "pilot_raw_b6KEEP.csv").drop_duplicates("item").set_index("item").story_id
    Xs = X.assign(item=X.item.map(st))                 # the bootstrap unit becomes the story
    per_seed = {}
    for seed in SEEDS:
        res = b6.run_tests(Xs, seed)
        v = b6.verdicts(res)
        for t in res:
            res[t] |= dict(p_holm=v[t][1], verdict=v[t][0], n_items=X.item.nunique(),
                           n_stories=Xs.item.nunique())
        per_seed[seed] = res
    return per_seed


def collect(phase, family, per_seed, registered):
    rows = []
    prim = per_seed[SEEDS[0]]
    for t, r in prim.items():
        vs = {per_seed[s][t]["verdict"] for s in SEEDS}
        ps = [per_seed[s][t]["p_boot"] for s in SEEDS]
        rows.append(dict(phase=phase, family=family, test=t, estimate_pp=round(r["estimate_pp"], 2),
                         ci_lo=round(r["ci_lo"], 2), ci_hi=round(r["ci_hi"], 2),
                         ci90_lo=round(r["ci90_lo"], 2) if "ci90_lo" in r else "",
                         ci90_hi=round(r["ci90_hi"], 2) if "ci90_hi" in r else "",
                         p_boot=round(r["p_boot"], 4), p_holm=round(r["p_holm"], 4),
                         p_min_seeds=round(min(ps), 4), p_max_seeds=round(max(ps), 4),
                         n_items=r["n_items"], n_stories=r["n_stories"],
                         verdict=vs.pop() if len(vs) == 1 else "borderline (changes with the bootstrap seed)",
                         registered_verdict=registered.get(t, "")))
    return rows


# ---------------------------------------------------------------- 2. prior
def prior_rows(family, models, prefix):
    S = conf_data(prefix, None)
    K, P, Lad = S["main"]["KEEP"], S["main"]["PSEUDO"], S["ladder"]
    qp = K.drop_duplicates("item").set_index("item").question_property
    cls = qp.map(lambda q: "commonsense" if q in PLAUSIBLE else q)

    def keep_minus(lo, cond):
        cols = []
        for m in models:
            x = K[(K.model == m) & (K.cond == cond)]
            y = lo[(lo.model == m) & (lo.cond == cond)]
            x = x[x.parsed == 1].set_index("item").correct
            y = y[y.parsed == 1].set_index("item").correct
            i = x.index.intersection(y.index)
            cols.append(pd.Series(x[i].values - y[i].values, index=i, name=m))
        return pd.concat(cols, axis=1)

    q = {"H3b: RAW, KEEP minus SYMBOL": step_df(Lad["KEEP"], Lad["SYMBOL"], models),
         "RAW, KEEP minus PSEUDO": keep_minus(P, "RAW"),
         "ORACLE, KEEP minus PSEUDO": keep_minus(P, "ORACLE")}
    rows = []
    for what, W in q.items():
        W = W.loc[W.index.isin(cls.index)]
        lab = np.asarray(W.index.map(cls))
        A = {g: W.values[lab == g] for g in ("commonsense", "anticommonsense")}
        for g, a in A.items():
            d = cluster_boot(len(a), lambda i: 100 * np.nanmean(a[i]), SEEDS[0], NBOOT)
            rows.append(dict(family=family, quantity=what, group=g, estimate_pp=round(100 * np.nanmean(a), 2),
                             ci_lo=round(np.percentile(d, 2.5), 2), ci_hi=round(np.percentile(d, 97.5), 2),
                             p_boot=round(boot_p(d, NBOOT), 4), n_items=len(a)))
        a, b = A["commonsense"], A["anticommonsense"]
        rng = np.random.default_rng(SEEDS[0])
        d = np.array([100 * (np.nanmean(a[rng.integers(0, len(a), len(a))])
                             - np.nanmean(b[rng.integers(0, len(b), len(b))])) for _ in range(NBOOT)])
        rows.append(dict(family=family, quantity=what, group="commonsense minus anticommonsense",
                         estimate_pp=round(100 * (np.nanmean(a) - np.nanmean(b)), 2),
                         ci_lo=round(np.percentile(d, 2.5), 2), ci_hi=round(np.percentile(d, 97.5), 2),
                         p_boot=round(boot_p(d, NBOOT), 4), n_items=len(a) + len(b)))
    return rows


def main() -> int:
    conf = pd.read_csv(RESULTS / "confirmatory.csv")
    conf = conf[conf.analysis == "primary"]
    cls = classes(CONF["n"], CONF["seed"], read_ids(CONF["exclude"]))
    rows, prior = [], []
    for family, (models, prefix, _) in FAMILIES.items():
        if conf_data(prefix, None) is None:
            continue
        reg = conf[conf.family == family].set_index("test").verdict.to_dict()
        rows += collect("B5", family, b5_family(family, models, prefix, cls), reg)
        prior += prior_rows(family, models, prefix)
    b6reg = pd.read_csv(RESULTS / "b6.csv")
    b6reg = b6reg[b6reg.analysis == "primary"].set_index("test").verdict.to_dict()
    rows += collect("B6", "gpt", b6_stories(), b6reg)

    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "story_clusters.csv", index=False)
    P = pd.DataFrame(prior)
    P.to_csv(RESULTS / "prior_split.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(R.to_string(index=False))
        print()
        print(P.to_string(index=False))
    print("\n  wrote results/cladder/story_clusters.csv, prior_split.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
