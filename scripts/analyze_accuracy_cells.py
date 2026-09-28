"""Absolute accuracy in every cell, and every registered contrast split into
its Yes and No questions.

    python scripts/analyze_accuracy_cells.py

Exploratory, added after the registered analyses. A difference in accuracy
does not say where the accuracies sit, and on yes/no questions it mixes two
things: a shift in how often the model says Yes, and a change in how well it
tells Yes questions from No questions. So, for B5 (the fresh CLadder sample,
every family) and CaLM (every family):

  1. CELLS. Per model and per family (models pooled), per names x graph:
     accuracy, share of Yes answers, accuracy on gold-Yes and on gold-No
     questions, balanced accuracy (their mean). Parsed answers only.
  2. SPLIT. Each registered contrast (H1 to H3b, C1 to C3) computed exactly as
     registered, then on the gold-Yes questions alone and the gold-No
     questions alone, their mean (the change in balanced accuracy), and the
     change in the share of Yes answers. A pure shift towards Yes raises
     accuracy on Yes questions and lowers it on No questions by the same
     amount; a gain in discrimination raises both. The "all" row must equal
     the published primary estimate, or the script stops.
  3. PER MODEL. H1 to H3b for each model of the GPT-4.1 family, which the
     registered tests average.

Bootstrap as registered: questions for B5, stories for CaLM; 4,000 draws.
Writes results/cladder/accuracy_cells.csv, effect_split.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from analyze_confirmatory import FAMILIES, conf_data
from analyze_querygroup import ARITH, IDENT
from stats import boot_p, cluster_boot

RESULTS = ROOT / "results" / "cladder"
CALM = ROOT / "results" / "calm"
NBOOT = 4000
SEED = 20260907
CALM_FAMILIES = {"gpt": (["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"], "gpt"),
                 "llama": (["meta-llama/llama-3.3-70b-instruct"], "llama")}


def prep(d):
    d = d[(d.parsed == 1) & ~d.query_type.isin(ARITH | IDENT)] if "query_type" in d else d[d.parsed == 1]
    return d.assign(yes=(d.pred.astype(str).str.lower() == "yes").astype(float))


# ---------------------------------------------------------------- 1. cells
def cell_rows(bench, family, lex, d, models):
    out = []
    for who, dd in [(m, d[d.model == m]) for m in models] + [("all", d[d.model.isin(models)])]:
        for c, g in dd.groupby("cond"):
            ay = g[g.gold == "yes"].correct.mean()
            an = g[g.gold == "no"].correct.mean()
            out.append(dict(benchmark=bench, family=family, model=who, lexicon=lex, cond=c,
                            n=len(g), accuracy_pct=round(100 * g.correct.mean(), 2),
                            says_yes_pct=round(100 * g.yes.mean(), 2),
                            acc_gold_yes_pct=round(100 * ay, 2), acc_gold_no_pct=round(100 * an, 2),
                            balanced_pct=round(50 * (ay + an), 2),
                            gold_yes_pct=round(100 * (g.gold == "yes").mean(), 2)))
    return out


# ---------------------------------------------------------------- 2. split
def per_unit(a, b, models, col, key):
    """cond a minus cond b per (model, key), parsed pairs; rows = key, cols = model."""
    cols = []
    for m in models:
        x = a[a.model == m].drop_duplicates(key).set_index(key)[col]
        y = b[b.model == m].drop_duplicates(key).set_index(key)[col]
        i = x.index.intersection(y.index)
        if len(i) >= 10:
            cols.append(pd.Series((x[i] - y[i]).values, index=i, name=m))
    return pd.concat(cols, axis=1).groupby(level=0).mean() if cols else None


def did(a1, b1, a2, b2, models, col, key):
    """(a1 - b1) - (a2 - b2) per (model, key), every item in all four cells."""
    cols = []
    for m in models:
        s = [d[d.model == m].drop_duplicates(key).set_index(key)[col] for d in (a1, b1, a2, b2)]
        i = s[0].index.intersection(s[1].index).intersection(s[2].index).intersection(s[3].index)
        if len(i) >= 10:
            cols.append(pd.Series(((s[0][i] - s[1][i]) - (s[2][i] - s[3][i])).values,
                                  index=i, name=m))
    return pd.concat(cols, axis=1).groupby(level=0).mean() if cols else None


def boot(W, cluster_of=None):
    """Mean over every cell; resample rows, or clusters of rows."""
    if cluster_of is None:
        A = W.values
        d = cluster_boot(len(A), lambda i: 100 * np.nanmean(A[i]), SEED, NBOOT)
    else:
        groups = [g.values for _, g in W.groupby(W.index.map(cluster_of), sort=True)]
        d = cluster_boot(len(groups), lambda i: 100 * np.nanmean(
            np.concatenate([groups[j] for j in i]).ravel()), SEED, NBOOT)
    return 100 * np.nanmean(W.values), d


def split_rows(bench, family, models, test, build, gold_of, cluster_of=None, published=None):
    """build(col, subset) -> matrix; subset selects questions by gold label."""
    out, draws = [], {}
    for part in ("all", "gold yes", "gold no"):
        W = build("correct")
        if part != "all":
            W = W[W.index.map(gold_of) == part.split()[1]]
        e, d = boot(W, cluster_of)
        draws[part] = (e, d)
        if part == "all" and published is not None and round(e, 2) != published:
            raise SystemExit(f"{bench} {family} {test}: {e:.2f} is not the published {published}")
        out.append((part, e, d, len(W)))
    eb = (draws["gold yes"][0] + draws["gold no"][0]) / 2
    # The balanced change needs both halves from ONE resample of the rows, so
    # its interval is drawn jointly rather than from the two halves' draws.
    W = build("correct")
    g = np.asarray(W.index.map(gold_of))[:, None]
    stack = np.stack([np.where(g == "yes", W.values, np.nan), np.where(g == "no", W.values, np.nan)])
    if cluster_of is None:
        dd = cluster_boot(len(W), lambda i: 50 * (np.nanmean(stack[0][i]) + np.nanmean(stack[1][i])),
                          SEED, NBOOT)
    else:
        idx = [np.flatnonzero(np.asarray(W.index.map(cluster_of) == c))
               for c in sorted(set(W.index.map(cluster_of)))]
        dd = cluster_boot(len(idx), lambda i: 50 * (
            np.nanmean(stack[0][np.concatenate([idx[j] for j in i])])
            + np.nanmean(stack[1][np.concatenate([idx[j] for j in i])])), SEED, NBOOT)
    out.append(("balanced", eb, dd, len(W)))
    e, d = boot(build("yes"), cluster_of)
    out.append(("says yes", e, d, len(W)))
    rows = []
    for part, e, d, n in out:
        d = np.asarray(d, dtype=float)
        d = d[np.isfinite(d)]
        rows.append(dict(benchmark=bench, family=family, models="+".join(models) if len(models) < 3
                         else "GPT-4.1 family", test=test, part=part, estimate_pp=round(e, 2),
                         ci_lo=round(np.percentile(d, 2.5), 2), ci_hi=round(np.percentile(d, 97.5), 2),
                         p_boot=round(boot_p(d, NBOOT), 4), n_questions=n))
    return rows


def b5_rows(family, S, models, published):
    K, P = prep(S["main"]["KEEP"]), prep(S["main"]["PSEUDO"])
    L = {lex: prep(d) for lex, d in S["ladder"].items()}
    gold_of = pd.concat([K, P]).drop_duplicates("id").set_index("id").gold
    gold_item = pd.concat([K, P]).drop_duplicates("item").set_index("item").gold
    c = lambda d, cond: d[d.cond == cond]
    tests = {
        "H1": (lambda col: did(c(K, "RAW"), c(P, "RAW"), c(K, "ORACLE"), c(P, "ORACLE"),
                               models, col, "id"), gold_of),
        "H2": (lambda col: per_unit(c(P, "ORACLE"), c(P, "DR_k1"), models, col, "id"), gold_of),
        "H3a": (lambda col: per_unit(c(L["IRRELEVANT"], "RAW"), c(L["PERMUTE"], "RAW"),
                                     models, col, "item"), gold_item),
        "H3b": (lambda col: per_unit(c(L["KEEP"], "RAW"), c(L["SYMBOL"], "RAW"),
                                     models, col, "item"), gold_item),
        # not registered: the cost of anonymising, with and without the graph,
        # the two halves of H1
        "RAW: KEEP minus PSEUDO": (lambda col: per_unit(c(K, "RAW"), c(P, "RAW"),
                                                        models, col, "id"), gold_of),
        "ORACLE: KEEP minus PSEUDO": (lambda col: per_unit(c(K, "ORACLE"), c(P, "ORACLE"),
                                                           models, col, "id"), gold_of),
    }
    rows = []
    for t, (build, g) in tests.items():
        rows += split_rows("CLadder B5", family, models, t, build, g,
                           published=published.get(t) if published else None)
    return rows


def main() -> int:
    conf = pd.read_csv(RESULTS / "confirmatory.csv")
    conf = conf[conf.analysis == "primary"]
    cells, split = [], []
    for family, (models, prefix, _) in FAMILIES.items():
        S = conf_data(prefix, None)
        if S is None:
            continue
        for lex, d in {**S["main"], **{k: v for k, v in S["ladder"].items()
                                       if k not in S["main"]}}.items():
            cells += cell_rows("CLadder B5", family, lex, prep(d), models)
        pub = conf[conf.family == family].set_index("test").estimate_pp.to_dict()
        split += b5_rows(family, S, models, pub)
        if family == "gpt":
            for m in models:
                split += b5_rows(family, S, [m], None)

    items = pd.read_csv(CALM / "calm_items.csv").set_index("item")
    story_of, gold_of = items.story, items.gold
    pub = pd.read_csv(CALM / "calm.csv")
    pub = pub[pub.analysis == "primary"]
    for family, (models, prefix) in CALM_FAMILIES.items():
        K = prep(pd.read_csv(CALM / "raw" / f"calm_raw_{prefix}KEEP.csv"))
        P = prep(pd.read_csv(CALM / "raw" / f"calm_raw_{prefix}PSEUDO.csv"))
        for lex, d in (("KEEP", K), ("PSEUDO", P)):
            cells += cell_rows("CaLM", family, lex, d, models)
        c = lambda d, cond: d[d.cond == cond]
        p = pub[pub.family == family].set_index("test").estimate_pp.to_dict()
        tests = {
            "C1": lambda col: did(c(K, "RAW"), c(P, "RAW"), c(K, "ORACLE"), c(P, "ORACLE"),
                                  models, col, "item"),
            "C2": lambda col: per_unit(c(P, "ORACLE"), c(P, "DR_k1"), models, col, "item"),
            "C3": lambda col: per_unit(c(K, "RAW"), c(P, "RAW"), models, col, "item"),
        }
        for t, build in tests.items():
            split += split_rows("CaLM", family, models, t, build, gold_of, story_of, p[t])

    C = pd.DataFrame(cells)
    C.to_csv(RESULTS / "accuracy_cells.csv", index=False)
    X = pd.DataFrame(split)
    X.to_csv(RESULTS / "effect_split.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 400):
        print(C[C.model == "all"].to_string(index=False))
        print()
        print(X.to_string(index=False))
    print("\n  wrote results/cladder/accuracy_cells.csv, effect_split.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
