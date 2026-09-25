"""Second benchmark: CaLM's ATE items, the three tests fixed in prereg/CALM.md.

    python scripts/analyze_calm.py

1. CHECK, every run. The two CaLM files are the ones registered (SHA-256 of
   commit 1c1e93a of OpenCausaLab/CaLM); the recomputed gold (src/calm.gold)
   agrees with all 100 answers CaLM publishes in its Lite release, and
   src/calm.mode with all 100 of its mode labels. Any failure exits 1. Writes
   results/calm/calm_items.csv: every item, its mode, gold and whether it is used.

2. TESTS, for each model family whose records exist
   (results/calm/raw/calm_raw_{prefix}{KEEP,PSEUDO}.csv, scripts/calm_run.py):

     C1  per item [(KEEP - PSEUDO) | RAW] - [(KEEP - PSEUDO) | ORACLE]   > 0
     C2  per item, PSEUDO: ORACLE - DR_k1                                > 0
     C3  per item, RAW: KEEP - PSEUDO                                    > 0

   One value per (item, model) cell, averaged over every cell. The bootstrap
   resamples STORIES: the 520 items come from 92 stories that share names and a
   graph, so items are not independent units. Holm over the three, family-wise
   alpha 0.05; bootstrap seeds 20260907, 1, 2, 3, 4, and a verdict that changes
   with the seed is borderline. Then the registered sensitivity analyses
   (items resampled instead of stories; unparsed answers scored as wrong; for
   Llama, cut-off answers re-asked) and an exploratory accuracy table.
   Writes results/calm/calm.csv and results/calm/calm_accuracy.csv.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

import calm
from stats import boot_cells, boot_p, cluster_boot

RAW = ROOT / "results" / "calm" / "raw"
NBOOT = 4000
SEEDS = [20260907, 1, 2, 3, 4]
ALPHA = 0.05
SHA256 = {
    calm.FULL: "0928be65f495862fab154dd206342ecbc8000cb76d814d22f7b7fcc3a7194e62",
    calm.LITE: "70809205b956dcd38895e37df2d079ce71057c0569f4b507410093e3fc6947f1",
}
TIER = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]
LLAMA = "meta-llama/llama-3.3-70b-instruct"
# family -> (models, file prefix, re-ask cap or None)
FAMILIES = {"llama": ([LLAMA], "llama", 1500), "gpt": (TIER, "gpt", None)}
TESTS = [
    ("C1", "DiD [(KEEP - PSEUDO) | RAW] - [(KEEP - PSEUDO) | ORACLE]"),
    ("C2", "PSEUDO: ORACLE minus DR_k1"),
    ("C3", "RAW: KEEP minus PSEUDO"),
]


# ---------------------------------------------------------------- check

def check() -> pd.DataFrame:
    print("=" * 78)
    print("1. CHECK - data, recomputed gold and mode against CaLM's own labels")
    print("=" * 78)
    for name, want in SHA256.items():
        got = hashlib.sha256((calm.DATA / name).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"  {name}: SHA-256 {got[:16]}... is not the registered file")
    print("  both files are the registered ones (SHA-256)")
    full, lite = calm.load(calm.FULL), calm.load(calm.LITE)
    freq = calm.name_freq(full)
    g = sum(calm.gold(x) == x["gt_answer"].lower() for x in lite)
    m = sum(calm.mode(x, freq) == x["Mode"] for x in lite)
    print(f"  gold agrees with CaLM Lite on {g}/{len(lite)}, mode on {m}/{len(lite)}")
    if g != len(lite) or m != len(lite):
        raise SystemExit("  the recomputed labels do not reproduce CaLM's; stopping")

    from calm_run import items
    used = {it["index"] for it in items()}
    rows = [dict(item=int(x["index"]), story=calm.story(x), mode=calm.mode(x, freq),
                 gold=calm.gold(x), n_nodes=len(calm.node_names(x)),
                 has_data=bool(x["Background"]["data_info"].strip()),
                 used=int(x["index"] in used)) for x in full]
    T = pd.DataFrame(rows).sort_values("item")
    T.to_csv(ROOT / "results" / "calm" / "calm_items.csv", index=False)
    u = T[T.used == 1]
    print(f"  {len(T)} items: {({k: int(v) for k, v in T['mode'].value_counts().items()})}; used {len(u)} REAL items "
          f"from {u.story.nunique()} stories, gold yes {int((u.gold == 'yes').sum())} / "
          f"no {int((u.gold == 'no').sum())}")
    print("  wrote results/calm/calm_items.csv")
    return T


# ---------------------------------------------------------------- tests

def cell(d, model, cond):
    s = d[(d.model == model) & (d.cond == cond) & (d.parsed == 1)]
    return s.set_index("item").correct


def matrix(K, P, models, test):
    cols = []
    for m in models:
        if test == "C1":
            kr, pr, ko, po = cell(K, m, "RAW"), cell(P, m, "RAW"), cell(K, m, "ORACLE"), cell(P, m, "ORACLE")
            i = kr.index.intersection(pr.index).intersection(ko.index).intersection(po.index)
            v = (kr[i] - pr[i]) - (ko[i] - po[i])
        elif test == "C2":
            o, r = cell(P, m, "ORACLE"), cell(P, m, "DR_k1")
            i = o.index.intersection(r.index)
            v = o[i] - r[i]
        else:
            k, p = cell(K, m, "RAW"), cell(P, m, "RAW")
            i = k.index.intersection(p.index)
            v = k[i] - p[i]
        cols.append(pd.Series(v.values, index=i, name=m))
    return pd.concat(cols, axis=1).groupby(level=0).mean()


def boot_stories(W, story_of, seed):
    """Resample stories; the statistic is the mean over every cell of the drawn stories."""
    groups = [g.values for _, g in W.groupby(W.index.map(story_of), sort=True)]
    draws = cluster_boot(len(groups), lambda i: 100 * np.nanmean(
        np.concatenate([groups[j] for j in i]).ravel()), seed, NBOOT)
    est = 100 * np.nanmean(W.values)
    return est, np.percentile(draws, 2.5), np.percentile(draws, 97.5), boot_p(draws, NBOOT)


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    adj, run = np.empty_like(ps), 0.0
    for r, i in enumerate(np.argsort(ps, kind="stable")):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def run_tests(K, P, models, story_of, label, family, by="story"):
    rows, per_seed = [], {}
    Ws = {t: matrix(K, P, models, t) for t, _ in TESTS}
    for s in SEEDS:
        res = {t: (boot_stories(W, story_of, s) if by == "story" else boot_cells(W, s, NBOOT))
               for t, W in Ws.items()}
        adj = holm([res[t][3] for t, _ in TESTS])
        per_seed[s] = {t: (res[t], a,
                           ("confirmed" if res[t][0] > 0 else "contradicted") if a < ALPHA
                           else "not confirmed")
                       for (t, _), a in zip(TESTS, adj)}
    for t, what in TESTS:
        (e, lo, hi, p), a, _ = per_seed[SEEDS[0]][t]
        vs = {per_seed[s][t][2] for s in SEEDS}
        ps = [per_seed[s][t][0][3] for s in SEEDS]
        verdict = vs.pop() if len(vs) == 1 else "borderline (changes with the bootstrap seed)"
        rows.append(dict(family=family, analysis=label, test=t, quantity=what,
                         predicted="positive", estimate_pp=round(e, 2), ci_lo=round(lo, 2),
                         ci_hi=round(hi, 2), p_boot=round(p, 4), p_holm=round(a, 4),
                         p_min_seeds=round(min(ps), 4), p_max_seeds=round(max(ps), 4),
                         n_items=len(Ws[t]), n_stories=Ws[t].index.map(story_of).nunique(),
                         verdict=verdict))
        print(f"  {t}  {e:+7.2f} [{lo:+6.2f} ; {hi:+6.2f}]  p={p:.4f}  Holm {a:.4f}  "
              f"n={len(Ws[t])} items  {verdict}")
    return rows


def with_recap(d, path):
    r = pd.read_csv(path)
    key = ["model", "item", "cond"]
    cut = d.finish == "length"
    if set(map(tuple, r[key].values)) != set(map(tuple, d.loc[cut, key].values)):
        raise SystemExit(f"{path.name}: the re-asked rows are not the cut-off rows")
    return pd.concat([d[~cut], r], ignore_index=True)


def main() -> int:
    T = check()
    story_of = T.set_index("item").story
    rows, acc = [], []
    for family, (models, prefix, recap) in FAMILIES.items():
        f = {lex: RAW / f"calm_raw_{prefix}{lex}.csv" for lex in ("KEEP", "PSEUDO")}
        if not all(p.exists() for p in f.values()):
            print(f"\n  {family}: no CaLM records yet (results/calm/raw/calm_raw_{prefix}*.csv)")
            continue
        K, P = pd.read_csv(f["KEEP"]), pd.read_csv(f["PSEUDO"])
        print("\n" + "=" * 78)
        print(f"2. TESTS - {family}")
        print("=" * 78)
        for lex, d in (("KEEP", K), ("PSEUDO", P)):
            print(f"  parse rate {lex:7s} {100 * d.parsed.mean():6.2f}%  ({len(d)} rows)")
        print("\n  primary: stories resampled")
        rows += run_tests(K, P, models, story_of, "primary", family)
        print("\n  sensitivity: items resampled")
        rows += run_tests(K, P, models, story_of, "items resampled", family, by="item")
        w = lambda d: d.assign(correct=np.where(d.parsed == 1, d.correct, 0), parsed=1)
        print("\n  sensitivity: unparsed answers scored as wrong")
        rows += run_tests(w(K), w(P), models, story_of, "unparsed scored as wrong", family)
        if recap:
            RK = with_recap(K, f["KEEP"].with_name(f"calm_raw_{prefix}KEEP_recap{recap}.csv"))
            RP = with_recap(P, f["PSEUDO"].with_name(f"calm_raw_{prefix}PSEUDO_recap{recap}.csv"))
            print(f"\n  sensitivity: cut-off answers re-asked at {recap} tokens")
            rows += run_tests(RK, RP, models, story_of, f"re-asked at {recap}", family)
        # Exploratory, not a test: accuracy by condition, split by whether the
        # answer needs the probabilities (has_data) or only the graph, and by
        # whether a directed path runs from treatment to outcome at all (with
        # none, the answer is No whatever the probabilities). The share of Yes
        # answers shows which way a prior pushes. "all" pools the family's models.
        for lex, d in (("KEEP", K), ("PSEUDO", P)):
            d = d[d.parsed == 1]
            for who, dd in [(m, d[d.model == m]) for m in models] + [("all", d)]:
                for (c, hd, pa), g in dd.groupby(["cond", "has_data", "path"]):
                    acc.append(dict(family=family, model=who, lexicon=lex, cond=c,
                                    needs_probabilities=bool(hd), path=bool(pa), n=len(g),
                                    accuracy_pct=round(100 * g.correct.mean(), 2),
                                    says_yes_pct=round(100 * (g.pred == "yes").mean(), 2)))
    if rows:
        pd.DataFrame(rows).to_csv(ROOT / "results" / "calm" / "calm.csv", index=False)
        pd.DataFrame(acc).to_csv(ROOT / "results" / "calm" / "calm_accuracy.csv", index=False)
        print("\n  wrote results/calm/calm.csv, results/calm/calm_accuracy.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
