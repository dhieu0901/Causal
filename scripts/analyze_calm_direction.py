"""CaLM: does C3 run the wrong way because of how the question is aimed?

    python scripts/analyze_calm_direction.py
    python scripts/analyze_calm_direction.py --dump 60    # a blind sheet for hand coding
    python scripts/analyze_calm_direction.py --score      # score the filled sheet
    python scripts/analyze_calm_direction.py --dump-unrelated   # second sheet (task 2.2c)

The pre-registered test C3 (prereg/CALM.md) predicted that real names beat
pseudowords when no graph is given. On CaLM it ran the other way for both
model families, and all of the reversal sat in the 283 items with no directed
path from treatment to outcome. REPORT section 3 offered a reading after
seeing the numbers: a familiar name carries a prior that helps when the
question runs the way the story does and misleads when it runs against it or
between two variables the story does not link. This file puts that reading to
the data it came from, so it stays exploratory.

Each used item is classed by where its question points in the story's own
graph (src/calm.py reads both):

  forward     a directed path runs from treatment to outcome; the answer
              depends on the probabilities
  reverse     a directed path runs from outcome to treatment: the question asks
              whether an effect moves its own cause, and the answer is No
  unrelated   no directed path either way; the answer is No. Split once more by
              whether the two share a cause in the graph, since a shared cause
              makes them correlated.

The class comes from the graph, not from reading the names. For REAL items the
graph is the story, so the two should agree; --dump writes a blind sheet (the
two names only) so a person can say which way the real world runs, and --score
compares that with the graph class (plan.md, task 2.2b).

The no-path items carry no probabilities at all. Under RAW, with the graph
removed, nothing in the prompt decides them; the model can only fall back on
the names. So the share of Yes answers under RAW is the readout of the prior.

Once the sheet is filled, every run also splits the items by the HAND code
of their pair (the code travels to every item that asks about the same two
names, in the same order): if the names carry the real world's causal
relation, the models should say Yes most on no-path items whose pair the real
world runs forward. That split is written to calm_direction_hand.csv.

The estimator is analyze_calm's own: one value per (item, model) cell,
averaged over every cell, stories resampled, five seeds. "reverse minus
unrelated" resamples stories once and takes both means from the same draw,
because a story can hold items of both classes.

Writes: results/calm/calm_direction.csv
        results/calm/calm_direction_hand.csv, once a sheet is filled
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

import calm
from analyze_calm import FAMILIES, NBOOT, RAW, SEEDS, boot_stories, matrix
from stats import boot_p, cluster_boot

OUT = ROOT / "results" / "calm"
SHEET = OUT / "direction_sample_for_hand_coding.csv"
# Task 2.2c: every graph-unrelated pair the first sheet did not hold, so that the
# no-path items whose pair the real world runs forward are numerous enough to
# compare. Read together with the first sheet.
SHEET2 = OUT / "direction_sample_for_hand_coding_2.csv"
CLASSES = ["forward", "reverse", "unrelated"]


def ancestors(edges, x):
    par: dict[str, list[str]] = {}
    for a, b in edges:
        par.setdefault(b, []).append(a)
    seen, todo = set(), [x]
    while todo:
        for p in par.get(todo.pop(), []):
            if p not in seen:
                seen.add(p)
                todo.append(p)
    return seen


def classify() -> pd.DataFrame:
    from calm_run import items
    rows = []
    for it in items():
        t, o = calm._treatment_outcome(it)
        e = calm.sym_edges(it)
        fwd, rev = calm._path(e, t, o), calm._path(e, o, t)
        nm = calm.node_names(it)
        rows.append(dict(item=int(it["index"]), story=calm.story(it),
                         direction="forward" if fwd else "reverse" if rev else "unrelated",
                         shared_cause=bool(ancestors(e, t) & ancestors(e, o)),
                         has_data=bool(it["Background"]["data_info"].strip()),
                         treatment=nm[t], outcome=nm[o]))
    C = pd.DataFrame(rows).sort_values("item").reset_index(drop=True)
    # Forward is exactly the path flag analyze_calm splits on, and every no-path
    # item carries no probabilities; both are assumed below, so both are checked.
    P = pd.read_csv(RAW / "calm_raw_gptKEEP.csv").drop_duplicates("item").set_index("item").path
    if not (C.set_index("item").direction.eq("forward") == P.reindex(C.item).values).all():
        raise SystemExit("direction class disagrees with the path flag in the records")
    if C[C.direction != "forward"].has_data.any():
        raise SystemExit("a no-path item carries probabilities; the RAW readout assumes none")
    return C


def diff_stories(W, cls, story_of, a, b, seed):
    """mean over class a minus mean over class b, stories resampled together."""
    st = W.index.map(story_of)
    stories = sorted(set(st))
    by = {s: (W[(st == s) & (cls.reindex(W.index).values == a)].values,
              W[(st == s) & (cls.reindex(W.index).values == b)].values) for s in stories}

    def stat(i):
        A = [by[stories[j]][0] for j in i]
        B = [by[stories[j]][1] for j in i]
        A = np.concatenate([x.ravel() for x in A]) if A else np.array([])
        B = np.concatenate([x.ravel() for x in B]) if B else np.array([])
        return 100 * (np.nanmean(A) - np.nanmean(B))

    draws = cluster_boot(len(stories), stat, seed, NBOOT)
    wa = W[cls.reindex(W.index).values == a].values
    wb = W[cls.reindex(W.index).values == b].values
    est = 100 * (np.nanmean(wa) - np.nanmean(wb))
    d = draws[np.isfinite(draws)]
    return est, np.percentile(d, 2.5), np.percentile(d, 97.5), boot_p(d, NBOOT)


def says_yes(d, models, cond, items_):
    s = d[d.model.isin(models) & (d.cond == cond) & (d.parsed == 1) & d.item.isin(items_)]
    return round(100 * (s.pred == "yes").mean(), 2), round(100 * s.correct.mean(), 2)


def main_tables(C) -> pd.DataFrame:
    story_of = C.set_index("item").story
    cls = C.set_index("item").direction
    rows = []
    for family, (models, prefix, _) in FAMILIES.items():
        K = pd.read_csv(RAW / f"calm_raw_{prefix}KEEP.csv")
        P = pd.read_csv(RAW / f"calm_raw_{prefix}PSEUDO.csv")
        W = matrix(K, P, models, "C3")
        print(f"\n  {family}: C3 = RAW, KEEP minus PSEUDO, by where the question points")
        subsets = [(c, C[C.direction == c].item) for c in CLASSES]
        subsets += [("unrelated, shared cause", C[(C.direction == "unrelated") & C.shared_cause].item),
                    ("unrelated, no shared cause", C[(C.direction == "unrelated") & ~C.shared_cause].item)]
        for name, its in subsets:
            w = W[W.index.isin(set(its))]
            per = [boot_stories(w, story_of, s) for s in SEEDS]
            e, lo, hi, p = per[0]
            ps = [x[3] for x in per]
            yk, ak = says_yes(K, models, "RAW", set(its))
            yp, ap = says_yes(P, models, "RAW", set(its))
            yo, ao = says_yes(K, models, "ORACLE", set(its))
            rows.append(dict(family=family, direction=name, quantity="RAW: KEEP minus PSEUDO",
                             estimate_pp=round(e, 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                             p_boot=round(p, 4), p_min_seeds=round(min(ps), 4),
                             p_max_seeds=round(max(ps), 4), n_items=len(w),
                             n_stories=w.index.map(story_of).nunique(),
                             raw_keep_acc_pct=ak, raw_pseudo_acc_pct=ap,
                             raw_keep_says_yes_pct=yk, raw_pseudo_says_yes_pct=yp,
                             oracle_keep_acc_pct=ao))
            print(f"    {name:27s} {e:+7.2f} [{lo:+6.2f} ; {hi:+6.2f}]  p {min(ps):.4f}-"
                  f"{max(ps):.4f}  n={len(w):3d}  says Yes under RAW: KEEP {yk:5.2f}%  "
                  f"PSEUDO {yp:5.2f}%")
        per = [diff_stories(W, cls, story_of, "reverse", "unrelated", s) for s in SEEDS]
        e, lo, hi, p = per[0]
        ps = [x[3] for x in per]
        rows.append(dict(family=family, direction="reverse minus unrelated",
                         quantity="difference in RAW: KEEP minus PSEUDO",
                         estimate_pp=round(e, 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                         p_boot=round(p, 4), p_min_seeds=round(min(ps), 4),
                         p_max_seeds=round(max(ps), 4),
                         n_items=int(W.index.isin(set(C[C.direction != "forward"].item)).sum()),
                         n_stories=W.index.map(story_of).nunique()))
        print(f"    {'reverse minus unrelated':27s} {e:+7.2f} [{lo:+6.2f} ; {hi:+6.2f}]  "
              f"p {min(ps):.4f}-{max(ps):.4f}")
    return pd.DataFrame(rows)


def dump(C, n):
    """A blind sheet: the two names only, shuffled, class withheld."""
    rng = random.Random(20260925)
    per = n // len(CLASSES)
    pick = []
    for c in CLASSES:
        pool = C[C.direction == c].drop_duplicates(["treatment", "outcome"])
        pick += rng.sample(list(pool.item), min(per, len(pool)))
    rng.shuffle(pick)
    S = C.set_index("item").loc[pick, ["treatment", "outcome"]].reset_index()
    S["your_code"] = ""
    S.to_csv(SHEET, index=False, encoding="utf-8-sig")
    print(f"  wrote {len(S)} pairs to {SHEET.relative_to(ROOT)}.\n"
          "  For each row fill your_code with one word, from the real world, not the "
          "story:\n    forward    changing the treatment would move the outcome\n"
          "    reverse    it runs the other way: the outcome moves the treatment\n"
          "    unrelated  neither moves the other\n"
          "  then run with --score.")


def read_sheet():
    parts = [pd.read_csv(f, encoding="utf-8-sig", dtype=str).fillna("")
             for f in (SHEET, SHEET2) if f.exists()]
    if not parts:
        return None
    S = pd.concat(parts, ignore_index=True)
    S["your_code"] = S.your_code.str.strip().str.lower()
    S = S[S.your_code.isin(CLASSES)]
    return S if len(S) else None


def dump_unrelated(C):
    """Every graph-unrelated (treatment, outcome) pair not on the first sheet, shuffled."""
    first = pd.read_csv(SHEET, encoding="utf-8-sig", dtype=str)
    done = set(zip(first.treatment, first.outcome))
    pool = C[C.direction == "unrelated"].drop_duplicates(["treatment", "outcome"])
    pool = pool[[(t, o) not in done for t, o in zip(pool.treatment, pool.outcome)]]
    pick = list(pool.item)
    random.Random(20260926).shuffle(pick)
    S = C.set_index("item").loc[pick, ["treatment", "outcome"]].reset_index()
    S["your_code"] = ""
    S["note"] = ""
    S.to_csv(SHEET2, index=False, encoding="utf-8-sig")
    print(f"  wrote {len(S)} pairs to {SHEET2.relative_to(ROOT)}")


def score(C):
    S = read_sheet()
    if S is None:
        raise SystemExit(f"{SHEET.name} has no filled rows")
    S["graph"] = S.item.astype(int).map(C.set_index("item").direction)
    print(f"  {len(S)} coded pairs; agreement with the graph class "
          f"{100 * (S.your_code == S.graph).mean():.1f}%\n")
    print(pd.crosstab(S.graph, S.your_code, margins=True).to_string())


def hand_split(C) -> pd.DataFrame | None:
    """C3 and the Yes rates on no-path items, split by the hand code of the pair."""
    S = read_sheet()
    if S is None:
        return None
    codes = S.groupby(["treatment", "outcome"]).your_code.agg(set)
    if (codes.map(len) > 1).any():
        raise SystemExit("one pair carries two different hand codes")
    code = codes.map(lambda x: next(iter(x)))
    C = C.assign(hand=[code.get((t, o)) for t, o in zip(C.treatment, C.outcome)])
    story_of = C.set_index("item").story
    rows = []
    agree = S.assign(graph=S.item.astype(int).map(C.set_index("item").direction))
    for g in CLASSES:
        for h in CLASSES:
            rows.append(dict(family="", part="agreement", graph_class=g, hand_code=h,
                             n_pairs=int(((agree.graph == g) & (agree.your_code == h)).sum())))
    print(f"\n  hand-coded sheet: {len(S)} pairs, agreement with the graph class "
          f"{100 * (agree.your_code == agree.graph).mean():.1f}%; the codes reach "
          f"{int(C.hand.notna().sum())} of {len(C)} items")
    for family, (models, prefix, _) in FAMILIES.items():
        K = pd.read_csv(RAW / f"calm_raw_{prefix}KEEP.csv")
        P = pd.read_csv(RAW / f"calm_raw_{prefix}PSEUDO.csv")
        W = matrix(K, P, models, "C3")
        print(f"  {family}: no-path items, RAW, by the hand code of the pair")
        for h in CLASSES:
            its = C[(C.direction != "forward") & C.hand.eq(h)].item
            w = W[W.index.isin(set(its))]
            if len(w) < 5:
                continue
            e, lo, hi, p = boot_stories(w, story_of, SEEDS[0])
            yk, _ = says_yes(K, models, "RAW", set(its))
            yp, _ = says_yes(P, models, "RAW", set(its))
            rows.append(dict(family=family, part="no-path items", hand_code=h,
                             estimate_pp=round(e, 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                             p_boot=round(p, 4), n_items=len(w),
                             n_stories=w.index.map(story_of).nunique(),
                             raw_keep_says_yes_pct=yk, raw_pseudo_says_yes_pct=yp))
            print(f"    real world {h:9s} {e:+7.2f} [{lo:+6.2f} ; {hi:+6.2f}]  n={len(w):3d}  "
                  f"says Yes under RAW: KEEP {yk:5.2f}%  PSEUDO {yp:5.2f}%")
        # Does the pull of a real name depend on which way the real world runs?
        # Stories resampled once, both means from the same draw.
        nop = C[C.direction != "forward"].set_index("item").hand
        w = W[W.index.isin(set(nop.dropna().index))]
        per = [diff_stories(w, nop, story_of, "forward", "reverse", sd) for sd in SEEDS]
        e, lo, hi, p = per[0]
        ps = [q[3] for q in per]
        rows.append(dict(family=family, part="no-path items", hand_code="forward minus reverse",
                         estimate_pp=round(e, 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                         p_boot=round(p, 4), n_items=len(w),
                         n_stories=w.index.map(story_of).nunique()))
        print(f"    forward minus reverse {e:+7.2f} [{lo:+6.2f} ; {hi:+6.2f}]  "
              f"p {min(ps):.4f}-{max(ps):.4f}")
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=int, default=0)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--dump-unrelated", action="store_true", dest="dump_unrelated")
    a = ap.parse_args()
    C = classify()
    if a.dump:
        dump(C, a.dump)
        return 0
    if a.dump_unrelated:
        dump_unrelated(C)
        return 0
    if a.score:
        score(C)
        return 0
    print("=" * 78)
    print("CaLM: WHERE THE QUESTION POINTS, AND WHAT THE NAMES DO THERE (exploratory)")
    print("=" * 78)
    n = C.direction.value_counts()
    sc = int(C[C.direction == "unrelated"].shared_cause.sum())
    print(f"  {len(C)} items: forward {n['forward']}, reverse {n['reverse']}, unrelated "
          f"{n['unrelated']} ({sc} of them share a cause)")
    R = main_tables(C)
    R.to_csv(OUT / "calm_direction.csv", index=False)
    print("\n  wrote results/calm/calm_direction.csv")
    H = hand_split(C)
    if H is not None:
        H.to_csv(OUT / "calm_direction_hand.csv", index=False)
        print("  wrote results/calm/calm_direction_hand.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
