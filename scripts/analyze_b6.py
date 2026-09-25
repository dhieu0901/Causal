"""B6: does a model rely on the graph it is given or obey the instruction to
use it, and is the harm of a wrong graph the harm of cutting the causal path?

    python scripts/analyze_b6.py

The tests, their direction and the decision rule are in prereg/B6.md, committed
and pushed before the first B6 call. Three parts, all run every time.

  1. CHECK. The five tests are computed by the functions below on one row per
     reversal draw: its group (analyze_answer_change.py: estimand kept;
     estimand changed, answer kept; answer changed, with the path cut or kept)
     and its harm, DR minus ORACLE averaged over models and over the KEEP and
     PSEUDO lexicons. Before any B6 record is read they run on the exploratory
     draws of price400, n600 and the confirmatory sample, and M1 and M2 must
     equal the "both" rows analyze_answer_change.py publishes in
     answer_change_harm.csv. A mismatch exits 1.
  2. CONFIRM, once results/cladder/raw/pilot_raw_b6{KEEP,PSEUDO}.csv exist
     (scripts/run_b6.sh): the five tests, Holm across the five, at five
     bootstrap seeds; a verdict that changes with the seed is borderline. Then
     the registered sensitivity analysis (unparsed answers scored as wrong)
     and the registered descriptive quantities. Writes results/cladder/b6.csv
     and b6_descriptive.csv.
  3. PROBE, once results/cladder/raw/probe_direction_raw.csv exists
     (scripts/probe_direction.py): descriptive. For the B5 items, the share of
     name pairs whose real-world direction, asked directly, agrees with the
     item's edge, runs against it, or is "neither", under the real names and
     under the permuted names of H3a. Writes results/cladder/probe_direction.csv.

A change to a test after the B6 data exist is exploratory and must say so.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from analyze_answer_change import Matcher, draws_for
from pilot import read_ids
from stats import boot_p, cluster_boot

RESULTS = ROOT / "results" / "cladder"
RAW = RESULTS / "raw"
NBOOT = 4000
SEEDS = [20260907, 1, 2, 3, 4]
ALPHA = 0.05
MARGIN = 5.0
MODELS = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]
# pilot.py --n 340 --seed 20260926 --sample-kmax 1 --kmax 3 --drop-nonsense
# --query-types ate,ett --exclude-ids prereg/excluded_ids_b6.txt --no-instr
B6 = dict(n_items=340, sample_kmax=1, kmax=3, seed=20260926,
          exclude_ids=read_ids("prereg/excluded_ids_b6.txt"), query_types=["ate", "ett"])
KEEP_ANSWER = "estimand changed, answer kept"
TESTS = [
    ("M1", "path cut: DR minus ORACLE", "negative"),
    ("M2", "estimand changed, answer kept: DR minus ORACLE, within +-5", "within +-5"),
    ("M3", "path cut minus answer changed with the path kept (DR minus ORACLE)", "negative"),
    ("R1", "path cut, no instruction: DR_NI minus ORACLE_NI", "negative"),
    ("R2", "path cut: harm with the instruction minus harm without, within +-5", "within +-5"),
]


# ---------------------------------------------------------------- the tests

def units(x: pd.DataFrame) -> list[np.ndarray]:
    """Row positions of each (sample, item), in sorted order: the bootstrap unit."""
    return list(x.groupby(["sample", "item"]).indices.values())


def boot_mean(x: pd.DataFrame, col: str, seed: int):
    """Mean over draws, items resampled with all their draws; the construction of
    analyze_answer_change.boot_groups, returning the draws as well."""
    keys, vals = units(x), x[col].values
    d = cluster_boot(len(keys), lambda i: 100 * vals[np.concatenate([keys[j] for j in i])].mean(),
                     seed, NBOOT)
    return 100 * vals.mean(), d


def boot_diff(a: pd.DataFrame, b: pd.DataFrame, col: str, seed: int):
    """mean(a) - mean(b); items resampled once, both groups taken from the same draw."""
    a = a.assign(_g="a")
    b = b.assign(_g="b")
    x = pd.concat([a, b], ignore_index=True)
    keys = units(x)
    va, vb = x[col].values, x._g.values == "a"

    def stat(i):
        idx = np.concatenate([keys[j] for j in i])
        m = vb[idx]
        if not m.any() or m.all():
            return np.nan
        return 100 * (va[idx][m].mean() - va[idx][~m].mean())

    d = cluster_boot(len(keys), stat, seed, NBOOT)
    return 100 * (a[col].mean() - b[col].mean()), d


def tost_p(d):
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    return max(max(float((d <= -MARGIN).mean()), 1 / NBOOT),
               max(float((d >= MARGIN).mean()), 1 / NBOOT))


def summary(est, d, tost=False):
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    row = dict(estimate_pp=est, ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5),
               p_boot=tost_p(d) if tost else boot_p(d, NBOOT))
    if tost:
        row |= dict(ci90_lo=np.percentile(d, 5), ci90_hi=np.percentile(d, 95))
    return row


def cut(x):
    return x[(x.group == "answer changed") & (x.route == "no path")]


def run_tests(X: pd.DataFrame, seed: int, with_ni=True) -> dict:
    out = {}
    e, d = boot_mean(cut(X), "h", seed)
    out["M1"] = summary(e, d) | dict(n_draws=len(cut(X)))
    k = X[X.group == KEEP_ANSWER]
    e, d = boot_mean(k, "h", seed)
    out["M2"] = summary(e, d, tost=True) | dict(n_draws=len(k))
    kept = X[(X.group == "answer changed") & (X.route == "backdoor")]
    e, d = boot_diff(cut(X), kept, "h", seed)
    out["M3"] = summary(e, d) | dict(n_draws=len(cut(X)) + len(kept))
    if with_ni:
        c = cut(X).dropna(subset=["h_ni"])
        e, d = boot_mean(c, "h_ni", seed)
        out["R1"] = summary(e, d) | dict(n_draws=len(c))
        c = c.dropna(subset=["h"]).assign(delta=lambda t: t.h - t.h_ni)
        e, d = boot_mean(c, "delta", seed)
        out["R2"] = summary(e, d, tost=True) | dict(n_draws=len(c))
    return out


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    adj, run = np.empty_like(ps), 0.0
    for r, i in enumerate(np.argsort(ps, kind="stable")):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def verdicts(res):
    adj = holm([res[t]["p_boot"] for t, *_ in TESTS])
    v = {}
    for (t, _, pred), pa in zip(TESTS, adj):
        if pred == "within +-5":
            v[t] = ("confirmed" if pa < ALPHA else "not confirmed", pa)
        elif pa >= ALPHA:
            v[t] = ("not confirmed", pa)
        else:
            v[t] = ("confirmed" if res[t]["estimate_pp"] < 0 else "contradicted", pa)
    return v


# ---------------------------------------------------------------- 1. check

def exploratory_draws() -> pd.DataFrame:
    D = pd.read_csv(RESULTS / "answer_change.csv")
    keys = ["sample", "item", "cond"]
    both = (D.groupby(keys)[["vs_ORACLE"]].mean()
            .join(D[D.lexicon == "KEEP"].set_index(keys)[["group", "route"]]).reset_index())
    both = both.rename(columns={"vs_ORACLE": "h"})
    return both.dropna(subset=["h"])


def check() -> None:
    print("=" * 78)
    print("1. CHECK - the test functions on the exploratory draws")
    print("=" * 78)
    X = exploratory_draws()
    res = run_tests(X, SEEDS[0], with_ni=False)
    H = pd.read_csv(RESULTS / "answer_change_harm.csv")
    H = H[(H.lexicon == "both") & (H.reference == "ORACLE") & (H.cond == "all")
          & (H.gold == "all")]
    want = {"M1": H[(H.group == "answer changed") & (H.route == "no path")],
            "M2": H[(H.group == KEEP_ANSWER) & (H.route == "all")]}
    bad = 0
    for t, row in want.items():
        if len(row) != 1:
            raise SystemExit(f"  {t}: no single published 'both' row in answer_change_harm.csv")
        row = row.iloc[0]
        r = res[t]
        got = [round(r["estimate_pp"], 2), round(r["ci_lo"], 2), round(r["ci_hi"], 2)]
        exp = [round(row.harm_pp, 2), round(row.ci_lo, 2), round(row.ci_hi, 2)]
        ok = got == exp
        bad += not ok
        print(f"  {t}  {'OK  ' if ok else 'DIFF'} {got[0]:+7.2f} [{got[1]:+6.2f} ; {got[2]:+6.2f}]"
              f"  published {exp[0]:+7.2f} [{exp[1]:+6.2f} ; {exp[2]:+6.2f}]")
    r = res["M3"]
    print(f"  M3  (no published row) {r['estimate_pp']:+7.2f} [{r['ci_lo']:+6.2f} ; "
          f"{r['ci_hi']:+6.2f}]  p={r['p_boot']:.4f}")
    r = res["M2"]
    print(f"  M2  90% interval [{r['ci90_lo']:+6.2f} ; {r['ci90_hi']:+6.2f}], TOST p {r['p_boot']:.4f}")
    if bad:
        raise SystemExit("\n  The test functions do not reproduce the published rows; stopping.")
    print("\n  M1 and M2 reproduce. The B6 tests are the exploratory constructions.")


# ---------------------------------------------------------------- 2. confirm

def correct_by(d, cond):
    s = d[(d.cond == cond) & (d.parsed == 1)]
    return s.set_index(["model", "item"]).correct


def b6_draws(unparsed_wrong=False) -> pd.DataFrame | None:
    f = {lex: RAW / f"pilot_raw_b6{lex}.csv" for lex in ("KEEP", "PSEUDO")}
    if not all(p.exists() for p in f.values()):
        return None
    G = draws_for("b6", Matcher(), kw=B6)
    G = G[G.lexicon == "KEEP"].set_index(["item", "cond"])[["group", "route", "gold"]]
    rows = []
    for lex, p in f.items():
        d = pd.read_csv(p)
        if unparsed_wrong:
            d = d.assign(correct=np.where(d.parsed == 1, d.correct, 0), parsed=1)
        for k in (1, 2, 3):
            for tag, dr, orc in (("h", f"DR_k{k}", "ORACLE"), ("h_ni", f"DR_k{k}_NI", "ORACLE_NI")):
                a, b = correct_by(d, dr), correct_by(d, orc)
                i = a.index.intersection(b.index)
                s = (a[i] - b[i]).groupby(level="item").mean()
                rows.append(pd.DataFrame({"item": s.index, "cond": f"DR_k{k}", "lexicon": lex,
                                          "col": tag, "v": s.values}))
    V = pd.concat(rows).groupby(["item", "cond", "col"]).v.mean().unstack("col").reset_index()
    X = V.join(G, on=["item", "cond"], how="inner").assign(sample="b6")
    return X.dropna(subset=["group"])


def confirm(X, label):
    rows, per_seed = [], {s: run_tests(X, s) for s in SEEDS}
    v_seed = {s: verdicts(per_seed[s]) for s in SEEDS}
    for t, what, pred in TESTS:
        vs = {v_seed[s][t][0] for s in SEEDS}
        verdict = vs.pop() if len(vs) == 1 else "borderline (changes with the bootstrap seed)"
        ps = [per_seed[s][t]["p_boot"] for s in SEEDS]
        r = per_seed[SEEDS[0]][t]
        rows.append(dict(analysis=label, test=t, quantity=what, predicted=pred,
                         estimate_pp=round(r["estimate_pp"], 2), ci_lo=round(r["ci_lo"], 2),
                         ci_hi=round(r["ci_hi"], 2),
                         ci90_lo=round(r["ci90_lo"], 2) if "ci90_lo" in r else "",
                         ci90_hi=round(r["ci90_hi"], 2) if "ci90_hi" in r else "",
                         p_boot=round(r["p_boot"], 4), p_holm=round(v_seed[SEEDS[0]][t][1], 4),
                         p_min_seeds=round(min(ps), 4), p_max_seeds=round(max(ps), 4),
                         n_draws=r["n_draws"], n_items=None, verdict=verdict))
        print(f"  {t}  {r['estimate_pp']:+7.2f} [{r['ci_lo']:+6.2f} ; {r['ci_hi']:+6.2f}]"
              f"  p={r['p_boot']:.4f}  Holm {v_seed[SEEDS[0]][t][1]:.4f}  draws={r['n_draws']:4d}"
              f"  {verdict}")
    return rows


def descriptive(X) -> list[dict]:
    """Registered as descriptive in prereg/B6.md: no verdicts."""
    rows = []

    def add(what, x, col):
        if len(x) < 10:
            return
        e, d = boot_mean(x, col, SEEDS[0])
        s = summary(e, d)
        rows.append(dict(quantity=what, estimate_pp=round(s["estimate_pp"], 2),
                         ci_lo=round(s["ci_lo"], 2), ci_hi=round(s["ci_hi"], 2),
                         p_boot=round(s["p_boot"], 4), n_draws=len(x)))

    for g in ("estimand kept", KEEP_ANSWER, "answer changed"):
        add(f"{g}: DR minus ORACLE", X[X.group == g], "h")
        add(f"{g}: DR_NI minus ORACLE_NI", X[X.group == g].dropna(subset=["h_ni"]), "h_ni")
    kept = X[(X.group == "answer changed") & (X.route == "backdoor")]
    add("answer changed, path kept: DR minus ORACLE", kept, "h")
    add("answer changed, path kept: DR_NI minus ORACLE_NI", kept.dropna(subset=["h_ni"]), "h_ni")
    for k in (1, 2, 3):
        c = cut(X[X.cond == f"DR_k{k}"])
        add(f"path cut, DR_k{k}: DR minus ORACLE", c, "h")
    return rows


def per_item(d, a, b):
    """cond a minus cond b per item, averaged over models (parsed pairs)."""
    x, y = correct_by(d, a), correct_by(d, b)
    i = x.index.intersection(y.index)
    return (x[i] - y[i]).groupby(level="item").mean()


def arm_contrasts() -> list[dict]:
    rows = []
    for lex in ("KEEP", "PSEUDO"):
        d = pd.read_csv(RAW / f"pilot_raw_b6{lex}.csv")
        for a, b in (("ORACLE", "RAW"), ("ORACLE_NI", "RAW"), ("ORACLE_NI", "ORACLE"),
                     ("DR_k1", "RAW"), ("DR_k1_NI", "RAW")):
            v = per_item(d, a, b)
            x = pd.DataFrame({"sample": "b6", "item": v.index, "v": v.values})
            e, dd = boot_mean(x, "v", SEEDS[0])
            s = summary(e, dd)
            rows.append(dict(quantity=f"{lex}: {a} minus {b}", estimate_pp=round(s["estimate_pp"], 2),
                             ci_lo=round(s["ci_lo"], 2), ci_hi=round(s["ci_hi"], 2),
                             p_boot=round(s["p_boot"], 4), n_draws=len(x)))
    return rows


# ---------------------------------------------------------------- 3. probe

def probe() -> None:
    f = RAW / "probe_direction_raw.csv"
    if not f.exists():
        print("\n  probe: no records yet (results/cladder/raw/probe_direction_raw.csv)")
        return
    P = pd.read_csv(f)
    rows = []
    for (lex, m), g in P.groupby(["lexicon", "model"]):
        n = len(g)
        rel = g.relation.value_counts()
        rows.append(dict(lexicon=lex, model=m, n_pairs=n,
                         agrees_pct=round(100 * rel.get("agrees", 0) / n, 2),
                         reversed_pct=round(100 * rel.get("reversed", 0) / n, 2),
                         neither_pct=round(100 * rel.get("neither", 0) / n, 2),
                         unparsed_pct=round(100 * rel.get("unparsed", 0) / n, 2)))
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "probe_direction.csv", index=False)
    print("\n" + "=" * 78)
    print("3. PROBE - the real-world direction of each edge's two names, asked directly")
    print("=" * 78)
    print(R.to_string(index=False))
    print("  wrote results/cladder/probe_direction.csv")


def main() -> int:
    check()
    X = b6_draws()
    if X is None:
        print("\n  B6: no records yet (results/cladder/raw/pilot_raw_b6*.csv)")
    else:
        print("\n" + "=" * 78)
        print("2. CONFIRM - the five pre-registered tests")
        print("=" * 78)
        print(f"  {len(X)} reversal draws on {X.item.nunique()} items")
        print("\n  primary")
        rows = confirm(X, "primary")
        print("\n  sensitivity: unparsed answers scored as wrong")
        rows += confirm(b6_draws(unparsed_wrong=True), "unparsed scored as wrong")
        pd.DataFrame(rows).to_csv(RESULTS / "b6.csv", index=False)
        desc = pd.DataFrame(descriptive(X) + arm_contrasts())
        desc.to_csv(RESULTS / "b6_descriptive.csv", index=False)
        print("\n  descriptive:")
        print(desc.to_string(index=False))
        print("\n  wrote results/cladder/b6.csv, b6_descriptive.csv")
    probe()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
