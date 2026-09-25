"""Confirmatory phase: the six tests fixed in prereg/CONFIRMATORY.md, on a fresh sample.

    python scripts/analyze_confirmatory.py

Two parts, both run every time.

  1. CHECK. Every test below is computed by a function written here, on
     dataframes, so the confirmatory sample and a re-ask variant can go through
     it. Those functions are copies of the constructions that produced the
     exploratory numbers, and copies drift, so they are run on the exploratory
     n600 sample first and must return exactly the rows already published for
     it: pooled_headline.csv (H1), structure_arms.csv (H2),
     ladder5_steps_n600.csv (H3) and perturbation_conditional_slope.csv (H4).
     Any mismatch exits 1 before a confirmatory number is printed.

  2. CONFIRM. Once the confirmatory records exist (scripts/run_confirmatory.sh):
     the six tests, Holm-adjusted across the six, at each of five bootstrap
     seeds; a verdict that changes with the seed is reported as borderline, not
     as confirmed. Then the pre-registered sensitivity analysis (unparsed
     answers scored as wrong). Writes results/confirmatory.csv. Before the
     records exist it says so and writes nothing.

The tests, their direction and their decision rule are in prereg/CONFIRMATORY.md,
committed before any confirmatory call was sent. Do not change them here without
changing that file, and a change after the data exist is exploratory.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from stats import boot_cells, boot_items, boot_p

from analyze_querygroup import ARITH, IDENT, TIER
from pilot import read_ids

RAW = ROOT / "results" / "raw"
NBOOT = 4000
SEEDS = [20260907, 1, 2, 3, 4]         # the project's seed rule; the first is primary
ALPHA = 0.05
MARGIN = 5.0                            # H3a equivalence margin, pp
SEVEN = {"IV", "arrowhead", "confounding", "diamond", "diamondcut", "frontdoor", "mediation"}
STRICT_TYPES = {"nde", "nie", "det-counterfactual"}
LLAMA = "meta-llama/llama-3.3-70b-instruct"

# The confirmatory draw: pilot.py --n 1000 --seed 20260925 --sample-kmax 1
# --kmax 3 --drop-nonsense --exclude-ids prereg/excluded_ids.txt
CONF = dict(n=1000, seed=20260925, exclude="prereg/excluded_ids.txt")
# family -> (models, file prefix, re-ask cap or None)
FAMILIES = {"gpt": (TIER, "conf", None), "llama": ([LLAMA], "llamaconf", 1500)}

# (test, what, kind, predicted sign). kind "two" = two-sided bootstrap test,
# "tost" = two one-sided tests against +-MARGIN.
TESTS = [
    ("H1", "DiD [(KEEP - PSEUDO) | RAW] - [(KEEP - PSEUDO) | ORACLE]", "two", +1),
    ("H2", "PSEUDO: ORACLE minus DR_k1", "two", +1),
    ("H3a", "RAW: IRRELEVANT minus PERMUTE, equivalent within 5 pp", "tost", 0),
    ("H3b", "RAW: KEEP minus SYMBOL", "two", +1),
    ("H4-KEEP", "slope per reversed edge, estimand-changing items, KEEP", "two", -1),
    ("H4-PSEUDO", "slope per reversed edge, estimand-changing items, PSEUDO", "two", -1),
]


# ---------------------------------------------------------------------------
# the four constructions

def cell_id(d, model, cond, qs):
    s = d[(d.model == model) & (d.cond == cond) & (d.parsed == 1) & d.query_type.isin(qs)]
    return s.set_index("id").correct


def h1_matrix(K, L, models):
    """pool_samples.did_sample with POOL_LEXICON: one column per model, rows = id."""
    qs = set(K.query_type.unique()) - ARITH - IDENT
    cols = []
    for m in models:
        kr, lr = cell_id(K, m, "RAW", qs), cell_id(L, m, "RAW", qs)
        ka, la = cell_id(K, m, "ORACLE", qs), cell_id(L, m, "ORACLE", qs)
        i = kr.index.intersection(lr.index).intersection(ka.index).intersection(la.index)
        if len(i) < 10:
            continue
        cols.append(pd.Series(((kr[i] - lr[i]) - (ka[i] - la[i])).values, index=i, name=m))
    return pd.concat(cols, axis=1).groupby(level=0).mean()


def h2_matrix(L, models):
    """analyze_structure_arms section 4b, one sample."""
    qs = set(L.query_type.unique()) - ARITH - IDENT
    cols = []
    for m in models:
        o, r = cell_id(L, m, "ORACLE", qs), cell_id(L, m, "DR_k1", qs)
        i = o.index.intersection(r.index)
        if len(i) >= 10:
            cols.append(pd.Series((o[i] - r[i]).values, index=i, name=m))
    return pd.concat(cols, axis=1).groupby(level=0).mean()


def cell_item(d, model, cond="RAW"):
    s = d[(d.model == model) & (d.cond == cond) & (d.parsed == 1)]
    return s[~s.query_type.isin(ARITH | IDENT)].set_index("item").correct


def step_matrix(hi, lo, models):
    """analyze_ladder5: one row per model, items sorted, NaN where a model lacks the pair."""
    by = {}
    for m in models:
        x, y = cell_item(hi, m), cell_item(lo, m)
        idx = x.index.intersection(y.index)
        if len(idx):
            by[m] = pd.Series(x[idx].values - y[idx].values, index=idx)
    items = sorted(set().union(*[set(s.index) for s in by.values()]))
    return np.vstack([by[m].reindex(items).values for m in by])


def paired_causal(d, cond):
    """classify_perturbations.paired_causal: per item, (cond - RAW) averaged over models."""
    d = d[~d.query_type.isin(ARITH | IDENT)]
    cols = []
    for m in sorted(d.model.unique()):
        r = d[(d.model == m) & (d.cond == "RAW")].set_index("item").correct
        c = d[(d.model == m) & (d.cond == cond)].set_index("item").correct
        i = r.index.intersection(c.index)
        if len(i) >= 5:
            cols.append(pd.Series((c[i] - r[i]).values, index=i, name=m))
    return pd.concat(cols, axis=1).mean(axis=1).dropna() if cols else pd.Series(dtype=float)


def h4_slopes(d, cls, lex):
    """classify_perturbations.conditional_slope, one sample: items whose k=1 draw
    already changes the estimand, one slope per item across its own k = 1, 2, 3."""
    d = d[(d.parsed == 1) & d.graph_id.isin(SEVEN)]
    M = pd.concat({k: paired_causal(d, f"DR_k{k}") for k in (1, 2, 3)}, axis=1).dropna()
    flag = cls[(cls.cond == "DR_k1") & (cls.lexicon == lex)].set_index("item").unchanged_qt
    M = M[flag.reindex(M.index).eq(False).values]
    return ((M[3] - M[1]) / 2).values


def classes(n, seed, exclude_ids):
    """The estimand verdict of every DR draw actually shown (classify_perturbations)."""
    from classify_perturbations import check_replay, classify
    N = classify(n, 1, 3, arms=("DR",), seed=seed, exclude_ids=exclude_ids)
    # Only the causal query types are used by any test, and the confirmatory
    # run sent only those (pilot.py --causal-only), so the replay proof covers
    # exactly them. Checking every type failed on the first confirmatory run:
    # prompts never sent are not in the cache. No test changes.
    N = N[~N.query_type.isin(ARITH | IDENT)].reset_index(drop=True)
    check_replay(N)
    strict = N.estimand_unchanged & ~N.touches_causal_path
    N["unchanged_qt"] = np.where(N.query_type.isin(STRICT_TYPES), strict, N.estimand_unchanged)
    return N


# ---------------------------------------------------------------------------
# one pass of the six tests

def tost_p(draws, n):
    lo = max(float((np.asarray(draws) <= -MARGIN).mean()), 1.0 / n)
    hi = max(float((np.asarray(draws) >= MARGIN).mean()), 1.0 / n)
    return max(lo, hi)


def run_tests(S, models, cls, seed):
    """S: {"main": {lex: df}, "slope": {lex: df}, "ladder": {lex: df}}. One row per test."""
    out = {}
    W = h1_matrix(S["main"]["KEEP"], S["main"]["PSEUDO"], models)
    e, lo, hi, p = boot_cells(W, seed, NBOOT)
    out["H1"] = dict(estimate_pp=e, ci_lo=lo, ci_hi=hi, p_boot=p, n_items=len(W))

    W = h2_matrix(S["main"]["PSEUDO"], models)
    e, lo, hi, p = boot_cells(W, seed, NBOOT)
    out["H2"] = dict(estimate_pp=e, ci_lo=lo, ci_hi=hi, p_boot=p, n_items=len(W))

    Lad = S["ladder"]
    for t, a, b in (("H3a", "IRRELEVANT", "PERMUTE"), ("H3b", "KEEP", "SYMBOL")):
        M = step_matrix(Lad[a], Lad[b], models)
        e, dr = boot_cells(M, seed, NBOOT, axis=1, return_draws=True)
        row = dict(estimate_pp=e, ci_lo=np.percentile(dr, 2.5), ci_hi=np.percentile(dr, 97.5),
                   p_boot=boot_p(dr, NBOOT), n_items=M.shape[1])
        if t == "H3a":
            row |= dict(ci90_lo=np.percentile(dr, 5), ci90_hi=np.percentile(dr, 95),
                        p_boot=tost_p(dr, NBOOT))
        out[t] = row

    for lex in ("KEEP", "PSEUDO"):
        x = h4_slopes(S["slope"][lex], cls, lex)
        e, lo, hi, p = boot_items(x, seed, NBOOT)
        out[f"H4-{lex}"] = dict(estimate_pp=e, ci_lo=lo, ci_hi=hi, p_boot=p, n_items=len(x))
    return out


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps, kind="stable")
    adj, run = np.empty_like(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def verdicts(res):
    adj = holm([res[t]["p_boot"] for t, *_ in TESTS])
    v = {}
    for (t, _, kind, sign), pa in zip(TESTS, adj):
        if kind == "tost":
            v[t] = ("confirmed" if pa < ALPHA else "not confirmed", pa)
        elif pa >= ALPHA:
            v[t] = ("not confirmed", pa)
        else:
            v[t] = ("confirmed" if np.sign(res[t]["estimate_pp"]) == sign else "contradicted", pa)
    return v


# ---------------------------------------------------------------------------
# data

def read(name):
    return pd.read_csv(RAW / name)


def n600_data():
    main = {lex: read(f"pilot_raw_n600{lex}.csv") for lex in ("KEEP", "PSEUDO")}
    slope = {lex: pd.concat([main[lex], read(f"pilot_raw_n600arms{lex}.csv")], ignore_index=True)
             for lex in ("KEEP", "PSEUDO")}
    ladder = {"KEEP": main["KEEP"], "PSEUDO": main["PSEUDO"]}
    ladder |= {lex: read(f"pilot_raw_n600ladder{lex}.csv")
               for lex in ("PERMUTE", "IRRELEVANT", "SYMBOL")}
    return dict(main=main, slope=slope, ladder=ladder)


def conf_files(prefix):
    main = {lex: RAW / f"pilot_raw_{prefix}{lex}.csv" for lex in ("KEEP", "PSEUDO")}
    lad = {lex: RAW / f"pilot_raw_{prefix}ladder{lex}.csv"
           for lex in ("PERMUTE", "IRRELEVANT", "SYMBOL")}
    return main, lad


def with_recap(d, path):
    """Every answer cut off at the cap replaced by its re-ask (as analyze_second_family)."""
    if not path.exists():
        return None
    r = pd.read_csv(path)
    key = ["model", "item", "cond"]
    cut = d.finish == "length"
    if set(map(tuple, r[key].values)) != set(map(tuple, d.loc[cut, key].values)):
        raise SystemExit(f"{path.name}: the re-asked rows are not the cut-off rows")
    return pd.concat([d[~cut], r], ignore_index=True)


def conf_data(prefix, recap):
    main_f, lad_f = conf_files(prefix)
    if not all(p.exists() for p in [*main_f.values(), *lad_f.values()]):
        return None
    main = {lex: pd.read_csv(p) for lex, p in main_f.items()}
    lad = {lex: pd.read_csv(p) for lex, p in lad_f.items()}
    if recap:
        rec = {}
        for lex, p in [*main_f.items(), *lad_f.items()]:
            src = main[lex] if lex in main else lad[lex]
            rec[lex] = with_recap(src, p.with_name(p.stem + f"_recap{recap}.csv"))
        if any(v is None for v in rec.values()):
            raise SystemExit(f"{prefix}: a re-ask file is missing")
        main = {lex: rec[lex] for lex in main}
        lad = {lex: rec[lex] for lex in lad}
    ladder = {"KEEP": main["KEEP"], "PSEUDO": main["PSEUDO"]} | lad
    return dict(main=main, slope=main, ladder=ladder)


def unparsed_as_wrong(S):
    f = lambda d: d.assign(correct=np.where(d.parsed == 1, d.correct, 0), parsed=1)
    return {k: {lex: f(d) for lex, d in v.items()} for k, v in S.items()}


# ---------------------------------------------------------------------------

def check_n600() -> None:
    print("=" * 78)
    print("1. CHECK - the confirmatory code on n600 must give the published rows")
    print("=" * 78)
    cls = classes(600, 20260907, None)
    res = run_tests(n600_data(), TIER, cls, SEEDS[0])

    ph = pd.read_csv(ROOT / "results" / "pooled_headline.csv").set_index("pooling")
    sa = pd.read_csv(ROOT / "results" / "structure_arms.csv").set_index("quantity")
    ld = pd.read_csv(ROOT / "results" / "ladder5_steps_n600.csv").set_index("step")
    cs = pd.read_csv(ROOT / "results" / "perturbation_conditional_slope.csv")
    cs = cs[(cs["sample"] == "n600") & (cs.quantity == "slope per reversed edge")].set_index("lexicon")
    want = {
        "H1": (ph.loc["n600", ["did_pp", "ci_lo", "ci_hi", "p_boot"]], "pooled_headline.csv"),
        "H2": (sa.loc["PSEUDO branch, ORACLE minus DR_k1 | n600",
                      ["estimate_pp", "ci_lo", "ci_hi", "p_boot"]], "structure_arms.csv"),
        "H3a": (ld.loc["IRRELEVANT -> PERMUTE", ["delta_pp", "ci_lo", "ci_hi"]],
                "ladder5_steps_n600.csv"),
        "H3b": (ld.loc["KEEP -> SYMBOL", ["delta_pp", "ci_lo", "ci_hi", "p"]],
                "ladder5_steps_n600.csv"),
        "H4-KEEP": (cs.loc["KEEP", ["delta_pp", "ci_lo", "ci_hi", "p_boot"]],
                    "perturbation_conditional_slope.csv"),
        "H4-PSEUDO": (cs.loc["PSEUDO", ["delta_pp", "ci_lo", "ci_hi", "p_boot"]],
                      "perturbation_conditional_slope.csv"),
    }
    bad = 0
    for t, (row, src) in want.items():
        r = res[t]
        # H3a's p here is the TOST p, which ladder5 does not publish
        got = [r["estimate_pp"], r["ci_lo"], r["ci_hi"]] + ([] if t == "H3a" else [r["p_boot"]])
        exp = [float(x) for x in row.values]
        digits = [2, 2, 2, 4][:len(exp)]
        ok = all(round(g, k) == round(x, k) for g, x, k in zip(got, exp, digits))
        bad += not ok
        print(f"  {t:10s} {'OK  ' if ok else 'DIFF'} {r['estimate_pp']:+7.2f} "
              f"[{r['ci_lo']:+6.2f} ; {r['ci_hi']:+6.2f}]  published {exp[0]:+7.2f} "
              f"[{exp[1]:+6.2f} ; {exp[2]:+6.2f}]  ({src})")
    if bad:
        raise SystemExit(f"\n  {bad} test(s) do not reproduce their published n600 row. "
                         f"The confirmatory code is not the exploratory analysis; stopping.")
    print("\n  All six reproduce. The confirmatory tests are the exploratory constructions.")


def confirm(family, S, models, cls, label):
    rows = []
    per_seed = {s: run_tests(S, models, cls, s) for s in SEEDS}
    v_seed = {s: verdicts(per_seed[s]) for s in SEEDS}
    prim = per_seed[SEEDS[0]]
    for t, what, kind, sign in TESTS:
        vs = {v_seed[s][t][0] for s in SEEDS}
        verdict = vs.pop() if len(vs) == 1 else "borderline (changes with the bootstrap seed)"
        ps = [per_seed[s][t]["p_boot"] for s in SEEDS]
        r = prim[t]
        rows.append(dict(family=family, analysis=label, test=t, quantity=what,
                         predicted={1: "positive", -1: "negative", 0: "within +-5 pp"}[sign],
                         estimate_pp=round(r["estimate_pp"], 2), ci_lo=round(r["ci_lo"], 2),
                         ci_hi=round(r["ci_hi"], 2),
                         ci90_lo=round(r["ci90_lo"], 2) if "ci90_lo" in r else "",
                         ci90_hi=round(r["ci90_hi"], 2) if "ci90_hi" in r else "",
                         p_boot=round(r["p_boot"], 4),
                         p_holm=round(v_seed[SEEDS[0]][t][1], 4),
                         p_min_seeds=round(min(ps), 4), p_max_seeds=round(max(ps), 4),
                         n_items=r["n_items"], verdict=verdict))
        print(f"  {t:10s} {r['estimate_pp']:+7.2f} [{r['ci_lo']:+6.2f} ; {r['ci_hi']:+6.2f}]"
              f"  p={r['p_boot']:.4f}  Holm {v_seed[SEEDS[0]][t][1]:.4f}  n={r['n_items']:4d}"
              f"  {verdict}")
    return rows


def main() -> int:
    check_n600()
    exclude = read_ids(CONF["exclude"])
    rows = []
    for family, (models, prefix, recap) in FAMILIES.items():
        S = conf_data(prefix, None)
        if S is None:
            print(f"\n  {family}: no confirmatory records yet (results/raw/pilot_raw_{prefix}*.csv)")
            continue
        print("\n" + "=" * 78)
        print(f"2. CONFIRM - {family}, the six pre-registered tests")
        print("=" * 78)
        cls = classes(CONF["n"], CONF["seed"], exclude)
        ids = pd.concat([d.id for d in S["main"].values()]).astype(int)
        if ids.isin(exclude).any():
            raise SystemExit("a confirmatory record carries an excluded id")
        for lex, d in {**S["main"], **S["ladder"]}.items():
            print(f"  parse rate {lex:10s} {100 * d.parsed.mean():6.2f}%  ({len(d)} rows)")
        print("\n  primary")
        rows += confirm(family, S, models, cls, "primary")
        print("\n  sensitivity: unparsed answers scored as wrong")
        rows += confirm(family, unparsed_as_wrong(S), models, cls, "unparsed scored as wrong")
        if recap:
            R = conf_data(prefix, recap)
            print(f"\n  sensitivity: cut-off answers re-asked at {recap} tokens")
            rows += confirm(family, R, models, cls, f"re-asked at {recap}")
    if rows:
        pd.DataFrame(rows).to_csv(ROOT / "results" / "confirmatory.csv", index=False)
        print("\n  wrote results/confirmatory.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
