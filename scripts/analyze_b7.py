"""B7, exploratory: what gpt-5.6-luna does with the graph, beside GPT-4.1.

    python scripts/analyze_b7.py

The six pre-registered tests of B7 (prereg/B7.md) are computed by
analyze_confirmatory.py, family "luna". prereg/B7.md section 5 reads any
comparison between gpt-5.6-luna and GPT-4.1 on the same items as descriptive,
not a test. This file is that description; nothing in it was pre-registered.
Both families answered the same 484 confirmatory causal items with the same
prompts. GPT-4.1 appears twice: as its three models averaged per item, as in
every B5 table, and as each model on its own. The average mixes a model that
barely reacts to the direction of an edge (gpt-4.1-nano) with two that do, so
a comparison with one model is read against each of the three as well
(review round 13).

  1. accuracy per lexicon and condition, and each condition against RAW:
     analyze_b5_vs_raw's contrasts with its estimator
  2. the lexical ladder under RAW, the steps of REPORT section 4.3: why H3a
     failed on luna
  3. where the harm of a reversed edge lives, ate and ett items: the groups of
     analyze_answer_change.py, DR minus ORACLE, one value per draw averaged
     over the two lexicons (the unit of the B6 tests), with analyze_b6's test
     functions, read here without verdicts

Each part first recomputes, with the functions it uses, rows that are already
published for GPT-4.1 (b5_vs_raw.csv, confirmatory.csv, answer_change.csv) and
stops if one differs. Every p is a range over the project's five bootstrap
seeds.

Writes: results/cladder/b7_accuracy.csv
        results/cladder/b7_descriptive.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

import analyze_answer_change as AC
from stats import boot_cells, boot_items, boot_p
from analyze_querygroup import ARITH, IDENT, TIER
from analyze_b5_vs_raw import ROWS as VS_RAW_ROWS, paired_vs
from analyze_confirmatory import FAMILIES, conf_data, step_matrix
from analyze_b6 import run_tests

RESULTS = ROOT / "results" / "cladder"
NBOOT = 4000
SEEDS = [20260907, 1, 2, 3, 4]
FAMS = ("gpt", "luna")
LUNA = FAMILIES["luna"][0]
CONDS = ("RAW", "ORACLE", "DR_k1", "DR_k2", "DR_k3")
# (high, low) under RAW; the row reads "high minus low", as in REPORT section 4.3.
STEPS = [("KEEP", "IRRELEVANT"), ("IRRELEVANT", "PERMUTE"), ("IRRELEVANT", "SYMBOL"),
         ("PERMUTE", "SYMBOL"), ("SYMBOL", "PSEUDO"), ("KEEP", "SYMBOL")]


def seeds_row(per):
    """per: [(est, lo, hi, p)] over SEEDS -> the columns every row carries."""
    e, lo, hi, p = per[0]
    ps = [q[3] for q in per]
    return dict(estimate_pp=round(e, 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                p_boot=round(p, 4), p_min_seeds=round(min(ps), 4), p_max_seeds=round(max(ps), 4))


def stop_if_differs(what, got, want):
    got, want = [round(float(x), 2) for x in got], [round(float(x), 2) for x in want]
    print(f"  {what:44s} {'OK  ' if got == want else 'DIFF'} {got}  published {want}")
    if got != want:
        raise SystemExit(f"  {what}: does not reproduce the published row; stopping.")


# ------------------------------------------------------------ 1. against RAW
def accuracy(S, family):
    rows = []
    for lex, d in S["main"].items():
        d = d[~d.query_type.isin(ARITH | IDENT)]
        for cond in CONDS:
            x = d[d.cond == cond]
            per_model = x[x.parsed == 1].groupby("model").correct.mean()
            rows.append(dict(family=family, lexicon=lex, cond=cond,
                             accuracy_pct=round(100 * per_model.mean(), 2),
                             parse_pct=round(100 * x.parsed.mean(), 2),
                             n_models=len(per_model), n_rows=len(x)))
    for lex, d in S["ladder"].items():
        if lex in S["main"]:
            continue
        x = d[(d.cond == "RAW") & ~d.query_type.isin(ARITH | IDENT)]
        per_model = x[x.parsed == 1].groupby("model").correct.mean()
        rows.append(dict(family=family, lexicon=lex, cond="RAW",
                         accuracy_pct=round(100 * per_model.mean(), 2),
                         parse_pct=round(100 * x.parsed.mean(), 2),
                         n_models=len(per_model), n_rows=len(x)))
    return rows


def vs_raw(S, family):
    rows = []
    for lex, cond, ref in VS_RAW_ROWS:
        d = S["main"][lex]
        v = paired_vs(d[d.parsed == 1], cond, ref)
        rows.append(dict(part="1 against RAW", family=family, lexicon=lex,
                         quantity=f"{cond} minus {ref}", n_items=len(v), n_draws=None)
                    | seeds_row([boot_items(v.values, s, NBOOT) for s in SEEDS]))
    return rows


# ------------------------------------------------------------ 2. the ladder
def ladder(S, family, models):
    rows = []
    for hi, lo in STEPS:
        M = step_matrix(S["ladder"][hi], S["ladder"][lo], models)
        per = []
        for s in SEEDS:
            e, dr = boot_cells(M, s, NBOOT, axis=1, return_draws=True)
            per.append((e, np.percentile(dr, 2.5), np.percentile(dr, 97.5), boot_p(dr, NBOOT)))
        rows.append(dict(part="2 ladder, RAW", family=family, lexicon="ladder",
                         quantity=f"{hi} minus {lo}", n_items=M.shape[1], n_draws=None)
                    | seeds_row(per))
    return rows


# ------------------------------------------------------------ 3. where the harm lives
def harm(prefix, lex, models):
    """analyze_answer_change.harm against ORACLE, on `prefix`'s records and
    `models` only: per (item, cond), DR minus ORACLE averaged over the models,
    parsed answers only. main() checks it against answer_change.csv."""
    d = pd.read_csv(RESULTS / "raw" / f"pilot_raw_{prefix}{lex}.csv")
    d = d[(d.parsed == 1) & d.query_type.isin(AC.TYPES) & d.model.isin(models)]
    out = []
    for m in sorted(d.model.unique()):
        dm = d[d.model == m]
        r = dm[dm.cond == "ORACLE"].drop_duplicates("item").set_index("item").correct
        for k in (1, 2, 3):
            c = dm[dm.cond == f"DR_k{k}"].drop_duplicates("item").set_index("item").correct
            i = r.index.intersection(c.index)
            out.append(pd.DataFrame({"item": i, "cond": f"DR_k{k}", "diff": (c[i] - r[i]).values}))
    return pd.concat(out).groupby(["item", "cond"])["diff"].mean()


def draws(prefix, models):
    """answer_change.csv's groups of the confirmatory draws, with the harm of
    `models` in `prefix`'s records. The draws are the confirmatory sample's
    (same items, seed and prompts), so their groups do not depend on which
    model answered."""
    G = pd.read_csv(RESULTS / "answer_change.csv")
    G = G[G["sample"].eq("conf") & G.lexicon.eq("KEEP")][["item", "cond", "group", "route",
                                                          "implied_answer"]]
    h = pd.concat([harm(prefix, lex, models) for lex in ("KEEP", "PSEUDO")])
    h = h.groupby(level=["item", "cond"]).mean().rename("h").reset_index()
    X = G.merge(h, on=["item", "cond"], how="inner").assign(sample="conf")
    return X.dropna(subset=["h"])


def harm_rows(X, family):
    """The four cells of REPORT section 4.8, then the B6 M3 contrast. Every p is
    two-sided: nothing here is an equivalence test."""
    changed = X.group.eq("answer changed")
    cells = [("estimand kept", X.group.eq("estimand kept")),
             ("estimand changed, answer kept", X.group.eq("estimand changed, answer kept")),
             ("answer changed, path cut", changed & X.route.eq("no path")),
             ("answer changed, path kept", changed & X.route.eq("backdoor"))]
    # A path-cutting graph always implies No; a path-keeping one implies either.
    # Split, so a harm here can be told apart from a lean towards No.
    cells += [(f"answer changed, path kept, implied {a}",
               changed & X.route.eq("backdoor") & X.implied_answer.eq(a)) for a in ("yes", "no")]
    rows = []
    for g, sel in cells:
        x = X[sel]
        rows.append(dict(part="3 where the harm lives", family=family, lexicon="both",
                         quantity=f"{g}: DR minus ORACLE",
                         n_items=x.groupby(["sample", "item"]).ngroups, n_draws=len(x))
                    | seeds_row([AC.boot_groups(x, "h", s) for s in SEEDS]))
    per = {s: run_tests(X, s, with_ni=False)["M3"] for s in SEEDS}
    x = X[changed & X.route.isin(["no path", "backdoor"])]
    rows.append(dict(part="3 where the harm lives", family=family, lexicon="both",
                     quantity="path cut minus path kept, answer changed (B6 M3 construction)",
                     n_items=x.groupby(["sample", "item"]).ngroups, n_draws=len(x))
                | seeds_row([(per[s]["estimate_pp"], per[s]["ci_lo"], per[s]["ci_hi"],
                              per[s]["p_boot"]) for s in SEEDS]))
    return rows


def main() -> int:
    data = {f: conf_data(FAMILIES[f][1], None) for f in FAMS}
    if any(S is None for S in data.values()):
        raise SystemExit("  B7 or B5 records missing (results/cladder/raw/pilot_raw_*conf*.csv)")
    for lex in data["gpt"]["ladder"]:
        a = data["gpt"]["ladder"][lex][["item", "id", "cond"]].drop_duplicates()
        b = data["luna"]["ladder"][lex][["item", "id", "cond"]].drop_duplicates()
        if not a.sort_values(list(a.columns)).reset_index(drop=True).equals(
                b.sort_values(list(b.columns)).reset_index(drop=True)):
            raise SystemExit(f"  {lex}: gpt and luna were not asked the same (item, condition) cells")

    # every family this file reports: the GPT-4.1 average, each of its models, luna
    only = lambda S, m: {k: {lex: d[d.model == m] for lex, d in v.items()} for k, v in S.items()}
    fams = {"gpt": (data["gpt"], TIER, "conf")}
    fams |= {m: (only(data["gpt"], m), [m], "conf") for m in TIER}
    fams["luna"] = (data["luna"], LUNA, "lunaconf")

    print("=" * 92)
    print("1. CHECK - the functions of this file on GPT-4.1 against the published rows")
    print("=" * 92)
    rows = {f: vs_raw(S, f) + ladder(S, f, models) for f, (S, models, _) in fams.items()}
    pub = pd.read_csv(RESULTS / "b5_vs_raw.csv")
    for r in pub.itertuples():
        got = [x for x in rows["gpt"] if x["lexicon"] == r.lexicon
               and x["quantity"] == f"{r.cond} minus {r.reference}"][0]
        stop_if_differs(f"b5_vs_raw {r.lexicon} {r.cond} - {r.reference}",
                        [got["estimate_pp"], got["ci_lo"], got["ci_hi"]], [r.delta_pp, r.ci_lo, r.ci_hi])
    conf = pd.read_csv(RESULTS / "confirmatory.csv")
    conf = conf[conf.analysis.eq("primary")]
    for f in FAMS:
        for t, q in (("H3a", "IRRELEVANT minus PERMUTE"), ("H3b", "KEEP minus SYMBOL")):
            got = [x for x in rows[f] if x["quantity"] == q][0]
            want = conf[conf.family.eq(f) & conf.test.eq(t)].iloc[0]
            stop_if_differs(f"confirmatory {f} {t}", [got["estimate_pp"], got["ci_lo"], got["ci_hi"]],
                            [want.estimate_pp, want.ci_lo, want.ci_hi])
    X = {f: draws(prefix, models) for f, (_, models, prefix) in fams.items()}
    D = pd.read_csv(RESULTS / "answer_change.csv")
    D = D[D["sample"].eq("conf")].groupby(["item", "cond"]).vs_ORACLE.mean()
    g = X["gpt"].set_index(["item", "cond"]).h
    stop_if_differs("answer_change.csv, conf draws, mean harm",
                    [100 * g.mean(), len(g)], [100 * D.reindex(g.index).mean(), D.notna().sum()])
    if not np.allclose(g.values, D.reindex(g.index).values):
        raise SystemExit("  per-draw harm differs from answer_change.csv; stopping.")
    print("\n  All reproduce.")

    out = []
    for f in fams:
        out += rows[f] + harm_rows(X[f], f)
    R = pd.DataFrame(out)
    R.to_csv(RESULTS / "b7_descriptive.csv", index=False)
    A = pd.DataFrame([r for f, (S, _, _) in fams.items() for r in accuracy(S, f)])
    A.to_csv(RESULTS / "b7_accuracy.csv", index=False)

    print("\n" + "=" * 92)
    print("2. gpt-5.6-luna beside GPT-4.1 on the 484 confirmatory causal items (descriptive)")
    print("=" * 92)
    print("\n  accuracy, %:")
    print(A.pivot_table(index=["lexicon", "cond"], columns="family", values="accuracy_pct",
                        sort=False).to_string())
    for part in R.part.unique():
        print(f"\n  {part}:")
        for r in R[R.part == part].itertuples():
            print(f"    {r.family:12s} {r.lexicon:8s} {r.quantity:60s} {r.estimate_pp:+7.2f} "
                  f"[{r.ci_lo:+6.2f} ; {r.ci_hi:+6.2f}]  p {r.p_min_seeds:.4f}-{r.p_max_seeds:.4f}"
                  f"  n={r.n_items}")
    print("\n  wrote results/cladder/b7_descriptive.csv, b7_accuracy.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
