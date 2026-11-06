"""B8: a graph offered as possibly wrong, and deleted and spurious edges.

    python scripts/analyze_b8.py

Registered in prereg/B8.md. Fresh ate and ett questions, real names (KEEP),
every graph without the instruction line, three families with separate verdicts:
the GPT-4.1 pair (gpt-4.1-mini and gpt-4.1, their mean; gpt-4.1-nano is no longer
served), gpt-5.6-luna and Llama 3.3 70B (pinned to CoreWeave, fp16).
Five tests per family:

  T0  reversals (DR_k_NI minus ORACLE_NI) over draws that cut every X -> Y path
      and change the implied answer: negative (Study 3's R1 on fresh questions)
  T1  deleted edges (ED_k_NI minus ORACLE_NI) over the same group: negative
  T2  spurious edges (FE_k_NI minus ORACLE_NI) over all draws: equivalent to 0
      within 5 points
  T3  reversals under the doubt framing (DR_k_U minus ORACLE_U) over the T0
      draws: negative, the model still follows a graph it is told may be wrong
  T4  the T0 draws, harm under the doubt framing minus harm without it
      (h_U minus h_NI): positive, the doubt reduces the harm

The design is targeted: only the draws the tests use were sent, as listed in
prereg/b8_selection.csv by scripts/b8_select.py before the first call. A draw
is one (item, corruption type, dose k); its harm is the mean over the family's
models, parsed answers only. The bootstrap resamples items with every draw of
an item.

Before it reads a B8 record the script runs the T1 and T2 constructions on the
exploratory price400 draws and stops unless they return the published rows of
results/cladder/edge_types_harm.csv.

Writes results/cladder/b8.csv (the tests), b8_descriptive.csv (shares of
answers per condition), b8_per_model.csv (each test per GPT-4.1 model, no
verdict) and b8_study3_reference.csv (Study 3's R1 construction over the same
models and real names only, descriptive).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

import analyze_edge_types as ET
from analyze_answer_change import Matcher, TYPES, draws_for
from analyze_b6 import boot_mean, holm, summary
from pilot import read_ids

RESULTS = ROOT / "results" / "cladder"
RAW = RESULTS / "raw"
SELECTION = ROOT / "prereg" / "b8_selection.csv"
SEEDS = [20260907, 1, 2, 3, 4]
ALPHA = 0.05
LEXICON = "KEEP"
# replay_model: the replay check proves each listed draw by its key in the API
# cache, by default under gpt-4.1-nano; B8 never called it (retired before the
# run, prereg/B8.md), so the check uses gpt-4.1-mini, which answered every
# listed call. A deviation from the registered code, reported with the results.
B8 = dict(n_items=340, sample_kmax=1, kmax=3, seed=20260927,
          query_types=["ate", "ett"], replay_condition="PERTURB_NI",
          replay_model="gpt-4.1-mini")
FAMILIES = {
    "gpt": ("GPT-4.1 pair", ["gpt-4.1-mini", "gpt-4.1"], "pilot_raw_b8KEEP.csv"),
    "luna": ("gpt-5.6-luna", ["gpt-5.6-luna"], "pilot_raw_b8lunaKEEP.csv"),
    "llama": ("Llama 3.3 70B", ["meta-llama/llama-3.3-70b-instruct"],
              "pilot_raw_b8llamaKEEP.csv"),
}
TESTS = [("T0", "reversal, path cut: DR_NI minus ORACLE_NI", "negative"),
         ("T1", "deleted edge, path cut: ED_NI minus ORACLE_NI", "negative"),
         ("T2", "spurious edge, all draws: FE_NI minus ORACLE_NI, within +-5", "within +-5"),
         ("T3", "reversal under doubt, path cut: DR_U minus ORACLE_U", "negative"),
         ("T4", "path cut: harm under doubt minus harm without it", "positive")]


def cut(x):
    return x[(x.group == "answer changed") & (x.route == "no path")]


def run(X, seed):
    out = {}
    dr = cut(X[X.arm.eq("DR")])
    for t, x, col, tost in (
            ("T0", dr, "h_ni", False),
            ("T1", cut(X[X.arm.eq("ED")]), "h_ni", False),
            ("T2", X[X.arm.eq("FE") & X.group.notna()], "h_ni", True),
            ("T3", dr, "h_u", False),
            ("T4", dr.assign(delta=dr.h_u - dr.h_ni), "delta", False)):
        x = x.dropna(subset=[col])
        e, d = boot_mean(x, col, seed)
        out[t] = summary(e, d, tost=tost) | dict(n_draws=len(x))
    return out


def verdicts(res):
    adj = holm([res[t]["p_boot"] for t, *_ in TESTS])
    v = {}
    for (t, _, pred), pa in zip(TESTS, adj):
        if pred == "within +-5":
            v[t] = ("confirmed" if pa < ALPHA else "not confirmed", pa)
        elif pa >= ALPHA:
            v[t] = ("not confirmed", pa)
        else:
            ok = (res[t]["estimate_pp"] < 0) == (pred == "negative")
            v[t] = ("confirmed" if ok else "contradicted", pa)
    return v


def check() -> None:
    """T1 and T2 constructions on the exploratory price400 draws (instructed)."""
    D = draws_for("price400", Matcher(), arms=("DR", "ED", "FE"))
    H = pd.concat([ET.harm("KEEP"), ET.harm("PSEUDO")], ignore_index=True)
    D = D.merge(H, on=["item", "lexicon", "cond"], how="left").assign(
        arm=lambda t: t.cond.str[:2], sample="price400", h_ni=lambda t: t.h)
    pub = pd.read_csv(RESULTS / "edge_types_harm.csv").set_index(["arm", "k", "group"])
    for arm, x, row in (("ED", cut(D[D.arm.eq("ED")]), ("ED", "all", "answer changed, path cut")),
                        ("FE", D[D.arm.eq("FE") & D.group.notna()], ("FE", "all", "all draws"))):
        x = x.dropna(subset=["h_ni"])
        r = summary(*boot_mean(x, "h_ni", SEEDS[0]))
        got = [round(r["estimate_pp"], 2), round(r["ci_lo"], 2), round(r["ci_hi"], 2)]
        want = list(pub.loc[row, ["harm_pp", "ci_lo", "ci_hi"]])
        ok = got == [round(w, 2) for w in want]
        print(f"  check {arm}: {got}  published {want}  {'OK' if ok else 'DIFF'}")
        if not ok:
            raise SystemExit("  the B8 constructions do not reproduce edge_types_harm.csv; stopping")


def selection() -> pd.DataFrame:
    S = pd.read_csv(SELECTION)
    return S[S.lexicon == LEXICON].reset_index(drop=True)


def classified(S: pd.DataFrame) -> pd.DataFrame:
    """Every B8 draw with its group; the replay check covers exactly the draws sent."""
    sent = {(int(i), LEXICON, c.rsplit("_", 1)[0])
            for i, c in zip(S.item, S.cond) if c[:2] in ("DR", "ED", "FE")}
    kw = B8 | {"exclude_ids": read_ids("prereg/excluded_ids_b8.txt")}
    G = draws_for("b8", Matcher(), kw=kw, lexicons=(LEXICON,), arms=("DR", "ED", "FE"),
                  replay_sent=sent)
    return G[G.lexicon == LEXICON]


def missing_jobs(d: pd.DataFrame, models, S: pd.DataFrame) -> int:
    """Listed jobs with no row for some model: a run stopped before it sent them."""
    have = set(zip(d.model, d.item, d.cond))
    return sum((m, int(i), c) not in have for m in models for i, c in zip(S.item, S.cond))


def harms(d: pd.DataFrame, G: pd.DataFrame, unparsed_wrong=False) -> pd.DataFrame:
    """One row per sent draw: h_ni and h_u, each the mean over the models of
    (corrupted graph correct) minus (correct graph correct) under that framing."""
    d = d[d.query_type.isin(TYPES)]
    if unparsed_wrong:
        d = d.assign(correct=d.correct.where(d.parsed == 1, 0), parsed=1)
    d = d[d.parsed == 1]
    by = {c: g.drop_duplicates(["model", "item"]).set_index(["model", "item"]).correct
          for c, g in d.groupby("cond")}
    rows = []
    for cond in sorted({c.rsplit("_", 1)[0] for c in by if c[:2] in ("DR", "ED", "FE")}):
        for col, suf, orc in (("h_ni", "_NI", "ORACLE_NI"), ("h_u", "_U", "ORACLE_U")):
            if cond + suf not in by or orc not in by:
                continue
            a, b = by[cond + suf], by[orc]
            i = a.index.intersection(b.index)
            s = (a[i] - b[i]).groupby(level="item").mean()
            rows.append(pd.DataFrame({"item": s.index, "cond": cond, "col": col, "v": s.values}))
    V = (pd.concat(rows).pivot_table(index=["item", "cond"], columns="col", values="v")
         .reset_index())
    for c in ("h_ni", "h_u"):
        if c not in V:
            V[c] = np.nan
    X = V.merge(G[["item", "cond", "group", "route", "gold"]], on=["item", "cond"], how="inner")
    return X.assign(arm=X.cond.str[:2], sample="b8")


def confirm(X, label):
    per = {s: run(X, s) for s in SEEDS}
    vs = {s: verdicts(per[s]) for s in SEEDS}
    rows = []
    for t, what, pred in TESTS:
        v = {vs[s][t][0] for s in SEEDS}
        verdict = v.pop() if len(v) == 1 else "borderline (changes with the bootstrap seed)"
        r = per[SEEDS[0]][t]
        ps = [per[s][t]["p_boot"] for s in SEEDS]
        rows.append(dict(analysis=label, test=t, quantity=what, predicted=pred,
                         estimate_pp=round(r["estimate_pp"], 2), ci_lo=round(r["ci_lo"], 2),
                         ci_hi=round(r["ci_hi"], 2),
                         ci90_lo=round(r["ci90_lo"], 2) if "ci90_lo" in r else "",
                         ci90_hi=round(r["ci90_hi"], 2) if "ci90_hi" in r else "",
                         p_boot=round(r["p_boot"], 4), p_holm=round(vs[SEEDS[0]][t][1], 4),
                         p_min_seeds=round(min(ps), 4), p_max_seeds=round(max(ps), 4),
                         n_draws=r["n_draws"], verdict=verdict))
        print(f"  {t}  {r['estimate_pp']:+7.2f} [{r['ci_lo']:+6.2f} ; {r['ci_hi']:+6.2f}]"
              f"  p {min(ps):.4f}-{max(ps):.4f}  draws {r['n_draws']:4d}  {verdict}")
    return rows


def describe(d: pd.DataFrame, G: pd.DataFrame, fam: str, models) -> list[dict]:
    """Descriptive, no verdict: on the items and draws of T0 and T3, how each
    framing is answered, and how the answers compare with no graph at all."""
    d = d[d.query_type.isin(TYPES) & (d.parsed == 1)]
    tg = cut(G[G.cond.str.startswith("DR")])
    items = set(tg.item)
    out = []
    for m in list(models) + (["all"] if len(models) > 1 else []):
        x = d if m == "all" else d[d.model == m]
        raw = x[x.cond == "RAW"].set_index(["model", "item"]).pred

        def on_items(cond):
            return x[(x.cond == cond) & x.item.isin(items)]

        def on_draws(suf):
            y = x.merge(tg[["item", "cond"]].assign(cond=lambda t: t.cond + suf),
                        on=["item", "cond"], how="inner")
            same = [p == raw.get((mm, i)) for mm, i, p in zip(y.model, y.item, y.pred)]
            return y, np.mean(same) if len(y) else np.nan

        ni, same_ni = on_draws("_NI")
        u, same_u = on_draws("_U")
        r = on_items("RAW")
        o_ni, o_u = on_items("ORACLE_NI"), on_items("ORACLE_U")
        out.append(dict(family=fam, model=m, items=len(items), draws=len(tg),
                        acc_raw=100 * r.correct.mean(), says_yes_raw=100 * (r.pred == "yes").mean(),
                        acc_oracle_ni=100 * o_ni.correct.mean(),
                        acc_oracle_u=100 * o_u.correct.mean(),
                        says_no_oracle_ni=100 * (o_ni.pred == "no").mean(),
                        says_no_oracle_u=100 * (o_u.pred == "no").mean(),
                        follow_ni=100 * (ni.pred == "no").mean(),
                        follow_u=100 * (u.pred == "no").mean(),
                        acc_dr_ni=100 * ni.correct.mean(), acc_dr_u=100 * u.correct.mean(),
                        same_as_raw_ni=100 * same_ni, same_as_raw_u=100 * same_u))
    return [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()} for r in out]


def study3_reference() -> list[dict]:
    """Descriptive (prereg/B8.md, section 3): Study 3's R1, the construction of
    T0 (path-cutting reversals without the instruction line), over the models
    B8 runs and real names only, to set beside T0. Study 3's registered R1
    averages both lexicons and, for the GPT-4.1 family, three models; these
    rows do not. Llama's Study 3 answers came from another provider (Novita)."""
    import analyze_b6 as B6
    rows = []
    for fam, prefix, models in (("gpt", "b6", ["gpt-4.1-mini", "gpt-4.1"]),
                                ("llama", "b6llama", None)):
        X = B6.b6_draws(prefix=prefix, models=models, lexicons=(LEXICON,))
        if X is None:
            continue
        c = B6.cut(X).dropna(subset=["h_ni"])
        r = summary(*boot_mean(c, "h_ni", SEEDS[0]))
        rows.append(dict(family=fam, models=", ".join(models) if models else "Llama 3.3 70B (Novita)",
                         lexicon=LEXICON,
                         quantity="Study 3, R1 construction: DR_NI minus ORACLE_NI, path cut",
                         estimate_pp=round(r["estimate_pp"], 2), ci_lo=round(r["ci_lo"], 2),
                         ci_hi=round(r["ci_hi"], 2), n_draws=len(c)))
    return rows


def per_model(d: pd.DataFrame, G: pd.DataFrame, fam: str, models) -> list[dict]:
    rows = []
    for m in models:
        res = run(harms(d[d.model == m], G), SEEDS[0])
        for t, what, _ in TESTS:
            r = res[t]
            rows.append(dict(family=fam, model=m, test=t, quantity=what,
                             estimate_pp=round(r["estimate_pp"], 2), ci_lo=round(r["ci_lo"], 2),
                             ci_hi=round(r["ci_hi"], 2), n_draws=r["n_draws"]))
    return rows


def main() -> int:
    check()
    present = {f: v for f, v in FAMILIES.items() if (RAW / v[2]).exists()}
    if not present:
        print("  no B8 records yet (results/cladder/raw/pilot_raw_b8*.csv)")
        return 0
    S = selection()
    G = classified(S)
    rows, desc, pm = [], [], []
    for fam, (label, models, fname) in present.items():
        d = pd.read_csv(RAW / fname)
        miss = missing_jobs(d, models, S)
        print(f"\n=== {label}: {len(d)} rows; {miss} listed jobs without a row")
        if miss:
            # prereg/B8.md: a test is computed only if all its data exist.
            rows.append(dict(family=fam, analysis="incomplete run, no verdict",
                             quantity=f"{miss} listed jobs without a row",
                             verdict="incomplete run, no verdict"))
            continue
        rows += [r | {"family": fam} for r in confirm(harms(d, G), "primary")]
        print("  sensitivity: unparsed answers scored as wrong")
        rows += [r | {"family": fam}
                 for r in confirm(harms(d, G, unparsed_wrong=True), "unparsed scored as wrong")]
        desc += describe(d, G, fam, models)
        if len(models) > 1:
            pm += per_model(d, G, fam, models)
    out = pd.DataFrame(rows)
    out[["family"] + [c for c in out.columns if c != "family"]].to_csv(RESULTS / "b8.csv",
                                                                      index=False)
    pd.DataFrame(desc).to_csv(RESULTS / "b8_descriptive.csv", index=False)
    if pm:
        pd.DataFrame(pm).to_csv(RESULTS / "b8_per_model.csv", index=False)
    ref = pd.DataFrame(study3_reference())
    ref.to_csv(RESULTS / "b8_study3_reference.csv", index=False)
    print("\n  Study 3 over the same models, real names only (descriptive):")
    print(ref.to_string(index=False))
    print("\n  wrote results/cladder/b8.csv, b8_descriptive.csv, b8_study3_reference.csv"
          + (", b8_per_model.csv" if pm else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
