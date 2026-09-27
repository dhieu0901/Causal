"""B8: deleted and spurious edges, and a graph offered as possibly wrong.

    python scripts/analyze_b8.py

Registered in prereg/B8.md (draft until committed). Four tests on fresh ate and
ett questions, three GPT-4.1 models, both lexicons, every graph without the
instruction line:

  T1  deleted edges (ED_k_NI minus ORACLE_NI) over draws that cut every X -> Y
      path and change the implied answer: negative
  T2  spurious edges (FE_k_NI minus ORACLE_NI) over all draws: equivalent to 0
      within 5 points
  T3  reversals under the doubt framing (DR_k_U minus ORACLE_U) over draws that
      cut the path: negative, the model still follows a graph it is told may be
      wrong
  T4  the same path-cutting draws, harm under the doubt framing minus harm
      without it (h_U minus h_NI): positive, the doubt reduces the harm

A draw is one (item, lexicon, corruption type, dose k): spurious edges differ
between the lexicons, so the two are not averaged into one draw, as in
analyze_edge_types.py. Its harm is the mean over the three models, parsed answers
only. The bootstrap resamples items with every draw of an item.

Before it reads a B8 record the script runs the T1 and T2 constructions on the
exploratory price400 draws and stops unless they return the published rows of
results/cladder/edge_types_harm.csv.

Writes results/cladder/b8.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

import analyze_edge_types as ET
from analyze_answer_change import Matcher, TYPES, draws_for
from analyze_b6 import boot_mean, holm, summary
from pilot import read_ids

RESULTS = ROOT / "results" / "cladder"
RAW = RESULTS / "raw"
SEEDS = [20260907, 1, 2, 3, 4]
ALPHA = 0.05
B8 = dict(n_items=340, sample_kmax=1, kmax=3, seed=20260927,
          exclude_ids=read_ids("prereg/excluded_ids_b8.txt"), query_types=["ate", "ett"],
          replay_condition="PERTURB_NI")
TESTS = [("T1", "deleted edge, path cut: ED_NI minus ORACLE_NI", "negative"),
         ("T2", "spurious edge, all draws: FE_NI minus ORACLE_NI, within +-5", "within +-5"),
         ("T3", "reversal under doubt, path cut: DR_U minus ORACLE_U", "negative"),
         ("T4", "path cut: harm under doubt minus harm without it", "positive")]


def cut(x):
    return x[(x.group == "answer changed") & (x.route == "no path")]


def run(X, seed):
    out = {}
    ed = cut(X[X.arm.eq("ED")]).dropna(subset=["h_ni"])
    e, d = boot_mean(ed, "h_ni", seed)
    out["T1"] = summary(e, d) | dict(n_draws=len(ed))
    fe = X[X.arm.eq("FE") & X.group.notna()].dropna(subset=["h_ni"])
    e, d = boot_mean(fe, "h_ni", seed)
    out["T2"] = summary(e, d, tost=True) | dict(n_draws=len(fe))
    dr = cut(X[X.arm.eq("DR")]).dropna(subset=["h_u"])
    e, d = boot_mean(dr, "h_u", seed)
    out["T3"] = summary(e, d) | dict(n_draws=len(dr))
    dd = dr.dropna(subset=["h_ni"]).assign(delta=lambda t: t.h_u - t.h_ni)
    e, d = boot_mean(dd, "delta", seed)
    out["T4"] = summary(e, d) | dict(n_draws=len(dd))
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


def b8_draws(unparsed_wrong=False, files=None, kw=None, tag="b8") -> pd.DataFrame | None:
    """`files` and `kw` default to B8's; a test passes other records through them."""
    f = files or {lex: RAW / f"pilot_raw_b8{lex}.csv" for lex in ("KEEP", "PSEUDO")}
    if not all(p.exists() for p in f.values()):
        return None
    G = draws_for(tag, Matcher(), kw=kw or B8, arms=("DR", "ED", "FE"))
    rows = []
    for lex, p in f.items():
        d = pd.read_csv(p)
        d = d[d.query_type.isin(TYPES)]
        if unparsed_wrong:
            d = d.assign(correct=d.correct.where(d.parsed == 1, 0), parsed=1)
        d = d[d.parsed == 1]
        by = {c: g.drop_duplicates(["model", "item"]).set_index(["model", "item"]).correct
              for c, g in d.groupby("cond")}
        for cond in sorted({c.rsplit("_", 1)[0] for c in by if c[:2] in ("DR", "ED", "FE")}):
            for col, suf, orc in (("h_ni", "_NI", "ORACLE_NI"), ("h_u", "_U", "ORACLE_U")):
                if cond + suf not in by:
                    continue
                a, b = by[cond + suf], by[orc]
                i = a.index.intersection(b.index)
                s = (a[i] - b[i]).groupby(level="item").mean()
                rows.append(pd.DataFrame({"item": s.index, "lexicon": lex, "cond": cond,
                                          "col": col, "v": s.values}))
    V = (pd.concat(rows).pivot_table(index=["item", "lexicon", "cond"], columns="col", values="v")
         .reset_index())
    X = V.merge(G[["item", "lexicon", "cond", "group", "route", "gold"]],
                on=["item", "lexicon", "cond"], how="inner")
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


def main() -> int:
    check()
    X = b8_draws()
    if X is None:
        print("  no B8 records yet (results/cladder/raw/pilot_raw_b8*.csv)")
        return 0
    rows = confirm(X, "primary")
    print("\n  sensitivity: unparsed answers scored as wrong")
    rows += confirm(b8_draws(unparsed_wrong=True), "unparsed scored as wrong")
    pd.DataFrame(rows).to_csv(RESULTS / "b8.csv", index=False)
    print("\n  wrote results/cladder/b8.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
