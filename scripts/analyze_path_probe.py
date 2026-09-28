"""Does the model read a path-cutting graph correctly, and does its reading
predict its answer? prereg/PATH_PROBE.md.

    python scripts/analyze_path_probe.py

Joins the probe (scripts/probe_path.py: "is there a directed path from X to Y
in this graph?", asked of every graph B6 showed) to B6's own answers.

A cell is one (model, item, lexicon, reversal dose k): the model's answer to
the B6 question under DR_kk and under ORACLE (instructed, parsed only), its
harm h = DR_kk - ORACLE, and the model's own reading of the two graphs. The
draw's group (path cut or not) is analyze_answer_change.py's, exactly as B6
used it. The bootstrap resamples items, every cell of an item together
(analyze_b6.boot_mean, boot_diff).

Tests (prereg/PATH_PROBE.md section 4):
  PP2  mean h over path-cut cells the model read as "no path" while reading
       the correct graph as "path": negative
  PP3  mean h over path-cut cells read as "path" (misread) minus mean h over
       those read as "no path": positive; run only with at least 20 misread
       cells on at least 10 items, else reported without a verdict
Holm across the tests run, seeds 20260907, 1, 2, 3, 4.

Check that stops the script: the cells, averaged over models and lexicons per
draw, must give analyze_b6.b6_draws()'s harm on every draw.

Writes: results/cladder/path_probe.csv, results/cladder/path_probe_reading.csv

Replication on Llama 3.3 70B (prereg/B6_LLAMA.md), once its B6 and probe
records exist: the same cells, check and tests on pilot_raw_b6llama*.csv and
probe_path_raw_llama.csv, written to path_probe_llama.csv and
path_probe_reading_llama.csv.

The probe asked inside the B6 question (prereg/PATH_PROBE_CONTEXT.md), once
probe_path_context_raw.csv exists: the same cells and tests, written to
path_probe_context.csv and path_probe_reading_context.csv.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from analyze_answer_change import Matcher, draws_for
from analyze_b6 import B6, b6_draws, boot_diff, boot_mean, holm, summary

RESULTS = ROOT / "results" / "cladder"
RAW = RESULTS / "raw"
SEEDS = [20260907, 1, 2, 3, 4]
ALPHA = 0.05
MIN_MISREAD, MIN_ITEMS = 20, 10
OPENING = re.compile(r"^\W*(yes|no)\b", re.IGNORECASE)


def cells(prefix="b6") -> pd.DataFrame:
    """One row per (model, item, lexicon, k): h and h_ni from B6's records."""
    G = draws_for("b6", Matcher(), kw=B6)
    G = G[G.lexicon == "KEEP"].set_index(["item", "cond"])[["group", "route", "gold"]]
    rows = []
    for lex in ("KEEP", "PSEUDO"):
        d = pd.read_csv(RAW / f"pilot_raw_{prefix}{lex}.csv")
        d = d[d.parsed == 1]
        by = {c: g.drop_duplicates(["model", "item"]).set_index(["model", "item"]).correct
              for c, g in d.groupby("cond")}
        for k in (1, 2, 3):
            for col, dr, orc in (("h", f"DR_k{k}", "ORACLE"), ("h_ni", f"DR_k{k}_NI", "ORACLE_NI")):
                if dr not in by:
                    continue
                a, b = by[dr], by[orc]
                i = a.index.intersection(b.index)
                rows.append(pd.DataFrame({"model": i.get_level_values(0),
                                          "item": i.get_level_values(1), "lexicon": lex,
                                          "cond": f"DR_k{k}", "col": col,
                                          "v": (a[i] - b[i]).values}))
    C = (pd.concat(rows).pivot_table(index=["model", "item", "lexicon", "cond"], columns="col",
                                     values="v").reset_index())
    return C.join(G, on=["item", "cond"], how="inner").assign(sample="b6")


def check(C: pd.DataFrame, prefix="b6") -> None:
    X = b6_draws(prefix=prefix).set_index(["item", "cond"])
    # b6_draws averages over models within a lexicon, then over the two lexicons
    mine = (C.groupby(["item", "cond", "lexicon"]).h.mean()
            .groupby(level=["item", "cond"]).mean())
    both = X.h.reindex(mine.index)
    if not np.allclose(mine.values, both.values, equal_nan=True):
        raise SystemExit("  the cells do not reproduce analyze_b6.b6_draws(); stopping")
    print(f"  check: {len(mine)} B6 draws reproduced from {len(C)} cells")


def reading(C: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    P = P[P.parsed == 1]
    shown = P[P.cond != "ORACLE"].set_index(["model", "item", "lexicon", "cond"]).pred
    orc = P[P.cond == "ORACLE"].set_index(["model", "item", "lexicon"]).pred
    C = C.join(shown.rename("read_shown"), on=["model", "item", "lexicon", "cond"])
    return C.join(orc.rename("read_oracle"), on=["model", "item", "lexicon"])


def lenient(P: pd.DataFrame, context=False) -> pd.DataFrame:
    """P with every unparsed answer read by its opening yes/no (text from the cache)."""
    from probe_path import MAX_TOKENS, context_jobs, jobs
    from runner import read_cached
    J = pd.DataFrame(context_jobs() if context else jobs())[["item", "lexicon", "cond", "prompt"]]
    U = P[P.parsed == 0].merge(J, on=["item", "lexicon", "cond"])
    fix = {}
    for r in U.itertuples():
        rec = read_cached(r.model, 0.0, r.prompt, max_tokens=MAX_TOKENS) or {}
        m = OPENING.search(rec.get("text", "") or "")
        if m:
            fix[(r.model, r.item, r.lexicon, r.cond)] = m.group(1).lower()
    P = P.copy()
    key = list(zip(P.model, P.item, P.lexicon, P.cond))
    new = [fix.get(k) for k in key]
    hit = pd.Series([v is not None for v in new], index=P.index)
    P.loc[hit, "pred"] = [v for v in new if v is not None]
    P.loc[hit, "parsed"] = 1
    P.loc[hit, "correct"] = ((P.loc[hit, "pred"] == "yes") == P.loc[hit, "path_true"].astype(bool)).astype(int)
    print(f"  lenient reading: {int(hit.sum())} of {len(U)} unparsed answers read by their opening word")
    return P


def tests(C, seed):
    cut = C[(C.group == "answer changed") & (C.route == "no path")]
    ok = cut[(cut.read_oracle == "yes")]
    right, wrong = ok[ok.read_shown == "no"].dropna(subset=["h"]), ok[ok.read_shown == "yes"].dropna(subset=["h"])
    out = {}
    e, d = boot_mean(right, "h", seed)
    out["PP2"] = summary(e, d) | dict(n_cells=len(right), n_items=right.item.nunique(),
                                      predicted="negative")
    # PP3 is a test only with enough misread cells; below that, the registration
    # asks for the same number as a description with no verdict
    e, d = boot_diff(wrong, right, "h", seed)
    enough = len(wrong) >= MIN_MISREAD and wrong.item.nunique() >= MIN_ITEMS
    out["PP3" if enough else "PP3 (too few misread cells, descriptive)"] = summary(e, d) | dict(
        n_cells=len(wrong) + len(right), n_items=pd.concat([wrong, right]).item.nunique(),
        predicted="positive" if enough else "descriptive")
    # sensitivity, no verdict: PP2 without the instruction line
    r_ni = ok[ok.read_shown == "no"].dropna(subset=["h_ni"])
    e, d = boot_mean(r_ni, "h_ni", seed)
    out["PP2 without instruction"] = summary(e, d) | dict(n_cells=len(r_ni),
                                                         n_items=r_ni.item.nunique(),
                                                         predicted="descriptive")
    return out, len(wrong), wrong.item.nunique()


def main() -> int:
    run("b6", "probe_path_raw.csv", "")
    if (RAW / "pilot_raw_b6llamaKEEP.csv").exists():
        print("\n" + "=" * 78)
        print("REPLICATION on Llama 3.3 70B (prereg/B6_LLAMA.md)")
        print("=" * 78)
        run("b6llama", "probe_path_raw_llama.csv", "_llama")
    if (RAW / "probe_path_context_raw.csv").exists():
        print("\n" + "=" * 78)
        print("THE PROBE INSIDE THE QUESTION (prereg/PATH_PROBE_CONTEXT.md)")
        print("=" * 78)
        run("b6", "probe_path_context_raw.csv", "_context", context=True)
    return 0


def run(prefix, probe_file, suffix, context=False) -> None:
    C0 = cells(prefix)
    check(C0, prefix)
    f = RAW / probe_file
    if not f.exists():
        print(f"  no probe records yet ({probe_file}; scripts/probe_path.py)")
        return
    P = pd.read_csv(f)

    # descriptive: how well each model reads path existence
    rd = []
    for m, g in P.groupby("model"):
        for name, sel in (("correct graph", g.cond == "ORACLE"),
                          ("reversal, path present", (g.cond != "ORACLE") & (g.path_true == 1)),
                          ("reversal, no path", (g.cond != "ORACLE") & (g.path_true == 0))):
            x = g[sel]
            rd.append(dict(model=m, graphs=name, n=len(x),
                           parsed_pct=round(100 * x.parsed.mean(), 2),
                           read_correctly_pct=round(100 * x[x.parsed == 1].correct.mean(), 2)))
    RD = pd.DataFrame(rd)

    C = reading(C0, P)
    cut = C[(C.group == "answer changed") & (C.route == "no path")]
    for m, g in cut.groupby("model"):
        g = g.dropna(subset=["read_shown"])
        rd.append(dict(model=m, graphs="path-cut draws of B6", n=len(g), parsed_pct=None,
                       read_correctly_pct=round(100 * (g.read_shown == "no").mean(), 2)))
    RD = pd.DataFrame(rd)
    RD.to_csv(RESULTS / f"path_probe_reading{suffix}.csv", index=False)
    print("\n  reading of path existence, % read correctly:")
    print(RD.to_string(index=False))

    per = {s: tests(C, s) for s in SEEDS}
    # Not registered: gpt-4.1-nano often skips the ANSWER line when the graph has
    # a direct X -> Y edge and opens with "Yes, there is a directed path ...".
    # Read every unparsed answer by its opening yes/no instead, and redo PP2.
    PL = lenient(P, context)
    L = reading(C0, PL)
    lp = tests(L, SEEDS[0])[0]["PP2"]
    per_l = [tests(L, s)[0]["PP2"]["p_boot"] for s in SEEDS]
    names = [t for t in per[SEEDS[0]][0] if t in ("PP2", "PP3")]
    rows = []
    for t in per[SEEDS[0]][0]:
        r = per[SEEDS[0]][0][t]
        ps = [per[s][0][t]["p_boot"] for s in SEEDS]
        if t in names:
            vs = set()
            for s in SEEDS:
                adj = dict(zip(names, holm([per[s][0][n]["p_boot"] for n in names])))
                sign_ok = (per[s][0][t]["estimate_pp"] < 0) == (r["predicted"] == "negative")
                vs.add("not confirmed" if adj[t] >= ALPHA else
                       ("confirmed" if sign_ok else "contradicted"))
            verdict = vs.pop() if len(vs) == 1 else "borderline (changes with the bootstrap seed)"
            p_holm = dict(zip(names, holm([per[SEEDS[0]][0][n]["p_boot"] for n in names])))[t]
        else:
            verdict, p_holm = "no verdict", None
        rows.append(dict(test=t, predicted=r["predicted"], estimate_pp=round(r["estimate_pp"], 2),
                         ci_lo=round(r["ci_lo"], 2), ci_hi=round(r["ci_hi"], 2),
                         p_boot=round(r["p_boot"], 4),
                         p_holm=None if p_holm is None else round(p_holm, 4),
                         p_min_seeds=round(min(ps), 4), p_max_seeds=round(max(ps), 4),
                         n_cells=r["n_cells"], n_items=r["n_items"], verdict=verdict))
    _, n_wrong, i_wrong = per[SEEDS[0]]
    if "PP3" not in names:
        print(f"\n  PP3 not run: {n_wrong} misread cells on {i_wrong} items "
              f"(needs {MIN_MISREAD} on {MIN_ITEMS})")
    ps = per_l
    rows.append(dict(test="PP2, unparsed probe answers read by their opening yes/no (not registered)",
                     predicted="descriptive", estimate_pp=round(lp["estimate_pp"], 2),
                     ci_lo=round(lp["ci_lo"], 2), ci_hi=round(lp["ci_hi"], 2),
                     p_boot=round(lp["p_boot"], 4), p_holm=None,
                     p_min_seeds=round(min(ps), 4), p_max_seeds=round(max(ps), 4),
                     n_cells=lp["n_cells"], n_items=lp["n_items"], verdict="no verdict"))
    for m, g in PL[PL.model == "gpt-4.1-nano"].groupby(PL.cond.eq("ORACLE")):
        x = g[g.parsed == 1]
        rd.append(dict(model="gpt-4.1-nano", graphs=("correct graph" if m else "reversals")
                       + ", unparsed read by opening word (not registered)", n=len(g),
                       parsed_pct=round(100 * g.parsed.mean(), 2),
                       read_correctly_pct=round(100 * x.correct.mean(), 2)))
    pd.DataFrame(rd).to_csv(RESULTS / f"path_probe_reading{suffix}.csv", index=False)
    R = pd.DataFrame(rows)
    if context:
        # prereg/PATH_PROBE_CONTEXT.md names the same tests PC2 and PC3
        R["test"] = R.test.str.replace(r"^PP(\d)", r"PC\1", regex=True)
    R.to_csv(RESULTS / f"path_probe{suffix}.csv", index=False)
    print("\n  tests:")
    for r in R.itertuples():
        print(f"    {r.test:26s} {r.estimate_pp:+7.2f} [{r.ci_lo:+6.2f} ; {r.ci_hi:+6.2f}]  "
              f"p {r.p_min_seeds:.4f}-{r.p_max_seeds:.4f}  cells {r.n_cells}  items {r.n_items}"
              f"  {r.verdict}")
    print(f"\n  wrote results/cladder/path_probe{suffix}.csv, path_probe_reading{suffix}.csv")


if __name__ == "__main__":
    raise SystemExit(main())
