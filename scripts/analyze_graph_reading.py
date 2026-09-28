"""How the models use a supplied graph: do they follow it, could they compute
with it, and do the question's own numbers contradict it?

    python scripts/analyze_graph_reading.py

Exploratory, added after B6 and the path probe; nothing here was registered.
Three questions a reader of M1 to M3 asks:

  1. FOLLOW. How often does the answer under a reversed graph equal the answer
     that graph implies (analyze_answer_change.implied)? Under ORACLE the
     implied answer is the gold label, so accuracy under ORACLE and "follows
     the shown graph" under DR are one measurement on two graphs.
  2. COMPUTE. CLadder states only the probabilities the TRUE estimand needs.
     Can the formula of the SHOWN graph be computed from them? A term is
     available if the prompt states it for every value of its conditioning
     set, or if it follows from available terms by the law of total
     probability, by Bayes' rule, or by an independence the shown graph
     implies (a reasoner who believes the graph may use those). Every valid
     backdoor set, front-door mediator and instrument is tried. A graph with
     no X -> Y path needs no number. The TRUE graph must be computable on
     every item, or the script stops.
  3. CONTRADICT. Do the stated numbers contradict the shown graph? They do
     when the graph implies T independent of W given the rest of a stated
     conditioning set and the stated P(T | W, rest) differ between the two
     values of W. The TRUE graph must never be contradicted, or the script
     stops.

Also: the share of items whose answer does not depend on the structure (the
sign of P(Y | X=1) - P(Y | X=0) gives the true answer), and M3 with the
comparison made within graph family and query type.

Samples: B6 (GPT-4.1 family, both lexicons) and the ate/ett questions of B5
(every family). Harm is DR minus ORACLE as in analyze_b6.py; the bootstrap
resamples items with all their draws.

Writes results/cladder/graph_reading.csv and graph_reading_models.csv.
"""
from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from analyze_answer_change import SAMPLES, Matcher, answer, draws_for
from analyze_b6 import B6, b6_draws, boot_mean, summary
from classify_perturbations import backdoor_sets, d_separated, descendants
from stats import boot_p, cluster_boot

RESULTS = ROOT / "results" / "cladder"
RAW = RESULTS / "raw"
NBOOT = 4000
SEED = 20260907
QTY = re.compile(r"^P\((\w+)=(\d)(?:\s*\|\s*([^)]*))?\)\s*=\s*(-?[0-9.]+)$")
GPT = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]
LLAMA = "meta-llama/llama-3.3-70b-instruct"
B5_FILES = {"gpt": "pilot_raw_conf{}.csv", "luna": "pilot_raw_lunaconf{}.csv",
            "llama": "pilot_raw_llamaconf{}.csv"}
KEEP_ANSWER = "estimand changed, answer kept"


# ---------------------------------------------------------------- the prompt's numbers
def stated_terms(reasoning) -> dict:
    """(target, conditioning set) -> {assignment: P(target=1 | assignment)}, for
    every term the prompt states under all values of its conditioning set."""
    fam: dict = {}
    for line in str(reasoning).splitlines():
        m = QTY.match(line.strip())
        if not m:
            continue
        cond = {}
        if m.group(3):
            for part in m.group(3).split(","):
                k, v = part.split("=")
                cond[k.strip()] = int(v)
        p = float(m.group(4))
        p = p if int(m.group(2)) == 1 else 1 - p
        fam.setdefault((m.group(1), frozenset(cond)), {})[tuple(sorted(cond.items()))] = p
    return {k: v for k, v in fam.items() if len(v) == 2 ** len(k[1])}


class Available:
    """Which terms P(t | C) a reasoner holding graph G can obtain from the stated ones."""

    def __init__(self, terms, G, obs):
        self.terms, self.G, self.obs = set(terms), G, obs
        self.nodes = sorted({n for e in G for n in e})
        self.memo: dict = {}

    def indep(self, t, w, C):
        return d_separated(self.G, self.nodes, t, w, set(C))

    def term(self, t, C, depth=4) -> bool:
        C = frozenset(C)
        if (t, C) in self.terms:
            return True
        if depth == 0:
            return False
        key = (t, C, depth)
        if key in self.memo:
            return self.memo[key]
        self.memo[key] = False                      # no cycles through itself
        others = [w for w in self.obs if w != t and w not in C]
        ok = (any(self.indep(t, w, C - {w}) and self.term(t, C - {w}, depth - 1) for w in C)
              or any(self.indep(t, w, C) and self.term(t, C | {w}, depth - 1) for w in others)
              or any(self.term(t, C | {w}, depth - 1) and self.term(w, C, depth - 1)
                     for w in others)
              or any(self.term(w, (C - {w}) | {t}, depth - 1) and self.term(t, C - {w}, depth - 1)
                     for w in C))
        self.memo[key] = ok
        return ok

    def joint(self, Z, C) -> bool:
        Z = sorted(Z)
        if not Z:
            return True
        return any(all(self.term(z, frozenset(C) | frozenset(o[:i])) for i, z in enumerate(o))
                   for o in itertools.permutations(Z))


def computable(G, latent, qt, terms) -> bool:
    """Can some formula valid under G be computed from the stated terms?"""
    G = [tuple(e) for e in G]
    nodes = sorted({n for e in G for n in e})
    if "Y" not in descendants(G, "X"):
        return True                                 # the effect is 0, no number needed
    obs = [n for n in nodes if n not in latent]
    A = Available(terms, G, obs)
    # ett weights the adjustment by P(Z | X=1). In the `confounding` family
    # CLadder states P(Z) only, and its own worked solutions put P(Z) in its
    # place; the same substitution is accepted here, or the TRUE graph would be
    # uncomputable on every such item.
    bases = [{"X"}, set()] if qt == "ett" else [set()]
    for Z in backdoor_sets(G, nodes, "X", "Y"):
        if not (Z & latent) and A.term("Y", {"X"} | Z) and any(A.joint(Z, b) for b in bases):
            return True
    from analyze_answer_change import frontdoor_ok, instrument_ok
    for m in obs:
        if m in ("X", "Y"):
            continue
        if frontdoor_ok(G, nodes, m) and A.term(m, {"X"}) and A.term("Y", {"X", m}) \
                and (qt == "ett" or A.term("X", set())):
            return True
        if qt == "ate" and instrument_ok(G, nodes, m) and A.term("Y", {m}) and A.term("X", {m}):
            return True
    return False


def contradicted(G, terms) -> bool:
    """Do the stated numbers break an independence that G implies?"""
    G = [tuple(e) for e in G]
    nodes = sorted({n for e in G for n in e})
    for (t, C), vals in terms.items():
        if t not in nodes:
            continue
        for w in C:
            R = C - {w}
            if w not in nodes or not d_separated(G, nodes, t, w, set(R)):
                continue
            for rest in itertools.product((0, 1), repeat=len(R)):
                a = dict(zip(sorted(R), rest))
                p0 = vals[tuple(sorted({**a, w: 0}.items()))]
                p1 = vals[tuple(sorted({**a, w: 1}.items()))]
                if abs(p1 - p0) > 1e-9:
                    return True
    return False


def hook(r, per, shown, latent, true_graph):
    """Per draw: computability and contradiction under the shown graph(s), the
    same checks under the true graph, and whether the structure-free contrast
    already gives the true answer."""
    terms = stated_terms(r.reasoning)
    qt = r.query_type
    naive_ok = []
    for s, true, pol in per:
        naive = s.prob({"Y": 1}, given={"X": 1}) - s.prob({"Y": 1}, given={"X": 0})
        naive_ok.append(answer(naive, pol) == answer(true, pol))
    return dict(computable=all(computable(G, latent, qt, terms) for G in shown),
                contradicted=any(contradicted(G, terms) for G in shown),
                true_computable=computable(true_graph, latent, qt, terms),
                true_contradicted=contradicted(true_graph, terms),
                naive_gives_true=all(naive_ok))


def draws(tag, kw=None) -> pd.DataFrame:
    D = draws_for(tag, Matcher(), kw=kw, per_draw=hook)
    bad = D[~D.true_computable | D.true_contradicted].drop_duplicates("item")
    if len(bad):
        raise SystemExit(f"{tag}: the TRUE graph fails the checks on {len(bad)} items, "
                         f"e.g. id {bad.id.iloc[0]}; the machinery is wrong")
    return D[D.lexicon == "KEEP"].drop(columns="lexicon")


# ---------------------------------------------------------------- answers
def answers(files, models) -> pd.DataFrame:
    """Per (lexicon, model, item, cond): parsed answer and correctness."""
    out = []
    for lex in ("KEEP", "PSEUDO"):
        d = pd.read_csv(RAW / files.format(lex))
        d = d[d.model.isin(models) & (d.parsed == 1) & d.query_type.isin(["ate", "ett"])]
        out.append(d.drop_duplicates(["model", "item", "cond"])
                   .assign(lexicon=lex)[["lexicon", "model", "item", "cond", "pred", "correct"]])
    return pd.concat(out, ignore_index=True)


def per_draw_rates(A: pd.DataFrame, D: pd.DataFrame, by_model=False) -> pd.DataFrame:
    """Accuracy under ORACLE and DR and the share of DR answers equal to the
    implied answer, per draw: mean over models within a lexicon, then over
    lexicons (the averaging of analyze_b6.b6_draws)."""
    o = A[A.cond == "ORACLE"][["lexicon", "model", "item", "correct"]].rename(
        columns={"correct": "acc_oracle"})
    x = A[A.cond.str.fullmatch(r"DR_k\d")].merge(o, on=["lexicon", "model", "item"])
    x = x.merge(D[["item", "cond", "implied_answer"]], on=["item", "cond"])
    x["follows"] = (x.pred.astype(str).str.lower() == x.implied_answer).astype(float)
    keys = ["item", "cond"] + (["model"] if by_model else [])
    lex = x.groupby(keys + ["lexicon"])[["correct", "acc_oracle", "follows"]].mean()
    return lex.groupby(level=keys).mean().rename(columns={"correct": "acc_dr"}).reset_index()


# ---------------------------------------------------------------- summaries
def groups(X: pd.DataFrame) -> dict:
    cut = (X.group == "answer changed") & (X.route == "no path")
    kept = (X.group == "answer changed") & (X.route == "backdoor")
    return {"estimand kept": X[X.group == "estimand kept"],
            KEEP_ANSWER: X[X.group == KEEP_ANSWER],
            "answer changed, path cut": X[cut], "answer changed, path kept": X[kept]}


def row(sample, what, x, harm=True):
    r = dict(sample=sample, quantity=what, n_draws=len(x), n_items=x.item.nunique(),
             computable_pct=round(100 * x.computable.mean(), 2),
             contradicted_pct=round(100 * x.contradicted.mean(), 2),
             acc_oracle_pct=round(100 * x.acc_oracle.mean(), 2),
             acc_dr_pct=round(100 * x.acc_dr.mean(), 2),
             follows_graph_pct=round(100 * x.follows.mean(), 2))
    if harm and x.h.notna().sum() >= 10:
        e, d = boot_mean(x.dropna(subset=["h"]).assign(sample=sample), "h", SEED)
        s = summary(e, d)
        r |= dict(harm_pp=round(s["estimate_pp"], 2), ci_lo=round(s["ci_lo"], 2),
                  ci_hi=round(s["ci_hi"], 2), p_boot=round(s["p_boot"], 4))
    return r


def stratified_m3(X: pd.DataFrame):
    """Path cut minus path kept, compared within (graph family, query type) and
    pooled with weights n_cut * n_kept / (n_cut + n_kept); items resampled."""
    g = groups(X)
    Z = pd.concat([g["answer changed, path cut"].assign(_c=1),
                   g["answer changed, path kept"].assign(_c=0)], ignore_index=True)
    Z = Z.dropna(subset=["h"])
    Z["stratum"] = Z.family + "/" + Z.query_type
    keys = list(Z.groupby("item").indices.values())

    def stat(idx):
        z = Z.iloc[idx]
        num = den = 0.0
        for _, s in z.groupby("stratum"):
            a, b = s[s._c == 1].h, s[s._c == 0].h
            if len(a) and len(b):
                w = len(a) * len(b) / (len(a) + len(b))
                num += w * (a.mean() - b.mean())
                den += w
        return 100 * num / den if den else np.nan

    est = stat(np.arange(len(Z)))
    d = cluster_boot(len(keys), lambda i: stat(np.concatenate([keys[j] for j in i])), SEED, NBOOT)
    d = d[np.isfinite(d)]
    both = [s for s, t in Z.groupby("stratum") if t._c.nunique() == 2]
    return dict(sample="b6", quantity="M3 within graph family and query type",
                n_draws=int(Z.stratum.isin(both).sum()),
                n_items=Z[Z.stratum.isin(both)].item.nunique(),
                harm_pp=round(est, 2), ci_lo=round(np.percentile(d, 2.5), 2),
                ci_hi=round(np.percentile(d, 97.5), 2), p_boot=round(boot_p(d, NBOOT), 4),
                n_strata=len(both))


def main() -> int:
    rows = []

    # ---- B6: GPT-4.1 family
    print("=" * 78)
    print("B6 - follow, compute, contradict (GPT-4.1 family, both lexicons)")
    print("=" * 78)
    D = draws("b6", B6)
    H = b6_draws()[["item", "cond", "h"]]
    R = per_draw_rates(answers("pilot_raw_b6{}.csv", GPT), D)
    X = D.merge(R, on=["item", "cond"]).merge(H, on=["item", "cond"], how="left")
    items = X.drop_duplicates("item")
    rows.append(dict(sample="b6", quantity="items whose structure-free contrast gives the true answer",
                     n_items=len(items), share_pct=round(100 * items.naive_gives_true.mean(), 2)))
    rows.append(row("b6", "all reversals", X))
    for g, x in groups(X).items():
        rows.append(row("b6", g, x))
    g = groups(X)
    for flag, name in ((True, "contradicted by the stated numbers"),
                       (False, "consistent with the stated numbers")):
        x = g["answer changed, path cut"]
        rows.append(row("b6", f"answer changed, path cut, {name}", x[x.contradicted == flag]))
    for flag, name in ((True, "computable"), (False, "not computable")):
        x = g["answer changed, path kept"]
        rows.append(row("b6", f"answer changed, path kept, {name}", x[x.computable == flag]))
    rows.append(stratified_m3(X))

    # ---- B5 ate/ett: every family, per model
    print("\n" + "=" * 78)
    print("B5 - the same, per model")
    print("=" * 78)
    D5 = draws("conf", SAMPLES["conf"][0])
    it5 = D5.drop_duplicates("item")
    rows.append(dict(sample="conf", quantity="items whose structure-free contrast gives the true answer",
                     n_items=len(it5), share_pct=round(100 * it5.naive_gives_true.mean(), 2)))
    X5 = D5.assign(h=np.nan)
    for g, x in groups(X5).items():
        rows.append(dict(sample="conf", quantity=g, n_draws=len(x), n_items=x.item.nunique(),
                         computable_pct=round(100 * x.computable.mean(), 2),
                         contradicted_pct=round(100 * x.contradicted.mean(), 2)))
    mrows = []
    for fam, f in B5_FILES.items():
        models = {"gpt": GPT, "luna": ["gpt-5.6-luna"], "llama": [LLAMA]}[fam]
        Rm = per_draw_rates(answers(f, models), D5, by_model=True)
        Xm = D5.merge(Rm, on=["item", "cond"])
        for m, xm in Xm.groupby("model"):
            for g, x in groups(xm).items():
                mrows.append(dict(model=m, group=g, n_draws=len(x), n_items=x.item.nunique(),
                                  acc_oracle_pct=round(100 * x.acc_oracle.mean(), 2),
                                  acc_dr_pct=round(100 * x.acc_dr.mean(), 2),
                                  follows_graph_pct=round(100 * x.follows.mean(), 2)))
    Rows = pd.DataFrame(rows)
    Rows.to_csv(RESULTS / "graph_reading.csv", index=False)
    M = pd.DataFrame(mrows)
    M.to_csv(RESULTS / "graph_reading_models.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(Rows.to_string(index=False))
        print()
        print(M.to_string(index=False))
    print("\n  wrote results/cladder/graph_reading.csv, graph_reading_models.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
