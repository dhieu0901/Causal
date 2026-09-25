"""Reversed edges in three groups: estimand kept, estimand changed but the
answer kept, answer changed.

    python scripts/analyze_answer_change.py

Why. classify_perturbations.py splits the reversals by whether they change the
ESTIMAND, and the harm sits in the ones that do. But a changed estimand need
not change the ANSWER: the formula the wrong graph implies can still land on
the same side of zero. If the model computes with the graph it is handed, a
reversal should hurt only when it changes the answer. If it hurts about as much
when only the estimand moves, the graph is doing something other than
feeding a computation.

What "the answer the wrong graph implies" means here. The value a reasoner
would get if it believed the shown graph and could read the TRUE observational
distribution of the item's SCM, turned into yes/no with the item's own question
direction. That distribution is exact: every CLadder item is matched to its SCM
in data/cladder/cladder-meta.json through the probabilities its `reasoning`
field states, and the SCM is enumerated (verify_groundtruth.SCM). In order:

  1. no directed X -> Y path in the shown graph: the effect is 0, answer No
  2. otherwise a backdoor set of OBSERVED variables (the latent variable of
     IV, frontdoor and arrowhead is declared unobserved in every prompt):
     the adjusted contrast; if two valid sets disagree in sign, "ambiguous"
  3. otherwise a single front-door mediator, then a single instrument (Wald)
  4. otherwise the shown graph leaves the effect unidentified

The prompt only states the numbers the TRUE estimand needs, so a model may not
be able to compute the wrong graph's formula from what it is shown. The groups
say what a faithful reasoner with the wrong graph would conclude, not what the
model could compute.

Scope. `ate` and `ett` only: the backdoor machinery is exact for them. The
natural effects and det-counterfactual depend on mediator roles and structural
equations, and are left out. Every DR draw on those items in the three samples
that sent them: price400, n600 (DR_k2 and DR_k3 from its arms files) and the
confirmatory sample. Harm is DR minus ORACLE (and DR minus RAW) per item,
averaged over models, parsed answers only; the bootstrap resamples items, all
draws of one item together.

Checks that stop the script: every draw's prompt is a key in the API cache
(classify_perturbations.check_replay); every item matches exactly one SCM, or
several that agree; the machinery applied to the TRUE graph gives the true
value's sign on every item. Exploratory: none of this was pre-registered.

Writes: results/cladder/answer_change.csv (one row per draw)
        results/cladder/answer_change_shares.csv
        results/cladder/answer_change_harm.csv
        results/cladder/answer_change_slope.csv
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from classify_perturbations import (backdoor_sets, check_replay, classify, d_separated,
                                    descendants)
from pilot import make_items, read_ids
from stats import boot_items, boot_p, cluster_boot

RESULTS = ROOT / "results" / "cladder"
RAWDIR = RESULTS / "raw"
SEED = 20260907
NBOOT = 4000
SEEDS = [20260907, 1, 2, 3, 4]
TYPES = ("ate", "ett")
LATENT = {"IV": {"V1"}, "frontdoor": {"V1"}, "arrowhead": {"V2"}}
TOL = 0.0051                       # the reasoning field states two decimals

# tag -> (classify / make_items arguments, record files)
SAMPLES = {
    "price400": (dict(n_items=399, sample_kmax=3, kmax=3, seed=SEED, exclude_ids=None),
                 ["pilot_raw_price400{}.csv"]),
    "n600": (dict(n_items=600, sample_kmax=1, kmax=3, seed=SEED, exclude_ids=None),
             ["pilot_raw_n600{}.csv", "pilot_raw_n600arms{}.csv"]),
    "conf": (dict(n_items=1000, sample_kmax=1, kmax=3, seed=20260925,
                  exclude_ids=read_ids("prereg/excluded_ids.txt")),
             ["pilot_raw_conf{}.csv"]),
}
GROUPS = ["estimand kept", "estimand changed, answer kept", "answer changed",
          "unidentified under the shown graph", "ambiguous"]


def _vg():
    spec = importlib.util.spec_from_file_location(
        "vg", str(ROOT / "scripts" / "verify_groundtruth.py"))
    vg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vg)
    return vg


VG = _vg()
QTY = re.compile(r"^P\((\w+)=(\d)(?:\s*\|\s*([^)]*))?\)\s*=\s*(-?[0-9.]+)$")


# ------------------------------------------------------------ item -> SCM
class Matcher:
    def __init__(self):
        meta = json.loads((ROOT / "data" / "cladder" / "cladder-meta.json").read_text(
            encoding="utf-8"))
        self.by = {}
        for m in meta:
            self.by.setdefault((m["story_id"], m["graph_id"]), []).append(m)
        self.scm = {}

    def get(self, m):
        if m["model_id"] not in self.scm:
            self.scm[m["model_id"]] = VG.SCM(m["params"])
        return self.scm[m["model_id"]]

    @staticmethod
    def stated(m, s, t, c):
        """The value CLadder printed for P(t | c): its published figure where it
        keeps one (in `arrowhead` those come from the faulty formula of
        REPORT section 1.1 and do not equal the SCM's), else the SCM's."""
        (var, val), = t.items()
        key = f"P({var}=1" + (" | " + ", ".join(f"{k}={v}" for k, v in c.items()) if c else "") + ")"
        if key in m["groundtruth"]:
            p = m["groundtruth"][key]
            return p if val == 1 else 1 - p
        return s.prob(t, given=c)

    def match(self, r):
        """Every SCM of the item's story and family that reproduces each stated quantity."""
        qs = []
        for line in str(r.reasoning).splitlines():
            m = QTY.match(line.strip())
            if m:
                cond = {}
                if m.group(3):
                    for part in m.group(3).split(","):
                        k, v = part.split("=")
                        cond[k.strip()] = int(v)
                qs.append(({m.group(1): int(m.group(2))}, cond, float(m.group(4))))
        if not qs:
            return []
        err = []
        for m in self.by.get((r.story_id, r.graph_id), []):
            s = self.get(m)
            err.append((max(abs(self.stated(m, s, t, c) - v) for t, c, v in qs), m))
        if not err:
            return []
        # Every SCM within rounding; failing that, the nearest one if it is
        # within one more unit (a few stated values are off by 0.0002 past
        # rounding, e.g. 0.615 printed as 0.61; the next SCM is then 0.05 away).
        ok = [m for e, m in err if e <= TOL]
        best = min(e for e, _ in err)
        return ok or ([m for e, m in err if e == best] if best <= 2 * TOL else [])


# ------------------------------------------------ what a graph implies
def adjusted(scm, Z, qt):
    Z = sorted(Z)
    tot = 0.0
    for vals in itertools.product((0, 1), repeat=len(Z)):
        z = dict(zip(Z, vals))
        w = (scm.prob(z, given={"X": 1}) if qt == "ett" else scm.prob(z)) if z else 1.0
        if w == 0.0:
            continue
        tot += w * (scm.prob({"Y": 1}, given={"X": 1, **z})
                    - scm.prob({"Y": 1}, given={"X": 0, **z}))
    return tot


def frontdoor_ok(G, nodes, m):
    rest = [(a, b) for a, b in G if m not in (a, b)]
    if "Y" in descendants(rest, "X"):
        return False                                    # m misses a directed path
    no_x_out = [(a, b) for a, b in G if a != "X"]
    no_m_out = [(a, b) for a, b in G if a != m]
    return (d_separated(no_x_out, nodes, "X", m, set())
            and d_separated(no_m_out, nodes, m, "Y", {"X"}))


def frontdoor(scm, m, qt):
    pm = {x: scm.prob({m: 1}, given={"X": x}) for x in (0, 1)}
    px1 = scm.prob({"X": 1})
    tot = 0.0
    for v in (0, 1):
        dm = (pm[1] - pm[0]) if v else (pm[0] - pm[1])
        if qt == "ett":
            inner = scm.prob({"Y": 1}, given={"X": 1, m: v})
        else:
            inner = sum(scm.prob({"Y": 1}, given={"X": x, m: v}) * (px1 if x else 1 - px1)
                        for x in (0, 1))
        tot += dm * inner
    return tot


def instrument_ok(G, nodes, z):
    if (z, "X") not in G or z in descendants(G, "X"):
        return False
    no_x_out = [(a, b) for a, b in G if a != "X"]
    return d_separated(no_x_out, nodes, z, "Y", set())


def wald(scm, z):
    dy = scm.prob({"Y": 1}, given={z: 1}) - scm.prob({"Y": 1}, given={z: 0})
    dx = scm.prob({"X": 1}, given={z: 1}) - scm.prob({"X": 1}, given={z: 0})
    return dy / dx if dx else float("nan")


def implied(scm, G, latent, qt):
    """(values, route): the effect the graph G implies on the SCM's true joint."""
    G = [tuple(e) for e in G]
    nodes = sorted({n for e in G for n in e})
    if "Y" not in descendants(G, "X"):
        return [0.0], "no path"
    sets = [Z for Z in backdoor_sets(G, nodes, "X", "Y") if not (Z & latent)]
    if sets:
        return [adjusted(scm, Z, qt) for Z in sorted(sets, key=sorted)], "backdoor"
    obs = [n for n in nodes if n not in ("X", "Y") and n not in latent]
    fd = [m for m in obs if frontdoor_ok(G, nodes, m)]
    if fd:
        return [frontdoor(scm, m, qt) for m in fd], "frontdoor"
    if qt == "ate":
        iv = [z for z in obs if instrument_ok(G, nodes, z)]
        if iv:
            return [wald(scm, z) for z in iv], "instrument"
    return [], "unidentified"


def answer(value, polarity):
    """yes/no for a signed effect; a zero effect answers No either way."""
    if value == 0.0:
        return "no"
    return "yes" if (value > 0) == polarity else "no"


# ------------------------------------------------------------ per sample
def draws_for(tag, M, kw=None, lexicons=("KEEP", "PSEUDO")):
    """One row per DR draw on an ate/ett item. `kw` defaults to the sample's
    entry in SAMPLES; analyze_r1_chains.py passes the lex sample's."""
    kw = kw or SAMPLES[tag][0]
    items = make_items(kw["n_items"], kw["seed"], kw["sample_kmax"], "full_v1.5_default.csv",
                       None, True, kw["exclude_ids"], kw.get("query_types"))
    C = classify(kw["n_items"], kw["sample_kmax"], kw["kmax"], arms=("DR",), seed=kw["seed"],
                 exclude_ids=kw["exclude_ids"], lexicons=lexicons, keep_shown=True,
                 query_types=kw.get("query_types"))
    C = C[C.query_type.isin(TYPES)].reset_index(drop=True)
    check_replay(C)
    canon = __import__("classify_perturbations").canonical_structures()
    rows, info = [], {}
    for i in sorted(C.item.unique()):
        r = items.loc[i]
        cands = M.match(r)
        if not cands:
            raise SystemExit(f"{tag} item {i} (id {r.id}): no SCM reproduces its quantities")
        latent = LATENT.get(r.graph_id, set())
        for v in latent:                              # the prompt must say so
            nm = cands[0]["variable_mapping"][f"{v}name"]
            if f"{nm.lower()} is unobserved" not in r.prompt.lower():
                raise SystemExit(f"{tag} item {i}: '{nm}' is not declared unobserved")
        key = "ATE(Y | X)" if r.query_type == "ate" else "ETT(Y | X)"
        per = []
        for m in cands:
            s = M.get(m)
            true = s.ate() if r.query_type == "ate" else s.ett()
            pol = (str(r.label).strip().lower() == "yes") == (m["groundtruth"][key] > 0)
            tv, route = implied(s, canon[r.graph_id], latent, r.query_type)
            if not tv or any(np.sign(v) != np.sign(true) for v in tv):
                raise SystemExit(f"{tag} item {i}: the machinery on the TRUE graph gives "
                                 f"{tv} ({route}), the SCM {true:+.4f}")
            per.append((s, true, pol))
        info[i] = (r, per)
    for x in C.itertuples():
        r, per = info[x.item]
        latent = LATENT.get(r.graph_id, set())
        outs = set()
        for s, true, pol in per:
            for G in x.shown_sym:
                vals, route = implied(s, G, latent, r.query_type)
                if not vals:
                    outs.add(("unidentified", route))
                    continue
                ans = {answer(v, pol) for v in vals}
                outs.add((ans.pop() if len(ans) == 1 else "ambiguous", route))
        gold = str(r.label).strip().lower()
        true_ans = {answer(true, pol) for _, true, pol in per}
        if len(outs) != 1:
            group, route, ans = "ambiguous", "several", ""
        else:
            (ans, route), = outs
            if x.estimand_unchanged:
                group = "estimand kept"
            elif ans == "unidentified":
                group = "unidentified under the shown graph"
            elif ans == "ambiguous":
                group = "ambiguous"
            else:
                group = "estimand changed, answer kept" if ans == gold else "answer changed"
        rows.append(dict(sample=tag, item=x.item, id=int(r.id), lexicon=x.lexicon,
                         family=r.graph_id, query_type=r.query_type, cond=x.cond, k=x.k,
                         estimand_unchanged=x.estimand_unchanged, route=route,
                         implied_answer=ans, gold=gold,
                         gold_is_true_answer=true_ans == {gold}, n_scm=len(per),
                         group=group))
    return pd.DataFrame(rows)


def harm(tag, lex, ref):
    """Per (item, cond): cond minus ref, averaged over models, parsed only."""
    _, files = SAMPLES[tag]
    d = pd.concat([pd.read_csv(RAWDIR / f.format(lex)) for f in files], ignore_index=True)
    d = d[(d.parsed == 1) & d.query_type.isin(TYPES)]
    out = []
    for m in sorted(d.model.unique()):
        dm = d[d.model == m]
        r = dm[dm.cond == ref].drop_duplicates("item").set_index("item").correct
        for k in (1, 2, 3):
            c = dm[dm.cond == f"DR_k{k}"].drop_duplicates("item").set_index("item").correct
            i = r.index.intersection(c.index)
            out.append(pd.DataFrame({"item": i, "cond": f"DR_k{k}", "model": m,
                                     "diff": (c[i] - r[i]).values}))
    h = pd.concat(out).groupby(["item", "cond"])["diff"].mean()
    return h.rename(f"vs_{ref}")


def boot_groups(x: pd.DataFrame, col, seed):
    """Mean over draws; resample (sample, item) clusters, all their draws together."""
    keys = list(x.groupby(["sample", "item"]).indices.values())
    vals = x[col].values
    draws = cluster_boot(len(keys), lambda i: 100 * vals[np.concatenate([keys[j] for j in i])].mean(),
                         seed, NBOOT)
    return 100 * vals.mean(), np.percentile(draws, 2.5), np.percentile(draws, 97.5), \
        boot_p(draws, NBOOT)


def main() -> int:
    M = Matcher()
    D = []
    for tag in SAMPLES:
        print(f"\n  {tag}:")
        D.append(draws_for(tag, M))
    D = pd.concat(D, ignore_index=True)

    H = []
    for tag in SAMPLES:
        for lex in ("KEEP", "PSEUDO"):
            h = pd.concat([harm(tag, lex, "ORACLE"), harm(tag, lex, "RAW")], axis=1).reset_index()
            H.append(h.assign(sample=tag, lexicon=lex))
    H = pd.concat(H, ignore_index=True)
    D = D.merge(H, on=["sample", "lexicon", "item", "cond"], how="left")
    D.to_csv(RESULTS / "answer_change.csv", index=False)

    print("\n" + "=" * 92)
    print("REVERSED EDGES IN THREE GROUPS (ate and ett; exploratory)")
    print("=" * 92)
    K = D[D.lexicon == "KEEP"]
    print(f"  {len(K)} draws on {K.groupby(['sample', 'item']).ngroups} items; "
          f"gold equals the SCM's answer on {int(K.drop_duplicates(['sample', 'item']).gold_is_true_answer.sum())}"
          f" of them")
    print("\n  draws per group and dose (KEEP):")
    print(pd.crosstab(K.group, K.cond, margins=True).reindex(GROUPS + ["All"]).fillna(0)
          .astype(int).to_string())
    print("\n  route of the implied answer, estimand-changed draws (KEEP):")
    print(pd.crosstab(K[~K.estimand_unchanged].route, K[~K.estimand_unchanged].group).to_string())

    # The composition of each dose. A shown graph that cuts every X -> Y path
    # on a Yes item is where the harm lives (below), and its share grows with
    # k - which is a composition channel for the dose slope of section 1.2 and
    # of H4, of the same kind that classify_perturbations.py documents.
    S = K.assign(cuts_path=K.route.eq("no path"),
                 cuts_path_on_yes=K.route.eq("no path") & K.gold.eq("yes"))
    shares = pd.concat([
        pd.crosstab(S.cond, S.group, normalize="index").mul(100),
        S.groupby("cond")[["cuts_path", "cuts_path_on_yes"]].mean().mul(100),
        S.groupby("cond").size().rename("n_draws")], axis=1).round(1).reset_index()
    shares.to_csv(RESULTS / "answer_change_shares.csv", index=False)
    print("\n  share of each dose's draws, % (KEEP):")
    print(shares.to_string(index=False))

    # Two splits beyond the group. `route`: most answer changes come from a
    # shown graph with no X -> Y path at all (implied answer No on a Yes item),
    # the rest from a changed adjustment set; the prompt's numbers fit only the
    # TRUE formula, so a model could follow the first but hardly compute the
    # second. `gold`: a shown graph that cuts the path always implies No, so a
    # model that merely says No more often under any odd graph would look the
    # same; the Yes items whose estimand is kept are the check on that.
    # "both": one value per draw, the mean of its KEEP and PSEUDO harms. A
    # draw's group is the same under both lexicons (checked here), so pooling
    # them doubles the answers behind each draw without mixing groups. This is
    # the unit of the pre-registered B6 tests (prereg/B6.md).
    g2 = D.pivot_table(index=["sample", "item", "cond"], columns="lexicon",
                       values="group", aggfunc="first")
    if (g2["KEEP"] != g2["PSEUDO"]).any():
        raise SystemExit("a draw falls in different groups under KEEP and PSEUDO")
    keys = ["sample", "item", "cond"]
    both = (D.groupby(keys)[["vs_ORACLE", "vs_RAW"]].mean()
            .join(D[D.lexicon == "KEEP"].set_index(keys)[["group", "route", "gold"]])
            .reset_index().assign(lexicon="both"))
    D = pd.concat([D, both], ignore_index=True)

    rows = []
    splits = [(c, "all", "all") for c in ("all", "DR_k1", "DR_k2", "DR_k3")]
    splits += [("all", rt, gd) for rt in ("no path", "backdoor") for gd in ("all", "yes", "no")]
    splits += [("all", "all", gd) for gd in ("yes", "no")]
    for lex in ("KEEP", "PSEUDO", "both"):
        for g in GROUPS[:3]:
            for cond, route, gold in splits:
                x = D[(D.lexicon == lex) & (D.group == g) & D.vs_ORACLE.notna()]
                if cond != "all":
                    x = x[x.cond == cond]
                if route != "all":
                    x = x[x.route == route]
                if gold != "all":
                    x = x[x.gold == gold]
                if x.groupby(["sample", "item"]).ngroups < 10:
                    continue
                for ref in ("ORACLE", "RAW"):
                    xx = x[x[f"vs_{ref}"].notna()]
                    per = [boot_groups(xx, f"vs_{ref}", s) for s in SEEDS]
                    e, lo, hi, p = per[0]
                    ps = [q[3] for q in per]
                    rows.append(dict(lexicon=lex, group=g, cond=cond, route=route, gold=gold,
                                     reference=ref, harm_pp=round(e, 2), ci_lo=round(lo, 2),
                                     ci_hi=round(hi, 2), p_boot=round(p, 4),
                                     p_min_seeds=round(min(ps), 4),
                                     p_max_seeds=round(max(ps), 4), n_draws=len(xx),
                                     n_items=xx.groupby(["sample", "item"]).ngroups))
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "answer_change_harm.csv", index=False)
    print("\n  harm, DR minus ORACLE and DR minus RAW, pp (items resampled):")
    show = R[(R.cond == "all") & (R.gold == "all") | (R.route == "all") & (R.gold != "all")]
    for r in show.itertuples():
        print(f"    {r.lexicon:6s} {r.group:30s} {r.route:8s} gold {r.gold:3s} vs {r.reference:6s}"
              f" {r.harm_pp:+7.2f} [{r.ci_lo:+6.2f} ; {r.ci_hi:+6.2f}]  p {r.p_min_seeds:.4f}-"
              f"{r.p_max_seeds:.4f}  draws {r.n_draws:4d}  items {r.n_items}")
    slope(D)
    print("\n  wrote results/cladder/answer_change.csv, answer_change_harm.csv, "
          "answer_change_slope.csv")
    return 0


def slope(D: pd.DataFrame) -> None:
    """Does the dose slope survive the change in composition across doses?

    H4's construction on these draws: items whose k=1 reversal already changes
    the estimand and that have all three doses, one slope (h3 - h1) / 2 per
    item, harm against RAW. The composition prediction replaces each draw's
    harm by the mean harm of its cell - answer kept; answer changed with the
    path cut; answer changed with the path kept - pooled over the doses, so it
    moves only because the mix of cells moves. The residual is what is left of
    the slope once the mix is accounted for.
    """
    rows = []
    for lex in ("KEEP", "PSEUDO"):
        x = D[(D.lexicon == lex) & D.group.isin(GROUPS[1:3]) & D.vs_RAW.notna()].copy()
        x["cell"] = np.where(x.group.eq("answer changed"), "changed, " + x.route, "kept")
        x["pred"] = x.cell.map(x.groupby("cell").vs_RAW.mean())
        W = x.pivot_table(index=["sample", "item"], columns="cond",
                          values=["vs_RAW", "pred"]).dropna()
        obs = (W[("vs_RAW", "DR_k3")] - W[("vs_RAW", "DR_k1")]) / 2
        prd = (W[("pred", "DR_k3")] - W[("pred", "DR_k1")]) / 2
        for name, v in (("observed", obs), ("composition only", prd),
                        ("residual", obs - prd)):
            per = [boot_items(v.values, sd, NBOOT) for sd in SEEDS]
            e, lo, hi, p = per[0]
            ps = [q[3] for q in per]
            rows.append(dict(lexicon=lex, slope=name, pp_per_edge=round(e, 2),
                             ci_lo=round(lo, 2), ci_hi=round(hi, 2), p_boot=round(p, 4),
                             p_min_seeds=round(min(ps), 4), p_max_seeds=round(max(ps), 4),
                             n_items=len(v)))
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "answer_change_slope.csv", index=False)
    print("\n  slope per reversed edge vs RAW, items whose k=1 draw changes the estimand:")
    for r in R.itertuples():
        print(f"    {r.lexicon:6s} {r.slope:16s} {r.pp_per_edge:+6.2f} [{r.ci_lo:+6.2f} ; "
              f"{r.ci_hi:+6.2f}]  p {r.p_min_seeds:.4f}-{r.p_max_seeds:.4f}  n={r.n_items}")


if __name__ == "__main__":
    raise SystemExit(main())
