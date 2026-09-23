"""Is a reversed edge actually a WRONG graph? For a quarter of them, no.

Why this file exists. Every dose claim in this project assumes that k reversed
edges means "k units of wrongness". It does not. Reversing an edge that no
adjustment set depends on leaves the correct ATE estimand untouched, so the
model is handed a graph that is different but not misleading. Nothing in the
repository ever checked this - src/noise.py line 121 defers the question to a
`scripts/check_identifiability.py` that was never written.

The answer matters because the share of harmless draws is not constant across k:

    reversals that leave the estimand unchanged:  20.6% at k=1, 0.0% at k=2, 0.0% at k=3

So the "dose-response curve" in results/vs_raw_trend.csv is confounded with
sample composition. Going from k=1 to k=2 does not only add a reversal, it also
removes every harmless draw from the cell. Conditioning on the perturbation
actually changing the answer, more reversed edges is NOT more harmful.

How the drawn perturbation is recovered. scripts/pilot.py line 168 picks it with
`random.Random(f"{seed}:{i}:{t}:{k}").choice(opts)`, where `i` is the FRAME INDEX
from `items.iterrows()`. Re-running make_items with the same arguments and
replaying that generator reproduces the exact perturbation each item was shown.

Where the graph comes from. Line 2 of CLadder's `reasoning` field carries the
structure, but 67 of the 399 price400 items have no reasoning text. The
per-family structure from data/cladder-meta.json covers all of them, and this
script ASSERTS that the two agree wherever both exist (332 of 332, 0 mismatches)
before relying on it.

Limit of the criterion, stated plainly. The backdoor test is exact for `ate`. For
`ett`, `nde`, `nie` and `det-counterfactual` it is an approximation: those
estimands can depend on parts of the graph that the backdoor sets for (X, Y) do
not capture, so a perturbation called harmless here may still change such a
label. The split is therefore reported on the causal group as a whole and read
as a lower bound on how many draws are harmless, not as an exact partition.

Run:  python scripts/classify_perturbations.py
Writes: results/perturbation_classes.csv
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from stats import boot_interval, boot_p, cluster_boot

from perturb import to_edges                      # noqa: E402
from pilot import make_items, ENUMERATORS         # noqa: E402
from analyze_querygroup import ARITH, IDENT       # noqa: E402

SEED = 20260907
NBOOT = 4000
N_ITEMS, KMAX = 399, 3


# ---------------------------------------------------------------- graph tools
def parents(edges):
    p = {}
    for a, b in edges:
        p.setdefault(b, set()).add(a)
    return p


def _reach(adj, x):
    seen, stack = set(), [x]
    while stack:
        n = stack.pop()
        for q in adj.get(n, ()):
            if q not in seen:
                seen.add(q)
                stack.append(q)
    return seen


def ancestors(edges, x):
    return _reach(parents(edges), x)


def descendants(edges, x):
    ch = {}
    for a, b in edges:
        ch.setdefault(a, set()).add(b)
    return _reach(ch, x)


def d_separated(edges, nodes, x, y, Z):
    """Is x d-separated from y given Z? Brute force over simple paths.

    The graphs here have 3 to 5 nodes, so enumerating every simple path is both
    fast and much easier to check by eye than a Bayes-ball implementation.
    """
    anc_Z = set(Z)
    for z in Z:
        anc_Z |= ancestors(edges, z)
    adj = {n: set() for n in nodes}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    def walk(u, seen):
        if u == y:
            yield [u]
            return
        for w in adj[u]:
            if w in seen:
                continue
            for rest in walk(w, seen | {w}):
                yield [u] + rest

    for path in walk(x, {x}):
        for i in range(1, len(path) - 1):
            a, b, c = path[i - 1], path[i], path[i + 1]
            collider = (a, b) in edges and (c, b) in edges
            if (collider and b not in anc_Z) or (not collider and b in Z):
                break
        else:
            return False        # an open path survives, so not d-separated
    return True


def backdoor_sets(edges, nodes, x, y):
    """Every Z satisfying the backdoor criterion for (x, y)."""
    blocked = descendants(edges, x) | {x}
    gx = [(a, b) for a, b in edges if a != x]     # cut the arrows out of x
    others = [n for n in nodes if n not in (x, y)]
    ok = set()
    for r in range(len(others) + 1):
        for Z in itertools.combinations(others, r):
            if any(z in blocked for z in Z):
                continue
            if d_separated(gx, nodes, x, y, set(Z)):
                ok.add(frozenset(Z))
    return frozenset(ok)


def same_estimand(true_edges, bad_edges, nodes, x="X", y="Y"):
    """Does the corrupted graph still imply the same ATE estimand for (x, y)?"""
    if (y in descendants(true_edges, x)) != (y in descendants(bad_edges, x)):
        return False
    return (backdoor_sets(true_edges, nodes, x, y)
            == backdoor_sets(bad_edges, nodes, x, y))


# ------------------------------------------------------- canonical structures
def canonical_structures() -> dict[str, list[tuple[str, str]]]:
    spec = importlib.util.spec_from_file_location(
        "vg", str(ROOT / "scripts" / "verify_groundtruth.py"))
    vg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vg)
    meta = json.loads((ROOT / "data" / "cladder-meta.json").read_text(encoding="utf-8"))
    out = {}
    for m in meta:
        if m["graph_id"] in out:
            continue
        s = vg.SCM(m["params"])
        out[m["graph_id"]] = sorted((p, n) for n in s.nodes for p in s.parents[n])
    return out


def check_canonical(items, canon) -> tuple[int, int]:
    """Assert the per-family structure agrees with every reasoning line present."""
    agree = missing = 0
    for _, r in items.iterrows():
        txt = r["reasoning"]
        line = txt.splitlines()[1] if isinstance(txt, str) and len(txt.splitlines()) > 1 else None
        if not line or line == "nan":
            missing += 1
            continue
        if sorted(to_edges(line)) != canon[r["graph_id"]]:
            raise SystemExit(
                f"structure mismatch for {r['graph_id']}: reasoning says "
                f"{sorted(to_edges(line))}, cladder-meta says {canon[r['graph_id']]}. "
                "The per-family shortcut is not safe; classify from reasoning only.")
        agree += 1
    return agree, missing


# ------------------------------------------------------------------- analysis
def classify() -> pd.DataFrame:
    items = make_items(N_ITEMS, SEED, KMAX, "full_v1.5_default.csv", None, True)
    canon = canonical_structures()
    agree, missing = check_canonical(items, canon)
    print(f"  structure source: {agree} items verified against their reasoning line, "
          f"0 mismatches;\n  {missing} items have no reasoning line and use the "
          f"per-family structure.\n")

    rows = []
    for i, r in items.iterrows():
        edges = canon[r["graph_id"]]
        nodes = sorted({n for e in edges for n in e})
        if "X" not in nodes or "Y" not in nodes:
            continue
        for t in ("DR", "ED", "FE"):
            for k in range(1, KMAX + 1):
                opts = ENUMERATORS[t](edges, nodes, k)
                if not opts:
                    continue
                bad = random.Random(f"{SEED}:{i}:{t}:{k}").choice(opts)
                rows.append(dict(item=i, family=r["graph_id"], query_type=r["query_type"],
                                 cond=f"{t}_k{k}", arm=t, k=k,
                                 estimand_unchanged=same_estimand(edges, bad, nodes)))
    return pd.DataFrame(rows)


def benjamini_hochberg(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Step-up procedure. Returns a boolean mask of the rejected hypotheses."""
    m = len(p)
    order = np.argsort(p)
    below = p[order] <= (np.arange(1, m + 1) / m) * alpha
    keep = np.zeros(m, dtype=bool)
    if below.any():
        keep[order[: np.where(below)[0].max() + 1]] = True
    return keep


def boot(x: np.ndarray):
    if len(x) < 5:
        return (np.nan,) * 4
    b = cluster_boot(len(x), lambda i: x[i].mean(), SEED, NBOOT)
    est, lo, hi, p = boot_interval(x.mean(), b, NBOOT)
    return 100 * est, 100 * lo, 100 * hi, p


def paired_causal(d: pd.DataFrame, cond: str) -> pd.Series:
    d = d[~d.query_type.isin(ARITH | IDENT)]
    cols = []
    for m in sorted(d.model.unique()):
        r = d[(d.model == m) & (d.cond == "RAW")].set_index("item").correct
        c = d[(d.model == m) & (d.cond == cond)].set_index("item").correct
        i = r.index.intersection(c.index)
        if len(i) >= 5:
            cols.append(pd.Series((c[i] - r[i]).values, index=i, name=m))
    return pd.concat(cols, axis=1).mean(axis=1).dropna() if cols else pd.Series(dtype=float)


def main() -> int:
    C = classify()

    print("=" * 78)
    print("1. SHARE OF THE DRAWN PERTURBATIONS THAT LEAVE THE ATE ESTIMAND UNCHANGED")
    print("=" * 78 + "\n")
    share = C.pivot_table(index="arm", columns="k", values="estimand_unchanged",
                          aggfunc=lambda s: round(100 * float(np.mean(s)), 1))
    n = C.pivot_table(index="arm", columns="k", values="estimand_unchanged", aggfunc="size")
    print("  percent harmless:"); print(share.to_string())
    print("\n  items per cell:"); print(n.to_string())
    print("""
  The reversal arm goes 20.6 -> 0.0 -> 0.0. That collapse, not the dose, is what
  separates the k=1 cell from the k=2 and k=3 cells.""")

    print("\n" + "=" * 78)
    print("2. THE EFFECT SPLIT BY THAT COLUMN  -  causal query group only")
    print("=" * 78 + "\n")
    rows = []
    for lex in ("KEEP", "PSEUDO"):
        d = pd.read_csv(ROOT / "results" / f"pilot_raw_price400{lex}.csv")
        d = d[d.parsed == 1]
        print(f"  --- {lex} ---")
        print(f"  {'cond':8s} {'all':>22s} {'estimand UNCHANGED':>24s} {'estimand CHANGED':>24s}")
        for cond in ("DR_k1", "DR_k2", "DR_k3", "ED_k1", "ED_k2", "FE_k1"):
            v = paired_causal(d, cond)
            flag = C[C.cond == cond].set_index("item").estimand_unchanged.reindex(v.index)
            line, cells = f"  {cond:8s}", {}
            for label, mask in (("all", None), ("unchanged", flag == True), ("changed", flag == False)):
                x = v.values if mask is None else v[mask.fillna(False)].values
                if len(x) < 5:
                    line += f" {'n=' + str(len(x)):>22s}"
                    cells[label] = (np.nan, len(x))
                    continue
                e, lo, hi, p = boot(x)
                line += f" {e:+6.2f}[{lo:+6.2f};{hi:+6.2f}]{'*' if p < 0.05 else ' '}n={len(x):<3d}"
                cells[label] = (e, len(x))
                rows.append(dict(lexicon=lex, cond=cond, subset=label,
                                 delta_pp=round(e, 2), ci_lo=round(lo, 2),
                                 ci_hi=round(hi, 2), p_boot=round(p, 4), n_items=len(x)))
            print(line)
        print()

    print("""  HOW TO READ THIS.

  The dose axis does not measure wrongness. It measures how often the draw was
  harmless, and that share falls to zero by k=2. Conditioning on the perturbation
  actually changing the answer removes most of the apparent gradient.

  Consequences for the write-up, all of them subtractions:
  - Do not call the curve monotone in the dose.
  - Do not quote a pp-per-edge slope without this table beside it.
  - Do not present reversal against omission as a qualitative dissociation: the
    harmless share differs too (DR 20.6%, ED 31.8%, FE 55.1% at k=1), so part of
    the ordering is composition rather than error type.

  What survives the correction in the split table is narrow: KEEP DR_k2 and
  KEEP DR_k3 on the answer-changing side, -7.36 and -6.90. The k=1 answer-
  changing cell is -6.36 and does NOT survive, so the split does not by itself
  establish damage at one edge; the unsplit causal-group table in vs_raw.csv
  does that, at 2 of 3 samples.

  The third surviving cell, PSEUDO ED_k2 unchanged at -20.37, rests on n=18 and
  should be read as noise that the correction was too weak to remove rather than
  as a finding. It is printed rather than hidden for exactly that reason.

  Note also what the answer-changing column shows across k: -6.36, -7.36, -6.90.
  That is FLAT. Once the harmless draws are held out, adding reversed edges does
  not add damage - which is the cleanest statement of why the dose reading was an
  artefact.""")

    out = pd.DataFrame(rows)
    # This table splits six arms two ways across two lexicons, so it needs the
    # same correction the rest of the project applies. Without it the n=18 cells
    # read as findings.
    sub = out.subset != "all"
    out["survives_BH"] = False
    out.loc[sub, "survives_BH"] = benjamini_hochberg(out.loc[sub, "p_boot"].values)
    surv = out[sub & out.survives_BH]
    print(f"\n  Of the {int(sub.sum())} split cells, {len(surv)} survive "
          f"Benjamini-Hochberg at 0.05:")
    for _, r in surv.sort_values("p_boot").iterrows():
        print(f"    {r.lexicon:7s} {r.cond:7s} {r.subset:10s} {r.delta_pp:+7.2f}"
              f"  p={r.p_boot:.4f}  n={r.n_items}")

    C.to_csv(ROOT / "results" / "perturbation_classes.csv", index=False)
    out.to_csv(ROOT / "results" / "perturbation_split.csv", index=False)
    print(f"\n  wrote results/perturbation_classes.csv ({len(C)} rows), "
          f"perturbation_split.csv ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
