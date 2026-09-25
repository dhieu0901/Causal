"""Is a reversed edge actually a WRONG graph? For a quarter of them, no.

Why this file exists. Every dose claim in this project assumes that k reversed
edges means "k units of wrongness". It does not. Reversing an edge that no
adjustment set depends on leaves the correct ATE estimand untouched, so the
model is handed a graph that is different but not misleading. Nothing in the
repository ever checked this - src/noise.py line 121 defers the question to a
`scripts/check_identifiability.py` that was never written.

The answer matters because the share of harmless draws is not constant across k:

    reversals that leave the estimand unchanged, price400:  24.8% at k=1, 0.0% at k=2 and k=3
    the same on n600, same seven families:                  22.1% at k=1, 0.0% at k=2 and k=3

So the "dose-response curve" in results/vs_raw_trend.csv is confounded with
sample composition. Going from k=1 to k=2 does not only add a reversal, it also
removes every harmless draw from the cell. Whether the harm grows with k once
the estimand changes is a separate, model-dependent question. On price400 it
does not; on n600 it does. The n600 doses were answered a week apart, but
scripts/check_drift.py re-asked the earlier prompts and found no drift
(results/drift_check.csv). Holding the items fixed and pooling both samples
(section 4), the harm grows by about 1.8 pp per reversed edge under both
lexicons, short of 0.05 in both. The flat reading once stated here is
withdrawn.

How the drawn perturbation is recovered. pilot.py picks it with
`random.Random(f"{seed}:{i}:{t}:{k}").choice(opts)`, where `i` is the FRAME INDEX
from `items.iterrows()` and `opts` is enumerated over the NAME graph parsed from
the item's prose. classify() replays exactly that and only then maps the draw to
symbols. Until 2026-09-24 it replayed over the sorted SYMBOL graph instead - a
differently ordered list, so the same index named the wrong perturbation for
about half the draws. check_replay() now proves every replay by rebuilding the
prompt and finding it in the API cache.

Where the graph comes from. Line 2 of CLadder's `reasoning` field carries the
structure, but 67 of the 399 price400 items have no reasoning text. The
per-family structure from data/cladder-meta.json covers all of them, and this
script ASSERTS that the two agree wherever both exist (332 of 332, 0 mismatches)
before relying on it.

Limit of the criterion, stated plainly. The backdoor test is exact for `ate`
(and for `ett` and `backadj`, which use the same adjustment sets). For `nde`,
`nie` and `det-counterfactual` it is an approximation: those estimands depend on
the mediator's role, which the (X, Y) backdoor sets do not capture, so a
perturbation called harmless here may still change the label.

An earlier version of this paragraph called the harmless share a LOWER bound.
The error runs the other way. Section 1b checks it: every `nde` and `nie`
reversal the backdoor test calls harmless touches an edge on a directed X -> Y
path, i.e. changes the mediator structure. So for those types the backdoor
share OVER-counts harmless draws. Section 2b re-runs the split with a
query-type-aware flag (backdoor test for ate/ett/backadj, additionally requiring
every directed X -> Y path untouched for nde/nie/det-counterfactual); on
price400 the answer-changing harm is -8.38 / -7.36 / -6.90 at k = 1, 2, 3.

Run:  python scripts/classify_perturbations.py
Writes: results/perturbation_classes.csv, perturbation_classes_by_query.csv,
        perturbation_split.csv, perturbation_split_qt.csv, and for n600
        perturbation_classes_n600.csv, perturbation_split_n600.csv,
        perturbation_shares_n600.csv
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

from stats import boot_interval, boot_items, boot_p, boot_two_sample, cluster_boot

from perturb import to_edges, enumerate_scramble  # noqa: E402
from pilot import make_items, ENUMERATORS         # noqa: E402
from prompts import build, parse_prose_graph, strip_structure  # noqa: E402
from lexical import relabel_item                  # noqa: E402
from runner import _key, CACHE as CACHE_DIR       # noqa: E402
from analyze_querygroup import ARITH, IDENT       # noqa: E402

SEED = 20260907
NBOOT = 4000
N_ITEMS, KMAX = 399, 3
# Any model that ran every condition; its cache keys prove the replay.
REPLAY_MODEL = "gpt-4.1-nano"


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


def touches_causal_path(true_edges, bad_edges, x="X", y="Y"):
    """Does the corruption change an edge that lies on a directed x -> y path?

    An edge (u, v) is on such a path when x reaches u and v reaches y. Checked
    in both graphs, over the edges the corruption added or removed. For `nde`
    and `nie` this is what the backdoor test cannot see: reversing M -> Y keeps
    X an ancestor of Y through X -> Y and leaves the (X, Y) backdoor sets alone,
    so same_estimand() calls it harmless, yet M is no longer a mediator.
    """
    def on_path(e, edges):
        u, v = e
        return ((u == x or u in descendants(edges, x))
                and (v == y or v in ancestors(edges, y)))
    diff = set(map(tuple, true_edges)) ^ set(map(tuple, bad_edges))
    return any(on_path(e, true_edges) or on_path(e, bad_edges) for e in diff)


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
def meta_mappings() -> dict:
    """(story_id, graph_id) -> every distinct {symbol: lower-case name} in cladder-meta."""
    meta = json.loads((ROOT / "data" / "cladder-meta.json").read_text(encoding="utf-8"))
    out: dict = {}
    for m in meta:
        vm = m["variable_mapping"]
        s2n = tuple(sorted((k[:-4], v.lower()) for k, v in vm.items() if k.endswith("name")))
        out.setdefault((m["story_id"], m["graph_id"]), set()).add(s2n)
    return out


def prose_graph(prompt: str) -> list[tuple[str, str]]:
    """The name graph exactly as pilot.build_jobs parses it, edges in prose order."""
    _, removed = strip_structure(prompt)
    return parse_prose_graph(removed)


def classify(n_items=N_ITEMS, sample_kmax=KMAX, kmax=KMAX, arms=("DR", "ED", "FE"),
             lexicons=("KEEP", "PSEUDO"), scramble=False) -> pd.DataFrame:
    """Classify the perturbation each item was SHOWN, per lexicon.

    Until 2026-09-24 this replayed the draw over the canonical SYMBOL graph,
    whose edges are sorted, while pilot.py draws over the NAME graph parsed from
    the prose, whose edges come in sentence order. Same seed, same index, two
    differently ordered lists: on price400 the replay named the perturbation
    actually shown in only about half the draws, so every split built on it
    attached each item's answer to another graph's verdict.

    Now the draw is replayed exactly as build_jobs makes it - relabel, parse,
    enumerate over the name graph, choose - and only then mapped to symbols,
    through the cladder-meta variable mapping under which the prose graph IS
    the canonical structure. Where several mappings fit (an automorphism), the
    verdict must agree under all of them or the draw is left unclassified.

    Proof that the replay is right, not an argument: the prompt rebuilt from the
    replayed draw must be a key in the API cache. `replay_in_cache` records it;
    main() refuses to go on if any rebuilt prompt is missing while the cache is
    present.

    Lexicons are classified separately because FE draws differ between them:
    candidate_new_edges walks the nodes in sorted-name order, and sorting
    pseudowords orders the nodes differently. DR and ED depend on edge order
    only, which relabelling preserves.
    """
    items = make_items(n_items, SEED, sample_kmax, "full_v1.5_default.csv", None, True)
    canon = canonical_structures()
    agree, missing = check_canonical(items, canon)
    maps = meta_mappings()
    print(f"  structure source: {agree} items verified against their reasoning line, "
          f"0 mismatches;\n  {missing} items have no reasoning line and use the "
          f"per-family structure.")

    rows, no_map, ambiguous = [], 0, 0
    for i, r in items.iterrows():
        sym = canon[r["graph_id"]]
        sym_nodes = sorted({n for e in sym for n in e})
        if "X" not in sym_nodes or "Y" not in sym_nodes:
            continue
        keep_prompt, _ = relabel_item(r.prompt, "KEEP", seed=str(r.id))
        keep_edges = prose_graph(keep_prompt)
        n2s_all = []
        for m in maps.get((r.story_id, r.graph_id), ()):
            s2n = dict(m)
            if (all(a in s2n and b in s2n for a, b in sym)
                    and sorted((s2n[a], s2n[b]) for a, b in sym) == sorted(keep_edges)):
                n2s_all.append({v: s for s, v in s2n.items()})
        if not n2s_all:
            no_map += 1
            continue
        for lex in lexicons:
            prompt, clean = relabel_item(r.prompt, lex, seed=str(r.id))
            if not clean:
                continue                 # build_jobs drops it too
            edges = prose_graph(prompt)
            if len(edges) != len(sym):
                continue                 # build_jobs drops it too
            # Relabelling keeps sentence order, so edge j here is edge j in KEEP.
            pos: dict = {}
            for (a, b), (ka, kb) in zip(edges, keep_edges):
                if pos.setdefault(a, ka) != ka or pos.setdefault(b, kb) != kb:
                    raise SystemExit(f"item {i} {lex}: relabelled prose does not align "
                                     f"with KEEP edge by edge")
            nodes = sorted({n for e in edges for n in e})
            draws = []
            for t in arms:
                for k in range(1, kmax + 1):
                    opts = ENUMERATORS[t](edges, nodes, k)
                    if opts:
                        draws.append((f"{t}_k{k}", t, k, random.Random(
                            f"{SEED}:{i}:{t}:{k}").choice(opts)))
            if scramble:
                draws.append(("SCRAMBLE", "SCRAMBLE", 0, random.Random(
                    f"{SEED}:{i}:SCRAMBLE").choice(enumerate_scramble(edges, nodes))))
            for cond, t, k, shown in draws:
                verdicts = set()
                for n2s in n2s_all:
                    bad = sorted((n2s[pos[a]], n2s[pos[b]]) for a, b in shown)
                    verdicts.add((same_estimand(sym, bad, sym_nodes),
                                  touches_causal_path(sym, bad)))
                if len(verdicts) != 1:
                    ambiguous += 1
                    continue
                (unch, touch), = verdicts
                key = _key(REPLAY_MODEL, 0.0, build(prompt, "PERTURB", shown))
                rows.append(dict(item=i, lexicon=lex, family=r["graph_id"],
                                 query_type=r["query_type"], cond=cond, arm=t, k=k,
                                 estimand_unchanged=unch, touches_causal_path=touch,
                                 replay_in_cache=key.exists()))
    print(f"  {no_map} items with no fitting name mapping, {ambiguous} draws whose "
          f"verdict depends on an automorphism - both left out.\n")
    return pd.DataFrame(rows)


def check_replay(C: pd.DataFrame) -> None:
    """Every replayed prompt must be one the model was actually sent."""
    if not any(CACHE_DIR.glob("*.json")):
        print("  replay check SKIPPED: cache/ is empty (it is not in the repository).")
        return
    hit = C.replay_in_cache.mean()
    print(f"  replay check: {int(C.replay_in_cache.sum())}/{len(C)} rebuilt prompts "
          f"are keys in the API cache ({100 * hit:.1f}%).")
    if hit < 1:
        bad = C[~C.replay_in_cache].groupby(["lexicon", "cond"]).size()
        raise SystemExit(f"  REPLAY DOES NOT MATCH WHAT WAS SENT:\n{bad}")


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
    return boot_items(x, SEED, NBOOT)               # convention A, src/stats.py


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
    check_replay(C)
    # DR and ED draws are the same under both lexicons; FE draws are not (see
    # classify). Composition is reported on KEEP, the splits use each lexicon's own.
    K = C[C.lexicon == "KEEP"]

    print("=" * 78)
    print("1. SHARE OF THE DRAWN PERTURBATIONS THAT LEAVE THE ATE ESTIMAND UNCHANGED")
    print("=" * 78 + "\n")
    share = K.pivot_table(index="arm", columns="k", values="estimand_unchanged",
                          aggfunc=lambda s: round(100 * float(np.mean(s)), 1))
    n = K.pivot_table(index="arm", columns="k", values="estimand_unchanged", aggfunc="size")
    print("  percent harmless:"); print(share.to_string())
    print("\n  items per cell:"); print(n.to_string())
    dr = " -> ".join(f"{v:.1f}" for v in share.loc["DR"].dropna())
    print(f"""
  The reversal arm goes {dr}. That collapse, not the dose, is what
  separates the k=1 cell from the k=2 and k=3 cells.""")

    # ------------------------------------------------------------------
    # 1b. Which way does the approximation err? (review item MOI-3)
    # The backdoor test is exact for `ate` only. An earlier docstring called
    # the harmless share a LOWER bound; for the natural effects it errs the
    # other way, because same_estimand() cannot see a mediator losing its
    # role. So report the exact `ate`-only share next to the pooled one, and a
    # strict share that also requires every directed X -> Y path untouched.
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("1b. HARMLESS SHARE BY QUERY TYPE - where the criterion is exact, and where not")
    print("=" * 78 + "\n")
    C["strict_unchanged"] = C.estimand_unchanged & ~C.touches_causal_path
    K = C[C.lexicon == "KEEP"]
    qrows = []
    for (arm, k), g in K.groupby(["arm", "k"]):
        for label, s in [("all query types", g), ("ate only (exact)", g[g.query_type == "ate"])] + \
                        [(q, g[g.query_type == q]) for q in sorted(g.query_type.unique())]:
            if not len(s):
                continue
            qrows.append({"arm": arm, "k": k, "subset": label, "n": len(s),
                          "harmless_pct": round(100 * s.estimand_unchanged.mean(), 1),
                          "harmless_strict_pct": round(100 * s.strict_unchanged.mean(), 1),
                          "harmless_but_touches_path": int((s.estimand_unchanged & s.touches_causal_path).sum())})
    Q = pd.DataFrame(qrows)
    Q.to_csv(ROOT / "results" / "perturbation_classes_by_query.csv", index=False)
    show = Q[(Q.arm == "DR")]
    print(show.to_string(index=False))
    nat = K[(K.arm == "DR") & K.query_type.isin(["nde", "nie"]) & K.estimand_unchanged]
    print(f"""
  nde/nie reversals the backdoor test calls harmless: {len(nat)}, of which
  {int(nat.touches_causal_path.sum())} touch a directed X -> Y path.

  harmless_pct is the backdoor test; harmless_strict_pct additionally requires
  that no edge on a directed X -> Y path was touched. Which one is RIGHT
  depends on the query type:

    ate, ett, backadj   the backdoor test. If X stays an ancestor of Y and the
                        valid adjustment sets are unchanged, the adjustment
                        formula is unchanged, so the answer is too - touching
                        a directed path elsewhere does not matter. The strict
                        test is too conservative here.
    nde, nie,           the strict test. These depend on the mediator's role,
    det-counterfactual  which the (X, Y) backdoor test cannot see. Every nde
                        and nie reversal it called harmless touches a directed
                        X -> Y path, so the backdoor share OVER-counts there:
                        it is an upper bound, not the lower bound an earlier
                        docstring claimed.""")
    # Query-type-aware flag, used by the split in section 2b.
    strict_types = {"nde", "nie", "det-counterfactual"}
    C["unchanged_qt"] = np.where(C.query_type.isin(strict_types),
                                 C.strict_unchanged, C.estimand_unchanged)

    print("\n" + "=" * 78)
    print("2. THE EFFECT SPLIT BY THAT COLUMN  -  causal query group only")
    print("=" * 78 + "\n")
    rows = []
    for lex in ("KEEP", "PSEUDO"):
        d = pd.read_csv(ROOT / "results" / "raw" / f"pilot_raw_price400{lex}.csv")
        d = d[d.parsed == 1]
        print(f"  --- {lex} ---")
        print(f"  {'cond':8s} {'all':>22s} {'estimand UNCHANGED':>24s} {'estimand CHANGED':>24s}")
        for cond in ("DR_k1", "DR_k2", "DR_k3", "ED_k1", "ED_k2", "FE_k1"):
            v = paired_causal(d, cond)
            flag = C[(C.cond == cond) & (C.lexicon == lex)].set_index("item").estimand_unchanged.reindex(v.index)
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

    out = pd.DataFrame(rows)
    k1 = {a: share.loc[a, 1] for a in ("DR", "ED", "FE")}
    chg = out[(out.lexicon == "KEEP") & (out.subset == "changed")].set_index("cond").delta_pp
    print(f"""  HOW TO READ THIS.

  The dose axis does not measure wrongness. It measures how often the draw was
  harmless, and that share falls to zero by k=2. Conditioning on the perturbation
  actually changing the answer removes the apparent gradient.

  Consequences for the write-up, all of them subtractions:
  - Do not call the curve monotone in the dose.
  - Do not quote a pp-per-edge slope without this table beside it.
  - Do not present reversal against omission as a qualitative dissociation: the
    harmless share differs too (DR {k1["DR"]:.1f}%, ED {k1["ED"]:.1f}%, FE {k1["FE"]:.1f}% at
    k=1), so part of the ordering is composition rather than error type.

  The KEEP answer-changing column across k: {chg["DR_k1"]:+.2f}, {chg["DR_k2"]:+.2f}, {chg["DR_k3"]:+.2f}.
  Once the harmless draws are held out, adding reversed edges does not add
  damage.

  Until 2026-09-24 this table was built on a replay that named the wrong draw
  for about half the items (see classify). Two cells it reported do not survive
  the correction: PSEUDO ED_k2 unchanged at -20.37 and KEEP FE_k1 unchanged at
  -7.50. Both were artefacts of mislabelled items.""")
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

    # ------------------------------------------------------------------
    # 2b. The same split with the query-type-aware flag of section 1b. The
    # table above uses the (X, Y) backdoor test for every query type, which
    # section 1b shows over-counts harmless reversals for nde, nie and
    # det-counterfactual. Only k=1 has harmless draws, so only k=1 can move.
    # Written to its own file so the table above keeps its numbers.
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("2b. THE SPLIT AGAIN, WITH THE QUERY-TYPE-AWARE FLAG")
    print("=" * 78 + "\n")
    qt_rows = []
    for lex in ("KEEP", "PSEUDO"):
        d = pd.read_csv(ROOT / "results" / "raw" / f"pilot_raw_price400{lex}.csv")
        d = d[d.parsed == 1]
        for cond in ("DR_k1", "DR_k2", "DR_k3"):
            v = paired_causal(d, cond)
            flag = C[(C.cond == cond) & (C.lexicon == lex)].set_index("item").unchanged_qt.reindex(v.index)
            for label, mask in (("unchanged (qt)", flag == True), ("changed (qt)", flag == False)):
                x = v[mask.fillna(False)].values
                if len(x) < 5:
                    qt_rows.append(dict(lexicon=lex, cond=cond, subset=label, n_items=len(x)))
                    continue
                e, lo, hi, p = boot(x)
                print(f"  {lex:7s} {cond:6s} {label:15s} {e:+7.2f} [{lo:+7.2f} ; {hi:+7.2f}]"
                      f"  p={p:.4f}  n={len(x)}")
                qt_rows.append(dict(lexicon=lex, cond=cond, subset=label,
                                    delta_pp=round(e, 2), ci_lo=round(lo, 2),
                                    ci_hi=round(hi, 2), p_boot=round(p, 4), n_items=len(x)))
        # The check on the classification itself: split ORACLE by the DR_k1
        # flag. Where the reversal leaves the estimand alone, DR_k1 should do
        # about as well as the correct graph; where it changes it, worse.
        v = paired_causal(d, "ORACLE")
        flag = C[(C.cond == "DR_k1") & (C.lexicon == lex)].set_index("item").unchanged_qt.reindex(v.index)
        for label, mask in (("unchanged (qt)", flag == True), ("changed (qt)", flag == False)):
            x = v[mask.fillna(False)].values
            e, lo, hi, p = boot(x)
            print(f"  {lex:7s} ORACLE on the DR_k1 split, {label:15s} {e:+7.2f} "
                  f"[{lo:+7.2f} ; {hi:+7.2f}]  p={p:.4f}  n={len(x)}")
            qt_rows.append(dict(lexicon=lex, cond="ORACLE, items split by DR_k1", subset=label,
                                delta_pp=round(e, 2), ci_lo=round(lo, 2),
                                ci_hi=round(hi, 2), p_boot=round(p, 4), n_items=len(x)))
    pd.DataFrame(qt_rows).to_csv(ROOT / "results" / "perturbation_split_qt.csv", index=False)

    N = n600_dose()
    if N is not None:
        conditional_slope(C, N)

    # replay_in_cache proves the replay here but is not written: the cache is
    # not in the repository, so a fresh clone would write False everywhere and
    # the file would stop reproducing byte for byte.
    C.drop(columns="replay_in_cache").to_csv(ROOT / "results" / "perturbation_classes.csv",
                                            index=False)
    out.to_csv(ROOT / "results" / "perturbation_split.csv", index=False)
    print(f"\n  wrote results/perturbation_classes.csv ({len(C)} rows), "
          f"perturbation_split.csv ({len(out)} rows)")
    return 0


def n600_dose() -> pd.DataFrame | None:
    """Section 3: the dose line on all ten families (runs of 2026-09-24).

    price400 was drawn with k up to 3, which the three 2-edge families cannot
    support, so its dose line covers seven families. n600 covers all ten, and
    scripts/run_n600_extensions.sh added DR_k2, DR_k3 and SCRAMBLE to it without
    changing the sample. Same replay, same cache proof, same split.

    Two restrictions are reported side by side, because they answer different
    questions: all ten families (k=3 exists on seven only), and the seven
    families where every k exists, which holds the items fixed across k.
    """
    print("\n" + "=" * 78)
    print("3. n600: THE DOSE LINE ON ALL TEN FAMILIES, SPLIT THE SAME WAY")
    print("=" * 78 + "\n")
    arms_files = [ROOT / "results" / "raw" / f"pilot_raw_n600arms{lex}.csv" for lex in ("KEEP", "PSEUDO")]
    if not all(f.exists() for f in arms_files):
        print("  SKIPPED: run scripts/run_n600_extensions.sh first.")
        return None
    N = classify(600, 1, 3, arms=("DR",), scramble=True)
    check_replay(N)
    strict_types = {"nde", "nie", "det-counterfactual"}
    strict = N.estimand_unchanged & ~N.touches_causal_path
    N["unchanged_qt"] = np.where(N.query_type.isin(strict_types), strict, N.estimand_unchanged)
    K = N[N.lexicon == "KEEP"]
    shares = K.groupby("cond").agg(n=("unchanged_qt", "size"),
                                   harmless_backdoor_pct=("estimand_unchanged", "mean"),
                                   harmless_qt_pct=("unchanged_qt", "mean"))
    shares[["harmless_backdoor_pct", "harmless_qt_pct"]] *= 100
    print("  harmless share, KEEP (DR and SCRAMBLE draws do not depend on the lexicon):")
    print(shares.round(1).to_string())

    seven = {"IV", "arrowhead", "confounding", "diamond", "diamondcut", "frontdoor", "mediation"}
    rows = []
    for lex in ("KEEP", "PSEUDO"):
        base = pd.read_csv(ROOT / "results" / "raw" / f"pilot_raw_n600{lex}.csv")
        extra = pd.read_csv(ROOT / "results" / "raw" / f"pilot_raw_n600arms{lex}.csv")
        d = pd.concat([base, extra], ignore_index=True)
        d = d[d.parsed == 1]
        print(f"\n  --- {lex} ---")
        for fams, label in ((None, "all 10"), (seven, "7 with k=3")):
            dd = d if fams is None else d[d.graph_id.isin(fams)]
            per_k = {}
            for cond in ("DR_k1", "DR_k2", "DR_k3", "SCRAMBLE"):
                v = paired_causal(dd, cond)
                if len(v) < 5:
                    continue
                per_k[cond] = v
                flag = N[(N.cond == cond) & (N.lexicon == lex)].set_index("item").unchanged_qt.reindex(v.index)
                for sub_, mask in (("all", None), ("unchanged (qt)", flag == True),
                                   ("changed (qt)", flag == False)):
                    x = v.values if mask is None else v[mask.fillna(False)].values
                    if len(x) < 5:
                        rows.append(dict(lexicon=lex, families=label, cond=cond,
                                         subset=sub_, n_items=len(x)))
                        continue
                    e, lo, hi, p = boot(x)
                    print(f"  {label:10s} {cond:8s} {sub_:15s} {e:+7.2f} [{lo:+7.2f} ; {hi:+7.2f}]"
                          f"  p={p:.4f}  n={len(x)}")
                    rows.append(dict(lexicon=lex, families=label, cond=cond, subset=sub_,
                                     delta_pp=round(e, 2), ci_lo=round(lo, 2),
                                     ci_hi=round(hi, 2), p_boot=round(p, 4), n_items=len(x)))
            if label == "7 with k=3" and all(c in per_k for c in ("DR_k1", "DR_k2", "DR_k3")):
                # One slope per item across its own three k, as in analyze_vs_raw.
                M = pd.concat({k: per_k[f"DR_k{k}"] for k in (1, 2, 3)}, axis=1).dropna()
                kc = np.array([-1.0, 0.0, 1.0])
                slope = (M.values * kc).sum(axis=1) / (kc ** 2).sum()
                e, lo, hi, p = boot(slope)
                print(f"  {label:10s} slope per extra reversed edge {e:+.2f} [{lo:+.2f} ; {hi:+.2f}]"
                      f"  p={p:.4f}  n={len(slope)}")
                rows.append(dict(lexicon=lex, families=label, cond="DR slope",
                                 subset="all", delta_pp=round(e, 2), ci_lo=round(lo, 2),
                                 ci_hi=round(hi, 2), p_boot=round(p, 4), n_items=len(slope)))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "perturbation_split_n600.csv", index=False)
    N.drop(columns="replay_in_cache").to_csv(
        ROOT / "results" / "perturbation_classes_n600.csv", index=False)
    shares.round(1).reset_index().to_csv(ROOT / "results" / "perturbation_shares_n600.csv",
                                         index=False)
    print("\n  wrote results/perturbation_split_n600.csv, perturbation_classes_n600.csv,"
          " perturbation_shares_n600.csv")
    return N


SEVEN = {"IV", "arrowhead", "confounding", "diamond", "diamondcut", "frontdoor", "mediation"}


def conditional_slope(C: pd.DataFrame, N: pd.DataFrame) -> None:
    """Section 4: does harm grow with k on items whose every draw changes the estimand?

    The split tables compare k=1, 2, 3 cells that hold DIFFERENT item sets on
    the changed side (at k=1 the harmless items drop out, at k=2 and 3 none
    do). This holds the items fixed instead: keep an item only if its k=1 draw
    already changes the estimand - then its k=2 and k=3 draws do too - and fit
    one slope per item across its own three k. That is the test "flat once
    conditioned" actually needs.

    Per sample, then pooled over price400 and n600 by CLadder id (an id drawn
    in both contributes the mean of its two slopes; the two samples drew
    different corruptions for it). On n600, k=1 was answered on a different day
    from k=2 and k=3, so the n600 slope, and with it the pooled one, would carry
    any drift between the runs; scripts/check_drift.py found none. The k=2 to
    k=3 step is within one run and is reported beside it. The difference between the two samples is bootstrapped
    with the samples treated as independent, which ignores the 139 shared ids.
    """
    print("\n" + "=" * 78)
    print("4. CONDITIONAL SLOPE - items fixed, every draw changes the estimand")
    print("=" * 78 + "\n")
    maps = {t: pd.read_csv(ROOT / "results" / f"_itemmap_{t}.csv").set_index("item").id
            for t in ("price400", "n600")}
    rows = []
    for lex in ("KEEP", "PSEUDO"):
        per_sample = {}
        for tag, cls, files in (
                ("price400", C, [f"pilot_raw_price400{lex}.csv"]),
                ("n600", N, [f"pilot_raw_n600{lex}.csv", f"pilot_raw_n600arms{lex}.csv"])):
            d = pd.concat([pd.read_csv(ROOT / "results" / "raw" / f) for f in files], ignore_index=True)
            d = d[(d.parsed == 1) & d.graph_id.isin(SEVEN)]
            M = pd.concat({k: paired_causal(d, f"DR_k{k}") for k in (1, 2, 3)}, axis=1).dropna()
            flag = cls[(cls.cond == "DR_k1") & (cls.lexicon == lex)].set_index("item").unchanged_qt
            # Before conditioning: every item with all three k, harmless or not.
            # The difference from the conditional slope below is how much the
            # composition confound actually moves a slope in these data
            # (review round 11, M11-1).
            uncond = ((M[3] - M[1]) / 2).values
            M = M[flag.reindex(M.index).eq(False).values]
            slope = pd.Series(((M[3] - M[1]) / 2).values, index=maps[tag].reindex(M.index).values)
            step = (M[3] - M[2]).values
            eu = boot(uncond)
            print(f"  {lex:7s} {tag:9s} {'unconditional slope':24s} {eu[0]:+7.2f} "
                  f"[{eu[1]:+7.2f} ; {eu[2]:+7.2f}]  p={eu[3]:.4f}  n={len(uncond)}")
            rows.append(dict(lexicon=lex, sample=tag, quantity="unconditional slope",
                             delta_pp=round(eu[0], 2), ci_lo=round(eu[1], 2),
                             ci_hi=round(eu[2], 2), p_boot=round(eu[3], 4),
                             n_items=len(uncond)))
            ec = boot(slope.values)
            rows.append(dict(lexicon=lex, sample=tag,
                             quantity="composition effect (conditional minus unconditional slope)",
                             delta_pp=round(ec[0] - eu[0], 2), n_items=len(slope)))
            per_sample[tag] = slope
            for what, x in (("slope per reversed edge", slope.values), ("k=2 to k=3 step", step)):
                e, lo, hi, p = boot(x)
                print(f"  {lex:7s} {tag:9s} {what:24s} {e:+7.2f} [{lo:+7.2f} ; {hi:+7.2f}]"
                      f"  p={p:.4f}  n={len(x)}")
                rows.append(dict(lexicon=lex, sample=tag, quantity=what, delta_pp=round(e, 2),
                                 ci_lo=round(lo, 2), ci_hi=round(hi, 2), p_boot=round(p, 4),
                                 n_items=len(x)))
        pooled = pd.concat(per_sample.values()).groupby(level=0).mean()
        e, lo, hi, p = boot(pooled.values)
        print(f"  {lex:7s} {'pooled':9s} {'slope per reversed edge':24s} {e:+7.2f} "
              f"[{lo:+7.2f} ; {hi:+7.2f}]  p={p:.4f}  n={len(pooled)}")
        rows.append(dict(lexicon=lex, sample="pooled", quantity="slope per reversed edge",
                         delta_pp=round(e, 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                         p_boot=round(p, 4), n_items=len(pooled)))
        # Sorted by CLadder id: the bootstrap resamples POSITIONS, so without a
        # fixed order the same seed gives a different interval in every script
        # that rebuilds these vectors (check_consistency.py caught one).
        a_ = per_sample["n600"].sort_index().values
        b_ = per_sample["price400"].sort_index().values
        # The same two-sample bootstrap as analyze_prior_strength (src/stats.py).
        e, lo, hi, p = boot_two_sample(a_, b_, SEED, NBOOT)
        print(f"  {lex:7s} {'n600 - price400':24s}           {e:+7.2f} "
              f"[{lo:+7.2f} ; {hi:+7.2f}]  p={p:.4f}\n")
        rows.append(dict(lexicon=lex, sample="n600 minus price400",
                         quantity="slope per reversed edge", delta_pp=round(e, 2),
                         ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                         p_boot=round(p, 4), n_items=len(a_) + len(b_)))
    pd.DataFrame(rows).to_csv(ROOT / "results" / "perturbation_conditional_slope.csv", index=False)
    print("  wrote results/perturbation_conditional_slope.csv")
    validate_against_models(C, N)


def validate_against_models(C: pd.DataFrame, N: pd.DataFrame) -> None:
    """Section 5: does the estimand classification predict what the models do?

    If a reversal that leaves the estimand intact is truly harmless, the model
    given it should do as well as the model given the correct graph, on the
    same items; where the estimand changes, worse. Paired within item, both
    samples, both lexicons, seven families with k=3. Review round 11 (M11-2)
    found the manuscript claiming this on one sample only.
    """
    print("\n" + "=" * 78)
    print("5. THE CLASSIFICATION AGAINST THE MODELS: DR_k1 minus ORACLE, paired")
    print("=" * 78 + "\n")
    rows = []
    for tag, cls, files in (("price400", C, ["pilot_raw_price400{}.csv"]),
                            ("n600", N, ["pilot_raw_n600{}.csv"])):
        for lex in ("KEEP", "PSEUDO"):
            d = pd.concat([pd.read_csv(ROOT / "results" / "raw" / f.format(lex)) for f in files])
            d = d[(d.parsed == 1) & d.graph_id.isin(SEVEN)]
            o, r_ = paired_causal(d, "ORACLE"), paired_causal(d, "DR_k1")
            flag = cls[(cls.cond == "DR_k1") & (cls.lexicon == lex)].set_index("item").unchanged_qt
            i = o.index.intersection(r_.index)
            f_ = flag.reindex(i)
            for label, mask in (("estimand unchanged", f_.eq(True).values),
                                ("estimand changed", f_.eq(False).values)):
                idx = i[mask]
                for what, x in (("DR_k1 minus RAW", r_[idx].values),
                                ("ORACLE minus RAW", o[idx].values),
                                ("DR_k1 minus ORACLE", (r_[idx] - o[idx]).values)):
                    e, lo, hi, p = boot(x)
                    if what == "DR_k1 minus ORACLE":
                        print(f"  {tag:9s} {lex:7s} {label:19s} {what:19s} {e:+7.2f} "
                              f"[{lo:+7.2f} ; {hi:+7.2f}]  p={p:.4f}  n={len(x)}")
                    rows.append(dict(sample=tag, lexicon=lex, subset=label, quantity=what,
                                     delta_pp=round(e, 2), ci_lo=round(lo, 2),
                                     ci_hi=round(hi, 2), p_boot=round(p, 4), n_items=len(x)))
    pd.DataFrame(rows).to_csv(ROOT / "results" / "perturbation_validation.csv", index=False)
    print("\n  wrote results/perturbation_validation.csv")


if __name__ == "__main__":
    raise SystemExit(main())
