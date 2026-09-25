"""The gold answer of a CLadder item after CI_ACTIVE, from the extended graph.

    python scripts/ci_active_gold.py

What CI_ACTIVE does. src/noise.py's inject_ci_active adds one sentence before
the question: "An unmeasured factor, C, raises both the chance of receiving the
treatment and the chance of the outcome occurring." That is a latent U with
U -> X and U -> Y, a bidirected edge X <-> Y in the latent projection. It comes
with no numbers, so the SCM it extends is not parameterised, and the new gold
answer can only be one of two things:

  preserved     the item's own answer, because what the question asks is still
                computed correctly from the numbers the prompt states
  undetermined  no yes/no follows from the prompt any more; the right response
                is that the stated numbers do not settle the question

Nothing here is decided by family name. Each rule reads the structure:

  1. rung 1                 marginal, correlation, exp_away ask about the stated
                            probabilities themselves; a latent cause does not
                            change a stated number.                -> preserved
  2. det-counterfactual     the prompt's structural equations fully determine Y;
                            an extra, unmodelled cause of Y means they no longer
                            do.                                    -> undetermined
  3. backadj               the question asks which of two adjustment sets is
                            correct. Neither set named in the prompt contains
                            U, so neither blocks X <- U -> Y. This holds in the
                            IV and frontdoor families too, whose backadj items
                            offer the latent confounder itself as a set to
                            adjust for.                            -> undetermined
  4. X <-> Y already there  a latent node with edges into both X and Y (read
                            off the "... is unobserved." sentence and the item's
                            variable mapping, asserted identical across a
                            family) means adding U changes nothing in the latent
                            projection, so a formula that never adjusts for that
                            latent - IV's ratio, the frontdoor sum - still holds.
                                                                   -> preserved
  5. collider_bias          the answer is that X does not cause Y; a common
                            cause adds no directed path from X to Y.
                                                                   -> preserved
  6. otherwise              CLadder's formula (a backdoor adjustment, or none)
                            assumes no unmeasured X-Y confounding, which U
                            breaks. The one alternative identification, a
                            frontdoor through mediators when X has no direct
                            edge to Y, needs P(mediator | X); the script reads
                            the probabilities the prompt states (from CLadder's
                            own reasoning, asserted identical across a family
                            and query type) and checks for it. If any item could
                            be recomputed that way the script STOPS rather than
                            call it undetermined; none can.       -> undetermined

This makes CI_ACTIVE usable: items marked preserved keep their label, items
marked undetermined need a third answer ("cannot be determined") and must not be
scored against the clean label. NoisyCausal scores its confounder injection
against the clean SCM; this is the table that avoids that.

Writes: results/cladder/ci_active_gold.csv (one row per family x query type)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from perturb import FAMILY_STRUCTURE, to_edges

LATENT = re.compile(r"(?:^|[.:]\s*)([^.:]*?) is unobserved\.", re.IGNORECASE)
TERM = re.compile(r"P\(\s*([A-Z]\w*)\s*=\s*\d\s*\|\s*([^)]*)\)")


def symbol_names(reasoning: str) -> dict[str, str]:
    first = str(reasoning).splitlines()[0].removeprefix("Let ").rstrip(".")
    return {k.strip(): v.strip().lower()
            for k, v in (p.split("=", 1) for p in first.split(";") if "=" in p)}


def has_reasoning(d: pd.DataFrame) -> pd.Series:
    """CLadder leaves some reasoning empty and some as lines of the STRING "nan"."""
    first = d.reasoning.astype(str).str.split(chr(10)).str[0].str.strip()
    return d.reasoning.notna() & (first != "nan")


def latent_symbols(d: pd.DataFrame) -> dict[str, set]:
    """Per family, the symbols the prompts declare unobserved. Must agree across items."""
    out = {}
    for fam, s in d[has_reasoning(d)].groupby("graph_id"):
        seen = set()
        for _, r in s.iterrows():
            names = {m.strip().lower() for m in LATENT.findall(r.prompt)}
            sym = {k for k, v in symbol_names(r.reasoning).items() if v in names}
            if names and len(sym) != len(names):
                raise SystemExit(f"{fam} item {r.id}: an unobserved name maps to no symbol")
            seen.add(frozenset(sym))
        if len(seen) != 1:
            raise SystemExit(f"{fam}: items disagree on which node is unobserved: {seen}")
        out[fam] = set(next(iter(seen)))
    return out


def stated_terms(reasoning: str) -> frozenset:
    """(outcome, conditioning set) of every probability the prompt states."""
    terms = set()
    for line in str(reasoning).splitlines()[4:]:
        for lhs, rhs in TERM.findall(line):
            if "=" in line.split(")")[-1]:
                cond = frozenset(v.split("=")[0].strip() for v in rhs.split(","))
                terms.add((lhs, cond))
    return frozenset(terms)


def mediators(edges) -> set:
    """Nodes on a directed X -> Y path, X and Y excluded."""
    ch = {}
    for a, b in edges:
        ch.setdefault(a, set()).add(b)

    def reach(x):
        seen, st = set(), [x]
        while st:
            for q in ch.get(st.pop(), ()):
                if q not in seen:
                    seen.add(q)
                    st.append(q)
        return seen
    return {m for m in reach("X") if m != "Y" and "Y" in reach(m)}


def main() -> int:
    d = pd.read_csv(ROOT / "data" / "cladder" / "full_v1.5_default.csv", low_memory=False)
    lat = latent_symbols(d)
    rows = []
    for (fam, qt), s in d.groupby(["graph_id", "query_type"]):
        edges = to_edges(FAMILY_STRUCTURE[fam])
        rung = int(s.rung.iloc[0])
        xy_latent = any(("X" in {b for a, b in edges if a == L}) and
                        ("Y" in {b for a, b in edges if a == L}) for L in lat.get(fam, ()))
        if rung == 1:
            rule, gold = "1 rung 1: stated probabilities", "preserved"
        elif qt == "det-counterfactual":
            rule, gold = "2 structural equations no longer determine Y", "undetermined"
        elif qt == "backadj":
            rule, gold = "3 no adjustment set in either method blocks X <- U -> Y", "undetermined"
        elif xy_latent:
            rule, gold = "4 latent X-Y common cause already in the graph", "preserved"
        elif qt == "collider_bias":
            rule, gold = "5 no directed X -> Y path to add", "preserved"
        else:
            stated = {stated_terms(r) for r in s[has_reasoning(s)].reasoning}
            stated.discard(frozenset())
            if len(stated) > 1:
                raise SystemExit(f"{fam} {qt}: items state different probabilities: {stated}")
            terms = next(iter(stated)) if stated else frozenset()
            direct = ("X", "Y") in edges
            med = mediators(edges)
            has_pm = any(o in med and "X" in c for o, c in terms)
            if not direct and med and has_pm:
                raise SystemExit(f"{fam} {qt}: a frontdoor recomputation looks possible "
                                 f"from {sorted(terms)}; compute it, do not call it undetermined")
            why = ("X -> Y direct: not identified" if direct else
                   "frontdoor needs P(mediator | X), not stated")
            rule, gold = f"6 formula assumes no X-Y confounding; {why}", "undetermined"
        n = len(s)
        n_real = int((~s.story_id.astype(str).str.startswith("nonsense")).sum())
        rows.append(dict(graph_id=fam, query_type=qt, rung=rung, n_items=n,
                         n_real_word=n_real, gold=gold, rule=rule))
    T = pd.DataFrame(rows)
    T.to_csv(ROOT / "results" / "cladder" / "ci_active_gold.csv", index=False)
    print(T.to_string(index=False))
    tot = T.n_items.sum()
    pres = T.loc[T.gold == "preserved", "n_items"].sum()
    print(f"\n  latent nodes per family: { {k: sorted(v) for k, v in lat.items() if v} }")
    print(f"  preserved {pres}/{tot} items ({100 * pres / tot:.1f}%), "
          f"undetermined {tot - pres}/{tot} ({100 * (tot - pres) / tot:.1f}%)")
    print("  wrote results/cladder/ci_active_gold.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
