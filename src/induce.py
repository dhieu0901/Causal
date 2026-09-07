"""Graph induction: make the model build the DAG itself, then score it.

This is the bridge between the two halves of the study. Phase 1 measures what a
corrupted graph costs. Induction measures how corrupted a real agent's own graph
actually is. Only together do they say whether graph-guided reasoning pays off in
practice.

The variable list is GIVEN to the model and it is asked only for the edges. That
isolates structure discovery from entity extraction, matches how NoisyCausal
reports edge-level P/R/F1 in its Table 3, and makes the output parseable without
fuzzy name matching - a parse failure would otherwise be scored as a graph error
and silently inflate the induction-error estimate.
"""
from __future__ import annotations
import re

# The model is allowed to reason before committing. Forcing a bare edge list with
# "do not explain" measures how well it answers under a gag, not how well it can
# recover structure, and understates induction quality. Everything before the
# EDGES: marker is ignored by the parser.
INDUCE_TEMPLATE = """{body}

The variables in this world are:
{varlist}

Work out the causal structure. Which variables directly cause which? The
conditional probabilities stated above tell you which variables a quantity
depends on, and therefore which arrows exist.

Think it through first, then give your final answer as a list of directed edges
after a line containing only EDGES:, one edge per line in this format:

EDGES:
CAUSE -> EFFECT
CAUSE -> EFFECT

Use only the variable names listed above, spelled exactly as written. Include
every direct cause-effect pair you can justify, and no others."""

EDGE_LINE = re.compile(r"^\s*(.+?)\s*(?:->|→|-->)\s*(.+?)\s*$")
_ARTICLES = re.compile(r"^(?:the|a|an)\s+")


def _norm(s: str) -> str:
    s = re.sub(r"[*_`\"']", "", s).strip().lower()
    return _ARTICLES.sub("", s).strip(" .:;")


def resolve_name(raw: str, nodes: list[str]) -> str | None:
    """Map what the model wrote onto a known variable, or None.

    Models paraphrase: given the variable "yield per acre" a model may write
    "crop yield per acre". Rejecting that would score a correct edge as a parse
    failure, and parse failures are counted as graph errors - so sloppy matching
    here silently understates induction quality, which is the very thing being
    measured. Containment is accepted; an ambiguous match is not.
    """
    r = _norm(raw)
    if not r:
        return None
    exact = [n for n in nodes if _norm(n) == r]
    if exact:
        return exact[0]
    contained = [n for n in nodes if _norm(n) in r or r in _norm(n)]
    if len(contained) == 1:
        return contained[0]
    if len(contained) > 1:                    # prefer the longest overlap
        best = max(contained, key=lambda n: len(_norm(n)))
        ties = [n for n in contained if len(_norm(n)) == len(_norm(best))]
        return best if len(ties) == 1 else None
    return None


def build_induce_prompt(body: str, nodes: list[str]) -> str:
    varlist = "\n".join(f"- {n}" for n in sorted(nodes))
    return INDUCE_TEMPLATE.format(body=body.rstrip(), varlist=varlist)


def parse_edges(text: str, nodes: list[str]) -> list[tuple[str, str]]:
    """Keep only edges whose endpoints are known variables. Order preserved.

    Only the text after the last EDGES: marker is read, so arrows written while
    the model is thinking out loud are not mistaken for its final answer. If the
    marker is absent the whole response is scanned, which keeps a model that
    ignored the format from being scored as having produced no graph.
    """
    body = (text or "")
    if "EDGES:" in body.upper():
        idx = body.upper().rfind("EDGES:")
        body = body[idx + len("EDGES:"):]

    out, seen = [], set()
    for line in body.splitlines():
        line = line.strip().strip("-*0123456789. ")
        if not line or line.upper().startswith(("BEGIN", "END")):
            continue
        m = EDGE_LINE.match(line)
        if not m:
            continue
        a = resolve_name(m.group(1), nodes)
        b = resolve_name(m.group(2), nodes)
        if a and b and a != b and (a, b) not in seen:
            seen.add((a, b))
            out.append((a, b))
    return out


def edge_f1(pred: list[tuple[str, str]], true: list[tuple[str, str]]) -> dict:
    """Direction-sensitive edge-level scores, plus a breakdown by error type.

    A reversed edge costs twice under F1 - once as a false positive and once as a
    false negative - which is why F1 alone understates how many EDGES are wrong.
    The breakdown separates the three error types the perturbation study prices:
    reversed, spurious (false edge), and missing (deleted edge).
    """
    P, T = set(pred), set(true)
    tp = len(P & T)
    fp, fn = len(P - T), len(T - P)
    prec = tp / len(P) if P else 0.0
    rec = tp / len(T) if T else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    reversed_ = {(a, b) for (a, b) in P - T if (b, a) in T}
    spurious = (P - T) - reversed_
    missing = {(a, b) for (a, b) in T - P if (b, a) not in P}
    return {
        "precision": prec, "recall": rec, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "n_pred": len(P), "n_true": len(T),
        "n_reversed": len(reversed_),      # DR-type errors
        "n_spurious": len(spurious),       # FE-type errors
        "n_missing": len(missing),         # ED-type errors
        "n_wrong_edges": len(reversed_) + len(spurious) + len(missing),
        "exact_match": int(P == T),
    }
