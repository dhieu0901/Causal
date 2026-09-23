"""Controlled DAG perturbations: edge deletion, false edge, direction reversal.

Graphs in CLadder are tiny (3-5 nodes), so every perturbation set is enumerated
exhaustively rather than sampled. That gives exact feasibility counts instead of
Monte-Carlo estimates, and lets the experiment draw perturbations uniformly from
the true space of valid k-error graphs.
"""
from __future__ import annotations
from itertools import combinations
import networkx as nx

# graph_id -> structure, verified 1:1 across all 7064 CLadder models
FAMILY_STRUCTURE = {
    "chain":       "X->V2,V2->Y",
    "fork":        "X->Y,V2->Y",
    "collision":   "X->V3,Y->V3",
    "confounding": "V1->X,V1->Y,X->Y",
    "mediation":   "X->V2,X->Y,V2->Y",
    "IV":          "V1->X,V2->X,V1->Y,X->Y",
    "diamond":     "X->V3,X->V2,V2->Y,V3->Y",
    "diamondcut":  "V1->V3,V1->X,X->Y,V3->Y",
    "frontdoor":   "V1->X,X->V3,V1->Y,V3->Y",
    "arrowhead":   "X->V3,V2->V3,X->Y,V2->Y,V3->Y",
}


def to_edges(structure: str) -> list[tuple[str, str]]:
    return [tuple(p.strip().split("->")) for p in structure.split(",") if p.strip()]


def to_structure(edges) -> str:
    return ",".join(f"{a}->{b}" for a, b in edges)


def is_dag(edges, nodes) -> bool:
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return nx.is_directed_acyclic_graph(g)


def enumerate_ed(edges, nodes, k):
    """All ways to delete exactly k edges. Deletion can never create a cycle."""
    return [sorted(set(edges) - set(drop)) for drop in combinations(edges, k)]


def enumerate_dr(edges, nodes, k):
    """All ways to reverse exactly k edges, keeping the graph acyclic."""
    out = []
    for flip in combinations(edges, k):
        new = [(b, a) if (a, b) in flip else (a, b) for a, b in edges]
        if len(set(new)) != len(new):      # a reversal collided with an existing edge
            continue
        if is_dag(new, nodes):
            out.append(sorted(new))
    return out


def candidate_new_edges(edges, nodes):
    """Every edge that could be added while keeping the graph a DAG."""
    have = set(edges) | {(b, a) for a, b in edges}
    cands = []
    for a in nodes:
        for b in nodes:
            if a == b or (a, b) in have:
                continue
            if is_dag(list(edges) + [(a, b)], nodes):
                cands.append((a, b))
    return cands


def enumerate_fe(edges, nodes, k):
    """All ways to add exactly k spurious edges, keeping the graph acyclic."""
    cands = candidate_new_edges(edges, nodes)
    out = []
    for add in combinations(cands, k):
        new = list(edges) + list(add)
        if is_dag(new, nodes):
            out.append(sorted(new))
    return out


def enumerate_scramble(edges, nodes):
    """Every DAG on the same nodes, with the same number of edges, that keeps
    NOT ONE of the true directed edges.

    The matched control between NAMES_ONLY and DR_k: the block carries the same
    names and the same number of arrows as ORACLE, but the arrows are random.
    A true edge may come back reversed, since a reversal is not the true edge;
    forbidding reversals as well is impossible on the 3-node families, whose
    complement graph has too few pairs left.

    Consequence, stated rather than hidden: on the complete 3-node families
    (confounding, mediation) the only such DAG is the full reversal, so there
    SCRAMBLE is DR_k3 and carries no randomness. No node is ever left isolated -
    the edge counts rule it out on every family - so the block names exactly the
    nodes ORACLE names.
    """
    true = set(edges)
    pool = [(a, b) for a in nodes for b in nodes if a != b and (a, b) not in true]
    return [sorted(pick) for pick in combinations(pool, len(edges))
            if is_dag(pick, nodes)]


ENUMERATORS = {"ED": enumerate_ed, "FE": enumerate_fe, "DR": enumerate_dr}


def feasibility(family: str, kmax: int = 3) -> dict:
    """How many distinct k-error graphs exist for each perturbation type."""
    edges = to_edges(FAMILY_STRUCTURE[family])
    nodes = sorted({n for e in edges for n in e})
    row = {
        "graph_id": family,
        "nodes": len(nodes),
        "edges": len(edges),
        "addable": len(candidate_new_edges(edges, nodes)),
    }
    for t, fn in ENUMERATORS.items():
        for k in range(1, kmax + 1):
            row[f"{t}_k{k}"] = len(fn(edges, nodes, k)) if k <= _cap(t, row) else 0
    return row


def _cap(t, row):
    return row["edges"] if t in ("ED", "DR") else row["addable"]


def max_k(family: str, t: str, kmax: int = 6) -> int:
    """Largest k with at least one valid perturbation."""
    edges = to_edges(FAMILY_STRUCTURE[family])
    nodes = sorted({n for e in edges for n in e})
    best = 0
    for k in range(1, kmax + 1):
        try:
            if ENUMERATORS[t](edges, nodes, k):
                best = k
        except ValueError:
            break
    return best
