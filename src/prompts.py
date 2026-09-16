"""Build the experimental prompts.

CLadder prompts already state the DAG in prose ("X has a direct effect on Y and
Z."). If that is left in, RAW is not a no-graph condition and a corrupted graph
merely contradicts the text. So every condition is built from a *stripped* body,
and the graph is re-attached only where the condition calls for it.
"""
from __future__ import annotations
import re

PREAMBLE = ("Imagine a self-contained, hypothetical world with only the following "
            "conditions, and without any unmentioned factors or causal relationships:")

# "Husband has a direct effect on wife and alarm clock."
EFFECT_SENT = re.compile(
    r"\s*[A-Z][^.]*?\bhas a direct effect on\b[^.]*?\.", re.IGNORECASE)

ANSWER_RULE = ("\n\nAnswer the question with a single word on the last line, "
               "in exactly this format:\nANSWER: yes\nor\nANSWER: no")


def strip_structure(prompt: str) -> tuple[str, list[str]]:
    """Remove the prose DAG sentences. Returns (stripped_prompt, removed)."""
    head, sep, body = prompt.partition(PREAMBLE)
    if not sep:                              # unexpected shape, leave untouched
        return prompt, []
    removed = EFFECT_SENT.findall(body)
    cleaned = EFFECT_SENT.sub("", body).strip()
    return f"{head}{PREAMBLE} {cleaned}", [s.strip() for s in removed]


PARENT_CHILDREN = re.compile(
    r"^(.*?)\s+has a direct effect on\s+(.*?)\.\s*$", re.IGNORECASE | re.DOTALL)


def parse_prose_graph(removed: list[str]) -> list[tuple[str, str]]:
    """Rebuild the DAG over the STORY's own variable names.

    The graph must be stated in the same vocabulary as the body. Emitting
    "X has a direct effect on V2" beside a body about husbands and wives gives
    the model two unlinked namespaces, and the structure block becomes noise -
    which silently turns the oracle condition into a no-op.

    The name graph is isomorphic to the symbol DAG by construction, so
    perturbations can be enumerated directly on it.

    Node names are lower-cased: a name appears capitalised when it starts a
    sentence and lower-case when it is an object, so "Wife" and "wife" would
    otherwise become two distinct nodes and the DAG would gain phantom vertices.
    """
    edges = []
    for s in removed:
        m = PARENT_CHILDREN.match(s.strip())
        if not m:
            continue
        parent = m.group(1).strip().lower()
        tail = m.group(2).strip()
        for child in re.split(r",\s*and\s+|\s+and\s+|,\s*", tail):
            child = child.strip().lower()
            if child and child != parent:
                edges.append((parent, child))
    return edges


def describe_graph(edges, var_names: dict[str, str] | None = None) -> str:
    """Render an edge list as the same prose CLadder itself uses."""
    def nm(v):
        return (var_names or {}).get(v, v)
    parents: dict[str, list[str]] = {}
    for a, b in edges:
        parents.setdefault(a, []).append(b)
    out = []
    for a in sorted(parents):
        kids = [nm(k) for k in parents[a]]
        tail = kids[0] if len(kids) == 1 else ", ".join(kids[:-1]) + " and " + kids[-1]
        out.append(f"{nm(a).capitalize()} has a direct effect on {tail}.")
    return " ".join(out)


def build(prompt: str, condition: str, edges=None, var_names=None) -> str:
    """condition: RAW | RAW_INSTR | ORACLE | PERTURB (edges) | PROSE.

    PROSE reproduces the untouched CLadder prompt, as a sanity anchor showing how
    much the stripping itself costs.

    RAW_INSTR exists because ORACLE adds two things at once - a block of graph
    content AND a line telling the model to reason causally - so Delta_struct =
    ORACLE - RAW has been charging the whole gap to the graph when part of it
    may be the instruction. RAW_INSTR carries the instruction with no graph, so
    the gap splits:

        RAW_INSTR - RAW    what telling the model to think causally is worth
        ORACLE - RAW_INSTR what the graph CONTENT is worth on top of that

    The wording cannot be byte-identical: ORACLE says "this causal structure"
    and there is no "this" without a block. The instruction is reworded to point
    at the world instead of at a block, which keeps its function and changes its
    surface. That residual wording difference is the ablation's known limit and
    is reported with the result rather than papered over.
    """
    if condition == "PROSE":
        return prompt + ANSWER_RULE

    stripped, _ = strip_structure(prompt)
    if condition == "RAW":
        return stripped + ANSWER_RULE
    if condition == "RAW_INSTR":
        return (stripped + "\n\nReason about the causal structure of this world "
                "when answering." + ANSWER_RULE)

    desc = describe_graph(edges, var_names)
    block = (f"\n\nThe causal structure of this world is:\n{desc}\n"
             f"Use this causal structure when reasoning.")
    return stripped + block + ANSWER_RULE


ANSWER_RE = re.compile(r"ANSWER:\s*(yes|no)", re.IGNORECASE)
FALLBACK_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse_answer(text: str) -> str | None:
    """Strict format first, last bare yes/no as fallback. None if unparseable."""
    m = ANSWER_RE.findall(text or "")
    if m:
        return m[-1].lower()
    m = FALLBACK_RE.findall(text or "")
    return m[-1].lower() if m else None
