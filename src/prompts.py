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


# "Gender is unobserved." strip_structure() leaves this sentence in every item
# of the IV, arrowhead and frontdoor families (scripts/measure_raw_leak.py): it
# names no edge, but it tells the model a latent confounder exists, which is
# structure. RAW_CLEAN removes it. It is a separate condition rather than a
# change to strip_structure(), because every condition is built on the stripped
# body: widening the strip itself would change ~19,500 cached prompts across all
# arms, where RAW_CLEAN adds one prompt per affected item.
#
# The other survivor, "X causes Y" in det-counterfactual items, is NOT removed
# and cannot be: those sentences are the item's structural equations, and a
# det-counterfactual prompt carries no probabilities, so without them the
# question has no answer. That leak is handled by excluding the query type.
LATENT_SENT = re.compile(r"\s*[^.:?]*\bis unobserved\.", re.IGNORECASE)

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


def list_nodes(edges, var_names: dict[str, str] | None = None) -> str:
    """The variable names, SORTED ALPHABETICALLY.

    Order is a leakage channel. Listing them in topological order would tell the
    model which node is a root and which is a sink - exactly the structure this
    condition exists to withhold. Alphabetical order is independent of the graph.
    """
    def nm(v):
        return (var_names or {}).get(v, v)
    names = sorted({nm(n) for e in edges for n in e})
    if not names:
        return ""
    head = names[0].capitalize()
    return ", ".join([head] + names[1:]) + "."


def build(prompt: str, condition: str, edges=None, var_names=None) -> str:
    """condition: RAW | RAW_CLEAN | RAW_INSTR | NAMES_ONLY | ORACLE | PERTURB (edges) | PROSE.

    NAMES_ONLY is the matched control for ORACLE. ORACLE adds FOUR things at
    once: a block of text in a fixed position, the variable names restated, THE
    EDGES, and an instruction telling the model to use the structure. NAMES_ONLY
    keeps the first three and removes only the edges, so ORACLE minus NAMES_ONLY
    isolates what the EDGES are worth.

    If ORACLE does not beat NAMES_ONLY, then what helps the model is not the
    CONTENT of the structure but merely having a set of symbols to anchor on -
    and the study becomes one about symbol re-anchoring. Both outcomes are
    publishable.

    Two surface differences remain, stated rather than hidden:

      1. The opening sentence has to change: "The causal structure of this world
         is" is not true of a block that contains no structure. It becomes "The
         variables of this world are". This is the same class of limitation
         RAW_INSTR already carries.
      2. The closing instruction is KEPT WORD FOR WORD. Changing it would add a
         second difference and destroy the isolation. The consequence is that
         "Use this causal structure when reasoning" follows a block with no
         structure in it - which is precisely the condition being measured: does
         the model need the CONTENT, or only a scaffold to hold on to?

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
    if condition == "RAW_CLEAN":
        # Identical to RAW wherever there is no latent sentence, so those items
        # hit RAW's cache entry and cost nothing.
        return LATENT_SENT.sub("", stripped) + ANSWER_RULE
    if condition == "RAW_INSTR":
        return (stripped + "\n\nReason about the causal structure of this world "
                "when answering." + ANSWER_RULE)

    if condition == "NAMES_ONLY":
        body = list_nodes(edges, var_names)
        head = "The variables of this world are:"
    else:
        body = describe_graph(edges, var_names)
        head = "The causal structure of this world is:"
    block = f"\n\n{head}\n{body}\nUse this causal structure when reasoning."
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
