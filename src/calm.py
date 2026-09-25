"""CaLM's ATE items (Chen et al. 2024, arXiv:2405.00622) as a second benchmark.

Why these items. The design needs a question whose causal graph is stated apart
from the answer, whose variable names can be swapped inside the item, and whose
gold answer can be recomputed. CaLM's average-treatment-effect task, natural
mode, English, has all three, and it is NOT built from CLadder: its graphs are
random DAGs on 3 to 5 nodes, where CLadder has ten fixed families. (Several
other CaLM tasks - backadj, det-counterfactual, collider-bias, correlation,
exp-away - carry CLadder's own query-type names and are left out.)

The items are written in CLadder's own frame - the same "Imagine a
self-contained, hypothetical world..." preamble and one "X has a direct effect
on Y." sentence per edge - so src/prompts.py strips and rebuilds the graph for
them exactly as it does for CLadder, and RAW, ORACLE and DR_k1 mean the same
thing on both benchmarks.

Three kinds of item, which CaLM labels only in its 100-item Lite release:
  REAL    real names and a story that makes sense (exercise -> fitness)
  RANDOM  real names drawn from a fixed pool onto a random graph
  FAKE    four-letter pseudowords
mode() recovers the label for every item: RANDOM names come from a pool of
about thirty, so each recurs in dozens of items, while every REAL story has a
name of its own. It agrees with all 100 Lite labels (scripts/analyze_calm.py
checks this on every run).

The full release carries no answers. gold() recomputes them from the graph and
the probabilities in the item, the way CaLM's own worked solutions do: no
directed path from treatment to outcome means No; otherwise the backdoor-
adjusted difference in the direction the question asks, Yes when positive. It
agrees with all 100 Lite answers (same check).
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "calm"
FULL = "calm_dataset__intervention__average_treatment_effect__ATE-B_ATE-natural_EN.json"
LITE = "calm_lite_dataset__intervention__average_treatment_effect__ATE-B_ATE-natural_EN.json"
RANDOM_MIN_FREQ = 40      # REAL stories: a name seen <= 24 times; RANDOM: every name >= 56


def load(name: str = FULL) -> list[dict]:
    raw = (DATA / name).read_text(encoding="utf-8").strip()
    return json.loads(raw) if raw.startswith("[") else [
        json.loads(line) for line in raw.splitlines() if line.strip()]


def node_names(it) -> dict[str, str]:
    """Letter -> the item's name for it."""
    m = it["Background"]["real_world_meaning"]
    pairs = re.findall(r"([A-Z]) represents (.+?)(?=, [A-Z] represents| and [A-Z] represents|\.\s*$)", m)
    return {k: v.strip() for k, v in pairs}


def sym_edges(it) -> list[tuple[str, str]]:
    return re.findall(r"([A-Z])->([A-Z])", it["Background"]["graph"])


def story(it) -> str:
    """The unit items share: one set of names on one graph. 1,600 items, 279 stories."""
    nm = node_names(it)
    return " | ".join(sorted(nm.values())) + " || " + ",".join(
        f"{a}{b}" for a, b in sorted(sym_edges(it)))


def name_freq(items) -> Counter:
    return Counter(n for it in items for n in set(node_names(it).values()))


def mode(it, freq: Counter) -> str:
    ns = list(node_names(it).values())
    if all(re.fullmatch(r"[a-z]{4}", n) for n in ns):
        return "FAKE"
    return "RANDOM" if min(freq[n] for n in ns) >= RANDOM_MIN_FREQ else "REAL"


def prompt(it) -> str:
    """given_info, the probabilities, the instruction and the question, in that order."""
    parts = [it["given_info"], it["Background"]["data_info"], it["Instruction"], it["Question"]]
    return re.sub(r"\s+", " ", " ".join(p.strip() for p in parts if p and p.strip())).strip()


# ---------------------------------------------------------------- gold

def _path(edges, a, b) -> bool:
    kids: dict[str, list[str]] = {}
    for u, v in edges:
        kids.setdefault(u, []).append(v)
    seen, todo = set(), [a]
    while todo:
        x = todo.pop()
        for y in kids.get(x, []):
            if y == b:
                return True
            if y not in seen:
                seen.add(y)
                todo.append(y)
    return False


TERM = re.compile(r"P\(([^)|]+)(?:\|([^)]+))?\)=([0-9.]+)")


def _terms(math: str):
    """[(outcome {var: val}, given {var: val}, p)] from 'P(C=0|A=1)=0.72; P(A=0)=0.32'."""
    out = []
    for lhs, rhs, p in TERM.findall(math):
        f = lambda s: {k.strip(): int(v) for k, v in (t.split("=") for t in s.split(","))}
        out.append((f(lhs), f(rhs) if rhs else {}, float(p)))
    return out


def _value_words(it) -> dict[tuple[str, str], int]:
    """(letter, value phrase) -> 0/1, by pairing each English probability sentence
    with its formula: the i-th sentence of data_info states the i-th term of
    data_info_math. Value phrases can run to several words ("abundance of",
    "not good"), so each is read up to the next comma, " and " or " is "."""
    nm = node_names(it)
    sents = [s for s in re.split(r"(?<=\d)\.\s*", it["Background"]["data_info"]) if s.strip()]
    terms = _terms(it["Background"]["data_info_math"])
    out = {}
    for s, (lhs, rhs, _) in zip(sents, terms):
        vals = {**lhs, **rhs}
        for k, name in nm.items():
            if k not in vals:
                continue
            for m in re.finditer(rf"(?<![\w']){re.escape(name)} being (.+?)(?=,| and | is |$)",
                                 s, re.IGNORECASE):
                out[(k, m.group(1).strip().lower())] = vals[k]
    return out


def _treatment_outcome(it) -> tuple[str, str] | None:
    """Matched against the item's own names: a name can contain " on " ("amount of
    trash left on the beach"), so splitting the sentence would cut it wrongly."""
    nm = node_names(it)
    ins = re.sub(r"\s+", " ", it["Instruction"].strip().lower())
    hits = [(a, b) for a, na in nm.items() for b, nb in nm.items()
            if a != b and ins.endswith(f"of {na.lower()} on {nb.lower()}.")]
    return hits[0] if len(hits) == 1 else None


def gold(it) -> str | None:
    """'yes' / 'no', or None when the item cannot be scored."""
    nm = node_names(it)
    to = _treatment_outcome(it)
    if to is None:
        return None
    t, o = to
    edges = sym_edges(it)
    if not _path(edges, t, o):
        return "no"
    q = re.search(rf"If {re.escape(nm[t])} is changed to be (.+?), will {re.escape(nm[o])} "
                  rf"be more likely to be (.+?)\?", it["Question"], re.IGNORECASE)
    words = _value_words(it)
    if not q or not words:
        return None
    x1, y = words.get((t, q.group(1).strip().lower())), words.get((o, q.group(2).strip().lower()))
    if x1 is None or y is None:
        return None
    terms = _terms(it["Background"]["data_info_math"])
    cond = {}
    marg = {}
    for lhs, rhs, p in terms:
        if rhs and o in lhs:
            # P(O=v | T=x, Z=z): store as P(O=y | ...) whatever v was given
            key = tuple(sorted(rhs.items()))
            cond[key] = p if lhs[o] == y else 1 - p
        elif not rhs:
            key = tuple(sorted(lhs.items()))
            marg[key] = p
    zs = sorted({k for key in cond for k, _ in key if k != t})

    def pz(zv: dict) -> float | None:
        if not zs:
            return 1.0
        key = tuple(sorted(zv.items()))
        if key in marg:
            return marg[key]
        if len(zs) == 1:
            (z,) = zs
            other = ((z, 1 - zv[z]),)
            return 1 - marg[other] if other in marg else None
        return None

    def do(x):
        tot = 0.0
        for bits in range(2 ** len(zs)):
            zv = {z: (bits >> i) & 1 for i, z in enumerate(zs)}
            key = tuple(sorted({**zv, t: x}.items()))
            w = pz(zv)
            if key not in cond or w is None:
                return None
            tot += cond[key] * w
        return tot

    a, b = do(x1), do(1 - x1)
    if a is None or b is None:
        return None
    return "yes" if a - b > 0 else "no"


# ---------------------------------------------------------------- lexicon

def relabel(it, lexicon: str, seed: str) -> tuple[str, bool]:
    """KEEP leaves the prompt alone; PSEUDO swaps every name for a pseudoword from
    src/lexical.PSEUDOWORDS, the pool CLadder's own pseudoword split uses.
    Longest names first, so a name inside another is not swapped half-way.
    Returns (prompt, clean): clean is False if any original name survives."""
    p = prompt(it)
    if lexicon == "KEEP":
        return p, True
    if lexicon != "PSEUDO":
        raise ValueError(lexicon)
    from lexical import PSEUDOWORDS
    pool = list(PSEUDOWORDS)
    random.Random(f"pseudo:{seed}").shuffle(pool)
    names = sorted(set(node_names(it).values()), key=len, reverse=True)
    new = dict(zip(names, pool))
    for n in names:
        def sub(m, w=new[n]):
            return w.capitalize() if m.group(0)[0].isupper() else w
        p = re.sub(rf"(?<![\w']){re.escape(n)}(?![\w'])", sub, p, flags=re.IGNORECASE)
    clean = not any(re.search(rf"(?<![\w']){re.escape(n)}(?![\w'])", p, re.IGNORECASE)
                    for n in names)
    return p, clean
