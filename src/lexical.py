"""Relabel a CLadder item's variables in place, keeping everything else fixed.

The pilot's first lexical contrast compared CLadder's commonsense split against
its noncommonsense split. Those are different item pools - zero shared ids, ten
stories against thirty-seven - so the contrast is between-items and McNemar does
not apply to it. Caliper (arXiv:2606.04915) does the same manipulation within
item, which is why its design is the stronger one.

This module closes that gap. Every CLadder model ships a `variable_mapping`
giving the surface phrase for each variable and each of its two values, and
`background` maps to exactly one such mapping (209 backgrounds, none ambiguous).
So an item's lexicon can be swapped without touching its graph, its
probabilities, its query, or its gold label - which makes the lexical
manipulation paired at the item level.

Three lexicons:

  KEEP    the item as CLadder wrote it
  PSEUDO  CLadder's own pseudoword vocabulary (rixq, zuph, xevu)
  SYMBOL  single letters (A, B, C)

SYMBOL exists to separate two explanations that PSEUDO alone confounds. A
pseudoword item asks the model to bind four multi-syllable nonsense strings that
look alike (zuph / xevu / uvzi / wibl) and hold them through a multi-step
calculation. If accuracy recovers under single letters, what pseudowords destroy
is symbol binding, not causal reasoning. If it does not recover, the loss is
about the missing world knowledge itself.
"""
from __future__ import annotations
import json
import random
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# CLadder's own pseudoword list, taken from the nonsense stories rather than
# invented here, so the PSEUDO condition matches the shipped split's register.
PSEUDOWORDS = [
    "cwoi", "glimx", "gwet", "gyzp", "hwax", "jyka", "kraz", "kwox", "kwoz",
    "lirg", "muvq", "muvy", "pexu", "qwiu", "rixq", "rukz", "swoq", "swoy",
    "tijv", "tijw", "uvzi", "vubr", "wibl", "xevo", "xevu", "xyfo", "yomx",
    "yupt", "zory", "zuph", "zupj",
]

SYMBOLS = ["A", "B", "C", "D", "E", "F"]

# Xname/X1/X0, V2name/V21/V20, ...
VAR_RE = re.compile(r"^(X|Y|V\d+)(name|0|1)$")


@lru_cache(maxsize=1)
def _background_index() -> dict[str, dict]:
    """background -> variable_mapping. One entry per distinct background."""
    meta = json.loads((ROOT / "data" / "cladder-meta.json").read_text(encoding="utf-8"))
    idx = {}
    for m in meta:
        idx.setdefault(m["background"].strip(), m["variable_mapping"])
    return idx


def find_mapping(prompt: str) -> dict | None:
    """Recover the item's variable_mapping by matching its background.

    Matched on the background being a prefix of the prompt rather than on any id
    column, because the v1.5 CSVs carry no model id. Longest match wins: some
    backgrounds are prefixes of longer ones.
    """
    idx = _background_index()
    best = None
    for bg, vm in idx.items():
        if prompt.startswith(bg) and (best is None or len(bg) > len(best[0])):
            best = (bg, vm)
    return best[1] if best else None


def _symbols_of(vm: dict) -> list[str]:
    """Variable symbols present in this mapping, in a stable order."""
    seen = []
    for k in vm:
        m = VAR_RE.match(k)
        if m and m.group(1) not in seen:
            seen.append(m.group(1))
    order = {"X": 0, "V1": 1, "V2": 2, "V3": 3, "Y": 9}
    return sorted(seen, key=lambda s: (order.get(s, 5), s))


def _derangement(n: int, rng: random.Random) -> list[int]:
    """A permutation with no fixed point, so every variable changes role."""
    if n < 2:
        return list(range(n))
    idx = list(range(n))
    for _ in range(200):
        rng.shuffle(idx)
        if all(i != j for i, j in enumerate(idx)):
            return idx
    return idx[1:] + idx[:1]          # rotation is a derangement for n >= 2


def build_lexicon(vm: dict, lexicon: str, seed: str = "") -> dict[str, str]:
    """New surface phrases keyed exactly like variable_mapping.

    PSEUDO and SYMBOL follow CLadder's own nonsense convention: the variable,
    its positive value, and its negative value read as `w`, `w`, and `not w`.
    That keeps the negation cue the model needs while removing the world
    knowledge, and it is the register the shipped pseudoword split uses.

    PERMUTE is the control the other two cannot provide. Replacing "medicine"
    with "zuph" removes world knowledge, but it also shortens the prompt,
    changes how it tokenises, and asks the model to bind an unfamiliar string.
    PERMUTE instead reassigns the item's OWN variable phrases to different
    positions in its own graph: identical vocabulary, identical length,
    identical tokenisation, and the only thing destroyed is whether the causal
    direction is plausible. A derangement is used so no variable keeps its own
    name - otherwise part of the item would still carry a usable prior.
    """
    syms = _symbols_of(vm)

    if lexicon == "PERMUTE":
        order = _derangement(len(syms), random.Random(f"permute:{seed}"))
        return {f"{s}{suf}": vm[f"{syms[j]}{suf}"]
                for s, j in zip(syms, order)
                for suf in ("name", "1", "0")
                if f"{syms[j]}{suf}" in vm}

    if lexicon == "PSEUDO":
        pool = list(PSEUDOWORDS)
        random.Random(f"pseudo:{seed}").shuffle(pool)
        words = pool[:len(syms)]
    elif lexicon == "SYMBOL":
        words = SYMBOLS[:len(syms)]
    else:
        raise ValueError(f"unknown lexicon {lexicon!r}")

    out = {}
    for s, w in zip(syms, words):
        out[f"{s}name"] = w
        out[f"{s}1"] = w
        out[f"{s}0"] = f"not {w}"
    return out


def relabel(prompt: str, vm: dict, new: dict[str, str]) -> tuple[str, bool]:
    """Swap every surface phrase for its replacement, all in one pass.

    One pass, not one substitution per phrase. PERMUTE draws its replacements
    from the item's own vocabulary, so a sequential loop would cascade: rewrite
    "husband" to "wife", then the next step rewrites that fresh "wife" again.
    A single alternation pass consumes each span exactly once.

    Alternatives are ordered longest first, because a variable's name is
    usually a substring of its own value phrases ("husband" inside "alarm set
    by husband"), and Python's regex alternation takes the first branch that
    matches rather than the longest.

    Returns (relabelled_prompt, clean); `clean` is False when the swap did not
    fully take, and such an item is dropped rather than scored with half its
    vocabulary intact.
    """
    pairs = sorted(((vm[k], new[k]) for k in vm if k in new),
                   key=lambda p: -len(p[0]))
    if not pairs:
        return prompt, False
    lookup = {old.lower(): rep for old, rep in pairs}
    rx = re.compile("|".join(re.escape(old) for old, _ in pairs), re.IGNORECASE)
    hits = set()

    def sub(m):
        hits.add(m.group(0).lower())
        return lookup[m.group(0).lower()]

    out = rx.sub(sub, prompt)

    # Sentence-initial capitals: the phrases went in lower-case. The colon case
    # matters because the first structure sentence follows the preamble's colon,
    # which is exactly where CLadder's own nonsense split capitalises.
    out = re.sub(r"(?<=[.!?:]\s)([a-z])", lambda m: m.group(1).upper(), out)
    out = re.sub(r"^([a-z])", lambda m: m.group(1).upper(), out)

    # PERMUTE reuses the item's own vocabulary, so "no original phrase survives"
    # is the wrong test for it - every phrase is meant to survive, in a different
    # role. What has to hold for every lexicon is that each phrase the mapping
    # defines and that occurs in the prompt was actually consumed; for the
    # lexicons that bring in new vocabulary, nothing old may remain on top.
    # A phrase can legitimately go unhit on its own: "husband" never matches
    # alone when every occurrence sits inside "alarm set by husband", which the
    # longer alternative consumes first. So absence from `hits` proves nothing,
    # and the real test is what is left in the output.
    reuses_vocab = {v.lower() for v in new.values()} & {o.lower() for o, _ in pairs}
    if reuses_vocab:
        return out, out != prompt and bool(hits)
    survived = any(re.search(re.escape(old), out, re.IGNORECASE) for old, _ in pairs)
    return out, not survived


def relabel_item(prompt: str, lexicon: str, seed: str = "") -> tuple[str, bool]:
    """KEEP returns the prompt untouched; anything else swaps the lexicon."""
    if lexicon == "KEEP":
        return prompt, True
    vm = find_mapping(prompt)
    if vm is None:
        return prompt, False
    return relabel(prompt, vm, build_lexicon(vm, lexicon, seed))
