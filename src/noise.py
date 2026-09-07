"""Structured noise injection on top of CLadder - a deterministic reimplementation
of the NoisyCausal noise taxonomy.

Why rebuild it: NoisyCausal released neither code nor data, and its Appendix D.1
shows all five construction steps were LLM prompts with no formal solver for the
gold answer. Here every injection is a deterministic text or graph operation, and
the gold answer stays the CLadder label, which was verified against exact
enumeration of the SCM (see scripts/verify_groundtruth.py).

Scoring rule, stated explicitly because NoisyCausal left it ambiguous:

  answer_preserving = True   the injected text does not change the queried
                             quantity, so the correct behaviour is to ignore it
                             and the gold label is unchanged.

  answer_preserving = False  the injected text changes the queried quantity. The
                             item then needs its own recomputed gold answer and
                             is NOT scored against the clean label. These are
                             emitted only by CI_ACTIVE, which is out of scope for
                             the break-even study and reserved for the selective
                             -use follow-up.

NoisyCausal scores confounder-injected items against the clean SCM while its own
Table 6 says the injection "introduces a backdoor path". That rewards ignoring
the injection and penalises correctly adjusting for it. Separating the two cases
above is what avoids inheriting that defect.
"""
from __future__ import annotations
import random
import re

# Distractor variables that plausibly co-occur with an outcome but sit outside
# any mechanism. Kept domain-neutral so they attach to any CLadder story.
DISTRACTORS = [
    ("living in a sunny area", 41), ("owning a bicycle", 37),
    ("subscribing to a newspaper", 29), ("having a library card", 44),
    ("preferring tea to coffee", 52), ("being left-handed", 12),
]

CONFOUNDERS = [
    "general baseline health", "household income", "overall motivation",
    "regional infrastructure quality", "prior experience",
]

PROB_SENT = re.compile(r"(?:For|The probability of)[^.]*?\bis\b\s*(\d+)%\.")


def _rng(seed, tag):
    return random.Random(f"{seed}:{tag}")


# CLadder puts the query in the final sentence. Noise appended after it reads as
# an afterthought and is unnaturally easy to discount, so injections go in just
# before the question, where a real distractor would sit.
QUESTION_TAIL = re.compile(r"(?:(?<=\.)|(?<=\?))\s*([^.?!]*\?)\s*$")


def _insert_before_question(prompt: str, sentence: str) -> str:
    m = QUESTION_TAIL.search(prompt.rstrip())
    if not m:
        return prompt.rstrip() + sentence
    head = prompt.rstrip()[:m.start()].rstrip()
    return f"{head}{sentence} {m.group(1).strip()}"


def inject_iv(prompt, seed=0, **_):
    """Irrelevant Variable: a spuriously correlated factor outside the mechanism."""
    r = _rng(seed, "IV")
    name, pct = r.choice(DISTRACTORS)
    add = (f" Separately, {pct}% of the population report {name}, "
           f"and this group is observed to have better outcomes overall.")
    return _insert_before_question(prompt, add), True


def inject_cs(prompt, seed=0, **_):
    """Causal Swap: prose hinting the reverse direction, contradicting the DAG."""
    add = (" It is also commonly observed that the outcome tends to precede and "
           "predict the treatment rather than follow from it.")
    return _insert_before_question(prompt, add), True


def inject_bip(prompt, seed=0, **_):
    """Belief-Inconsistent Perturbation: a stated misconception (NoisyCausal 5.7)."""
    add = (" Many people in this world believe the treatment makes the outcome "
           "less likely, contrary to the recorded statistics.")
    return _insert_before_question(prompt, add), True


def inject_ci_inert(prompt, seed=0, **_):
    """Confounder Injection, INERT: a common cause named but off the query path.

    Answer preserving by construction - the factor is stated to influence only
    variables that are not on any backdoor path for the queried effect.
    """
    r = _rng(seed, "CIi")
    c = r.choice(CONFOUNDERS)
    add = (f" An unmeasured factor, {c}, varies across this population, but it "
           f"is known to influence neither the treatment nor the outcome here.")
    return _insert_before_question(prompt, add), True


def inject_ci_active(prompt, seed=0, **_):
    """Confounder Injection, ACTIVE: a genuine backdoor path.

    NOT answer preserving. Emitted for the selective-use follow-up only; these
    items require a recomputed gold answer and must never be scored against the
    clean CLadder label.
    """
    r = _rng(seed, "CIa")
    c = r.choice(CONFOUNDERS)
    add = (f" An unmeasured factor, {c}, raises both the chance of receiving the "
           f"treatment and the chance of the outcome occurring.")
    return _insert_before_question(prompt, add), False


def inject_pm(prompt, seed=0, **_):
    """Partial Masking: delete one stated probability.

    Returns preserved=True only as a flag that the CLADDER label is unchanged;
    whether the item is still ANSWERABLE is a separate question that
    scripts/check_identifiability.py must decide. Scoring a masked item against
    the clean label without that check is the defect found in NoisyCausal PM.
    """
    r = _rng(seed, "PM")
    hits = list(PROB_SENT.finditer(prompt))
    if not hits:
        return prompt, True
    h = r.choice(hits)
    return (prompt[:h.start()] + prompt[h.end():]).replace("  ", " "), True


def inject_vp(prompt, seed=0, **_):
    """Value Perturbation: alter one displayed probability by +-15 points.

    The clean label is kept, i.e. the benchmark asks the model to reason from the
    ORIGINAL structural model. That is NoisyCausal's stated rule; it is recorded
    here explicitly so the ambiguity is visible rather than buried.
    """
    r = _rng(seed, "VP")
    hits = list(PROB_SENT.finditer(prompt))
    if not hits:
        return prompt, True
    h = r.choice(hits)
    old = int(h.group(1))
    new = max(1, min(99, old + r.choice([-15, 15])))
    span = h.group(0).replace(f"is {old}%", f"is {new}%")
    return prompt[:h.start()] + span + prompt[h.end():], True


INJECTORS = {
    "NONE": lambda p, seed=0, **k: (p, True),
    "IV": inject_iv,
    "CS": inject_cs,
    "BIP": inject_bip,
    "CI_INERT": inject_ci_inert,
    "CI_ACTIVE": inject_ci_active,
    "PM": inject_pm,
    "VP": inject_vp,
}

ANSWER_PRESERVING = [k for k in INJECTORS if k != "CI_ACTIVE"]


def apply(prompt: str, kind: str, seed: int = 0):
    """Returns (noisy_prompt, answer_preserving)."""
    return INJECTORS[kind](prompt, seed=seed)
