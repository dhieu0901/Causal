"""Is the label of every question we sampled right, and decidable from what it states?

    python scripts/verify_sample_labels.py

verify_labels.py counts, in CLadder's JSON release, the questions whose
published value and recomputed value fall on opposite sides of the decision
threshold (91, of which 83 carry the label of the wrong value), and
verify_counterfactual.py checks the JSON release's counterfactual and collider
questions. The questions this project samples come from full_v1.5_default.csv,
a different set of questions on the same 7,064 SCMs. So every question of every
sample is checked here directly, whatever its query type:

  ate, ett, nde, nie, marginal   matched to its SCM through the probabilities its
                  reasoning field states (analyze_answer_change.Matcher); the
                  published and the recomputed value of its quantity are compared
                  with the decision threshold, read as CLadder reads it. For the
                  effects (threshold 0) a value that rounds to 0.00 means No
                  ("0.00 = 0" in CLadder's explanations); marginal compares the
                  unrounded value with 0.5. Where the two values fall on opposite
                  sides the label is decided by a published error, and the
                  question's own wording (its polarity) tells which of the two
                  values the label follows; elsewhere the label must follow the
                  recomputed value. An effect label that holds only by the
                  rounding (the unrounded value lies on the Yes side), and a
                  marginal label that P(X=1), P(Y=1 | X=0) and P(Y=1 | X=1) as
                  stated to two decimals put on the other side of 0.5, are flagged
                  as not decidable from the stated numbers
  exp_away        matched to its SCM the same way; the label must follow the sign
                  of P(Y=1 | X=1, V3=1) - P(Y=1 | V3=1). The question states only
                  P(X) and P(Y | X, V3), which do not determine P(Y=1 | V3=1);
                  CLadder's own explanation uses P(X) in place of P(X | V3=1). A
                  question whose label differs from what that formula gives is
                  flagged as not decidable from the stated numbers
  correlation     P(Y=1 | X=1) - P(Y=1 | X=0) from the SCM that reproduces the
                  stated P(X=1) and joint probabilities; the label must follow it.
                  CLadder reads a true value that rounds to 0.00 as no difference;
                  a question whose label holds only by that rounding, and one whose
                  two-decimal stated numbers give the other sign, is flagged as not
                  decidable from the stated numbers
  collider_bias   the causal effect of X on Y in X -> V3 <- Y is zero, so the
                  answer is Yes exactly when the question asks whether X does
                  NOT affect Y
  det-counterfactual  Pearl's three steps on the boolean mechanisms the reasoning
                  field prints (verify_counterfactual.counterfactual, with its
                  fix for the misprinted V3 equation of diamondcut)
  backadj         each method's adjustment set, read from the question's names,
                  against every set that satisfies the backdoor criterion on the
                  family's graph: "is Method 1 more correct than Method 2" is Yes
                  when Method 1's set is valid and Method 2's is not, and, when both
                  are valid (the fork family), when Method 1's set is the smaller.
                  This rule reproduces all 1,580 backadj labels of the CSV release

A question that cannot be checked stops the script: the check covers every
question or it reports nothing. The same checks are then run on every question
of the CSV release, to give the release-wide counts beside the sampled ones.

Writes results/cladder/sample_labels.csv (one row per sample file tag) and
results/cladder/release_labels.csv (one row per query type of the CSV release).
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from analyze_answer_change import VG, Matcher
from classify_perturbations import backdoor_sets
from lexical import find_mapping
from perturb import FAMILY_STRUCTURE, to_edges

RAW = ROOT / "results" / "cladder" / "raw"
KEY = {"ate": "ATE(Y | X)", "ett": "ETT(Y | X)", "nde": "NDE(Y | X)", "nie": "NIE(Y | X)",
       "marginal": "P(Y=1)"}
TOL = 1e-9
ROUND = 0.0051                      # stated probabilities carry two decimals
EQ = re.compile(r"^(X|Y|V\d)\s*=\s*([^=]+)$")
TERM = re.compile(r"P\(([^)]*)\)\s*=\s*(-?[0-9.]+)")
METHODS = re.compile(r"Method 1: (.*?)\. Method 2: (.*?)\.")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / "scripts" / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def question(prompt: str) -> str:
    """The question sentence (the last one ending in '?')."""
    return re.split(r"(?<=[.:])\s+", prompt.split("?")[-2])[-1].lower()


def polarity(r) -> int:
    """1 when the question asks whether the quantity is above its threshold.
    ett: "would it be more likely to see Y if X had been 0" asks whether the
    effect of treatment on the treated is negative."""
    q, qt = question(r.prompt), r.query_type
    # "will increased supply decrease the chance of ...": read the verb, not the names
    words = {"ate": ("increase the chance", "decrease the chance"), "nde": ("positively", "negatively"),
             "nie": ("positively", "negatively"), "marginal": ("more likely than", "less likely than"),
             "ett": ("less likely", "more likely")}[qt]
    up, down = words[0] in q, words[1] in q
    if up != down:
        return int(up)
    raise SystemExit(f"id {r.id} ({qt}): polarity not read: {q}")


def rounded_label(v, thr, pol, vl):
    """CLadder's label for a value: an effect that rounds to 0.00 means No."""
    return "no" if thr == 0 and abs(v) < 0.005 else vl.label_for(v, thr, pol)


def marginal_stated(r, pol, vl):
    """The label P(Y=1) gets from the three probabilities the question states."""
    t = {k.replace(" ", ""): float(v) for k, v in TERM.findall(str(r.reasoning))}
    if set(t) != {"X=1", "Y=1|X=0", "Y=1|X=1"}:
        raise SystemExit(f"id {r.id} (marginal): stated terms not read: {t}")
    v = t["X=1"] * t["Y=1|X=1"] + (1 - t["X=1"]) * t["Y=1|X=0"]
    return None if abs(v - 0.5) < TOL else vl.label_for(v, 0.5, pol)


def check_error(r, M, vl):
    cands = M.match(r)
    if not cands:
        raise SystemExit(f"id {r.id} ({r.query_type}): no SCM reproduces its stated quantities")
    thr, pol = vl.threshold(r.query_type), polarity(r)
    out = set()
    for m in cands:
        pub, tru = float(m["groundtruth"][KEY[r.query_type]]), float(vl.true_value(VG, m, r.query_type))
        lp, lt = rounded_label(pub, thr, pol, vl), rounded_label(tru, thr, pol, vl)
        if lp == lt:
            if r.label != lt:
                out.add("label differs from its SCM")
            elif lt != vl.label_for(tru, thr, pol):
                out.add("right, not decidable from the stated numbers")
            elif r.query_type == "marginal" and marginal_stated(r, pol, vl) != lt:
                out.add("right, not decidable from the stated numbers")
            elif lt == vl.label_for(tru, thr, pol):
                out.add("right")
            else:
                out.add("right, not decidable from the stated numbers")
        else:
            out.add("decided by a published error, label follows it" if r.label == lp
                    else "decided by a published error, label follows the recomputed value")
    if len(out) != 1:
        raise SystemExit(f"id {r.id} ({r.query_type}): the matching SCMs disagree: {out}")
    return out.pop()


def check_exp_away(r, M):
    cands = M.match(r)
    if not cands:
        raise SystemExit(f"id {r.id} (exp_away): no SCM reproduces its stated quantities")
    up = "increase" in question(r.prompt)
    want = set()
    for m in cands:
        s = M.get(m)
        v = s.prob({"Y": 1}, given={"X": 1, "V3": 1}) - s.prob({"Y": 1}, given={"V3": 1})
        want.add("yes" if (v > 0) == up else "no")
    if len(want) != 1:
        raise SystemExit(f"id {r.id} (exp_away): the matching SCMs disagree")
    if want.pop() != r.label:
        return "label differs from its SCM"
    t = {k.replace(" ", ""): float(v) for k, v in TERM.findall(str(r.reasoning))}
    px, a, b = t["X=1"], t["Y=1|X=1,V3=1"], t["Y=1|X=0,V3=1"]
    stated = a - (px * a + (1 - px) * b)
    return "right" if ("yes" if (stated > 0) == up else "no") == r.label else \
        "right, not decidable from the stated numbers"


def check_correlation(r, M):
    t = {}
    for k, v in TERM.findall(str(r.reasoning)):
        k = k.replace(" ", "")
        if k == "X=1=1":
            t["px"] = float(v)
        elif k == "Y=1,X=1=1":
            t["p11"] = float(v)
        elif k == "Y=1,X=0=1":
            t["p01"] = float(v)
    if set(t) != {"px", "p11", "p01"}:
        raise SystemExit(f"id {r.id} (correlation): stated terms not read: {t}")
    q = question(r.prompt)
    larger = "larger" in q
    if not larger and "smaller" not in q:
        raise SystemExit(f"id {r.id} (correlation): question direction not read")
    # a stated P(X=1) of 0.00 or 1.00 leaves the conditional undefined
    stated = (t["p11"] / t["px"] - t["p01"] / (1 - t["px"])) if 0 < t["px"] < 1 else None
    best, want = None, set()
    for m in M.by.get((r.story_id, r.graph_id), []):
        s = M.get(m)
        px = s.prob({"X": 1})
        p11 = s.prob({"X": 1, "Y": 1})
        p01 = s.prob({"X": 0, "Y": 1})
        err = max(abs(px - t["px"]), abs(p11 - t["p11"]), abs(p01 - t["p01"]))
        if err <= ROUND:
            v = s.prob({"Y": 1}, given={"X": 1}) - s.prob({"Y": 1}, given={"X": 0})
            strict = "yes" if (v > 0) == larger else "no"
            want.add("tie" if abs(v) < 0.005 and strict == "yes" else strict)
        best = err if best is None else min(best, err)
    if not want:
        raise SystemExit(f"id {r.id} (correlation): no SCM reproduces its stated numbers "
                         f"(nearest off by {best:.4f})")
    if len(want) != 1:
        raise SystemExit(f"id {r.id} (correlation): the matching SCMs disagree")
    w = want.pop()
    if w == "tie":
        return "right, not decidable from the stated numbers" if r.label == "no" else \
            "label differs from its SCM"
    if w != r.label:
        return "label differs from its SCM"
    if stated is None:
        return "right, not decidable from the stated numbers"
    return "right" if ("yes" if (stated > 0) == larger else "no") == r.label else \
        "right, not decidable from the stated numbers"


def check_collider(r):
    q = question(r.prompt)
    if "affect" not in q:
        raise SystemExit(f"id {r.id} (collider_bias): question not read: {q}")
    return "right" if ("yes" if "not affect" in q else "no") == r.label else \
        "label differs from a zero effect"


def check_counterfactual(r, vc):
    fm = vc.FORM.match(str(r.formal_form).strip())
    if not fm:
        raise SystemExit(f"id {r.id} (det-counterfactual): formal form not read")
    x_val, target = int(fm.group(1)), int(fm.group(2))
    evidence = {}
    for part in filter(None, [p.strip() for p in fm.group(3).split(",")]):
        k, v = part.split("=")
        evidence[k.strip()] = int(v)
    lines = [ln.strip() for ln in str(r.reasoning).splitlines()]
    txt = "\n".join(ln for ln in lines if EQ.match(ln) and not EQ.match(ln).group(2).strip().isdigit())
    if r.graph_id == "diamondcut":
        txt = vc.DIAMONDCUT_BUG.sub(lambda s: "V3 = %sV1" % (s.group(1) or ""), txt)
    eqs = vc.parse_eqs(txt)
    if not eqs:
        raise SystemExit(f"id {r.id} (det-counterfactual): no mechanisms read")
    got = vc.counterfactual(eqs, evidence, x_val, target)
    if got is None:
        raise SystemExit(f"id {r.id} (det-counterfactual): evidence not decisive")
    return "right" if ("yes" if got == 1 else "no") == r.label else "label differs from its mechanisms"


def check_backadj(r):
    vm = find_mapping(r.prompt)
    if vm is None:
        raise SystemExit(f"id {r.id} (backadj): no variable mapping")
    names = {vm[k].lower(): k[:-4] for k in vm if k.endswith("name")}
    mm = METHODS.search(r.prompt)
    if not mm:
        raise SystemExit(f"id {r.id} (backadj): methods not read")

    def adj_set(text):
        text = text.lower()
        if "case by case according to" not in text:
            if "in general" not in text:
                raise SystemExit(f"id {r.id} (backadj): method not read: {text}")
            return frozenset()
        tail = text.split("case by case according to", 1)[1].strip()
        out = set()
        for nm in sorted(names, key=len, reverse=True):
            if re.search(r"(?<![a-z])" + re.escape(nm) + r"(?![a-z])", tail):
                out.add(names[nm])
                tail = tail.replace(nm, " ")
        if not out or re.search(r"[a-z]{3,}", tail.replace("and", "")):
            raise SystemExit(f"id {r.id} (backadj): adjustment set not read: {text}")
        return frozenset(out)

    edges = to_edges(FAMILY_STRUCTURE[r.graph_id])
    nodes = sorted({n for e in edges for n in e})
    valid = backdoor_sets(edges, nodes, "X", "Y")
    s1, s2 = adj_set(mm.group(1)), adj_set(mm.group(2))
    v1, v2 = s1 in valid, s2 in valid
    if v1 and v2:
        if len(s1) == len(s2):
            raise SystemExit(f"id {r.id} (backadj): two valid sets of one size")
        want = "yes" if len(s1) < len(s2) else "no"
    else:
        want = "yes" if v1 else "no"
    return "right" if want == r.label else "label differs from the backdoor criterion"


def check(r, M, vl, vc):
    qt = r.query_type
    if qt in KEY:
        return check_error(r, M, vl)
    if qt == "exp_away":
        return check_exp_away(r, M)
    if qt == "correlation":
        return check_correlation(r, M)
    if qt == "collider_bias":
        return check_collider(r)
    if qt == "det-counterfactual":
        return check_counterfactual(r, vc)
    if qt == "backadj":
        return check_backadj(r)
    raise SystemExit(f"id {r.id}: query type {qt} has no check")


def release(full, M, vl, vc) -> pd.DataFrame:
    """Every question of the CSV release, by query type."""
    rows = []
    for qt, g in full.groupby("query_type"):
        c = {}
        for i in g.index:
            r = full.loc[i].copy()
            r["id"] = i
            v = check(r, M, vl, vc)
            c[v] = c.get(v, 0) + 1
        follows = c.get("decided by a published error, label follows it", 0)
        recomputed = c.get("decided by a published error, label follows the recomputed value", 0)
        rows.append(dict(query_type=qt, n=len(g), right=c.get("right", 0),
                         not_decidable_from_stated=c.get("right, not decidable from the stated numbers", 0),
                         decided_by_published_error=follows + recomputed,
                         label_follows_published_error=follows,
                         label_follows_recomputed=recomputed,
                         label_wrong_otherwise=len(g) - sum(c.get(k, 0) for k in (
                             "right", "right, not decidable from the stated numbers")) - follows - recomputed))
    R = pd.DataFrame(rows)
    R["not_decidable_pct"] = (100 * R.not_decidable_from_stated / R.n).round(2)
    return R


def main() -> int:
    vl, vc = _load("verify_labels"), _load("verify_counterfactual")
    full = pd.read_csv(ROOT / "data" / "cladder" / "full_v1.5_default.csv").set_index("id")
    tags = {}
    for p in sorted(RAW.glob("pilot_raw_*.csv")):
        if "_recap" in p.stem:
            continue
        tag = re.sub(r"^pilot_raw_", "", p.stem)
        tags[tag] = set(pd.read_csv(p, usecols=["id"]).id.astype(int))
    M = Matcher()
    verdict = {}
    for i in sorted(set().union(*tags.values())):
        r = full.loc[i].copy()
        r["id"] = i
        verdict[i] = check(r, M, vl, vc)
    rows = []
    for tag, ids in tags.items():
        v = pd.Series([verdict[i] for i in sorted(ids)], index=sorted(ids))
        bad = sorted(v[v.str.startswith("decided by a published error")].index)
        fol = sorted(v[v == "decided by a published error, label follows it"].index)
        nd = sorted(v[v == "right, not decidable from the stated numbers"].index)
        wrong = sorted(v[~(v.isin(["right", "right, not decidable from the stated numbers"])
                           | v.str.startswith("decided by a published error"))].index)
        rows.append(dict(sample=tag, n_questions=len(ids), n_checked=len(ids),
                         n_decided_by_error=len(bad),
                         query_types_decided=",".join(sorted({full.loc[i].query_type for i in bad})),
                         ids_decided=",".join(map(str, bad)),
                         n_label_follows_error=len(fol), ids_label_follows_error=",".join(map(str, fol)),
                         n_not_decidable=len(nd),
                         query_types_not_decidable=",".join(sorted({full.loc[i].query_type for i in nd})),
                         ids_not_decidable=",".join(map(str, nd)),
                         n_label_wrong_otherwise=len(wrong), ids_label_wrong_otherwise=",".join(map(str, wrong))))
    R = pd.DataFrame(rows)
    R.to_csv(ROOT / "results" / "cladder" / "sample_labels.csv", index=False)
    counts = pd.Series(verdict).value_counts()
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(counts.to_string())
        print()
        print(R.drop(columns=["ids_decided", "ids_label_follows_error", "ids_not_decidable",
                              "ids_label_wrong_otherwise"]).to_string(index=False))
    Rel = release(full, M, vl, vc)
    Rel.to_csv(ROOT / "results" / "cladder" / "release_labels.csv", index=False)
    print()
    print(Rel.to_string(index=False))
    print("\n  wrote results/cladder/sample_labels.csv, release_labels.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
