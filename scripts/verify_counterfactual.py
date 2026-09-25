"""Check the three query types verify_labels.py leaves out: rung 3 and collision.

Why this file exists. `verify_labels.py` states outright that its solver does not
cover `det-counterfactual` or `exp_away`, and it never touches `collider_bias`
either, because that type's `groundtruth` field is a string rather than a number.
Together that is 1,812 questions across CLadder and - worse - **28 of the 86 items
in the `causal` group**, the group carrying this project's headline result. Filing
an issue about someone else's labels while one's own labels are unverified is a
hole that has to be closed.

The three types need three different machines:

  exp_away          P(Y=1 | X=1, V3=1) - P(Y=1 | V3=1). Purely observational, so
                    the existing SCM solver handles it directly.

  collider_bias     The question is "does X affect Y", on the collision graph
                    X->V3<-Y. The true causal effect is exactly zero, because Y is
                    a root and do(X) does not touch it. Note: CLadder's
                    `formal_form` field writes
                    `E[Y|do(X=1),V3=1] - E[Y|do(X=0),V3=1]`, and that quantity is
                    NOT zero in any of the 168 items, because conditioning on a
                    collider creates a spurious association. The answer follows the
                    causal effect, not that formula.

  det-counterfactual  A deterministic SCM. The mechanisms are not probability
                    tables but boolean expressions printed in `reasoning.step4`,
                    such as `V2 = not X`, `Y = X or V2`. Computed with Pearl's
                    three steps: abduction (find the root values consistent with
                    the evidence), action, prediction.

The boolean expressions are walked with `ast`, allowing only and/or/not, variable
names and constants. No `eval`.

A CLADDER DISPLAY BUG, found while writing this file. In the `diamondcut` family,
`step1` states the graph as `V1->V3, V1->X, X->Y, V3->Y`, so V3's parent is V1.
But `step4` writes `V3 = X` in 132 questions. The two readings agree in the actual
world, because X and V1 take the same value there, but they DIFFER under the
intervention do(X): if V3's parent is V1, then do(X) leaves V3 unchanged.
Substituting the graph-consistent equation reproduces all 1,476 answers exactly;
leaving it as printed leaves 29 mismatched. So CLADDER'S ANSWERS ARE RIGHT and the
printed equation is wrong - the opposite way round from issue #15, where the value
was wrong and the labelling rule was right.

Run:  python scripts/verify_counterfactual.py
Writes: results/cladder/counterfactual_verification.csv
"""

from __future__ import annotations

import ast
import collections
import importlib.util
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

TOL = 1e-9
FORM = re.compile(r"^Y_\{X=(\d)\}\s*=\s*(\d)\s*\|\s*(.*)$")
# diamondcut only, and only the right-hand side of V3. Deliberately narrow: a
# broader fix rewrote 288 equations and broke 40 questions that were already right.
DIAMONDCUT_BUG = re.compile(r"^V3 = (not )?X$", re.M)


def load_solver():
    spec = importlib.util.spec_from_file_location(
        "vg", str(ROOT / "scripts" / "verify_groundtruth.py"))
    vg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vg)
    return vg


# --------------------------------------------------------------------------
# Deterministic SCM
# --------------------------------------------------------------------------
def bool_eval(node, env):
    """Evaluate a boolean expression. Only and/or/not, names and constants."""
    if isinstance(node, ast.BoolOp):
        vals = [bool_eval(v, env) for v in node.values]
        return int(all(vals) if isinstance(node.op, ast.And) else any(vals))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return int(not bool_eval(node.operand, env))
    if isinstance(node, ast.Name):
        return int(env[node.id])
    if isinstance(node, ast.Constant):
        return int(node.value)
    raise ValueError("expression not allowed: %s" % ast.dump(node))


def parse_eqs(step4):
    out = []
    for line in str(step4).split("\n"):
        if "=" not in line:
            continue
        lhs, rhs = line.split("=", 1)
        out.append((lhs.strip(), ast.parse(rhs.strip(), mode="eval").body))
    return out


def forward(eqs, roots, do=None):
    """Run the equations forward. An intervened node loses its equation."""
    env, do = dict(roots), do or {}
    env.update(do)
    for lhs, tree in eqs:
        if lhs in do:
            continue
        env[lhs] = bool_eval(tree, env)
    return env


def counterfactual(eqs, evidence, x_val, target):
    """Pearl's three steps. Returns 1/0, or None if the evidence is not decisive."""
    lhss = {l for l, _ in eqs}
    allv = set(lhss)
    for _, t in eqs:
        allv |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
    roots = sorted(allv - lhss)

    consistent = []                               # step 1: abduction
    for vals in itertools.product((0, 1), repeat=len(roots)):
        r = dict(zip(roots, vals))
        if all(forward(eqs, r).get(k) == v for k, v in evidence.items()):
            consistent.append(r)
    if not consistent:
        return None
    res = {forward(eqs, r, do={"X": x_val})["Y"] for r in consistent}  # steps 2, 3
    if len(res) > 1:
        return None
    return 1 if res.pop() == target else 0


def check_deterministic(qs, rows):
    print("=" * 84)
    print("1. DET-COUNTERFACTUAL - deterministic SCM, Pearl's three steps")
    print("=" * 84)
    st = collections.Counter()
    for q in qs:
        m = q["meta"]
        if m["query_type"] != "det-counterfactual":
            continue
        st["total"] += 1
        fm = FORM.match(m["formal_form"].strip())
        if not fm:
            st["formal_form unreadable"] += 1
            continue
        x_val, target = int(fm.group(1)), int(fm.group(2))
        evidence = {}
        for part in filter(None, [p.strip() for p in fm.group(3).split(",")]):
            k, v = part.split("=")
            evidence[k.strip()] = int(v)

        txt = str(q["reasoning"].get("step4", ""))
        if m["graph_id"] == "diamondcut" and DIAMONDCUT_BUG.search(txt):
            st["V3 equation misprinted, replaced per the graph"] += 1
            txt = DIAMONDCUT_BUG.sub(
                lambda s: "V3 = %sV1" % (s.group(1) or ""), txt)
        try:
            eqs = parse_eqs(txt)
        except Exception:
            st["step4 unreadable"] += 1
            continue
        if not eqs:
            st["step4 empty"] += 1
            continue
        got = counterfactual(eqs, evidence, x_val, target)
        if got is None:
            st["evidence not decisive"] += 1
        elif got == m["groundtruth"]:
            st["MATCH"] += 1
        else:
            st["MISMATCH"] += 1
    for k in ("total", "MATCH", "MISMATCH", "evidence not decisive",
              "V3 equation misprinted, replaced per the graph"):
        if st[k]:
            print(f"  {k:46s} {st[k]}")
    rows.append({"query_type": "det-counterfactual", "n": st["total"],
                 "match": st["MATCH"], "mismatch": st["MISMATCH"]})
    return st


def check_collision(vg, meta, qs, rows):
    print("\n" + "=" * 84)
    print("2. EXP_AWAY and COLLIDER_BIAS - the collision graph X->V3<-Y")
    print("=" * 84)
    scms = {m["model_id"]: m for m in meta}
    st = collections.Counter()
    for q in qs:
        m = q["meta"]
        qt = m["query_type"]
        if qt not in ("exp_away", "collider_bias"):
            continue
        st[qt + " total"] += 1
        s = vg.SCM(scms[m["model_id"]]["params"])
        if qt == "exp_away":
            val = (s.prob({"Y": 1}, given={"X": 1, "V3": 1})
                   - s.prob({"Y": 1}, given={"V3": 1}))
            if abs(val - m["groundtruth"]) < TOL:
                st["exp_away value MATCH"] += 1
            lab = "yes" if ((val > 0) == bool(m["polarity"])) else "no"
            st["exp_away label MATCH" if lab == q["answer"]
               else "exp_away label MISMATCH"] += 1
        else:
            ate = s.prob({"Y": 1}, do={"X": 1}) - s.prob({"Y": 1}, do={"X": 0})
            if abs(ate) < TOL:
                st["collider_bias causal effect is exactly 0"] += 1
            cond = (s.prob({"Y": 1}, do={"X": 1}, given={"V3": 1})
                    - s.prob({"Y": 1}, do={"X": 0}, given={"V3": 1}))
            if abs(cond) > TOL:
                st["collider_bias CLadder's formal_form is nonzero"] += 1
            # Compare against a TOLERANCE, not against 0. The true effect is zero,
            # but the enumeration returns values like +1e-17, and `> 0` is then
            # True, which flipped the label on 17 items. That is floating-point
            # noise, not signal.
            lab = "yes" if ((ate > TOL) == bool(m["polarity"])) else "no"
            st["collider_bias label MATCH" if lab == q["answer"]
               else "collider_bias label MISMATCH"] += 1
    for k in sorted(st):
        print(f"  {k:46s} {st[k]}")
    print("\n  A note on `collider_bias`. CLadder's `formal_form` field reads")
    print("  E[Y|do(X=1),V3=1] - E[Y|do(X=0),V3=1], and that quantity is nonzero in")
    print("  all 168 items, because conditioning on a collider creates a spurious")
    print("  association. The answer follows the UNCONDITIONAL causal effect, which is")
    print("  exactly zero. The answer is RIGHT; only the notation in formal_form is off.")
    for qt in ("exp_away", "collider_bias"):
        rows.append({"query_type": qt, "n": st[qt + " total"],
                     "match": st[qt + " label MATCH"],
                     "mismatch": st[qt + " label MISMATCH"]})
    return st


def coverage_in_sample():
    """How much of this project's own `causal` group do these three types cover?"""
    print("\n" + "=" * 84)
    print("3. COVERAGE WITHIN THIS PROJECT'S SAMPLE")
    print("=" * 84)
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from analyze_querygroup import load, qset
    except Exception as e:
        print("  could not read the sample:", e)
        return
    d = load()
    qs_set = qset(d["KEEP"], "causal")
    k = d["KEEP"]
    sub = k[k.query_type.isin(qs_set)]
    c = sub.groupby("query_type")["item"].nunique()
    three = [x for x in c.index
             if x in ("det-counterfactual", "collider_bias", "exp_away")]
    print(f"  the `causal` group has {sub.item.nunique()} items")
    print(f"  of which these three types account for: {int(c[three].sum())}")
    for x in three:
        print(f"    {x:20s} {int(c[x])}")


def main():
    vg = load_solver()
    meta = json.loads((ROOT / "data" / "cladder" / "cladder-meta.json").read_text(encoding="utf-8"))
    qs = json.loads((ROOT / "data" / "cladder" / "cladder-questions.json").read_text(encoding="utf-8"))

    rows = []
    check_deterministic(qs, rows)
    check_collision(vg, meta, qs, rows)
    coverage_in_sample()

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "cladder" / "counterfactual_verification.csv"
    df.to_csv(out, index=False)

    total = int(df.n.sum())
    match = int(df.match.sum())
    print("\n" + "=" * 84)
    print("CONCLUSION")
    print("=" * 84)
    print(f"  {match}/{total} labels reproduce exactly across the three remaining types.")
    if match == total:
        print("  No query type is left unverified. Every label this project scores")
        print("  against has been recomputed, either from the SCM or from the structural")
        print("  equations.")
    print("\n  Separately, in the `diamondcut` family the V3 equation printed in step4")
    print("  contradicts the graph printed in step1. That is a DISPLAY bug: CLadder's")
    print("  answers are correct.")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
