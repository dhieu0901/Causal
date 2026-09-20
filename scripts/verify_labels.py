"""Which value does CLadder's yes/no label follow: the PUBLISHED one or the CORRECT one?

Background. `verify_groundtruth.py` shows that CLadder's `meta.groundtruth` field
comes from a miscomputation: it multiplies the marginal probabilities of a node's
parents as if they were independent. The next question is whether that error
reaches the yes/no label used for scoring.

Method, taking no number from REPORT.md:
  1. recompute the CORRECT value with the SCM solver in verify_groundtruth.py
  2. read the PUBLISHED value from meta.groundtruth
  3. keep only the questions where the two fall on OPPOSITE sides of the
     decision threshold
  4. for each of those, see which side the yes/no label in the data agrees with

Run:  python scripts/verify_labels.py
Writes: a table to the console, and results/_label_flips.csv
"""

import csv
import importlib.util
import json
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# only `marginal` is a probability, so only it is compared against 0.5; every
# other quantity is a difference and is compared against 0
THRESHOLD = {"marginal": 0.5}

# the solver does not cover these two; named explicitly rather than skipped
# silently. scripts/verify_counterfactual.py covers them separately.
UNSUPPORTED = ("exp_away", "det-counterfactual")


def _load_solver():
    spec = importlib.util.spec_from_file_location(
        "vg", os.path.join(ROOT, "scripts", "verify_groundtruth.py"))
    vg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vg)
    return vg


def threshold(qt):
    return THRESHOLD.get(qt, 0.0)


def true_value(vg, scm_meta, qt):
    """Recompute the quantity with the SCM solver. None if unsupported."""
    scm = vg.SCM(scm_meta["params"])
    if qt == "marginal":
        return scm.prob({"Y": 1})
    if qt == "correlation":
        return (scm.prob({"Y": 1}, given={"X": 1})
                - scm.prob({"Y": 1}, given={"X": 0}))
    if qt == "ate":
        return scm.ate()
    if qt == "ett":
        return scm.ett()
    if qt in ("nde", "nie"):
        d = scm.nde_nie()
        # CLadder uses Pearl's both-versus-base convention, not telescoping
        return d["NDE"] if qt == "nde" else d["NIE_both_vs_base"]
    return None


def label_for(value, thr, polarity):
    """The yes/no label implied by a value. This rule was checked on the whole set."""
    return "yes" if ((value > thr) == bool(polarity)) else "no"


def count_label_flips(verbose=True, write_csv=True):
    """Returns (n_following_broken_value, n_decisive_questions, table_by_query_type)."""
    vg = _load_solver()
    meta = json.loads(open(os.path.join(ROOT, "data", "cladder-meta.json"),
                           encoding="utf-8").read())
    qs = json.loads(open(os.path.join(ROOT, "data", "cladder-questions.json"),
                         encoding="utf-8").read())
    scms = {m["model_id"]: m for m in meta}

    skipped = Counter()
    decisive = []
    for q in qs:
        m = q["meta"]
        qt = m["query_type"]
        pub = m.get("groundtruth")
        if not isinstance(pub, (int, float)):
            skipped["groundtruth is not a number"] += 1
            continue
        if qt in UNSUPPORTED:
            skipped["solver does not cover %s" % qt] += 1
            continue
        sm = scms.get(m["model_id"])
        if sm is None:
            skipped["no SCM"] += 1
            continue
        try:
            tru = true_value(vg, sm, qt)
        except Exception as e:
            skipped["error computing %s" % type(e).__name__] += 1
            continue
        if tru is None:
            skipped["unsupported %s" % qt] += 1
            continue
        t = threshold(qt)
        if (float(pub) > t) == (float(tru) > t):
            continue                      # same side, the label is unaffected
        decisive.append((q, m, qt, float(pub), float(tru), t))

    per_qt = defaultdict(Counter)
    rows = []
    unresolved = 0
    for q, m, qt, pub, tru, t in decisive:
        pol = m.get("polarity")
        ans = q["answer"]
        if pol is None:
            unresolved += 1
            continue
        lp, lt = label_for(pub, t, pol), label_for(tru, t, pol)
        if lp == lt:
            unresolved += 1
            continue
        if ans == lp:
            side = "broken"
        elif ans == lt:
            side = "correct"
        else:
            unresolved += 1
            continue
        per_qt[qt][side] += 1
        rows.append({"question_id": q.get("question_id"),
                     "graph_id": m["graph_id"], "query_type": qt,
                     "published_value": pub, "correct_value": tru,
                     "threshold": t, "label": ans, "label_follows": side})

    broken = sum(c["broken"] for c in per_qt.values())
    total = broken + sum(c["correct"] for c in per_qt.values())

    if write_csv:
        out = os.path.join(ROOT, "results", "_label_flips.csv")
        with open(out, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    if verbose:
        print("CHECKING CLADDER'S ANSWER KEY")
        print("=" * 62)
        if skipped:
            print("  skipped:", dict(skipped))
        print("  DECISIVE questions (the two values straddle the threshold):",
              len(decisive))
        if unresolved:
            print("  could not be decided:", unresolved)
        print()
        print(f"  {'query_type':14s} {'BROKEN':>7s} {'CORRECT':>8s}")
        print("  " + "-" * 30)
        for qt in sorted(per_qt):
            print(f"  {qt:14s} {per_qt[qt]['broken']:7d} {per_qt[qt]['correct']:8d}")
        print("  " + "-" * 30)
        print(f"  {'TOTAL':14s} {broken:7d} {total - broken:8d}")
        print()
        print(f"  => {broken}/{total} labels follow the BROKEN value "
              f"({100.0 * broken / total:.1f}%)")
        if write_csv:
            print("  wrote results/_label_flips.csv")

    return broken, total, {k: dict(v) for k, v in per_qt.items()}


if __name__ == "__main__":
    count_label_flips()
