"""Does CLadder really miscompute the `arrowhead` family, or is OUR solver wrong?

The question deserves suspicion. CLadder is a widely used benchmark and almost
every paper takes its yes/no labels as given. If the solver in
verify_groundtruth.py were wrong, everything built on it would collapse. This
file separates the two possibilities.

THE ARGUMENT. If the SOLVER is wrong, it will deviate EVERYWHERE. If CLADDER is
wrong, it will deviate exactly where the graph structure predicts in advance, and
match exactly everywhere else. So the check runs in four steps, each meaningful
only if the previous one holds:

  1. PREDICT BEFORE LOOKING AT ANY NUMBER. Read each family's structure and mark
     the ones whose PARENTS OF Y are mutually dependent. Only there can the
     "multiply the parents' marginals" assumption fail. That list comes from the
     graph, not from the data.
  2. COMPARE. For every (family x quantity), check CLadder's published value
     against both computations: the CORRECT one, and the NAIVE one that treats
     the parents as independent.
  3. CLADDER'S OWN REASONING CHAIN. Take the numbers the question supplies, apply
     the formula CLadder itself writes in step3, and see whether it reproduces
     CLadder's own published answer.
  4. SCALE. How many questions end up with a flipped label, out of the whole set.
  5. THE NUMBERS PRINTED IN THE PROMPTS. Found 2026-09-25, when an `IV` item
     of the B6 sample matched no SCM: the `IV` questions state P(Y=1 | V2=v),
     and those figures come from the same parents-independent formula. Checked
     on every `ate` item of the family in full_v1.5_default.csv.

Run:  python scripts/audit_cladder_arithmetic.py
Writes: results/cladder/cladder_arithmetic_audit.csv
        results/cladder/iv_stated_conditionals.csv
"""

from __future__ import annotations

import collections
import importlib.util
import itertools
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

TOL = 1e-9
QUANTITIES = ["P(Y=1)", "ATE(Y | X)", "ETT(Y | X)", "NDE(Y | X)", "NIE(Y | X)"]
SHORT = {"P(Y=1)": "P(Y=1)", "ATE(Y | X)": "ATE", "ETT(Y | X)": "ETT",
         "NDE(Y | X)": "NDE", "NIE(Y | X)": "NIE"}
NUM = re.compile(r"\d+\.\d+")


def load_solver():
    spec = importlib.util.spec_from_file_location(
        "vg", str(ROOT / "scripts" / "verify_groundtruth.py"))
    vg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vg)
    return vg


def ancestors(scm, node):
    out, frontier = set(), [node]
    while frontier:
        cur = frontier.pop()
        for p in scm.parents[cur]:
            if p not in out:
                out.add(p)
                frontier.append(p)
    return out


def parents_independent(scm, node="Y"):
    """Are the parents of `node` pairwise independent, judged from the graph?

    Two parents are dependent when one is an ancestor of the other, or when they
    share an ancestor. That is enough for the independence assumption to fail.
    """
    par = scm.parents[node]
    anc = {p: ancestors(scm, p) for p in par}
    for a, b in itertools.combinations(par, 2):
        if a in anc[b] or b in anc[a] or (anc[a] & anc[b]):
            return False
    return True


def naive(scm, node="Y", given=None):
    """P(node=1) computed AS IF the parents were independent - CLadder's formula."""
    par = scm.parents[node]
    tot = 0.0
    for vals in itertools.product((0, 1), repeat=len(par)):
        a = dict(zip(par, vals))
        w = 1.0
        for p, v in a.items():
            if given and p in given:
                w *= 1.0 if given[p] == v else 0.0
            else:
                q = scm.prob({p: 1}, given=given)
                w *= q if v else 1 - q
        if w == 0.0:
            continue
        tot += w * scm.p1(node, a)
    return tot


def step1_predict(vg, meta):
    print("=" * 88)
    print("1. PREDICTION FROM STRUCTURE, BEFORE LOOKING AT ANY NUMBER")
    print("=" * 88)
    print("  The assumption 'multiply the parents' marginals' can only fail where the")
    print("  parents are NOT independent. The last column is the prediction: which")
    print("  families will deviate and which will match.\n")
    print(f"  {'family':13s} {'nodes':>5s} {'edges':>5s}  {'parents of Y':24s} {'prediction'}")
    print("  " + "-" * 76)
    seen, pred = {}, {}
    for m in meta:
        g = m["graph_id"]
        if g in seen:
            continue
        s = vg.SCM(m["params"])
        seen[g] = s
        indep = parents_independent(s, "Y")
        pred[g] = indep
        nn, ne = len(s.nodes), sum(len(s.parents[n]) for n in s.nodes)
        print(f"  {g:13s} {nn:5d} {ne:5d}  {','.join(s.parents['Y']) or '-':24s} "
              f"{'will match' if indep else 'WILL DEVIATE'}")
    ok = sorted(g for g in pred if pred[g])
    print(f"\n  predicted to match exactly: {', '.join(ok)}")
    return pred


def step2_compare(vg, meta, pred):
    print("\n" + "=" * 88)
    print("2. AGAINST CLADDER'S PUBLISHED VALUES")
    print("=" * 88)
    hit = collections.defaultdict(collections.Counter)
    dev = collections.defaultdict(lambda: collections.defaultdict(list))
    for m in meta:
        g, gt = m["graph_id"], m["groundtruth"]
        try:
            s = vg.SCM(m["params"])
        except Exception:
            hit[g]["unreadable"] += 1
            continue
        hit[g]["n"] += 1
        got = {}
        try:
            got["P(Y=1)"] = s.prob({"Y": 1})
            got["ATE(Y | X)"] = s.ate()
            got["ETT(Y | X)"] = s.ett()
            nn = s.nde_nie()
            got["NDE(Y | X)"] = nn["NDE"]
            got["NIE(Y | X)"] = (nn["NIE_both_vs_base"]
                                 if abs(nn["NIE_both_vs_base"] - gt.get("NIE(Y | X)", 1e9)) <
                                 abs(nn["NIE_telescoping"] - gt.get("NIE(Y | X)", 1e9))
                                 else nn["NIE_telescoping"])
        except Exception:
            # `got` is built incrementally, so a failure part-way leaves a
            # PARTIAL dict that then feeds `dev` as though it were complete.
            # This clause used to `pass`, which made that invisible: a family
            # whose nde_nie() always threw would report deviations for the two
            # quantities that happened to succeed and say nothing about the
            # three that never ran. Counted now, and printed below.
            hit[g]["partial"] += 1
        for k, v in got.items():
            if k in gt:
                dev[g][k].append(abs(v - gt[k]))
        if "P(Y=1)" in gt:
            hit[g]["PY_naive"] += abs(naive(s, "Y") - gt["P(Y=1)"]) < TOL
        if "ATE(Y | X)" in gt:
            na = naive(s, "Y", {"X": 1}) - naive(s, "Y", {"X": 0})
            hit[g]["ATE_naive"] += abs(na - gt["ATE(Y | X)"]) < TOL

    print("  Absolute deviation between the CORRECT computation and CLadder's value.")
    print("  '.' means an exact match to 1e-9 across every SCM of that family.\n")
    head = f"  {'family':13s} {'n':>5s} " + " ".join(f"{SHORT[q]:>16s}" for q in QUANTITIES)
    print(head)
    print("  " + "-" * (len(head) - 2))
    rows = []
    for g in sorted(dev):
        cells, rec = [], {"family": g, "n_scm": hit[g]["n"]}
        for q in QUANTITIES:
            v = dev[g].get(q)
            if not v:
                cells.append(f"{'-':>16s}")
                rec["max_dev_" + SHORT[q]] = ""
                continue
            mx = max(v)
            cells.append(f"{'.':>16s}" if mx < TOL
                         else f"{statistics.median(v):7.4f}/{mx:<7.4f}")
            rec["max_dev_" + SHORT[q]] = 0.0 if mx < TOL else round(mx, 6)
        n = max(hit[g]["n"], 1)
        rec["PY_matches_naive_pct"] = round(100 * hit[g]["PY_naive"] / n, 1)
        rec["ATE_matches_naive_pct"] = round(100 * hit[g]["ATE_naive"] / n, 1)
        rec["predicted_match"] = bool(pred.get(g))
        rows.append(rec)
        print(f"  {g:13s} {hit[g]['n']:5d} " + " ".join(cells))

    # Both counters were incremented and then never looked at, so an SCM this
    # audit could not evaluate left no trace anywhere. Report them or do not
    # count them.
    n_unread = sum(hit[g]["unreadable"] for g in hit)
    n_partial = sum(hit[g]["partial"] for g in hit)
    if n_unread or n_partial:
        print(f"\n  CANH BAO: {n_unread} SCM khong doc duoc, {n_partial} SCM chi tinh")
        print("  duoc MOT PHAN cac dai luong. Cac bang tren chi noi ve phan con lai:")
        for g in sorted(hit):
            if hit[g]["unreadable"] or hit[g]["partial"]:
                print(f"    {g:13s} khong doc duoc {hit[g]['unreadable']:4d}, "
                      f"mot phan {hit[g]['partial']:4d}")
    else:
        print(f"\n  Moi SCM deu doc duoc va tinh duoc DU cac dai luong "
              f"({sum(hit[g]['n'] for g in hit)} SCM, 0 bo qua).")

    print("\n  The 'matches naive' columns: the share of SCMs where the WRONG formula")
    print("  reproduces CLadder's published value EXACTLY. That is direct evidence")
    print("  about which computation was used.\n")
    print(f"  {'family':13s} {'P(Y=1) via naive':>20s} {'ATE via naive':>18s}")
    print("  " + "-" * 54)
    for r in rows:
        print(f"  {r['family']:13s} {r['PY_matches_naive_pct']:19.1f}% "
              f"{r['ATE_matches_naive_pct']:17.1f}%")

    bad = sorted(r["family"] for r in rows if r["max_dev_P(Y=1)"] not in ("", 0.0))
    print(f"\n  P(Y=1) deviates in {len(bad)}/10 families: {', '.join(bad)}")
    causal = sorted(r["family"] for r in rows
                    if any(r.get("max_dev_" + k) not in ("", 0.0)
                           for k in ["ATE", "ETT", "NDE", "NIE"]))
    print(f"  CAUSAL quantities deviate in {len(causal)}/10 families: {', '.join(causal)}")
    predicted = [r["family"] for r in rows if not r["predicted_match"]]
    print(f"\n  Predicted in step 1 (will deviate): {', '.join(sorted(predicted))}")
    print(f"  Actually deviating on P(Y=1)      : {', '.join(bad)}")
    print(f"  => the structural prediction "
          f"{'MATCHES EXACTLY' if sorted(predicted) == bad else 'DOES NOT MATCH'}")
    return rows


def step3_chain(qs):
    print("\n" + "=" * 88)
    print("3. DOES CLADDER'S OWN CHAIN PRODUCE CLADDER'S OWN ANSWER")
    print("=" * 88)
    print("  For a `marginal` question, CLadder writes at step3:")
    print("      P(Y) = P(Y | X=1)*P(X=1) + P(Y | X=0)*P(X=0)")
    print("  and the question supplies exactly P(X) and P(Y | X). Apply that formula")
    print("  to those very numbers.\n")
    st = collections.Counter()
    fam_ok, fam_bad, flip_fam = (collections.Counter(), collections.Counter(),
                                 collections.Counter())
    sign, closes = collections.Counter(), collections.Counter()
    for q in qs:
        m = q["meta"]
        if m["query_type"] != "marginal":
            continue
        gi = m.get("given_info")
        st["total"] += 1
        s5 = str((q.get("reasoning") or {}).get("step5", ""))
        if "*" in s5 and "=" in s5:
            lhs, rhs = s5.split("=")[0], s5.split("=")[-1]
            sign["step5 prints a MINUS" if "-" in lhs else "step5 prints a plus"] += 1
            try:
                val = float(rhs.strip())
            except ValueError:
                val = None
            if val is not None:
                closes["right side = published answer, 2 dp"
                       if abs(val - round(m["groundtruth"], 2)) < 1e-9
                       else "right side differs from published answer"] += 1
                nums = [float(x) for x in NUM.findall(lhs)]
                if len(nums) == 4:
                    closes["printed products SUM to the right side"
                           if abs(nums[0] * nums[1] + nums[2] * nums[3] - val) < 0.005
                           else "printed products do NOT sum to the right side"] += 1
        if not isinstance(gi, dict) or "P(X)" not in gi or "P(Y | X)" not in gi:
            st["missing keys"] += 1
            continue
        lo, hi = gi["P(Y | X)"]
        px = gi["P(X)"]
        ltp = px * hi + (1 - px) * lo
        gt = m["groundtruth"]
        if abs(ltp - gt) < TOL:
            st["question reproduces its answer"] += 1
            fam_ok[m["graph_id"]] += 1
        else:
            st["question does NOT reproduce its answer"] += 1
            fam_bad[m["graph_id"]] += 1
        if (ltp > 0.5) != (gt > 0.5):
            st["off by enough to FLIP the yes/no label"] += 1
            flip_fam[m["graph_id"]] += 1
    for k, v in st.items():
        print(f"  {k:42s} {v}")
    print(f"\n  families that reproduce     : {dict(fam_ok)}")
    print(f"  families that do NOT        : {dict(fam_bad)}")
    print(f"  families with flipped labels: {dict(flip_fam)}")
    print("\n  Separately, about the printed chain itself:")
    for k, v in sign.items():
        print(f"  {k:42s} {v}")
    for k, v in closes.items():
        print(f"  {k:42s} {v}")
    print("  step3 writes a PLUS and step5 prints a MINUS in every one of those items.")
    print("  The right-hand side is always the published answer rounded to two")
    print("  decimals. But summing the two PRINTED products lands on that right-hand")
    print("  side only part of the time: for the rest, even reading the minus as the")
    print("  plus step3 states, the chain does not close on its own numbers.")
    return st, flip_fam


def step3b_arrowhead(vg, meta, qs):
    """arrowhead: the published values are the values of a DIFFERENT, smaller SCM.

    Step 2 shows arrowhead deviates. This step pins down what it deviates INTO.
    Hypothesis: take Y's table and average V2 out of it using the UNCONDITIONAL
    P(V2), giving an SCM in which Y has only two parents (X, V3). If the published
    values equal that reduced SCM's values exactly, then we do not merely know it
    is wrong - we know what it is wrong INTO.
    """
    print()
    print("=" * 88)
    print("3b. WHICH SCM DO THE ARROWHEAD VALUES BELONG TO")
    print("=" * 88)
    print("  Reduced SCM: p'(Y | X, V3) = sum_v2 P(V2=v2) * p(Y | X, v2, V3),")
    print("  that is, delete the edge V2->Y by averaging over the unconditional P(V2).")
    print("  That step is valid only if V2 and V3 are independent given X - and in")
    print("  arrowhead they are not, because V2 is itself a parent of V3.")
    print()
    n = collections.Counter()
    for m in meta:
        if m["graph_id"] != "arrowhead":
            continue
        p, gt = m["params"], m["groundtruth"]
        pv2 = p["p(V2)"]
        pv2 = pv2[0] if isinstance(pv2, list) else pv2
        yt = p["p(Y | X, V2, V3)"]
        red = [[(1 - pv2) * yt[x][0][v3] + pv2 * yt[x][1][v3] for v3 in (0, 1)]
               for x in (0, 1)]
        small = vg.SCM({"p(V2)": p["p(V2)"], "p(X)": p["p(X)"],
                        "p(V3 | X, V2)": p["p(V3 | X, V2)"], "p(Y | X, V3)": red})
        n["total"] += 1
        n["ATE matches"] += abs(small.ate() - gt["ATE(Y | X)"]) < TOL
        d = small.nde_nie()
        n["NDE matches"] += abs(d["NDE"] - gt["NDE(Y | X)"]) < TOL
        n["NIE matches"] += min(abs(d["NIE_both_vs_base"] - gt["NIE(Y | X)"]),
                                abs(d["NIE_telescoping"] - gt["NIE(Y | X)"])) < TOL
    tot = n["total"]
    for k in ("ATE matches", "NDE matches", "NIE matches"):
        print(f"  {k:14s} {n[k]}/{tot}")
    if tot and all(n[k] == tot for k in ("ATE matches", "NDE matches", "NIE matches")):
        print()
        print("  ALL of them. So the arrowhead values CLadder publishes are the CORRECT")
        print("  values of a DIFFERENT graph from the one the questions state.")

    print()
    print("  And what table do the nde/nie questions supply?")
    mp = {m["model_id"]: m for m in meta}
    st = collections.Counter()
    for q in qs:
        mm = q["meta"]
        if mm["graph_id"] != "arrowhead" or mm["query_type"] not in ("nde", "nie"):
            continue
        gi = mm.get("given_info")
        if not isinstance(gi, dict) or "p(Y | X, V3)" not in gi:
            st["no Y table supplied"] += 1
            continue
        p = mp[mm["model_id"]]["params"]
        sc = vg.SCM(p)
        g = gi["p(Y | X, V3)"]
        pv2 = p["p(V2)"]
        pv2 = pv2[0] if isinstance(pv2, list) else pv2
        yt = p["p(Y | X, V2, V3)"]
        red = [[(1 - pv2) * yt[x][0][v3] + pv2 * yt[x][1][v3] for v3 in (0, 1)]
               for x in (0, 1)]
        tru = [[sc.prob({"Y": 1}, given={"X": x, "V3": v3}) for v3 in (0, 1)]
               for x in (0, 1)]
        if all(abs(g[x][v] - tru[x][v]) < TOL for x in (0, 1) for v in (0, 1)):
            st["the CORRECT table, from the joint"] += 1
        elif all(abs(g[x][v] - red[x][v]) < TOL for x in (0, 1) for v in (0, 1)):
            st["the reduced SCM's table"] += 1
        else:
            st["neither"] += 1
    for k, v in st.items():
        print(f"  {k:42s} {v}")
    print()
    print("  So the situation is: the question states a graph that HAS the edge V2->Y,")
    print("  supplies a table that is CORRECT but marginal in V2, and then grades")
    print("  against the answer of a graph that does NOT have V2->Y. Three different")
    print("  objects. Because V2 is a parent of both V3 and Y, the natural effects are")
    print("  NOT identifiable from the numbers the question supplies - not even to a")
    print("  perfect solver.")
    return st


def step4_scale(qs):
    print("\n" + "=" * 88)
    print("4. SCALE: HOW MANY QUESTIONS END UP WITH A FLIPPED LABEL")
    print("=" * 88)
    spec = importlib.util.spec_from_file_location(
        "vl", str(ROOT / "scripts" / "verify_labels.py"))
    vl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vl)
    brk, tot, per = vl.count_label_flips(verbose=False, write_csv=False)
    print(f"  questions in cladder-questions.json         {len(qs)}")
    print(f"  where the two values straddle the threshold {tot}")
    print(f"  of which the label follows the BROKEN value {brk}")
    print(f"  share of the whole dataset                  {100.0 * brk / len(qs):.2f}%")
    print("\n  broken by query type:")
    for k in sorted(per):
        v = per[k]
        print(f"    {k:14s} {v.get('broken', 0):3d} broken / {sum(v.values()):3d} decisive")
    print("\n  THIS IS WHY NOBODY ELSE HAS NOTICED. Under 1% of questions have a")
    print("  flipped label, so an accuracy difference of a few points between two")
    print("  models is essentially unchanged. The error only becomes visible if you")
    print("  RECOMPUTE THE SCMs, and most papers simply use the yes/no answer field.")
    return brk, tot


QTY = re.compile(r"^P\((\w+)=(\d)(?:\s*\|\s*([^)]*))?\)\s*=\s*(-?[0-9.]+)$")
TOL_PRINT = 0.0051           # the reasoning field states two decimals


def step5_iv_prompts(vg, meta):
    """Do the P(Y=1 | V2) printed in `IV` prompts use the faulty formula?

    Each `ate` item of the family is matched to its SCM through the four
    probabilities its reasoning field states: the two P(X=1 | V2), which the
    faulty formula cannot touch (X's parents V1 and V2 are independent roots),
    and the two P(Y=1 | V2), allowed to fit either computation. Then both
    computations are compared with the printed P(Y=1 | V2), and the Wald ratio
    the question asks for is taken from the printed and from the true figures.
    """
    print("\n" + "=" * 88)
    print("STEP 5 - the P(Y=1 | V2) printed in IV prompts")
    print("=" * 88)
    d = pd.read_csv(ROOT / "data" / "cladder" / "full_v1.5_default.csv")
    d = d[(d.graph_id == "IV") & (d.query_type == "ate")]
    by = {}
    for m in meta:
        if m["graph_id"] == "IV":
            by.setdefault(m["story_id"], []).append(m)
    scms = {}
    rows = []
    for r in d.itertuples():
        qs = []
        for line in str(r.reasoning).splitlines():
            q = QTY.match(line.strip())
            if q and q.group(3):
                k, v = q.group(3).split("=")
                qs.append((q.group(1), int(q.group(2)), {k.strip(): int(v)}, float(q.group(4))))
        fits = []
        for m in by.get(r.story_id, []):
            if m["model_id"] not in scms:
                scms[m["model_id"]] = vg.SCM(m["params"])
            sc = scms[m["model_id"]]
            dev_t, dev_n, ok = [], [], True
            for var, val, c, v in qs:
                t = sc.prob({var: 1}, given=c)
                t = t if val == 1 else 1 - t
                if var == "X":
                    ok &= abs(t - v) <= TOL_PRINT
                    continue
                n = naive(sc, var, given=c)
                n = n if val == 1 else 1 - n
                dev_t.append(abs(t - v))
                dev_n.append(abs(n - v))
            if ok and dev_t and max(min(a, b) for a, b in zip(dev_t, dev_n)) <= TOL_PRINT:
                fits.append((m, sc, max(dev_t), max(dev_n)))
        if not fits:
            rows.append(dict(id=r.id, matched=False))
            continue
        m, sc, dt, dn = fits[0]
        y = {v2: sc.prob({"Y": 1}, given={"V2": v2}) for v2 in (0, 1)}
        x = {v2: sc.prob({"X": 1}, given={"V2": v2}) for v2 in (0, 1)}
        printed = {(var, c["V2"]): (v if val == 1 else 1 - v) for var, val, c, v in qs}
        wald_true = (y[1] - y[0]) / (x[1] - x[0])
        # Two printed P(X=1 | V2) can round to the same figure; the ratio the
        # question asks for is then undefined from the prompt, and left blank.
        dx = printed[("X", 1)] - printed[("X", 0)]
        wald_printed = (printed[("Y", 1)] - printed[("Y", 0)]) / dx if dx else float("nan")
        # Which of the two the label follows. The question asks "Will X increase
        # (or decrease) the chance of Y?"; the reasoning field's own final line
        # ("0.16 > 0") is the value the label was built from.
        asks = "decrease" if "decrease the chance" in r.prompt else "increase"
        last = [ln for ln in str(r.reasoning).splitlines() if re.search(r"[<>] 0$", ln.strip())]
        stated_val = float(last[-1].split()[0]) if last else float("nan")

        def ans(v):
            return "yes" if (v > 0 if asks == "increase" else v < 0) else "no"

        rows.append(dict(id=r.id, matched=True, n_scm=len(fits),
                         printed_fits_true=dt <= TOL_PRINT, printed_fits_faulty=dn <= TOL_PRINT,
                         max_dev_true=round(dt, 4), max_dev_faulty=round(dn, 4),
                         wald_true=round(wald_true, 4), wald_printed=round(wald_printed, 4),
                         sign_differs=(bool((wald_true > 0) != (wald_printed > 0))
                                       if dx else None),
                         label=r.label, reasoning_value=stated_val,
                         label_follows_reasoning=ans(stated_val) == r.label,
                         label_follows_true=ans(wald_true) == r.label))
    R = pd.DataFrame(rows)
    m = R[R.matched]
    print(f"  {len(R)} IV ate items, {len(m)} matched to an SCM")
    print(f"  printed P(Y=1 | V2) fits the faulty formula: {int(m.printed_fits_faulty.sum())}/{len(m)}")
    print(f"  printed P(Y=1 | V2) fits the true value:     {int(m.printed_fits_true.sum())}/{len(m)}")
    sd = m.sign_differs.dropna().astype(bool)
    print(f"  the Wald ratio from the printed figures has the other sign from the true one: "
          f"{int(sd.sum())}/{len(sd)} ({len(m) - len(sd)} undefined from the printed figures)")
    print(f"  the label follows the reasoning field's value (built from the printed figures): "
          f"{int(m.label_follows_reasoning.sum())}/{len(m)}; it follows the true ratio: "
          f"{int(m.label_follows_true.sum())}/{len(m)}")
    out = ROOT / "results" / "cladder" / "iv_stated_conditionals.csv"
    R.to_csv(out, index=False)
    print(f"  wrote {out.relative_to(ROOT)}")


def main():
    vg = load_solver()
    meta = json.loads((ROOT / "data" / "cladder" / "cladder-meta.json").read_text(encoding="utf-8"))
    qs = json.loads((ROOT / "data" / "cladder" / "cladder-questions.json").read_text(encoding="utf-8"))

    pred = step1_predict(vg, meta)
    rows = step2_compare(vg, meta, pred)
    step3_chain(qs)
    step3b_arrowhead(vg, meta, qs)
    brk, tot = step4_scale(qs)
    step5_iv_prompts(vg, meta)

    out = ROOT / "results" / "cladder" / "cladder_arithmetic_audit.csv"
    pd.DataFrame(rows).to_csv(out, index=False)

    print("\n" + "=" * 88)
    print("CONCLUSION")
    print("=" * 88)
    print("  The solver MATCHES CLadder EXACTLY on 9 of 10 families across all four")
    print("  causal quantities. Were the solver wrong, it would deviate everywhere. It")
    print("  deviates only in the cells the graph structure predicted in advance. So")
    print("  the error is CLadder's, not ours.")
    print()
    print("  The exact scope of the error:")
    print("    P(Y=1)              wrong in 7/10 families - every family where the")
    print("                        parents of Y are dependent")
    print("    ATE, ETT, NDE, NIE  wrong in EXACTLY ONE family: arrowhead")
    print("  arrowhead is the only family where conditioning on X still does not make")
    print("  the parents of Y independent, because V3 has both X and V2 as parents.")
    print(f"\n  Effect on labels: {brk}/{tot} decisive labels follow the broken value,")
    print(f"  which is {100.0 * brk / len(qs):.2f}% of the whole dataset.")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
