"""Does any question we sampled carry a label that a published CLadder error decides?

    python scripts/verify_sample_labels.py

verify_labels.py counts, in CLadder's JSON release, the questions whose
published value and recomputed value fall on opposite sides of the decision
threshold (91, of which 83 carry the label of the wrong value). The questions
this project samples come from full_v1.5_default.csv, a different set of
questions on the same 7,064 SCMs. So the same
check is run here on every question of every sample: each is matched to its
SCM through the probabilities its reasoning field states
(analyze_answer_change.Matcher), and the published and recomputed values of
its quantity are compared. `correlation` is not checked (its stated joint
probabilities do not parse into the matcher; no test uses it), nor are the
query types the solver does not cover (their labels are checked by
verify_counterfactual.py).

Writes results/cladder/sample_labels.csv: one row per sample file tag.
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

RAW = ROOT / "results" / "cladder" / "raw"
KEY = {"ate": "ATE(Y | X)", "ett": "ETT(Y | X)", "nde": "NDE(Y | X)", "nie": "NIE(Y | X)",
       "marginal": "P(Y=1)"}


def _vl():
    spec = importlib.util.spec_from_file_location("vl", str(ROOT / "scripts" / "verify_labels.py"))
    vl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vl)
    return vl


def main() -> int:
    vl = _vl()
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
        qt = r.query_type
        if qt not in KEY:
            verdict[i] = "not checked"
            continue
        cands = M.match(r)
        if not cands:
            raise SystemExit(f"id {i} ({qt}): no SCM reproduces its stated quantities")
        thr = vl.threshold(qt)
        dec = [(float(m["groundtruth"][KEY[qt]]) > thr) != (float(vl.true_value(VG, m, qt)) > thr)
               for m in cands]
        verdict[i] = "decided by a published error" if any(dec) else "unaffected"
    rows = []
    for tag, ids in tags.items():
        v = pd.Series([verdict[i] for i in sorted(ids)], index=sorted(ids))
        bad = sorted(v[v == "decided by a published error"].index)
        rows.append(dict(sample=tag, n_questions=len(ids), n_checked=int((v != "not checked").sum()),
                         n_decided_by_error=len(bad),
                         query_types_decided=",".join(sorted({full.loc[i].query_type for i in bad})),
                         ids_decided=",".join(map(str, bad))))
    R = pd.DataFrame(rows)
    R.to_csv(ROOT / "results" / "cladder" / "sample_labels.csv", index=False)
    with pd.option_context("display.width", 200):
        print(R.to_string(index=False))
    print("\n  wrote results/cladder/sample_labels.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
