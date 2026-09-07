"""Independently verify CLadder's ground truth by exact enumeration of the SCM.

    python scripts/verify_groundtruth.py

This is the trust check that justifies building on CLadder rather than on
NoisyCausal, whose Appendix D.1 shows all five construction steps were LLM
prompts and which documents no answer-computation procedure at all.

For each model we recompute the causal quantity from the published conditional
probability tables and compare against the published `groundtruth` field. A match
to floating-point precision means the labels were produced by a formal
procedure, not generated.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

TOL = 1e-9


def ate_mediation(p):
    """mediation: X -> V2 -> Y, X -> Y.  ATE = E[Y|do(X=1)] - E[Y|do(X=0)].

    X has no parents, so do(X=x) coincides with conditioning on X=x.
    """
    pv2, py = p["p(V2 | X)"], p["p(Y | X, V2)"]

    def EY(x):
        return sum((pv2[x] if v2 else 1 - pv2[x]) * py[x][v2] for v2 in (0, 1))

    return EY(1) - EY(0)


def ate_chain(p):
    """chain: X -> V2 -> Y.  Y depends on X only through V2."""
    pv2, py = p["p(V2 | X)"], p["p(Y | V2)"]

    def EY(x):
        return sum((pv2[x] if v2 else 1 - pv2[x]) * py[v2] for v2 in (0, 1))

    return EY(1) - EY(0)


def ate_confounding(p):
    """confounding: V1 -> X, V1 -> Y, X -> Y.  Backdoor adjustment on V1."""
    pv1, py = p["p(V1)"], p["p(Y | V1, X)"]

    def EY(x):
        return sum((pv1 if v1 else 1 - pv1) * py[v1][x] for v1 in (0, 1))

    return EY(1) - EY(0)


ESTIMATORS = {
    "mediation": ate_mediation,
    "chain": ate_chain,
    "confounding": ate_confounding,
}


def main():
    models = json.loads((ROOT / "data" / "cladder-meta.json").read_text(encoding="utf-8"))
    rows = []
    for fam, fn in ESTIMATORS.items():
        errs, skipped = [], 0
        for m in models:
            if m["graph_id"] != fam:
                continue
            gt = m["groundtruth"].get("ATE(Y | X)")
            if gt is None:
                continue
            try:
                errs.append(abs(fn(m["params"]) - gt))
            except (KeyError, TypeError, IndexError):
                skipped += 1
        e = np.array(errs) if errs else np.array([np.nan])
        rows.append({
            "graph_id": fam,
            "checked": len(errs),
            "skipped": skipped,
            "max_abs_err": f"{np.nanmax(e):.3e}",
            "mean_abs_err": f"{np.nanmean(e):.3e}",
            "all_match": bool(np.all(e < TOL)),
        })

    df = pd.DataFrame(rows)
    print("KIEM CHUNG GROUND TRUTH CLADDER - liet ke vet can tu CPD")
    print(f"nguong khop: {TOL}\n")
    print(df.to_string(index=False))
    out = ROOT / "results" / "groundtruth_verification.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)

    ok = df["all_match"].all()
    print(f"\n{'GROUND TRUTH DUNG TOAN BO' if ok else 'CO SAI LECH - DIEU TRA THEM'}")
    print(f"da ghi {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
